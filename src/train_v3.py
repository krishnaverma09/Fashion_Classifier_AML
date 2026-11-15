"""v3 experiment: models from the reference papers, a residual CNN and a softmax ensemble.

Run from the project root:
  python -m src.train_v3 train --model resnet     # one model -> runs/v3/experiments/resnet/
  python -m src.train_v3 train --model all        # every model in MODELS
  python -m src.train_v3 report                   # validation table + learning curves
  python -m src.train_v3 test                     # freeze the choice, official test set exactly once
  python -m src.train_v3 install                  # copy the ResNet into models/resnet/ for the demo

Papers:
  [IJST] Sumera et al. (2022), Implementation of CNN and ANN for Fashion-MNIST using different optimizers.
  [JSR]  Kadam, Adamuthe & Patil (2020), CNN model for image classification on MNIST and Fashion-MNIST.
  [EY]   Edwards & Yen, ECE 471 final project: Fashion-MNIST.

Every model trains on the same 48,000 training images and is early-stopped on the same 12,000
validation images (minimum validation loss, as in v1/v2). Overfitting is controlled by early stopping,
dropout, weight decay, BatchNorm and, for the residual CNN, augmentation; the clean train-validation
gap is recorded for every model. Outputs go to runs/v3/ and reports/v3/ only; v1/v2 files are not written.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import random
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .data import CLASS_NAMES, TRAIN_FILES, load_training_data, load_or_create_split
from .model import count_parameters
from .model_v3 import ANN, IJSTCNN, KadamArch3, KadamArch5, KerasCNN, ResNet
from .train_v2 import augment, batches, tensors

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'runs/v3'
REPORT = ROOT / 'reports/v3'
SEED = 42
COMMON = {'batch_size': 128, 'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0001,
          'max_epochs': 50, 'patience': 6, 'lr_schedule': 'ReduceLROnPlateau(validation loss, factor 0.5, patience 2)',
          'selection_metric': 'minimum validation cross-entropy loss'}
ENSEMBLE_SIZE = 3


MODELS = {  # name: (class, augment, source)
    'ann':         (ANN, False, '[IJST] ANN 784-3000-1000-10 (+dropout 0.3)'),
    'ijst_cnn':    (IJSTCNN, False, '[IJST] CNN 32/64, dense 64'),
    'kadam_arch3': (KadamArch3, False, '[JSR] Architecture 3'),
    'kadam_arch5': (KadamArch5, False, '[JSR] Architecture 5'),
    'keras_cnn':   (KerasCNN, False, '[EY] Keras CNN'),
    'resnet':      (ResNet, True, 'Residual CNN with BatchNorm + flip/shift/random-erasing augmentation'),
}
AUGMENTATION = ('v2 augmentation (horizontal flip p=0.5, shift up to 2 px) followed by random erasing '
                '(Zhong et al., cited by [JSR]): p=0.5, one zero-filled rectangle covering 2-20% of the image')


def random_erase(x, generator, p=0.5, area=(0.02, 0.2), aspect=(0.3, 3.3)):
    """Zero one random rectangle per selected image (background is 0); vectorized over the batch."""
    n, _, h, w = x.shape
    target = h * w * (area[0] + (area[1] - area[0]) * torch.rand(n, generator=generator))
    log_ratio = np.log(aspect[0]) + (np.log(aspect[1]) - np.log(aspect[0])) * torch.rand(n, generator=generator)
    ratio = torch.exp(log_ratio)
    eh = torch.sqrt(target * ratio).round().long().clamp(1, h)
    ew = torch.sqrt(target / ratio).round().long().clamp(1, w)
    top = (torch.rand(n, generator=generator) * (h - eh + 1)).long()
    left = (torch.rand(n, generator=generator) * (w - ew + 1)).long()
    chosen = torch.rand(n, generator=generator) < p
    rows, cols = torch.arange(h)[None, :, None], torch.arange(w)[None, None, :]
    box = ((rows >= top[:, None, None]) & (rows < (top + eh)[:, None, None])
           & (cols >= left[:, None, None]) & (cols < (left + ew)[:, None, None]))
    return x.masked_fill((box & chosen[:, None, None])[:, None], 0.0)


def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def evaluate(model, x, y, dev):
    """Eval-mode mean loss, accuracy and softmax probabilities, in batches of 512."""
    model.eval()
    loss, correct, probabilities = 0.0, 0, []
    with torch.inference_mode():
        for xb, yb in batches(x, y, 512):
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            loss += F.cross_entropy(logits, yb, reduction='sum').item()
            correct += (logits.argmax(1) == yb).sum().item()
            probabilities.append(logits.softmax(1).cpu())
    return loss / len(x), correct / len(x), torch.cat(probabilities).numpy()


def prepare_root():
    """Training archives copied into runs/v3 (never data/), and the shared seed-42 split."""
    raw = V3 / 'data/raw/FashionMNIST'
    raw.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_FILES:
        if not (raw / name).exists() and (ROOT / 'data/raw/FashionMNIST' / name).exists():
            shutil.copy2(ROOT / 'data/raw/FashionMNIST' / name, raw / name)
    images, labels = load_training_data(V3, download=True)
    train_idx, val_idx = load_or_create_split(labels, V3)
    return images, labels, train_idx, val_idx


def train(name, threads=4):
    if (V3 / 'selection.json').exists():
        raise FileExistsError('v3 choice is frozen and the test set may be open; no further training.')
    out = V3 / 'experiments' / name
    if (out / 'summary.json').exists():
        print(f'{name} already complete; skipping.', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    images, labels, train_idx, val_idx = prepare_root()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    dev = device()
    build, use_augment, source = MODELS[name]
    x_fit, y_fit = tensors(images, labels, train_idx)
    x_val, y_val = tensors(images, labels, val_idx)
    model = build().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON['learning_rate'], weight_decay=COMMON['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    generator = torch.Generator().manual_seed(SEED)
    print(f"{name}: {count_parameters(model):,} parameters; {len(train_idx)} training images; "
          f"augmentation={'on' if use_augment else 'off'}; device={dev}", flush=True)
    history, best_loss, best_epoch, stale, start = [], float('inf'), 0, 0, time.perf_counter()
    for epoch in range(1, COMMON['max_epochs'] + 1):
        tick = time.perf_counter()
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in batches(x_fit, y_fit, COMMON['batch_size'], generator):
            if use_augment:
                xb = random_erase(augment(xb, generator), generator)
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, dev)
        scheduler.step(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'model': name, 'class_names': list(CLASS_NAMES)}, out / 'best.pt')
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(train_idx),
                        'train_accuracy': correct / len(train_idx), 'validation_loss': val_loss,
                        'validation_accuracy': val_acc, 'learning_rate': optimizer.param_groups[0]['lr'],
                        'seconds': time.perf_counter() - tick, 'checkpoint_saved': improved})
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f"  epoch {epoch:02d}: train loss={history[-1]['train_loss']:.4f} val loss={val_loss:.4f} "
              f"val acc={val_acc:.4f} {history[-1]['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if stale >= COMMON['patience']:
            break
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True)['state_dict'])
    val_loss, val_acc, probabilities = evaluate(model, x_val, y_val, dev)
    # Clean (eval-mode, unaugmented) scores on the training images: the overfitting gap.
    train_loss, train_acc, _ = evaluate(model, x_fit, y_fit, dev)
    pred, truth = probabilities.argmax(1), y_val.numpy()
    np.savez_compressed(out / 'validation_predictions.npz', validation_indices=val_idx, predictions=pred,
                        probabilities=probabilities.astype(np.float32))
    summary = {'model': name, 'source': source, 'seed': SEED, **COMMON,
               'augmentation': AUGMENTATION if use_augment else 'none',
               'parameters': count_parameters(model), 'training_images': len(train_idx),
               'validation_images': len(val_idx), 'epochs_run': len(history), 'best_epoch': best_epoch,
               'validation_loss': val_loss, 'validation_accuracy': val_acc,
               'validation_macro_f1': f1_score(truth, pred, average='macro'),
               'shirt_validation_recall': recall_score(truth, pred, labels=[6], average=None)[0],
               'train_loss': train_loss, 'train_accuracy': train_acc,
               'accuracy_gap': train_acc - val_acc, 'loss_gap': val_loss - train_loss,
               'seconds': time.perf_counter() - start, 'device': str(dev), 'torch': torch.__version__,
               'official_test_loaded': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{name}: best epoch {best_epoch}; val acc={val_acc:.4f}, val loss={val_loss:.4f}, "
          f"train-val acc gap={train_acc - val_acc:+.4f}", flush=True)


def collect():
    rows = [json.loads((V3 / 'experiments' / m / 'summary.json').read_text())
            for m in MODELS if (V3 / 'experiments' / m / 'summary.json').exists()]
    if not rows:
        raise FileNotFoundError('No completed v3 runs.')
    return pd.DataFrame(rows).set_index('model')


def validation_probabilities(names):
    return {m: np.load(V3 / 'experiments' / m / 'validation_predictions.npz')['probabilities'] for m in names}


def ensemble_choice(runs, labels, val_idx):
    """Top ENSEMBLE_SIZE single models by validation accuracy; their softmax average is the ensemble."""
    members = list(runs['validation_accuracy'].sort_values(ascending=False).index[:ENSEMBLE_SIZE])
    probabilities = np.mean(list(validation_probabilities(members).values()), axis=0)
    return members, accuracy_score(labels[val_idx], probabilities.argmax(1))


def report():
    runs = collect()
    REPORT.mkdir(parents=True, exist_ok=True)
    columns = ['source', 'parameters', 'augmentation', 'best_epoch', 'epochs_run', 'validation_accuracy',
               'validation_macro_f1', 'validation_loss', 'train_accuracy', 'accuracy_gap', 'loss_gap',
               'shirt_validation_recall', 'seconds']
    table = runs[columns].sort_values('validation_accuracy', ascending=False)
    if len(runs) >= ENSEMBLE_SIZE:
        _, labels, _, val_idx = prepare_root()
        members, accuracy = ensemble_choice(runs, labels, val_idx)
        table.loc['ensemble'] = pd.Series({'source': 'softmax average of ' + ' + '.join(members),
                                           'validation_accuracy': accuracy})
    table.to_csv(REPORT / 'validation_results.csv')
    figure, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3), sharey=True, squeeze=False)
    for axis, name in zip(axes[0], runs.index):
        history = pd.read_csv(V3 / 'experiments' / name / 'history.csv')
        axis.plot(history.epoch, history.train_loss, label='train (as trained)')
        axis.plot(history.epoch, history.validation_loss, label='validation')
        axis.axvline(runs.loc[name, 'best_epoch'], color='grey', linestyle=':', linewidth=1)
        axis.set_title(name)
        axis.set_xlabel('epoch')
    axes[0][0].set_ylabel('cross-entropy loss')
    axes[0][0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(REPORT / 'learning_curves.png', dpi=120)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.4f}'.format):
        print(table[['parameters', 'validation_accuracy', 'validation_loss', 'train_accuracy', 'accuracy_gap']])
    return runs


def test():
    """Freeze the validation-based choice, then score every v3 model on the official test set once."""
    from .evaluate import load_official_test
    if (V3 / 'selection.json').exists():
        raise FileExistsError('v3 test evaluation already ran; it runs once and is never repeated.')
    runs = report()
    if set(runs.index) != set(MODELS):
        raise ValueError(f'Train every model first; missing: {sorted(set(MODELS) - set(runs.index))}')
    _, labels, _, val_idx = prepare_root()
    members, ensemble_accuracy = ensemble_choice(runs, labels, val_idx)
    best_single = runs['validation_accuracy'].idxmax()
    chosen = 'ensemble' if ensemble_accuracy > runs.loc[best_single, 'validation_accuracy'] else best_single
    # Written before the test set is opened, so a crash can't be followed by a quiet second attempt.
    selection = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'chosen': chosen,
                 'best_single_model': best_single, 'ensemble_members': members,
                 'validation_accuracy': {**runs['validation_accuracy'].to_dict(), 'ensemble': ensemble_accuracy},
                 'rule': 'Highest validation accuracy among the single models and the top-3 softmax ensemble.',
                 'official_test_loaded': False}
    (V3 / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    cached = ROOT / 'runs/reproduce/data/raw/FashionMNIST'
    for name in ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'):
        if not (V3 / 'data/raw/FashionMNIST' / name).exists() and (cached / name).exists():
            shutil.copy2(cached / name, V3 / 'data/raw/FashionMNIST' / name)
    images, test_labels = load_official_test(V3)
    x, y = tensors(images, test_labels, np.arange(len(test_labels)))
    dev, rows, probabilities = device(), [], {}
    for name in MODELS:
        model = MODELS[name][0]().to(dev)
        model.load_state_dict(torch.load(V3 / 'experiments' / name / 'best.pt', weights_only=True)['state_dict'])
        _, _, probabilities[name] = evaluate(model, x, y, dev)
    probabilities['ensemble'] = np.mean([probabilities[m] for m in members], axis=0)
    for name, p in probabilities.items():
        pred = p.argmax(1)
        rows.append({'model': name, 'test_accuracy': accuracy_score(test_labels, pred),
                     'test_macro_f1': f1_score(test_labels, pred, average='macro'),
                     'validation_accuracy': selection['validation_accuracy'][name],
                     **dict(zip(CLASS_NAMES, recall_score(test_labels, pred, average=None, labels=np.arange(10))))})
    results = pd.DataFrame(rows).set_index('model').sort_values('validation_accuracy', ascending=False)
    results.to_csv(REPORT / 'test_results.csv')
    summary = {**selection, 'official_test_loaded': True,
               'chosen_test_accuracy': float(results.loc[chosen, 'test_accuracy']),
               'chosen_test_macro_f1': float(results.loc[chosen, 'test_macro_f1']),
               'v1_cnn_test_accuracy': 0.89,
               'note': 'Single test evaluation after the validation-based choice was frozen; not used for tuning.'}
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(results[['validation_accuracy', 'test_accuracy', 'test_macro_f1', 'Shirt']])
    print(f'Chosen on validation: {chosen}; test accuracy {summary["chosen_test_accuracy"]:.4f} (v1 CNN: 0.8900)')


def install():
    """Copy the tested ResNet checkpoint to models/resnet/, where the demo looks for it."""
    source = V3 / 'experiments/resnet/best.pt'
    results = pd.read_csv(REPORT / 'test_results.csv').set_index('model')
    target = ROOT / 'models/resnet'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / 'best.pt')
    config = {'model': 'resnet', 'architecture': 'src.model_v3.ResNet',
              'checkpoint_sha256': hashlib.sha256((target / 'best.pt').read_bytes()).hexdigest(),
              'source': str(source.relative_to(ROOT)), 'class_names': list(CLASS_NAMES),
              'preprocessing': 'uint8 -> float32 / 255; (N,1,28,28); original orientation',
              'test_accuracy': float(results.loc['resnet', 'test_accuracy']),
              'test_macro_f1': float(results.loc['resnet', 'test_macro_f1']),
              'validation_accuracy': float(results.loc['resnet', 'validation_accuracy'])}
    (target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f"Installed {target / 'best.pt'} (test accuracy {config['test_accuracy']:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--model', choices=[*MODELS, 'all'], required=True)
    run.add_argument('--threads', type=int, default=4)
    for name in ('report', 'test', 'install'):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == 'train':
        for name in (MODELS if args.model == 'all' else [args.model]):
            train(name, args.threads)
    else:
        {'report': report, 'test': test, 'install': install}[args.command]()


if __name__ == '__main__':
    main()
