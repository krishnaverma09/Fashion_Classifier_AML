# Automatic Clothing Category Recognition

Inventory image tagging with handcrafted features, CNNs and a residual network.

This project classifies Fashion-MNIST clothing images into ten categories. It compares
classical models (on flattened pixels and HOG features) with neural networks, including
architectures from three reference papers, a residual CNN and an ensemble. The residual CNN is
served through a local Gradio demo. On the official test set, the ensemble reaches **94.19%**
accuracy and the residual CNN **94.32%**. The small CNN reaches **89.00%**, and the best
classical baseline (pixel Random Forest) **85.27%**.

The full write-up, in IEEE conference format, is
[`reports/ieee_report/Fashion_Classifier_IEEE_Report.pdf`](reports/ieee_report/Fashion_Classifier_IEEE_Report.pdf).

## Team

| Name | Enroll |
|---|---|
| Satyam Kumar | 2401010428 |
| Krishna Verma | 2401010240 |
| Akshit Vats | 2401020085 |
| Chinmay Soni | 2401010134 |
| Lakshya Bapna | TBD |

## Results

Every model is evaluated once on the 10,000 official test images, after its selection was
frozen using validation data only.

| Model | Training images | Test accuracy | Test macro F1 | Validation accuracy | Train–val gap |
|---|---:|---:|---:|---:|---:|
| **Ensemble** (ResNet + Arch 5 + Keras CNN) | 48,000 | **94.19%** | **94.17%** | 95.15% | – |
| ResNet + augmentation (demo model) | 48,000 | 94.32% | 94.31% | 94.97% | 2.0 pts |
| JSR Architecture 5 | 48,000 | 92.74% | 92.72% | 93.76% | 3.4 pts |
| EY Keras CNN | 48,000 | 92.75% | 92.71% | 93.07% | 4.8 pts |
| JSR Architecture 3 | 48,000 | 91.90% | 91.90% | 92.87% | 2.8 pts |
| IJST CNN 32/64 | 48,000 | 91.35% | 91.39% | 91.84% | 4.4 pts |
| IJST ANN 784-3000-1000-10 | 48,000 | 90.01% | 89.99% | 91.00% | 4.5 pts |
| Small CNN | 12,000 | 89.00% | 88.85% | 89.93% | – |
| Pixels + Random Forest | 12,000 | 85.27% | 85.09% | 86.46% | – |
| HOG + Logistic Regression | 12,000 | 83.27% | 83.22% | 83.94% | – |
| HOG + Linear SVM | 12,000 | 82.56% | 82.40% | 84.03% | – |
| Pixels + kNN | 12,000 | 82.41% | 82.40% | 83.47% | – |
| Pixels + Logistic Regression | 12,000 | 82.27% | 82.22% | 83.75% | – |
| HOG + Random Forest | 12,000 | 81.97% | 81.94% | 82.84% | – |
| HOG + kNN | 12,000 | 81.09% | 81.15% | 82.24% | – |
| Pixels + Linear SVM | 12,000 | 80.03% | 79.84% | 81.93% | – |

The ensemble was chosen because it had the highest validation accuracy. On test, the ResNet
alone scored slightly higher (94.32% vs 94.19%), a difference within noise. The train–val gap
compares clean (eval-mode, unaugmented) training accuracy with validation accuracy. Test
accuracy is within about 1 point of validation accuracy for every neural model, so none of them
overfit the validation set.

Results for the 12,000-image models (per-class recall, confusion matrices, CPU timings, model
sizes, error analysis) are in `reports/final_evaluation/`, and figures in `reports/figures/`.
Results for the 48,000-image models, with per-class recall and learning curves, are in
`reports/v3/`.

## Project layout

```
.
├── src/                 data, baselines, CNN, training, evaluation, prediction, v2/v3 experiments
├── scripts/             setup, reproduction pipeline, verification scripts
├── space/               Hugging Face Space settings (README.md) and requirements.txt
├── tests/               pytest suite
├── app.py               Gradio demo
├── notebooks/           executed workflow notebook and the unmodified reference lab
├── assets/examples/     demo sample images
├── reports/             committed results, figures, integrity manifests, progress notes
├── data/  models/       generated datasets and weights (gitignored)
├── runs/                generated pipeline, v2 and v3 outputs (gitignored)
├── archive/             Step 1 scaffold and Steps 1–6 milestone backups and logs, for reference only
└── pyproject.toml  uv.lock  Dockerfile  compose.yaml
```

## Setup

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` (Python 3.11). Both
setups install exactly that lock. Run every command from the repository root.

**Docker** (recommended; needs only Docker with compose):

```bash
docker compose build                 # Python 3.11, CPU-only PyTorch, locked dependencies
docker compose run --rm pipeline     # download data, train, evaluate, install models/
docker compose up -d app             # Gradio demo at http://127.0.0.1:7860
docker compose run --rm verify       # artifact hashes, prediction contract, HTTP checks
docker compose up notebook           # JupyterLab at http://127.0.0.1:8888
```

`pipeline` runs the whole workflow on CPU in `runs/reproduce/`, so the committed reports are
never overwritten. It takes about 15–20 minutes, mostly for the linear-SVM baselines.
Rebuild the image after changing code, because the code is copied into it.

**Native** with [uv](https://docs.astral.sh/uv/) (uv downloads Python 3.11 if needed):

```bash
bash scripts/setup_mac.sh            # uv sync --locked, environment check, Jupyter kernel
uv run python app.py                 # Gradio demo
uv run pytest                        # test suite
```

To change dependencies, edit `pyproject.toml`, run `uv lock`, and commit both files.

## Data and method

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 28×28
  grayscale images in ten balanced classes. Archives are checked against the publisher's MD5
  checksums.
- **Split:** a stratified split with seed 42 divides the 60,000 training images into 48,000
  training and 12,000 validation images.
- **Training budget:** the baselines and the small CNN use the same saved 12,000-image subset
  (1,200 per class). The paper models and the ResNet use all 48,000 training images.
- **Test set:** the official 10,000 test images were loaded only after all model choices
  were frozen, in `reports/final_evaluation/frozen-models.json` and `runs/v3/selection.json`.
- **Features:**
  - Pixels: scaled to [0, 1].
  - HOG: 9 bins, 7×7 cells, 2×2 blocks, giving 324 descriptors.
- **Baselines:**
  - Logistic Regression, kNN, Random Forest and linear SVM, each on pixels and on HOG.
  - Two predeclared settings per combination.
  - Each combination is selected by validation macro F1.
- **CNN:**
  - Architecture: two convolution blocks (32/64 channels) with BatchNorm and pooling, dense
    layers of 128 and 64, and dropout 0.4. It has 429,450 parameters.
  - Training: Adam (learning rate 1e-3, weight decay 1e-4) with batch size 128. Early stopping
    watches validation loss with patience 5; the best epoch (9 of 14) was kept.
- **Models from the reference papers** (`src/train_v3.py`):
  - **[IJST]** Sumera et al. (2022), *Implementation of CNN and ANN for Fashion-MNIST using
    different optimizers*. The ANN 784-3000-1000-10 (5,366,010 parameters) and the CNN 32/64
    with a dense layer of 64 (121,930 parameters).
  - **[JSR]** Kadam, Adamuthe and Patil (2020), *CNN model for image classification on MNIST
    and Fashion-MNIST*. Architecture 3: three 2×2 convolutions of 64, dropout 0.25. Architecture
    5: VGG-style 32-32/64-64 convolutions, a dense layer of 512, dropout 0.25 and 0.5.
  - **[EY]** Edwards and Yen, *ECE 471 final project: Fashion-MNIST*. The Keras CNN: 32 and 64
    convolutions, dropout 0.25, a dense layer of 128, dropout 0.5.
- **ResNet:** six residual blocks (32/64/128 channels) with BatchNorm and skip connections,
  following Bhatnagar et al., cited by [JSR] and [EY]. It ends in global average pooling,
  dropout 0.3 and a linear layer, with 696,042 parameters.
- **Ensemble:** the average of the softmax scores of the three models with the highest
  validation accuracy.
- **Training of the 48,000-image models:** Adam (learning rate 1e-3, weight decay 1e-4), batch
  size 128 and seed 42. The learning rate halves when validation loss stops improving. The
  final model was the one with the highest validation accuracy, out of the six single models
  and the ensemble.
- **Avoiding overfitting:**
  - Every model keeps the checkpoint with the lowest validation loss (patience 6).
  - The paper models keep their dropout. The ANN, which the paper trains without dropout, gets
    dropout 0.3.
  - The ResNet trains with augmentation: a horizontal flip, a shift of up to 2 px, and random
    erasing of one rectangle covering 2–20% of the image (Zhong et al., cited by [JSR]). It has
    the smallest train–val gap.

```bash
python -m src.train_v3 train --model all   # or --model resnet; outputs in runs/v3/experiments/
python -m src.train_v3 report              # validation table and learning curves -> reports/v3/
python -m src.train_v3 test                # freeze the choice, official test set exactly once
python -m src.train_v3 install             # copy the ResNet into models/resnet/ for the demo
```

Training all six 48,000-image models took about 30 minutes on an Apple M3 (MPS). Each was trained
once (seed 42), so differences under about 0.5 points are not reliable.

## Prediction and demo

`FashionPredictor` loads the saved CNN once. It accepts an image path, a PIL image or a NumPy
array, and returns:

- the predicted label;
- all ten softmax scores;
- the ranked predictions;
- the exact 28×28 preview passed to the model.

```python
from src.predict import FashionPredictor

result = FashionPredictor().predict("your-image.png", background="dark", top_k=3)
print(result.label, result.scores)
```

From the command line: `python -m src.predict image.png` (add `--background light` for
light backgrounds).

`ResNetPredictor` in `src/predict_v3.py` is the same predictor with the ResNet (94.32% test
accuracy). It uses the same preprocessing and returns the same result, and it checks the weights
against the hash in `models/resnet/config.json`. `load_predictor()` returns the ResNet when it is
installed (`python -m src.train_v3 install`) and the small CNN otherwise. From the command line:
`python -m src.predict_v3 image.png`.

**Demo:** run `uv run python app.py` (or `docker compose up -d app`) and open
http://127.0.0.1:7860. The demo serves the ResNet from `models/resnet/`, which is committed to
the repository, so a fresh clone (including Docker) serves it without retraining. If
`models/resnet/` is removed, the demo falls back to the small CNN.

1. Upload an image or pick a sample.
2. Choose whether the image has a dark or light background.
3. Click **Predict category**.

The app binds to localhost only and never trains or downloads data.

**Limitations:**

- The model was trained on benchmark thumbnails. Real clothing photos work best when the
  item is isolated, upright and on a plain background.
- Scores are softmax values, not calibrated confidence.

## Verification

```bash
python scripts/verify_artifacts.py   # every hash-protected report, source and model file
python scripts/verify_prediction.py  # preprocessing and prediction contract
python scripts/verify_demo.py        # demo callbacks and HTTP uploads for the served model (server must be running)
uv run pytest                        # integrity, prediction, v2/v3 safeguards, install safety
```

## Model provenance

The original 89.00% checkpoints aren't in the repository; their hashes are recorded in
`reports/final_evaluation/frozen-models.json`. A fresh clone gets its models from `pipeline`,
which retrains them on CPU using the same code and the same data split.

The CPU reproduction scored **88.91%** for the CNN and 85.27% for pixel Random Forest. The
original run used MPS, so small differences are expected. Reproduced models are recorded in
`models/PROVENANCE.json`, and `verify_artifacts.py` reports which set is installed.

The ResNet is the exception: its tested checkpoint (94.32%) is committed as
`models/resnet/best.pt`, about 2.8 MB. `models/resnet/config.json` records its SHA-256 hash,
source run and scores, and `ResNetPredictor` refuses weights that don't match that hash. The
`pipeline` leaves `models/resnet/` untouched.

## v2 experiment

`src/train_v2.py` is a controlled ablation that tests whether accuracy improves with:

- all 48,000 training images;
- mild augmentation (horizontal flip plus a shift of up to 2 px);
- a deeper CNN.

Each configuration changes one factor and runs with seeds 42, 1 and 2.

| Config | Training images | Augmentation | Network |
|---|---|---|---|
| `v1_12k` | 12,000 | – | v1 CNN (reference) |
| `v1_48k` | 48,000 | – | v1 CNN |
| `v1_48k_aug` | 48,000 | flip + shift | v1 CNN |
| `deep_48k` | 48,000 | – | deeper CNN (585k parameters) |
| `deep_48k_aug` | 48,000 | flip + shift | deeper CNN |

```bash
python -m src.train_v2 train --config deep_48k --seed 42   # one run → runs/v2/experiments/
python -m src.train_v2 report                              # validation comparison table
python -m src.train_v2 select                              # freeze the choice (validation only)
python -m src.train_v2 test                                # official test set, exactly once
```

Selection uses validation data only, through a paired bootstrap. It picks the simplest
configuration that isn't distinguishable from the best, and adopts it only if it clearly
beats `v1_12k`. The test set can be evaluated only once. The v1 code, models and the 89.00%
result are never modified.

## Next steps

Step 8: choose a hosting platform, then deploy and verify the demo.  
