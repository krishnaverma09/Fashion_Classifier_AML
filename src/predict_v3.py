"""Dataset-free ResNet prediction, reusing the v1 preprocessing and prediction contract.

Example: python -m src.predict_v3 image.png --background dark
The ResNet is installed by `python -m src.train_v3 install`. load_predictor() falls back to the
v1 SmallCNN when it is not installed, so a fresh clone or the Docker pipeline still works.
"""
from pathlib import Path
import argparse
import hashlib
import json

import torch

from .model_v3 import ResNet
from .predict import FashionPredictor

ROOT = Path(__file__).resolve().parents[1]
RESNET_DIR = ROOT / 'models/resnet'


class ResNetPredictor(FashionPredictor):
    """FashionPredictor with the v3 ResNet; prepare_image() and predict() are inherited unchanged."""
    name = 'ResNet'

    def __init__(self, model_dir=RESNET_DIR, *, device='cpu'):
        model_dir = Path(model_dir)
        self.checkpoint_path = model_dir / 'best.pt'
        if device not in ('cpu', 'mps'):
            raise ValueError("device must be 'cpu' or 'mps'.")
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; choose device=cpu.')
        self.config = json.loads((model_dir / 'config.json').read_text())
        if hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest() != self.config['checkpoint_sha256']:
            raise ValueError(f'{self.checkpoint_path} does not match {model_dir / "config.json"}.')
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('model') != 'resnet' or checkpoint.get('class_names') != self.config['class_names']:
            raise ValueError('Checkpoint is not the installed v3 ResNet.')
        self.class_names = tuple(checkpoint['class_names'])
        self.device = torch.device(device)
        self.best_epoch = int(checkpoint['epoch'])
        self.model = ResNet().to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()


def load_predictor(device='cpu'):
    """The ResNet when installed, otherwise the v1 SmallCNN."""
    if (RESNET_DIR / 'best.pt').exists():
        return ResNetPredictor(device=device)
    predictor = FashionPredictor(ROOT / 'models/cnn/best.pt', device=device)
    predictor.name = 'SmallCNN'
    return predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--background', choices=('dark', 'light'), default='dark')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = load_predictor()
    try:
        result = predictor.predict(args.image, background=args.background, top_k=args.top_k)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'model': predictor.name, **result.as_dict()}, indent=2))


if __name__ == '__main__':
    main()
