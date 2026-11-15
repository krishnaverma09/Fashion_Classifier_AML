"""ResNet predictor: same contract as the v1 predictor, hash-checked weights, SmallCNN fallback."""
import json
import shutil

import pytest

from src import predict_v3
from src.data import CLASS_NAMES


@pytest.fixture(scope='module')
def predictor():
    if not (predict_v3.RESNET_DIR / 'best.pt').exists():
        pytest.skip('ResNet not installed; run python -m src.train_v3 install.')
    return predict_v3.ResNetPredictor()


def test_resnet_predictions_are_valid_and_deterministic(predictor, root):
    for record in json.loads((root / 'assets/examples/manifest.json').read_text()):
        first = predictor.predict(root / record['path'], background=record['background'], top_k=3)
        second = predictor.predict(root / record['path'], background=record['background'], top_k=3)
        assert first.label in CLASS_NAMES and list(first.scores) == list(CLASS_NAMES)
        assert abs(sum(first.scores.values()) - 1) < 1e-5
        assert first.scores == second.scores


def test_load_predictor_prefers_resnet(predictor):
    assert predict_v3.load_predictor().name == 'ResNet'


def test_altered_checkpoint_is_rejected(predictor, tmp_path):
    shutil.copytree(predict_v3.RESNET_DIR, tmp_path / 'resnet')
    with (tmp_path / 'resnet/best.pt').open('ab') as checkpoint:
        checkpoint.write(b'\0')
    with pytest.raises(ValueError, match='does not match'):
        predict_v3.ResNetPredictor(tmp_path / 'resnet')


def test_falls_back_to_small_cnn_without_resnet(root, tmp_path, monkeypatch):
    if not (root / 'models/cnn/best.pt').exists():
        pytest.skip('No installed CNN weights; run the pipeline first.')
    monkeypatch.setattr(predict_v3, 'RESNET_DIR', tmp_path / 'missing')
    assert predict_v3.load_predictor().name == 'SmallCNN'
