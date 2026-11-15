"""v3 architectures, random erasing and test-set guards (no training, no test data)."""
import json

import pytest
import torch

from src import train_v3
from src.model import count_parameters


@pytest.mark.parametrize('name', train_v3.MODELS)
def test_model_shapes(name):
    model = train_v3.MODELS[name][0]().eval()
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_paper_parameter_counts():
    # Figures 3 and 4 of the IJST paper.
    assert count_parameters(train_v3.ANN()) == 5_366_010
    assert count_parameters(train_v3.IJSTCNN()) == 121_930


def test_random_erase_zeroes_one_rectangle():
    x = torch.rand(256, 1, 28, 28) + 0.1  # strictly positive, so erased pixels are exactly the zeros
    out = train_v3.random_erase(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    erased = 0
    for original, moved in zip(x, out):
        zero = moved[0] == 0
        assert torch.equal(moved[0][~zero], original[0][~zero])
        if zero.any():
            erased += 1
            rows, cols = zero.nonzero(as_tuple=True)
            box = zero[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            assert box.all() and zero.sum() == box.numel()
            assert zero.sum() <= 0.2 * 28 * 28 + 28
    assert 0.3 < erased / len(x) < 0.7


def test_training_and_test_refused_after_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v3, 'V3', tmp_path)
    (tmp_path / 'selection.json').write_text(json.dumps({}))
    with pytest.raises(FileExistsError):
        train_v3.train('resnet')
    with pytest.raises(FileExistsError):
        train_v3.test()
