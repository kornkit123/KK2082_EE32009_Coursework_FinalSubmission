"""Tests for network.py and training.py."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import training
from config import CLS_POST, CLS_PRE, DET_POST, DET_PRE, MODEL_DIR, MODEL_TAG
from network import SpikeCNN


# ------------------------------------------------------------------ network
@pytest.mark.parametrize('n_out, length', [(2, DET_PRE + DET_POST), (5, CLS_PRE + CLS_POST)])
def test_spikecnn_output_shape(n_out, length):
    """Detector and classifier windows have different lengths but share the network."""
    model = SpikeCNN(n_out).eval()
    out = model(torch.zeros(8, 1, length))
    assert out.shape == (8, n_out)


@pytest.mark.parametrize('name, n_out', [('clf', 5), ('det', 2)])
def test_saved_weights_match_the_architecture(name, n_out):
    """The weight files shipped in models/ must load into the current SpikeCNN."""
    path = MODEL_DIR / f'{name}_{MODEL_TAG}.pt'
    if not path.exists():
        pytest.skip(f'{path.name} not present')
    SpikeCNN(n_out).load_state_dict(torch.load(path, map_location='cpu'))   # strict load


# ---------------------------------------------------------------- to_tensor
def test_to_tensor_normalises_and_adds_channel_axis():
    windows = np.array([[1.0, 3.0, 5.0], [5.0, 5.0, 5.0]])
    x = training.to_tensor(windows, mu=1.0, sigma=2.0)
    assert x.shape == (2, 1, 3)
    assert x.dtype == torch.float32
    assert torch.allclose(x[0, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_make_loader_batches():
    loader = training.make_loader(np.zeros((10, 65)), np.arange(10) % 2, shuffle=False,
                                  mu=0.0, sigma=1.0, batch_size=4)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 1, 65)
    assert yb.dtype == torch.long
    assert len(loader) == 3                           # 4 + 4 + 2


# ---------------------------------------------------------------- add_noise
def test_add_noise_with_zero_levels_changes_nothing():
    xb = torch.randn(16, 1, 65)
    out = training.add_noise(xb, levels_extra=torch.zeros(3))
    assert torch.equal(out, xb)


def test_add_noise_uses_only_levels_with_weight():
    """With all weight on level 2 (std 3), the added noise has std about 3."""
    torch.manual_seed(0)
    xb = torch.zeros(2000, 1, 65)
    out = training.add_noise(xb, levels_extra=torch.tensor([0.0, 1.0, 3.0]),
                             weights=torch.tensor([0.0, 0.0, 1.0]))
    assert out.std().item() == pytest.approx(3.0, rel=0.03)


def test_add_noise_gives_fresh_noise_each_call():
    xb = torch.zeros(4, 1, 65)
    levels = torch.tensor([1.0])
    assert not torch.equal(training.add_noise(xb, levels), training.add_noise(xb, levels))


# -------------------------------------------------------------- train_model
def _toy_problem(rng, n=400, length=65):
    """Two easy classes: windows with a positive bump (label 1) or without (label 0)."""
    x = rng.normal(0, 0.3, (n, length))
    y = rng.integers(0, 2, n)
    x[y == 1, 20:40] += 2.0
    return x, y


def test_train_model_learns_a_simple_task(rng):
    torch.manual_seed(0)
    x_train, y_train = _toy_problem(rng)
    x_val, y_val = _toy_problem(rng, n=200)
    model = training.train_model(
        SpikeCNN(n_out=2),
        training.make_loader(x_train, y_train, shuffle=True, mu=0.0, sigma=1.0),
        training.make_loader(x_val, y_val, shuffle=False, mu=0.0, sigma=1.0),
        nn.CrossEntropyLoss(), epochs=5, device='cpu', verbose=False)
    assert training.accuracy(model, x_val, y_val, 0.0, 1.0, device='cpu') > 0.95


def test_train_model_with_zero_epochs_returns_model_unchanged(rng):
    model = SpikeCNN(n_out=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x, y = _toy_problem(rng, n=20)
    loader = training.make_loader(x, y, shuffle=False, mu=0.0, sigma=1.0)
    training.train_model(model, loader, loader, nn.CrossEntropyLoss(), epochs=0,
                         device='cpu', verbose=False)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
