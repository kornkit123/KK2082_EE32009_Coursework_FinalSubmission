"""End-to-end tests of inference.py and the export step in main.py.

Real networks give unpredictable outputs, so these tests use two tiny "stub"
models whose behaviour is known exactly:
  - the detector says "spike" when the sample at the spike start is > 0.5
  - the classifier returns the class equal to the (rounded) amplitude there
A synthetic recording with impulses of known position and height then has a
known correct answer.
"""
import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn

import inference
import main
from config import CLS_PRE, DET_PRE


class StubDetector(nn.Module):
    """Logits [0, score], where score is large and positive when sig[t] > 0.5."""

    def forward(self, x):
        score = 50.0 * (x[:, 0, DET_PRE] - 0.5)
        return torch.stack([torch.zeros_like(score), score], dim=1)


class StubClassifier(nn.Module):
    """Picks class k (0-4) whose value k + 1 is closest to the amplitude at the spike start."""

    def forward(self, x):
        amplitude = x[:, 0, CLS_PRE][:, None]
        return -(torch.arange(1, 6, dtype=torch.float32, device=x.device)[None, :] - amplitude) ** 2


def _synthetic_recording(n_samples=20_000):
    """Zeros with impulses at known positions; impulse height = class label (1-5)."""
    positions = np.array([1000, 3000, 5000, 7000, 9000, 9030, 15000])
    classes = np.array([1, 2, 3, 4, 5, 2, 3])
    sig = np.zeros(n_samples)
    sig[positions] = classes
    return sig, positions, classes


def test_detection_probabilities_length_and_range():
    sig, positions, _ = _synthetic_recording()
    probs = inference.detection_probabilities(StubDetector(), sig, start=100, stop=12_000,
                                              mu=0.0, sigma=1.0, batch=1000, device='cpu')
    assert len(probs) == 12_000 - 100
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert probs[1000 - 100] > 0.99                   # spike at sample 1000
    assert probs[2000 - 100] < 0.01                   # nothing at sample 2000


def test_classify_returns_empty_for_no_spikes():
    sig, _, _ = _synthetic_recording()
    assert len(inference.classify(sig, np.array([], dtype=int), StubClassifier(),
                                  0.0, 1.0, device='cpu')) == 0


def test_predict_dataset_finds_every_spike_and_class():
    sig, positions, classes = _synthetic_recording()
    index, cls = inference.predict_dataset(sig, threshold=0.5, min_distance=15,
                                           detector=StubDetector(), classifier=StubClassifier(),
                                           mu=0.0, sigma=1.0)
    assert np.array_equal(index, positions)
    assert np.array_equal(cls, classes)               # labels 1-5, not 0-4


def test_min_distance_merges_close_spikes():
    """Spikes at 9000 and 9030 are merged when min_distance > 30."""
    sig, positions, _ = _synthetic_recording()
    index, _ = inference.predict_dataset(sig, 0.5, 40, StubDetector(), StubClassifier(), 0.0, 1.0)
    assert len(index) == len(positions) - 1


def test_export_writes_only_index_and_class(tmp_path):
    """The brief asks for Index and Class only (no raw d) in D2.mat ... D6.mat."""
    sig, positions, classes = _synthetic_recording()
    recordings = [sig] * 6
    settings = {n: (0.5, 15) for n in range(2, 7)}

    main.export_submission(recordings, StubDetector(), StubClassifier(), 0.0, 1.0,
                           settings=settings, out_dir=tmp_path)

    for n in range(2, 7):
        mat = spio.loadmat(tmp_path / f'D{n}.mat', squeeze_me=True)
        keys = {k for k in mat if not k.startswith('__')}
        assert keys == {'Index', 'Class'}
        assert np.array_equal(mat['Index'], positions)
        assert np.array_equal(mat['Class'], classes)
