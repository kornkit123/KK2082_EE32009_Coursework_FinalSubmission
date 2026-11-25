"""Tests for data.py: loading, drift removal, noise estimation and windowing."""
import numpy as np
import pytest
import scipy.io as spio
import data


# ---------------------------------------------------------------- load_data
def test_load_data_reads_all_six_recordings(tmp_path):
    """D1 provides d, Index and Class; D2-D6 provide only d."""
    spio.savemat(tmp_path / 'D1.mat', {'d': np.arange(10.0), 'Index': [2, 5], 'Class': [1, 3]})
    for n in range(2, 7):
        spio.savemat(tmp_path / f'D{n}.mat', {'d': np.full(10, float(n))})

    recordings, index1, class1 = data.load_data(tmp_path)

    assert len(recordings) == 6
    assert np.array_equal(recordings[0], np.arange(10.0))
    assert np.all(recordings[5] == 6.0)
    assert list(index1) == [2, 5]
    assert list(class1) == [1, 3]


# ---------------------------------------------------------- remove_baseline
def test_remove_baseline_removes_constant_offset():
    sig = np.full(5000, 7.0)
    assert np.allclose(data.remove_baseline(sig, size=101), 0.0)


def test_remove_baseline_removes_slow_drift_but_keeps_spike():
    t = np.arange(20_000)
    drift = 2.0 * np.sin(2 * np.pi * t / 20_000)      # one slow cycle
    spike = np.zeros_like(drift)
    spike[10_000:10_020] = 5.0                         # 20-sample spike
    cleaned = data.remove_baseline(drift + spike, size=501)

    away_from_edges = slice(1000, -1000)
    assert np.abs(cleaned - spike)[away_from_edges].max() < 0.05   # drift gone
    assert cleaned[10_010] == pytest.approx(5.0, abs=0.05)         # spike unchanged


# ---------------------------------------------------------------- noise_std
def test_noise_std_matches_true_std_for_gaussian_noise(rng):
    noise = rng.normal(0, 2.0, 200_000)
    assert data.noise_std(noise) == pytest.approx(2.0, rel=0.02)


def test_noise_std_ignores_sparse_spikes(rng):
    """Large, rare spikes should barely change the robust estimate."""
    sig = rng.normal(0, 1.0, 200_000)
    sig[::1000] += 50.0                               # 0.1% of samples are huge
    assert data.noise_std(sig) == pytest.approx(1.0, rel=0.03)
    assert np.std(sig) > 1.5                          # whereas plain std is inflated


# ------------------------------------------------------------ noisy_version
def test_noisy_version_reaches_target_noise_level(rng):
    clean = rng.normal(0, 1.0, 200_000)
    noisy = data.noisy_version(clean, target=3.0, base=1.0, rng=rng)
    assert np.std(noisy) == pytest.approx(3.0, rel=0.02)


def test_noisy_version_adds_nothing_when_target_below_base(rng):
    sig = rng.normal(0, 2.0, 1000)
    assert np.array_equal(data.noisy_version(sig, target=1.0, base=2.0, rng=rng), sig)


# ---------------------------------------------------------- extract_windows
def test_extract_windows_shape_and_content():
    sig = np.arange(100)
    windows = data.extract_windows(sig, positions=[20, 50], pre=5, post=10)
    assert windows.shape == (2, 15)
    assert np.array_equal(windows[0], np.arange(15, 30))     # sig[20-5 : 20+10]
    assert np.array_equal(windows[1], np.arange(45, 60))


def test_extract_windows_jitter_stays_within_limit(rng):
    sig = np.arange(1000)
    positions = np.full(500, 500)
    windows = data.extract_windows(sig, positions, pre=5, post=10, jitter=3, rng=rng)
    shifts = windows[:, 5] - 500                     # sample at "t" shows the shift
    assert shifts.min() >= -3 and shifts.max() <= 3
    assert len(np.unique(shifts)) > 1                # jitter actually happened


# ----------------------------------------------------- background_positions
def test_background_positions_are_far_from_spikes_and_in_range(rng):
    spikes = np.array([1000, 5000, 9000])
    positions = data.background_positions(spikes, lo=100, hi=9900, n_candidates=2000,
                                          min_gap=100, rng=rng)
    assert len(positions) > 0
    assert positions.min() >= 100 and positions.max() < 9900
    distance = np.abs(positions[:, None] - spikes[None, :]).min(axis=1)
    assert distance.min() > 100


# -------------------------------------------------- hard_negative_positions
def test_hard_negatives_are_near_but_not_on_spikes(rng):
    spikes = np.arange(1000, 100_000, 1000)           # well-separated spikes
    negatives = data.hard_negative_positions(spikes, lo=0, hi=200_000, all_spikes=spikes,
                                             rng=rng, min_offset=15, max_offset=60)
    distance = np.abs(negatives[:, None] - spikes[None, :]).min(axis=1)
    assert len(negatives) == len(spikes)              # none rejected here
    assert distance.min() >= 15 and distance.max() <= 60


def test_hard_negatives_respect_bounds_and_neighbouring_spikes(rng):
    spikes = np.array([100, 130])                     # two spikes 30 samples apart
    negatives = data.hard_negative_positions(np.repeat(spikes, 200), lo=90, hi=200,
                                             all_spikes=spikes, rng=rng)
    assert np.all((negatives >= 90) & (negatives < 200))
    distance = np.abs(negatives[:, None] - spikes[None, :]).min(axis=1)
    assert distance.min() >= 15                       # never lands on the other spike
