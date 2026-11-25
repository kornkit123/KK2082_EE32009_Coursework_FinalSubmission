"""Loading, cleaning and windowing of the recordings, plus noise simulation."""
from pathlib import Path

import numpy as np
import scipy.io as spio
from scipy.ndimage import median_filter

from config import BASELINE_WINDOW, HARD_NEG_MAX_OFFSET, HARD_NEG_MIN_OFFSET, MAD_TO_STD


def load_data(data_dir):
    """Load all six recordings, plus the labels of D1.

    Args:
        data_dir: folder containing D1.mat ... D6.mat.

    Returns:
        recordings: list of six 1-D arrays [D1, D2, ..., D6] (the `d` vectors).
        index1:     spike start positions in D1 (samples).
        class1:     neuron class (1-5) of each spike in D1.
    """
    data_dir = Path(data_dir)
    d1_mat = spio.loadmat(data_dir / 'D1.mat', squeeze_me=True)
    recordings = [d1_mat['d']]
    for n in range(2, 7):
        recordings.append(spio.loadmat(data_dir / f'D{n}.mat', squeeze_me=True)['d'])
    return recordings, d1_mat['Index'], d1_mat['Class']


def remove_baseline(sig, size=BASELINE_WINDOW):
    """Subtract the slow baseline drift from a recording.

    A long median filter (default ~0.1 s) follows the slow drift but ignores
    the short spikes, so subtracting it flattens the baseline while leaving
    the spike shapes unchanged. Without this, drift in the noisy datasets
    (D5, D6) caused thousands of false detections (see README results log).

    Args:
        sig:  1-D recording.
        size: median filter length in samples (should be much longer than a spike).

    Returns:
        1-D array of the same length with the baseline removed.
    """
    return sig - median_filter(sig, size=size)


def noise_std(sig):
    """Estimate the noise standard deviation of a recording.

    Uses the median absolute deviation (MAD), which is robust: the spikes are
    rare, large outliers and barely move the median, whereas they would inflate
    a plain np.std(). For Gaussian noise, std = 1.4826 * MAD.

    Args:
        sig: 1-D recording.

    Returns:
        Estimated noise standard deviation (float).
    """
    return MAD_TO_STD * np.median(np.abs(sig - np.median(sig)))


def noisy_version(sig, target, base, rng):
    """Add Gaussian noise to a signal so its total noise std equals `target`.

    Used to turn the clean, labelled D1 into a stand-in for a noisier dataset
    (D2-D6) for validation.

    Args:
        sig:    recording that already contains noise of std `base`.
        target: desired total noise std (e.g. the measured noise of D5).
        base:   noise std already present in `sig`.
        rng:    numpy random Generator, for reproducibility.

    Returns:
        1-D array, same shape as `sig`, with extra noise added.
    """
    # Independent noise variances add, so the extra std is sqrt(target^2 - base^2).
    # If target <= base no noise is added.
    extra = np.sqrt(max(target**2 - base**2, 0))
    return sig + rng.normal(0, extra, sig.shape)


def extract_windows(sig, positions, pre, post, jitter=0, rng=None):
    """Cut a fixed-length window around each position in a recording.

    Window k covers sig[t - pre : t + post], where t = positions[k].

    Args:
        sig:       1-D recording.
        positions: spike start positions (samples).
        pre:       samples to keep before each position.
        post:      samples to keep after each position.
        jitter:    if > 0, shift each window by a random amount in [-jitter, +jitter].
                   This teaches the networks to cope with slightly mis-timed detections.
        rng:       numpy random Generator (needed only when jitter > 0).

    Returns:
        2-D array of shape (len(positions), pre + post).
    """
    out = []
    for t in positions:
        if jitter:
            t = t + rng.integers(-jitter, jitter + 1)
        out.append(sig[t - pre : t + post])
    return np.array(out)


def background_positions(spike_index, lo, hi, n_candidates, min_gap, rng):
    """Draw random positions that are far from every spike ("no spike here" examples).

    Args:
        spike_index:  all known spike positions.
        lo, hi:       positions are drawn uniformly from [lo, hi).
        n_candidates: number of random positions to draw before filtering.
        min_gap:      keep only positions more than this many samples from any spike.
        rng:          numpy random Generator.

    Returns:
        1-D array of accepted positions (in the order they were drawn).
    """
    candidates = rng.integers(lo, hi, size=n_candidates)
    distance_to_spike = np.abs(candidates[:, None] - spike_index[None, :]).min(axis=1)
    return candidates[distance_to_spike > min_gap]


def hard_negative_positions(spike_subset, lo, hi, all_spikes, rng,
                            min_offset=HARD_NEG_MIN_OFFSET, max_offset=HARD_NEG_MAX_OFFSET):
    """Positions slightly before or after real spikes, labelled "no spike".

    These windows contain part of a spike but at the wrong time, so they teach
    the detector to fire only at the true spike start rather than anywhere
    near a spike.

    Args:
        spike_subset: spikes to place hard negatives around (e.g. the training spikes).
        lo, hi:       keep only positions in [lo, hi).
        all_spikes:   every known spike, used to reject positions that land on another spike.
        rng:          numpy random Generator.
        min_offset, max_offset: distance range from the spike, in samples.

    Returns:
        1-D array of hard negative positions.
    """
    n = len(spike_subset)
    offsets = rng.integers(min_offset, max_offset + 1, n) * rng.choice([-1, 1], n)
    positions = spike_subset + offsets
    positions = positions[(positions >= lo) & (positions < hi)]
    # Drop any position that is actually the start of a neighbouring spike
    nearest = np.abs(positions[:, None] - all_spikes[None, :]).min(axis=1)
    return positions[nearest >= min_offset]
