"""Exploratory analysis of the recordings (not needed to produce the submission).

These plots and statistics justified the main design choices:
    - baseline drift      -> remove_baseline() before detection
    - white noise spectra -> noise can be simulated as white noise
    - Gaussian amplitudes -> simulated noise can be Gaussian
    - average spike shape -> window lengths, and which classes are easy to confuse

Run from the project root:

    python Script/analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis

from config import BASELINE_WINDOW, DATA_DIR, FS
from data import load_data, remove_baseline

PREVIEW_SAMPLES = 5_000        # samples shown in raw-signal plots (0.2 s)
DRIFT_SAMPLES = 100_000        # samples shown in the drift plots (4 s)
HIGH_PASS_HZ = 4_000           # above the spike band, so what remains is mostly noise
AVG_PRE, AVG_POST = 40, 150    # window for the average spike shape


def plot_baseline_drift(recordings):
    """Plot the start of each recording with its slow (median-filtered) baseline."""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    for n, (ax, sig) in enumerate(zip(axes.flatten(), recordings), 1):
        segment = sig[:DRIFT_SAMPLES]
        baseline = segment - remove_baseline(segment, size=BASELINE_WINDOW)
        ax.plot(segment, lw=0.5, label=f'D{n} raw')
        ax.plot(baseline, 'r', label='slow baseline')
        ax.set_title(f'D{n}')
        ax.legend()
    fig.tight_layout()


def plot_power_spectra(recordings):
    """Plot the power spectral density of each recording (flat = white noise)."""
    plt.figure()
    for n, sig in enumerate(recordings, 1):
        freqs, power = welch(sig, fs=FS, nperseg=4096)
        plt.semilogy(freqs, power, label=f'D{n}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density')
    plt.title('Power spectra of D1-D6')
    plt.legend()


def check_gaussian_noise(recordings):
    """Print the excess kurtosis of each high-passed recording and plot D6's histogram.

    High-passing removes most of the spike energy, leaving mainly noise.
    Excess kurtosis near 0 means the noise is close to Gaussian.
    """
    b, a = butter(4, HIGH_PASS_HZ / (FS / 2), btype='high')
    for n, sig in enumerate(recordings, 1):
        print(f'D{n}: excess kurtosis {kurtosis(filtfilt(b, a, sig)):.3f}')

    plt.figure()
    plt.hist(filtfilt(b, a, recordings[5]), bins=200, density=True)
    plt.xlabel('Amplitude after high-pass (mV)')
    plt.ylabel('Probability density')
    plt.title('D6 noise amplitude distribution')


def plot_raw_recordings(recordings, index1):
    """Plot the first samples of each recording; D1 also shows its labelled spike starts."""
    for n, sig in enumerate(recordings, 1):
        plt.figure()
        plt.plot(sig[:PREVIEW_SAMPLES])
        if n == 1:                      # only D1 has labels
            shown = index1[index1 < PREVIEW_SAMPLES]
            plt.plot(shown, sig[shown], 'rx', label='labelled spike start')
            plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (mV)')
        plt.title(f'D{n}: first {PREVIEW_SAMPLES} samples')


def plot_average_spikes(d1, index1, class1):
    """Plot the mean spike shape of each class in D1."""
    ok = (index1 >= AVG_PRE) & (index1 < len(d1) - AVG_POST)   # windows must fit
    index1, class1 = index1[ok], class1[ok]
    plt.figure()
    for k in range(1, 6):
        windows = np.array([d1[t - AVG_PRE : t + AVG_POST] for t in index1[class1 == k]])
        plt.plot(np.arange(-AVG_PRE, AVG_POST), windows.mean(axis=0), label=f'class {k}')
    plt.axvline(0, color='k', ls=':')
    plt.xlabel('Samples from spike start')
    plt.ylabel('Amplitude (mV)')
    plt.title('Average spike per class (D1)')
    plt.legend()


def main():
    """Load the data, print summary statistics and show all exploratory plots."""
    recordings, index1, class1 = load_data(DATA_DIR)
    classes, counts = np.unique(class1, return_counts=True)
    print(f'D1: {len(recordings[0])} samples, {len(index1)} labelled spikes')
    print('Spikes per class:', dict(zip(classes.tolist(), counts.tolist())))

    plot_baseline_drift(recordings)
    plot_power_spectra(recordings)
    check_gaussian_noise(recordings)
    plot_raw_recordings(recordings, index1)
    plot_average_spikes(recordings[0], index1, class1)
    plt.show()


if __name__ == '__main__':
    main()
