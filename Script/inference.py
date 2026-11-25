"""Applying the trained detector and classifier to a full recording."""
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

from config import CLS_POST, CLS_PRE, DET_POST, DET_PRE, DEVICE
from data import extract_windows
from training import to_tensor


def detection_probabilities(detector, sig, start, stop, mu, sigma, batch=8192, device=DEVICE):
    """Probability that a spike starts at each sample t in [start, stop).

    The detector is slid along the recording one sample at a time. A strided
    view avoids copying the 1.44 M overlapping windows into memory; they are
    copied one batch at a time instead.

    Args:
        detector:  trained SpikeCNN with 2 outputs.
        sig:       1-D recording (baseline already removed).
        start:     first sample to score (must be >= DET_PRE).
        stop:      one past the last sample to score.
        mu, sigma: normalisation constants used in training.
        batch:     windows per forward pass (memory vs speed).
        device:    'cuda' or 'cpu'.

    Returns:
        1-D array of length stop - start; element i is P(spike starts at start + i).
    """
    view = sliding_window_view(sig, DET_PRE + DET_POST)   # view[k] = sig[k : k + L]
    detector.eval()                                       # so window k is centred on t = k + DET_PRE
    probs = []
    with torch.no_grad():
        for k in range(start - DET_PRE, stop - DET_PRE, batch):
            xb = to_tensor(view[k : min(k + batch, stop - DET_PRE)], mu, sigma).to(device)
            probs.append(torch.softmax(detector(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def classify_windows(classifier, windows, mu, sigma, batch=4096, device=DEVICE):
    """Predict the class (0-4) of each pre-cut classifier window.

    Args:
        classifier: trained SpikeCNN with 5 outputs.
        windows:    2-D array (n_windows, CLS_PRE + CLS_POST).
        mu, sigma:  normalisation constants used in training.
        batch:      windows per forward pass (keeps GPU memory bounded).
        device:     'cuda' or 'cpu'.

    Returns:
        1-D integer array of classes 0-4.
    """
    if len(windows) == 0:
        return np.array([], dtype=int)
    x = to_tensor(windows, mu, sigma)
    classifier.eval()
    with torch.no_grad():
        return np.concatenate([classifier(x[i:i + batch].to(device)).argmax(1).cpu().numpy()
                               for i in range(0, len(x), batch)])


def classify(sig, index, classifier, mu, sigma, device=DEVICE):
    """Predict the class of the spike starting at each position in `index`.

    Args:
        sig:        1-D recording (baseline already removed).
        index:      spike start positions (samples).
        classifier: trained SpikeCNN with 5 outputs.
        mu, sigma:  normalisation constants used in training.
        device:     'cuda' or 'cpu'.

    Returns:
        1-D integer array of classes 0-4 (add 1 for the 1-5 labels of the brief).
    """
    if len(index) == 0:
        return np.array([], dtype=int)
    windows = extract_windows(sig, index, CLS_PRE, CLS_POST)
    return classify_windows(classifier, windows, mu, sigma, device=device)


def predict_dataset(sig, threshold, min_distance, detector, classifier, mu, sigma):
    """Detect and classify every spike in a recording.

    Steps:
      1. Score every sample with the detector.
      2. Keep local maxima of the probability that exceed `threshold` and are
         at least `min_distance` samples apart (one detection per spike).
      3. Classify the window at each detection.

    Args:
        sig:          1-D recording (baseline already removed).
        threshold:    minimum detector probability for a detection.
        min_distance: minimum spacing between detections, in samples.
        detector:     trained SpikeCNN with 2 outputs.
        classifier:   trained SpikeCNN with 5 outputs.
        mu, sigma:    normalisation constants used in training.

    Returns:
        index: spike start positions (samples).
        cls:   class labels 1-5, one per spike.
    """
    # Only score positions whose classifier window fits inside the recording
    start, stop = CLS_PRE, len(sig) - CLS_POST
    probs = detection_probabilities(detector, sig, start, stop, mu, sigma)
    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance)
    index = peaks + start
    cls = classify(sig, index, classifier, mu, sigma) + 1     # 0-4 -> labels 1-5
    return index, cls
