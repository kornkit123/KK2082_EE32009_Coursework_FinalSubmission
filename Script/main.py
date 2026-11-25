"""EE32009 Coursework: spike detection and classification pipeline.

Run from the project root:

    python Script/main.py

Pipeline:
    1. Load D1-D6 and remove baseline drift
    2. Measure the noise level of every recording
    3. Split labelled D1 into training and validation parts (by time)
    4. Build training windows (clean D1) and validation windows (D1 + simulated noise)
    5. Set up normalisation and the per-batch noise augmentation
    6. Train the detector and classifier, or load saved weights
    7. Report classifier accuracy at each noise level
    8. (Optional) sweep detection threshold and spacing on validation data
    9. Detect and classify spikes in D2-D6 and export the .mat files
"""
import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

import data
import inference
import scoring
import training
from config import (CLASS3_REPEATS, CLS_POST, CLS_PRE, CLF_EPOCHS, DATA_DIR, DATASET_WEIGHTS,
                    DET_EPOCHS, DET_POST, DET_PRE, DETECTION_SETTINGS, DEVICE, EDGE_MARGIN,
                    JITTER, MODEL_DIR, MODEL_TAG, N_BACKGROUND_CANDIDATES, BACKGROUND_MIN_GAP,
                    NEGATIVES_PER_SPIKE, NOISE_LEVEL_WEIGHTS, PATIENCE, SEED, SUBMISSION_DIR,
                    TRAIN_FRACTION)
from network import SpikeCNN

# ----------------------------------------------------------------------
# Run switches
# ----------------------------------------------------------------------
RETRAIN = False     # True: train both models and save them. False: load saved weights
                    # (if no saved weights exist, the models are trained anyway).
RUN_SWEEP = False   # True: run the threshold/spacing sweep (slow, only needed when tuning).

# Thresholds and spacings tried by the sweep
SWEEP_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98)
SWEEP_DISTANCES = (15, 25, 40)


# ----------------------------------------------------------------------
# 1-2. Loading and noise levels
# ----------------------------------------------------------------------
def load_recordings():
    """Load D1-D6, remove baseline drift from every one, and return them with D1's labels."""
    recordings, index1, class1 = data.load_data(DATA_DIR)
    # Same cleaning for training (D1) and test (D2-D6) data
    recordings = [data.remove_baseline(sig) for sig in recordings]
    return recordings, index1, class1


def measure_noise_levels(recordings):
    """Print and return the estimated noise std of each recording.

    Returns:
        base:   noise std of D1.
        levels: list of noise std for D1..D6 (levels[0] == base).
    """
    levels = [data.noise_std(sig) for sig in recordings]
    for n, level in enumerate(levels, 1):
        print(f'D{n} noise std: {level:.4f}')
    return levels[0], levels


# ----------------------------------------------------------------------
# 3. Train / validation split of D1
# ----------------------------------------------------------------------
def split_d1(n_samples, index1, class1, rng):
    """Split D1 by time into training (first 80%) and validation (last 20%) positions.

    A gap of one window plus EDGE_MARGIN is left on each side of the split
    point, so no training window overlaps a validation window.

    Returns:
        dict with spike positions/classes and background/hard-negative
        positions for each part, plus the split point.
    """
    split = int(TRAIN_FRACTION * n_samples)

    # Keep only spikes whose (jittered) classifier window fits inside the recording
    ok = (index1 > CLS_PRE + EDGE_MARGIN) & (index1 < n_samples - CLS_POST - EDGE_MARGIN)
    index1, class1 = index1[ok], class1[ok]

    train_end = split - CLS_POST - EDGE_MARGIN        # last usable training position
    val_start = split + CLS_PRE + EDGE_MARGIN         # first usable validation position
    is_train, is_val = index1 < train_end, index1 >= val_start

    # Background ("no spike") positions, split the same way
    background = data.background_positions(
        index1, CLS_PRE + EDGE_MARGIN, n_samples - CLS_POST - EDGE_MARGIN,
        N_BACKGROUND_CANDIDATES, BACKGROUND_MIN_GAP, rng)
    n_train_spikes, n_val_spikes = is_train.sum(), is_val.sum()

    parts = {
        'split': split,
        'index': index1, 'class': class1,                       # all usable D1 spikes
        'train_index': index1[is_train], 'train_class': class1[is_train],
        'val_index': index1[is_val], 'val_class': class1[is_val],
        'train_background': background[background < train_end][:NEGATIVES_PER_SPIKE * n_train_spikes],
        'val_background': background[background >= val_start][:NEGATIVES_PER_SPIKE * n_val_spikes],
    }
    # Hard negatives: 15-60 samples from a real spike (teach the detector exact timing)
    parts['train_hard_neg'] = data.hard_negative_positions(
        parts['train_index'], DET_PRE + EDGE_MARGIN, split - DET_POST - EDGE_MARGIN, index1, rng)
    parts['val_hard_neg'] = data.hard_negative_positions(
        parts['val_index'], split + DET_PRE + EDGE_MARGIN, n_samples - DET_POST - EDGE_MARGIN, index1, rng)
    return parts


# ----------------------------------------------------------------------
# 4. Training and validation windows
# ----------------------------------------------------------------------
def build_training_windows(d1, parts, rng):
    """Cut CLEAN training windows from D1 (noise is added later, per batch).

    Returns:
        (clf_x, clf_y): classifier windows and labels 0-4 (spikes only, jittered).
        (det_x, det_y): detector windows and labels (1 = spike start, 0 = not).
    """
    train_index, train_class = parts['train_index'], parts['train_class']

    clf_x = data.extract_windows(d1, train_index, CLS_PRE, CLS_POST, jitter=JITTER, rng=rng)
    clf_y = train_class.astype(int) - 1                   # labels 1-5 -> 0-4

    # Class 3 is the smallest spike and the one most often missed, so the detector
    # sees extra jittered copies of it
    class3_repeated = np.repeat(train_index[train_class == 3], CLASS3_REPEATS)

    det_x = np.concatenate([
        data.extract_windows(d1, train_index, DET_PRE, DET_POST, jitter=JITTER, rng=rng),
        data.extract_windows(d1, class3_repeated, DET_PRE, DET_POST, jitter=JITTER, rng=rng),
        data.extract_windows(d1, parts['train_background'], DET_PRE, DET_POST),
        data.extract_windows(d1, parts['train_hard_neg'], DET_PRE, DET_POST),
    ])
    det_y = np.concatenate([
        np.ones(len(train_index)),
        np.ones(len(class3_repeated)),
        np.zeros(len(parts['train_background'])),
        np.zeros(len(parts['train_hard_neg'])),
    ])
    return (clf_x, clf_y), (det_x, det_y)


def build_validation_windows(d1, parts, levels, base, rng):
    """Make FIXED noisy copies of D1, one per noise level D1..D6, and cut validation windows.

    Fixed (rather than fresh) noise means validation loss is comparable between epochs.

    Returns:
        clf_val_sets: list of dicts {'level', 'x', 'y'}, one per noise level.
        (det_x, det_y): detector validation windows and labels, all levels together.
    """
    val_index, val_class = parts['val_index'], parts['val_class']
    clf_val_sets, det_x, det_y = [], [], []
    for level in levels:
        sig = data.noisy_version(d1, level, base, rng)

        det_x.append(np.concatenate([
            data.extract_windows(sig, val_index, DET_PRE, DET_POST),
            data.extract_windows(sig, parts['val_background'], DET_PRE, DET_POST),
            data.extract_windows(sig, parts['val_hard_neg'], DET_PRE, DET_POST),
        ]))
        det_y.append(np.concatenate([
            np.ones(len(val_index)),
            np.zeros(len(parts['val_background'])),
            np.zeros(len(parts['val_hard_neg'])),
        ]))
        clf_val_sets.append({'level': level,
                             'x': data.extract_windows(sig, val_index, CLS_PRE, CLS_POST),
                             'y': val_class - 1})
    return clf_val_sets, (np.concatenate(det_x), np.concatenate(det_y))


# ----------------------------------------------------------------------
# 5-6. Augmentation, training and loading
# ----------------------------------------------------------------------
def make_augmenter(levels, base, sigma):
    """Return a function that adds fresh noise at a random D1..D6 level to a batch.

    Noise is in normalised units (divided by sigma), because the batches are
    already normalised when the function is applied.
    """
    levels_extra = torch.tensor([np.sqrt(max(level**2 - base**2, 0)) for level in levels],
                                dtype=torch.float32, device=DEVICE) / sigma
    level_weights = torch.tensor(NOISE_LEVEL_WEIGHTS, device=DEVICE)
    return lambda xb: training.add_noise(xb, levels_extra, level_weights)


def model_paths():
    """File paths of the saved classifier and detector weights."""
    return MODEL_DIR / f'clf_{MODEL_TAG}.pt', MODEL_DIR / f'det_{MODEL_TAG}.pt'


def train_or_load_models(train_sets, val_sets, mu, sigma, augment, retrain):
    """Train and save both networks, or load previously saved weights.

    Args:
        train_sets: ((clf_x, clf_y), (det_x, det_y)) clean training windows.
        val_sets:   (clf_val_sets, (det_x, det_y)) validation windows.
        mu, sigma:  normalisation constants.
        augment:    per-batch noise function from make_augmenter.
        retrain:    force training even if saved weights exist.

    Returns:
        (classifier, detector), both on DEVICE in eval mode.
    """
    (clf_x, clf_y), (det_x, det_y) = train_sets
    clf_val_sets, (det_val_x, det_val_y) = val_sets
    clf_path, det_path = model_paths()

    classifier, detector = SpikeCNN(n_out=5), SpikeCNN(n_out=2)

    if retrain or not (clf_path.exists() and det_path.exists()):
        print('Training classifier...')
        clf_val_x = np.concatenate([v['x'] for v in clf_val_sets])
        clf_val_y = np.concatenate([v['y'] for v in clf_val_sets])
        classifier = training.train_model(
            classifier,
            training.make_loader(clf_x, clf_y, shuffle=True, mu=mu, sigma=sigma),
            training.make_loader(clf_val_x, clf_val_y, shuffle=False, mu=mu, sigma=sigma),
            nn.CrossEntropyLoss(), epochs=CLF_EPOCHS, patience=PATIENCE, augment=augment)

        print('Training detector...')
        detector = training.train_model(
            detector,
            training.make_loader(det_x, det_y, shuffle=True, mu=mu, sigma=sigma),
            training.make_loader(det_val_x, det_val_y, shuffle=False, mu=mu, sigma=sigma),
            nn.CrossEntropyLoss(), epochs=DET_EPOCHS, patience=PATIENCE, augment=augment)

        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(classifier.state_dict(), clf_path)
        torch.save(detector.state_dict(), det_path)
        print(f'Saved weights to {clf_path.name} and {det_path.name}')
    else:
        print(f'Loading saved weights {clf_path.name} and {det_path.name}')
        classifier.load_state_dict(torch.load(clf_path, map_location=DEVICE))
        detector.load_state_dict(torch.load(det_path, map_location=DEVICE))

    return classifier.to(DEVICE).eval(), detector.to(DEVICE).eval()


# ----------------------------------------------------------------------
# 7-8. Validation
# ----------------------------------------------------------------------
def validate_classifier(classifier, clf_val_sets, d1, parts, base, mu, sigma, rng):
    """Print classifier accuracy and confusion matrix at each noise level.

    Accuracy is reported twice: on perfectly aligned windows, and on windows
    shifted by up to +/- JITTER samples, which is closer to real detections.
    """
    for v in clf_val_sets:
        predicted = inference.classify_windows(classifier, v['x'], mu, sigma)
        aligned_acc = (predicted == v['y']).mean()

        sig = data.noisy_version(d1, v['level'], base, rng)
        jittered = data.extract_windows(sig, parts['val_index'], CLS_PRE, CLS_POST,
                                        jitter=JITTER, rng=rng)
        jittered_acc = training.accuracy(classifier, jittered, v['y'], mu, sigma)

        print(f"noise std {v['level']:.3f} | aligned acc {aligned_acc:.3f} "
              f"| jittered acc {jittered_acc:.3f}")
        print(confusion_matrix(v['y'], predicted))


def sweep_detection_settings(detector, classifier, d1, parts, levels, base, mu, sigma, rng):
    """Score every (threshold, spacing) pair on noisy validation copies of D1.

    Prints every result, then the best pair per dataset and the estimated
    F_Final. Copy the best pairs into DETECTION_SETTINGS in config.py.

    Returns:
        dict {dataset number: (best F_dataset, threshold, spacing)} for D2-D6.
    """
    start, stop = parts['split'] + DET_PRE + EDGE_MARGIN, len(d1) - DET_POST - EDGE_MARGIN
    val_index, val_class = parts['val_index'], parts['val_class']
    best = {}
    for n, level in enumerate(levels, 1):                 # n = 1 is D1, n = 2 is D2, ...
        sig = data.noisy_version(d1, level, base, rng)
        probs = inference.detection_probabilities(detector, sig, start, stop, mu, sigma)
        for min_distance in SWEEP_DISTANCES:
            for threshold in SWEEP_THRESHOLDS:
                peaks, _ = find_peaks(probs, height=threshold, distance=min_distance)
                index = peaks + start
                index = index[index < len(sig) - CLS_POST]   # classifier window must fit
                cls = inference.classify(sig, index, classifier, mu, sigma)
                f_id, f_cl, f_ds = scoring.dataset_score(index, cls, val_index, val_class - 1)
                recall = scoring.recall_per_class(index, val_index, val_class)
                recall_text = ' '.join(f'{r:.2f}' for r in recall.values())
                print(f'D{n} noise {level:.3f} | dist {min_distance} | thr {threshold} '
                      f'| F_ident {f_id:.3f} F_class {f_cl:.3f} F_dataset {f_ds:.3f} '
                      f'| class recall {recall_text}')
                if n >= 2 and f_ds > best.get(n, (-1,))[0]:
                    best[n] = (f_ds, threshold, min_distance)

    print('\nBest per dataset:')
    for n in sorted(best):
        print(f'D{n}: F_dataset {best[n][0]:.3f} at thr {best[n][1]}, dist {best[n][2]}')
    print(f'Estimated F_Final: {sum(DATASET_WEIGHTS[n] * best[n][0] for n in DATASET_WEIGHTS):.3f}')
    return best


# ----------------------------------------------------------------------
# 9. Export
# ----------------------------------------------------------------------
def export_submission(recordings, detector, classifier, mu, sigma,
                      settings=DETECTION_SETTINGS, out_dir=SUBMISSION_DIR):
    """Detect and classify spikes in D2-D6 and save D2.mat ... D6.mat.

    Each file holds only `Index` and `Class`. The raw signal `d` is left out,
    as the brief asks, to keep the files small.

    Args:
        recordings: list [D1, ..., D6] (baseline removed).
        settings:   {dataset number: (threshold, min spacing)}.
        out_dir:    folder to write the .mat files to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in range(2, 7):
        threshold, min_distance = settings[n]
        index, cls = inference.predict_dataset(recordings[n - 1], threshold, min_distance,
                                               detector, classifier, mu, sigma)
        print(f'D{n}: {len(index)} spikes, class counts {np.bincount(cls, minlength=6)[1:]}')
        spio.savemat(out_dir / f'D{n}.mat', {'Index': index.astype(np.int64),
                                             'Class': cls.astype(np.int64)})
    print(f'Submission files written to {out_dir}')


def main():
    """Run the full pipeline: load, prepare data, train/load, validate, export."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    recordings, index1, class1 = load_recordings()
    d1 = recordings[0]
    base, levels = measure_noise_levels(recordings)

    parts = split_d1(len(d1), index1, class1, rng)
    train_sets = build_training_windows(d1, parts, rng)
    clf_val_sets, det_val = build_validation_windows(d1, parts, levels, base, rng)

    # Every window is normalised as (x - mu) / sigma, in training and at prediction time
    mu, sigma = float(np.median(d1)), float(d1.std())
    augment = make_augmenter(levels, base, sigma)

    classifier, detector = train_or_load_models(train_sets, (clf_val_sets, det_val),
                                                mu, sigma, augment, RETRAIN)

    validate_classifier(classifier, clf_val_sets, d1, parts, base, mu, sigma, rng)
    if RUN_SWEEP:
        sweep_detection_settings(detector, classifier, d1, parts, levels, base, mu, sigma, rng)

    print(f"D1 reference: {len(parts['index'])} spikes")
    export_submission(recordings, detector, classifier, mu, sigma)


if __name__ == '__main__':
    main()
