"""Project-wide constants for the spike sorting pipeline.

Every tunable number lives here so it can be found and changed in one place,
rather than being scattered through the code as unexplained "magic numbers".
"""
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Paths (relative to the project root, so the code runs on any machine)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'Data'               # D1.mat ... D6.mat
MODEL_DIR = PROJECT_ROOT / 'models'            # saved network weights
SUBMISSION_DIR = PROJECT_ROOT / 'submission'   # exported D2.mat ... D6.mat

MODEL_TAG = 'v.3'   # suffix of the weight files, e.g. models/clf_v.3.pt

# ----------------------------------------------------------------------
# Recording properties
# ----------------------------------------------------------------------
FS = 25_000          # sample rate in Hz (given in the brief)

# ----------------------------------------------------------------------
# Window sizes, in samples relative to the spike start (Index)
# ----------------------------------------------------------------------
# Detector: short window, just enough to see the spike rise and peak.
DET_PRE, DET_POST = 15, 50          # 65 samples = 2.6 ms
# Classifier: longer window that also includes the undershoot after the peak,
# which is where Classes 1 and 2 differ most.
CLS_PRE, CLS_POST = 15, 90          # 105 samples = 4.2 ms

# ----------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------
BASELINE_WINDOW = 2501   # median filter length (~0.1 s); much longer than a spike,
                         # so it follows the slow drift but ignores the spikes
MAD_TO_STD = 1.4826      # converts median absolute deviation to std for Gaussian noise

# ----------------------------------------------------------------------
# Training data preparation (D1)
# ----------------------------------------------------------------------
SEED = 0                    # random seed for numpy and torch (reproducible runs)
TRAIN_FRACTION = 0.8        # first 80% of D1 trains, last 20% validates
EDGE_MARGIN = 30            # extra samples kept clear of edges/split, so jittered windows fit
JITTER = 10                 # +/- samples of random shift applied to training spike windows
N_BACKGROUND_CANDIDATES = 30_000   # random positions drawn as possible "no spike" windows
BACKGROUND_MIN_GAP = 100    # a background window must be this far from every spike
NEGATIVES_PER_SPIKE = 3     # background windows kept per spike window
HARD_NEG_MIN_OFFSET = 15    # hard negatives sit 15-60 samples from a real spike:
HARD_NEG_MAX_OFFSET = 60    # close enough to look similar, far enough to be "wrong timing"
CLASS3_REPEATS = 3          # extra copies of Class 3 spikes (smallest, most often missed)

# Probability of drawing each noise level D1..D6 when augmenting a training batch.
# Rises with noise, roughly following the marking weights of D2..D6.
NOISE_LEVEL_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
BATCH_SIZE = 256
CLF_EPOCHS, DET_EPOCHS = 90, 70     # upper limits; early stopping usually ends sooner
PATIENCE = 12                       # epochs without improvement before stopping

# ----------------------------------------------------------------------
# Detection and scoring
# ----------------------------------------------------------------------
MATCH_TOLERANCE = 50     # +/- samples, the window the automarker uses

# Detection settings per dataset, chosen from the validation sweep in main.py:
# {dataset number: (probability threshold, minimum distance between spikes in samples)}
DETECTION_SETTINGS = {2: (0.85, 15), 3: (0.9, 15), 4: (0.9, 25), 5: (0.85, 25), 6: (0.7, 25)}

# Marking weights from the brief.
F_IDENT_WEIGHT, F_CLASS_WEIGHT = 0.3, 0.7
DATASET_WEIGHTS = {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
