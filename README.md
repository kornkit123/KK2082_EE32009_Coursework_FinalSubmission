# EE32009 Coursework: Spike Detection and Classification

Detects and classifies neuronal spikes (five neuron classes) in extracellular
recordings sampled at 25 kHz. The system is trained on the labelled recording D1
and applied to the unlabelled, noisier recordings D2-D6. For each recording it
produces an `Index` vector (spike start positions) and a `Class` vector (1-5).

## Quick start

```bash
pip install -r requirements.txt
# put D1.mat ... D6.mat in Data/
python Script/main.py
```

This loads the trained weights in `models/` and writes `submission/D2.mat` to
`submission/D6.mat` in a few minutes on a CPU. No training is needed.

## Method summary

1. **Drift removal.** A 2501-sample (~0.1 s) median filter estimates the slow
   baseline, which is subtracted from every recording (D1-D6) before anything else.
2. **Noise estimation.** The noise level of each recording is measured with a
   robust median-absolute-deviation estimate. A spectrum and kurtosis check
   (`analysis.py`) showed the noise is white and Gaussian, so it can be simulated.
3. **Training data.** Spike, background and hard-negative windows are cut from
   clean D1. Fresh Gaussian noise at a randomly chosen D1-D6 level is added to
   every training batch, so the models never see the same noise twice. Levels are
   sampled with weights that rise with noise (D1 0.05 up to D6 0.30), roughly
   matching the marking weights. Class 3 spikes are repeated three times in the
   detector's positive set because they are the smallest and easiest to miss.
4. **Two 1-D CNNs** (`SpikeCNN` in `network.py`), same architecture:
   - detector: is a spike starting at this sample? (65-sample window, 2 outputs)
   - classifier: which of the 5 neuron classes? (105-sample window, 5 outputs)
5. **Peak picking.** `find_peaks` on the detector probabilities, with a
   threshold and minimum spacing chosen per dataset from the validation sweep
   (`DETECTION_SETTINGS` in `config.py`).
6. **Classification** of each detected spike, then export to `.mat`.

## Project structure

```
Data/                 D1.mat ... D6.mat (from Moodle)
models/               Saved weights: clf_v.3.pt (classifier), det_v.3.pt (detector)
Script/
    main.py           Full pipeline: prepare data, train/load, validate, export
    config.py         All constants and settings (paths, window sizes, training, detection)
    data.py           Loading, drift removal, noise estimation/simulation, windowing
    network.py        SpikeCNN, the 1-D CNN used for detection and classification
    training.py       Tensor conversion, noise augmentation, training loop
    inference.py      Running the trained models over a full recording
    scoring.py        Local copy of the automarker scoring (matching, F1, F_dataset)
    analysis.py       Exploratory plots that motivated the design (not needed to run)
tests/                pytest unit tests for every module
submission/           Exported D2.mat ... D6.mat (created by main.py)
Assets/               Figures and notes (not needed to run the code)
submission_v*/        Earlier submissions and automarker feedback
```

Data flows through the modules like this:

```
data.py ──> training.py ──> network.py (trained weights in models/)
   │                                          │
   └──────────────> inference.py <────────────┘ ──> submission/*.mat
                         │
                    scoring.py  (validation only)
```

## Requirements

- Python 3.10 or newer
- Libraries in `requirements.txt`: numpy, scipy, matplotlib, scikit-learn, torch, pytest

A GPU is optional. The code uses CUDA if available and falls back to the CPU.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Place `D1.mat` to `D6.mat` in the `Data/` folder. All paths are set relative to
the project folder in `Script/config.py`, so nothing needs editing.

## Running

```bash
python Script/main.py        # full pipeline -> submission/D2.mat ... D6.mat
python Script/analysis.py    # optional exploratory plots
python -m pytest             # run the unit tests (a few seconds)
```

### Settings

Run switches at the top of `Script/main.py`:

| Setting | Meaning |
|---|---|
| `RETRAIN = False` | Load saved weights from `models/` (default; trains automatically if none exist) |
| `RETRAIN = True` | Train both models from scratch and save them (slow without a GPU) |
| `RUN_SWEEP = True` | Score every threshold/spacing pair on validation data, print the best per dataset |

Everything else is in `Script/config.py`, including:

| Setting | Meaning |
|---|---|
| `DETECTION_SETTINGS` | `{dataset: (threshold, min spike distance)}` used at export |
| `MODEL_TAG` | Suffix of the weight files, so versions do not overwrite each other |
| `NOISE_LEVEL_WEIGHTS` | How often each noise level D1..D6 is drawn during training |
| `CLASS3_REPEATS` | Extra copies of Class 3 spikes in the detector training set |
| `DET_PRE/POST`, `CLS_PRE/POST` | Window sizes around a spike start |

Typical tuning workflow:

1. Set `RETRAIN = True` and `RUN_SWEEP = True`, and change `MODEL_TAG` so the old
   weights are kept. Run, then read the best `(threshold, distance)` per dataset
   from the sweep output.
2. Copy those values into `DETECTION_SETTINGS`.
3. Set both switches back to `False` to re-export quickly.

Retrain whenever the preprocessing, model architecture, noise augmentation or
window sizes change. Changing only `DETECTION_SETTINGS` does not need a retrain.

## Output

`submission/D2.mat` to `submission/D6.mat`, each containing only:

- `Index`: spike start positions in samples
- `Class`: class labels 1-5, one per spike

Zip the `.mat` files (directly in the ZIP, no subfolder) and upload to Moodle.
The raw signal `d` is deliberately not included.

The console also prints the spike count and per-class counts for each dataset,
which is a quick sanity check against the 2,432 spikes in D1.

## Testing

`tests/` contains 40 pytest tests (run with `python -m pytest`, a few seconds,
no data files needed). They check:

- **data.py**: drift removal keeps spikes but removes offsets and slow drift; the
  noise estimate is accurate and robust to spikes; simulated noise reaches the
  target level; windows, background and hard-negative positions are correct.
- **scoring.py**: matching follows the automarker rules (±50 samples, one-to-one,
  closest wins), and F1 and F_dataset give the expected values for hand-worked cases.
- **network.py / training.py**: output shapes for both window lengths; the saved
  weights load into the current architecture; noise augmentation; the training
  loop learns a simple task.
- **inference.py / main.py**: on a synthetic recording with known spikes and
  "stub" models of known behaviour, every spike and class is recovered, close
  spikes are merged by the minimum spacing, and the exported files hold only
  `Index` and `Class`.

## Results log

### Automarker feedback (detection precision / recall)

| Submission | Change | D2 P / R | D3 P / R | D4 P / R | D5 P / R | D6 P / R |
|---|---|---|---|---|---|---|
| 1 | Baseline CNNs, no drift removal | 0.970 / 0.912 | 0.958 / 0.875 | 0.944 / 0.794 | 0.322 / 0.717 | 0.132 / 0.592 |
| 2 | Drift removal | 1.000 / 0.942 | 0.996 / 0.938 | 0.995 / 0.832 | 0.979 / 0.784 | 0.915 / 0.741 |
| 3 | Graded noise weights, Class 3 x3 | 1.000 / 0.947 | 0.999 / 0.935 | 0.985 / 0.864 | 0.981 / 0.775 | 0.910 / 0.741 |

Submission 1 returned 6,437 spikes for D5 and 14,148 for D6, against roughly
2,900 and 3,200 real spikes. Most of those were false detections caused by
baseline drift, which grows with the noise level. After drift removal the counts
fell to 2,315 and 2,556.

### Validation comparison of training variants

Best `F_dataset` (0.3 x detection F1 + 0.7 x classification F1) per dataset from
the validation sweep. Validation uses the held-out last 20% of D1 with simulated
noise, with `thr` and `dist` re-tuned for every variant. Differences below about
0.005 are within run-to-run variation.

| Variant | D2 | D3 | D4 | D5 | D6 | Est. F_Final |
|---|---|---|---|---|---|---|
| No Class 3 oversampling, equal noise weights | - | - | 0.871 | 0.741 | 0.622 | - |
| Class 3 x3, equal noise weights | 0.960 | 0.959 | 0.894 | 0.763 | 0.632 | 0.799 |
| **Class 3 x3, graded weights (0.05 to 0.30)** | 0.961 | 0.958 | 0.894 | 0.766 | **0.645** | **0.804** |
| Class 3 x5, graded weights | 0.961 | 0.957 | 0.893 | 0.752 | 0.623 | 0.793 |
| Class 3 x3, high-noise weights (D1, D2 = 0) | 0.959 | 0.958 | 0.903 | 0.770 | 0.632 | 0.802 |

Estimated F_Final = 0.1 x D2 + 0.15 x D3 + 0.2 x D4 + 0.25 x D5 + 0.3 x D6, using
the validation scores as a stand-in for the automarker's.

Findings:

- **Class 3** has the weakest and smallest spike (peak about 1.5, against 2.7 to
  5.4 for the others). At D5 and D6 noise its amplitude is about the noise level,
  so detector recall for it stays around 0.1 whatever the training. Lowering the
  threshold recovers some Class 3 spikes but precision falls faster, so the best
  F1 accepts the misses.
- **Oversampling Class 3 x3** helped at D4, where the spike is still visible
  (Class 3 recall roughly doubled). x5 was no better.
- **Noise weighting** made no meaningful difference. Three weightings land within
  0.002 of each other on F_Final, so training distribution is not the bottleneck.
- **Remaining loss** is classification at D6: about 30% of found spikes get the
  wrong label (mainly Class 4 vs 5, and 1 vs 2).

## Notes and limitations

- Validation noise is simulated white Gaussian noise added to the last 20% of D1.
  It is optimistic compared with the real D2-D6 recordings, so the automarker
  feedback is the final check.
- Overlapping spikes closer than the minimum spacing are merged into one detection.
- Classification accuracy falls at high noise (D5, D6), mainly through confusion
  between Classes 4 and 5, and between Classes 1 and 2.
