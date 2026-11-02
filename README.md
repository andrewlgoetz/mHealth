# Cross-dataset human activity recognition

Data preparation for the group proposal: compare UCI HAR and WISDM in both transfer directions, then study adaptation with a small labelled target sample. The code uses ordinary Python files so everyone can work on separate parts of the project.

## Run the preparation

Use Python 3.11 or newer. From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe data_process.py
.\.venv\Scripts\python.exe -m unittest discover -v
```

The datasets have already been downloaded and prepared locally. Downloads and generated arrays live in `data/`, which is ignored by Git. Each teammate can reproduce them with these commands. Downloading again reuses the archives and checks their SHA-256 hashes against `reports/data_sources.json`. The original READMEs and other release documentation are retained under `data/raw/`.

To change the window length, overlap, subject split seed, or maximum WISDM interpolation gap:

```powershell
.\.venv\Scripts\python.exe data_process.py --window-seconds 5 --overlap 0.5 --seed 42 --max-gap-seconds 0.5
```

This replaces the generated datasets and summary reports. Sampling rate stays at 20 Hz. Window and step lengths are rounded to whole samples; the reports record the actual durations.

## Files

| File | Purpose |
| --- | --- |
| `download_data.py` | Download and extract the official releases; record hashes and source URLs. |
| `data_process.py` | Read signals, clean recordings, harmonize units and sampling, make windows and subject splits. |
| `features.py` | Extract the same 76 features from either dataset. |
| `model_data.py` | Load arrays and prepare transfer experiments using source training statistics. |
| `test_features.py`, `test_data_process.py` | Check known signals, recording boundaries, subject splits, and scaling. |
| `reports/` | Download provenance, cleaning counts, class counts, and the exact subject assignments. |

## Datasets and preprocessing

The sources are [UCI HAR, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and [Fordham WISDM Activity Prediction v1.1](https://www.cis.fordham.edu/wisdm/dataset.php). WISDM here is the original six-activity smartphone release described in the proposal. The downloader records the archive checksum because the WISDM download URL uses the name `latest`.

Both datasets use only total x/y/z acceleration, including gravity. Gyroscope readings and the publishers' precomputed features are excluded. UCI acceleration is converted from g to m/s² using 9.80665; WISDM is already in m/s². Axis order is x, y, z. Device orientation and placement differences remain for the experiments to measure.

The shared label mapping is fixed:

| Label | Activity |
| --- | --- |
| 0 | walking |
| 1 | upstairs |
| 2 | downstairs |
| 3 | sitting |
| 4 | standing |

UCI laying and WISDM jogging are excluded. Every window has one subject, one activity, and one continuous recording segment. Incomplete final windows are discarded; there is no padding, class balancing, or per-window normalization.

**UCI HAR:** The original release contains filtered, overlapping 128-sample windows at 50 Hz, rather than continuous raw recordings. Adjacent rows are joined only when their subject and label agree and the previous final 64 samples exactly match the next first 64 samples on all axes. Only the new 64 samples are appended. This check detects 457 same-subject, same-label discontinuities that would otherwise create artificial recordings. Each verified run is downsampled to 20 Hz with an anti-aliasing polyphase FIR filter before making the shared windows. Time starts at zero within each reconstructed segment; original wall-clock times are unavailable. Longer windows are reconstructed from this release, rather than substituted from a different UCI dataset.

**WISDM:** Semicolon-delimited records are read in their original order, including multiple records on a line. One extra trailing comma is repaired when all six fields are present. Consecutive records with identical subject, activity, timestamp, and acceleration are removed. Malformed readings, nonpositive timestamps, backwards clocks, conflicting readings at the same timestamp, activity or subject changes, and gaps over 0.5 seconds end a segment. No sorting joins separate sessions. Each remaining segment is interpolated onto a uniform 20 Hz timestamp grid without extrapolation. This uses elapsed time rather than assuming that every 100 source rows span five seconds. The default maximum interpolation gap is configurable and is a preprocessing assumption to revisit in sensitivity analyses.

## Prepared data

Default configuration: 5-second windows (100 samples), 2.5-second step (50% overlap), seed 42.

| Dataset | Subjects | Train windows | Validation windows | Test windows | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI HAR | 30 | 2,018 | 557 | 1,034 | 3,609 |
| WISDM | 36 | 8,368 | 3,088 | 2,595 | 14,051 |

UCI keeps its official nine test subjects. Four of its 21 training subjects are reserved for validation, leaving 17 for training. WISDM uses 22 training, seven validation, and seven test subjects. Subjects are disjoint within each dataset, so overlapping windows cannot cross splits. Subject identifiers are local to each dataset: UCI subject 1 and WISDM subject 1 are different people. All five classes occur in each split. Use the subject identifiers as groups for later subject-level cross-validation; changing `--seed` changes the validation selection and WISDM split, while preserving UCI's official test subjects.

The final WISDM cleaning removed 41,867 exact repeated records and 12,846 invalid records (12,845 nonpositive timestamps and one missing acceleration value), and repaired 11,733 trailing commas. Its retained windows remain imbalanced: walking 8,095; upstairs 2,173; downstairs 1,754; sitting 1,111; standing 918. Keep macro F1 and per-class metrics in the model evaluation. Full counts are in the JSON reports.

Each `data/processed/{uci_har,wisdm}/` folder contains:

| File | Contents |
| --- | --- |
| `dataset.npz` | `X` (windows × 100 × 3), `features` (windows × 76), `y`, `subjects`, `split`, `segment_ids`, `start_seconds`, `feature_names`. Model inputs are float32; labels are integers. |
| `features.csv` | The same unscaled feature matrix with named columns, in NPZ row order. |
| `metadata.csv` | Subject, label, activity, segment, start time within the segment, original partition, and assigned split, in the same row order. |
| `segments.csv` | Segment provenance, source starting record, resampled sample count, and number of output windows; includes segments too short for a window. WISDM also records the original starting timestamp in nanoseconds. |

Arrays are saved without Python objects and load with `allow_pickle=False`. Subject IDs, segment IDs, timestamps, and split names are metadata; do not include them as model features.

## Features

Features are computed for x, y, z, and acceleration magnitude. Each channel has mean, population standard deviation, min, max, median, interquartile range, RMS, mean absolute deviation about the mean, mean-square energy, demeaned zero-crossing rate, range, skewness, and excess kurtosis. Frequency features use a demeaned signal and a Hann-windowed FFT, with DC excluded: dominant frequency, spectral centroid, normalized spectral entropy, and power fractions in [0.1, 3) Hz and [3, 10) Hz. Power fractions use all non-DC power as their denominator, including the Nyquist bin. The remaining four features are the three axis correlations and signal magnitude area, defined as mean(|x| + |y| + |z|). Constant-signal spectral, standardized-moment, and correlation features return zero.

## Start model development

Load unscaled inputs for tree models, or fit your own preprocessing using training subjects:

```python
from model_data import load_dataset

train = load_dataset("uci_har", split="train")
val = load_dataset("uci_har", split="val")
test = load_dataset("uci_har", split="test")

X_train, y_train = train["features"], train["y"]  # classical models
windows_train = train["X"]                     # CNN/LSTM: (N, 100, 3)
groups = train["subjects"]                     # subject-level CV
```

For a transfer experiment with optional standardization already applied:

```python
from model_data import load_transfer

experiment = load_transfer("uci_har", "wisdm")
train = experiment["source_train"]
val = experiment["source_val"]
source_test = experiment["source_test"]
target_all = experiment["target_all"]
target_adapt = experiment["target_adapt"]
target_test = experiment["target_test"]
scaler = experiment["scaler"]
```

The helper fits channel and feature means/standard deviations on source training windows only, and applies those same values to every source and target partition. Reverse the dataset arguments for WISDM → UCI. The saved arrays remain unscaled. When using cross-validation, fit new scalers inside each training fold.

`target_all` supports the proposal's full-target zero-shot evaluation. For adaptation experiments, sample labelled target windows only from `target_adapt` (target train + validation subjects), and compare adapted and unadapted models on the identical `target_test` subjects. Fit target-domain normalization or alignment on the adaptation pool as well. The target test labels must remain unused for training, tuning, or alignment. Exact adaptation label budgets and repeated sampling seeds belong in the later model experiments; all required labels and subject groups are available here.

The same preprocessing supports statistical-feature classifiers and raw-window neural models. No models are trained by these scripts.

## Dataset citations

- Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN. Dataset DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
- Kwapisz, J. R., Weiss, G. M., and Moore, S. A. (2010). *Activity Recognition using Cell Phone Accelerometers*. KDD Sensor Data workshop. [Original paper](https://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf).

Retain the supplied READMEs when redistributing the data, as requested by the publishers.
