# Cross-dataset activity recognition

A basic workflow for comparing UCI HAR and WISDM. Both datasets get the same
20 features, five activity labels, acceleration units, and 2.5-second windows.

## Run it

Use Python 3.11 or newer. From this folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe data_process.py
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe sample_data.py
```

The raw data are already downloaded on this machine. Preparation creates:

- `data/processed/uci_har.npz`
- `data/processed/wisdm.npz`

`check_data.py` runs the unit tests and checks both actual saved datasets. It
checks shapes, labels, finite values, subject separation, matching feature names,
and whether saved features match their windows. A failure exits with an error.
`sample_data.py` prints one training example per activity with every feature
visible. Use `sample_data.py --rows 2` for a smaller sample.

## What each file does

| File | Purpose |
| --- | --- |
| `data_process.py` | Read and clean data, make windows, split subjects, save arrays. |
| `features.py` | Calculate the same 20 features for either dataset. |
| `check_data.py` | Run tests and check the prepared data. |
| `sample_data.py` | Print training examples with all features and their labels. |
| `test_data_process.py`, `test_features.py` | Small tests with known input signals. |

## Data sources

On another machine, download [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
and [WISDM Activity Prediction v1.1](https://www.cis.fordham.edu/wisdm/dataset.php).
Extract the archives so these paths exist (the UCI download may contain another ZIP):

```text
data/raw/uci_har/UCI HAR Dataset/train/
data/raw/uci_har/UCI HAR Dataset/test/
data/raw/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
```

Use the original six-activity WISDM smartphone release. Keep the publishers'
READMEs with the data. The `data/` folder is ignored by Git; teammates prepare
their own local copies.

## Preprocessing choices

- Use total acceleration, including gravity, in x/y/z order. Convert UCI from g
  to m/s^2 using 9.80665. WISDM already uses m/s^2. Gyroscopes are excluded.
- Keep five shared activities: `0=walking`, `1=upstairs`, `2=downstairs`,
  `3=sitting`, `4=standing`. Exclude UCI laying and WISDM jogging.
- For UCI, use the first 125 samples of each released window (2.5 seconds at
  50 Hz). Downsample with an anti-aliasing filter to 50 samples at 20 Hz.
  Each released window is handled independently; no recording reconstruction.
- For WISDM, repair an extra trailing comma and remove exact repeated readings.
  Invalid readings, clock resets, conflicting timestamps, subject/activity
  changes, and gaps over 0.5 seconds end a recording. Preserve file order.
  Interpolate within each recording to 20 Hz, then take 50-sample windows with
  a 25-sample step. Discard incomplete windows; do not extrapolate or pad.
- Split by subject with seed 42. UCI keeps its official nine test subjects,
  with four validation and 17 training subjects. WISDM has seven test, seven
  validation, and 22 training subjects. Subject IDs belong to their own dataset.
- Save unscaled inputs. Do not fit normalization on test or target data when
  measuring unadapted transfer. Fit any scaler on source training data only,
  then apply it to both datasets. Keep per-window normalization out of this
  baseline because it removes potentially useful gravity/orientation signals.

The same window duration and features support both transfer directions.
WISDM remains imbalanced, so evaluate macro F1 and per-class results alongside
accuracy. This preparation checks data quality; it does not establish model accuracy.

## Changes from the proposal and previous implementation

1. **2.5-second windows instead of the suggested 5 seconds.** This fits inside
   UCI's released windows and removes all stitching logic. Frequency resolution
   is now 0.4 Hz. Both datasets use 50 readings per window at 20 Hz.
2. **Approximately matching overlap.** UCI retains its released 1.28-second
   stride, giving 48.8% overlap after cropping; WISDM uses a 1.25-second stride
   and 50% overlap. The slight stride difference avoids reconstructing UCI.
3. **20 simple features instead of 76.** For x, y, z, and magnitude, calculate
   mean, population standard deviation, RMS, range, and dominant frequency.
   Frequency is calculated after demeaning and Hann tapering, excluding DC;
   constant signals return zero frequency. Magnitude is sqrt(x^2 + y^2 + z^2).
4. **Less machinery.** Download/extraction automation, checksum tracking,
   configurable window settings, transfer/scaling helpers, detailed reports,
   and duplicate CSV exports have been removed. Raw windows, labels, subjects,
   and train/validation/test splits remain available for later experiments.

## Load the training data

Each NPZ contains `windows`, `features`, `labels`, `subjects`, `split`, and
`feature_names`. Rows correspond across all arrays. Windows have shape
`(N, 50, 3)` and features have shape `(N, 20)`; names are in the same order for
both datasets. Subject IDs and labels are metadata, not input features.

```python
import numpy as np

with np.load("data/processed/uci_har.npz", allow_pickle=False) as data:
    train = data["split"] == "train"
    X_train = data["features"][train]
    y_train = data["labels"][train]
    subjects_train = data["subjects"][train]
    windows_train = data["windows"][train]  # For a CNN/LSTM later.
    feature_names = data["feature_names"]
```

Use `wisdm.npz` for the other dataset and `val` or `test` for held-out subjects.
For zero-shot transfer, evaluate on the other dataset without fitting to it.
For adaptation, use target train/validation subjects and reserve target test
subjects for evaluation; compare adapted and unadapted models on that same test set.

Current preparation produces 8,355 UCI windows and 29,168 WISDM windows:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| UCI HAR | 4,683 | 1,262 | 2,410 |
| WISDM | 17,331 | 6,396 | 5,441 |

Cite Anguita et al. (2013), *A Public Domain Dataset for Human Activity
Recognition Using Smartphones*, and Kwapisz, Weiss, and Moore (2010), *Activity
Recognition using Cell Phone Accelerometers*, when reporting results.
