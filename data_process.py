"""Prepare UCI HAR and WISDM with matching windows, labels, and features."""

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from features import extract_features

ROOT = Path(__file__).resolve().parent
UCI_DIR = ROOT / "data/raw/uci_har/UCI HAR Dataset"
WISDM_FILE = ROOT / "data/raw/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
OUTPUT_DIR = ROOT / "data/processed"

CLASSES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
SAMPLE_RATE = 20
WINDOW_SAMPLES = 50  # 2.5 seconds at 20 Hz.
STEP_SAMPLES = 25    # WISDM windows overlap by 50%.
GRAVITY = 9.80665    # Convert UCI acceleration from g to m/s^2.


def load_uci(directory=UCI_DIR):
    """Use each released UCI window independently; do not join recordings."""
    windows, labels, subjects = [], [], []

    for partition in ("train", "test"):
        folder = Path(directory) / partition
        activity = np.loadtxt(folder / f"y_{partition}.txt", dtype=int, ndmin=1)
        people = np.loadtxt(folder / f"subject_{partition}.txt", dtype=int, ndmin=1)
        axes = []
        for axis in "xyz":
            path = folder / "Inertial Signals" / f"total_acc_{axis}_{partition}.txt"
            axes.append(np.loadtxt(path, ndmin=2))

        signals = np.stack(axes, axis=2)
        if signals.shape != (len(activity), 128, 3) or len(people) != len(activity):
            raise ValueError("UCI signal, label, and subject rows do not match.")
        if not np.isin(activity, np.arange(1, 7)).all():
            raise ValueError("UCI contains an unknown activity label.")

        # Keep the five shared activities. Label 6 is laying.
        keep = activity <= 5
        signals = signals[keep, :125] * GRAVITY  # First 2.5 seconds at 50 Hz.

        # The polyphase filter prevents aliasing when reducing 50 Hz to 20 Hz.
        signals = resample_poly(signals, up=2, down=5, axis=1, padtype="line")
        windows.append(signals)
        labels.append(activity[keep] - 1)
        subjects.append(people[keep])
        if partition == "test":
            test_subjects = np.unique(people)

    return (
        np.concatenate(windows).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(subjects),
        test_subjects,
    )


def read_wisdm_recordings(path):
    """Yield (timestamp/x/y/z readings, subject, activity) for each valid run."""
    records = []
    previous = None
    valid_activities = {activity.title() for activity in CLASSES} | {"Jogging"}

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            # Some lines contain several records, separated by semicolons.
            for record in line.split(";"):
                if not record.strip():
                    continue
                fields = record.strip().split(",")
                if len(fields) == 7 and not fields[-1].strip():
                    fields.pop()  # Repair one extra trailing comma.

                try:
                    if len(fields) != 6:
                        raise ValueError("Expected six fields")
                    subject = int(fields[0])
                    activity = fields[1].strip()
                    timestamp = int(fields[2])
                    acceleration = [float(value) for value in fields[3:]]
                    if (
                        subject < 1
                        or timestamp <= 0
                        or activity not in valid_activities
                        or not np.isfinite(acceleration).all()
                    ):
                        raise ValueError("Invalid reading")
                except ValueError:
                    # Do not join data across a missing or invalid reading.
                    if records:
                        yield np.asarray(records), previous[0], previous[1]
                    records = []
                    previous = None
                    continue

                if previous is not None:
                    old_subject, old_activity, old_time, old_acceleration = previous
                    same_activity = subject == old_subject and activity == old_activity
                    elapsed = (timestamp - old_time) / 1e9
                    is_duplicate = (
                        same_activity and elapsed == 0
                        and acceleration == old_acceleration
                    )
                    if is_duplicate:
                        continue  # Exact duplicate; keep the recording intact.
                    if not same_activity or not 0 < elapsed <= 0.5:
                        # Start again after a subject/activity change, clock
                        # reset, conflicting timestamp, or gap over 0.5 seconds.
                        yield np.asarray(records), old_subject, old_activity
                        records = []

                records.append([timestamp, *acceleration])
                previous = (subject, activity, timestamp, acceleration)

    if records:
        yield np.asarray(records), previous[0], previous[1]


def load_wisdm(path=WISDM_FILE):
    """Interpolate continuous WISDM runs to 20 Hz and make 2.5-second windows."""
    windows, labels, subjects = [], [], []

    for records, subject, activity in read_wisdm_recordings(path):
        if activity == "Jogging":
            continue
        times = (records[:, 0] - records[0, 0]) / 1e9
        sample_count = int(np.floor(times[-1] * SAMPLE_RATE + 1e-7)) + 1
        grid = np.arange(sample_count) / SAMPLE_RATE
        axes = [np.interp(grid, times, records[:, axis]) for axis in (1, 2, 3)]
        signal = np.column_stack(axes)

        # Use elapsed time, not raw row counts, because timestamps are irregular.
        # Each window stays within one subject, activity, and recording run.
        for start in range(0, len(signal) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            windows.append(signal[start:start + WINDOW_SAMPLES])
            labels.append(CLASSES.index(activity.lower()))
            subjects.append(subject)

    if not windows:
        raise ValueError("WISDM has no complete windows after cleaning.")
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
    )


def make_splits(subjects, test_subjects=None):
    """Keep each person's windows together to prevent train/test leakage."""
    rng = np.random.default_rng(42)
    people = np.unique(subjects)
    if test_subjects is None:
        people = rng.permutation(people)
        count = max(1, round(0.2 * len(people)))
        test_subjects = people[:count]
        validation_subjects = people[count:2 * count]
    else:
        remaining = np.setdiff1d(people, test_subjects)
        count = max(1, round(0.2 * len(remaining)))
        validation_subjects = rng.permutation(remaining)[:count]

    split = np.full(len(subjects), "train", dtype="U5")
    split[np.isin(subjects, validation_subjects)] = "val"
    split[np.isin(subjects, test_subjects)] = "test"
    return split


def save_dataset(name, windows, labels, subjects, test_subjects=None):
    # Save one file per dataset. Metadata stays separate from model features.
    features, names = extract_features(windows, SAMPLE_RATE)
    split = make_splits(subjects, test_subjects)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / f"{name}.npz",
        windows=windows,
        features=features,
        labels=labels,
        subjects=subjects,
        split=split,
        feature_names=np.asarray(names),
    )
    print(f"{name}: {len(labels):,} examples, {len(names)} features")
    for part in ("train", "val", "test"):
        print(f"  {part}: {(split == part).sum():,} examples")


if __name__ == "__main__":
    if not UCI_DIR.exists() or not WISDM_FILE.exists():
        raise SystemExit("Raw data are missing. See the download steps in README.md.")

    windows, labels, subjects, test_subjects = load_uci()
    save_dataset("uci_har", windows, labels, subjects, test_subjects)

    windows, labels, subjects = load_wisdm()
    save_dataset("wisdm", windows, labels, subjects)
