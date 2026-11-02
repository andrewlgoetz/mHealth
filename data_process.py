"""Prepare comparable UCI HAR and WISDM windows, features, and subject splits."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from features import extract_features

ROOT = Path(__file__).resolve().parent
CLASSES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
WISDM_LABELS = {name.title(): index for index, name in enumerate(CLASSES)}
SAMPLE_RATE = 20
GRAVITY = 9.80665


def find_file(directory, filename):
    matches = list(Path(directory).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} under {directory}; run download_data.py first.")
    return matches[0]


def uci_segments(directory, audit):
    """Join only windows whose 64 shared samples agree exactly on all axes."""
    audit["original_windows"] = 0
    audit["verified_overlaps"] = 0
    audit["same_label_discontinuities"] = 0
    for partition in ("train", "test"):
        subject_file = find_file(directory, f"subject_{partition}.txt")
        subjects = np.loadtxt(subject_file, dtype=int)
        labels = np.loadtxt(subject_file.parent / f"y_{partition}.txt", dtype=int)
        signals = np.stack([
            np.loadtxt(subject_file.parent / "Inertial Signals" / f"total_acc_{axis}_{partition}.txt")
            for axis in "xyz"
        ], axis=2)
        if signals.shape != (len(labels), 128, 3) or len(subjects) != len(labels):
            raise ValueError("Unexpected UCI HAR signal or label shape.")
        if not np.isfinite(signals).all() or not np.isin(labels, np.arange(1, 7)).all():
            raise ValueError("Invalid UCI HAR values or labels.")
        audit["original_windows"] += len(labels)
        start = 0
        for index in range(1, len(signals) + 1):
            same_label = index < len(signals) and subjects[index] == subjects[index - 1] and labels[index] == labels[index - 1]
            continuous = same_label and np.array_equal(signals[index - 1, 64:], signals[index, :64])
            if continuous:
                audit["verified_overlaps"] += 1
                continue
            if same_label:
                audit["same_label_discontinuities"] += 1
            label = int(labels[start]) - 1
            if label < len(CLASSES):
                # The first half of later rows is already present in the run.
                signal = np.concatenate((signals[start], signals[start + 1:index, 64:].reshape(-1, 3)))
                yield signal * GRAVITY, {
                    "subject": int(subjects[start]), "label": label,
                    "original_partition": partition, "source_start_record": start,
                }
            else:
                audit["excluded_laying_windows"] = audit.get("excluded_laying_windows", 0) + index - start
            start = index


def wisdm_segments(directory, audit, max_gap_seconds=0.5):
    """Remove exact repeats; bad rows, resets, gaps, and label changes end a run."""
    path = find_file(directory, "WISDM_ar_v1.1_raw.txt")
    counts = Counter()
    activity_counts = Counter()
    intervals = []
    records, previous, start_record = [], None, 0
    record_number = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            # A few lines contain more than one semicolon-delimited record.
            for record in line.split(";"):
                if not record.strip():
                    continue
                record_number += 1
                counts["records_seen"] += 1
                fields = record.strip().split(",")
                if len(fields) == 7 and not fields[-1].strip():
                    fields.pop()
                    counts["repaired_trailing_comma_records"] += 1
                try:
                    if len(fields) != 6:
                        raise ValueError("Expected six fields")
                    subject, activity, timestamp = int(fields[0]), fields[1].strip(), int(fields[2])
                    acceleration = [float(value) for value in fields[3:]]
                    if subject < 1 or activity not in {*WISDM_LABELS, "Jogging"} or not np.isfinite(acceleration).all():
                        raise ValueError("Invalid sensor values")
                    if timestamp <= 0:
                        counts["nonpositive_timestamp_records"] += 1
                        raise ValueError("Invalid timestamp")
                except ValueError:
                    counts["invalid_records"] += 1
                    if records:
                        yield np.asarray(records), previous[0], previous[1], start_record
                    records, previous = [], None
                    continue
                activity_counts[activity] += 1
                new_run = previous is None
                if previous is not None:
                    delta = (timestamp - previous[2]) / 1e9
                    if subject != previous[0] or activity != previous[1]:
                        counts["subject_or_activity_changes"] += 1
                        new_run = True
                    elif delta == 0 and records[-1][1:] == acceleration:
                        counts["removed_exact_duplicate_records"] += 1
                        continue
                    elif delta <= 0:
                        counts["timestamp_resets_or_conflicting_duplicates"] += 1
                        new_run = True
                    elif delta > max_gap_seconds:
                        counts["large_time_gaps"] += 1
                        new_run = True
                    else:
                        intervals.append(delta)
                if new_run:
                    if records:
                        yield np.asarray(records), previous[0], previous[1], start_record
                    records, start_record = [], record_number
                records.append([timestamp, *acceleration])
                previous = (subject, activity, timestamp)
    if records:
        yield np.asarray(records), previous[0], previous[1], start_record
    audit.update(dict(counts))
    audit["valid_records_by_activity"] = dict(activity_counts)
    if intervals:
        audit["timestamp_interval_seconds_p5_p50_p95"] = np.percentile(intervals, [5, 50, 95]).tolist()


def uniform_wisdm_segments(directory, audit, max_gap_seconds):
    for records, subject, activity, start_record in wisdm_segments(directory, audit, max_gap_seconds):
        if activity == "Jogging":
            continue
        # Subtract before converting nanoseconds to seconds, preserving precision.
        timestamps = (records[:, 0] - records[0, 0]) / 1e9
        grid = np.arange(int(np.floor(timestamps[-1] * SAMPLE_RATE + 1e-7)) + 1) / SAMPLE_RATE
        signal = np.column_stack([np.interp(grid, timestamps, records[:, axis]) for axis in (1, 2, 3)])
        yield signal, {
            "subject": subject, "label": WISDM_LABELS[activity],
            "original_partition": "raw", "source_start_record": start_record,
            "source_start_timestamp_ns": int(records[0, 0]),
        }


def make_windows(segments, window_samples, step_samples):
    windows, metadata, segment_metadata = [], [], []
    for segment_id, (signal, info) in enumerate(segments):
        starts = range(0, len(signal) - window_samples + 1, step_samples)
        segment_metadata.append({**info, "segment_id": segment_id, "resampled_samples": len(signal), "windows": len(starts)})
        for start in starts:
            windows.append(signal[start:start + window_samples])
            metadata.append({
                "subject": info["subject"], "label": info["label"],
                "activity": CLASSES[info["label"]], "segment_id": segment_id,
                "start_seconds": start / SAMPLE_RATE,
                "original_partition": info["original_partition"],
            })
    if not windows:
        raise ValueError("No complete windows remain. Check signal continuity and window length.")
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(metadata), pd.DataFrame(segment_metadata)


def assign_splits(metadata, name, seed):
    rng = np.random.default_rng(seed)
    subjects = np.sort(metadata.subject.unique())
    if name == "uci_har":
        # Preserve the official nine test subjects, choose validation from train.
        test = metadata.loc[metadata.original_partition == "test", "subject"].unique()
        remaining = np.setdiff1d(subjects, test)
        val = rng.permutation(remaining)[:max(1, round(0.2 * len(remaining)))]
    else:
        shuffled = rng.permutation(subjects)
        count = max(1, round(0.2 * len(subjects)))
        test, val = shuffled[:count], shuffled[count:2 * count]
    metadata["split"] = np.where(metadata.subject.isin(test), "test", np.where(metadata.subject.isin(val), "val", "train"))
    return metadata


def validate_dataset(windows, features, metadata, window_samples):
    if windows.shape != (len(metadata), window_samples, 3) or len(features) != len(metadata):
        raise ValueError("Window, feature, and metadata rows do not align.")
    if not np.isfinite(windows).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite model inputs remain.")
    if metadata.groupby("subject")["split"].nunique().max() != 1:
        raise ValueError("A subject appears in multiple splits.")
    if metadata.groupby("segment_id")["split"].nunique().max() != 1:
        raise ValueError("Overlapping windows appear in multiple splits.")
    for split in ("train", "val", "test"):
        if set(metadata.loc[metadata.split == split, "label"]) != set(range(len(CLASSES))):
            raise ValueError(f"{split} is missing an activity. Review the subject split.")


def prepare_dataset(name, data_dir, report_dir, window_seconds, overlap, seed, max_gap_seconds):
    print(f"Preparing {name}", flush=True)
    directory = data_dir / "raw" / name
    audit = {}
    window_samples = round(window_seconds * SAMPLE_RATE)
    step_samples = round(window_samples * (1 - overlap))
    if name == "uci_har":
        def resampled_segments():
            for signal, info in uci_segments(directory, audit):
                # Polyphase FIR removes frequencies above the new Nyquist limit.
                resampled = resample_poly(signal, up=2, down=5, axis=0, padtype="line")
                valid_length = int(np.floor((len(signal) - 1) * SAMPLE_RATE / 50)) + 1
                yield resampled[:valid_length], info
        segments = resampled_segments()
    else:
        segments = uniform_wisdm_segments(directory, audit, max_gap_seconds)
    windows, metadata, segment_metadata = make_windows(segments, window_samples, step_samples)
    metadata = assign_splits(metadata, name, seed)
    features, feature_names = extract_features(windows, SAMPLE_RATE)
    validate_dataset(windows, features, metadata, window_samples)
    output = data_dir / "processed" / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "dataset.npz", X=windows, features=features,
        y=metadata.label.to_numpy(dtype=np.int64), subjects=metadata.subject.to_numpy(dtype=np.int64),
        feature_names=np.asarray(feature_names), split=metadata.split.to_numpy(dtype="U5"),
        segment_ids=metadata.segment_id.to_numpy(dtype=np.int64),
        start_seconds=metadata.start_seconds.to_numpy(dtype=float),
    )
    metadata.to_csv(output / "metadata.csv", index=False)
    pd.DataFrame(features, columns=feature_names).to_csv(output / "features.csv", index=False)
    segment_metadata.to_csv(output / "segments.csv", index=False)
    summary = {
        "dataset": name, "classes": dict(enumerate(CLASSES)),
        "sample_rate_hz": SAMPLE_RATE, "acceleration_units": "m/s^2 (gravity retained)",
        "window_seconds": window_samples / SAMPLE_RATE, "step_seconds": step_samples / SAMPLE_RATE,
        "requested_overlap": overlap, "seed": seed, "max_wisdm_gap_seconds": max_gap_seconds,
        "windows": len(windows), "subjects": int(metadata.subject.nunique()),
        "features": len(feature_names), "segments": len(segment_metadata),
        "segments_too_short": int((segment_metadata.windows == 0).sum()),
        "class_windows": {activity: int((metadata.activity == activity).sum()) for activity in CLASSES},
        "splits": {}, "cleaning": audit,
    }
    for split in ("train", "val", "test"):
        rows = metadata[metadata.split == split]
        summary["splits"][split] = {
            "windows": len(rows), "subjects": sorted(int(subject) for subject in rows.subject.unique()),
            "class_windows": {activity: int((rows.activity == activity).sum()) for activity in CLASSES},
        }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {len(windows):,} windows, {metadata.subject.nunique()} subjects, {len(feature_names)} features", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if not np.isfinite(args.window_seconds) or args.window_seconds * SAMPLE_RATE < 2:
        parser.error("--window-seconds must give at least two samples at 20 Hz")
    if not np.isfinite(args.overlap) or not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1)")
    if round(round(args.window_seconds * SAMPLE_RATE) * (1 - args.overlap)) < 1:
        parser.error("--overlap gives a step smaller than one sample")
    if not np.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
        parser.error("--max-gap-seconds must be positive")
    for name in ("uci_har", "wisdm"):
        prepare_dataset(name, args.data_dir, ROOT / "reports", args.window_seconds, args.overlap, args.seed, args.max_gap_seconds)


if __name__ == "__main__":
    main()
