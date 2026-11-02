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
SAMPLE_RATE = 20  # Both datasets use this sampling rate after preprocessing.
GRAVITY = 9.80665  # Convert UCI acceleration from g to m/s².


def find_file(directory, filename):
    """Find a release file without depending on the archive's folder layout."""
    matches = list(Path(directory).rglob(filename))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {filename} under {directory}; "
            "run download_data.py first."
        )

    return matches[0]


def uci_segments(directory, audit):
    """Reconstruct continuous recordings from UCI's overlapping windows."""
    audit["original_windows"] = 0
    audit["verified_overlaps"] = 0
    audit["same_label_discontinuities"] = 0

    for partition in ("train", "test"):
        subject_file = find_file(directory, f"subject_{partition}.txt")
        subjects = np.loadtxt(subject_file, dtype=int)
        labels = np.loadtxt(subject_file.parent / f"y_{partition}.txt", dtype=int)

        # Stack the three axes into (original windows, 128 samples, 3 axes).
        axis_signals = []
        for axis in "xyz":
            signal_file = (
                subject_file.parent
                / "Inertial Signals"
                / f"total_acc_{axis}_{partition}.txt"
            )
            axis_signals.append(np.loadtxt(signal_file))

        signals = np.stack(axis_signals, axis=2)

        if signals.shape != (len(labels), 128, 3) or len(subjects) != len(labels):
            raise ValueError("Unexpected UCI HAR signal or label shape.")

        if not np.isfinite(signals).all() or not np.isin(labels, np.arange(1, 7)).all():
            raise ValueError("Invalid UCI HAR values or labels.")

        audit["original_windows"] += len(labels)
        segment_start = 0

        # The extra iteration at the end saves the final recording segment.
        for index in range(1, len(signals) + 1):
            same_subject_and_activity = (
                index < len(signals)
                and subjects[index] == subjects[index - 1]
                and labels[index] == labels[index - 1]
            )

            # Consecutive 128-sample windows share 64 samples. All shared
            # x/y/z readings must match before we treat them as one recording.
            overlap_matches = same_subject_and_activity and np.array_equal(
                signals[index - 1, 64:],
                signals[index, :64],
            )

            if overlap_matches:
                audit["verified_overlaps"] += 1
                continue

            if same_subject_and_activity:
                audit["same_label_discontinuities"] += 1

            label = int(labels[segment_start]) - 1  # UCI labels start at 1.

            if label < len(CLASSES):
                # Keep the first window in full, then append only the new
                # 64 samples from each later window to avoid duplicating data.
                first_window = signals[segment_start]
                new_samples = signals[segment_start + 1:index, 64:].reshape(-1, 3)
                signal = np.concatenate((first_window, new_samples))

                yield signal * GRAVITY, {
                    "subject": int(subjects[segment_start]),
                    "label": label,
                    "original_partition": partition,
                    "source_start_record": segment_start,
                }
            else:
                # Laying is not part of the shared five-activity task.
                excluded_windows = audit.get("excluded_laying_windows", 0)
                audit["excluded_laying_windows"] = (
                    excluded_windows + index - segment_start
                )

            segment_start = index


def wisdm_segments(directory, audit, max_gap_seconds=0.5):
    """Read WISDM records in file order and separate continuous recordings."""
    path = find_file(directory, "WISDM_ar_v1.1_raw.txt")
    counts = Counter()
    activity_counts = Counter()
    sampling_intervals = []

    records = []
    previous_record = None
    start_record = 0
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

                # Some complete records have an extra comma after the z value.
                if len(fields) == 7 and not fields[-1].strip():
                    fields.pop()
                    counts["repaired_trailing_comma_records"] += 1

                try:
                    if len(fields) != 6:
                        raise ValueError("Expected six fields")

                    subject = int(fields[0])
                    activity = fields[1].strip()
                    timestamp = int(fields[2])
                    acceleration = [float(value) for value in fields[3:]]

                    if (
                        subject < 1
                        or activity not in {*WISDM_LABELS, "Jogging"}
                        or not np.isfinite(acceleration).all()
                    ):
                        raise ValueError("Invalid sensor values")

                    if timestamp <= 0:
                        counts["nonpositive_timestamp_records"] += 1
                        raise ValueError("Invalid timestamp")

                except ValueError:
                    # An invalid reading ends the current segment. Do not
                    # interpolate across a record whose time or values are bad.
                    counts["invalid_records"] += 1
                    if records:
                        previous_subject, previous_activity, _ = previous_record
                        yield (
                            np.asarray(records),
                            previous_subject,
                            previous_activity,
                            start_record,
                        )

                    records = []
                    previous_record = None
                    continue

                activity_counts[activity] += 1
                starts_new_segment = previous_record is None

                if previous_record is not None:
                    previous_subject, previous_activity, previous_timestamp = (
                        previous_record
                    )
                    elapsed_seconds = (timestamp - previous_timestamp) / 1e9

                    if subject != previous_subject or activity != previous_activity:
                        counts["subject_or_activity_changes"] += 1
                        starts_new_segment = True

                    elif elapsed_seconds == 0 and records[-1][1:] == acceleration:
                        # Skip exact copies without breaking a valid recording.
                        counts["removed_exact_duplicate_records"] += 1
                        continue

                    elif elapsed_seconds <= 0:
                        # A backwards clock or conflicting duplicate timestamp
                        # cannot be interpolated as part of the current segment.
                        counts["timestamp_resets_or_conflicting_duplicates"] += 1
                        starts_new_segment = True

                    elif elapsed_seconds > max_gap_seconds:
                        counts["large_time_gaps"] += 1
                        starts_new_segment = True

                    else:
                        sampling_intervals.append(elapsed_seconds)

                if starts_new_segment:
                    if records:
                        yield (
                            np.asarray(records),
                            previous_subject,
                            previous_activity,
                            start_record,
                        )

                    records = []
                    start_record = record_number

                records.append([timestamp, *acceleration])
                previous_record = (subject, activity, timestamp)

    # There may be one last segment left when the file ends.
    if records:
        previous_subject, previous_activity, _ = previous_record
        yield np.asarray(records), previous_subject, previous_activity, start_record

    audit.update(dict(counts))
    audit["valid_records_by_activity"] = dict(activity_counts)

    if sampling_intervals:
        interval_percentiles = np.percentile(sampling_intervals, [5, 50, 95])
        audit["timestamp_interval_seconds_p5_p50_p95"] = interval_percentiles.tolist()


def uniform_wisdm_segments(directory, audit, max_gap_seconds):
    """Interpolate each WISDM recording onto an evenly spaced 20 Hz grid."""
    segments = wisdm_segments(directory, audit, max_gap_seconds)

    for records, subject, activity, start_record in segments:
        if activity == "Jogging":
            continue

        # Subtract before converting nanoseconds to seconds, preserving precision.
        timestamps = (records[:, 0] - records[0, 0]) / 1e9

        # End the grid at or before the last real reading; do not extrapolate.
        # The small tolerance avoids losing a sample to floating-point rounding.
        sample_count = int(np.floor(timestamps[-1] * SAMPLE_RATE + 1e-7)) + 1
        time_grid = np.arange(sample_count) / SAMPLE_RATE

        resampled_axes = []
        for axis in (1, 2, 3):
            resampled_axis = np.interp(time_grid, timestamps, records[:, axis])
            resampled_axes.append(resampled_axis)

        signal = np.column_stack(resampled_axes)

        yield signal, {
            "subject": subject,
            "label": WISDM_LABELS[activity],
            "original_partition": "raw",
            "source_start_record": start_record,
            "source_start_timestamp_ns": int(records[0, 0]),
        }


def make_windows(segments, window_samples, step_samples):
    """Make complete windows without crossing recording boundaries."""
    windows = []
    metadata = []
    segment_metadata = []

    for segment_id, (signal, info) in enumerate(segments):
        starts = range(0, len(signal) - window_samples + 1, step_samples)

        # Keep a record of short segments too, even if they produce no windows.
        segment_metadata.append({
            **info,
            "segment_id": segment_id,
            "resampled_samples": len(signal),
            "windows": len(starts),
        })

        for start in starts:
            windows.append(signal[start:start + window_samples])
            metadata.append({
                "subject": info["subject"],
                "label": info["label"],
                "activity": CLASSES[info["label"]],
                "segment_id": segment_id,
                "start_seconds": start / SAMPLE_RATE,
                "original_partition": info["original_partition"],
            })

    if not windows:
        raise ValueError(
            "No complete windows remain. "
            "Check signal continuity and window length."
        )

    return (
        np.asarray(windows, dtype=np.float32),
        pd.DataFrame(metadata),
        pd.DataFrame(segment_metadata),
    )


def assign_splits(metadata, name, seed):
    """Assign whole subjects to train, validation, or test using a fixed seed."""
    rng = np.random.default_rng(seed)
    subjects = np.sort(metadata.subject.unique())

    if name == "uci_har":
        # Keep the official test subjects; reserve part of train for validation.
        test_rows = metadata.original_partition == "test"
        test_subjects = metadata.loc[test_rows, "subject"].unique()
        remaining_subjects = np.setdiff1d(subjects, test_subjects)

        validation_count = max(1, round(0.2 * len(remaining_subjects)))
        validation_subjects = rng.permutation(remaining_subjects)[:validation_count]

    else:
        # WISDM has no official split. Reserve about 20% of subjects for each
        # of validation and test, leaving the rest for training.
        shuffled_subjects = rng.permutation(subjects)
        held_out_count = max(1, round(0.2 * len(subjects)))

        test_subjects = shuffled_subjects[:held_out_count]
        validation_subjects = shuffled_subjects[held_out_count:2 * held_out_count]

    is_test = metadata.subject.isin(test_subjects)
    is_validation = metadata.subject.isin(validation_subjects)
    training_or_validation = np.where(is_validation, "val", "train")
    metadata["split"] = np.where(is_test, "test", training_or_validation)

    return metadata


def validate_dataset(windows, features, metadata, window_samples):
    """Check row alignment, finite inputs, subject isolation, and class coverage."""
    expected_shape = (len(metadata), window_samples, 3)

    if windows.shape != expected_shape or len(features) != len(metadata):
        raise ValueError("Window, feature, and metadata rows do not align.")

    if not np.isfinite(windows).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite model inputs remain.")

    # Keeping each subject in one split also keeps their overlapping windows
    # together, preventing shared samples from leaking between splits.
    if metadata.groupby("subject")["split"].nunique().max() != 1:
        raise ValueError("A subject appears in multiple splits.")

    if metadata.groupby("segment_id")["split"].nunique().max() != 1:
        raise ValueError("Overlapping windows appear in multiple splits.")

    expected_labels = set(range(len(CLASSES)))
    for split in ("train", "val", "test"):
        actual_labels = set(metadata.loc[metadata.split == split, "label"])
        if actual_labels != expected_labels:
            raise ValueError(
                f"{split} is missing an activity. Review the subject split."
            )


def prepare_dataset(
    name,
    data_dir,
    report_dir,
    window_seconds,
    overlap,
    seed,
    max_gap_seconds,
):
    """Prepare and save one dataset, then report the cleaning and split counts."""
    print(f"Preparing {name}", flush=True)
    directory = data_dir / "raw" / name
    audit = {}
    window_samples = round(window_seconds * SAMPLE_RATE)
    step_samples = round(window_samples * (1 - overlap))

    # Read each continuous recording and bring it to the shared sampling rate.
    if name == "uci_har":
        def resampled_segments():
            for signal, info in uci_segments(directory, audit):
                # Downsample 50 Hz to 20 Hz (2/5). The FIR filter removes
                # frequencies that the lower sampling rate cannot represent.
                resampled = resample_poly(
                    signal,
                    up=2,
                    down=5,
                    axis=0,
                    padtype="line",
                )

                # Discard any final sample beyond the original recording time.
                valid_length = int(np.floor((len(signal) - 1) * SAMPLE_RATE / 50)) + 1
                yield resampled[:valid_length], info

        segments = resampled_segments()
    else:
        segments = uniform_wisdm_segments(directory, audit, max_gap_seconds)

    # Both datasets now use the same window shape, labels, and feature function.
    windows, metadata, segment_metadata = make_windows(
        segments,
        window_samples,
        step_samples,
    )
    metadata = assign_splits(metadata, name, seed)
    features, feature_names = extract_features(windows, SAMPLE_RATE)
    validate_dataset(windows, features, metadata, window_samples)

    # Save unscaled model inputs. Scaling is fitted later on source train only.
    output = data_dir / "processed" / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "dataset.npz",
        X=windows,
        features=features,
        y=metadata.label.to_numpy(dtype=np.int64),
        subjects=metadata.subject.to_numpy(dtype=np.int64),
        feature_names=np.asarray(feature_names),
        split=metadata.split.to_numpy(dtype="U5"),
        segment_ids=metadata.segment_id.to_numpy(dtype=np.int64),
        start_seconds=metadata.start_seconds.to_numpy(dtype=float),
    )

    # CSV rows stay in the same order as the arrays in dataset.npz.
    metadata.to_csv(output / "metadata.csv", index=False)
    feature_table = pd.DataFrame(features, columns=feature_names)
    feature_table.to_csv(output / "features.csv", index=False)
    segment_metadata.to_csv(output / "segments.csv", index=False)

    # Record the actual configuration and counts for the shared experiment log.
    summary = {
        "dataset": name,
        "classes": dict(enumerate(CLASSES)),
        "sample_rate_hz": SAMPLE_RATE,
        "acceleration_units": "m/s^2 (gravity retained)",
        "window_seconds": window_samples / SAMPLE_RATE,
        "step_seconds": step_samples / SAMPLE_RATE,
        "requested_overlap": overlap,
        "seed": seed,
        "max_wisdm_gap_seconds": max_gap_seconds,
        "windows": len(windows),
        "subjects": int(metadata.subject.nunique()),
        "features": len(feature_names),
        "segments": len(segment_metadata),
        "segments_too_short": int((segment_metadata.windows == 0).sum()),
        "class_windows": {
            activity: int((metadata.activity == activity).sum())
            for activity in CLASSES
        },
        "splits": {},
        "cleaning": audit,
    }

    for split in ("train", "val", "test"):
        rows = metadata[metadata.split == split]
        summary["splits"][split] = {
            "windows": len(rows),
            "subjects": sorted(int(subject) for subject in rows.subject.unique()),
            "class_windows": {
                activity: int((rows.activity == activity).sum())
                for activity in CLASSES
            },
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_file = report_dir / f"{name}_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{name}: {len(windows):,} windows, "
        f"{metadata.subject.nunique()} subjects, {len(feature_names)} features",
        flush=True,
    )
    return summary


def main():
    """Read command-line settings and prepare both datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    args = parser.parse_args()

    # Reject settings that cannot produce a valid window or sampling interval.
    if not np.isfinite(args.window_seconds) or args.window_seconds * SAMPLE_RATE < 2:
        parser.error("--window-seconds must give at least two samples at 20 Hz")

    if not np.isfinite(args.overlap) or not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1)")

    window_samples = round(args.window_seconds * SAMPLE_RATE)
    step_samples = round(window_samples * (1 - args.overlap))
    if step_samples < 1:
        parser.error("--overlap gives a step smaller than one sample")

    if not np.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
        parser.error("--max-gap-seconds must be positive")

    for name in ("uci_har", "wisdm"):
        prepare_dataset(
            name,
            args.data_dir,
            ROOT / "reports",
            args.window_seconds,
            args.overlap,
            args.seed,
            args.max_gap_seconds,
        )


if __name__ == "__main__":
    main()
