"""Small synthetic checks for recording boundaries and subject isolation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_process import (
    GRAVITY,
    assign_splits,
    uci_segments,
    uniform_wisdm_segments,
    wisdm_segments,
)


def write_uci_partition(root, partition, signals, subjects, labels):
    # Build the same small folder layout that the UCI reader expects.
    folder = root / partition
    signal_folder = folder / "Inertial Signals"
    signal_folder.mkdir(parents=True)

    np.savetxt(folder / f"subject_{partition}.txt", subjects, fmt="%d")
    np.savetxt(folder / f"y_{partition}.txt", labels, fmt="%d")

    for axis, values in zip("xyz", np.moveaxis(signals, 2, 0)):
        signal_file = signal_folder / f"total_acc_{axis}_{partition}.txt"
        np.savetxt(signal_file, values)


def wisdm_record(subject, activity, timestamp, value, trailing_comma=False):
    suffix = "," if trailing_comma else ""
    return f"{subject},{activity},{timestamp},{value},2,3{suffix};"


class RecordingTests(unittest.TestCase):
    def test_uci_stitches_verified_overlap_and_stops_at_discontinuity(self):
        # The first two windows share 64 samples; the third is unrelated even
        # though its subject and activity labels match.
        time = np.arange(192, dtype=float)
        continuous = np.column_stack((time, 2 * time, -time))
        separate = continuous[:128] + 1000
        train = np.stack((continuous[:128], continuous[64:], separate))
        test = np.stack((continuous[:128] + 2000, continuous[:128] + 3000))

        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_uci_partition(root, "train", train, [1, 1, 1], [1, 1, 1])
            write_uci_partition(root, "test", test, [9, 9], [3, 6])

            audit = {}
            segments = list(uci_segments(root, audit))

        # Joining overlapping windows should produce 192 samples, not 256.
        self.assertEqual([len(signal) for signal, _ in segments], [192, 128, 128])
        np.testing.assert_array_equal(segments[0][0], continuous * GRAVITY)
        np.testing.assert_array_equal(segments[1][0], separate * GRAVITY)
        self.assertEqual(segments[1][1]["source_start_record"], 2)
        self.assertEqual(segments[2][1]["original_partition"], "test")
        self.assertEqual(segments[2][1]["label"], 2)
        self.assertEqual(audit["original_windows"], 5)
        self.assertEqual(audit["verified_overlaps"], 1)
        self.assertEqual(audit["same_label_discontinuities"], 1)
        self.assertEqual(audit["excluded_laying_windows"], 1)

    def test_wisdm_parsing_preserves_order_and_recording_boundaries(self):
        # The initial line holds three records, including an exact duplicate.
        lines = [
            wisdm_record(1, "Walking", 1_000_000_000, 10)
            + wisdm_record(1, "Walking", 1_050_000_000, 11)
            + wisdm_record(1, "Walking", 1_050_000_000, 11),
            wisdm_record(1, "Walking", 1_050_000_000, 12),  # Conflicting duplicate.
            "broken record;",
            wisdm_record(1, "Walking", 1_100_000_000, 13, trailing_comma=True),
            wisdm_record(1, "Walking", 0, 99),
            wisdm_record(1, "Walking", 1_150_000_000, 14),
            wisdm_record(1, "Walking", 1_100_000_000, 15),  # Clock reset.
            wisdm_record(1, "Walking", 1_150_000_000, 16),
            wisdm_record(1, "Walking", 2_000_000_000, 17),  # Long gap.
            wisdm_record(2, "Walking", 2_050_000_000, 18),
            wisdm_record(2, "Jogging", 2_100_000_000, 19),
            wisdm_record(2, "Standing", 2_150_000_000, 20),
            wisdm_record(2, "Standing", 2_200_000_000, "nan"),
            wisdm_record(2, "Standing", 2_250_000_000, 21),
        ]

        with TemporaryDirectory() as folder:
            raw_file = Path(folder) / "WISDM_ar_v1.1_raw.txt"
            raw_file.write_text("\n".join(lines), encoding="utf-8")

            audit = {}
            segments = list(wisdm_segments(folder, audit, max_gap_seconds=0.5))

        values = [records[:, 1].tolist() for records, *_ in segments]
        self.assertEqual(
            values,
            [[10, 11], [12], [13], [14], [15, 16], [17], [18], [19], [20], [21]],
        )

        # The post-reset run remains after its predecessor; nothing is sorted.
        self.assertLess(segments[4][0][0, 0], segments[3][0][0, 0])
        self.assertEqual(audit["records_seen"], 16)
        self.assertEqual(audit["invalid_records"], 3)
        self.assertEqual(audit["nonpositive_timestamp_records"], 1)
        self.assertEqual(audit["removed_exact_duplicate_records"], 1)
        self.assertEqual(audit["repaired_trailing_comma_records"], 1)
        self.assertEqual(audit["timestamp_resets_or_conflicting_duplicates"], 2)
        self.assertEqual(audit["large_time_gaps"], 1)
        self.assertEqual(audit["subject_or_activity_changes"], 3)

    def test_wisdm_resamples_elapsed_time_and_excludes_jogging(self):
        # A linear signal has a known value at every interpolated timestamp,
        # even when the original readings are not evenly spaced.
        seconds = np.array([0.0, 0.07, 0.15, 0.25])
        lines = []

        activities = (("Walking", 1_000_000_000), ("Jogging", 2_000_000_000))
        for activity, base in activities:
            for elapsed in seconds:
                timestamp = base + round(elapsed * 1e9)
                lines.append(
                    f"1,{activity},{timestamp},"
                    f"{elapsed},{2 * elapsed + 3},{-elapsed};"
                )

        with TemporaryDirectory() as folder:
            raw_file = Path(folder) / "WISDM_ar_v1.1_raw.txt"
            raw_file.write_text("\n".join(lines), encoding="utf-8")

            audit = {}
            segments = list(uniform_wisdm_segments(folder, audit, max_gap_seconds=0.5))

        # Only walking remains, sampled at 0, 0.05, ..., 0.25 seconds.
        self.assertEqual(len(segments), 1)
        signal, info = segments[0]
        grid = np.arange(6) / 20
        expected = np.column_stack((grid, 2 * grid + 3, -grid))
        np.testing.assert_allclose(signal, expected, atol=1e-12)
        self.assertEqual(info["label"], 0)
        self.assertEqual(info["source_start_timestamp_ns"], 1_000_000_000)
        self.assertEqual(audit["valid_records_by_activity"]["Jogging"], 4)


class SubjectSplitTests(unittest.TestCase):
    def make_metadata(self, official_test=()):
        # Give every subject three rows so a row-based split would be detectable.
        return pd.DataFrame([
            {
                "subject": subject,
                "original_partition": "test" if subject in official_test else "train",
                "segment_id": subject * 10 + segment,
            }
            for subject in range(1, 11)
            for segment in range(3)
        ])

    def assert_subjects_disjoint(self, metadata):
        self.assertTrue((metadata.groupby("subject")["split"].nunique() == 1).all())
        self.assertEqual(set(metadata["split"]), {"train", "val", "test"})

        groups = {
            split: set(metadata.loc[metadata.split == split, "subject"])
            for split in ("train", "val", "test")
        }
        self.assertTrue(groups["train"].isdisjoint(groups["val"]))
        self.assertTrue(groups["train"].isdisjoint(groups["test"]))
        self.assertTrue(groups["val"].isdisjoint(groups["test"]))

    def test_uci_keeps_official_test_subjects(self):
        metadata = self.make_metadata(official_test=(2, 7))
        result = assign_splits(metadata.copy(), "uci_har", seed=42)
        self.assert_subjects_disjoint(result)
        self.assertEqual(set(result.loc[result.split == "test", "subject"]), {2, 7})

        # Running again with the same seed should preserve every assignment.
        repeat = assign_splits(metadata.copy(), "uci_har", seed=42)
        pd.testing.assert_frame_equal(result, repeat)

    def test_wisdm_subject_splits_are_reproducible(self):
        metadata = self.make_metadata()
        metadata["original_partition"] = "raw"
        result = assign_splits(metadata.copy(), "wisdm", seed=42)
        self.assert_subjects_disjoint(result)
        self.assertEqual(
            result.groupby("split").subject.nunique().to_dict(),
            {"test": 2, "train": 6, "val": 2},
        )

        repeat = assign_splits(metadata.copy(), "wisdm", seed=42)
        pd.testing.assert_frame_equal(result, repeat)


if __name__ == "__main__":
    unittest.main()
