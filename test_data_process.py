"""Small checks for the dataset readers and subject splits."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data_process import (
    GRAVITY, load_uci, load_wisdm, make_splits, read_wisdm_recordings,
)


def record(subject, activity, timestamp, value):
    return f"{subject},{activity},{timestamp},{value},2,3;"


class DataTests(unittest.TestCase):
    def test_uci_windows_have_shared_shape_units_and_labels(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for partition, subject, label, value in (
                ("train", 1, 1, 2.0), ("test", 9, 3, 3.0),
            ):
                directory = root / partition
                signals = directory / "Inertial Signals"
                signals.mkdir(parents=True)
                np.savetxt(
                    directory / f"subject_{partition}.txt", [subject] * 2, fmt="%d",
                )
                np.savetxt(directory / f"y_{partition}.txt", [label, 6], fmt="%d")
                for axis in "xyz":
                    np.savetxt(
                        signals / f"total_acc_{axis}_{partition}.txt",
                        np.full((2, 128), value),
                    )
            windows, labels, subjects, test_subjects = load_uci(root)

        # Each released UCI window stays independent; laying is excluded.
        self.assertEqual(windows.shape, (2, 50, 3))
        # Allow the tiny amplitude ripple from the anti-aliasing filter.
        np.testing.assert_allclose(windows[0], 2 * GRAVITY, rtol=1e-4)
        np.testing.assert_allclose(windows[1], 3 * GRAVITY, rtol=1e-4)
        self.assertAlmostEqual(windows[0].mean(dtype=float), 2 * GRAVITY, places=5)
        self.assertAlmostEqual(windows[1].mean(dtype=float), 3 * GRAVITY, places=5)
        np.testing.assert_array_equal(labels, [0, 2])
        np.testing.assert_array_equal(subjects, [1, 9])
        self.assertEqual(set(test_subjects), {9})

    def test_wisdm_keeps_valid_readings_and_recording_boundaries(self):
        lines = [
            record(1, "Walking", 1_000_000_000, 10)
            + record(1, "Walking", 1_050_000_000, 11)
            + record(1, "Walking", 1_050_000_000, 11),  # Exact copy.
            record(1, "Walking", 1_050_000_000, 12),  # Conflicting timestamp.
            "broken record;",
            "1,Walking,1100000000,13,2,3,;",  # Complete record, extra comma.
            record(1, "Walking", 0, 99),
            record(1, "Walking", 1_150_000_000, 14),
            record(1, "Walking", 1_100_000_000, 15),  # Clock reset.
            record(1, "Walking", 1_150_000_000, 16),
            record(1, "Walking", 2_000_000_000, 17),  # Long gap.
            record(2, "Walking", 2_050_000_000, 18),
            record(2, "Jogging", 2_100_000_000, 19),
            record(2, "Standing", 2_150_000_000, 20),
            record(2, "Standing", 2_200_000_000, "nan"),
            record(2, "Standing", 2_250_000_000, 21),
            record(2, "Unknown", 2_300_000_000, 99),
            record(2, "Standing", 2_350_000_000, 22),
        ]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "raw.txt"
            path.write_text("\n".join(lines), encoding="utf-8")
            recordings = list(read_wisdm_recordings(path))

        values = [rows[:, 1].tolist() for rows, _, _ in recordings]
        self.assertEqual(values, [
            [10, 11], [12], [13], [14], [15, 16], [17], [18], [19], [20], [21], [22],
        ])
        # Preserve file order, even when the clock goes backwards.
        self.assertLess(recordings[4][0][0, 0], recordings[3][0][0, 0])

    def test_wisdm_windows_use_elapsed_time_and_exclude_jogging(self):
        # Irregular readings of a linear signal have known interpolated values.
        seconds = np.concatenate(([0, 0.07, 0.15, 0.25], np.arange(0.35, 5, 0.1)))
        lines = []
        for activity in ("Walking", "Jogging"):
            for time in seconds:
                timestamp = 1_000_000_000 + round(time * 1e9)
                lines.append(
                    f"1,{activity},{timestamp},{time},{2 * time + 3},{-time};"
                )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "raw.txt"
            path.write_text("\n".join(lines), encoding="utf-8")
            windows, labels, subjects = load_wisdm(path)

        self.assertEqual(windows.shape, (3, 50, 3))
        for index, window in enumerate(windows):
            time = np.arange(50) / 20 + index * 1.25
            expected = np.column_stack((time, 2 * time + 3, -time))
            np.testing.assert_allclose(window, expected, atol=1e-6)
        np.testing.assert_array_equal(labels, [0, 0, 0])
        np.testing.assert_array_equal(subjects, [1, 1, 1])

    def test_splits_keep_subjects_separate_and_preserve_uci_test(self):
        subjects = np.repeat(np.arange(1, 11), 3)
        for official_test in (None, [2, 7]):
            splits = make_splits(subjects, test_subjects=official_test)
            np.testing.assert_array_equal(
                splits, make_splits(subjects, test_subjects=official_test),
            )
            self.assertEqual(set(splits), {"train", "val", "test"})
            for subject in np.unique(subjects):
                self.assertEqual(len(set(splits[subjects == subject])), 1)
            if official_test is not None:
                self.assertEqual(set(subjects[splits == "test"]), {2, 7})


if __name__ == "__main__":
    unittest.main()
