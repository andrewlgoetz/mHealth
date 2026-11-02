"""Small checks for feature extraction and leakage-free model inputs."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from features import extract_features
from model_data import load_transfer


class FeatureTests(unittest.TestCase):
    def test_sine_frequency_and_axis_correlation(self):
        time = np.arange(100) / 20
        signal = np.sin(2 * np.pi * 2 * time)
        windows = np.stack((signal, -signal, np.zeros(100)), axis=1)[None]
        values, names = extract_features(windows, sample_rate=20)
        self.assertEqual(values.shape, (1, 76))
        self.assertEqual(len(set(names)), 76)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values[0, names.index("x_dominant_frequency")], 2)
        self.assertAlmostEqual(values[0, names.index("correlation_xy")], -1)
        self.assertAlmostEqual(values[0, names.index("x_energy")], 0.5)

    def test_constant_signals_have_finite_features_and_zero_spectra(self):
        values, names = extract_features(np.full((2, 100, 3), 9.80665))
        self.assertTrue(np.isfinite(values).all())
        zero_columns = [
            i for i, name in enumerate(names)
            if any(part in name for part in (
                "dominant_frequency", "spectral_", "power_fraction_",
                "correlation_", "_std", "zero_crossing_rate",
            ))
        ]
        np.testing.assert_array_equal(values[:, zero_columns], 0)

    def test_empty_batch_retains_feature_names_and_shape(self):
        empty, names = extract_features(np.empty((0, 100, 3)))
        _, populated_names = extract_features(np.zeros((1, 100, 3)))
        self.assertEqual(empty.shape, (0, 76))
        self.assertEqual(names, populated_names)

    def test_invalid_windows_and_sample_rates_are_rejected(self):
        for windows in (np.zeros((100, 3)), np.zeros((1, 1, 3)),
                        np.full((1, 100, 3), np.nan)):
            with self.subTest(shape=windows.shape), self.assertRaises(ValueError):
                extract_features(windows)
        for rate in (0, -1, np.inf, np.nan):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                extract_features(np.zeros((1, 100, 3)), sample_rate=rate)


class TransferTests(unittest.TestCase):
    def test_scaling_uses_only_source_train_and_target_subjects_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, levels, first_subject in (
                ("uci_har", [1, 3, 100, 200], 1),
                ("wisdm", [1000, 2000, 3000, 4000], 10),
            ):
                values = np.array(levels, dtype=np.float32)
                folder = root / name
                folder.mkdir()
                np.savez(
                    folder / "dataset.npz",
                    X=np.broadcast_to(values[:, None, None], (4, 100, 3)).copy(),
                    features=np.column_stack((values * 10, np.full(4, 5))),
                    feature_names=np.array(["varying_feature", "constant_feature"]),
                    y=np.arange(4),
                    subjects=np.arange(first_subject, first_subject + 4),
                    segment_ids=np.arange(4),
                    start_seconds=np.zeros(4),
                    split=np.array(["train", "train", "val", "test"]),
                )
            data = load_transfer("uci_har", "wisdm", data_dir=root)

        np.testing.assert_allclose(data["scaler"]["X_mean"], 2)
        np.testing.assert_allclose(data["scaler"]["X_std"], 1)
        np.testing.assert_allclose(data["scaler"]["feature_mean"], [20, 5])
        np.testing.assert_allclose(data["scaler"]["feature_std"], [10, 1])
        np.testing.assert_allclose(data["source_train"]["X"][:, 0, 0], [-1, 1])
        np.testing.assert_allclose(data["target_all"]["X"][:, 0, 0], [998, 1998, 2998, 3998])
        np.testing.assert_allclose(data["target_all"]["features"][:, 0], [998, 1998, 2998, 3998])
        self.assertTrue(np.isfinite(data["target_all"]["features"]).all())
        adapt_subjects = set(data["target_adapt"]["subjects"])
        test_subjects = set(data["target_test"]["subjects"])
        self.assertEqual(adapt_subjects, {10, 11, 12})
        self.assertEqual(test_subjects, {13})
        self.assertTrue(adapt_subjects.isdisjoint(test_subjects))


if __name__ == "__main__":
    unittest.main()
