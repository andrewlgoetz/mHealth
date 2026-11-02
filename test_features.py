"""Check feature values using simple signals with known answers."""

import unittest

import numpy as np

from features import extract_features


class FeatureTests(unittest.TestCase):
    def test_known_sine_wave(self):
        # Five cycles of a 2 Hz sine wave fit in a 2.5-second window.
        time = np.arange(50) / 20
        signal = np.sin(2 * np.pi * 2 * time)
        windows = np.stack((signal, -signal, np.zeros(50)), axis=1)[None]
        features, names = extract_features(windows)

        self.assertEqual(features.shape, (1, 20))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(set(names)), 20)
        self.assertAlmostEqual(features[0, names.index("x_mean")], 0)
        self.assertAlmostEqual(features[0, names.index("x_std")], np.sqrt(0.5))
        self.assertAlmostEqual(features[0, names.index("x_rms")], np.sqrt(0.5))
        self.assertAlmostEqual(features[0, names.index("x_range")], np.ptp(signal))
        self.assertEqual(features[0, names.index("x_dominant_frequency")], 2)

    def test_constant_signal(self):
        features, names = extract_features(np.full((2, 50, 3), 9.80665))
        self.assertTrue(np.isfinite(features).all())
        for channel in ("x", "y", "z", "magnitude"):
            self.assertAlmostEqual(features[0, names.index(f"{channel}_std")], 0)
            self.assertEqual(features[0, names.index(f"{channel}_range")], 0)
            self.assertEqual(features[0, names.index(f"{channel}_dominant_frequency")], 0)
        self.assertAlmostEqual(features[0, names.index("x_rms")], 9.80665, places=5)
        self.assertAlmostEqual(
            features[0, names.index("magnitude_mean")], np.sqrt(3) * 9.80665,
            places=5,
        )

    def test_empty_batch_keeps_column_order(self):
        features, names = extract_features(np.empty((0, 50, 3)))
        self.assertEqual(features.shape, (0, 20))
        self.assertEqual(names[:5], [
            "x_mean", "x_std", "x_rms", "x_range", "x_dominant_frequency",
        ])
        self.assertEqual(names[-1], "magnitude_dominant_frequency")

    def test_invalid_inputs_are_rejected(self):
        for windows in (np.zeros((50, 3)), np.zeros((1, 1, 3)),
                        np.full((1, 50, 3), np.nan)):
            with self.subTest(shape=windows.shape), self.assertRaises(ValueError):
                extract_features(windows)
        for rate in (0, -1, np.nan, np.inf):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                extract_features(np.zeros((1, 50, 3)), sample_rate=rate)


if __name__ == "__main__":
    unittest.main()
