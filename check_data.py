"""Run the tests, then check that both prepared datasets are ready to use."""

import unittest
from pathlib import Path

import numpy as np

from features import extract_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"


def check_dataset(path):
    """Check saved windows, features, labels, and subject splits."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run python data_process.py first.")

    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    count = len(data["labels"])
    assert count > 0, f"{path.stem}: no examples were saved"
    assert data["windows"].shape == (count, 50, 3), "Expected 50 samples and 3 axes"
    assert data["features"].shape == (count, 20), "Expected 20 features per example"

    for name in ("labels", "subjects", "split"):
        assert data[name].shape == (count,), f"{name} rows do not match the windows"
    for name in ("windows", "features"):
        assert np.isfinite(data[name]).all(), f"{name} contains missing or infinite values"

    assert data["labels"].dtype.kind in "iu", "Labels must be integers"
    assert data["subjects"].dtype.kind in "iu", "Subjects must be integers"
    assert np.all(data["subjects"] > 0), "Subject IDs must be positive"
    assert set(data["labels"]) == set(range(5)), "Expected all five shared activities"
    assert set(data["split"]) == {"train", "val", "test"}, "Unexpected split names"

    # Every subject belongs to one split, keeping overlapping windows together.
    for subject in np.unique(data["subjects"]):
        subject_splits = data["split"][data["subjects"] == subject]
        assert len(set(subject_splits)) == 1, f"Subject {subject} crosses splits"

    for split in ("train", "val", "test"):
        selected = data["split"] == split
        assert set(data["labels"][selected]) == set(range(5)), f"{split} misses a class"

    # Recompute a few examples to check that feature rows match their windows.
    sample_rows = np.linspace(0, count - 1, min(8, count), dtype=int)
    expected_features, expected_names = extract_features(data["windows"][sample_rows])
    np.testing.assert_array_equal(data["feature_names"], expected_names)
    np.testing.assert_allclose(
        data["features"][sample_rows], expected_features, rtol=1e-5, atol=1e-5
    )

    split_counts = {
        split: int((data["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    print(f"{path.stem}: {count:,} examples, 20 features; {split_counts}")
    return data


def main():
    tests = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise SystemExit(1)

    try:
        uci = check_dataset(DATA_DIR / "uci_har.npz")
        wisdm = check_dataset(DATA_DIR / "wisdm.npz")
        np.testing.assert_array_equal(uci["feature_names"], wisdm["feature_names"])
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Data check failed: {error}") from error

    print("Both datasets passed: matching features and separate subject splits.")


if __name__ == "__main__":
    main()
