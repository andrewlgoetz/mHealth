"""Load prepared arrays and make source-scaled cross-dataset experiments."""

from pathlib import Path

import numpy as np


DATASETS = ("uci_har", "wisdm")
SPLITS = ("train", "val", "test")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"


def _subset(data, mask):
    """Feature names describe columns; every other array describes windows."""
    return {
        name: values if name == "feature_names" else values[mask]
        for name, values in data.items()
    }


def load_dataset(name, split=None, data_dir=None):
    """Load one dataset, optionally selecting train, val, or test subjects.

    ``data_dir`` is the directory containing the uci_har and wisdm folders.
    X has shape (windows, samples, 3); features has shape (windows, features).
    Metadata columns and feature names are separate from model inputs.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}.")
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}.")
    root = DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    path = root / name / "dataset.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if split is not None:
        data = _subset(data, data["split"] == split)
    return data


def load_transfer(source, target, data_dir=None):
    """Prepare both transfer directions with scaling fitted on source train.

    ``target_all`` is for zero-shot evaluation only. For adaptation, use
    ``target_adapt`` (target train + val subjects), then evaluate on the fixed,
    disjoint ``target_test`` subjects. Compare adapted and unadapted models on
    that same target_test set. No target data are used to fit these scalers.

    Scaling preserves gravity and orientation differences: channel statistics
    are learned over all source training windows and times, not per window.
    """
    source_data = load_dataset(source, data_dir=data_dir)
    target_data = load_dataset(target, data_dir=data_dir)
    if source == target:
        raise ValueError("Source and target must be different datasets.")
    if source_data["X"].shape[1:] != target_data["X"].shape[1:]:
        raise ValueError("Source and target window shapes must match.")
    if not np.array_equal(source_data["feature_names"], target_data["feature_names"]):
        raise ValueError("Source and target feature names must match in order.")

    train = _subset(source_data, source_data["split"] == "train")
    if len(train["y"]) == 0:
        raise ValueError("Source training split is empty.")
    scaler = {
        "X_mean": train["X"].mean(axis=(0, 1), dtype=np.float64),
        "X_std": train["X"].std(axis=(0, 1), dtype=np.float64),
        "feature_mean": train["features"].mean(axis=0, dtype=np.float64),
        "feature_std": train["features"].std(axis=0, dtype=np.float64),
    }
    # Constant source channels/features should remain finite after scaling.
    scaler["X_std"][scaler["X_std"] == 0] = 1.0
    scaler["feature_std"][scaler["feature_std"] == 0] = 1.0

    for data in (source_data, target_data):
        data["X"] = ((data["X"] - scaler["X_mean"]) / scaler["X_std"]).astype(
            np.float32
        )
        data["features"] = (
            (data["features"] - scaler["feature_mean"]) / scaler["feature_std"]
        ).astype(np.float32)

    return {
        "source_train": _subset(source_data, source_data["split"] == "train"),
        "source_val": _subset(source_data, source_data["split"] == "val"),
        "source_test": _subset(source_data, source_data["split"] == "test"),
        "target_all": target_data,
        "target_adapt": _subset(target_data, np.isin(target_data["split"], ("train", "val"))),
        "target_test": _subset(target_data, target_data["split"] == "test"),
        "scaler": scaler,
    }
