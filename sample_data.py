"""Print training examples with every feature from both prepared datasets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
ACTIVITIES = ["walking", "upstairs", "downstairs", "sitting", "standing"]


def print_samples(name, rows=5):
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run python data_process.py first.")

    with np.load(path, allow_pickle=False) as data:
        training_rows = np.flatnonzero(data["split"] == "train")

        # Start with one example per activity, rather than five near-identical
        # neighbouring windows from the same recording.
        selected_rows = []
        for label in range(len(ACTIVITIES)):
            matches = training_rows[data["labels"][training_rows] == label]
            if len(matches):
                selected_rows.append(int(matches[0]))

        extra_rows = [row for row in training_rows if row not in selected_rows]
        selected_rows = (selected_rows + extra_rows)[:rows]
        examples = pd.DataFrame(
            data["features"][selected_rows], columns=data["feature_names"]
        )
        examples.insert(0, "subject", data["subjects"][selected_rows])
        examples.insert(1, "label", data["labels"][selected_rows])
        examples.insert(
            2, "activity", [ACTIVITIES[label] for label in data["labels"][selected_rows]]
        )

    # Transposing keeps all 20 features visible without a very wide table.
    examples.index = [f"example_{index + 1}" for index in range(len(examples))]
    print(f"\n{name} - training examples (unscaled)")
    print(examples.T.to_string(float_format=lambda value: f"{value:.5f}"))
    print("subject, label, and activity are metadata; the remaining rows are features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")

    try:
        for dataset in ("uci_har", "wisdm"):
            print_samples(dataset, args.rows)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
