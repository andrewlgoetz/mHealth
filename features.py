"""Calculate the same 20 simple features for either dataset."""

import numpy as np


def extract_features(windows, sample_rate=20.0):
    """Return a feature matrix and column names for (windows, samples, 3) data."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[2] != 3 or windows.shape[1] < 2:
        raise ValueError("Expected windows with shape (N, samples >= 2, 3).")
    if not np.isfinite(windows).all():
        raise ValueError("Acceleration values must be finite.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sampling rate must be finite and positive.")

    channels = ("x", "y", "z", "magnitude")
    statistics = ("mean", "std", "rms", "range", "dominant_frequency")
    names = [f"{channel}_{stat}" for channel in channels for stat in statistics]
    if len(windows) == 0:
        return np.empty((0, len(names)), dtype=np.float32), names

    # Magnitude combines the axes and is less sensitive to phone orientation.
    magnitude = np.linalg.norm(windows, axis=2, keepdims=True)
    signals = np.concatenate((windows, magnitude), axis=2)
    frequencies = np.fft.rfftfreq(windows.shape[1], d=1 / sample_rate)[1:]
    taper = np.hanning(windows.shape[1])
    columns = []

    for axis in range(4):
        signal = signals[:, :, axis]
        mean = signal.mean(axis=1)
        spread = np.ptp(signal, axis=1)

        # Remove the mean and taper before finding the strongest frequency.
        # Ignore the first FFT bin, which represents the constant/DC component.
        centered = signal - mean[:, None]
        spectrum = np.abs(np.fft.rfft(centered * taper, axis=1))[:, 1:]
        dominant_frequency = frequencies[np.argmax(spectrum, axis=1)]
        dominant_frequency[spread == 0] = 0

        columns.extend([
            mean,
            signal.std(axis=1),
            np.sqrt(np.mean(signal ** 2, axis=1)),
            spread,
            dominant_frequency,
        ])

    return np.column_stack(columns).astype(np.float32), names
