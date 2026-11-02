"""Shared statistical and frequency features for accelerometer windows."""

import numpy as np


def extract_features(
    windows: np.ndarray, sample_rate: float = 20.0
) -> tuple[np.ndarray, list[str]]:
    """Return 76 features per window, using x, y, z and acceleration magnitude.

    Input shape is (windows, samples, 3). Features use population moments;
    energy is mean squared acceleration, and zero crossings use demeaned data.
    Frequency features use the power of a demeaned, Hann-windowed FFT, without
    DC. Entropy is divided by log(number of frequency bins). A constant signal
    has zero frequency features and zero correlation with any other signal.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[2] != 3 or windows.shape[1] < 2:
        raise ValueError("windows must have shape (n_windows, n_samples >= 2, 3)")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not np.isfinite(windows).all():
        raise ValueError("windows must contain only finite values")

    channels = ("x", "y", "z", "magnitude")
    time_names = (
        "mean", "std", "min", "max", "median", "iqr", "rms",
        "mean_abs_deviation", "energy", "zero_crossing_rate", "range",
        "skewness", "excess_kurtosis",
    )
    frequency_names = (
        "dominant_frequency", "spectral_centroid", "spectral_entropy",
        "power_fraction_0.1_to_3hz", "power_fraction_3hz_to_nyquist",
    )
    names = [
        f"{channel}_{feature}"
        for channel in channels
        for feature in (*time_names, *frequency_names)
    ] + ["correlation_xy", "correlation_xz", "correlation_yz", "signal_magnitude_area"]
    if len(windows) == 0:
        return np.empty((0, len(names)), dtype=np.float32), names

    magnitude = np.linalg.norm(windows, axis=2, keepdims=True)
    signals = np.concatenate((windows, magnitude), axis=2)
    mean = signals.mean(axis=1)
    centered = signals - mean[:, None, :]
    spread = np.ptp(signals, axis=1)
    centered = np.where(spread[:, None, :] == 0, 0, centered)
    std = np.sqrt(np.mean(centered ** 2, axis=1))
    standardized = np.divide(
        centered, std[:, None, :], out=np.zeros_like(centered),
        where=std[:, None, :] > 0,
    )
    quartiles = np.percentile(signals, [25, 75], axis=1)
    energy = np.mean(signals ** 2, axis=1)
    kurtosis = np.mean(standardized ** 4, axis=1) - 3
    kurtosis[std == 0] = 0
    time_features = [
        mean,
        std,
        signals.min(axis=1),
        signals.max(axis=1),
        np.median(signals, axis=1),
        quartiles[1] - quartiles[0],
        np.sqrt(energy),
        np.mean(np.abs(centered), axis=1),
        energy,
        np.mean(np.signbit(centered[:, 1:]) != np.signbit(centered[:, :-1]), axis=1),
        spread,
        np.mean(standardized ** 3, axis=1),
        kurtosis,
    ]

    sample_count = signals.shape[1]
    frequencies = np.fft.rfftfreq(sample_count, d=1 / sample_rate)[1:]
    tapered = centered * np.hanning(sample_count)[None, :, None]
    power = np.abs(np.fft.rfft(tapered, axis=1)[:, 1:]) ** 2
    total_power = power.sum(axis=1)
    fractions = np.divide(
        power, total_power[:, None, :], out=np.zeros_like(power),
        where=total_power[:, None, :] > 0,
    )
    dominant = frequencies[np.argmax(power, axis=1)]
    dominant[total_power == 0] = 0
    log_fractions = np.log(np.maximum(fractions, np.finfo(float).tiny))
    low_band = (frequencies >= 0.1) & (frequencies < 3)
    high_band = (frequencies >= 3) & (frequencies < sample_rate / 2)
    frequency_features = [
        dominant,
        (fractions * frequencies[None, :, None]).sum(axis=1),
        -(fractions * log_fractions).sum(axis=1) / np.log(max(len(frequencies), 2)),
        fractions[:, low_band].sum(axis=1),
        fractions[:, high_band].sum(axis=1),
    ]
    # Stacking this way keeps each channel's features together in column order.
    features = np.stack(time_features + frequency_features, axis=2).reshape(len(windows), -1)

    correlations = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        covariance = np.mean(centered[:, :, first] * centered[:, :, second], axis=1)
        denominator = std[:, first] * std[:, second]
        correlation = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
        )
        correlations.append(np.clip(correlation, -1, 1))
    # Mean over time of |x| + |y| + |z|, without removing gravity.
    sma = np.abs(windows).sum(axis=2).mean(axis=1)
    features = np.column_stack((features, *correlations, sma))
    return features.astype(np.float32), names
