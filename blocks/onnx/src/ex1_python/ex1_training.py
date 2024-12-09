#!/usr/bin/env python3
"""Multi-peak spectral detector v3: Heatmap-based with proper supervision."""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import voigt_profile
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.ndimage import gaussian_filter1d

sys.modules["brotli"] = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    import keras

    KERAS_3 = int(keras.__version__.split('.')[0]) >= 3
except ImportError:
    from tensorflow import keras

    KERAS_3 = False

import matplotlib.pyplot as plt


# ============================================================================
# Device Detection
# ============================================================================

def print_device_info():
    print("=" * 60)
    print("TensorFlow Device Configuration")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__} (Keras 3: {KERAS_3})")

    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"CPUs: {len(cpus)}, GPUs: {len(gpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get('device_name', 'Unknown') if details else 'Unknown'
                print(f"  GPU {i}: {name}")
            except:
                print(f"  GPU {i}: {gpu.name}")
        device, device_name = "/GPU:0", "GPU"
    else:
        device, device_name = "/CPU:0", "CPU"

    print(f"Using: {device_name}")
    print("=" * 60)
    return device, device_name


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    fft_size: int = 4096
    kmax: int = 8
    snr_min_db: float = 6.0
    snr_max_db: float = 40.0
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    train_samples: int = 50000
    val_samples: int = 6250
    test_samples: int = 6250
    seed: int = 42
    peak_width_min_bins: float = 1.0  # Allow single-bin peaks
    peak_width_max_frac: float = 0.15
    baseline_slope_max: float = 0.3
    lambda_heatmap: float = 1.0
    lambda_reg: float = 2.0
    num_workers: int = 8
    heatmap_res: int = 512  # Increased from 256 for better position accuracy
    edge_margin: float = 0.05
    narrow_peak_prob: float = 0.3
    min_epochs: int = 30  # Minimum epochs before early stopping


# ============================================================================
# Peak Shape Generators
# ============================================================================

def peak_gaussian(bins: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((bins - center) / sigma) ** 2)


def peak_asymmetric_gaussian(bins: np.ndarray, center: float, sigma_l: float, sigma_r: float) -> np.ndarray:
    left = bins <= center
    out = np.empty_like(bins)
    out[left] = np.exp(-0.5 * ((bins[left] - center) / sigma_l) ** 2)
    out[~left] = np.exp(-0.5 * ((bins[~left] - center) / sigma_r) ** 2)
    return out


def peak_lorentzian(bins: np.ndarray, center: float, gamma: float) -> np.ndarray:
    return gamma ** 2 / (gamma ** 2 + (bins - center) ** 2)


def peak_sinc2(bins: np.ndarray, center: float, width: float) -> np.ndarray:
    x = np.pi * (bins - center) / max(width, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.sin(x) / x
        s = np.where(np.abs(x) < 1e-10, 1.0, s)
    return s ** 2


def peak_parabolic(bins: np.ndarray, center: float, width: float) -> np.ndarray:
    x = (bins - center) / max(width, 1.0)
    return np.clip(1.0 - x ** 2, 0.0, 1.0)


def peak_voigt(bins: np.ndarray, center: float, sigma: float, gamma: float) -> np.ndarray:
    v = voigt_profile(bins - center, max(sigma, 0.1), max(gamma, 0.1))
    return v / (v.max() + 1e-12)


def peak_dual_gaussian(bins: np.ndarray, center: float, sigma1: float, sigma2: float,
                       sep: float, ratio: float) -> np.ndarray:
    g1 = peak_gaussian(bins, center - sep / 2, max(sigma1, 1.0))
    g2 = peak_gaussian(bins, center + sep / 2, max(sigma2, 1.0)) * ratio
    combined = g1 + g2
    return combined / (combined.max() + 1e-12)


PEAK_GENERATORS: list[tuple[str, Callable, Callable]] = [
    ("gaussian", peak_gaussian,
     lambda rng, w: {"sigma": w}),
    ("asymmetric_gaussian", peak_asymmetric_gaussian,
     lambda rng, w: {"sigma_l": w * rng.uniform(0.5, 1.0), "sigma_r": w * rng.uniform(1.0, 2.0)}),
    ("lorentzian", peak_lorentzian,
     lambda rng, w: {"gamma": w}),
    ("sinc2", peak_sinc2,
     lambda rng, w: {"width": w * 2}),
    ("parabolic", peak_parabolic,
     lambda rng, w: {"width": w * 1.5}),
    ("voigt", peak_voigt,
     lambda rng, w: {"sigma": w * 0.7, "gamma": w * 0.5}),
    ("dual_gaussian", peak_dual_gaussian,
     lambda rng, w: {"sigma1": w * 0.6, "sigma2": w * 0.8,
                     "sep": w * rng.uniform(0.3, 1.2), "ratio": rng.uniform(0.3, 1.0)}),
]


# ============================================================================
# Width & Kurtosis
# ============================================================================

def compute_energy_containment_widths(peak_shape: np.ndarray, center_idx: int,
                                      thresholds: tuple = (0.68, 0.96, 0.997)) -> tuple[float, ...]:
    total_energy = np.sum(peak_shape ** 2)
    if total_energy < 1e-12:
        return tuple(1.0 for _ in thresholds)
    results = []
    for thresh in thresholds:
        target = thresh * total_energy
        for half_width in range(1, len(peak_shape)):
            lo, hi = max(0, center_idx - half_width), min(len(peak_shape), center_idx + half_width + 1)
            if np.sum(peak_shape[lo:hi] ** 2) >= target:
                results.append(2 * half_width)
                break
        else:
            results.append(float(len(peak_shape)))
    return tuple(results)


def compute_gaussian_equivalent_sigma(peak_shape: np.ndarray, center_idx: int) -> float:
    bins = np.arange(len(peak_shape))
    weights = peak_shape ** 2
    total = weights.sum()
    if total < 1e-12:
        return 1.0
    variance = np.sum(weights * (bins - center_idx) ** 2) / total
    return max(np.sqrt(variance), 1.0)


def compute_kurtosis_adaptive(spectrum: np.ndarray, center_idx: int, sigma: float) -> tuple[float, float]:
    half_win = max(3, int(3 * sigma))
    lo, hi = max(0, center_idx - half_win), min(len(spectrum), center_idx + half_win + 1)
    window = spectrum[lo:hi]
    if len(window) < 4:
        return 3.0, 0.0
    k = scipy_kurtosis(window, fisher=False)
    return float(k), float(k - 3.0)


# ============================================================================
# Spectrum Generator with Realistic Noise
# ============================================================================

@dataclass
class PeakParams:
    center: float
    amplitude: float
    sigma_equiv: float
    w68: float
    w96: float
    w99: float
    kurtosis: float
    excess_kurtosis: float
    shape_name: str = ""
    peak_shape: np.ndarray = None


def generate_realistic_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate noise with realistic FFT characteristics."""
    noise = rng.normal(0, 1, n)

    # Add colored noise component (low-frequency)
    if rng.random() < 0.5:
        color_scale = rng.uniform(0.1, 0.5)
        colored = gaussian_filter1d(rng.normal(0, 1, n), sigma=n / 20)
        noise += color_scale * colored / (np.std(colored) + 1e-10)

    # Add edge effects (higher variance at edges) - simulates FFT artifacts
    if rng.random() < 0.7:
        edge_width = int(n * 0.1)
        edge_boost = rng.uniform(1.5, 3.0)
        edge_taper = np.ones(n)
        edge_taper[:edge_width] = np.linspace(edge_boost, 1.0, edge_width)
        edge_taper[-edge_width:] = np.linspace(1.0, edge_boost, edge_width)
        noise *= edge_taper

    # Occasional spurious spikes (to train the model to reject them)
    if rng.random() < 0.3:
        num_spikes = rng.integers(1, 5)
        for _ in range(num_spikes):
            spike_pos = rng.integers(0, n)
            spike_amp = rng.uniform(2, 5)  # 2-5 sigma spikes
            noise[spike_pos] += spike_amp * (1 if rng.random() > 0.5 else -1)

    return noise


def generate_baseline(n: int, rng: np.random.Generator, slope_max: float) -> np.ndarray:
    if rng.random() < 0.4:
        return np.zeros(n)
    # Linear slope
    slope = rng.uniform(-slope_max, slope_max)
    intercept = rng.uniform(-0.5, 0.5)
    baseline = intercept + slope * np.linspace(-0.5, 0.5, n)
    # Add curvature
    if rng.random() < 0.3:
        curve = rng.uniform(-0.2, 0.2)
        baseline += curve * (np.linspace(-1, 1, n) ** 2)
    return baseline


def generate_single_spectrum(cfg: TrainConfig, rng: np.random.Generator,
                             store_shapes: bool = False
                             ) -> tuple[np.ndarray, list[PeakParams], np.ndarray]:
    n = cfg.fft_size
    bins = np.arange(n, dtype=np.float64)

    baseline = generate_baseline(n, rng, cfg.baseline_slope_max)
    noise = generate_realistic_noise(n, rng)
    spectrum = baseline + noise

    edge_margin = int(n * cfg.edge_margin)
    num_peaks = rng.integers(0, cfg.kmax + 1)
    peaks: list[PeakParams] = []

    for _ in range(num_peaks):
        snr_db = rng.uniform(cfg.snr_min_db, cfg.snr_max_db)
        amplitude = 1.0 * (10 ** (snr_db / 20))

        # Decide if this is a narrow peak
        if rng.random() < cfg.narrow_peak_prob:
            # Narrow peak: 1-5 bins
            width_bins = rng.uniform(1.0, 5.0)
            # Use only Gaussian or sinc2 for narrow peaks (most realistic)
            idx = rng.choice([0, 3])  # gaussian or sinc2
        else:
            # Normal width distribution
            width_bins = rng.uniform(cfg.peak_width_min_bins, cfg.peak_width_max_frac * n)
            idx = rng.integers(0, len(PEAK_GENERATORS))

        margin = max(edge_margin, int(3 * width_bins))
        if margin >= n // 2:
            margin = edge_margin
        center = rng.uniform(margin, n - margin)

        shape_name, shape_fn, param_sampler = PEAK_GENERATORS[idx]
        shape_params = param_sampler(rng, max(width_bins, 1.0))
        peak_shape = shape_fn(bins, center, **shape_params)

        sigma_equiv = compute_gaussian_equivalent_sigma(peak_shape, int(center))
        w68, w96, w99 = compute_energy_containment_widths(peak_shape, int(center))

        scaled_peak = amplitude * peak_shape
        spectrum += scaled_peak

        kurt, excess_kurt = compute_kurtosis_adaptive(scaled_peak, int(center), sigma_equiv)

        peaks.append(PeakParams(
            center=center, amplitude=amplitude, sigma_equiv=sigma_equiv,
            w68=w68, w96=w96, w99=w99, kurtosis=kurt, excess_kurtosis=excess_kurt,
            shape_name=shape_name, peak_shape=scaled_peak if store_shapes else None
        ))

    peaks.sort(key=lambda p: p.amplitude, reverse=True)
    return spectrum, peaks, baseline


def spectrum_to_normalized(spectrum: np.ndarray) -> np.ndarray:
    """Robust normalization that handles negative values properly."""
    # Use minimum to ensure all values become positive
    floor = np.min(spectrum)
    shifted = spectrum - floor + 1.0  # Now all values >= 1.0

    # Log transform (safe since all values >= 1.0)
    log_spec = np.log10(shifted)

    # Check for any NaN/inf and replace
    if not np.all(np.isfinite(log_spec)):
        log_spec = np.nan_to_num(log_spec, nan=0.0, posinf=10.0, neginf=-10.0)

    # Robust normalization (median and MAD)
    median = np.median(log_spec)
    mad = np.median(np.abs(log_spec - median))
    if mad < 1e-10:
        mad = np.std(log_spec) + 1e-10
    normalized = (log_spec - median) / (1.4826 * mad + 1e-10)

    # Clip extremes
    normalized = np.clip(normalized, -5, 10)

    return normalized.astype(np.float32)


# ============================================================================
# Heatmap Target Generation
# ============================================================================

def generate_heatmap_target(peaks: list[PeakParams], cfg: TrainConfig) -> np.ndarray:
    """Generate Gaussian heatmap target for peak positions.

    Use FIXED sigma for sharp, localized targets regardless of peak width.
    This teaches the model to output sharp spikes at peak centers.
    """
    n = cfg.fft_size
    heatmap_res = cfg.heatmap_res
    heatmap = np.zeros(heatmap_res, dtype=np.float32)

    # Fixed sigma: ~2 heatmap bins = ~32 spectrum bins for N=4096, H=256
    sigma_hm = 2.0

    for p in peaks:
        center_hm = p.center * heatmap_res / n
        x = np.arange(heatmap_res)
        gaussian = np.exp(-0.5 * ((x - center_hm) / sigma_hm) ** 2)
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


def generate_regression_target(peaks: list[PeakParams], cfg: TrainConfig) -> np.ndarray:
    """Generate per-position regression targets."""
    n = cfg.fft_size
    heatmap_res = cfg.heatmap_res

    # Regression channels: offset, amplitude, sigma, w68, w96, w99, kurtosis, excess_kurt
    reg_target = np.zeros((heatmap_res, 8), dtype=np.float32)
    reg_mask = np.zeros(heatmap_res, dtype=np.float32)

    for p in peaks:
        center_hm = int(p.center * heatmap_res / n)
        center_hm = np.clip(center_hm, 0, heatmap_res - 1)

        # Offset from grid position to true position (in normalized coords)
        grid_pos = center_hm * n / heatmap_res
        offset = (p.center - grid_pos) / n  # Normalized offset

        # Fill regression target at peak position
        frame_max_amp = max(pp.amplitude for pp in peaks) if peaks else 1.0

        # Sanitize values
        def safe_val(v, default=0.0):
            return float(v) if np.isfinite(v) else default

        reg_target[center_hm, 0] = safe_val(offset, 0.0)
        reg_target[center_hm, 1] = safe_val(p.amplitude / frame_max_amp, 0.5)
        reg_target[center_hm, 2] = safe_val(p.sigma_equiv / n, 0.01)
        reg_target[center_hm, 3] = safe_val(p.w68 / n, 0.01)
        reg_target[center_hm, 4] = safe_val(p.w96 / n, 0.02)
        reg_target[center_hm, 5] = safe_val(p.w99 / n, 0.03)
        reg_target[center_hm, 6] = np.clip(safe_val(p.kurtosis / 10.0, 0.3), 0, 1)
        reg_target[center_hm, 7] = np.clip(safe_val((p.excess_kurtosis + 5) / 10.0, 0.5), 0, 1)
        reg_mask[center_hm] = 1.0

    return reg_target, reg_mask


# ============================================================================
# Dataset Generation
# ============================================================================

def _generate_batch_heatmap(args: tuple) -> tuple:
    start_idx, count, cfg_dict, seed = args
    cfg = TrainConfig(**cfg_dict)
    rng = np.random.default_rng(seed)

    X = np.empty((count, cfg.fft_size), dtype=np.float32)
    Y_hm = np.empty((count, cfg.heatmap_res), dtype=np.float32)
    Y_reg = np.empty((count, cfg.heatmap_res, 8), dtype=np.float32)
    Y_mask = np.empty((count, cfg.heatmap_res), dtype=np.float32)

    for i in range(count):
        spectrum, peaks, _ = generate_single_spectrum(cfg, rng)
        X[i] = spectrum_to_normalized(spectrum)
        Y_hm[i] = generate_heatmap_target(peaks, cfg)
        Y_reg[i], Y_mask[i] = generate_regression_target(peaks, cfg)

    # Replace any NaN with zeros
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    Y_hm = np.nan_to_num(Y_hm, nan=0.0)
    Y_reg = np.nan_to_num(Y_reg, nan=0.0)
    Y_mask = np.nan_to_num(Y_mask, nan=0.0)

    return start_idx, X, Y_hm, Y_reg, Y_mask

    return start_idx, X, Y_hm, Y_reg, Y_mask


def generate_dataset(cfg: TrainConfig, num_samples: int, seed: int,
                     desc: str = "Generating", parallel: bool = True) -> tuple:
    X = np.empty((num_samples, cfg.fft_size), dtype=np.float32)
    Y_hm = np.empty((num_samples, cfg.heatmap_res), dtype=np.float32)
    Y_reg = np.empty((num_samples, cfg.heatmap_res, 8), dtype=np.float32)
    Y_mask = np.empty((num_samples, cfg.heatmap_res), dtype=np.float32)

    cfg_dict = {k: getattr(cfg, k) for k in [
        'fft_size', 'kmax', 'snr_min_db', 'snr_max_db', 'peak_width_min_bins',
        'peak_width_max_frac', 'baseline_slope_max', 'heatmap_res', 'edge_margin',
        'narrow_peak_prob'
    ]}

    if parallel and cfg.num_workers > 1:
        num_workers = min(cfg.num_workers, num_samples)
        chunk_size = num_samples // num_workers
        remainder = num_samples % num_workers

        tasks = []
        start = 0
        for i in range(num_workers):
            count = chunk_size + (1 if i < remainder else 0)
            if count > 0:
                tasks.append((start, count, cfg_dict, seed + i * 10000))
                start += count

        print(f"{desc} (n={num_samples}, workers={num_workers})...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_batch_heatmap, t): t[0] for t in tasks}
            completed = 0
            for future in as_completed(futures):
                idx, bx, bhm, breg, bmask = future.result()
                count = bx.shape[0]
                X[idx:idx + count] = bx
                Y_hm[idx:idx + count] = bhm
                Y_reg[idx:idx + count] = breg
                Y_mask[idx:idx + count] = bmask
                completed += 1
                if completed % max(1, len(tasks) // 5) == 0 or completed == len(tasks):
                    print(f"  Progress: {100 * completed // len(tasks):3d}%")
    else:
        print(f"{desc} (n={num_samples}, sequential)...")
        rng = np.random.default_rng(seed)
        for i in range(num_samples):
            spectrum, peaks, _ = generate_single_spectrum(cfg, rng)
            X[i] = spectrum_to_normalized(spectrum)
            Y_hm[i] = generate_heatmap_target(peaks, cfg)
            Y_reg[i], Y_mask[i] = generate_regression_target(peaks, cfg)
            if (i + 1) % (num_samples // 10) == 0:
                print(f"  Progress: {100 * (i + 1) // num_samples:3d}%")

    return X, Y_hm, Y_reg, Y_mask


# ============================================================================
# Model Architecture: Proper Heatmap Detection
# ============================================================================

def build_model(cfg: TrainConfig) -> keras.Model:
    """
    Encoder architecture with dynamic downsampling to match heatmap_res.
    Output: heatmap (H,) + regression (H, 8)
    """
    n = cfg.fft_size
    H = cfg.heatmap_res

    # Calculate required downsampling factor
    downsample_factor = n // H  # e.g., 4096/512 = 8
    num_pools = int(np.log2(downsample_factor))  # e.g., log2(8) = 3

    inp = keras.Input(shape=(n, 1), name="spectrum")

    def conv_block(x, filters, kernel=5):
        x = keras.layers.Conv1D(filters, kernel, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters, kernel, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    # Dynamic encoder based on required downsampling
    filters_list = [32, 64, 128, 256, 512][:num_pools + 1]
    kernels_list = [7, 5, 5, 3, 3][:num_pools + 1]

    x = inp
    for i in range(num_pools):
        x = conv_block(x, filters_list[i], kernels_list[i])
        x = keras.layers.MaxPool1D(2)(x)

    # Bottleneck
    x = conv_block(x, filters_list[-1], 3)
    x = keras.layers.Conv1D(filters_list[-1], 3, dilation_rate=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(filters_list[-1], 3, dilation_rate=4, padding="same", activation="relu")(x)

    # Detection heads
    head = keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    head = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(head)

    # Heatmap output
    heatmap = keras.layers.Conv1D(1, 1, activation="sigmoid", name="heatmap_raw")(head)
    heatmap = keras.layers.Reshape((H,), name="heatmap")(heatmap)

    # Regression output
    regression = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(head)
    regression = keras.layers.Conv1D(8, 1, activation="linear", name="regression")(regression)

    model = keras.Model(inputs=inp, outputs=[heatmap, regression], name="peak_detector")
    return model


# ============================================================================
# Loss Functions
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for heatmap - handles class imbalance."""

    def loss_fn(y_true, y_pred):
        eps = 1e-6
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)

        # Binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        # Focal weight
        pt = tf.where(y_true > 0.5, y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt + eps, gamma)

        # Alpha weighting (higher for positives since they're rare)
        alpha_weight = tf.where(y_true > 0.5, alpha, 1 - alpha)

        loss = focal_weight * alpha_weight * bce

        # Filter out any NaN
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

        return tf.reduce_mean(loss)

    return loss_fn


def masked_regression_loss(y_true, y_pred, mask):
    """Huber loss only at peak positions."""
    diff = y_true - y_pred
    abs_diff = tf.abs(diff)
    delta = 1.0
    huber = tf.where(abs_diff <= delta, 0.5 * tf.square(diff), delta * (abs_diff - 0.5 * delta))

    # Filter NaN
    huber = tf.where(tf.math.is_finite(huber), huber, tf.zeros_like(huber))

    # Average over channels, then mask
    loss_per_pos = tf.reduce_mean(huber, axis=-1)  # (batch, H)
    masked = loss_per_pos * mask

    total_mask = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(masked) / total_mask


class CombinedLoss(keras.losses.Loss):
    def __init__(self, lambda_heatmap=1.0, lambda_reg=2.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_heatmap = lambda_heatmap
        self.lambda_reg = lambda_reg
        self.focal = focal_loss(gamma=2.0, alpha=0.75)

    def call(self, y_true, y_pred):
        # y_true: concatenated [heatmap (H), regression (H*8), mask (H)]
        # y_pred: [heatmap (H), regression (H, 8)]
        H = y_pred[0].shape[-1]

        hm_true = y_true[:, :H]
        reg_true = tf.reshape(y_true[:, H:H + H * 8], (-1, H, 8))
        mask = y_true[:, H + H * 8:]

        hm_pred = y_pred[0]
        reg_pred = y_pred[1]

        loss_hm = self.focal(hm_true, hm_pred)
        loss_reg = masked_regression_loss(reg_true, reg_pred, mask)

        return self.lambda_heatmap * loss_hm + self.lambda_reg * loss_reg


# ============================================================================
# Training
# ============================================================================

def train(cfg: TrainConfig, device: str, parallel: bool = True) -> keras.Model:
    X_train, Y_hm_train, Y_reg_train, Y_mask_train = generate_dataset(
        cfg, cfg.train_samples, cfg.seed, "Training data", parallel)
    X_val, Y_hm_val, Y_reg_val, Y_mask_val = generate_dataset(
        cfg, cfg.val_samples, cfg.seed + 1000, "Validation data", parallel)

    # Reshape for model
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # Concatenate targets for loss function
    H = cfg.heatmap_res
    Y_train = np.concatenate([Y_hm_train, Y_reg_train.reshape(-1, H * 8), Y_mask_train], axis=1)
    Y_val = np.concatenate([Y_hm_val, Y_reg_val.reshape(-1, H * 8), Y_mask_val], axis=1)

    print(f"\nData: X={X_train.shape}, Y={Y_train.shape}")

    with tf.device(device):
        model = build_model(cfg)

        # Learning rate schedule: warmup + cosine decay
        steps_per_epoch = cfg.train_samples // cfg.batch_size
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = 3 * steps_per_epoch  # 3 epoch warmup

        def lr_schedule(step):
            step = tf.cast(step, tf.float32)
            warmup_steps_f = tf.cast(warmup_steps, tf.float32)
            total_steps_f = tf.cast(total_steps, tf.float32)

            # Warmup phase
            warmup_lr = cfg.learning_rate * (step / warmup_steps_f)

            # Cosine decay phase
            progress = (step - warmup_steps_f) / (total_steps_f - warmup_steps_f)
            cosine_lr = cfg.learning_rate * 0.5 * (1.0 + tf.cos(np.pi * progress))

            return tf.where(step < warmup_steps_f, warmup_lr, tf.maximum(cosine_lr, 1e-6))

        lr_scheduler = keras.optimizers.schedules.LearningRateSchedule

        # Use Adam with gradient clipping for stability
        optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0)
        focal = focal_loss(gamma=2.0, alpha=0.85)  # Higher alpha weights positives more (better recall)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        @tf.function
        def train_step(x, y):
            H = cfg.heatmap_res
            hm_true = y[:, :H]
            reg_true = tf.reshape(y[:, H:H + H * 8], (-1, H, 8))
            mask = y[:, H + H * 8:]

            with tf.GradientTape() as tape:
                hm_pred, reg_pred = model(x, training=True)
                loss_hm = focal(hm_true, hm_pred)
                loss_reg = masked_regression_loss(reg_true, reg_pred, mask)
                loss = cfg.lambda_heatmap * loss_hm + cfg.lambda_reg * loss_reg

            grads = tape.gradient(loss, model.trainable_variables)
            # Clip gradients
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss, loss_hm, loss_reg

        @tf.function
        def val_step(x, y):
            H = cfg.heatmap_res
            hm_true = y[:, :H]
            reg_true = tf.reshape(y[:, H:H + H * 8], (-1, H, 8))
            mask = y[:, H + H * 8:]

            hm_pred, reg_pred = model(x, training=False)
            loss_hm = focal(hm_true, hm_pred)
            loss_reg = masked_regression_loss(reg_true, reg_pred, mask)
            return cfg.lambda_heatmap * loss_hm + cfg.lambda_reg * loss_reg

        # Training loop
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(cfg.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(cfg.batch_size)

        best_val_loss = float('inf')
        patience = 20  # Increased patience
        patience_counter = 0
        best_weights = model.get_weights()

        # Track moving average of validation loss for stability
        val_loss_ema = None
        ema_alpha = 0.3

        model.summary()

        for epoch in range(cfg.epochs):
            # Update learning rate (manual warmup + decay)
            if epoch < 3:
                new_lr = cfg.learning_rate * (epoch + 1) / 3
            else:
                progress = (epoch - 3) / max(cfg.epochs - 3, 1)
                new_lr = cfg.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
                new_lr = max(new_lr, 1e-6)
            optimizer.learning_rate.assign(new_lr)

            # Training
            train_losses = []
            for x_batch, y_batch in train_ds:
                loss, loss_hm, loss_reg = train_step(x_batch, y_batch)
                loss_val = loss.numpy()
                if np.isfinite(loss_val):
                    train_losses.append(loss_val)

            # Validation
            val_losses = []
            for x_batch, y_batch in val_ds:
                val_loss_batch = val_step(x_batch, y_batch).numpy()
                if np.isfinite(val_loss_batch):
                    val_losses.append(val_loss_batch)

            train_loss = np.mean(train_losses) if train_losses else float('nan')
            val_loss = np.mean(val_losses) if val_losses else float('nan')

            # EMA smoothing for more stable early stopping
            if val_loss_ema is None:
                val_loss_ema = val_loss
            else:
                val_loss_ema = ema_alpha * val_loss + (1 - ema_alpha) * val_loss_ema

            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs}: train={train_loss:.4f}, val={val_loss:.4f} (ema={val_loss_ema:.4f}), lr={new_lr:.2e}")

            # Check for NaN
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                print(f"  Warning: NaN detected")
                patience_counter += 1
                if patience_counter >= 5:
                    print("Too many NaN epochs, stopping.")
                    break
                continue

            # Early stopping based on EMA (only after min_epochs)
            if val_loss_ema < best_val_loss - 1e-4:  # Require meaningful improvement
                best_val_loss = val_loss_ema
                patience_counter = 0
                best_weights = model.get_weights()
            elif epoch >= cfg.min_epochs:  # Only count patience after min_epochs
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        model.set_weights(best_weights)

    return model


# ============================================================================
# Inference: Extract Peaks from Heatmap
# ============================================================================

def extract_peaks_from_heatmap(heatmap: np.ndarray, regression: np.ndarray,
                               cfg: TrainConfig, threshold: float = 0.3,
                               min_distance: int = 8) -> list[dict]:
    """Extract peaks using local maxima detection with proper NMS.

    min_distance is in HEATMAP space (multiply by n/H for spectrum bins).
    """
    H = len(heatmap)
    n = cfg.fft_size

    # Find local maxima: position i is a max if heatmap[i] > heatmap[i-1] and heatmap[i] > heatmap[i+1]
    local_max = np.zeros(H, dtype=bool)
    for i in range(1, H - 1):
        if heatmap[i] > heatmap[i - 1] and heatmap[i] > heatmap[i + 1]:
            local_max[i] = True
    # Handle edges
    if heatmap[0] > heatmap[1]:
        local_max[0] = True
    if heatmap[-1] > heatmap[-2]:
        local_max[-1] = True

    # Get candidate positions above threshold
    candidates = []
    for i in range(H):
        if local_max[i] and heatmap[i] >= threshold:
            candidates.append((heatmap[i], i))

    # Sort by confidence (highest first)
    candidates.sort(reverse=True)

    peaks = []
    suppressed = np.zeros(H, dtype=bool)

    for conf, idx in candidates:
        if suppressed[idx]:
            continue
        if len(peaks) >= cfg.kmax:
            break

        # Get position with offset correction
        grid_pos = idx * n / H
        offset = regression[idx, 0] * n
        center = grid_pos + offset
        center = np.clip(center, 0, n - 1)

        # Get other parameters
        peaks.append({
            'confidence': float(conf),
            'center': float(center),
            'amplitude': float(regression[idx, 1]),
            'sigma': float(regression[idx, 2] * n),
            'w68': float(regression[idx, 3] * n),
            'w96': float(regression[idx, 4] * n),
            'w99': float(regression[idx, 5] * n),
            'kurtosis': float(regression[idx, 6] * 10),
            'excess_kurtosis': float(regression[idx, 7] * 10 - 5),
        })

        # Suppress nearby positions
        lo = max(0, idx - min_distance)
        hi = min(H, idx + min_distance + 1)
        suppressed[lo:hi] = True

    return peaks


def predict_peaks(model: keras.Model, spectrum: np.ndarray, cfg: TrainConfig,
                  threshold: float = 0.5) -> list[dict]:
    """Run inference on a single spectrum."""
    normalized = spectrum_to_normalized(spectrum)
    X = normalized[np.newaxis, :, np.newaxis]

    heatmap, regression = model.predict(X, verbose=0)

    return extract_peaks_from_heatmap(heatmap[0], regression[0], cfg, threshold)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: keras.Model, cfg: TrainConfig, parallel: bool = True, threshold: float = 0.5):
    X_test, Y_hm_test, Y_reg_test, Y_mask_test = generate_dataset(
        cfg, cfg.test_samples, cfg.seed + 2000, "Test data", parallel)

    print(f"\nEvaluating with threshold={threshold}...")

    n_eval = min(500, cfg.test_samples)

    center_errors = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Tolerance in bins - should be a bit larger than heatmap grid size
    match_tolerance = cfg.fft_size / cfg.heatmap_res * 3  # ~48 bins for N=4096, H=256

    for i in range(n_eval):
        # Get ground truth peaks from mask
        gt_peaks = []
        for j in range(cfg.heatmap_res):
            if Y_mask_test[i, j] > 0.5:
                center = j * cfg.fft_size / cfg.heatmap_res + Y_reg_test[i, j, 0] * cfg.fft_size
                gt_peaks.append(center)

        # Predict
        X = X_test[i:i + 1, :, np.newaxis]
        heatmap, regression = model.predict(X, verbose=0)
        pred_peaks = extract_peaks_from_heatmap(heatmap[0], regression[0], cfg, threshold=threshold)
        pred_centers = [p['center'] for p in pred_peaks]

        # Match predictions to ground truth (greedy matching)
        matched_gt = set()
        for pred_c in pred_centers:
            best_match = None
            best_dist = float('inf')
            for gi, gt_c in enumerate(gt_peaks):
                if gi not in matched_gt:
                    dist = abs(pred_c - gt_c)
                    if dist < best_dist and dist < match_tolerance:
                        best_dist = dist
                        best_match = gi

            if best_match is not None:
                center_errors.append(best_dist)
                matched_gt.add(best_match)
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += len(gt_peaks) - len(matched_gt)

    print(f"\nResults on {n_eval} samples:")
    print(f"  True positives:  {true_positives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")

    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"\nPosition accuracy (matched peaks):")
        print(f"  MAE:      {np.mean(center_errors):.2f} bins")
        print(f"  Median:   {np.median(center_errors):.2f} bins")
        print(f"  90th pct: {np.percentile(center_errors, 90):.2f} bins")


# ============================================================================
# Visualization
# ============================================================================

def plot_spectrum_with_peaks(cfg: TrainConfig, model: keras.Model = None,
                             seed: int = 12345, threshold: float = 0.3,
                             save_path: str = None, show: bool = True):
    rng = np.random.default_rng(seed)
    n = cfg.fft_size
    bins = np.arange(n)

    spectrum, true_peaks, baseline = generate_single_spectrum(cfg, rng, store_shapes=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 2, 1]})

    # Top: Linear spectrum
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(true_peaks), 1)))
    for i, peak in enumerate(true_peaks):
        if peak.peak_shape is not None:
            ax1.fill_between(bins, 0, peak.peak_shape, alpha=0.3, color=colors[i],
                             label=f'P{i + 1}: {peak.shape_name}')
    ax1.plot(bins, spectrum, 'b-', alpha=0.8, linewidth=0.5, label='Spectrum')
    for i, peak in enumerate(true_peaks):
        ax1.plot(peak.center, peak.amplitude, 'g^', markersize=10, markeredgecolor='darkgreen',
                 markeredgewidth=2, zorder=10)
        ax1.annotate(f'T{i + 1}', (peak.center, peak.amplitude), textcoords="offset points",
                     xytext=(5, 5), fontsize=9, color='darkgreen', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Spectrum (N={n}, {len(true_peaks)} true peaks)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Middle: Normalized + detections
    ax2 = axes[1]
    normalized = spectrum_to_normalized(spectrum)
    ax2.plot(bins, normalized, 'b-', linewidth=0.5, label='Normalized')

    for i, peak in enumerate(true_peaks):
        ax2.axvline(peak.center, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

    detected_peaks = []
    if model is not None:
        detected_peaks = predict_peaks(model, spectrum, cfg, threshold)
        for i, p in enumerate(detected_peaks):
            ax2.axvline(p['center'], color='red', linestyle='-', alpha=0.7, linewidth=2)
            y_pos = normalized[int(np.clip(p['center'], 0, n - 1))]
            ax2.plot(p['center'], y_pos, 'rv', markersize=10, markeredgecolor='darkred',
                     markeredgewidth=2, zorder=11)
            ax2.annotate(f"D{i + 1}\np={p['confidence']:.2f}", (p['center'], y_pos),
                         textcoords="offset points", xytext=(-15, -30), fontsize=8,
                         color='darkred', fontweight='bold')

    ax2.set_ylabel('Normalized')
    ax2.set_title(f'Detections: {len(detected_peaks)} (threshold={threshold})')
    ax2.grid(True, alpha=0.3)

    # Bottom: Heatmap
    ax3 = axes[2]
    if model is not None:
        X = normalized[np.newaxis, :, np.newaxis]
        heatmap, _ = model.predict(X, verbose=0)
        hm_x = np.linspace(0, n, cfg.heatmap_res)
        ax3.fill_between(hm_x, 0, heatmap[0], alpha=0.5, color='orange')
        ax3.plot(hm_x, heatmap[0], 'r-', linewidth=1)
        ax3.axhline(threshold, color='gray', linestyle='--', label=f'Threshold={threshold}')
        ax3.set_ylabel('Heatmap')
        ax3.set_xlabel('Bin')
        ax3.legend(loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'No model', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlim(0, n)
    ax3.grid(True, alpha=0.3)

    # Info box
    if true_peaks:
        info = f"True peaks ({len(true_peaks)}):\n"
        for i, p in enumerate(true_peaks[:8]):
            info += f"  T{i + 1}: c={p.center:.0f}, A={p.amplitude:.1f}, σ={p.sigma_equiv:.1f}\n"
        if detected_peaks:
            info += f"\nDetected ({len(detected_peaks)}):\n"
            for i, p in enumerate(detected_peaks[:8]):
                info += f"  D{i + 1}: c={p['center']:.0f}, p={p['confidence']:.2f}\n"
        fig.text(0.02, 0.02, info, fontsize=8, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()

    return fig


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model: keras.Model, cfg: TrainConfig, output_dir: str = "."):
    import subprocess
    import shutil

    n, H = cfg.fft_size, cfg.heatmap_res
    snr_int = int(cfg.snr_min_db)
    base_name = f"peak_detector_N{n}_H{H}_SNR{snr_int}"
    saved_model_dir = f"{output_dir}/{base_name}_saved"
    onnx_path = f"{output_dir}/{base_name}.onnx"

    # Create wrapper model with single concatenated output for ONNX
    inp = keras.Input(shape=(1, n), dtype="float32", name="spectrum")
    x = keras.ops.transpose(inp, [0, 2, 1])
    heatmap, regression = model(x, training=False)
    # Concatenate outputs: (batch, H + H*8)
    reg_flat = keras.ops.reshape(regression, (-1, H * 8))
    output = keras.ops.concatenate([heatmap, reg_flat], axis=-1)
    export_model = keras.Model(inputs=inp, outputs=output, name="peak_detector_export")

    export_model.export(saved_model_dir)
    print(f"SavedModel: {saved_model_dir}")

    # Try tf2onnx conversion
    result = subprocess.run(
        ["python", "-m", "tf2onnx.convert", "--saved-model", saved_model_dir,
         "--output", onnx_path, "--opset", "17"],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"ONNX: {onnx_path}")
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    else:
        print(f"ONNX export failed!")
        print(f"  stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"  stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        print(f"  SavedModel retained: {saved_model_dir}")

        # Try alternative: direct onnx export if available
        try:
            import tf2onnx
            import onnx
            print("Trying in-process tf2onnx conversion...")
            spec = (tf.TensorSpec((None, 1, n), tf.float32, name="spectrum"),)
            model_proto, _ = tf2onnx.convert.from_keras(export_model, input_signature=spec, opset=17)
            onnx.save(model_proto, onnx_path)
            print(f"ONNX (in-process): {onnx_path}")
            shutil.rmtree(saved_model_dir, ignore_errors=True)
        except Exception as e:
            print(f"In-process conversion also failed: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Peak detector v3")
    parser.add_argument("--fft-size", type=int, default=4096, choices=[1024, 2048, 4096, 8192])
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--snr-min", type=float, default=6.0)
    parser.add_argument("--snr-max", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--plot-seed", type=int, default=12345)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--heatmap-res", type=int, default=None,
                        help="Heatmap resolution (default: fft_size/8)")
    args = parser.parse_args()

    heatmap_res = args.heatmap_res if args.heatmap_res else args.fft_size // 8

    cfg = TrainConfig(
        fft_size=args.fft_size, kmax=args.kmax, snr_min_db=args.snr_min, snr_max_db=args.snr_max,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, num_workers=args.workers,
        heatmap_res=heatmap_res,
    )

    device, _ = print_device_info()
    print(f"\nConfig: N={cfg.fft_size}, Kmax={cfg.kmax}, SNR=[{cfg.snr_min_db}, {cfg.snr_max_db}] dB")
    print(f"Heatmap resolution: {cfg.heatmap_res}")

    if args.plot_only:
        plot_spectrum_with_peaks(cfg, model=None, seed=args.plot_seed,
                                 save_path=f"{args.output_dir}/example_N{cfg.fft_size}.png",
                                 show=not args.no_show)
        return

    model = train(cfg, device, parallel=not args.sequential)
    evaluate(model, cfg, parallel=not args.sequential, threshold=args.threshold)
    export_to_onnx(model, cfg, args.output_dir)

    plot_spectrum_with_peaks(cfg, model=model, seed=args.plot_seed, threshold=args.threshold,
                             save_path=f"{args.output_dir}/detection_N{cfg.fft_size}.png",
                             show=not args.no_show)


if __name__ == "__main__":
    main()
