#!/usr/bin/env python3
"""Sequential spectrum data generator for history-based peak tracking (ex2/ex3).

Generates sequences of M consecutive spectra with slowly evolving peaks,
simulating real spectral monitoring scenarios. Reuses ex1's single-spectrum
generation utilities.
"""

import os
import sys
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1_python'))
from ex1_training import (
    TrainConfig, PeakParams, PEAK_GENERATORS,
    generate_realistic_noise, generate_baseline,
    compute_gaussian_equivalent_sigma, compute_energy_containment_widths,
    compute_kurtosis_adaptive, spectrum_to_normalized,
    generate_heatmap_target, generate_regression_target,
)


@dataclass
class SequenceConfig:
    history_depth: int = 16
    drift_sigma_min: float = 0.1  # min position drift (bins/slice)
    drift_sigma_max: float = 1.0  # max position drift (bins/slice)
    amp_drift_db: float = 1.0  # amplitude drift std (dB/slice)
    width_drift_rel: float = 0.03  # relative width drift std per slice
    peak_appear_prob: float = 0.03  # new peak per slice probability
    peak_disappear_prob: float = 0.02


def generate_evolving_sequence(
        cfg: TrainConfig, seq_cfg: SequenceConfig, rng: np.random.Generator
) -> tuple[np.ndarray, list[PeakParams]]:
    """Generate M consecutive spectra with slowly evolving peaks.

    Returns:
        spectra: (M, N) normalised spectra
        last_peaks: PeakParams list for the last slice (targets)
    """
    M = seq_cfg.history_depth
    N = cfg.fft_size
    bins = np.arange(N, dtype=np.float64)
    edge_margin = int(N * cfg.edge_margin)

    # initialise peak tracks for the full sequence
    num_peaks = rng.integers(0, cfg.kmax + 1)
    tracks = []
    for _ in range(num_peaks):
        snr_db = rng.uniform(cfg.snr_min_db, cfg.snr_max_db)
        if rng.random() < cfg.narrow_peak_prob:
            width = rng.uniform(1.0, 5.0)
            shape_idx = rng.choice([0, 3])  # gaussian or sinc2
        else:
            width = rng.uniform(cfg.peak_width_min_bins, cfg.peak_width_max_frac * N)
            shape_idx = int(rng.integers(0, len(PEAK_GENERATORS)))

        margin = max(edge_margin, int(3 * width))
        if margin >= N // 2:
            margin = edge_margin

        tracks.append({
            'center': float(rng.uniform(margin, N - margin)),
            'snr_db': float(snr_db),
            'width': float(width),
            'shape_idx': shape_idx,
            'drift_sigma': float(rng.uniform(seq_cfg.drift_sigma_min, seq_cfg.drift_sigma_max)),
        })

    spectra = np.empty((M, N), dtype=np.float32)

    for t in range(M):
        # evolve peaks after the first slice
        if t > 0:
            for tr in tracks:
                tr['center'] += rng.normal(0, tr['drift_sigma'])
                tr['center'] = float(np.clip(tr['center'], edge_margin, N - edge_margin))
                tr['snr_db'] += rng.normal(0, seq_cfg.amp_drift_db)
                tr['snr_db'] = float(np.clip(tr['snr_db'], cfg.snr_min_db - 3, cfg.snr_max_db + 3))
                tr['width'] *= float(np.exp(rng.normal(0, seq_cfg.width_drift_rel)))
                tr['width'] = float(np.clip(tr['width'], 1.0, cfg.peak_width_max_frac * N))

            # occasional new peak
            if len(tracks) < cfg.kmax and rng.random() < seq_cfg.peak_appear_prob:
                w = float(rng.uniform(cfg.peak_width_min_bins, cfg.peak_width_max_frac * N * 0.5))
                tracks.append({
                    'center': float(rng.uniform(edge_margin, N - edge_margin)),
                    'snr_db': float(rng.uniform(cfg.snr_min_db, cfg.snr_max_db)),
                    'width': w,
                    'shape_idx': int(rng.integers(0, len(PEAK_GENERATORS))),
                    'drift_sigma': float(rng.uniform(seq_cfg.drift_sigma_min, seq_cfg.drift_sigma_max)),
                })

            # occasional disappearance
            tracks = [tr for tr in tracks if rng.random() > seq_cfg.peak_disappear_prob]

        # render this slice
        baseline = generate_baseline(N, rng, cfg.baseline_slope_max)
        noise = generate_realistic_noise(N, rng)
        spectrum = baseline + noise

        slice_peaks = []
        for tr in tracks:
            amplitude = 1.0 * (10 ** (tr['snr_db'] / 20))
            _, shape_fn, param_sampler = PEAK_GENERATORS[tr['shape_idx']]
            shape_params = param_sampler(rng, max(tr['width'], 1.0))
            peak_shape = shape_fn(bins, tr['center'], **shape_params)

            scaled = amplitude * peak_shape
            spectrum += scaled

            if t == M - 1:
                sigma_eq = compute_gaussian_equivalent_sigma(peak_shape, int(tr['center']))
                w68, w96, w99 = compute_energy_containment_widths(peak_shape, int(tr['center']))
                kurt, excess = compute_kurtosis_adaptive(scaled, int(tr['center']), sigma_eq)
                slice_peaks.append(PeakParams(
                    center=tr['center'], amplitude=amplitude, sigma_equiv=sigma_eq,
                    w68=w68, w96=w96, w99=w99, kurtosis=kurt, excess_kurtosis=excess,
                    shape_name=PEAK_GENERATORS[tr['shape_idx']][0],
                ))

        spectra[t] = spectrum_to_normalized(spectrum)

        if t == M - 1:
            last_peaks = sorted(slice_peaks, key=lambda p: p.amplitude, reverse=True)

    if not tracks:
        last_peaks = []

    return spectra, last_peaks


# -- parallel dataset generation ------------------------------------------------

def _generate_sequence_batch(args: tuple) -> tuple:
    start_idx, count, cfg_dict, seq_cfg_dict, seed = args
    cfg = TrainConfig(**cfg_dict)
    seq_cfg = SequenceConfig(**seq_cfg_dict)
    rng = np.random.default_rng(seed)
    M = seq_cfg.history_depth
    N = cfg.fft_size
    R = cfg.n_regression_channels

    X = np.empty((count, M, N), dtype=np.float32)
    Y_hm = np.empty((count, N), dtype=np.float32)
    Y_reg = np.empty((count, N, R), dtype=np.float32)
    Y_mask = np.empty((count, N), dtype=np.float32)

    for i in range(count):
        spectra, last_peaks = generate_evolving_sequence(cfg, seq_cfg, rng)
        X[i] = spectra
        Y_hm[i] = generate_heatmap_target(last_peaks, cfg)
        reg, mask = generate_regression_target(last_peaks, cfg)
        Y_reg[i] = reg
        Y_mask[i] = mask

    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    Y_hm = np.nan_to_num(Y_hm, nan=0.0)
    Y_reg = np.nan_to_num(Y_reg, nan=0.0)
    Y_mask = np.nan_to_num(Y_mask, nan=0.0)

    return start_idx, X, Y_hm, Y_reg, Y_mask


def generate_sequence_dataset(
        cfg: TrainConfig, seq_cfg: SequenceConfig,
        num_samples: int, seed: int, desc: str = "Generating", parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = seq_cfg.history_depth
    N = cfg.fft_size
    R = cfg.n_regression_channels

    X = np.empty((num_samples, M, N), dtype=np.float32)
    Y_hm = np.empty((num_samples, N), dtype=np.float32)
    Y_reg = np.empty((num_samples, N, R), dtype=np.float32)
    Y_mask = np.empty((num_samples, N), dtype=np.float32)

    cfg_dict = {k: getattr(cfg, k) for k in [
        'fft_size', 'kmax', 'snr_min_db', 'snr_max_db', 'peak_width_min_bins',
        'peak_width_max_frac', 'baseline_slope_max', 'edge_margin',
        'narrow_peak_prob', 'n_regression_channels', 'heatmap_sigma_min', 'heatmap_sigma_scale',
    ]}
    seq_dict = {k: getattr(seq_cfg, k) for k in [
        'history_depth', 'drift_sigma_min', 'drift_sigma_max', 'amp_drift_db',
        'width_drift_rel', 'peak_appear_prob', 'peak_disappear_prob',
    ]}

    if parallel and cfg.num_workers > 1:
        num_workers = min(cfg.num_workers, num_samples)
        chunk = num_samples // num_workers
        remainder = num_samples % num_workers

        tasks = []
        start = 0
        for i in range(num_workers):
            count = chunk + (1 if i < remainder else 0)
            if count > 0:
                tasks.append((start, count, cfg_dict, seq_dict, seed + i * 10000))
                start += count

        print(f"{desc} (n={num_samples}, M={M}, workers={num_workers})...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_sequence_batch, t): t[0] for t in tasks}
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
        print(f"{desc} (n={num_samples}, M={M}, sequential)...")
        rng = np.random.default_rng(seed)
        for i in range(num_samples):
            spectra, last_peaks = generate_evolving_sequence(cfg, seq_cfg, rng)
            X[i] = spectra
            Y_hm[i] = generate_heatmap_target(last_peaks, cfg)
            Y_reg[i], Y_mask[i] = generate_regression_target(last_peaks, cfg)
            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  Progress: {100 * (i + 1) // num_samples:3d}%")

    return X, Y_hm, Y_reg, Y_mask
