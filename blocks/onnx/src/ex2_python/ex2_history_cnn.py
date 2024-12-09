#!/usr/bin/env python3
"""History-based peak tracker using multi-channel 1D CNN (ex2).

Extends ex1's U-Net peak detector from [1,1,N] single-slice input to [1,M,N]
multi-slice input, enabling temporal continuity for improved low-SNR detection.

Input shape:  [batch, M, N]  (M history slices of N frequency bins)
Output shape: [batch, N*(R+1)]  (heatmap + regression for last slice)

The key architectural change from ex1: the first Conv1D layer receives M input
channels instead of 1.  The model learns which history slices are most informative
through channel weights — recent slices naturally receive higher weight.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.modules["brotli"] = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    import keras

    KERAS_3 = int(keras.__version__.split('.')[0]) >= 3
except ImportError:
    from tensorflow import keras

    KERAS_3 = False

# ex1 utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1_python'))
from ex1_training import (
    TrainConfig, focal_loss, masked_regression_loss,
    extract_peaks_from_heatmap, print_device_info,
)

# ex2 sequence generator
from ex2_generate_sequences import SequenceConfig, generate_sequence_dataset


# ============================================================================
# Model: ex1 U-Net with M input channels
# ============================================================================

def build_history_model(cfg: TrainConfig, M: int) -> keras.Model:
    n = cfg.fft_size
    R = cfg.n_regression_channels

    depth = 3
    filters = [32, 64, 128, 256]
    drop_rate = 0.15

    # M channels (history slices) instead of ex1's single channel
    inp = keras.Input(shape=(n, M), name="spectrum_history")

    def conv_block(x, f, kernel=5, dropout=0.0):
        x = keras.layers.Conv1D(f, kernel, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if dropout > 0:
            x = keras.layers.SpatialDropout1D(dropout)(x)
        x = keras.layers.Conv1D(f, kernel, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    skips = []
    x = inp
    for i in range(depth):
        x = conv_block(x, filters[i], kernel=7 if i == 0 else 5, dropout=drop_rate)
        skips.append(x)
        x = keras.layers.MaxPool1D(2)(x)

    x = conv_block(x, filters[depth], 3, dropout=drop_rate)
    x = keras.layers.Conv1D(filters[depth], 3, dilation_rate=2, padding="same", activation="relu")(x)
    x = keras.layers.SpatialDropout1D(drop_rate)(x)
    x = keras.layers.Conv1D(filters[depth], 3, dilation_rate=4, padding="same", activation="relu")(x)

    for i in range(depth - 1, -1, -1):
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Concatenate()([x, skips[i]])
        x = conv_block(x, filters[i], kernel=5, dropout=drop_rate if i > 0 else 0.0)

    head = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    heatmap = keras.layers.Conv1D(1, 1, activation="sigmoid", name="heatmap_raw")(head)
    heatmap = keras.layers.Reshape((n,), name="heatmap")(heatmap)

    regression = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(head)
    regression = keras.layers.Conv1D(R, 1, activation="linear", name="regression")(regression)

    model = keras.Model(inputs=inp, outputs=[heatmap, regression],
                        name=f"peak_detector_history_M{M}")
    return model


# ============================================================================
# Training
# ============================================================================

def train(cfg: TrainConfig, seq_cfg: SequenceConfig, device: str,
          parallel: bool = True) -> keras.Model:
    M = seq_cfg.history_depth

    X_train, Y_hm_train, Y_reg_train, Y_mask_train = generate_sequence_dataset(
        cfg, seq_cfg, cfg.train_samples, cfg.seed, "Training data", parallel)
    X_val, Y_hm_val, Y_reg_val, Y_mask_val = generate_sequence_dataset(
        cfg, seq_cfg, cfg.val_samples, cfg.seed + 1000, "Validation data", parallel)

    # transpose (samples, M, N) → (samples, N, M) for Conv1D channel convention
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))

    N = cfg.fft_size
    R = cfg.n_regression_channels
    Y_train = np.concatenate([Y_hm_train, Y_reg_train.reshape(-1, N * R), Y_mask_train], axis=1)
    Y_val = np.concatenate([Y_hm_val, Y_reg_val.reshape(-1, N * R), Y_mask_val], axis=1)

    print(f"\nData: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  History depth M={M}, FFT size N={N}")

    with tf.device(device):
        model = build_history_model(cfg, M)

        optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0)
        focal = focal_loss(gamma=2.0, alpha=0.85)

        @tf.function
        def train_step(x, y):
            hm_true = y[:, :N]
            reg_true = tf.reshape(y[:, N:N + N * R], (-1, N, R))
            mask = y[:, N + N * R:]

            with tf.GradientTape() as tape:
                hm_pred, reg_pred = model(x, training=True)
                loss_hm = focal(hm_true, hm_pred)
                loss_reg = masked_regression_loss(reg_true, reg_pred, mask)
                loss = cfg.lambda_heatmap * loss_hm + cfg.lambda_reg * loss_reg

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss, loss_hm, loss_reg

        @tf.function
        def val_step(x, y):
            hm_true = y[:, :N]
            reg_true = tf.reshape(y[:, N:N + N * R], (-1, N, R))
            mask = y[:, N + N * R:]

            hm_pred, reg_pred = model(x, training=False)
            loss_hm = focal(hm_true, hm_pred)
            loss_reg = masked_regression_loss(reg_true, reg_pred, mask)
            return cfg.lambda_heatmap * loss_hm + cfg.lambda_reg * loss_reg

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(cfg.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(cfg.batch_size)

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_weights = model.get_weights()
        val_loss_ema = None
        ema_alpha = 0.3

        model.summary()

        for epoch in range(cfg.epochs):
            if epoch < 3:
                new_lr = cfg.learning_rate * (epoch + 1) / 3
            else:
                progress = (epoch - 3) / max(cfg.epochs - 3, 1)
                new_lr = cfg.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
                new_lr = max(new_lr, 1e-6)
            optimizer.learning_rate.assign(new_lr)

            train_losses = []
            for x_batch, y_batch in train_ds:
                loss, loss_hm, loss_reg = train_step(x_batch, y_batch)
                v = loss.numpy()
                if np.isfinite(v):
                    train_losses.append(v)

            val_losses = []
            for x_batch, y_batch in val_ds:
                v = val_step(x_batch, y_batch).numpy()
                if np.isfinite(v):
                    val_losses.append(v)

            train_loss = np.mean(train_losses) if train_losses else float('nan')
            val_loss = np.mean(val_losses) if val_losses else float('nan')

            if val_loss_ema is None:
                val_loss_ema = val_loss
            else:
                val_loss_ema = ema_alpha * val_loss + (1 - ema_alpha) * val_loss_ema

            print(f"Epoch {epoch + 1:3d}/{cfg.epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f} (ema={val_loss_ema:.4f}), "
                  f"lr={new_lr:.2e}")

            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                patience_counter += 1
                if patience_counter >= 5:
                    print("Too many NaN epochs, stopping.")
                    break
                continue

            if val_loss_ema < best_val_loss - 1e-4:
                best_val_loss = val_loss_ema
                patience_counter = 0
                best_weights = model.get_weights()
            elif epoch >= cfg.min_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        model.set_weights(best_weights)

    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: keras.Model, cfg: TrainConfig, seq_cfg: SequenceConfig,
             parallel: bool = True, threshold: float = 0.5):
    M = seq_cfg.history_depth

    X_test, Y_hm_test, Y_reg_test, Y_mask_test = generate_sequence_dataset(
        cfg, seq_cfg, cfg.test_samples, cfg.seed + 2000, "Test data", parallel)

    print(f"\nEvaluating with threshold={threshold}...")
    n_eval = min(500, cfg.test_samples)

    center_errors = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    match_tolerance = max(10, cfg.fft_size // 100)

    for i in range(n_eval):
        gt_peaks = []
        for j in range(cfg.fft_size):
            if Y_mask_test[i, j] > 0.5:
                center = j + Y_reg_test[i, j, 0] * cfg.fft_size
                gt_peaks.append(center)

        X = np.transpose(X_test[i:i + 1], (0, 2, 1))  # (1, N, M)
        heatmap, regression = model.predict(X, verbose=0)
        pred_peaks = extract_peaks_from_heatmap(heatmap[0], regression[0], cfg, threshold)
        pred_centers = [p['center'] for p in pred_peaks]

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

    print(f"\nResults on {n_eval} samples (M={M}):")
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
# ONNX Export
# ============================================================================

def export_to_onnx(model: keras.Model, cfg: TrainConfig, seq_cfg: SequenceConfig,
                   output_dir: str = "."):
    import shutil

    M = seq_cfg.history_depth
    N = cfg.fft_size
    R = cfg.n_regression_channels
    base_name = f"history_cnn_M{M}_N{N}"
    saved_model_dir = f"{output_dir}/{base_name}_saved"
    onnx_path = f"{output_dir}/{base_name}.onnx"

    # export wrapper: input [batch, M, N] → transpose → model → concat output
    inp = keras.Input(shape=(M, N), dtype="float32", name="spectrum_history")
    if KERAS_3:
        x = keras.ops.transpose(inp, [0, 2, 1])  # → [batch, N, M]
    else:
        x = tf.transpose(inp, [0, 2, 1])
    heatmap, regression = model(x, training=False)
    if KERAS_3:
        reg_flat = keras.ops.reshape(regression, (-1, N * R))
        output = keras.ops.concatenate([heatmap, reg_flat], axis=-1)
    else:
        reg_flat = tf.reshape(regression, (-1, N * R))
        output = tf.concat([heatmap, reg_flat], axis=-1)
    export_model = keras.Model(inputs=inp, outputs=output, name=f"{base_name}_export")

    export_model.export(saved_model_dir)
    print(f"SavedModel: {saved_model_dir}")

    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        import tf2onnx

        loaded = tf.saved_model.load(saved_model_dir)
        concrete = loaded.signatures["serving_default"]
        frozen = convert_variables_to_constants_v2(concrete)
        graph_def = frozen.graph.as_graph_def()

        inputs = [node.name + ":0" for node in graph_def.node if node.op == "Placeholder"]
        outputs = [frozen.outputs[0].name]

        tf2onnx.convert.from_graph_def(
            graph_def, input_names=inputs, output_names=outputs,
            opset=17, output_path=onnx_path,
        )

        _add_onnx_metadata(onnx_path, cfg, seq_cfg)
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print(f"SavedModel retained: {saved_model_dir}")


def _add_onnx_metadata(onnx_path: str, cfg: TrainConfig, seq_cfg: SequenceConfig):
    try:
        import onnx
        model = onnx.load(onnx_path)
        metadata = {
            "input_size": str(cfg.fft_size),
            "history_depth": str(seq_cfg.history_depth),
            "n_regression_channels": str(cfg.n_regression_channels),
            "kmax": str(cfg.kmax),
            "snr_min_db": str(cfg.snr_min_db),
            "snr_max_db": str(cfg.snr_max_db),
            "normalise_mode": "LogMAD",
            "architecture": "history_cnn_unet",
        }
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.save(model, onnx_path)
    except ImportError:
        pass


def validate_onnx(onnx_path: str, cfg: TrainConfig, seq_cfg: SequenceConfig, n_samples: int = 10):
    import onnxruntime as ort
    from ex2_generate_sequences import generate_evolving_sequence

    M = seq_cfg.history_depth
    N = cfg.fft_size
    R = cfg.n_regression_channels

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    print(f"ONNX model: {onnx_path}")
    print(f"  Input:  {input_name} {input_shape}")
    print(f"  Output: {output_shape}")

    meta = session.get_modelmeta()
    if meta.custom_metadata_map:
        print(f"  Metadata: {dict(meta.custom_metadata_map)}")

    rng = np.random.default_rng(cfg.seed)
    for i in range(n_samples):
        spectra, _ = generate_evolving_sequence(cfg, seq_cfg, rng)
        x = spectra[np.newaxis, :, :].astype(np.float32)  # [1, M, N]

        ort_output = session.run(None, {input_name: x})[0]
        heatmap = ort_output[0, :N]
        regression = ort_output[0, N:].reshape(N, R)

        print(f"  Sample {i}: heatmap max={heatmap.max():.4f}, "
              f"regression max={np.abs(regression).max():.4f}")

    print(f"\nValidation passed")
    print(f"  Expected output shape: [1, {N + N * R}] = [1, {N * (1 + R)}]")
    print(f"  Actual output shape:   {list(ort_output.shape)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="History peak tracker — 1D CNN (ex2)")
    parser.add_argument("--fft-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--history-depth", type=int, default=16, help="M: number of history slices")
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--snr-min", type=float, default=3.0, help="lower SNR for history benefit")
    parser.add_argument("--snr-max", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--validate-onnx", type=str, default=None,
                        help="Path to .onnx file: run ORT validation and exit")
    args = parser.parse_args()

    cfg = TrainConfig(
        fft_size=args.fft_size, kmax=args.kmax,
        snr_min_db=args.snr_min, snr_max_db=args.snr_max,
        epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, num_workers=args.workers,
        train_samples=args.train_samples,
        val_samples=max(1000, args.train_samples // 10),
        test_samples=max(1000, args.train_samples // 10),
    )
    seq_cfg = SequenceConfig(history_depth=args.history_depth)

    device, _ = print_device_info()
    print(f"\nConfig: N={cfg.fft_size}, M={seq_cfg.history_depth}, "
          f"Kmax={cfg.kmax}, SNR=[{cfg.snr_min_db}, {cfg.snr_max_db}] dB")

    if args.validate_onnx:
        validate_onnx(args.validate_onnx, cfg, seq_cfg)
        return

    model = train(cfg, seq_cfg, device, parallel=not args.sequential)
    evaluate(model, cfg, seq_cfg, parallel=not args.sequential, threshold=args.threshold)
    export_to_onnx(model, cfg, seq_cfg, args.output_dir)


if __name__ == "__main__":
    main()
