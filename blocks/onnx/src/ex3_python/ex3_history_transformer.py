#!/usr/bin/env python3
"""History-based peak tracker using 1D Vision Transformer (ex3).

Same task as ex2 (M×N history → peak detection on last slice) but using a
Transformer architecture with self-attention for temporal reasoning.

Input shape:  [batch, M, N]  (M history slices of N frequency bins)
Output shape: [batch, N*(R+1)]  (heatmap + regression for last slice)

Architecture:
  Input [1, M, N]
    → patch embedding: Conv1d per-slice (N → n_patches × d_model)
    → flatten: [M * n_patches, d_model] tokens
    → positional encoding (learned)
    → N_layers × TransformerEncoder
    → extract last-slice tokens → Conv1d heads → heatmap + regression
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2_python'))
from ex2_generate_sequences import SequenceConfig, generate_sequence_dataset


# ============================================================================
# Transformer configuration
# ============================================================================

@dataclass
class TransformerConfig:
    patch_size: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    warmup_fraction: float = 0.1  # LR warmup as fraction of total epochs


# ============================================================================
# Model: 1D Vision Transformer
# ============================================================================

class PatchEmbedding(keras.layers.Layer):
    """Conv1D-based patch embedding applied independently per history slice."""

    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = keras.layers.Conv1D(d_model, patch_size, strides=patch_size,
                                        padding="valid")

    def call(self, x):
        # x: [batch, N, M] → process each channel independently?
        # Actually: reshape to process all M slices through the same Conv1D
        # x: [batch, N, M] → [batch*M, N, 1] → Conv1D → [batch*M, n_patches, d_model]
        shape = tf.shape(x)
        batch, N, M = shape[0], shape[1], shape[2]

        # transpose to [batch, M, N] then reshape to [batch*M, N, 1]
        x_t = tf.transpose(x, [0, 2, 1])  # [batch, M, N]
        x_flat = tf.reshape(x_t, [batch * M, N, 1])  # [batch*M, N, 1]
        patches = self.proj(x_flat)  # [batch*M, n_patches, d_model]

        n_patches = tf.shape(patches)[1]
        # reshape to [batch, M * n_patches, d_model]
        return tf.reshape(patches, [batch, M * n_patches, self.d_model])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dim_ff, activation="gelu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout),
        ])
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # pre-norm transformer block
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, training=training)
        x = x + self.dropout(attn_out, training=training)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed, training=training)
        return x + ffn_out


def build_transformer_model(cfg: TrainConfig, M: int, t_cfg: TransformerConfig) -> keras.Model:
    N = cfg.fft_size
    R = cfg.n_regression_channels
    n_patches = N // t_cfg.patch_size
    total_tokens = M * n_patches

    inp = keras.Input(shape=(N, M), name="spectrum_history")

    # patch embedding → [batch, M * n_patches, d_model]
    x = PatchEmbedding(t_cfg.patch_size, t_cfg.d_model)(inp)

    # learned positional encoding
    pos_embed = keras.layers.Embedding(total_tokens, t_cfg.d_model)
    positions = tf.range(total_tokens)
    x = x + pos_embed(positions)
    x = keras.layers.Dropout(t_cfg.dropout)(x)

    # transformer encoder blocks
    for _ in range(t_cfg.n_layers):
        x = TransformerEncoderBlock(
            t_cfg.d_model, t_cfg.n_heads, t_cfg.dim_feedforward, t_cfg.dropout)(x)

    x = keras.layers.LayerNormalization()(x)

    # extract last-slice tokens: indices [(M-1)*n_patches .. M*n_patches-1]
    last_start = (M - 1) * n_patches
    x = x[:, last_start:last_start + n_patches, :]  # [batch, n_patches, d_model]

    # upsample back to N bins via transposed conv / reshape + conv
    # [batch, n_patches, d_model] → [batch, N, channels]
    x = keras.layers.Dense(t_cfg.patch_size * 64)(x)  # [batch, n_patches, patch_size*64]
    x = keras.layers.Reshape((N, 64))(x)  # [batch, N, 64]

    head = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    heatmap = keras.layers.Conv1D(1, 1, activation="sigmoid", name="heatmap_raw")(head)
    heatmap = keras.layers.Reshape((N,), name="heatmap")(heatmap)

    regression = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(head)
    regression = keras.layers.Conv1D(R, 1, activation="linear", name="regression")(regression)

    model = keras.Model(inputs=inp, outputs=[heatmap, regression],
                        name=f"peak_detector_transformer_M{M}")
    return model


# ============================================================================
# Training (with warmup + cosine decay, AdamW)
# ============================================================================

def train(cfg: TrainConfig, seq_cfg: SequenceConfig, t_cfg: TransformerConfig,
          device: str, parallel: bool = True) -> keras.Model:
    M = seq_cfg.history_depth

    X_train, Y_hm_train, Y_reg_train, Y_mask_train = generate_sequence_dataset(
        cfg, seq_cfg, cfg.train_samples, cfg.seed, "Training data", parallel)
    X_val, Y_hm_val, Y_reg_val, Y_mask_val = generate_sequence_dataset(
        cfg, seq_cfg, cfg.val_samples, cfg.seed + 1000, "Validation data", parallel)

    X_train = np.transpose(X_train, (0, 2, 1))  # (samples, N, M)
    X_val = np.transpose(X_val, (0, 2, 1))

    N = cfg.fft_size
    R = cfg.n_regression_channels
    Y_train = np.concatenate([Y_hm_train, Y_reg_train.reshape(-1, N * R), Y_mask_train], axis=1)
    Y_val = np.concatenate([Y_hm_val, Y_reg_val.reshape(-1, N * R), Y_mask_val], axis=1)

    print(f"\nData: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  History depth M={M}, FFT size N={N}")
    print(f"  Transformer: d_model={t_cfg.d_model}, heads={t_cfg.n_heads}, "
          f"layers={t_cfg.n_layers}, patch={t_cfg.patch_size}")

    with tf.device(device):
        model = build_transformer_model(cfg, M, t_cfg)

        # AdamW with weight decay
        try:
            optimizer = keras.optimizers.AdamW(
                learning_rate=cfg.learning_rate, weight_decay=1e-4, clipnorm=1.0)
        except AttributeError:
            optimizer = keras.optimizers.Adam(
                learning_rate=cfg.learning_rate, clipnorm=1.0)

        focal = focal_loss(gamma=2.0, alpha=0.85)

        warmup_epochs = max(1, int(t_cfg.warmup_fraction * cfg.epochs))

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
        patience = 30  # more patience for transformers
        patience_counter = 0
        best_weights = model.get_weights()
        val_loss_ema = None
        ema_alpha = 0.3

        model.summary()

        for epoch in range(cfg.epochs):
            # warmup + cosine decay schedule
            if epoch < warmup_epochs:
                new_lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
                new_lr = cfg.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
                new_lr = max(new_lr, 1e-6)
            optimizer.learning_rate.assign(new_lr)

            train_losses = []
            for x_batch, y_batch in train_ds:
                loss, _, _ = train_step(x_batch, y_batch)
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

        X = np.transpose(X_test[i:i + 1], (0, 2, 1))
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

    print(f"\nResults on {n_eval} samples (M={M}, Transformer):")
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
                   t_cfg: TransformerConfig, output_dir: str = "."):
    import shutil

    M = seq_cfg.history_depth
    N = cfg.fft_size
    R = cfg.n_regression_channels
    base_name = f"history_transformer_M{M}_N{N}"
    saved_model_dir = f"{output_dir}/{base_name}_saved"
    onnx_path = f"{output_dir}/{base_name}.onnx"

    inp = keras.Input(shape=(M, N), dtype="float32", name="spectrum_history")
    if KERAS_3:
        x = keras.ops.transpose(inp, [0, 2, 1])
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

        _add_onnx_metadata(onnx_path, cfg, seq_cfg, t_cfg)
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print(f"SavedModel retained: {saved_model_dir}")


def _add_onnx_metadata(onnx_path: str, cfg: TrainConfig, seq_cfg: SequenceConfig,
                       t_cfg: TransformerConfig):
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
            "architecture": "history_transformer_vit",
            "patch_size": str(t_cfg.patch_size),
            "d_model": str(t_cfg.d_model),
            "n_heads": str(t_cfg.n_heads),
            "n_layers": str(t_cfg.n_layers),
        }
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.save(model, onnx_path)
    except ImportError:
        pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="History peak tracker — Transformer (ex3)")
    parser.add_argument("--fft-size", type=int, default=1024, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--history-depth", type=int, default=16)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--snr-min", type=float, default=3.0)
    parser.add_argument("--snr-max", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16, help="smaller for transformer memory")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.4)
    # transformer hyperparameters
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--warmup", type=float, default=0.1, help="LR warmup fraction")
    args = parser.parse_args()

    cfg = TrainConfig(
        fft_size=args.fft_size, kmax=args.kmax,
        snr_min_db=args.snr_min, snr_max_db=args.snr_max,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed, num_workers=args.workers,
        train_samples=args.train_samples,
        val_samples=max(1000, args.train_samples // 10),
        test_samples=max(1000, args.train_samples // 10),
    )
    seq_cfg = SequenceConfig(history_depth=args.history_depth)
    t_cfg = TransformerConfig(
        patch_size=args.patch_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dim_feedforward=args.dim_ff, warmup_fraction=args.warmup,
    )

    device, _ = print_device_info()
    print(f"\nConfig: N={cfg.fft_size}, M={seq_cfg.history_depth}, "
          f"Kmax={cfg.kmax}, SNR=[{cfg.snr_min_db}, {cfg.snr_max_db}] dB")
    print(f"Transformer: d={t_cfg.d_model}, heads={t_cfg.n_heads}, "
          f"layers={t_cfg.n_layers}, patch={t_cfg.patch_size}, ff={t_cfg.dim_feedforward}")

    model = train(cfg, seq_cfg, t_cfg, device, parallel=not args.sequential)
    evaluate(model, cfg, seq_cfg, parallel=not args.sequential, threshold=args.threshold)
    export_to_onnx(model, cfg, seq_cfg, t_cfg, args.output_dir)


if __name__ == "__main__":
    main()
