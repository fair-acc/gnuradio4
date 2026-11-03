#!/usr/bin/env python3
"""Build the identity model (y = x, N=64) in TensorFlow/Keras, convert it to ONNX via
tf2onnx, stamp the GR4 custom metadata, and emit both .onnx and .ort.

Companion to the hand-built-graph reference blocks/onnx/models/identity_N64.onnx
(src/ex0_python/ex0_identity_model.py): same shape, metadata and semantics, but built
from the API a Keras/TensorFlow user actually trains in. Writes identity_N64_tf.onnx /
.ort so the tracked CI fixtures stay untouched.

Usage: python ex00_identity_gen.py [output_dir]
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
):
    os.environ.setdefault(_v, "6")

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow import keras

N = 64
GENERATOR_CMD = "python blocks/onnx/src/ex00_identity_gen.py"


def provenance() -> dict[str, str]:
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return {
        "generator_cmd": GENERATOR_CMD,
        "git_rev": rev.stdout.strip() if rev.returncode == 0 else "unknown",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def build_keras_model() -> keras.Model:
    inp = keras.Input(shape=(1, N), dtype="float32", name="input")
    out = keras.layers.Identity(name="output")(inp)
    return keras.Model(inputs=inp, outputs=out, name="identity_tf")


def convert_to_onnx(model: keras.Model) -> tuple[onnx.ModelProto, dict[str, str]]:
    # keras.Input(shape=...) leaves the batch dimension dynamic; pinning it to 1 here
    # keeps the exported graph's input shape fully concrete, so OnnxSession::bindIo
    # skips its symbolic-shape probe inference at load
    spec = (tf.TensorSpec((1, 1, N), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17)

    # tf2onnx always imports the ai.onnx.ml domain even when unused; drop it to match
    # the plain single-opset graph OnnxSession expects from a hand-built model
    used_domains = {node.domain for node in onnx_model.graph.node}
    kept = [
        o for o in onnx_model.opset_import if o.domain == "" or o.domain in used_domains
    ]
    del onnx_model.opset_import[:]
    onnx_model.opset_import.extend(kept)

    metadata = {
        "input_size": str(N),
        "normalise_mode": "None",
        "architecture": "identity-tf",
        **provenance(),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value

    onnx.checker.check_model(onnx_model)
    return onnx_model, metadata


def convert_to_ort(onnx_path: Path, ort_path: Path) -> None:
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(ort_path)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 6
    ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def self_test(onnx_path: Path, ort_path: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 6
    session = ort.InferenceSession(
        str(onnx_path), session_options, providers=["CPUExecutionProvider"]
    )
    x = np.linspace(-3.0, 3.0, N, dtype=np.float32).reshape(1, 1, N)

    (y,) = session.run(None, {"input": x})
    max_error = float(np.max(np.abs(y - x)))
    assert max_error < 1e-7, f"identity: max error {max_error:.2e}"

    ort_session = ort.InferenceSession(
        str(ort_path), session_options, providers=["CPUExecutionProvider"]
    )
    (y_ort,) = ort_session.run(None, {"input": x})
    assert np.allclose(y_ort, y), ".ort output diverges from .onnx"

    print(f"self-test OK: y = x (max error {max_error:.2e}); .onnx and .ort agree")


def main() -> int:
    out_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "models"
    )
    onnx_path = out_dir / "identity_N64_tf.onnx"
    ort_path = out_dir / "identity_N64_tf.ort"

    keras_model = build_keras_model()
    model, metadata = convert_to_onnx(keras_model)
    onnx.save(model, str(onnx_path))
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")
    print("stamped metadata:")
    for key, value in metadata.items():
        print(f"  {key} = {value}")

    convert_to_ort(onnx_path, ort_path)
    print(f"wrote {ort_path} ({ort_path.stat().st_size} bytes)")

    self_test(onnx_path, ort_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
