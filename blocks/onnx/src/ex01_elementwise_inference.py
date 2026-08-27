#!/usr/bin/env python3
"""Load mvp_affine_N8.onnx and verify it against the closed-form y = a*x + b.

Prints the metadata and I/O shapes GR4's OnnxSession reads to configure a
block, then runs one inference and compares against the analytic reference —
the check to run before wiring a model into a flow graph and blaming the
GR4 block for a mismatch that is actually in the model.

Usage: python ex01_elementwise_inference.py [input_dir]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

N = 8
A_DEFAULT = 2.0
B_DEFAULT = 1.0


def print_model_info(session: ort.InferenceSession) -> None:
    print("declared metadata:")
    for key, value in session.get_modelmeta().custom_metadata_map.items():
        print(f"  {key} = {value}")
    print("I/O shapes:")
    for tensor in session.get_inputs():
        print(f"  in  {tensor.name:<6} {tensor.shape} {tensor.type}")
    for tensor in session.get_outputs():
        print(f"  out {tensor.name:<6} {tensor.shape} {tensor.type}")
    overridable = session.get_overridable_initializers()
    if overridable:
        print("config (model_overrides / config_in tunable, not required inputs):")
        for tensor in overridable:
            print(f"  cfg {tensor.name:<6} {tensor.shape} {tensor.type}")


def main() -> int:
    in_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "models"
    )
    model_path = in_dir / f"mvp_affine_N{N}.onnx"
    if not model_path.exists():
        print(
            f"model not found: {model_path} (run ex01_elementwise_gen.py first)",
            file=sys.stderr,
        )
        return 1

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 6
    session = ort.InferenceSession(
        str(model_path), session_options, providers=["CPUExecutionProvider"]
    )
    print(f"model: {model_path}")
    print_model_info(session)

    x = np.linspace(-3.0, 3.0, N, dtype=np.float32).reshape(1, 1, N)
    (y,) = session.run(None, {"x": x})
    expected = A_DEFAULT * x.reshape(1, N) + B_DEFAULT

    print(f"\n{'bin':>4} {'input':>9} {'expected':>9} {'measured':>9}")
    for i in range(N):
        print(f"{i:>4} {x[0, 0, i]:>9.4f} {expected[0, i]:>9.4f} {y[0, i]:>9.4f}")

    max_error = float(np.max(np.abs(y - expected)))
    print(f"\nmax |error| = {max_error:.2e}")
    if max_error > 1e-5:
        print("FAIL: model does not compute y = a*x + b", file=sys.stderr)
        return 1

    a_override, b_override = np.array([3.0], dtype=np.float32), np.array(
        [0.0], dtype=np.float32
    )
    (y_override,) = session.run(None, {"x": x, "a": a_override, "b": b_override})
    expected_override = float(a_override[0]) * x.reshape(1, N) + float(b_override[0])
    override_error = float(np.max(np.abs(y_override - expected_override)))
    print(
        f"override a=3, b=0 (as model_overrides would set them): max |error| = {override_error:.2e}"
    )
    if override_error > 1e-5:
        print("FAIL: overridable initializer did not take effect", file=sys.stderr)
        return 1

    print("OK — safe to wire into a GR4 flow graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
