#!/usr/bin/env python3
"""Build the minimal ONNX round-trip example: y = a*x + b, elementwise.

Hand-built opset-17 graph (no training, no framework) matching the
OnnxInference contract: input [1, 1, N] -> output [1, N]. `a` and `b` are
overridable initializers (default 2.0 / 1.0) — a name present in both
graph.input and initializer, so ORT reports the model as needing only `x`
while `a`/`b` stay tunable via OnnxInference's config_in ports / model_overrides.
Writes both mvp_affine_N8.onnx (full metadata, dev builds) and .ort (stripped,
minimal/Emscripten builds).

Usage: python ex01_elementwise_gen.py [output_dir]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

N = 8
A_DEFAULT = 2.0
B_DEFAULT = 1.0
GENERATOR_CMD = "python blocks/onnx/src/ex01_elementwise_gen.py"


def const(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


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


def build_model() -> tuple[onnx.ModelProto, dict[str, str]]:
    initializers = [
        const("shape_1N", np.array([1, N], dtype=np.int64)),
        const("a", np.array([A_DEFAULT], dtype=np.float32)),
        const("b", np.array([B_DEFAULT], dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "shape_1N"], ["x2d"]),
        helper.make_node("Mul", ["x2d", "a"], ["scaled"]),
        helper.make_node("Add", ["scaled", "b"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gr_onnx_mvp_affine",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, N]),
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(N),
        "history_depth": "1",
        "normalise_mode": "None",
        "architecture": "mvp-affine",
        **provenance(),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model, metadata


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

    (y,) = session.run(None, {"x": x})
    expected = A_DEFAULT * x.reshape(1, N) + B_DEFAULT
    max_error = float(np.max(np.abs(y - expected)))
    assert max_error < 1e-6, f"default coefficients: max error {max_error:.2e}"

    a_override, b_override = np.array([3.0], dtype=np.float32), np.array(
        [0.0], dtype=np.float32
    )
    (y_override,) = session.run(None, {"x": x, "a": a_override, "b": b_override})
    expected_override = 3.0 * x.reshape(1, N) + 0.0
    override_error = float(np.max(np.abs(y_override - expected_override)))
    assert (
        override_error < 1e-6
    ), f"overridden coefficients: max error {override_error:.2e}"
    assert not np.allclose(y, y_override), "override had no effect on the output"

    ort_session = ort.InferenceSession(
        str(ort_path), session_options, providers=["CPUExecutionProvider"]
    )
    (y_ort,) = ort_session.run(None, {"x": x})
    (y_ort_override,) = ort_session.run(
        None, {"x": x, "a": a_override, "b": b_override}
    )
    assert np.allclose(y_ort, y), ".ort default output diverges from .onnx"
    assert np.allclose(
        y_ort_override, y_override
    ), ".ort override diverges from .onnx (initializer got baked in?)"

    print(
        f"self-test OK: y = {A_DEFAULT}*x + {B_DEFAULT} (max error {max_error:.2e}); "
        f"override a=3,b=0 verified live in both .onnx and .ort (max error {override_error:.2e})"
    )


def main() -> int:
    out_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "models"
    )
    onnx_path = out_dir / f"mvp_affine_N{N}.onnx"
    ort_path = out_dir / f"mvp_affine_N{N}.ort"

    model, metadata = build_model()
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
