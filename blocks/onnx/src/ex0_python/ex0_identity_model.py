#!/usr/bin/env python3
"""Generate an identity ONNX model for CI and streaming mode integration tests.

The model computes y = x (identity function) with input/output shape [1, 1, 64].
It uses only the ONNX helper API (no training framework needed).

Output files (committed via Git LFS):
    blocks/onnx/models/identity_N64.onnx
    blocks/onnx/models/identity_N64.ort

Usage:
    python3 ex0_identity_model.py [--output-dir ../../models]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def create_identity_model(n: int = 64, m: int = 1) -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, m, n])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, m, n])

    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

    graph = helper.make_graph([identity_node], "identity_graph", [input_tensor], [output_tensor])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # custom metadata for OnnxSession auto-detection
    metadata = [("input_size", str(n)), ("normalise_mode", "None"), ("architecture", "identity")]
    if m > 1:
        metadata.append(("history_depth", str(m)))
    for key, value in metadata:
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.checker.check_model(model)
    return model


def convert_to_ort(onnx_path: Path, ort_path: Path) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.optimized_model_filepath = str(ort_path)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort.InferenceSession(str(onnx_path), so)


def validate(onnx_path: Path, n: int = 64, m: int = 1) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    test_input = np.arange(m * n, dtype=np.float32).reshape(1, m, n)
    result = session.run(None, {"input": test_input})[0]

    if not np.allclose(test_input, result, atol=1e-7):
        print("FAIL: identity model output does not match input", file=sys.stderr)
        sys.exit(1)

    print(f"validation passed: max error = {np.max(np.abs(test_input - result)):.2e}")


def main() -> None:
    default_output = Path(__file__).resolve().parent.parent.parent / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--m", type=int, default=1, help="history depth (channels)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = create_identity_model(args.n, args.m)

    suffix = f"_M{args.m}" if args.m > 1 else ""
    onnx_path = args.output_dir / f"identity_N{args.n}{suffix}.onnx"
    ort_path = args.output_dir / f"identity_N{args.n}{suffix}.ort"

    onnx.save(model, str(onnx_path))
    print(f"saved {onnx_path} ({onnx_path.stat().st_size} bytes)")

    validate(onnx_path, args.n, args.m)

    convert_to_ort(onnx_path, ort_path)
    print(f"saved {ort_path} ({ort_path.stat().st_size} bytes)")

    # summary
    print(f"\nmodel: identity y=x, shape [1, {args.m}, {args.n}]")
    meta_str = f"fft_size={args.n}, normalise_mode=None, architecture=identity"
    if args.m > 1:
        meta_str += f", history_depth={args.m}"
    print(f"metadata: {meta_str}")
    print(f"ops: {len(model.graph.node)} (Identity)")


if __name__ == "__main__":
    main()
