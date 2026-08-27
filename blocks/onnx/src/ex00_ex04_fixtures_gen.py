#!/usr/bin/env python3
"""Generate the deterministic analytic ONNX test fixtures for the onnx qa suite.

Every fixture is a hand-built graph (no training, no random weights) so the tracked
repository can rebuild its own test models without TensorFlow or the untracked
ex*_python research trees. Two families:

peaks_fixture_N1024.onnx (target "peaks") implements
classical peak detection and emit the peaks-tensor contract consumed by
gr::blocks::onnx::OnnxPeakDetector:

    spectra[1, M, N] -> heatmap[1, N], peaks[1, K, 13], peak_rescore[1, K],
                        reconstruction[1, N], residual[1, N]

The temporal variant (M=16, metadata history_depth=16) slices the NEWEST frame
(row M-1) out of the [1, M, N] input and runs the identical detection graph on it,
so a peak present only in an older frame must not be reported — which is exactly
what the C++ history-buffer ordering test needs.

Per peak the graph computes: score (min-max normalised height), sub-bin centre
(parabolic interpolation), amplitude above the spectrum minimum, and a Gaussian
log-ratio width estimate sigma = h / sqrt(-2 ln r) with r = (y[i-h]+y[i+h])/(2 y[i])
(amplitudes above the spectrum minimum; exact for a Gaussian). Two stencils are
blended: h=2 for narrow peaks, h=8 when the h=2 ratio saturates (wide peaks).
`score_scale` is an overridable initializer (default 1.0) multiplying the emitted
heatmap and peak_rescore outputs (not the internal peak selection), used to
exercise the config-input/model-overrides path of OnnxInference.

The remaining targets are the small graphs behind the OnnxInference use-case
examples (ex00..ex04): identity (target "identity"), 1:1 elementwise affine
("affine"), N:1 decimating mean ("frame_mean"), sliding-window delta
("frame_delta"), and a multi-output mean+rms model ("mean_rms").

Usage: python ex00_ex04_fixtures_gen.py [output_dir] [--only TARGET [TARGET ...]]
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N = 1024  # spectrum bins (peaks fixture)
K = 8  # maximum peaks
M_TEMPORAL = 16  # history depth of the temporal peaks fixture

N_SMALL = 64  # spectrum bins (identity/affine/frame_mean/frame_delta/mean_rms)
M_IDENTITY = 4  # history depth of the identity_N64_M4 CI fixture
M_HISTORY = 16  # history depth of the peaks fixture covering the M>1 path
N_ALT = 512  # second spectrum size, so the model-swap test proves a genuine re-size
M_FRAME_MEAN = 4  # frames averaged by the frame_mean fixture
M_FRAME_DELTA = 8  # sliding-window depth of the frame_delta fixture
AFFINE_SCALE = 2.0
AFFINE_OFFSET = 1.0


def const(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def provenance(target: str) -> dict[str, str]:
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return {
        "generator_cmd": f"python blocks/onnx/src/ex00_ex04_fixtures_gen.py --only {target}",
        "git_rev": rev.stdout.strip() if rev.returncode == 0 else "unknown",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def build_peaks_model(m: int = 1, n: int = N) -> onnx.ModelProto:
    initializers = [
        const("shape_1N", np.array([1, n], dtype=np.int64)),
        const("k_topk", np.array([K], dtype=np.int64)),
        const("one_i", np.array(1, dtype=np.int64)),
        const("two_i", np.array(2, dtype=np.int64)),
        const("eight_i", np.array(8, dtype=np.int64)),
        const("zero_i", np.array(0, dtype=np.int64)),
        const("nmax_i", np.array(n - 1, dtype=np.int64)),
        const("half_f", np.array(0.5, dtype=np.float32)),
        const("two_f", np.array(2.0, dtype=np.float32)),
        const("eps_f", np.array(1e-9, dtype=np.float32)),
        const("neg_half_f", np.array(-0.5, dtype=np.float32)),
        const("neg_two_f", np.array(-2.0, dtype=np.float32)),
        const("h2_f", np.array(2.0, dtype=np.float32)),
        const("h8_f", np.array(8.0, dtype=np.float32)),
        const("r_lo_f", np.array(1e-3, dtype=np.float32)),
        const("r_hi_f", np.array(0.999, dtype=np.float32)),
        const("r_wide_f", np.array(0.9, dtype=np.float32)),
        const("axes_2", np.array([2], dtype=np.int64)),
        const("axes_1", np.array([1], dtype=np.int64)),
        const("positions", np.arange(n, dtype=np.float32).reshape(1, 1, n)),
        const("is_top1", np.eye(1, K, dtype=np.float32)),  # [1, K]: 1, 0, 0, ...
        const("zeros_K", np.zeros((1, K), dtype=np.float32)),
        const(
            "gate_K", np.full((1, K), 0.5, dtype=np.float32)
        ),  # in-graph keep gate for residual
        const("score_scale", np.array([1.0], dtype=np.float32)),  # overridable
        const("slice_a0", np.array([0], dtype=np.int64)),
        const("slice_a1", np.array([1], dtype=np.int64)),
        const("slice_n1", np.array([n - 1], dtype=np.int64)),
        const("slice_n", np.array([n], dtype=np.int64)),
    ]

    if m > 1:
        initializers += [
            const("slice_m1", np.array([m - 1], dtype=np.int64)),
            const("slice_m", np.array([m], dtype=np.int64)),
        ]
        # temporal input: detection runs on the NEWEST frame (row m-1)
        input_nodes = [
            helper.make_node(
                "Slice", ["spectra", "slice_m1", "slice_m", "axes_1"], ["newest"]
            ),
            helper.make_node("Reshape", ["newest", "shape_1N"], ["x"]),
        ]
    else:
        input_nodes = [helper.make_node("Reshape", ["spectra", "shape_1N"], ["x"])]

    nodes = [
        *input_nodes,
        # heatmap: min-max normalised spectrum
        helper.make_node("ReduceMin", ["x"], ["xmin"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["x"], ["xmax"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["x", "xmin"], ["x_shift"]),
        helper.make_node("Sub", ["xmax", "xmin"], ["xrange"]),
        helper.make_node("Add", ["xrange", "eps_f"], ["xrange_safe"]),
        helper.make_node("Div", ["x_shift", "xrange_safe"], ["heatmap_base"]),
        helper.make_node("Mul", ["heatmap_base", "score_scale"], ["heatmap"]),
        # local-maximum mask from edge-replicated neighbour shifts
        helper.make_node("Slice", ["x", "slice_a0", "slice_a1", "axes_1"], ["x_first"]),
        helper.make_node("Slice", ["x", "slice_a0", "slice_n1", "axes_1"], ["x_head"]),
        helper.make_node("Slice", ["x", "slice_a1", "slice_n", "axes_1"], ["x_tail"]),
        helper.make_node("Slice", ["x", "slice_n1", "slice_n", "axes_1"], ["x_last"]),
        helper.make_node("Concat", ["x_first", "x_head"], ["xl"], axis=1),
        helper.make_node("Concat", ["x_tail", "x_last"], ["xr"], axis=1),
        helper.make_node("Greater", ["x", "xl"], ["gt_left"]),
        helper.make_node("GreaterOrEqual", ["x", "xr"], ["ge_right"]),
        helper.make_node("And", ["gt_left", "ge_right"], ["is_max"]),
        helper.make_node("Cast", ["is_max"], ["is_max_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["heatmap_base", "is_max_f"], ["masked"]),
        # strongest K local maxima
        helper.make_node(
            "TopK", ["masked", "k_topk"], ["score", "idx"], axis=1, largest=1, sorted=1
        ),
        helper.make_node("Cast", ["idx"], ["idx_f"], to=TensorProto.FLOAT),
        # neighbour samples for parabolic interpolation and curvature width
        helper.make_node("Sub", ["idx", "one_i"], ["idx_m1_raw"]),
        helper.make_node("Add", ["idx", "one_i"], ["idx_p1_raw"]),
        helper.make_node("Clip", ["idx_m1_raw", "zero_i", "nmax_i"], ["idx_m1"]),
        helper.make_node("Clip", ["idx_p1_raw", "zero_i", "nmax_i"], ["idx_p1"]),
        helper.make_node("GatherElements", ["x", "idx"], ["yc"], axis=1),
        helper.make_node("GatherElements", ["x", "idx_m1"], ["yl"], axis=1),
        helper.make_node("GatherElements", ["x", "idx_p1"], ["yr"], axis=1),
        # parabolic sub-bin offset: (yl - yr) / (2*(yl + yr - 2*yc) - eps), clipped to +-0.5
        helper.make_node("Sub", ["yl", "yr"], ["dy"]),
        helper.make_node("Add", ["yl", "yr"], ["yl_yr"]),
        helper.make_node("Mul", ["yc", "two_f"], ["two_yc"]),
        helper.make_node("Sub", ["yl_yr", "two_yc"], ["d2y"]),  # negative at a peak
        helper.make_node("Mul", ["d2y", "two_f"], ["d2y2"]),
        helper.make_node("Sub", ["d2y2", "eps_f"], ["d2y2_safe"]),
        helper.make_node("Div", ["dy", "d2y2_safe"], ["offset_raw"]),
        helper.make_node("Clip", ["offset_raw", "neg_half_f", "half_f"], ["offset"]),
        helper.make_node("Add", ["idx_f", "offset"], ["centre"]),
        # amplitude above spectrum minimum
        helper.make_node("Sub", ["yc", "xmin"], ["amp_raw"]),
        helper.make_node("Max", ["amp_raw", "eps_f"], ["amp"]),
        # Gaussian log-ratio width per stencil h: sigma = h / sqrt(-2 ln r),
        # r = (a[i-h] + a[i+h]) / (2 a[i]) with amplitudes above the spectrum minimum
        *[
            node
            for h_i, h_f, tag in [("two_i", "h2_f", "h2"), ("eight_i", "h8_f", "h8")]
            for node in [
                helper.make_node("Sub", ["idx", h_i], [f"idx_m_{tag}_raw"]),
                helper.make_node("Add", ["idx", h_i], [f"idx_p_{tag}_raw"]),
                helper.make_node(
                    "Clip", [f"idx_m_{tag}_raw", "zero_i", "nmax_i"], [f"idx_m_{tag}"]
                ),
                helper.make_node(
                    "Clip", [f"idx_p_{tag}_raw", "zero_i", "nmax_i"], [f"idx_p_{tag}"]
                ),
                helper.make_node(
                    "GatherElements", ["x", f"idx_m_{tag}"], [f"yl_{tag}"], axis=1
                ),
                helper.make_node(
                    "GatherElements", ["x", f"idx_p_{tag}"], [f"yr_{tag}"], axis=1
                ),
                helper.make_node("Add", [f"yl_{tag}", f"yr_{tag}"], [f"ysum_{tag}"]),
                helper.make_node("Sub", [f"ysum_{tag}", "xmin"], [f"asum_{tag}_half"]),
                helper.make_node("Sub", [f"asum_{tag}_half", "xmin"], [f"asum_{tag}"]),
                helper.make_node("Mul", ["amp", "two_f"], [f"amp2_{tag}"]),
                helper.make_node(
                    "Div", [f"asum_{tag}", f"amp2_{tag}"], [f"ratio_{tag}_raw"]
                ),
                helper.make_node(
                    "Clip", [f"ratio_{tag}_raw", "r_lo_f", "r_hi_f"], [f"ratio_{tag}"]
                ),
                helper.make_node("Log", [f"ratio_{tag}"], [f"lnr_{tag}"]),
                helper.make_node("Mul", [f"lnr_{tag}", "neg_two_f"], [f"m2ln_{tag}"]),
                helper.make_node("Sqrt", [f"m2ln_{tag}"], [f"sq_{tag}"]),
                helper.make_node("Div", [h_f, f"sq_{tag}"], [f"sigma_{tag}"]),
            ]
        ],
        # narrow peaks resolve with h=2; when that ratio saturates the peak is wide -> h=8
        helper.make_node("Greater", ["ratio_h2", "r_wide_f"], ["is_wide"]),
        helper.make_node("Where", ["is_wide", "sigma_h8", "sigma_h2"], ["sigma"]),
        # peaks tensor [1, K, 13]
        helper.make_node("Unsqueeze", ["score", "axes_2"], ["score_3"]),
        helper.make_node("Unsqueeze", ["centre", "axes_2"], ["centre_3"]),
        helper.make_node("Unsqueeze", ["amp", "axes_2"], ["amp_3"]),
        helper.make_node("Unsqueeze", ["sigma", "axes_2"], ["sigma_3"]),
        helper.make_node("Unsqueeze", ["zeros_K", "axes_2"], ["zeros_3"]),
        helper.make_node("Unsqueeze", ["is_top1", "axes_2"], ["is_top1_3"]),
        helper.make_node(
            "Concat",
            [
                "score_3",
                "centre_3",
                "amp_3",
                "sigma_3",
                "sigma_3",
                "zeros_3",
                "sigma_3",
                "score_3",
                "zeros_3",
                "is_top1_3",
                "zeros_3",
                "zeros_3",
                "zeros_3",
            ],
            ["peaks"],
            axis=2,
        ),
        # rescore output scaled by the overridable initializer
        helper.make_node("Mul", ["score", "score_scale"], ["peak_rescore"]),
        # Gaussian reconstruction of the kept peaks (in-graph gate at 0.5, mirroring the
        # block's default gate) and the residual with those peaks stripped — sub-gate
        # peaks stay in the residual so a detect-subtract-repeat cascade can find them
        helper.make_node("GreaterOrEqual", ["score", "gate_K"], ["is_real"]),
        helper.make_node("Cast", ["is_real"], ["is_real_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["is_real_f", "axes_2"], ["is_real_3"]),
        helper.make_node("Sub", ["positions", "centre_3"], ["dx"]),  # [1, K, n]
        helper.make_node("Div", ["dx", "sigma_3"], ["z"]),
        helper.make_node("Mul", ["z", "z"], ["z_sq"]),
        helper.make_node("Mul", ["z_sq", "neg_half_f"], ["exponent"]),
        helper.make_node("Exp", ["exponent"], ["gauss"]),
        helper.make_node("Mul", ["gauss", "amp_3"], ["gauss_amp"]),
        helper.make_node("Mul", ["gauss_amp", "is_real_3"], ["gauss_real"]),
        helper.make_node(
            "ReduceSum", ["gauss_real", "axes_1"], ["recon_peaks"], keepdims=0
        ),  # [1, n]
        helper.make_node("Add", ["recon_peaks", "xmin"], ["reconstruction"]),
        helper.make_node("Sub", ["x", "recon_peaks"], ["residual"]),
    ]

    graph = helper.make_graph(
        nodes,
        "gr_onnx_peak_fixture",
        inputs=[
            helper.make_tensor_value_info("spectra", TensorProto.FLOAT, [1, m, n]),
            helper.make_tensor_value_info("score_scale", TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("heatmap", TensorProto.FLOAT, [1, n]),
            helper.make_tensor_value_info("peaks", TensorProto.FLOAT, [1, K, 13]),
            helper.make_tensor_value_info("peak_rescore", TensorProto.FLOAT, [1, K]),
            helper.make_tensor_value_info("reconstruction", TensorProto.FLOAT, [1, n]),
            helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, n]),
        ],
        initializer=initializers,
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "history_depth": str(m),
        "n_max_peaks": str(K),
        "n_peak_properties": "13",
        "property_layout": "peak_present,centre,amplitude,sigma_left,sigma_right,eta,sigma_avg,score,type_tag,is_top1,reserved0,reserved1,reserved2",
        "score_output": "peak_rescore",
        "normalise_mode": "None",
        "architecture": "analytic-fixture",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_peaks(path: Path, m: int = 1, n: int = N) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    spectrum = 0.1 + 0.01 * rng.standard_normal(n).astype(np.float32)
    x = np.arange(n, dtype=np.float32)
    # both probes are placed as a fraction of n so the self-test holds at every spectrum size
    narrow_centre, wide_centre = 0.1958 * n, 0.5859 * n
    for centre, amp, sig in [(narrow_centre, 5.0, 5.0), (wide_centre, 2.5, 20.0)]:
        spectrum += amp * np.exp(-0.5 * ((x - centre) / sig) ** 2)
    # older frames flat: detection must come from the newest frame (row m-1) only
    frames = np.full((1, m, n), 0.1, dtype=np.float32)
    frames[0, m - 1, :] = spectrum
    outputs = session.run(None, {"spectra": frames})
    heatmap, peaks, rescore = outputs[0], outputs[1], outputs[2]
    assert (
        heatmap.shape == (1, n)
        and peaks.shape == (1, K, 13)
        and rescore.shape == (1, K)
    )
    top, second = peaks[0, 0], peaks[0, 1]
    assert abs(top[1] - narrow_centre) < 1.0, f"centre {top[1]} != {narrow_centre}"
    assert abs(top[2] - 5.0) < 0.5, f"amplitude {top[2]} != 5.0"
    assert abs(top[3] - 5.0) < 0.5, f"sigma {top[3]} != 5.0"
    assert abs(second[1] - wide_centre) < 1.5, f"centre {second[1]} != {wide_centre}"
    assert abs(second[3] - 20.0) < 2.0, f"wide sigma {second[3]} != 20"
    scaled_rescore, scaled_heatmap = session.run(
        ["peak_rescore", "heatmap"],
        {"spectra": frames, "score_scale": np.array([2.0], dtype=np.float32)},
    )
    assert np.allclose(
        scaled_rescore, 2.0 * rescore, atol=1e-6
    ), "score_scale override ignored on peak_rescore"
    assert np.allclose(
        scaled_heatmap, 2.0 * heatmap, atol=1e-6
    ), "score_scale override ignored on heatmap"
    if m > 1:
        # a peak only in the OLDEST frame must not fire (flat newest frame -> no local maxima)
        stale = np.full((1, m, n), 0.1, dtype=np.float32)
        stale[0, 0, :] = spectrum
        (stale_scores,) = session.run(["peak_rescore"], {"spectra": stale})
        assert (
            float(stale_scores.max()) < 0.5
        ), f"stale-frame peak leaked: {stale_scores.max()}"
    print(
        f"self-test OK (M={m}): centre={top[1]:.2f} amplitude={top[2]:.2f} sigma={top[3]:.2f}"
    )


def build_identity_model(n: int = N_SMALL, m: int = 1) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity_graph",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, m, n])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, m, n])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "normalise_mode": "None",
        "architecture": "identity",
    }
    if m > 1:
        metadata["history_depth"] = str(m)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_identity(path: Path, n: int, m: int) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.arange(m * n, dtype=np.float32).reshape(1, m, n)
    (y,) = session.run(None, {"input": x})
    assert y.shape == (1, m, n), f"output shape {y.shape} != (1, {m}, {n})"
    assert np.allclose(
        x, y, atol=1e-7
    ), f"identity mismatch, max error {np.max(np.abs(x - y)):.2e}"
    print(f"self-test OK (identity, M={m}): max error {np.max(np.abs(x - y)):.2e}")


def build_affine_model() -> onnx.ModelProto:
    n = N_SMALL
    initializers = [
        const("shape_1N", np.array([1, n], dtype=np.int64)),
        const("scale", np.array(AFFINE_SCALE, dtype=np.float32)),
        const("offset", np.array(AFFINE_OFFSET, dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "shape_1N"], ["x"]),
        helper.make_node("Mul", ["x", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "offset"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gr_onnx_affine_fixture",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, n])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "normalise_mode": "None",
        "architecture": "analytic-affine",
        **provenance("affine"),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_affine(path: Path) -> None:
    import onnxruntime as ort

    n = N_SMALL
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.linspace(-3.0, 3.0, n, dtype=np.float32).reshape(1, 1, n)
    (y,) = session.run(None, {"input": x})
    expected = AFFINE_SCALE * x.reshape(1, n) + AFFINE_OFFSET
    assert y.shape == (1, n), f"output shape {y.shape} != (1, {n})"
    assert np.allclose(
        y, expected, atol=1e-6
    ), f"max error {np.max(np.abs(y - expected)):.2e}"
    print(
        f"self-test OK (affine): y = {AFFINE_SCALE}*x + {AFFINE_OFFSET}, max error {np.max(np.abs(y - expected)):.2e}"
    )


def build_frame_mean_model() -> onnx.ModelProto:
    n, m = N_SMALL, M_FRAME_MEAN
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["output"], axes=[1], keepdims=0)
    ]
    graph = helper.make_graph(
        nodes,
        "gr_onnx_frame_mean_fixture",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, m, n])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "history_depth": str(m),
        "normalise_mode": "None",
        "architecture": "analytic-frame-mean",
        **provenance("frame_mean"),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_frame_mean(path: Path) -> None:
    import onnxruntime as ort

    n, m = N_SMALL, M_FRAME_MEAN
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frames = np.stack(
        [np.full(n, float(k), dtype=np.float32) for k in range(m)]
    ).reshape(1, m, n)
    (y,) = session.run(None, {"input": frames})
    expected = np.full((1, n), (m - 1) / 2.0, dtype=np.float32)  # mean of 0..m-1
    assert y.shape == (1, n), f"output shape {y.shape} != (1, {n})"
    assert np.allclose(
        y, expected, atol=1e-6
    ), f"max error {np.max(np.abs(y - expected)):.2e}"
    print(
        f"self-test OK (frame_mean): mean of frames 0..{m - 1} = {(m - 1) / 2.0}, max error {np.max(np.abs(y - expected)):.2e}"
    )


def build_frame_delta_model() -> onnx.ModelProto:
    n, m = N_SMALL, M_FRAME_DELTA
    initializers = [
        const("shape_1N", np.array([1, n], dtype=np.int64)),
        const("axes_1", np.array([1], dtype=np.int64)),
        const("slice_0", np.array([0], dtype=np.int64)),
        const("slice_1", np.array([1], dtype=np.int64)),
        const("slice_m1", np.array([m - 1], dtype=np.int64)),
        const("slice_m", np.array([m], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node(
            "Slice", ["input", "slice_m1", "slice_m", "axes_1"], ["newest"]
        ),
        helper.make_node(
            "Slice", ["input", "slice_0", "slice_1", "axes_1"], ["oldest"]
        ),
        helper.make_node("Sub", ["newest", "oldest"], ["delta"]),
        helper.make_node("Reshape", ["delta", "shape_1N"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gr_onnx_frame_delta_fixture",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, m, n])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "history_depth": str(m),
        "normalise_mode": "None",
        "architecture": "analytic-frame-delta",
        **provenance("frame_delta"),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_frame_delta(path: Path) -> None:
    import onnxruntime as ort

    n, m = N_SMALL, M_FRAME_DELTA
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frames = np.stack(
        [np.full(n, float(k), dtype=np.float32) for k in range(m)]
    ).reshape(1, m, n)
    (y,) = session.run(None, {"input": frames})
    expected = np.full(
        (1, n), float(m - 1), dtype=np.float32
    )  # newest (m-1) minus oldest (0)
    assert y.shape == (1, n), f"output shape {y.shape} != (1, {n})"
    assert np.allclose(
        y, expected, atol=1e-6
    ), f"max error {np.max(np.abs(y - expected)):.2e}"
    (y_rev,) = session.run(None, {"input": frames[:, ::-1, :].copy()})
    assert np.allclose(
        y_rev, -expected, atol=1e-6
    ), "reversed frame order must negate the delta"
    print(
        f"self-test OK (frame_delta): newest - oldest = {float(m - 1)}, order-sensitivity confirmed"
    )


def build_mean_rms_model() -> onnx.ModelProto:
    n = N_SMALL
    initializers = [const("shape_1N", np.array([1, n], dtype=np.int64))]
    nodes = [
        helper.make_node("Reshape", ["input", "shape_1N"], ["x"]),
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["x", "x"], ["x_sq"]),
        helper.make_node("ReduceMean", ["x_sq"], ["mean_sq"], axes=[1], keepdims=1),
        helper.make_node("Sqrt", ["mean_sq"], ["rms"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gr_onnx_mean_rms_fixture",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, n])],
        outputs=[
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("rms", TensorProto.FLOAT, [1, 1]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    metadata = {
        "input_size": str(n),
        "normalise_mode": "None",
        "architecture": "analytic-mean-rms",
        **provenance("mean_rms"),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def self_test_mean_rms(path: Path) -> None:
    import onnxruntime as ort

    n = N_SMALL
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.arange(1, n + 1, dtype=np.float32)  # 1..n
    mean, rms = session.run(["mean", "rms"], {"input": x.reshape(1, 1, n)})
    assert mean.shape == (1, 1) and rms.shape == (
        1,
        1,
    ), f"shapes {mean.shape}, {rms.shape}"
    assert abs(mean.item() - np.mean(x)) < 1e-4, f"mean {mean.item()} != {np.mean(x)}"
    assert (
        abs(rms.item() - np.sqrt(np.mean(x * x))) < 1e-3
    ), f"rms {rms.item()} != {np.sqrt(np.mean(x * x))}"
    print(f"self-test OK (mean_rms): mean={mean.item():.4f} rms={rms.item():.4f}")


def convert_to_ort(onnx_path: Path, keep_uncompressed: bool = False) -> Path:
    """Emit the pre-optimised .ort form the minimal and Emscripten builds require.

    Only the .gz forms are tracked, so the uncompressed intermediates are removed once packed —
    otherwise every run leaves clutter that has to be carried by ignore rules. `keep_uncompressed`
    is for identity_N64, which is tracked in all four forms because it *is* the
    format-equivalence test.
    """
    import onnxruntime as ort

    ort_path = onnx_path.with_suffix(".ort")
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(ort_path)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 6
    options.log_severity_level = 3
    ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    print(f"wrote {ort_path} ({ort_path.stat().st_size} bytes)")
    gzip_alongside(onnx_path)  # native reads the protobuf form
    gzip_alongside(ort_path)  # WASM/minimal runtimes read only the flatbuffer form
    if not keep_uncompressed:
        onnx_path.unlink(missing_ok=True)
        ort_path.unlink(missing_ok=True)
        print(f"  removed the uncompressed intermediates of {onnx_path.stem}")
    return ort_path


def gzip_alongside(path: Path) -> Path:
    """Write path.gz next to path. LFS stores blobs verbatim, so this is a real saving there.

    Run natively only; the CI never executes this script and needs no Python environment for it.
    """
    import gzip as _gzip
    import shutil

    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, _gzip.open(gz_path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    print(f"wrote {gz_path} ({gz_path.stat().st_size} bytes)")
    return gz_path


def build_heatmap_only_model(n: int = N) -> onnx.ModelProto:
    """A model that emits a heatmap and NO peaks tensor — the negative fixture proving
    OnnxPeakDetector stops rather than silently forwarding when the contract is unmet.
    """
    spectra = onnx.helper.make_tensor_value_info(
        "spectra", onnx.TensorProto.FLOAT, [1, 1, n]
    )
    heatmap = onnx.helper.make_tensor_value_info(
        "heatmap", onnx.TensorProto.FLOAT, [1, n]
    )
    nodes = [
        onnx.helper.make_node("Squeeze", ["spectra", "axis_1"], ["frame"]),
        onnx.helper.make_node("Sigmoid", ["frame"], ["heatmap"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "heatmap_only",
        [spectra],
        [heatmap],
        initializer=[const("axis_1", np.array([1], dtype=np.int64))],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 9
    for key, value in (
        {
            "architecture": "heatmap-only-fixture",
            "input_size": str(n),
            "history_depth": "1",
        }
        | provenance("heatmap_only")
    ).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def generate_peaks_expr(out_dir: Path) -> None:
    """The peaks fixture declaring an identity ExprTk normalisation in its metadata — the only
    coverage of the model-declared-expression path, so it is generated and tracked like the rest
    rather than left for the qa suite to skip over."""
    model = build_peaks_model(1, N)
    for key, value in {
        "normalise_mode": "Expression",
        "normalise_expr": "vecOut := vecIn",
    }.items():
        for existing in model.metadata_props:
            if existing.key == key:
                existing.value = value
                break
        else:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
    path = out_dir / f"peaks_fixture_expr_N{N}.onnx"
    onnx.save(model, str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    convert_to_ort(path)


def generate_heatmap_only(out_dir: Path) -> None:
    path = out_dir / f"heatmap_only_N{N}.onnx"
    onnx.save(build_heatmap_only_model(N), str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    convert_to_ort(path)


def generate_peaks(out_dir: Path) -> None:
    """M=16 covers the history path and N_ALT the model-swap path, so neither needs a trained
    detector tracked to test it."""
    for m, n in [(1, N), (M_HISTORY, N), (1, N_ALT)]:
        suffix = f"_M{m}" if m > 1 else ""
        path = out_dir / f"peaks_fixture_N{n}{suffix}.onnx"
        onnx.save(build_peaks_model(m, n), str(path))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        self_test_peaks(path, m, n)
        convert_to_ort(path)


def generate_identity(out_dir: Path) -> None:
    for m in (1, M_IDENTITY):
        suffix = f"_M{m}" if m > 1 else ""
        path = out_dir / f"identity_N{N_SMALL}{suffix}.onnx"
        onnx.save(build_identity_model(N_SMALL, m), str(path))
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        self_test_identity(path, N_SMALL, m)
        convert_to_ort(path, keep_uncompressed=(m == 1))


def generate_affine(out_dir: Path) -> None:
    path = out_dir / f"affine_N{N_SMALL}.onnx"
    onnx.save(build_affine_model(), str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    self_test_affine(path)
    convert_to_ort(path)


def generate_frame_mean(out_dir: Path) -> None:
    path = out_dir / f"frame_mean_N{N_SMALL}_M{M_FRAME_MEAN}.onnx"
    onnx.save(build_frame_mean_model(), str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    self_test_frame_mean(path)
    convert_to_ort(path)


def generate_frame_delta(out_dir: Path) -> None:
    path = out_dir / f"frame_delta_N{N_SMALL}_M{M_FRAME_DELTA}.onnx"
    onnx.save(build_frame_delta_model(), str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    self_test_frame_delta(path)
    convert_to_ort(path)


def generate_mean_rms(out_dir: Path) -> None:
    path = out_dir / f"mean_rms_N{N_SMALL}.onnx"
    onnx.save(build_mean_rms_model(), str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    self_test_mean_rms(path)
    convert_to_ort(path)


FIXTURES = {
    "peaks": generate_peaks,
    "heatmap_only": generate_heatmap_only,
    "peaks_expr": generate_peaks_expr,
    "identity": generate_identity,
    "affine": generate_affine,
    "frame_mean": generate_frame_mean,
    "frame_delta": generate_frame_delta,
    "mean_rms": generate_mean_rms,
}


def main() -> int:
    default_out = Path(__file__).resolve().parent.parent / "models"
    parser = argparse.ArgumentParser(description="Build the analytic ONNX qa fixtures.")
    parser.add_argument("out_dir", nargs="?", type=Path, default=default_out)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIXTURES),
        metavar="TARGET",
        help=f"subset of {{{', '.join(sorted(FIXTURES))}}} (default: all)",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = args.out_dir.resolve()
    if not out_dir.is_relative_to(repo_root):
        sys.exit(f"fixtures belong inside {repo_root}, not '{out_dir}'")
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in args.only or sorted(FIXTURES):
        FIXTURES[target](out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
