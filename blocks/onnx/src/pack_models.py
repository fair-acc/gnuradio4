"""Package a model for deployment: convert to .ort and gzip.

The fp32 .onnx stays the master artefact; .ort is the form WASM and other minimal runtimes consume,
and both ship gzip-compressed because `fileio` inflates a `.gz` source on read.

Run natively; the CI never executes this and needs no Python environment for it.

    python pack_models.py <model.onnx>
"""

import gzip, os, shutil, subprocess, sys
from pathlib import Path

import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed


def freeze_input_shape(onnx_path: Path) -> None:
    """Pin the free batch/frame dims so the runtime skips symbolic shape resolution at load.

    The exporter leaves `spectra` as [unk, unk, N]; resolving that costs ~0.7 s of every session
    construction and dominates load time for the .ort form -- 0.778 s -> 0.045 s once pinned. The
    block feeds one DataSet per call, and the model states its own frame count, so the concrete
    shape is [1, history_depth, input_size]. Bit-identical output: only the declared dims change.
    """
    model = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in model.metadata_props}
    shape = [1, int(meta.get("history_depth", 1)), int(meta["input_size"])]
    make_input_shape_fixed(model.graph, model.graph.input[0].name, shape)
    fix_output_shapes(model)
    onnx.save(model, str(onnx_path))


def to_ort(onnx_path: Path) -> Path:
    """Convert to .ort using only ops a stock ONNX Runtime has.

    "Fixed" style bakes ORT's fusions into the flatbuffer, and those fusions emit com.microsoft
    contrib ops -- FusedGemm/FusedMatMul. A runtime built without contrib ops (a distribution
    package, or a minimal WASM/TPU build) then refuses the model with a bare "failed to load".
    "Runtime" style defers those fusions to load time, so the file holds standard ops and the
    runtime applies whatever fusions it actually has.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(onnx_path),
            "--output_dir",
            str(onnx_path.parent),
            "--optimization_style",
            "Runtime",
        ],
        check=True,
        capture_output=True,
    )
    produced = onnx_path.with_name(onnx_path.stem + ".with_runtime_opt.ort")
    ort = onnx_path.with_suffix(".ort")
    produced.replace(ort)
    for leftover in onnx_path.parent.glob(
        onnx_path.stem + "*.required_operators*.config"
    ):
        leftover.unlink()
    with ort.open("rb") as f, gzip.open(str(ort) + ".gz", "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    return ort


if len(sys.argv) != 2:
    sys.exit(__doc__)
# resolve first: this reaches an argv list, where a leading '-' would read as a flag
src = Path(sys.argv[1]).resolve()
if not src.is_file() or src.suffix != ".onnx":
    sys.exit(f"expected an existing .onnx file, got '{src}'")
freeze_input_shape(src)
ort = to_ort(src)
with src.open("rb") as f, gzip.open(str(src) + ".gz", "wb", compresslevel=9) as g:
    shutil.copyfileobj(f, g)
for p in (src, Path(str(src) + ".gz"), ort, Path(str(ort) + ".gz")):
    print(f"  {p.name:52} {os.path.getsize(p)/1048576:7.2f} MB")
