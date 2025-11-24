from __future__ import annotations
import time
import statistics as stats
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, List

import numpy as np
import tensorflow as tf

_physical_gpus = tf.config.list_physical_devices("GPU")
_TF_DEFAULT_DEVICE = "/GPU:0" if _physical_gpus else "/CPU:0"


@dataclass
class BenchmarkResult:
    name: str
    kind: str  # "GEMM" or "GEMV"
    impl: str  # "naive" / "numpy"
    dtype: str
    M: int
    K: int
    N: Optional[int]  # None for GEMV
    n_repeats: int
    passed: bool
    max_abs_err: float
    times_ns: List[int]
    scaling: int  # operations per run (e.g. 2*M*K*N)

    @property
    def total_time_ns(self) -> int:
        return sum(self.times_ns)

    @property
    def min_ns(self) -> int:
        return min(self.times_ns)

    @property
    def max_ns(self) -> int:
        return max(self.times_ns)

    @property
    def mean_ns(self) -> float:
        return stats.fmean(self.times_ns)

    @property
    def stddev_ns(self) -> float:
        return stats.pstdev(self.times_ns) if len(self.times_ns) > 1 else 0.0

    @property
    def median_ns(self) -> float:
        return stats.median(self.times_ns)

    @property
    def ops_per_second(self) -> float:
        if self.total_time_ns == 0:
            return float("inf")
        total_ops = self.scaling * self.n_repeats
        total_seconds = self.total_time_ns * 1e-9
        return total_ops / total_seconds


def format_time(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.0f} ns"
    us = ns / 1e3
    if us < 1e3:
        return f"{us:.1f} us"
    ms = us / 1e3
    if ms < 1e3:
        return f"{ms:.2f} ms"
    s = ms / 1e3
    return f"{s:.2f} s"


def format_ops_per_s(ops: float) -> str:
    if ops >= 1e9:
        return f"{ops/1e9:.1f}G"
    if ops >= 1e6:
        return f"{ops/1e6:.1f}M"
    if ops >= 1e3:
        return f"{ops/1e3:.1f}k"
    return f"{ops:.1f}"


# --- naive Python-loop implementations over NumPy tensors --------------------


def gemm_naive(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> None:
    """Naive triple-loop GEMM: C = A * B, using Python loops."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += float(A[i, p]) * float(B[p, j])
            out[i, j] = s


def gemm_numpy(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> None:
    """Vectorised GEMM using NumPy's matmul / BLAS backend."""
    out[...] = A @ B


def gemm_tf_gpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> None:
    """
    GEMM using TensorFlow on the selected device (GPU if available).

    This function:
      - copies A and B from host (NumPy) to device (TensorFlow tensor),
      - performs C = A @ B on the device,
      - copies the result back into the given NumPy 'out' array.

    All transfers happen on every call, so H2D/D2H overhead is included
    in the benchmark.
    """
    with tf.device(_TF_DEFAULT_DEVICE):
        # Host -> device copies
        A_tf = tf.convert_to_tensor(A)  # lives on _TF_DEFAULT_DEVICE
        B_tf = tf.convert_to_tensor(B)

        # Device matmul
        C_tf = tf.matmul(A_tf, B_tf)

    # Device -> host copy
    out[...] = C_tf.numpy()


def gemv_naive(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Naive GEMV: y = A * x, using Python loops."""
    m, k = A.shape
    assert x.shape[0] == k
    for i in range(m):
        s = 0.0
        for p in range(k):
            s += float(A[i, p]) * float(x[p])
        y[i] = s


def gemv_numpy(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Vectorised GEMV using NumPy matmul."""
    y[...] = A @ x


def gemv_tf_gpu(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Vectorised GEMV using TensorFlow matmul."""
    with tf.device(_TF_DEFAULT_DEVICE):
        A_tf = tf.convert_to_tensor(A)
        x_tf = tf.convert_to_tensor(x)
        y_tf = tf.matmul(A_tf, tf.expand_dims(x_tf, -1))
    y[...] = tf.squeeze(y_tf, -1).numpy()


# --- generic benchmark harness -----------------------------------------------


def run_benchmark(
    name: str,
    kind: str,
    impl: str,
    func: Callable,
    ref_func: Callable,
    dtype: np.dtype,
    M: int,
    K: int,
    N: Optional[int],
    n_repeats: int,
    atol: float = 1e-4,
) -> BenchmarkResult:
    rng = np.random.default_rng(12345)
    dtype = np.dtype(dtype)

    if kind == "GEMM":
        assert N is not None
        A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(dtype)
        B = rng.uniform(-1.0, 1.0, size=(K, N)).astype(dtype)
        C_ref = ref_func(A, B)
        C = np.empty_like(C_ref)
        scaling = 2 * M * K * N
        args = (A, B, C)
        ref = C_ref
    elif kind == "GEMV":
        assert N is None
        A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(dtype)
        x = rng.uniform(-1.0, 1.0, size=(K,)).astype(dtype)
        y_ref = ref_func(A, x)
        y = np.empty_like(y_ref)
        scaling = 2 * M * K
        args = (A, x, y)
        ref = y_ref
    else:
        raise ValueError(f"Unknown kind {kind}")

    times: List[int] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        func(*args)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    max_abs_err = float(np.max(np.abs(args[-1] - ref)))
    passed = bool(max_abs_err <= atol)

    return BenchmarkResult(
        name=name,
        kind=kind,
        impl=impl,
        dtype=dtype.name,
        M=M,
        K=K,
        N=N,
        n_repeats=n_repeats,
        passed=passed,
        max_abs_err=max_abs_err,
        times_ns=times,
        scaling=scaling,
    )


def print_results(results: Sequence[BenchmarkResult]) -> None:
    # Columns to show (status and max|err removed)
    headers = [
        "benchmark",
        "impl",
        "dtype",
        "#N",
        "min",
        "mean",
        "stddev",
        "median",
        "max",
        "total",
        "ops/s",
    ]

    # Build all rows first so we can compute widths and know impl per row
    rows: List[List[str]] = []
    impls: List[str] = []

    for r in results:
        dims = f"M={r.M},K={r.K}" + (f",N={r.N}" if r.N is not None else "")
        row = [
            f"{r.kind} {dims}",
            r.impl,
            r.dtype,
            str(r.n_repeats),
            format_time(r.min_ns),
            format_time(r.mean_ns),
            format_time(r.stddev_ns),
            format_time(r.median_ns),
            format_time(r.max_ns),
            format_time(r.total_time_ns),
            format_ops_per_s(r.ops_per_second),
        ]
        rows.append(row)
        impls.append(r.impl)

    # Compute per-column widths from headers + all rows
    col_widths: List[int] = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            max_len = max(max_len, len(row[col_idx]))
        col_widths.append(max_len)

    def fmt_row(columns: List[str]) -> str:
        return " | ".join(col.ljust(width) for col, width in zip(columns, col_widths))

    # Common separator line length
    sep_line = "-" * (sum(col_widths) + 3 * (len(col_widths) - 1))

    # Print header and top separator
    print(fmt_row(headers))
    print(sep_line)

    # Print rows, inserting a separator whenever impl changes
    prev_impl: Optional[str] = None
    for row, impl in zip(rows, impls):
        if prev_impl is not None and impl != prev_impl:
            print(sep_line)
        print(fmt_row(row))
        prev_impl = impl


def main():
    # Same shapes as your C++ benchmark
    gemm_sizes = [(6, 6, 6), (64, 64, 64), (256, 256, 256), (1024, 1024, 1024)]
    gemv_sizes = [(6, 6), (64, 64), (256, 256), (1024, 1024)]

    # For GEMM(naive) a full 1024^3 loop in Python is impractical, so we skip
    # that case by default. Set this True if you *really* want to try.
    RUN_LARGE_NAIVE_GEMM = False

    dtypes = [np.float32, np.float64]

    results: List[BenchmarkResult] = []

    # Ordering:
    #  1) kind: GEMM, GEMV
    #  2) impl: naive, numpy, TensorFlow
    #  3) sizes: increasing
    for kind in ("GEMM", "GEMV"):
        for impl in ("naive", "numpy", "TensorFlow"):
            for dt in dtypes:
                atol = 1e-4 if dt == np.float32 else 1e-8

                if kind == "GEMM":
                    for M, K, N in gemm_sizes:
                        # Select implementation and skip unsupported combos
                        if impl == "naive":
                            if not (RUN_LARGE_NAIVE_GEMM or max(M, K, N) <= 256):
                                continue  # skip 1024³ naive
                            n_repeats = 10 if max(M, K, N) <= 64 else 3
                            func = gemm_naive
                        elif impl == "numpy":
                            n_repeats = 10
                            func = gemm_numpy
                        elif impl == "TensorFlow":
                            n_repeats = 10
                            func = gemm_tf_gpu
                        else:
                            continue

                        r = run_benchmark(
                            name=f"{kind}({impl}) {np.dtype(dt).name} M={M},K={K},N={N}",
                            kind="GEMM",
                            impl=impl,
                            func=func,
                            ref_func=lambda A, B: A @ B,
                            dtype=dt,
                            M=M,
                            K=K,
                            N=N,
                            n_repeats=n_repeats,
                            atol=atol,
                        )
                        results.append(r)

                else:  # kind == "GEMV"
                    # You already have gemv_tf_gpu, so we treat all three
                    # implementations analogously here.
                    for M, K in gemv_sizes:
                        if impl == "naive":
                            if max(M, K) <= 64:
                                n_repeats = 10_000
                            elif max(M, K) <= 256:
                                n_repeats = 1_000
                            else:
                                n_repeats = 200
                            func = gemv_naive
                        elif impl == "numpy":
                            n_repeats = 10_000
                            func = gemv_numpy
                        elif impl == "TensorFlow":
                            # You can dial this down if it becomes too slow,
                            # but 10k gives you a comparable baseline.
                            n_repeats = 10_000
                            func = gemv_tf_gpu
                        else:
                            continue

                        r = run_benchmark(
                            name=f"{kind}({impl}) {np.dtype(dt).name} M={M},K={K}",
                            kind="GEMV",
                            impl=impl,
                            func=func,
                            ref_func=lambda A, x: A @ x,
                            dtype=dt,
                            M=M,
                            K=K,
                            N=None,
                            n_repeats=n_repeats,
                            atol=atol,
                        )
                        results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
