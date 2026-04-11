# GPU Compute Integration for GNU Radio 4.0 — Consolidated Design (V3)

## Status: WIP — Stages 0–6 + 7A done. Tier 1 (minimum viable wiring) pending.

**Author**: Ralph J. Steinhagen (GSI/FAIR)
**Date**: 2026-02-16 (initial), 2026-04-03 (V2), 2026-04-06 (V3), 2026-04-07 (V4 — Van Loan fix + tests),
2026-04-16 (V5 — rebased onto coreCleanUp, updated remaining work)
**Branch**: `syclExperiments` on `fair-acc/gnuradio4` (rebased onto `coreCleanUp`)

---

## 0. Process Constraints

All implementation follows the CLAUDE.md style guide plus these constraints:

1. **AI-assisted workflow.** Claude Code implements; the user guides, reviews, and approves each
   step. All estimates are AI-assisted (user as guide/reviewer).
2. **Review gate between stages.** Each stage ends with a review + Q&A session. Open points from
   the current and next stage are discussed. Advance to the next stage only after an explicit
   **"go ahead"** from the user.
3. **No autonomous commits or pushes.** Claude Code does not `git commit` or `git push` unless
   the user explicitly authorises it. Work is presented as uncommitted changes for review.
4. **Each step is testable.** Every implementation step must compile and pass its tests before
   the review gate.

---

## 1. Executive Summary

This document describes the design for integrating heterogeneous compute (GPU, FPGA, etc.) into
GNU Radio 4.0 (GR4). The primary backend is SYCL via AdaptiveCpp; GL Compute (WebGPU/WGSL) serves
as the WASM fallback; native CUDA/ROCm are future extension points.

**Core principle:** A user who writes `processOne(...) const noexcept` or `processBulk(...)` for CPU
can run the same code on a GPU without changes and achieve ~70–80% of device performance. An escape
hatches (`processBulk_sycl`, `generateShader`, etc.) allow expert users to write native device
code for 100% performance with backend-specific parameters.

Two use-cases drive the design:

- **Case A — Standalone GPU blocks**: A block (or implicit single-block chain) transfers data to
  device, computes, and returns results. Explicit `HostToDevice<T>` / `DeviceToHost<T>` blocks
  control batch/DMA size. When absent, the scheduler handles transfers internally.
- **Case B — GPU sub-graphs** (Phase 2): A chain of GPU-capable blocks executes on-device. Data
  stays on device between blocks. `HostToDevice<T>` at entry, `DeviceToHost<T>` at exit.

---

## 2. Design Goals and Constraints

### 2.1 Goals

1. **Preserve the Block API.** `processOne(T) const noexcept` and `processBulk(...)` unchanged.
2. **Auto-parallelisation.** `const noexcept` processOne maps to `bulk` dispatch automatically.
3. **Backend-specific escape hatches.** `processBulk_sycl(sycl::queue&, ...)`,
   `generateShader(settings)`, `processBulk_cuda(CUstream, ...)` — native parameter types,
   zero-cost abstraction per backend.
4. **Backend-agnostic.** SYCL (primary), GL Compute (WASM), CUDA/ROCm (future). One `compute_domain`
   setting selects the target; `ComputeRegistry` resolves it.
5. **Zero cost for CPU-only graphs.** GPU code paths are compile-guarded. Existing CPU + SIMD
   dispatch is completely untouched.
6. **CI-testable without GPU hardware.** AdaptiveCpp CPU/OpenMP fallback for SYCL tests; Mesa
   llvmpipe for GL Compute tests.
7. **P2300-compatible execution model.** `gr::execution` layer uses `std::execution` names and
   concepts. Migration to the standard is a namespace swap.
8. **Composition over modification.** GPU dispatch logic lives outside `Block<T>` in a composed
   `gpu::ExecutionStrategy`. `Block<T>` gains one `if constexpr` branch, not hundreds of lines.

### 2.2 Constraints

- **Primary GPU target**: NVIDIA, with AMD and Intel compatible via SYCL abstraction.
- **Compilers**: AdaptiveCpp (SSCP, clang ≥ 20), GCC 15+, Clang 20+, Emscripten. GCC and
  Emscripten must compile cleanly with GPU features disabled.
- **No external GPU library dependencies** beyond AdaptiveCpp itself for Phase 1.
- **No `stdexec` dependency.** Compile-time cost too high; WASM untested.
- **float32 focus.** Initial GPU support targets `float`. `gr::complex<float>` is introduced
  for complex-valued blocks (layout-compatible with `std::complex<float>`, works on device).
- **SIMD retained.** CPU execution uses `vir::simd` / `std::simd` as today. GPU uses scalar
  `processOne` per work-item (GPU hardware handles SIMD via warps/wavefronts).

---

## 3. Existing Infrastructure (preserve as-is)

### 3.1 `compute_domain` setting (Block.hpp:706)

```cpp
A<std::string, "compute domain", Doc<"compute domain/IO thread pool name">> compute_domain = "default_io";
```

Already a reflected setting on every block. Currently selects thread pool. Extended to select
GPU targets via string format:

| Value           | Target                                  |
|-----------------|-----------------------------------------|
| `"default_cpu"` | CPU thread pool (existing default)      |
| `"default_io"`  | I/O thread pool (existing)              |
| `"gpu"`         | any available GPU, framework picks best |
| `"gpu:sycl"`    | SYCL backend, default device            |
| `"gpu:sycl:0"`  | SYCL backend, device index 0            |
| `"gpu:gl"`      | GL Compute backend                      |

If `compute_domain = "gpu:..."` but no matching device is available: **warn and fall back to CPU**.

### 3.2 `ComputeRegistry` (ComputeDomain.hpp)

Extended to resolve both memory resources and execution schedulers from `compute_domain` strings:

```cpp
ComputeRegistry::instance().resolve("gpu:sycl:0")           → memory_resource*
ComputeRegistry::instance().resolveScheduler("gpu:sycl:0")  → GpuScheduler&
```

Single registry, one key for both memory and execution.

### 3.3 SIMD dispatch (Block.hpp:1747–1774)

The existing dispatch tree in `Block::work()`:

```
const + SIMD capable  →  invokeProcessOneSimd (vectorised loop)
const + no SIMD       →  invokeProcessOnePure (scalar loop)
non-const             →  invokeProcessOneNonConst (sample-by-sample)
```

The `const` qualifier already gates SIMD eligibility (with a `static_assert` preventing non-const
SIMD). GPU uses the same gate. The GPU branch is inserted *before* the SIMD branch via composition.

### 3.4 `AtomicRef<T>` (AtomicRef.hpp)

Dual-path atomics: `sycl::atomic_ref` on device, `std::atomic_ref` on host. Already used
throughout `CircularBuffer`. Unchanged.

### 3.5 Thread pool infrastructure (thread_pool.hpp)

`BasicThreadPool`, `TaskExecutor`, `Manager` singleton, `ThreadPoolWrapper`. All preserved.
The `gr::execution::PoolScheduler` wraps `TaskExecutor` as a P2300-compatible scheduler.

---

## 4. Dispatch Hierarchy

The framework selects the execution path at compile time based on block traits, `compute_domain`,
and the active backend:

```
1. Backend is SYCL  + has processBulk_sycl?  →  call processBulk_sycl(sycl::queue&, ...)
2. Backend is WGSL  + has generateShader?     →  compile shader → bind → dispatch
3. Backend is CUDA  + has processBulk_cuda?   →  call processBulk_cuda(CUstream, ...)
4. Backend is ROCm  + has processBulk_rocm?   →  call processBulk_rocm(hip_stream, ...)
5. Any GPU backend + const noexcept processOne? → auto-parallelise per backend
6. CPU domain + has processBulk_cpu?          →  explicit CPU specialisation
7. CPU domain + const + SIMD capable?         →  invokeProcessOneSimd (existing, unchanged)
8. CPU domain + const + no SIMD?              →  invokeProcessOnePure (existing, unchanged)
9. CPU domain + non-const processOne?         →  invokeProcessOneNonConst (existing, unchanged)
10. processBulk?                              →  call processBulk (existing, unchanged)
```

Levels 7–10 are completely untouched. Levels 1–4 are backend-specific escape hatches with native
parameters (no `void*` casting, type-safe). Level 5 is auto-parallelisation. Level 6 is optional
CPU specialisation.

### 4.1 Auto-parallelisation concept

```cpp
template<typename TBlock>
concept AutoParallelisable = requires(const TBlock& b, typename TBlock::input_type v) {
    { b.processOne(v) } noexcept -> std::same_as<typename TBlock::output_type>;
};
```

The `const noexcept` qualifiers are the compile-time gate. Matches the existing SIMD gate exactly.
Auto-parallelisation dispatches per backend:

- SYCL: `sycl::parallel_for(N, [state](id i) { out[i] = processOne(in[i]); })`
- WGSL: auto-generated shader from `processOne` (Phase 2)
- CPU: scalar loop or SIMD (existing)

### 4.2 Backend-specific escape hatch concepts

Each backend gets its own concept with native parameter types:

```cpp
template<typename TBlock>
concept HasSyclBulk = requires(TBlock& b, sycl::queue& q,
    const typename TBlock::input_type* in, std::size_t nIn,
    typename TBlock::output_type* out, std::size_t nOut) {
    { b.processBulk_sycl(q, in, nIn, out, nOut) } -> std::same_as<gr::work::Status>;
};

template<typename TBlock>
concept HasWgslShader = requires(const gr::property_map& settings) {
    { TBlock::generateShader(settings) } -> std::convertible_to<std::string>;
};

// future:
// concept HasCudaBulk = requires(TBlock& b, CUstream stream, ...) { ... };
// concept HasRocmBulk = requires(TBlock& b, hipStream_t stream, ...) { ... };
```

A block can provide escape hatches for multiple backends:

```cpp
struct FFT2 : gr::Block<FFT2<T>> {
    gr::work::Status processBulk(...);                      // CPU fallback
    gr::work::Status processBulk_sycl(sycl::queue&, ...);  // SYCL escape hatch
    static std::string generateShader(const property_map&); // WGSL escape hatch
};
```

### 4.3 Naming convention for specialised process functions

| Function                                  | Backend             | Parameters                                         |
|-------------------------------------------|---------------------|----------------------------------------------------|
| `processOne(T) const noexcept`            | all (auto-parallel) | scalar value                                       |
| `processOne(t_or_simd<T>) const noexcept` | CPU SIMD            | SIMD vector                                        |
| `processBulk(...)`                        | CPU                 | `InputSpanLike`, `OutputSpanLike`                  |
| `processBulk_sycl(sycl::queue&, ...)`     | SYCL                | native queue + `std::span` over USM memory         |
| `processBulk_cuda(CUstream, ...)`         | CUDA (future)       | native stream + `std::span` over device memory     |
| `processBulk_rocm(hipStream_t, ...)`      | ROCm (future)       | native stream + `std::span` over device memory     |
| `shaderFragment()`                        | GLSL/WGSL           | returns `ShaderFragment` (GLSL source + constants) |

**Revisit (7A follow-up):** consider unifying device escape hatches to use the same
`InputSpanLike`/`OutputSpanLike` as `processBulk`, with the backend handle as the only
distinguishing parameter: `processBulk(sycl::queue&, InputSpanLike auto&, OutputSpanLike auto&)`.
This would give identical UX for CPU and GPU code. The `_sycl` suffix is retained for now
because it makes trait detection trivial (`requires { &D::processBulk_sycl; }`). The overload
approach (same `processBulk` name, distinguish by first parameter type) is more intuitive for
block authors but harder to detect with concepts. Revisit once the full dispatch pipeline (7C)
is working and the USM edge semantics (7B) are validated.

---

## 5. Composition Architecture

### 5.1 Block<T> modification — minimal

`Block<T>::workInternal()` gains one `if constexpr` branch before the existing dispatch:

```cpp
if constexpr (gpu::isGpuEligible<Derived> && /* compute_domain indicates GPU */) {
    gpu::ExecutionStrategy<Derived>::dispatch(self(), inputSpans, outputSpans, processedIn);
} else if constexpr (HasConstProcessOneFunction<Derived>) {
    // existing SIMD / scalar paths — untouched
    ...
}
```

### 5.2 `gpu::ExecutionStrategy<TBlock>` — composed, not inherited

Lives in `core/include/gnuradio-4.0/gpu/ExecutionStrategy.hpp`. Handles:

- Backend detection from `compute_domain` string
- Trait detection: per-backend escape hatch concepts vs `AutoParallelisable`
- `DeviceBlockState<TBlock>` generation via reflection (scalar fields copied, PMR containers
  mirrored as `{T* ptr, size_t size}`)
- H2D/D2H transfer (delegates to `GpuScheduler` or explicit `HostToDevice<T>` blocks)
- Dispatch to backend-specific method or auto-parallelised `bulk`

### 5.3 Layer diagram

```
User Block Code
    processOne(T) const noexcept     — auto-parallelised on any backend
    processBulk_sycl(queue&, ...)    — SYCL escape hatch (native types)
    generateShader(settings)         — WGSL/GLSL escape hatch (shader text)
    processBulk_cuda(CUstream, ...)  — CUDA escape hatch (future)
    ┌──────────────────────────────────────────────┐
    │         Block<T>::workInternal()              │
    │  if constexpr (gpu eligible + gpu domain)     │
    │      → gpu::ExecutionStrategy<Derived>        │
    │  else                                         │
    │      → existing SIMD / scalar dispatch        │
    └──────────┬───────────────────────────────────┘
               │
    ┌──────────▼───────────────────────────────────┐
    │    gpu::ExecutionStrategy<TBlock>             │
    │  ┌────────────┐  ┌─────────────────────────┐ │
    │  │ DeviceBlock │  │ H2D / D2H transfer      │ │
    │  │ State<T>   │  │ (implicit or via         │ │
    │  │ (reflected) │  │  HostToDevice<T> block)  │ │
    │  └─────┬──────┘  └──────────┬──────────────┘ │
    │        └──────────┬─────────┘                 │
    │                   │                           │
    │    ┌──────────────▼───────────────────┐       │
    │    │    gr::execution::GpuScheduler   │       │
    │    │    bulk() CPO dispatch            │       │
    │    └──────────────┬───────────────────┘       │
    │                   │                           │
    └───────────────────┼───────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────┐
    │           Backend Dispatch                      │
    │  ┌─────────┐ ┌──────┐ ┌────────┐ ┌─────────┐  │
    │  │  SYCL   │ │ CUDA │ │ ROCm   │ │GL Comp. │  │
    │  │(Adaptive│ │(fut.)│ │ (fut.) │ │(WASM)   │  │
    │  │  Cpp)   │ │      │ │        │ │         │  │
    │  └─────────┘ └──────┘ └────────┘ └─────────┘  │
    └────────────────────────────────────────────────┘
```

---

## 6. `gr::execution` — P2300 Migration Layer

### 6.1 Purpose

Thin sender/scheduler layer using P2300 names. Wraps existing thread pool infrastructure.
Provides `GpuScheduler` for heterogeneous dispatch. Migration to `std::execution` is a namespace
swap when compiler support lands.

### 6.2 Core concepts (mirror P2300)

```cpp
namespace gr::execution {

template <typename... Sigs> struct completion_signatures {};

struct set_value_t  { template<typename R, typename... Vs> void operator()(R&&, Vs&&...) const; };
struct set_error_t  { template<typename R, typename E>     void operator()(R&&, E&&) const; };
struct set_stopped_t{ template<typename R>                  void operator()(R&&) const; };

inline constexpr set_value_t   set_value{};
inline constexpr set_error_t   set_error{};
inline constexpr set_stopped_t set_stopped{};

template <typename S> concept sender = requires { typename std::remove_cvref_t<S>::completion_signatures; };
template <typename R> concept receiver = /* ... */;
template <typename O> concept operation_state = requires(O& o) { { o.start() } -> std::same_as<void>; };
template <typename S> concept scheduler = requires(S&& s) { { std::forward<S>(s).schedule() } -> sender; };

} // namespace gr::execution
```

### 6.3 Required algorithms (8 total)

| Algorithm   | Signature                                  | Purpose                            |
|-------------|--------------------------------------------|------------------------------------|
| `just`      | `Vs... → sender<Vs...>`                    | value injection                    |
| `schedule`  | `scheduler → sender<>`                     | entry point on scheduler's context |
| `then`      | `sender<Vs...>, f → sender<R>`             | monadic map                        |
| `let_value` | `sender<Vs...>, f → sender<Us...>`         | dynamic sender chaining            |
| `when_all`  | `sender..., → sender<all...>`              | concurrent fan-out                 |
| `transfer`  | `sender<Vs...>, scheduler → sender<Vs...>` | context hop (H2D / D2H)            |
| `bulk`      | `sender<Vs...>, Shape, f → sender<Vs...>`  | data-parallel (GPU bridge)         |
| `sync_wait` | `sender<T> → optional<tuple<T>>`           | blocking terminal                  |

All are CPOs. `bulk` and `transfer` must be customisable by `GpuScheduler`.
Pipe syntax: `sender | then(f) | bulk(N, g)`.

Implemented in `core/include/gnuradio-4.0/execution/execution.hpp` (~500 lines). No external
dependency. Implemented within the existing thread pool header infrastructure.

### 6.4 `PoolScheduler`

Wraps `TaskExecutor` as a P2300 scheduler. `schedule()` returns a sender that completes on the
pool's threads.

```cpp
inline PoolScheduler cpuScheduler() { return PoolScheduler{Manager::defaultCpuPool()}; }
inline PoolScheduler ioScheduler()  { return PoolScheduler{Manager::defaultIoPool()};  }
```

### 6.5 `GpuScheduler`

```cpp
namespace gr::execution {

enum class GpuBackend { SYCL, CUDA, ROCm, GLCompute, CPU_Fallback };

class GpuScheduler {
    GpuBackend _backend;
    void*      _nativeHandle; // sycl::queue*, CUstream, GL context, nullptr

public:
    auto schedule() const -> /* GpuScheduleSender */;
    GpuBackend backend() const noexcept;
    void*      nativeHandle() const noexcept;
    // bulk() CPO override dispatches to backend-specific parallel execution
};

} // namespace gr::execution
```

`bulk` dispatches per backend:

- **SYCL**: `sycl::queue::parallel_for(sycl::range<1>(shape), ...)`
- **CUDA**: `kernel<<<grid,block>>>(...)`
- **GL Compute**: `glDispatchCompute(shape/wg, 1, 1)`
- **CPU fallback**: `std::for_each(std::execution::par_unseq, ...)`

### 6.6 Cross-context pipeline (conceptual)

```cpp
auto cpu = gr::execution::cpuScheduler();
auto gpu = gr::execution::GpuScheduler{GpuBackend::SYCL, &syclQueue};

auto pipeline = gr::execution::schedule(cpu)
    | gr::execution::then([] { return loadSamples(); })
    | gr::execution::transfer(gpu)
    | gr::execution::bulk(N, [](std::size_t i, auto& buf) { buf[i] *= 2.0f; })
    | gr::execution::transfer(cpu)
    | gr::execution::then([](auto r) { publishResult(r); });

gr::execution::sync_wait(std::move(pipeline));
```

### 6.7 Migration path

| Phase                  | Action                                                               |
|------------------------|----------------------------------------------------------------------|
| Now                    | `gr::execution` with 8 algorithms + `PoolScheduler` + `GpuScheduler` |
| Compiler support lands | CMake option `GR_USE_STD_EXECUTION`: aliases → `std::execution::`    |
| Long term              | `PoolScheduler` / `GpuScheduler` model `std::execution::scheduler`   |

Existing `pool.execute(callable)` API unchanged — senders are opt-in for new code.

---

## 7. Memory Model

### 7.1 Existing PMR Infrastructure (already in place)

The codebase is **already fully PMR-plumbed**:

| Component              | Allocator                                                                            | Location                  |
|------------------------|--------------------------------------------------------------------------------------|---------------------------|
| `CircularBuffer<T>`    | `std::pmr::polymorphic_allocator<T>` + `double_mapped_memory_resource`               | `CircularBuffer.hpp:231`  |
| `EdgeParameters`       | `dataResource` + `tagResource` (both `std::pmr::memory_resource*`) + `ComputeDomain` | `BlockModel.hpp:63-71`    |
| `Port`                 | Creates buffers with edge's PMR resources                                            | `Port.hpp:857,862`        |
| `Tag` / `property_map` | `pmt::Value::Map` = `std::pmr::unordered_map<std::pmr::string, Value>`               | `Tag.hpp:29,76`           |
| `Tensor<T>`            | `gr::pmr::vector<T, managed>`                                                        | `Tensor.hpp:199`          |
| `pmt::Value`           | Stores `std::pmr::memory_resource*`, all constructors accept it                      | `Value.hpp:159,180`       |
| `ComputeDomain`        | Provider → `memory_resource*`, `BoundDomain` + `bind()`                              | `ComputeDomain.hpp:49,93` |

The chain is: `ComputeDomain` → `ComputeRegistry::resolve()` → `memory_resource*` →
`EdgeParameters` → `Port::resizeBuffer()` → `CircularBuffer(size, allocator)`.

**What's missing:** A `gpu::UsmMemoryResource : std::pmr::memory_resource` that wraps
`sycl::malloc_shared` / `sycl::free`, registered with `ComputeRegistry` for `"gpu:..."` domains.
Once registered, `CircularBuffer`, `Tag`, `Tensor`, and `Value` all get device-accessible memory
automatically through the existing plumbing.

### 7.2 CPU edges — unchanged

`CircularBuffer<T>` with atomic counters (`gr::AtomicRef`). Lock-free single-writer/multi-reader.
Completely unmodified for CPU-only graphs.

### 7.3 Atomic counters and USM — confirmed incompatible

The same atomic variable cannot be coherently accessed from CPU and GPU simultaneously. This
confirms that `CircularBuffer` cannot transparently bridge host↔device. Explicit data transfer
is always required at CPU↔GPU boundaries.

### 7.4 Host↔Device transfers

**Explicit path** (recommended for production): User places `HostToDevice<T>` / `DeviceToHost<T>`
blocks in the graph. These control batch/DMA size for throughput vs. latency trade-off.

```cpp
gr::Graph flow;
auto& src    = flow.emplaceBlock<SignalSource<float>>();
auto& h2d    = flow.emplaceBlock<gpu::HostToDevice<float>>({{"chunk_size", 4096}});
auto& mul    = flow.emplaceBlock<MultiplyConst<float>>({{"factor", 2.0f}, {"compute_domain", "gpu"}});
auto& d2h    = flow.emplaceBlock<gpu::DeviceToHost<float>>();
auto& sink   = flow.emplaceBlock<DataSink<float>>();
flow.connect<"out", "in">(src, h2d);
flow.connect<"out", "in">(h2d, mul);    // device edge
flow.connect<"out", "in">(mul, d2h);    // device edge
flow.connect<"out", "in">(d2h, sink);
```

**Implicit path** (single GPU block without transition blocks): If the scheduler encounters a GPU
block without surrounding `HostToDevice` / `DeviceToHost`, it handles the transfer internally with
default chunk size. Data is transferred to device and back per invocation.

### 7.5 Device edges (Phase 2)

`DeviceBuffer<T>` — USM device allocation, no atomics, synchronisation via SYCL events. Used
between consecutive GPU blocks within a sub-graph. Not a `CircularBuffer`.

### 7.6 `DeviceBlockState<T>`

Generated via compile-time reflection over `GR_MAKE_REFLECTABLE` fields:

- **Scalar fields** (`float`, `int`, etc.): trivially copied to device memory.
- **`std::pmr::vector<T>` fields** (with USM-backed allocator): data already in device-accessible
  memory. Mirrored as `{T* ptr, size_t size}` in the device state struct.
- **Non-PMR containers** (`std::vector<T>`): not supported in auto-parallelisation path. Blocks
  with such state must use a backend-specific escape hatch (e.g. `processBulk_sycl`).

**Requirement:** GPU-eligible blocks with array/container settings must use `std::pmr::vector<T>`.

### 7.7 DeviceContext hierarchy (V3 refactor)

The original `DeviceContext` was a single struct with SYCL-specific members behind `#if` guards.
V3+ refactors this into a virtual base class with backend-specific implementations. Users can
implement their own backends (e.g. CUDA, ROCm) by subclassing `DeviceContext`.

```cpp
namespace gr::device {

enum class DeviceType { CPU, GPU, FPGA, Accelerator };

struct DeviceContext {
    virtual ~DeviceContext() = default;

    virtual DeviceBackend backend() const noexcept = 0;
    virtual DeviceType    deviceType() const noexcept = 0;
    virtual std::string   shortName() const = 0;  // "CPU", "SYCL:RTX 3070", "GLSL:RTX 3070"
    virtual std::string   name() const = 0;        // "NVIDIA GeForce RTX 3070"
    virtual std::string   version() const = 0;     // "OpenGL 4.3 NVIDIA 595.45.04"
    bool                  isGpu() const noexcept { return deviceType() == DeviceType::GPU; }

    template<typename T> T* allocateDevice(std::size_t count);
    template<typename T> T* allocateHost(std::size_t count);
    template<typename T> T* allocateShared(std::size_t count);
    template<typename T> void deallocate(T* ptr);

    virtual void copyHostToDevice(const void* host, void* device, std::size_t bytes) = 0;
    virtual void copyDeviceToHost(const void* device, void* host, std::size_t bytes) = 0;
    virtual void wait() = 0;

    virtual void* allocateDeviceRaw(std::size_t bytes, std::size_t alignment) = 0;
    virtual void* allocateHostRaw(std::size_t bytes, std::size_t alignment) = 0;
    virtual void* allocateSharedRaw(std::size_t bytes, std::size_t alignment) = 0;
    virtual void  deallocateRaw(void* ptr) = 0;
};

struct DeviceContextCpu  final : DeviceContext { /* shortName()="CPU", heap alloc, memcpy */ };
struct DeviceContextSycl final : DeviceContext { /* shortName()="SYCL:RTX 3070" or "SYCL:CPU", USM, parallelFor */ };
struct DeviceContextGLSL final : DeviceContext { /* shortName()="GLSL:RTX 3070", persistent SSBOs, dispatch */ };
// future: DeviceContextCUDA, DeviceContextROCm, DeviceContextWebGPU

} // namespace gr::device
```

**Important**: derived classes must declare `using DeviceContext::copyHostToDevice;` and
`using DeviceContext::copyDeviceToHost;` to un-hide the typed template wrappers from the base
class (C++ name-hiding with virtual methods + non-virtual templates).

Backend-specific operations (SYCL's `parallelFor`, GLSL's `dispatch(shader, ...)`) are accessed
through the concrete type via `dynamic_cast` or by the `ExecutionStrategy` which knows the backend
at compile time via block traits.

### 7.8 DeviceBuffer — RAII + shared ownership

```cpp
struct DeviceBuffer {
    void*          data = nullptr;
    std::size_t    size = 0;          // bytes
    DeviceContext* ctx  = nullptr;

    ~DeviceBuffer() { if (ctx && data) ctx->deallocateRaw(data); }

    DeviceBuffer(DeviceBuffer&&) noexcept;             // move-only by default
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
};

// for shared buffers between fused blocks:
using SharedDeviceBuffer = std::shared_ptr<DeviceBuffer>;
```

RAII is the default (block owns its buffers). When the scheduler fuses adjacent blocks into a
single shader dispatch, the intermediate buffers are replaced by a `SharedDeviceBuffer` that both
blocks reference. The `shared_ptr` ensures the buffer lives until the last block releases it.

A `DeviceBufferRegistry` tracks all live buffers per context for:

- lifetime validation (detect use-after-free in debug builds)
- false-sharing detection (warn if two blocks write overlapping regions)
- total device memory accounting

### 7.9 Shader fusion architecture

When adjacent blocks in a graph all provide GLSL/WGSL shader descriptions, the scheduler fuses
them into a single compute dispatch. This eliminates intermediate buffer round-trips.

**Block shader interface:**

```cpp
struct ShaderFragment {
    std::string              glslFunction;    // "float process(float x) { return x * GAIN; }"
    std::vector<ShaderConst> constants;       // {{"GAIN", gain}, {"OFFSET", offset}}
    std::size_t              inputChunkSize;  // 0 = element-wise, N = requires N-sample chunks
    std::size_t              outputChunkSize; // usually same as input
};

// blocks implement:
concept HasShaderFragment = requires(const TBlock& b) {
    { b.shaderFragment() } -> std::convertible_to<ShaderFragment>;
};
```

**Fusion rules:**

1. **Element-wise chain** (all blocks have `inputChunkSize == 0`): functions are composed inline.
   `outBuf.data[i] = processC(processB(processA(inBuf.data[i])));` — one dispatch, no barriers.

2. **Bulk block in chain** (e.g. FFT with `inputChunkSize == N`): the scheduler chunks the input
   as multiples of N. The fused shader becomes a multi-stage dispatch:
    - Stage 1: element-wise pre-processing (fused) over k×N samples
    - Barrier (separate dispatch)
    - Stage 2: bulk operation (FFT) over k chunks of N samples
    - Barrier
    - Stage 3: element-wise post-processing (fused)

3. **Settings change**: invalidates the fused shader. The `ShaderCache` (hash-keyed by the
   concatenation of all fragment sources + constants) recompiles on next dispatch.

**Dispatch backends — two paths, one shader model:**

Blocks provide GLSL `process()` functions via `ShaderFragment`. The runtime wraps these in
the appropriate dispatch mechanism depending on the platform:

```
ShaderFragment (same for all platforms)
     │
     ├──[native GL 4.3+]──→ compute shader + SSBOs (fast, current implementation)
     ├──[native GL 3.0+]──→ transform feedback (fallback when GL < 4.3)
     └──[WASM / WebGL2] ──→ transform feedback (ES 3.0, works in all browsers)
```

- **Native (GL 4.3+)**: compute shaders + SSBOs via EGL headless. Fast. Tested in CI via
  Mesa llvmpipe. This is the current implementation in `GlComputeContext`.
- **Native (GL 3.0+) / WASM (WebGL2)**: transform feedback GPGPU pattern. Input data in a
  vertex buffer, vertex shader applies `process()`, output captured via transform feedback
  into another buffer. No compute shaders, no SSBOs — works on OpenGL ES 3.0 / WebGL2.
  The `process()` function body is identical; only the boilerplate wrapper changes.
- **Future (WebGPU)**: `GLSL2WGSL` transpiler converts fragments to WGSL for WebGPU compute
  shaders. Available in Chrome + Node.js 24 (`--experimental-webgpu`). Deferred until
  WebGPU is universally available.

SSBOs: available on native (GL 4.3+) only. NOT in browsers (WebGL2 = ES 3.0) or Node.js.
The `GlComputeContext` auto-detects GL version and selects compute vs transform feedback.

**WASM init path:** `#ifdef __EMSCRIPTEN__` uses `emscripten_webgl_create_context()` instead
of EGL. Same GL ES 3.0 API surface, same transform feedback dispatch.

**CI testing:**

- **Native (Linux)**: Mesa llvmpipe via EGL → compute shaders + SSBOs (full GPU dispatch).
  This is the primary e2e test for the GLSL path.
- **WASM (Node.js 24)**: shader logic only (ShaderFragment, ShaderFusion, ShaderCache,
  GLSL2WGSL transpilation). Verified: compiles clean, 2171 asserts pass. No GPU dispatch
  (Node.js has no WebGPU adapter without browser context).
- **WASM (Emscripten + Dawn WebGPU)**: compiles with `--use-port=emdawnwebgpu`.
  Adapter available in browser only (Chrome, Edge, Firefox, Safari). Runtime testing via
  browser: load the Emscripten HTML output in Chrome to exercise WebGPU compute dispatch.
  Headless browser CI (Playwright/Puppeteer) deferred — manual browser testing for now.

**Compile flag:** `GR_DEVICE_HAS_GL` replaces `GR_DEVICE_HAS_GL_COMPUTE`. ON for both
native (EGL available) and WASM (Emscripten). Guards the `GlComputeContext` and
`DeviceContextGLSL` code.

**DeviceContext naming API:**

```cpp
struct DeviceContext {
    virtual std::string shortName() const = 0;  // "CPU", "SYCL:RTX3070", "GLSL:llvmpipe"
    virtual std::string name() const = 0;        // "NVIDIA GeForce RTX 3070"
    virtual std::string version() const = 0;     // "OpenGL 4.3 NVIDIA 595.45.04"
};
```

Used by benchmarks and device info printing. No hard-coded backend names.

---

## 8. `gr::complex<T>`

### 8.1 Motivation

`std::complex<float>` member functions do not compile through AdaptiveCpp's SSCP flow. The PoC
required a custom `complexf` struct. A project-wide `gr::complex<T>` solves this permanently.

### 8.2 Design

```cpp
namespace gr {

template<typename T>
struct complex {
    T re{}, im{};

    constexpr complex() = default;
    constexpr complex(T r, T i = T{}) : re(r), im(i) {}

    // implicit conversion from/to std::complex<T> (layout-compatible)
    constexpr complex(const std::complex<T>& c) : re(c.real()), im(c.imag()) {}
    constexpr operator std::complex<T>() const { return {re, im}; }

    constexpr T real() const { return re; }
    constexpr T imag() const { return im; }

    friend constexpr complex operator+(complex a, complex b) { return {a.re+b.re, a.im+b.im}; }
    friend constexpr complex operator-(complex a, complex b) { return {a.re-b.re, a.im-b.im}; }
    friend constexpr complex operator*(complex a, complex b) {
        return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
    }
    // ... division, abs, norm, conj, etc.
};

} // namespace gr
```

Properties:

- `constexpr`, trivially copyable, works on all backends (SYCL, GL Compute, CPU).
- Layout-compatible with `std::complex<T>` (two consecutive `T`s) — `reinterpret_cast` safe.
- Struct → reflectable/inspectable.
- `vir::simdize<gr::complex<T>>` specialisation needed for SIMD compatibility.

### 8.3 Integration with existing type system

`std::complex<T>` is hard-wired at 6 points in the type system. The table below documents each
integration point and what's needed for `gr::complex<T>` compatibility.

| # | Location | Current pattern | `gr::complex<T>` status | Fix |
|---|---|---|---|---|
| 1 | `meta::complex_like` (utils.hpp:682) | Closed list: `std::complex<float/double>` | Not matched | Extend concept (structural or explicit) |
| 2 | `pmt::detail::is_complex<T>` (PmtTypeHelpers.hpp:41) | Partial spec for `std::complex<T>` | Not matched | Add partial spec for `gr::complex<T>` |
| 3 | `Value::ValueScalarType` (Value.hpp:57) | Closed list | Implicit conversion works for ctor; `get_if<gr::complex<T>>` fails | Low priority — convert at boundary |
| 4 | `UncertainValue` (UncertainValue.hpp:126) | Uses `std::real()` / `std::imag()` free functions | Not found via ADL | Add `real()` / `imag()` free functions in `namespace gr` |
| 5 | `TensorMath::real/imag/conj` (TensorMath.hpp:590) | SFINAE: `std::is_same_v<U, std::complex<...>>` | Not matched | Extend guard to include `gr::complex<T>` |
| 6 | `TensorMath::squaredMagnitude` (TensorMath.hpp:731) | Uses `std::norm(val)` | Not found via ADL | `gr::norm()` already in Complex.hpp, ADL finds it |

**Minimal fixes applied to `Complex.hpp` (Phase 1):**
1. Add `using value_type = T;` — enables trait introspection, matches `std::complex<T>::value_type`
2. Add `real()` / `imag()` free functions in `namespace gr` for ADL compatibility with
   `UncertainValue`'s `std::real()` / `std::imag()` calls

**Deferred to Phase 2 (full `gr::complex` migration):**
- Make `complex_like` a structural concept (`.real()` + `.imag()` returning floating-point)
- Extend `Value::ValueScalarType` to include `gr::complex<float/double>`
- Extend `TensorMath` SFINAE guards
- Extend `is_complex` trait

**Design rationale:** For Phase 1, `std::complex<T>` remains the canonical type in `Value`,
`Tensor`, `Settings`, and the tag system. `gr::complex<T>` is used in block port types and
GPU device code. The implicit conversion between them handles boundaries. This avoids
doubling the type dispatch complexity in `Value` and `ValueHelper`.

**Future consideration:** A structural `complex_like` concept would be cleaner than the current
closed type list. Both `std::complex<T>` and `gr::complex<T>` have `.real()` and `.imag()`
methods, so a structural concept eliminates the need for explicit listing:
```cpp
template<typename T>
concept complex_like = requires(const T& z) {
    { z.real() } -> std::floating_point;
    { z.imag() } -> std::floating_point;
} && std::is_trivially_copyable_v<T>;
```
This also naturally supports any future complex-like types without modification.

### 8.4 Migration

Phase 1 targets `float` only. `gr::complex<float>` is introduced but only migrated to critical
blocks (FFT, FIR, IQ processing). Other blocks continue using `std::complex<T>` on CPU.

---

## 9. WebGPU/WGSL Compute and Runtime Shader Integration

### 9.1 Scope

WebGPU (WGSL shaders) is the GPU compute backend for WASM targets. WebGL 2.0 does not support
compute shaders (based on OpenGL ES 3.0). WebGPU is stable in Chrome, Firefox, and Safari as of

2025. Emscripten supports it via Dawn bindings.

For native desktop, OpenGL 4.3+ compute shaders (GLSL) are an alternative when SYCL is
unavailable. Both shader backends share the same infrastructure.

### 9.1a GL Compute Backend (implemented, Phase 1)

`device::GlComputeContext` provides headless OpenGL 4.3 compute shader execution via EGL.
Works without a display server (CI, Docker, SSH). Uses Mesa llvmpipe for software rendering
when no GPU is available, or delegates to hardware (NVIDIA, AMD, Intel) when present.

**CMake integration:** `GR_HAS_GL_COMPUTE` detected via pkg-config for EGL + OpenGL. Status
reported alongside `GR_USE_ADAPTIVE_CPP` in CMake output. Docker image includes
`libegl1-mesa-dev libgl-dev`.

**Escape hatch signature:**
```cpp
static device::ShaderDispatchInfo processBulk_glsl(const property_map& settings);
```

Returns GLSL source + workgroup size. The framework compiles, caches (keyed by source hash),
binds input/output SSBOs from port data, and dispatches via `glDispatchCompute`.

**Platform support:**

| Platform | GL compute? | Driver | Notes |
|---|---|---|---|
| Linux desktop (NVIDIA/AMD/Intel) | Yes | hardware | via EGL headless |
| Linux CI / Docker | Yes | Mesa llvmpipe | software, `libegl1-mesa-dev` |
| Raspberry Pi 4/5 | Yes | Mesa V3D | GLES 3.1 (`#version 310 es`) |
| Raspberry Pi 3/Zero | No | VideoCore IV | GLES 2.0 only, CPU fallback |
| macOS | No | deprecated GL 4.1 | no compute shaders |
| WASM | No | WebGL 2.0 | use WebGPU instead |

**Concept:**
```cpp
template<typename Derived>
concept HasShaderGenerator = requires(const gr::property_map& settings) {
    { Derived::processBulk_glsl(settings) };
};
```

### 9.2 Three levels of shader integration

| Level                              | Capability                                                        | Phase             |
|------------------------------------|-------------------------------------------------------------------|-------------------|
| 1. Static shader escape hatch      | Block provides `processBulk_glsl(settings)` returning GLSL source | Phase 1 (done)    |
| 2. Runtime-specialised shaders     | Block generates shader from settings, framework caches/recompiles | Phase 1 (done)    |
| 3. Auto-generation from processOne | Build-time transpilation of C++ → WGSL/GLSL                       | Phase 2           |

**Level 2 is the key differentiator** — shader backends can JIT-specialise kernels in ways that
SYCL (build-time SSCP) cannot. This is particularly powerful for FFT, FIR, and other blocks
where structural parameters (size, tap count) change infrequently.

### 9.3 Runtime shader specialisation design

Settings split into two categories per block:

| Category            | Behaviour                                           | Example                            |
|---------------------|-----------------------------------------------------|------------------------------------|
| **Baked constants** | Embedded in shader source; recompile on change      | `fft_size`, `tap_count`, `inverse` |
| **Uniforms**        | Passed via uniform buffer; update without recompile | `scale_factor`, `threshold`        |

The block author decides which settings are baked vs. uniform via a trait:

```cpp
template<typename TBlock>
concept ShaderCapable = requires(const property_map& settings) {
    { TBlock::generateShader(settings) } -> std::convertible_to<std::string>;
};

// Example: FFT block shader generation
template<typename T>
struct Fft : gr::Block<Fft<T>> {
    // ... ports, settings ...

    // generates WGSL with baked FFT size, unrolled stages, embedded twiddle LUT
    static std::string generateShader(const property_map& settings) {
        auto N = settings.at("fft_size").as<std::size_t>();
        auto nStages = std::countr_zero(N);
        return std::format(R"(
            const N: u32 = {N}u;
            const nStages: u32 = {nStages}u;
            // ... baked twiddle factors as const arrays ...
            @compute @workgroup_size(256)
            fn fftStage(@builtin(global_invocation_id) gid: vec3u) {{
                // ... unrolled butterfly stages for this specific N ...
            }}
        )", fmt::arg("N", N), fmt::arg("nStages", nStages));
    }
};
```

**Advantages of runtime specialisation:**

- Twiddle factors embedded as shader constants (no LUT indirection, optimal register usage)
- Butterfly loops unrolled for specific N (no loop overhead)
- Dead code elimination by the shader compiler for specific configurations
- Approaches cuFFT-level performance for fixed sizes
- Shader recompilation is milliseconds — acceptable when `fft_size` changes

### 9.4 Shader cache

```cpp
namespace gr::gpu {

struct ShaderKey {
    std::size_t sourceHash;
    GpuBackend  backend;
    bool operator==(const ShaderKey&) const = default;
};

class ShaderCache {
    std::unordered_map<ShaderKey, /* compiled program handle */> _cache;
public:
    auto getOrCompile(std::string_view source, GpuBackend backend) -> /* handle */;
    void invalidate(ShaderKey key);  // called when baked settings change
};

} // namespace gr::gpu
```

When `settingsChanged` fires and a baked setting was modified, the block's cached shader is
invalidated. On the next dispatch, `generateShader()` is called with the new settings and the
result is compiled and cached.

### 9.5 Dispatch flow for shader backends

When `GpuScheduler` (WebGPU/GL backend) dispatches a block:

1. **`ShaderCapable` block?** Call `TBlock::generateShader(currentSettings)` → shader source
2. **Hash source**, look up in `ShaderCache`
3. **Cache miss**: compile shader (WebGPU: `device.createComputePipeline()`; GL: `glCompileShader`)
4. **Bind uniforms** from non-baked settings → uniform buffer
5. **Bind data buffers** (input/output spans) → storage buffers
6. **Dispatch**: `pass.dispatchWorkgroups(N/wg, 1, 1)` (WebGPU) or `glDispatchCompute(...)` (GL)

### 9.6 Kernel-as-struct pattern (from TinyCompute)

Source: Koen Samyn, CppCon 2025 "From Pure ISO C++20 to Compute Shaders"
Repository: https://github.com/samynk/TinyCompute
Tutorial: https://github.com/samynk/ComputeShadersTutorial
(See §16.1 and §16.2 for detailed analysis and adoption roadmap.)

Pure C++ kernel description that runs on CPU directly and can drive shader generation:

```cpp
struct FloatAdder {
    uvec3 local_size{ 256, 1, 1 };
    BufferBinding<float, 0> A;
    BufferBinding<float, 1> B;
    BufferBinding<float, 2> C;
    void main() { auto i = gl_GlobalInvocationID.x; C[i] = A[i] + B[i]; }
};
```

TinyCompute uses a CRTP `ComputeBackend<Derived>` with `CPUBackend` (parallel algorithms,
`gl_GlobalInvocationID` as thread-local) and `GPUBackend` (Clang AST transpiler → GLSL).
`BufferBinding<T, B>` encodes the SSBO binding point in the type — `binding[i]` works
identically on CPU (vector access) and GPU (SSBO reference). `DimTraits<D>` specialises
index flattening for 1D/2D/3D dispatch at compile time.

**Transferable patterns:**

- Kernel-as-struct with typed bindings (pure ISO C++, works on GCC)
- `DimTraits` for 1D/2D/3D coordinate flattening (FFT + waterfall/spectrogram)
- `PixelConcept` + `GPUFormatTraits` (spectrum → texture colourmap conversion)
- CPU/GPU equivalence via backend swap (same `main()` for testing)
- Memory barriers between compute passes (from ComputeShadersTutorial — separable filters)
- Querying `GL_COMPUTE_WORK_GROUP_SIZE` post-compilation (adaptive dispatch)

**What we do NOT adopt:** the Clang AST transpiler (too heavy; we use `ShaderFragment` strings),
swizzle system (niche), `PixelConcept` (deferred to visualisation phase).

**Phase 2 goal:** kernel-as-struct becomes the common description for Level 3 auto-generation.
On CPU, `main()` runs directly. On SYCL, it compiles natively. On WebGPU/GL, `generateShader()`
produces WGSL/GLSL from the struct's bindings and logic.

### 9.7 Missing primitives for FFT shaders

TinyCompute lacks shared/local memory and barrier primitives. Must be added for the FFT local-
memory phase:

- `SharedMemory<T, N>` → `shared` (GLSL), `var<workgroup>` (WGSL), `sycl::local_accessor` (SYCL)
- `barrier()` → `barrier()` (GLSL), `workgroupBarrier()` (WGSL), `sycl::group_barrier()` (SYCL)

---

## 10. FFT Implementations

### 10.1 Algorithm: Van Loan Stockham auto-sort

All GPU paths (SYCL + GLSL) use the **Van Loan Stockham** radix-2 algorithm (Computational
Frameworks for the FFT, Ch. 2). Key properties:

- **No bit-reversal pass** — output is naturally in DFT order
- **Sequential reads** (`srcLo = j`), interleaved writes (`dstLo = group * Ls + k`)
- **Twiddle**: `w = tw[k * (N/Ls)]` indexes the base N/2-entry table directly
- **Butterfly**: `dst[lo] = a + w*b; dst[hi] = a - w*b` (twiddle applied to `b` before sum)
- Ping-pong buffers (out-of-place), negligible cost vs GPU VRAM

**CPU path**: delegates to `gr::algorithm::FFT` (SimdFFT, mixed-radix {2,3,4,5} with SIMD).
The `SyclFFT::forwardStockhamCpu()` method provides a CPU-side Van Loan implementation
(sequential scalar) for direct algorithm testing against SimdFFT.

**Implementations**:

- `SyclFFT` (`algorithm/fourier/SyclFFT.hpp`): GPU via SYCL, CPU via SimdFFT delegation
- `GlslFFT` (`algorithm/fourier/GlslFFT.hpp`): GPU via GLSL compute shaders
- Both share the same Van Loan index mapping, tested against SimdFFT

The algorithm choice is internal to `SyclFft`; the public API (`forward`, `forwardBatch`) is
unchanged. The bit-reversal is handled locally (not in H2D/D2H blocks).

### 10.2 GPU execution phases (Stockham)

**Phase 1 — Global memory stages**: stages where half-span > workgroup size. One kernel per stage,
all batches fused into a single launch (`sycl::range<2>{nBatches, N/2}`). Reads from `src`
buffer, writes to `dst` buffer, then swaps pointers.

**Phase 2 — Fused local-memory stages**: single kernel covering all remaining stages. Each
workgroup loads a tile into `local_accessor`, performs butterfly stages with `group_barrier`,
writes back. Twiddle sub-table for the workgroup loaded into shared memory once.

All stages chained via `sycl::event` dependencies — single `event.wait()` at the end.

Transition: `sLocal = max(0, log₂N - log₂(2 × wgSize))`.

### 10.3 Kernel configuration (`FftKernelParams`)

Following the `gr::math::config::KernelParams<T>` pattern from `TensorMath.hpp`:

```cpp
namespace gr::device::config {
struct FftKernelParams {
    std::size_t workgroupSize = 256;   // device-dependent: 256 (NVIDIA), 64 (AMD RDNA), 256 (Intel Arc)
    std::size_t radixPerThread = 4;    // register-level FFT: 2 (minimal), 4 (balanced), 8 (aggressive)
};
}
```

Default provided via `DeviceContext` (which knows the device). Users can override per-block.
The workgroup size determines the global/local stage split:

| wgSize | Shared mem (data+tw) | Local stages | Global stages (N=65536) | Launches |
|--------|----------------------|--------------|-------------------------|----------|
| 256    | 6 KB                 | 9            | 7                       | 8        |
| 512    | 12 KB                | 10           | 6                       | 7        |
| 1024   | 24 KB                | 11           | 5                       | 6        |

All fit within the 48 KB/SM shared-memory limit on current hardware.

### 10.4 Twiddle factor strategy: per-stage compact sub-tables

Current: single global table of N/2 entries, accessed with stride `2^stage`. At late stages
the stride causes 32× cache-line amplification per warp.

Revised: pre-compute per-stage sub-tables `twiddleStage[s]` with `halfSpan_s` entries each,
accessed sequentially. Total memory = N-1 entries (same as before, just rearranged).
For the local-memory kernel, the workgroup's twiddle slice (≤8 KB for wgSize=1024) is loaded
into shared memory once and reused across all fused stages.

### 10.5 Kernel arithmetic: `uint32_t` + bit-shifts

**Critical finding**: the V1 DIF kernels used `std::size_t` (64-bit) for all index arithmetic
including division/modulo. On NVIDIA GA104, 64-bit integer divide is emulated (~100+ cycles)
vs the butterfly itself (~20 cycles). **96% of kernel time was integer division.**

Fix (mandatory for all GPU kernels):

- Use `uint32_t` for kernel-internal indices (max index 128×65536 = 8M fits 32 bits)
- Replace `/ halfSpan` → `>> log2(halfSpan)`, `% halfSpan` → `& (halfSpan - 1)` (1 cycle each)
- Use `sycl::range<2>{nBatches, N/2}` so batch index comes from hardware thread mapping (free)

### 10.6 Radix-4/8 per-thread (register-level FFT)

Each thread loads R elements (R=4 or 8) from global/shared memory, computes a radix-R
butterfly entirely in registers (log₂(R) stages, no memory access), then writes results back.
This fuses log₂(R) stages into every kernel, reducing global-memory passes.

Implementation plan:

1. **Radix-4 global stage**: each thread loads 4 elements from src, computes 2 butterfly
   stages in registers (using twiddle factors), writes 4 elements to dst. Replaces 2
   standard global stages with 1 radix-4 launch. For N=65536 with wgSize=1024:
   5 standard stages → 3 radix-4 launches (2+2+1) → **saves 2 kernel launches + 2 global passes**.
2. **Radix-4/8 local stage**: the fused local-memory kernel does its innermost 2–3 stages
   per-thread in registers before exchanging via shared memory. Reduces barrier count and
   shared-memory traffic by log₂(R).
3. **Sub-group shuffle** (`sycl::sub_group`): for warp-level exchange (32 threads),
   `sub_group::shuffle_xor()` replaces shared memory for ~5 stages. Test before/after —
   keep only if measurably faster (some SYCL backends implement shuffles via shared memory).

Target with wgSize=1024 + radix-4:

- 2 stages in registers per kernel
- 11 stages in shared memory (fused)
- Total in one fused kernel: 13 stages
- Remaining global stages for N=65536: 3 radix-2 or 2 radix-4
- Total launches: 2–3 (global) + 1 (fused) = **3–4** (down from 6)

### 10.7 Measured performance (RTX 3070, GA104)

**Achievable bandwidth**: 405 GB/s through AdaptiveCpp (90% of 448 GB/s theoretical).
FFT arithmetic intensity: ~0.23 FLOP/byte → **memory roofline ~93 GFLOP/s**.

Standalone butterfly throughput (5 global stages, compute-only, sustained): **82 GFLOP/s**
(88% of roofline). Including H2D transfer (PCIe 4.0): **51 GFLOP/s**.

| Version | Description                        | x128 N=65536   | % roofline | vs CPU SimdFFT |
|---------|------------------------------------|----------------|------------|----------------|
| V0      | DIF, sync .wait()                  | 2.3 GFLOP/s    | 2%         | 0.16×          |
| V1      | DIF, async events                  | 11.3 GFLOP/s   | 12%        | 0.82×          |
| V2      | Stockham, uint32                   | 12 GFLOP/s     | 13%        | 0.82×          |
| V2+10   | V2, benchmark<10>                  | 51 GFLOP/s     | 55%        | **3.8×**       |
| V3      | radix-4 + JIT warmup + SimdFFT CPU | **59 GFLOP/s** | **63%**    | **3.2×**       |

**Key findings:**

- **Integer arithmetic dominates GPU kernels.** `std::size_t` (64-bit) division costs ~100+ cycles
  on NVIDIA (emulated). The butterfly is ~20 cycles. First kernel spent 96% on index math.
  Fix: `uint32_t` + bit-shifts for power-of-2 dimensions.
- **AdaptiveCpp SSCP cold-start: ~14ms per unique kernel.** Amortises to ~70μs sustained.
  Single-invocation benchmarks are misleading — use `benchmark<10>` with JIT warmup.
- **L2 cache cliff is dramatic.** x16 N=8192 (1 MB, fits L2): 36 GFLOP/s.
  x128 N=65536 (67 MB, GDDR6): 59 GFLOP/s at 3.2× the data volume.
- **Stockham > DIF on GPU.** No bit-reversal pass, natural auto-sort, clean ping-pong.
  Keep DIF for CPU (in-place is valuable).
- **Sub-group shuffle: 1.22× for inner stages.** Validated, not integrated (complexity vs gain).
- **SimdFFT as CPU fallback.** Replaced scalar DIF with SimdFFT delegation: 2.5G → 14G GFLOP/s.

### 10.8 CPU benchmark: SyclFFT CPU path (SimdFFT delegation)

SyclFFT delegates to `gr::algorithm::FFT` (SimdFFT) on CPU. No separate scalar implementation.

| Batch | N     | CPU SimdFFT (direct) | CPU via SyclFFT | Ratio |
|-------|-------|----------------------|-----------------|-------|
| x1    | 4096  | 15.9 GFLOP/s         | 8.9 GFLOP/s     | 1.8×  |
| x128  | 4096  | 21.2 GFLOP/s         | 13.4 GFLOP/s    | 1.6×  |
| x128  | 65536 | 18.2 GFLOP/s         | 8.7 GFLOP/s     | 2.1×  |

The 1.6-2× overhead is from the out-of-place copy (SimdFFT writes to a separate buffer, SyclFFT
copies back). Acceptable — the CPU path is a correctness fallback, not the primary target.

### 10.9 Benchmark

**File**: `algorithm/benchmarks/bm_FFT_backends.cpp` (zero `#if` macros, runtime backend discovery).
Backends auto-discovered via `FFTBackend` struct with `init`/`compute` lambdas.
`bm_FFT_backends_helpers.cpp` isolates SYCL headers (SSCP deadlock workaround).

Discovered backends (acpp build, RTX 3070): `SimdFFT`, `SyclFFT:CPU`, `SYCL:NVIDIA GeForce RTX 3070`,
`SYCL:CPU` (OpenMP), `GLSL:NVIDIA GeForce RTX 3070`.

Benchmark reproduction: run 3× per compiler, keep best-of-3 per entry. See memory note
`project_fft_benchmark_table.md` for the exact procedure.

**Note**: GPU numbers are sensitive to concurrent GPU load. Re-run with idle GPU.

### 10.10 Test coverage

**File**: `blocks/fourier/test/qa_FFT2.cpp` — 7 suites, 37 tests, 62 asserts, exit 0.

| Suite                        | Tests | What                                                                   |
|------------------------------|-------|------------------------------------------------------------------------|
| `FFT CPU SimdFFT`            | 3     | SimdFFT reference (forward, inverse, sizes)                            |
| `FFT SyclFFT:CPU`            | 3     | SyclFFT CPU fallback (SimdFFT delegation)                              |
| `FFT SyclFFT:CPU vs SimdFFT` | 11    | Full numerical comparison (9 param combos + batched + round-trip)      |
| `FFT Stockham vs SimdFFT`    | 16    | **Van Loan Stockham algorithm** vs SimdFFT for N∈{16,64,256,1024,4096} |
| `FFT GlslFFT`                | 2     | GLSL GPU Stockham (peak-bin + magnitude)                               |
| `FFT2 graph integration`     | 2     | Block-level graph: Source→FFT2→Sink                                    |

The `FFT Stockham vs SimdFFT` suite calls `forwardStockhamCpu()` — the exact same Van Loan
butterfly code that runs on GPU via SYCL and GLSL kernels. No GPU hardware needed.

**Known issue**: GLSL cross-test with full `maxError` comparison fails due to GL state
contamination between test suites (eglTerminate ordering). The standalone `FFT GlslFFT`
suite passes correctly. Fix: leaked GL singleton for the standalone suite prevents
premature `eglTerminate`.

### 10.11 Production gaps

- R2C/C2R optimisation for real-valued input
- Rader's/Bluestein's for non-power-of-2 lengths
- Async H2D/compute/D2H triple-buffered pipeline for streaming workloads
- Sub-group shuffle integration for inner local-memory stages (+22%)

---

## 11. Transition Blocks

### 11.1 `HostToDevice<T>`

```cpp
template<typename T>
struct HostToDevice : gr::Block<HostToDevice<T>> {
    using Description = Doc<"transfers samples from host CircularBuffer to device memory">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    Annotated<gr::Size_t, "min chunk size"> min_chunk_size = 1024UZ;
    Annotated<gr::Size_t, "max chunk size"> max_chunk_size = 65536UZ;
    Annotated<gr::Size_t, "chunk size">     chunk_size     = 4096UZ;

    GR_MAKE_REFLECTABLE(HostToDevice, in, out, min_chunk_size, max_chunk_size, chunk_size);

    gr::work::Status processBulk(const gr::ConsumableSpan auto& inSpan,
                                 gr::PublishableSpan auto& outSpan);
};
```

Adaptive chunking: throughput estimator doubles/halves `chunk_size` within bounds.

### 11.2 `DeviceToHost<T>`

```cpp
template<typename T>
struct DeviceToHost : gr::Block<DeviceToHost<T>> {
    using Description = Doc<"transfers samples from device memory back to host CircularBuffer">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(DeviceToHost, in, out);

    gr::work::Status processBulk(const gr::ConsumableSpan auto& inSpan,
                                 gr::PublishableSpan auto& outSpan);
};
```

### 11.3 Tag handling at boundaries

`HostToDevice` scans incoming tags for keys matching reflected fields of downstream GPU blocks.
If a settings-affecting tag is found at sample index N:

1. Chunk 1: samples 0..N-1, launched with current settings.
2. Settings update: `sync_state_to_device(...)` for affected block(s).
3. Chunk 2: samples N..end, launched with new settings.

Informational tags (no matching keys) are forwarded without splitting.

---

## 12. Staged Implementation Plan

Each stage produces one or more commits (only when authorised). Every stage is independently
testable. Each stage ends with a **review gate** (review + Q&A + explicit "go ahead").

All effort estimates are **AI-assisted** (Claude Code implements, user guides/reviews).

---

### Stage 0 — `gr::complex<T>` and `gr::execution` foundation (9d)

#### Step 0.1 — `gr::complex<T>` type (2d)

**File**: `core/include/gnuradio-4.0/Complex.hpp`
**Functionality**:

- `gr::complex<T>` struct: `{T re, im}`, constexpr, trivially copyable
- Arithmetic: `+`, `-`, `*`, `/`, `abs`, `norm`, `conj`, `polar`
- Implicit conversion to/from `std::complex<T>` (layout-compatible)
- `vir::simdize<gr::complex<T>>` specialisation

**Quality gate**: Compiles on GCC 15 + Clang 20. No warnings with `-Werror`.

#### Step 0.2 — `qa_Complex.cpp` (1d)

**Tests**:

- `"gr::complex round-trips with std::complex"` — construct from `std::complex`, convert back, verify exact equality
- `"arithmetic matches std::complex"` — +, -, *, / against `std::complex` reference for 100 random pairs
- `"layout compatibility with std::complex"` — `static_assert` on `sizeof`, `offsetof(re)`, `offsetof(im)`
- `"reinterpret_cast between gr::complex and std::complex is safe"` — cast array, verify all elements
- `"vir::simdize produces correct SIMD type"` — instantiate `simdize<gr::complex<float>>`, load/store, verify
- `"SIMD arithmetic matches scalar"` — SIMD add/mul vs scalar add/mul on same data

**Quality gate**: All tests pass. Layout `static_assert`s enforced.

#### Step 0.3 — `gr::execution` core (3d)

**File**: `core/include/gnuradio-4.0/execution/execution.hpp`
**Functionality**:

- P2300 concepts: `sender`, `receiver`, `operation_state`, `scheduler`, `completion_signatures`
- CPOs: `set_value`, `set_error`, `set_stopped`
- 8 algorithms: `just`, `schedule`, `then`, `let_value`, `when_all`, `transfer`, `bulk`, `sync_wait`
- Pipe syntax: `sender | then(f) | bulk(N, g)`
- All algorithms are CPOs with `tag_invoke` customisation points

**Quality gate**: Compiles on GCC 15 + Clang 20 + Emscripten. Template depth ≤ 20 for a 5-stage pipeline.

#### Step 0.4 — `PoolScheduler` (1d)

**File**: `core/include/gnuradio-4.0/execution/pool_scheduler.hpp`
**Functionality**:

- `PoolScheduler` wrapping `TaskExecutor` as P2300 scheduler
- `schedule()` returns sender completing on pool thread
- `cpuScheduler()`, `ioScheduler()` factory functions
- Escape hatch: `executor()` for direct pool access

**Quality gate**: Compiles. `schedule()` produces a valid `sender`.

#### Step 0.5 — `qa_Execution.cpp` (2d)

**Tests**:

- `"just produces value"` — `sync_wait(just(42))` == `optional(tuple(42))`
- `"then transforms value"` — `just(3) | then([](int x){ return x*2; })` == 6
- `"schedule runs on pool thread"` — verify `std::this_thread::get_id()` differs from caller
- `"transfer hops between pools"` — `schedule(cpu) | transfer(io)`, verify thread name change
- `"when_all joins two senders"` — `when_all(just(1), just(2))` == `tuple(1, 2)`
- `"bulk invokes N times"` — `just(vec) | bulk(N, [](size_t i, auto& v){ v[i]++; })`, verify all incremented
- `"let_value chains senders"` — `just(3) | let_value([](int x){ return just(x+1); })` == 4
- `"set_stopped propagates through when_all"` — verify cancellation
- `"pipe syntax compiles and produces correct result"` — 5-stage pipeline

**Quality gate**: All tests pass on GCC 15 + Clang 20. Emscripten compiles (tests may skip if no threading).

**Review gate 0 → 0.5**: Q&A on `gr::complex` SIMD behaviour, P2300 concept correctness, any API surprises.

---

### Stage 0.5 — USM memory resource for GPU edges (3.5d)

#### Step 0.5.1 — `gpu::UsmMemoryResource` (1.5d)

**File**: `core/include/gnuradio-4.0/gpu/UsmMemoryResource.hpp`
**Functionality**:

- `UsmMemoryResource : std::pmr::memory_resource`
- SYCL path: `do_allocate()` → `sycl::malloc_shared()`, `do_deallocate()` → `sycl::free()`
- CPU fallback (no SYCL): `do_allocate()` → `::operator new()`, `do_deallocate()` → `::operator delete()`
- Alignment-aware (`do_allocate(size, align)`)
- Stores `sycl::queue*` (or `nullptr` for CPU fallback)

**Quality gate**: Compiles with and without SYCL. CPU fallback path works standalone.

#### Step 0.5.2 — `ComputeRegistry` GPU resolution (0.5d)

**File**: `core/include/gnuradio-4.0/ComputeDomain.hpp` (extend existing)
**Functionality**:

- `ComputeRegistry::register_provider()` for `"gpu:sycl"` backend → `UsmMemoryResource`
- `resolveScheduler()` method stub (returns nullptr until Stage 1 implements `GpuScheduler`)
- `"gpu"` → picks first available GPU provider; `"gpu:sycl:0"` → specific device

**Quality gate**: `resolve("gpu:sycl")` returns a valid `memory_resource*` (CPU fallback).

#### Step 0.5.3 — `qa_UsmMemoryResource.cpp` (1d)

**Tests**:

- `"allocate and deallocate float array"` — 4096 floats, write/read, no crash
- `"CircularBuffer with USM resource"` — construct `CircularBuffer<float>` with USM allocator, push/pop samples
- `"Tag with USM-backed property_map"` — create `Tag` with USM resource, insert string + float values, read back
- `"Tensor with USM resource"` — `Tensor<float>({1024})` with USM allocator, fill, verify
- `"multiple allocations and deallocations"` — stress test: 100 alloc/dealloc cycles, verify no leaks

#### Step 0.5.4 — Full PMR chain validation (0.5d)

**Tests** (added to `qa_UsmMemoryResource.cpp`):

- `"ComputeDomain gpu_shared resolves to UsmMemoryResource"` — `bind(ComputeDomain::gpu_shared())` returns valid
  resource
- `"EdgeParameters with GPU domain creates USM-backed port buffers"` — construct port with GPU edge, verify buffer uses
  USM resource

**Quality gate**: All tests pass. PMR chain end-to-end validated on CPU fallback.

**Review gate 0.5 → 1**: Q&A on PMR chain correctness, any edge cases with `double_mapped_memory_resource` interaction,
readiness for `DeviceContext`.

---

### Stage 1 — GPU abstraction layer (8d)

#### Step 1.1 — `gpu::DeviceContext` (2d)

**File**: `core/include/gnuradio-4.0/gpu/DeviceContext.hpp`
**Functionality**:

- Wraps `sycl::queue` (SYCL path) or nullptr (CPU fallback)
- `allocate_device<T>(count)` → `sycl::malloc_device` / `new T[count]`
- `allocate_shared<T>(count)` → `sycl::malloc_shared` / `new T[count]`
- `deallocate(ptr)` → `sycl::free` / `delete[]`
- `copy_h2d(host, device, count)` → `q.memcpy().wait()` / `std::memcpy`
- `copy_d2h(device, host, count)` → same
- `parallel_for(count, f)` → `q.parallel_for(range<1>{count}, f)` / `for` loop
- `wait()` → `q.wait()`
- SYCL validation: compile `gr::complex<float>` arithmetic in a trivial kernel (G8 mitigation)

**Quality gate**: Compiles with and without SYCL. CPU fallback fully functional.

#### Step 1.2 — `gpu::GpuScheduler` (2d)

**File**: `core/include/gnuradio-4.0/execution/gpu_scheduler.hpp`
**Functionality**:

- `GpuScheduler` models `gr::execution::scheduler`
- `schedule()` returns sender completing on GPU context
- `bulk` CPO override: dispatches to `DeviceContext::parallel_for`
- `transfer` CPO override: dispatches to `DeviceContext::copy_h2d` / `copy_d2h`
- Stores `GpuBackend` enum + `DeviceContext*`

#### Step 1.3 — `ComputeRegistry::resolveScheduler()` (1d)

**File**: `core/include/gnuradio-4.0/ComputeDomain.hpp` (extend)
**Functionality**:

- `resolveScheduler("gpu:sycl:0")` → `GpuScheduler&`
- `resolveScheduler("default_cpu")` → `PoolScheduler&`
- Unknown domain + GPU prefix → warn, fall back to CPU scheduler

#### Step 1.4 — `gpu::DeviceBlockState<T>` (2d)

**File**: `core/include/gnuradio-4.0/gpu/DeviceBlockState.hpp`
**Functionality**:

- Compile-time reflection over `GR_MAKE_REFLECTABLE` fields
- Scalar fields (`float`, `int`, `bool`): direct copy
- `std::pmr::vector<T>`: extract `{T* ptr, size_t size}` view
- `sync_to_device(ctx, block, state)`: copy state struct to device memory

#### Step 1.5 — `qa_DeviceContext.cpp` (1d)

**Tests**:

- `"allocate and deallocate device memory"` — 4096 floats, no crash
- `"copy host to device and back"` — fill host array, H2D, D2H, verify exact match
- `"parallel_for multiplies array"` — `parallel_for(N, [](i){ out[i] = in[i] * 2; })`, verify
- `"parallel_for with gr::complex"` — same with `gr::complex<float>`, validates G8
- `"CPU fallback produces same results"` — same tests on CPU-only DeviceContext

**Quality gate**: All tests pass on CPU fallback. SYCL CPU backend if AdaptiveCpp available.

#### Step 1.6 — `qa_GpuScheduler.cpp` (1d — reduced from 1d, shares infrastructure with 1.5)

**Tests**:

- `"bulk dispatches N work items"` — `schedule(gpu) | bulk(N, f) | sync_wait`, verify
- `"transfer moves data between contexts"` — `schedule(cpu) | transfer(gpu) | transfer(cpu)`, verify
- `"GpuScheduler resolves from ComputeRegistry"` — `resolveScheduler("gpu:sycl")` returns valid scheduler

**Quality gate**: All tests pass. `bulk` CPO correctly dispatches to `DeviceContext`.

**Review gate 1 → 2**: Q&A on `DeviceContext` API surface, `DeviceBlockState` reflection correctness, `processBulk_sycl`
signature finalisation (pin it here).

---

### Stage 2 — `ExecutionStrategy` and Block<T> integration (6d)

#### Step 2.1 — GPU concepts in `BlockTraits.hpp` (1d)

**File**: `core/include/gnuradio-4.0/BlockTraits.hpp` (extend)
**Functionality**:

- `AutoParallelisable<TBlock>`: `processOne(...) const noexcept` with supported return type
- `HasSyclBulk<TBlock>`: has `processBulk_sycl(sycl::queue&, const T*, size_t, T*, size_t)`
- `HasWgslShader<TBlock>`: has `generateShader(property_map) → string`
- `GpuEligible<TBlock>`: `AutoParallelisable || HasSyclBulk || HasWgslShader || ...`
- `ShaderCapable<TBlock>`: has `static generateShader(property_map) → string`

#### Step 2.2 — `gpu::ExecutionStrategy<TBlock>` (3d)

**File**: `core/include/gnuradio-4.0/gpu/ExecutionStrategy.hpp`
**Functionality**:

- `dispatch(block, inputSpans, outputSpans, count)`:
    1. If `HasSyclBulk` (SYCL backend) → copy input H2D, call `processBulk_sycl`, copy output D2H
    2. Else if `AutoParallelisable` → mirror `DeviceBlockState`, copy input H2D, `bulk(N, processOne)`, copy output D2H
- State sync: call `sync_to_device` when block settings differ from cached device state
- Implicit transfer management (when no explicit `HostToDevice`/`DeviceToHost` present)

#### Step 2.3 — `Block::workInternal()` GPU branch (0.5d)

**File**: `core/include/gnuradio-4.0/Block.hpp` (one `if constexpr` branch)
**Functionality**:

- Before existing SIMD dispatch: `if constexpr (gpu::GpuEligible<Derived>)` + runtime `compute_domain` check
- Delegate to `gpu::ExecutionStrategy<Derived>::dispatch(...)`

#### Step 2.4 — `qa_GpuMultiplyConst.cpp` CPU fallback (1d)

**Tests**:

- `"auto-parallelised processOne produces correct output"` — `MultiplyConst<float>{factor=2}` with
  `compute_domain="gpu"` (CPU fallback), 4096 samples, verify `out[i] == in[i] * 2`
- `"DeviceBlockState mirrors factor correctly"` — change factor via settings, verify GPU path uses new value
- `"non-const processOne is not GPU-eligible"` — `static_assert` that a stateful block fails `AutoParallelisable`
- `"results match CPU scalar path exactly"` — same input, compare GPU-fallback output vs CPU output

#### Step 2.5 — `qa_GpuMultiplyConst.cpp` SYCL CPU backend (0.5d)

**Tests** (same scenarios, but compiled with AdaptiveCpp and run on SYCL CPU device):

- `"SYCL CPU backend matches CPU fallback"` — numerical equivalence

**Quality gate**: `MultiplyConst` runs correctly on CPU fallback and SYCL CPU. Concepts reject non-const blocks.

**Review gate 2 → 3**: Q&A on `ExecutionStrategy` dispatch correctness, per-backend concept design confirmed stable,
`Block::workInternal()` diff review.

---

### Stage 3 — Transition blocks and explicit H2D/D2H (8d)

#### Step 3.1 — `HostToDevice<T>` (1.5d)

**File**: `blocks/gpu/HostToDevice.hpp`
**Functionality**: As specified in section 11.1. Fixed `chunk_size`, `processBulk` copies from host `CircularBuffer` to
device allocation via `DeviceContext::copy_h2d`.

#### Step 3.2 — `DeviceToHost<T>` (1d)

**File**: `blocks/gpu/DeviceToHost.hpp`
**Functionality**: As specified in section 11.2. Copies from device allocation back to host `CircularBuffer`.

#### Step 3.3 — Edge-level compute domain detection (2d)

**File**: `core/include/gnuradio-4.0/Graph.hpp` or scheduler (extend)
**Functionality**:

- On `connect()`: detect `source.compute_domain != sink.compute_domain`
- If explicit `HostToDevice`/`DeviceToHost` already present → use them
- If not → scheduler handles transfer internally (implicit mode)

#### Step 3.4 — Adaptive chunking (1d)

**File**: `blocks/gpu/HostToDevice.hpp` (extend)
**Functionality**: Throughput estimator. Doubles `chunk_size` if throughput improves > 5%, halves if drops > 10%.
Bounded by `[min_chunk_size, max_chunk_size]`.

#### Step 3.5 — `qa_HostToDevice.cpp` (1.5d)

**Tests**:

- `"explicit H2D/D2H transfers data correctly"` — `Source → H2D → MultiplyConst(gpu) → D2H → Sink`, verify output
- `"chunk_size setting controls transfer size"` — set chunk_size=512, verify transfers happen in 512-sample blocks
- `"tag at sample boundary splits work"` — inject tag at sample 100 with settings change, verify two chunks processed
  with different settings
- `"informational tags forwarded without splitting"` — tag with non-matching key passes through

#### Step 3.6 — `qa_GpuImplicitTransfer.cpp` (1d)

**Tests**:

- `"implicit transfer for single GPU block"` — `Source → MultiplyConst(gpu) → Sink` (no H2D/D2H blocks), verify correct
  output
- `"implicit and explicit produce same results"` — compare output of implicit vs explicit graphs

**Quality gate**: Both explicit and implicit paths produce correct results. Tag splitting works.

**Review gate 3 → 4**: Q&A on transition block correctness, adaptive chunking behaviour, implicit/explicit path parity.
Discuss FFT2 design details before proceeding.

---

### Stage 4 — `FFT2<T>` block (CPU SIMD + GPU SYCL) (10.5d)

**Design context:** `FFT2<T>` is a new raw FFT block in `gr::blocks::fourier`. The existing
`gr::blocks::fft::FFT` (DataSet output, windowing) is retained until FFT2 passes external review.

#### Step 4.1 — `gr::complex<float>` in SYCL FFT (1d)

**File**: `core/include/gnuradio-4.0/gpu/SyclFft.hpp` (new)
**Functionality**: Migrate PoC `complexf`/`sycl::float2` → `gr::complex<float>`. Verify SSCP compilation.

#### Step 4.2 — `gpu::SyclFft` class (3d)

**File**: `core/include/gnuradio-4.0/gpu/SyclFft.hpp`
**Functionality**:

- DIF Cooley-Tukey: global-memory stages + fused local-memory stages + bit-reversal
- Twiddle cache: computed on host, copied to device-only allocation
- Persistent device buffers (resized on `fft_size` change)
- `forward(ctx, d_data, N)` → in-place forward FFT
- `inverse(ctx, d_data, N)` → in-place inverse FFT
- `forward_batch(ctx, d_data, totalSamples, fftSize)` → batched

#### Step 4.3 — Batched FFT (1d)

**File**: `core/include/gnuradio-4.0/gpu/SyclFft.hpp` (extend)
**Functionality**: `forward_batch` detects `nBatches = totalSamples / fftSize`, launches `nBatches` independent FFTs in
one dispatch. Each batch offset by `batch * fftSize`.

#### Step 4.4 — `FFT2<T>` block (2d)

**File**: `blocks/fourier/include/gnuradio-4.0/fourier/FFT2.hpp`
**Functionality**: As specified in the block definition above. `processBulk` → `SimdFFT`, `processBulk_sycl` →
`SyclFft`.
`settingsChanged` handles `fft_size` change (reinit both, update `input_chunk_size`).

#### Step 4.5 — `qa_FFT2.cpp` numerical validation (1.5d)

**Tests**:

- `"CPU forward FFT matches known DFT"` — single tone input, verify peak at correct bin, magnitude within 1e-6
- `"CPU forward+inverse round-trip"` — random input, FFT then IFFT, verify error < 1e-6
- `"GPU forward FFT matches CPU"` — same input, compare GPU output vs CPU output, max error < 1e-5
- `"GPU forward+inverse round-trip"` — error < 1e-5
- `"batched FFT: each batch independent"` — 16 different inputs batched, verify each matches individual FFT
- `"batched GPU matches batched CPU"` — same 16 inputs, compare
- `"power-of-2 sizes: 1024, 2048, 4096, 8192, 16384, 32768, 65536"` — all sizes produce correct results
- `"fft_size change reinitialises correctly"` — change from 4096 to 8192 via settings, verify output still correct

#### Step 4.6 — `qa_FFT2.cpp` end-to-end (1d)

**Tests**:

- `"explicit path: Source → H2D → FFT2 → D2H → Sink"` — full graph, verify output
- `"implicit path: Source → FFT2(gpu) → Sink"` — same, no transition blocks
- `"explicit and implicit produce same results"` — compare outputs

#### Step 4.7 — Benchmark (1d)

**Tests** (benchmark mode, not assertions):

- `"CPU SIMD FFT throughput"` — N = 1024..65536, report MS/s
- `"GPU kernel-only throughput"` — persistent device data, no H2D/D2H
- `"GPU end-to-end throughput"` — including H2D/D2H transfer
- `"batched GPU throughput"` — 1, 4, 16, 64 batches at N=4096
- Report crossover point where GPU beats CPU

**Quality gate**: All numerical tests pass. GPU matches CPU within tolerance. Benchmark numbers reported.

**Review gate 4 → 5**: Q&A on FFT2 numerical correctness, GPU performance vs expectations, benchmark analysis. Discuss
shader generation approach for FFT2 before Stage 5.

---

### Stage 5 — WebGPU/WGSL backend with runtime shader specialisation (11d)

#### Step 5.1 — `GpuScheduler` WebGPU backend (2d)

**File**: `core/include/gnuradio-4.0/execution/gpu_scheduler.hpp` (extend)

#### Step 5.0 — DeviceContext refactor

**Files**: `core/include/gnuradio-4.0/device/DeviceContext.hpp` (rewrite),
`DeviceContextSycl.hpp` (new), `DeviceContextGLSL.hpp` (new)

Refactor `DeviceContext` from a single struct with `#if` guards to a virtual base class.
Update all consumers (~15 files): `SyclFFT`, `FFT2`, `TransferBlocks`, `ExecutionStrategy`,
`DeviceBlockState`, all `qa_Device*` tests, benchmark helpers.

#### Step 5.1 — DeviceContextGLSL

**File**: `core/include/gnuradio-4.0/device/DeviceContextGLSL.hpp`

Wraps existing `GlComputeContext`. Device memory = persistent SSBOs (allocated once, reused).
H2D/D2H via `glBufferData`/`glMapBufferRange`. `dispatch(shader, input, output, N, wgSize)`
compiles/caches shader and dispatches.

#### Step 5.2 — DeviceBuffer + DeviceBufferRegistry

**File**: `core/include/gnuradio-4.0/device/DeviceBuffer.hpp`

`DeviceBuffer` (RAII, move-only) + `SharedDeviceBuffer` (`shared_ptr<DeviceBuffer>` for fused
blocks). `DeviceBufferRegistry` tracks all live buffers per context: lifetime validation,
false-sharing detection, total memory accounting.

#### Step 5.3 — ShaderCache

**File**: `core/include/gnuradio-4.0/device/ShaderCache.hpp`

`ShaderKey{sourceHash, backend}` → compiled program handle. `getOrCompile()`, `invalidate()`,
LRU eviction. Thread-safe. Used by both GLSL and future WGSL backends.

#### Step 5.4 — ShaderFragment concept + ExecutionStrategy

**File**: `core/include/gnuradio-4.0/BlockTraits.hpp` (extend),
`core/include/gnuradio-4.0/device/ExecutionStrategy.hpp` (extend)

`ShaderFragment` struct: GLSL function source, constants, input/output chunk sizes.
`HasShaderFragment<TBlock>` concept. `ExecutionStrategy` dispatches to the GLSL path
when `HasShaderFragment` and backend is GLSL/WGSL.

#### Step 5.5 — MultiplyConst shader example

**File**: `blocks/basic/include/gnuradio-4.0/basic/MultiplyConst.hpp` (extend)

`shaderFragment()` returns GLSL `float process(float x) { return x * GAIN; }` with
`GAIN` as a baked constant. Test: compare output vs CPU for 4096 samples.

#### Step 5.6 — Shader fusion

**File**: `core/include/gnuradio-4.0/device/ShaderFusion.hpp`

Scheduler identifies fusible chains (adjacent blocks with `HasShaderFragment` in same
compute domain). Fuses element-wise blocks into one dispatch. Bulk blocks (FFT, chunk-based)
create barriers in the fused pipeline — the scheduler chunks input as multiples of the bulk
block's required size.

Each fused block declares its `inputChunkSize` and `outputChunkSize` via `ShaderFragment`.
The fusion pass computes the LCM chunk size, allocates device buffers, generates the combined
GLSL source, and compiles via `ShaderCache`. Settings changes invalidate the fused shader.

#### Step 5.7 — FFT2 GLSL shader

**File**: `blocks/fourier/include/gnuradio-4.0/fourier/FFT2.hpp` (extend)

`shaderFragment()` returns GLSL with baked N, unrolled butterfly stages, twiddle LUT as
`const` arrays, workgroup size optimised for N. `inputChunkSize = fft_size`.

#### Step 5.8 — GLSL → WGSL transpiler

**File**: `core/include/gnuradio-4.0/device/GlslToWgsl.hpp`

Minimal translator for the compute-shader subset: layout declarations, `gl_GlobalInvocationID`
→ `@builtin(global_invocation_id)`, `buffer` → `storage`, GLSL types → WGSL types. ~200 lines.
Invoked by `DeviceContextWebGPU` (future) or WASM build path.

#### Step 5.9 — Emscripten/WASM validation

Build clean with Emscripten. WebGL2 path (no compute) for rendering. WebGPU compute via
Node.js 24 `--experimental-webgpu` (Dawn) for CI. Desktop browsers for manual validation.

#### Step 5.10 — Benchmark

Integrate GLSL path into `algorithm/benchmarks/bm_fft_sycl.cpp`. Compare all backends
(CPU SimdFFT, CPU SYCL, GPU SYCL, GPU GLSL) in one table.

**Quality gate**: All tests pass on GCC 15, acpp, Emscripten. GLSL path verified on
NVIDIA (hardware) + Mesa llvmpipe (software). Fused shader correctness verified.

**Review gate 5 → 6**: Q&A on DeviceContext hierarchy, shader fusion correctness,
WASM compatibility.

---

### Stage 6 — Polish and CI (3d)

#### Step 6.1 — Compile guard validation (1d)

**Tests**: Build with GCC 15, Clang 20, Emscripten — all with `GR_USE_ADAPTIVE_CPP=OFF`. Verify zero GPU-related
compilation errors.

#### Step 6.2 — CI integration (1d)

**Functionality**: AdaptiveCpp CPU fallback tests added to existing CI job. WebGPU tests in a separate optional job (may
require Dawn in Docker).

#### Step 6.3 — Documentation (1d)

**File**: `docs/USER_API_GPU_Blocks.md`
**Content**: How to write a GPU block (`processBulk_sycl`), how to write a shader-capable block (`generateShader`), how
to set `compute_domain`, examples.

**Quality gate**: CI green. Documentation reviewed.

**Final review**: Full integration review before any merge to main.

---

**Stages 0–6 complete.** See "Integration assessment (2026-04-17)" below for the
tiered roadmap (Tier 1 → Tier 3) covering remaining integration, production hardening,
and first-class GPU support.

The `FFT2<T>` block:

```cpp
namespace gr::blocks::fourier {

template<typename T>
struct FFT2 : gr::Block<FFT2<T>> {
    using Description = Doc<"raw forward/inverse FFT, dispatches to CPU SIMD or GPU">;

    gr::PortIn<gr::complex<T>>  in;
    gr::PortOut<gr::complex<T>> out;

    Annotated<gr::Size_t, "fft size">  fft_size = 4096UZ;
    Annotated<bool, "inverse">         inverse  = false;

    GR_MAKE_REFLECTABLE(FFT2, in, out, fft_size, inverse);

    gr::algorithm::SimdFFT<T, gr::algorithm::Transform::Complex> _cpuFft;
    gpu::SyclFft                                                  _gpuFft;

    // CPU path — delegates to existing SimdFFT (zero-copy reinterpret_cast)
    gr::work::Status processBulk(const ConsumableSpan auto& inSpan,
                                  PublishableSpan auto& outSpan) { /* _cpuFft */ }

    // SYCL escape hatch — native queue + USM pointers, single or batched
    gr::work::Status processBulk_sycl(sycl::queue& q,
        const gr::complex<T>* d_in, std::size_t nIn,
        gr::complex<T>* d_out, std::size_t nOut) { /* _gpuFft */ }

    // Shader path — generates specialised WGSL with baked N and twiddles (Stage 5)
    static std::string generateShader(const property_map& settings) { /* ... */ }

    void settingsChanged(const property_map& old, const property_map& newSettings) {
        // reinitialise _cpuFft and _gpuFft when fft_size changes
        // invalidate shader cache if ShaderCapable
        // update input_chunk_size = fft_size
    }
};

} // namespace gr::blocks::fourier
```

**Key design decisions for FFT2:**

- **Raw output**: `gr::complex<T>`, not `DataSet<T>`. Windowing, magnitude, phase extraction
  are separate blocks. This keeps FFT2 composable (FFT-based FIR, spectrum chains).
- **CPU path reuses `SimdFFT`**: No CPU FFT reimplementation. Zero-copy conversion between
  `gr::complex<T>` and `std::complex<T>` (layout-compatible, enforced by `static_assert`).
- **Internal dispatch** via `compute_domain`, not template parameter. Same block type, different
  target at runtime.
- **Backend-specific escape hatches**: `processBulk_sycl(sycl::queue&, ...)` gets native SYCL
  types. `generateShader(property_map)` returns WGSL/GLSL text. Each backend gets type-safe
  native parameters — no `void*`, no `DeviceContext&` indirection.
- **Namespace**: `gr::blocks::fourier` (not `gr::blocks::fft`) to avoid confusion with the old
  `FFT` during the transition period.
- **Resampling**: `input_chunk_size = fft_size`, `output_chunk_size = fft_size`. Updated in
  `settingsChanged` when `fft_size` changes.

Usage — identical graph, different `compute_domain`:

```cpp
auto& fft = flow.emplaceBlock<FFT2<float>>({{"fft_size", 4096}});                                  // CPU SIMD
auto& fft = flow.emplaceBlock<FFT2<float>>({{"fft_size", 4096}, {"compute_domain", "gpu:sycl"}});  // GPU SYCL
auto& fft = flow.emplaceBlock<FFT2<float>>({{"fft_size", 4096}, {"compute_domain", "gpu:wgsl"}});  // WebGPU
```

### Stage 5 — DeviceContext refactor + GLSL shader backend + fusion

#### Completed (5.0–5.16)

| Step | Task                                                                                        | Status |
|------|---------------------------------------------------------------------------------------------|--------|
| 5.0  | **DeviceContext refactor**: virtual base + `DeviceType` + naming API                        | done   |
| 5.1  | **DeviceContextGLSL**: persistent SSBOs via `GlComputeContext`                              | done   |
| 5.2  | **DeviceBuffer + DeviceBufferRegistry**: RAII + `shared_ptr`, lifetime tracking             | done   |
| 5.3  | **ShaderCache**: LRU, thread-safe, eviction callback                                        | done   |
| 5.4  | **ShaderFragment** + `HasShaderFragment` + `ExecutionStrategy` GLSL path                    | done   |
| 5.5  | **`builtin_multiply::shaderFragment()`** + end-to-end GLSL test                             | done   |
| 5.6  | **ShaderFusion**: element-wise composition + chunk-based barriers                           | done   |
| 5.7  | **FFT2::shaderFragment()** with chunk-based fusion                                          | done   |
| 5.8  | **GLSL2WGSL** minimal transpiler                                                            | done   |
| 5.9  | **Emscripten/WASM** build validation (Node.js 24)                                           | done   |
| 5.10 | **Benchmark** via `DeviceContextGLSL`                                                       | done   |
| 5.11 | **Naming API**: `shortName()`, `name()`, `version()`, `deviceType()` on all backends        | done   |
| 5.12 | **GlslFFT**: Van Loan Stockham FFT via GLSL compute shaders                                 | done   |
| 5.13 | **`bm_FFT_backends.cpp`**: zero `#if`, runtime backend discovery, `FFTBackend` struct       | done   |
| 5.14 | **SyclFFT Van Loan fix**: corrected Stockham index mapping, dropped radix-4 for correctness | done   |
| 5.15 | **`forwardStockhamCpu()`**: CPU-side Van Loan for direct algorithm testing                  | done   |
| 5.16 | **qa_FFT2 restructured**: 7 suites, 37 tests, 62 asserts, segfault-free                     | done   |

#### Deferred

| What                                           | Why                                                      |
|------------------------------------------------|----------------------------------------------------------|
| WebGPU dispatch (Emscripten + Dawn)            | Node.js has no WebGPU adapter; browser testing manual    |
| `GR_DEVICE_HAS_GL` flag (replace `GL_COMPUTE`) | requires WebGL2/WebGPU init path                         |
| GLSL cross-test full `maxError` comparison     | GL state contamination between suites                    |
| SyclFFT radix-4 re-add                         | dropped for Van Loan correctness; re-add as optimisation |
| Sub-group shuffle integration                  | validated +22%, not integrated                           |

### Stage 6 — Polish and CI

| Step | Task                                                                               | Status                                                                                                                |
|------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 6.1  | Compile guard validation: `__ACPP__` guard on `AtomicRef.hpp`, `BackendDetect.hpp` | done                                                                                                                  |
| 6.2  | CI integration: AdaptiveCpp CPU fallback tests in existing CI job                  | done ([fair-acc/gnuradio4 CI](https://github.com/fair-acc/gnuradio4/actions/runs/23943643903/job/69835147900?pr=754)) |
| 6.3  | escape hatch documentation (block author guide)                                    | pending                                                                                                               |

### Integration assessment (2026-04-17)

**Context:** The `coreCleanUp` PR ([fair-acc/gnuradio4#769](https://github.com/fair-acc/gnuradio4/pull/769))
delivered the core infrastructure that the remaining integration depends on:

- `ComputeDomain::parse()` for `"gpu:sycl:0"` strings
- `Graph::applyEdgeConnection()` resolves domain → PMR via `ComputeRegistry::tryResolve()`
- `Edge` stores `_domain` (was previously dropped)
- composable `workInternal()` extraction (clean insertion point for GPU dispatch)
- `processEpilogue()`, move-only spans, PMR settings support
- multi-tag support with composable forwarding policies

#### Current state: architecturally complete, functionally orphaned

The entire GPU stack (~7500 lines, 49 files) is **disconnected from block execution**.
The situation:

```
What exists:                          What's missing:

Block::compute_domain = "gpu:sycl"    <-- stored, never read at dispatch time
        |
        v
Edge::_domain resolved -> USM PMR    <-- edges allocate GPU memory correctly
        |
        v
Block::dispatchProcessing()           <-- NO GPU branch -- goes straight to
        |                                 CPU processOne/processBulk
        v
ExecutionStrategy::dispatch()         <-- fully implemented, ZERO call sites
        |
        v
DeviceContext / ShaderFusion / etc.   <-- all working, tested in isolation
```

`Block::dispatchProcessing()` (~line 1749) has no `if constexpr (DeviceEligible<Derived>)`
branch. `ExecutionStrategy::dispatch()` -- the sophisticated GPU router -- has zero call sites
in production code. `registerUsmProvider()` is never called at startup. `DeviceBlockState`
(settings mirroring) is never invoked from `ExecutionStrategy`. No test runs a full graph
with explicit `compute_domain="gpu:sycl"`.

---

### Tier 1 -- "It works" (minimum viable wiring)

Activates everything already built. Target: ~2-3 days.

| #    | Task                                                 | Scope                                                                                                                                                                                                     | Files                                                    | Status  |
|------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|---------|
| T1.1 | **GPU dispatch branch in `dispatchProcessing()`**    | `if constexpr (DeviceEligible<Derived>)` + runtime `compute_domain` check -> `ExecutionStrategy::dispatch()`. The composable extraction from `coreCleanUp` provides a clean insertion point (~line 1749). | `Block.hpp` (~20 lines)                                  | pending |
| T1.2 | **`registerUsmProvider()` at startup**               | Call from `Graph` init or static registration so `"gpu:sycl"` edges auto-resolve to USM allocators. Currently defined in `UsmMemoryResource.hpp:95` but never invoked.                                    | `Graph.hpp` or new init (~5 lines)                       | pending |
| T1.3 | **`SchedulerRegistry` <-> `ComputeRegistry` bridge** | `ComputeRegistry` creates `DeviceContext`, registers with `SchedulerRegistry`. GPU dispatch needs a `DeviceContext*` resolved from `compute_domain` at runtime.                                           | `ComputeDomain.hpp`, `SchedulerRegistry.hpp` (~15 lines) | pending |
| T1.4 | **`DeviceBlockState` wiring**                        | `ExecutionStrategy::dispatch()` calls `sync_to_device()` before each GPU dispatch so block settings (e.g. `factor` in MultiplyConst) are visible on device.                                               | `ExecutionStrategy.hpp` (~10 lines)                      | pending |
| T1.5 | **Tag forwarding in GPU path**                       | `ExecutionStrategy::dispatch()` must forward tags through unchanged. The new multi-tag policies from `coreCleanUp` work for CPU; GPU path must at minimum pass all tags to output.                        | `ExecutionStrategy.hpp` (~20 lines)                      | pending |
| T1.6 | **End-to-end test**                                  | `Source -> H2D -> MultiplyConst(gpu) -> D2H -> Sink`, verify output matches CPU. Also: `Source -> FFT2(gpu) -> Sink` via `processBulk_sycl`. Both using `compute_domain="gpu:sycl"` (CPU fallback).       | new `qa_DeviceIntegration.cpp` (~80 lines)               | pending |
| T1.7 | **Clang 20 build validation**                        | Reconfigure + rebuild `build-clang20` to verify merged device code compiles clean.                                                                                                                        | build system only                                        | pending |

**Quality gate:** A user can write `compute_domain = "gpu:sycl"` on any block with
`const noexcept processOne` and it runs on GPU (CPU fallback) through the framework,
not manual calls. FFT2's `processBulk_sycl` works via the dispatch path. End-to-end test
passes on GCC 15 + Clang 20.

---

### Tier 2 -- "It's usable" (production-quality integration)

Makes the GPU path robust enough for real signal processing chains. Target: ~2-3 weeks.

| #    | Task                                        | Why                                                                                                                                                                                                                          | Status  |
|------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| T2.1 | **Automatic transfer block insertion**      | Graph detects CPU<->GPU domain transitions in `applyEdgeConnection()`, inserts `HostToDevice`/`DeviceToHost` automatically. Without this, users must manually build `Source -> H2D -> ... -> D2H -> Sink`.                   | pending |
| T2.2 | **Adaptive chunk sizing**                   | `HostToDevice` accumulates samples to amortise DMA. Currently fixed `chunk_size`. Needs throughput-based auto-tuning: double if throughput improves >5%, halve if drops >10%. Bounded by `[min_chunk_size, max_chunk_size]`. | pending |
| T2.3 | **Settings change -> shader recompilation** | When `fft_size` changes, cached shaders must invalidate. `settingsChanged()` must notify `ShaderCache`. Currently `ShaderCache` exists but is never invalidated by settings changes.                                         | pending |
| T2.4 | **`processEpilogue()` interaction**         | Trailing samples (< chunk size) should fall back to CPU. If a GPU-eligible block also defines `processEpilogue()`, the epilogue must run on the CPU path. Verify and test.                                                   | pending |
| T2.5 | **Error reporting**                         | GPU allocation failure, shader compile error, device lost -- all need to surface through `std::expected` or tags, not silent fallback. Currently errors are swallowed.                                                       | pending |
| T2.6 | **CI with GPU backend**                     | AdaptiveCpp CPU fallback in CI (already partially done -- step 6.2). Mesa llvmpipe for GL Compute. Device tests should run in standard CI, not only when manually enabled.                                                   | pending |
| T2.7 | **`TransferBlocks.hpp` location**           | `HostToDevice<T>` and `DeviceToHost<T>` live in `blocks/basic/` but `blocks/device/` exists as a separate CMake target. Move transfer blocks to `blocks/device/` or remove the empty scaffold.                               | pending |
| T2.8 | **Block author documentation**              | `docs/USER_API_GPU_Blocks.md` -- how to write a GPU block (`processBulk_sycl`, `shaderFragment()`), how to set `compute_domain`, examples, performance guidelines. Was stage 6.3 / 7E.                                       | pending |

**Quality gate:** A user can build a graph with mixed CPU/GPU blocks, connect them normally,
and the framework handles transfer blocks, domain transitions, and chunk accumulation
automatically. Settings changes propagate to device. Errors are reported, not silent.

#### Transfer block design notes

Transfer blocks serve two purposes:

1. **Domain boundary markers** -- `HostToDevice<T>` and `DeviceToHost<T>` mark the CPU<->device
   boundary. Between them, data stays on-device (USM edges, no per-block round-trips).

2. **Chunk size control** -- GPU dispatch needs large batches for efficient DMA amortisation
   and kernel occupancy. `HostToDevice<T>` accumulates to `chunk_size` before crossing.

```
CPU blocks -> HostToDevice -> [device block] -> [device block] -> DeviceToHost -> CPU blocks
                  ^                 ^ data stays on device ^           ^
             host->device       device edges (USM)              device->host
             DMA boundary                                     DMA boundary
```

Edge memory resolution (in `Graph::applyEdgeConnection()`):

- CPU<->CPU edge: `std::pmr::new_delete_resource()` (default, unchanged)
- CPU<->`HostToDevice`: host-side edge -- `HostToDevice` copies into USM output
- `HostToDevice`<->device: USM shared edge (`UsmMemoryResource`)
- device<->device: USM shared edge (data stays on device)
- device<->`DeviceToHost`: USM shared edge -- `DeviceToHost` copies to host output
- `DeviceToHost`<->CPU: host-side edge (default resource)

USM shared memory IS host-accessible, so `std::ranges::copy` works directly.
The actual host->device DMA is implicit (hardware/driver, triggered by device access).

---

### Tier 3 -- "First-class citizen" (parity with CPU path)

Makes GPU compute a natural, optimised part of the GR4 framework. Long-tail work -- months
of iteration, profiling, and real-world usage feedback.

| #     | Task                                      | Why                                                                                                                                                                                                                                                                                                                                         |
|-------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T3.1  | **GPU sub-graph scheduling**              | Chain of GPU blocks should keep data on-device. Currently each block does independent H2D -> compute -> D2H. Need graph-level analysis: identify maximal GPU sub-graphs, wrap with single H2D/D2H pair, schedule blocks within via SYCL events or shader fusion.                                                                            |
| T3.2  | **Shader fusion in scheduler**            | Adjacent element-wise GPU blocks (Multiply -> Add -> Scale) should fuse into a single shader dispatch. `ShaderFusion` exists and works; the scheduler must invoke it when it detects fusible chains.                                                                                                                                        |
| T3.3  | **`compute_domain` propagation**          | When a block is `"gpu:sycl"`, downstream blocks in the same sub-graph should inherit the domain unless explicitly overridden. Avoids requiring every block in a chain to be individually annotated.                                                                                                                                         |
| T3.4  | **Device-resident CircularBuffer**        | Currently USM edges use shared memory (host-accessible). For maximum throughput, inter-device edges should use device-only memory with explicit DMA at sub-graph boundaries. Shared memory has PCIe coherency overhead on discrete GPUs.                                                                                                    |
| T3.5  | **`std::pmr::string` internal migration** | Block internal strings (`name`, `unique_name`, `compute_domain`) are `std::string`. On embedded/device systems, these should be `std::pmr::string`. The `coreCleanUp` PR added `to_pmr()`/`to_std()` helpers but ~200 site changes remain. Blocked by C++ `std::string`/`std::pmr::string` type incompatibility (no cross-type comparison). |
| T3.6  | **Multi-device scheduling**               | `"gpu:sycl:0"` vs `"gpu:sycl:1"` -- scheduling across multiple GPUs. `SchedulerRegistry` already supports prefix matching; needs scheduler logic to distribute sub-graphs across devices and manage cross-device transfers.                                                                                                                 |
| T3.7  | **Profiling integration**                 | Expose kernel execution time, memory transfer time, occupancy via tags or a profiling API. Essential for users to know if GPU is faster for their graph. SYCL events provide timing; GL queries likewise.                                                                                                                                   |
| T3.8  | **WebGPU runtime backend**                | `GLSL2WGSL` transpiler exists. Needs `DeviceContextWebGPU` wrapping Dawn or `navigator.gpu`. Enables browser-based GPU compute via Emscripten. Currently blocked by lack of stable WebGPU in Node.js for CI.                                                                                                                                |
| T3.9  | **Higher-radix FFT codelets**             | Radix-4, radix-8, radix-16 for SyclFFT/GlslFFT. R2C/C2R transforms. Non-power-of-2 via Bluestein. Sub-group shuffle integration (+22% validated but not integrated).                                                                                                                                                                        |
| T3.10 | **`gr::complex<T>` migration**            | Migrate all complex-valued blocks from `std::complex<T>` to `gr::complex<T>`. Only critical blocks (FFT2) use it today.                                                                                                                                                                                                                     |
| T3.11 | **DeviceContextCUDA / DeviceContextROCm** | Native CUDA/ROCm backends (user-implementable via the virtual base class established in Stage 5). Not needed while AdaptiveCpp SYCL covers NVIDIA/AMD, but required for vendor-specific optimisations.                                                                                                                                      |
| T3.12 | **Auto-GLSL from `processOne`**           | Expression-template DSL that transpiles `processOne` C++ to GLSL at compile time. Eliminates the need for manual `shaderFragment()` implementations. Research-grade -- may not be practical.                                                                                                                                                |

---

### Summary

| Stage      | Description                                                                                                                                                                                                  | Status              |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| 0-6        | Foundation: `gr::complex`, `gr::execution`, DeviceContext hierarchy, ShaderFragment/Fusion, SyclFFT, GlslFFT, FFT2, TransferBlocks, compile guards, CI                                                       | **done**            |
| 7A         | Escape hatch signatures (`std::span`)                                                                                                                                                                        | **done**            |
| **Tier 1** | **Minimum viable wiring** (T1.1-T1.7): dispatch branch, USM registration, SchedulerRegistry bridge, DeviceBlockState wiring, tag forwarding, end-to-end test, Clang 20 validation                            | **pending** (~2-3d) |
| **Tier 2** | **Production integration** (T2.1-T2.8): automatic transfer insertion, adaptive chunking, shader invalidation, processEpilogue interaction, error reporting, CI, docs                                         | **pending** (~2-3w) |
| **Tier 3** | **First-class citizen** (T3.1-T3.12): sub-graph scheduling, shader fusion in scheduler, domain propagation, device-only buffers, PMR migration, multi-device, profiling, WebGPU, higher-radix FFT, CUDA/ROCm | **future**          |

---

## 13. Known Gotchas and Mitigations

| #   | Gotcha                                                                           | Mitigation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Stage |
|-----|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| G1  | `gr::complex<T>` ↔ `std::complex<T>` layout compatibility                        | `static_assert(sizeof(gr::complex<T>) == sizeof(std::complex<T>))` + `static_assert` on member offsets. Enforce before any `reinterpret_cast`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0     |
| G2  | `vir::simdize<gr::complex<T>>` may not work without specialisation               | Validate in `qa_Complex.cpp`. May require upstream coordination with `vir-simd` or a local specialisation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 0     |
| G3  | Per-backend escape hatch signatures                                              | Pin `processBulk_sycl(sycl::queue&, const T*, size_t, T*, size_t)` in Stage 2. WGSL uses `generateShader`. Each backend's signature is independent — adding a new backend doesn't change existing ones.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 2     |
| G9  | Pointer-based escape hatches don't work for GL/WebGPU                            | GL/WebGPU uses buffer handles, not pointers. Blocks targeting GL/WebGPU must use `generateShader` (shader text), not `processBulk_sycl`. Two escape hatch families: pointer-based (SYCL, CUDA, ROCm) and shader-based (WGSL, GLSL).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 5     |
| G4  | `fft_size` change at runtime                                                     | `settingsChanged` must atomically: reinit `SimdFFT`, reinit `SyclFft` (device realloc + twiddle recompute), invalidate shader cache, update `input_chunk_size`. Test all three paths.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 4     |
| G5  | Twiddle cache memory location                                                    | Compute on host, `memcpy` to device allocation (not USM shared). Device-only memory is faster for GPU reads.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 4     |
| G6  | Resampling semantics for FFT2                                                    | `input_chunk_size = fft_size`, `output_chunk_size = fft_size`. Scheduler must deliver exact multiples of `fft_size`. Batched: `K × fft_size` input → `K × fft_size` output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 4     |
| G7  | Old `FFT` block coexistence                                                      | `FFT2` in `gr::blocks::fourier` namespace. Old `FFT` stays in `gr::blocks::fft`. Both registered under different names. Old block retired only after explicit approval.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 4     |
| G8  | AdaptiveCpp SSCP + `gr::complex<T>` on device                                    | Validate early: compile a trivial SYCL kernel using `gr::complex<float>` arithmetic through SSCP. If it fails, fall back to `sycl::float2` internally with conversion at boundaries.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 1     |
| G10 | acpp SSCP: `nd_range` HCF lookup fails during static init or with `boost/ut.hpp` | **Two confirmed triggers:** (1) `nd_range` kernel called during static init (global ctor) → HCF cache not ready. (2) `boost/ut.hpp` compiled by acpp in any TU of the same executable → HCF blob interference breaks `nd_range` lookup even at runtime. `basic parallel_for` unaffected in both cases. Suite-in-main does NOT help for (2) — the issue is the HCF blob generated from boost/ut's template machinery, not call timing. All linkers (mold, bfd, lld), all flags fail. **Only fix:** SHARED library for SYCL-kernel TU. **MVP:** `acpp_bug_reporting/acpp_hcf_repro.tar.gz` (5 files, 2 KB, zero dependencies). **Upstream:** [AdaptiveCpp#2042](https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2042). | 0–4   |
| G11 | SYCL + EGL coexistence                                                           | SYCL and EGL coexist in the same process and same TU (verified). The separate-TU + SHARED pattern is for G10, not header conflicts. Global Boost.UT suites work alongside SYCL — no workaround needed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 4     |
| G12 | Per-stage kernel launch overhead (GPU SYCL FFT)                                  | **Fixed (V1→V2):** removed `.wait()`, stages now chained via `sycl::event` deps + in-order queue. Single host wait at end. Remaining overhead: AdaptiveCpp dispatch ~70μs/launch. Reduce launches further via wgSize=1024 + radix-4 per-thread.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 4     |
| G13 | Serial batch loop in `SyclFft::forwardBatch`                                     | **Fixed (V1→V2):** batches fused into single kernel launches (`sycl::range` covers all batches). No host-side batch loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 4     |
| G14 | `std::size_t` (64-bit) integer division in GPU kernels                           | 64-bit div/mod on NVIDIA GA104 costs ~100+ cycles (emulated). Butterfly itself is ~20 cycles. **96% of kernel time is index arithmetic.** Fix: `uint32_t` + bit-shifts for power-of-2 dimensions, `sycl::range<2>` for batch dimension.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 4     |
| G15 | DIF bit-reversal pass on GPU                                                     | Random-access permutation as a separate kernel — worst-case memory pattern. Fix: switch GPU path to **Stockham auto-sort** (naturally ordered output, no permutation). Keep DIF for CPU fallback (in-place is valuable there).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 4     |
| G16 | Stockham index mapping was wrong                                                 | Initial Stockham used DIF-style indices (`srcLo = block*halfSpan + k`). Produced bit-reversed output, not natural order. **Fix:** Van Loan canonical mapping: `srcLo = j` (sequential), `dstLo = group * Ls + k` (interleaved), `butterfly: a ± w*b`. Applied to both SyclFFT and GlslFFT. Validated by `forwardStockhamCpu()` against SimdFFT.                                                                                                                                                                                                                                                                                                                                                                        | 4–5   |
| G17 | `DeviceContext` typed template name-hiding                                       | Derived classes that override `copyHostToDevice(void*, void*, size_t)` hide the base's typed template `copyHostToDevice<T>(T*, T*, size_t)`. **Fix:** `using DeviceContext::copyHostToDevice;` in every derived class. Without this, typed calls silently resolve to the `void*` overload, interpreting element count as bytes.                                                                                                                                                                                                                                                                                                                                                                                        | 5     |
| G18 | `GlComputeContext` destructor segfault                                           | `eglTerminate` in destructor invalidates GL context. If `GlslFFT` instances are destroyed after `eglTerminate` (e.g. test suite ordering), their `deallocateRaw` calls segfault. **Fix:** leaked Meyer's singleton (`*new GlState()`) for the GL context in test code — destructor never runs.                                                                                                                                                                                                                                                                                                                                                                                                                         | 5     |

---

## 14. Open Questions (to revisit during implementation)

1. **SYCL event overhead**: For sub-graphs with many small blocks, per-kernel event tracking may
   dominate. Profiling needed to determine if in-order queue is sufficient.
2. **Reflection depth for `DeviceBlockState`**: PMR vector mirroring (`{T*, size_t}`) needs
   validation with actual FIR filter taps.
3. **WebGPU shader compilation latency**: First-time shader compilation may cause a visible
   stall. Measure latency; consider async compilation with a CPU fallback during compile.
4. **Shader recompilation on settings change**: When `fft_size` changes, a new shader is
   compiled. How does this interact with in-flight data? May need to drain the pipeline
   before switching shaders.
5. **SHARED library for SYCL benchmark helpers**: acpp SSCP cannot resolve `nd_range` HCF
   when `boost/ut.hpp` is in the same binary (any TU, any linker, any flags). SHARED library
   is the only workaround. Suite-in-main doesn't help. MVP reproducer at
   `acpp_bug_reporting/acpp_hcf_repro.tar.gz`. File upstream, fix direction: lazy HCF init
   or isolate boost/ut's template IR from kernel IR during SSCP processing.
6. **CPU SYCL performance**: The scalar DIF radix-2 reaches ~3 GFLOP/s vs SimdFFT's ~20 GFLOP/s
   (GCC). To reach 80% parity, the CPU SYCL path needs SIMD: either `vir::simd` butterflies
   inside `parallelFor`, or a dedicated `processBulk_simd` path that uses the same DIF algorithm
   with explicit vectorisation.

---

## 14. File Structure

```
core/include/gnuradio-4.0/
├── Complex.hpp                         (gr::complex<T>)
├── ComputeDomain.hpp                   (ComputeDomain parser + ComputeRegistry — from coreCleanUp)
├── AtomicRef.hpp                       (existing, ACPP guard fixed — from coreCleanUp)
├── Block.hpp                           (composable workInternal — from coreCleanUp; GPU branch pending 7C)
├── BlockTraits.hpp                     (AutoParallelisable, HasSyclBulk, HasShaderFragment, DeviceEligible)
├── execution/
│   ├── execution.hpp                   (P2300 concepts, CPOs, 8 algorithms)
│   ├── pool_scheduler.hpp              (PoolScheduler wrapping TaskExecutor)
│   └── gpu_scheduler.hpp               (GpuScheduler + bulk CPO override)
├── device/
│   ├── BackendDetect.hpp               (compile-time SYCL/GL availability detection)
│   ├── DeviceContext.hpp               (virtual base class for device wrappers)
│   ├── DeviceContextSycl.hpp           (SYCL backend via AdaptiveCpp)
│   ├── DeviceContextGLSL.hpp           (GL Compute backend via GlComputeContext)
│   ├── DeviceBlockState.hpp            (reflection-based state mirroring — pending 7D wiring)
│   ├── DeviceBuffer.hpp                (RAII device buffer + SharedDeviceBuffer + registry)
│   ├── ExecutionStrategy.hpp           (composed dispatch: auto-bulk, escape hatch, or shader)
│   ├── GlComputeContext.hpp            (EGL/OpenGL compute context — headless)
│   ├── GLSL2WGSL.hpp                   (minimal GLSL → WGSL transpiler)
│   ├── SchedulerRegistry.hpp           (maps compute_domain strings to DeviceContext instances)
│   ├── ShaderCache.hpp                 (compiled shader program cache with LRU eviction)
│   ├── ShaderFragment.hpp              (per-block shader source + constants descriptor)
│   ├── ShaderFusion.hpp                (fuses element-wise shader chains into single dispatch)
│   └── UsmMemoryResource.hpp           (std::pmr::memory_resource wrapping sycl::malloc_shared)
algorithm/include/gnuradio-4.0/algorithm/fourier/
├── SyclFFT.hpp                         (Stockham Van Loan FFT, SYCL)
└── GlslFFT.hpp                         (Stockham Van Loan FFT, GLSL compute)
blocks/
├── basic/include/gnuradio-4.0/basic/
│   └── TransferBlocks.hpp              (HostToDevice<T> + DeviceToHost<T>)
├── device/                             (empty — CMake target only)
├── fourier/
│   └── include/gnuradio-4.0/fourier/
│       ├── fft.hpp                     (existing FFT block — retained until FFT2 passes review)
│       └── FFT2.hpp                    (new: raw FFT, CPU SimdFFT + GPU SyclFft + shader)
core/test/
├── qa_Complex.cpp
├── qa_Execution.cpp
├── qa_UsmMemoryResource.cpp
├── qa_DeviceContext.cpp
├── qa_DeviceMultiplyConst.cpp          (auto-parallelised processOne on CPU fallback)
├── qa_DeviceScheduler.cpp
├── device_test_helpers.{hpp,cpp}       (shared test infrastructure)
blocks/device/test/
└── qa_HostToDevice.cpp
blocks/fourier/test/
└── qa_FFT2.cpp                         (7 suites, 37 tests — CPU + GPU + batched)
algorithm/benchmarks/
├── bm_FFT_backends.cpp                 (runtime backend discovery, all 4 backends)
└── bm_FFT_backends_helpers.{hpp,cpp}
```

---

## 15. Glossary

| Term                | Definition                                                                         |
|---------------------|------------------------------------------------------------------------------------|
| SSCP                | Single-Source, Single Compiler Pass — AdaptiveCpp's generic compilation flow       |
| USM                 | Unified Shared Memory — SYCL memory model (device, shared, host tiers)             |
| H2D / D2H           | Host-to-Device / Device-to-Host memory transfer                                    |
| DIF                 | Decimation-In-Frequency — FFT algorithm variant, in-place                          |
| WGSL                | WebGPU Shading Language — shader language for WebGPU compute                       |
| `DeviceBlockState`  | Reflection-generated device-side mirror of block settings                          |
| `ExecutionStrategy` | Composed dispatch helper: routes to auto-bulk, escape hatch, or shader             |
| `ShaderCapable`     | Concept: block provides `generateShader(settings)` for runtime-specialised shaders |
| `ShaderCache`       | Compiled shader program cache, keyed by source hash + backend                      |
| `bulk`              | P2300 data-parallel algorithm — the GPU dispatch primitive                         |
| CPO                 | Customisation-Point Object — enables scheduler-specific overrides                  |
| `t_or_simd<V,T>`    | Existing GR4 concept accepting both scalar T and SIMD vector types                 |
| Baked constant      | Shader setting embedded in source code; recompile on change                        |
| Uniform             | Shader setting passed via buffer; update without recompile                         |

---

## 16. External References

### 16.1 TinyCompute (Koen Samyn)

**Repository**: https://github.com/samynk/TinyCompute
**Talk**: CppCon 2025 — "From Pure ISO C++20 to Compute Shaders"
**License**: (check repo)  |  **Language**: C++20 |  **Backend**: OpenGL 4.3 compute shaders

**What it is.** A header-heavy library for writing compute kernels as pure ISO C++ structs that
execute on CPU (via `std::execution` policies) or GPU (via a Clang-based AST transpiler that
rewrites the struct's `main()` to GLSL). The dual-backend model means identical kernel source
runs with breakpoints on CPU and at full speed on GPU.

**Architecture — key types:**

| Type / Concept                     | Purpose                                                               | GR4 relevance                                  |
|------------------------------------|-----------------------------------------------------------------------|------------------------------------------------|
| `BufferBinding<T, B, Set>`         | Typed SSBO wrapper; `binding[i]` works on CPU (vector) and GPU (SSBO) | high — maps to port data                       |
| `DimTraits<D>`                     | 1D/2D/3D index flattening via specialisation; compile-time dispatch   | high — FFT, spectrogram                        |
| `PixelConcept` / `GPUFormatTraits` | Channel-aware pixel types (RGBA32F, RGBA8UI, etc.) + format metadata  | medium — spectrum → texture                    |
| `Swizzle` (user-defined literals)  | `"xy"_sw`, `"rgba"_sw` → compile-time 2-bit mask extraction           | low — niche                                    |
| `ComputeBackend<Derived>` (CRTP)   | CPU/GPU backend swap; `CPUBackend` uses parallel algorithms           | high — same pattern as our `ExecutionStrategy` |
| `KernelRewriter`                   | Clang `RecursiveASTVisitor` that rewrites C++ → GLSL                  | low — we use `ShaderFragment` strings instead  |

**Transferable patterns (Phase 2):**

1. **Kernel-as-struct** — the struct _is_ the kernel description. `local_size`, buffer bindings,
   and `main()` all live together. For GR4, a signal-processing block's `ShaderFragment` could
   evolve into a kernel struct where `main()` doubles as the CPU reference implementation and
   the transpilation source.
2. **`DimTraits`** — clean abstraction for 1D (time-domain buffers), 2D (spectrogram / waterfall),
   3D (future multi-channel spectrogram). Our `ShaderFusion` already assumes 1D; DimTraits would
   generalise it for 2D FFT or image-domain blocks.
3. **CPU/GPU parity via backend swap** — TinyCompute's `CPUBackend` sets `gl_GlobalInvocationID`
   as a thread-local, then calls `main()` per work-item. This is exactly how our
   `DeviceContextCpu::dispatch()` should work for `ShaderFragment`-based blocks: run the same
   `process()` function on CPU for testing without GPU hardware.
4. **Typed buffer bindings** — `BufferBinding<float, 0>` encodes the SSBO binding point in the
   type. Our `ShaderFragment` currently uses untyped `inputBuffer` / `outputBuffer` names.
   Typed bindings would make multi-buffer blocks (e.g. FFT with twiddle + data + output) safer.

**What we do NOT adopt:**

- **Clang AST transpiler** — too heavy a dependency (requires Clang/LLVM at build time). We
  already have `ShaderFragment` returning GLSL strings; this is simpler and sufficient.
- **Swizzle system** — elegant but irrelevant for 1D signal processing. Only useful if we add
  image/texture rendering blocks (spectrum colourmap → texture).
- **`PixelConcept`** — relevant only for visualisation (spectrum → RGBA texture). Deferred
  until GR4 adds native GPU-rendered visualisation blocks.

**Gaps in TinyCompute that we already fill:**

- No shared/local memory or barrier primitives (we need these for FFT local-memory stages —
  see §9.7). Our SYCL path uses `sycl::local_accessor` + `group_barrier`; our GLSL path uses
  `shared` arrays + `barrier()`.
- No WGSL or SYCL backend (OpenGL-only). We already have SYCL via AdaptiveCpp and GLSL via EGL.
- No shader caching or runtime specialisation. We have `ShaderCache` (LRU, hash-keyed).
- No PMR / memory resource integration. We have the full PMR chain through `UsmMemoryResource`.

### 16.2 ComputeShadersTutorial (Koen Samyn)

**Repository**: https://github.com/samynk/ComputeShadersTutorial
**License**: CC0 (public domain)  |  **Language**: C++20 / C# (OpenTK)  |  **Backend**: OpenGL 4.3

**What it is.** Educational companion to TinyCompute — 11 progressive C++ projects (+ C#
mirrors) demonstrating compute shader fundamentals from image filtering to ray tracing. Not a
library; a teaching resource showing practical patterns.

**Notable patterns worth studying:**

| Project / Pattern             | What it demonstrates                                                                            | GR4 relevance                                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Separable filtering** (P01) | Two-pass 1D blur (horizontal then vertical) with `GL_TEXTURE_UPDATE_BARRIER_BIT` between passes | high — maps to separable convolution / FIR in freq domain             |
| **Ping-pong SSBO** (P03)      | Double-buffered SSBOs for in-place updates (Game of Life)                                       | high — identical to our FFT ping-pong buffers                         |
| **Two-stage pipeline** (P05)  | Ray generation → SSBO → visualisation as separate dispatches                                    | medium — maps to compute → render chains                              |
| **SSBO template wrapper**     | `ShaderStorageBufferObject<T>` with typed upload/download                                       | high — directly comparable to our `DeviceBuffer`                      |
| **Workgroup size query**      | `GL_COMPUTE_WORK_GROUP_SIZE` queried post-compilation                                           | medium — our `GlComputeContext` hard-codes sizes; could query instead |

**Transferable insights:**

1. **Memory barriers between passes** — the tutorial explicitly demonstrates where
   `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` is required between compute dispatches
   that share buffers. Our `ShaderFusion` assumes element-wise chains need no barriers; for
   multi-pass algorithms (separable filters, FFT stages), explicit barriers between dispatches
   are mandatory. The tutorial's pattern matches our GLSL FFT stage separation.
2. **Querying workgroup size from compiled shader** — rather than hard-coding workgroup sizes
   in the dispatch call, the tutorial queries `GL_COMPUTE_WORK_GROUP_SIZE` after compilation.
   This is more robust: the shader declares its own optimal size, and the host adapts dispatch
   dimensions accordingly. Our `GlComputeContext::dispatch()` could adopt this.
3. **Typed SSBO wrapper** — `ShaderStorageBufferObject<T>` with `Upload()`/`Download()` and
   1D/2D accessors. Similar to our `DeviceBuffer` but with templated element type and
   CPU↔GPU transfer built in. Validates our design direction.

**What we do NOT adopt:**

- The GLFW/GLEW windowing — we use EGL headless (no display server needed).
- The C# / OpenTK examples — not relevant.
- The ray-tracing and boid-flocking examples — interesting but outside signal processing.

### 16.3 gpuAgnosticFunctor (Nick Karpowicz)

**Repository**: https://github.com/NickKarpowicz/gpuAgnosticFunctor
**License**: MIT |  **Language**: C++17/20 |  **Backends**: CUDA, SYCL (oneAPI), OpenMP (CPU)

**What it is.** A minimal (~6.5 KB) proof-of-concept demonstrating functor-based GPU kernel
dispatch across CUDA, SYCL, and OpenMP. Each backend is a standalone header implementing an
identical device class API (`Malloc`, `Free`, `MemcpyDeviceToHost`, `LaunchKernel`). Backend
selection is compile-time via preprocessor flags — each backend produces a separate executable.

**Architecture — the functor dispatch pattern:**

```cpp
// kernel as a stateful functor — captures device pointers
struct MultiplyAdd {
    float* A;
    float* B;
    float  scale;
    deviceFunction void operator()(unsigned int i) const {
        B[i] = A[i] * scale + 1.0f;
    }
};

// dispatch — identical call across all backends
device d;
d.LaunchKernel(Nblocks, Nthreads, MultiplyAdd{d_A, d_B, 2.0f});
```

Each backend dispatches differently:

- **CUDA**: `kernel<<<Nblock, Nthread>>>(functor)` — native grid launch
- **SYCL**: `queue.parallel_for(Nblock*Nthread, functor)` — flattened range
- **OpenMP**: `#pragma omp parallel for` over blocks, inner loop over threads

**Key types:**

| Type / Pattern               | What it does                                        | GR4 relevance                                    |
|------------------------------|-----------------------------------------------------|--------------------------------------------------|
| `device` class (per-backend) | Uniform `Malloc`/`Free`/`LaunchKernel`/`Memcpy` API | medium — validates our `DeviceContext` design    |
| `deviceFunction` macro       | `__device__` (CUDA) / empty (SYCL, OpenMP)          | low — we use `if constexpr`                      |
| Functor-as-kernel            | Stateful callable with captured device pointers     | high — clean pattern for simple element-wise ops |

**What's transferable:**

1. **Functor dispatch as the lowest-level primitive** — the pattern of a stateful callable
   dispatched via `LaunchKernel(N, functor)` is the simplest possible kernel abstraction. For
   GR4, this maps to how `ExecutionStrategy` dispatches auto-parallelised `processOne`: the
   block's state + `processOne` method form a functor, dispatched over N samples. The
   gpuAgnosticFunctor pattern validates that this works cleanly across CUDA/SYCL/OpenMP.
2. **CUDA as a first-class backend** — gpuAgnosticFunctor shows the minimal API surface needed
   to add a CUDA backend: `cudaMalloc`, `cudaMemcpy`, `<<<grid, block>>>` launch. Our
   `DeviceContextCUDA` (deferred to Phase 2) would follow this pattern: implement the
   `DeviceContext` virtual interface using these same CUDA calls.

**What we do NOT adopt:**

- **Preprocessor-based backend selection** — gpuAgnosticFunctor builds three separate
  executables. We need runtime backend coexistence (SYCL + GLSL in the same process). Our
  virtual `DeviceContext` hierarchy is the right approach.
- **Manual memory management** (`Malloc`/`Free` with raw `void**`) — our PMR chain
  (`UsmMemoryResource` → `CircularBuffer` → `Port`) is type-safe and RAII.
- **No async/streaming** — all operations are blocking. We need overlapped H2D/compute/D2H
  for streaming signal processing.
- **No shared memory, barriers, or sub-group ops** — only global-memory element-wise kernels.
- **No FFT or signal-processing primitives** — purely element-wise array operations.
- **No error handling, no device selection, no multi-device** — research prototype only.

**Gaps in gpuAgnosticFunctor that we already fill:**

- Runtime backend coexistence → `DeviceContext` hierarchy
- Type-safe memory → PMR + `DeviceBuffer` RAII
- Shader specialisation → `ShaderFragment` + `ShaderCache`
- Complex arithmetic → `gr::complex<T>`
- FFT → `SyclFFT` + `GlslFFT` (Van Loan Stockham)
- Streaming dispatch → `HostToDevice<T>` / `DeviceToHost<T>` with adaptive chunking

### 16.4 Summary — what each reference contributes

**Comparison of the three references against GR4's existing device layer:**

| Capability                   | GR4 (current)                        | TinyCompute          | Tutorial          | gpuAgnosticFunctor     |
|------------------------------|--------------------------------------|----------------------|-------------------|------------------------|
| Backend abstraction          | virtual `DeviceContext` (runtime)    | CRTP (compile)       | —                 | preprocessor (compile) |
| Backends supported           | SYCL, GLSL, CPU                      | OpenGL only          | OpenGL only       | CUDA, SYCL, OpenMP     |
| Memory management            | PMR + RAII (`DeviceBuffer`)          | manual               | manual            | manual `void**`        |
| Kernel description           | `ShaderFragment` (GLSL string)       | kernel-as-struct     | raw GLSL          | functor class          |
| CPU/GPU parity               | `DeviceContextCpu` fallback          | full (same `main()`) | —                 | separate executables   |
| Shader caching               | `ShaderCache` (LRU, hash-keyed)      | —                    | —                 | —                      |
| Shader fusion                | `ShaderFusion` (element-wise chain)  | —                    | —                 | —                      |
| Runtime specialisation       | baked constants in GLSL              | —                    | —                 | —                      |
| Shared memory / barriers     | SYCL `local_accessor` + `barrier()`  | —                    | `glMemoryBarrier` | —                      |
| FFT                          | Van Loan Stockham (SYCL + GLSL)      | —                    | —                 | —                      |
| Complex arithmetic           | `gr::complex<T>` (device-safe)       | —                    | —                 | —                      |
| Streaming / async            | `HostToDevice<T>` chunking           | —                    | —                 | —                      |
| Multi-dimensional dispatch   | 1D only                              | `DimTraits` 1D/2D/3D | 2D textures       | 1D only                |
| Typed buffer bindings        | untyped `inputBuffer`/`outputBuffer` | `BufferBinding<T,B>` | `SSBO<T>`         | raw `void*`            |
| Transpilation (C++ → shader) | —                                    | Clang AST → GLSL     | —                 | —                      |

**Adoption roadmap (revised):**

| What                                                     | Source             | Phase       | Priority |
|----------------------------------------------------------|--------------------|-------------|----------|
| **Kernel-as-struct** for `ShaderFragment` evolution      | TinyCompute        | Phase 2     | high     |
| **CPU/GPU parity** for shader-based blocks               | TinyCompute        | Phase 2     | high     |
| **Shared-memory / barrier primitives** in kernel structs | — (we build)       | Phase 2     | high     |
| **Typed buffer bindings** in `ShaderFragment`            | TinyCompute        | Phase 2     | medium   |
| **`DimTraits`** for 2D dispatch (spectrogram)            | TinyCompute        | Phase 2     | medium   |
| **Functor dispatch pattern** for `DeviceContextCUDA`     | gpuAgnosticFunctor | Phase 2     | medium   |
| **CUDA backend minimal API surface** (reference impl)    | gpuAgnosticFunctor | Phase 2     | medium   |
| **Query workgroup size** post-compilation                | Tutorial           | Phase 1 fix | low      |
| **`PixelConcept`** for visualisation blocks              | TinyCompute        | Phase 3+    | low      |
