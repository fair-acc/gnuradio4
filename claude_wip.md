# GPU Compute Integration for GNU Radio 4.0

## Status: WIP Design — Phase 1

**Author**: Ralph J. Steinhagen (GSI/FAIR)
**Date**: 2026-02-16
**Branch**: `syclExperiments` on `fair-acc/gnuradio4`

---

## 1. Executive Summary

This document describes the design for integrating GPU compute into the GNU Radio 4.0 (GR4) framework via SYCL (
primarily AdaptiveCpp) with a vendor-agnostic abstraction layer that can be extended to support native CUDA and ROCm
code paths. The design preserves GR4's existing `Block<T>` user API — users who can write `processOne(T)` or
`processBulk(...)` for CPU execution can, with minimal changes, run the same algorithms on GPUs.

Two primary use-cases drive the design:

- **Case A — Standalone GPU blocks**: A single block receives data from the CPU pipeline, transfers it to the GPU for
  heavy compute, and returns results. Chunking and adaptive batching amortize host↔device transfer costs.
- **Case B — Composable GPU sub-graphs**: A chain of GPU-capable blocks executes entirely on-device. Explicit transition
  blocks mark the host↔device boundaries. Data flows between GPU blocks without returning to host memory. The sub-graph
  is scheduled as a unit, triggered by the parent CPU scheduler.

---

## 2. Design Goals and Constraints

### 2.1 Goals

1. **Preserve the Block API**: Users write `U processOne(T val, ...)` or `processBulk(...)` as before. The framework
   handles kernel generation, memory transfers, and scheduling.
2. **GPU-agnostic abstraction**: The primary implementation uses SYCL via AdaptiveCpp's generic SSCP compiler. A thin
   abstraction layer (inspired by [gpuAgnosticFunctor](https://github.com/NickKarpowicz/gpuAgnosticFunctor/)) allows
   future extension to native CUDA and ROCm code paths.
3. **Zero cost for CPU-only graphs**: CPU code paths must not be affected by the GPU extensions. Compile guards disable
   GPU features cleanly when SYCL is unavailable.
4. **Edge-based data flow**: The graph model of blocks, ports, and edges is preserved. GPU-to-GPU edges are a new edge
   type, not an abandonment of the edge concept.
5. **Tag and settings compatibility**: GPU blocks participate in GR4's tag propagation and settings infrastructure. Work
   is split at tag boundaries when tags affect block settings.
6. **CI-testable without GPU hardware**: All GPU blocks must be testable via AdaptiveCpp's CPU/OpenMP backend fallback.
7. **float32 focus**: Initial GPU support targets `float` and `std::complex<float>` (via SSCP flow). No 64-bit double
   requirement on GPU.

### 2.2 Constraints

- **Primary GPU target**: NVIDIA, with AMD and Intel kept compatible via SYCL abstraction.
- **Compiler**: AdaptiveCpp with generic SSCP (`--acpp-targets=generic`) requires clang ≥14; GR4's pipeline uses clang
  ≥20. GCC ≥14 and Emscripten builds must compile cleanly with GPU features disabled.
- **No external FFT libraries**: GPU FFT must be a vanilla, vendor-neutral SYCL implementation (radix-2 Cooley-Tukey,
  power-of-2 sizes 1024–65536). Numerically correct; up to ~5× slower than cuFFT is acceptable.
- **No external GPU library dependencies** beyond AdaptiveCpp itself for Phase 1.

---

## 3. Architecture Overview

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Block Code                       │
│          processOne(T) / processBulk(...)                │
│      Optional: processGpu(DeviceContext&, ...)           │
├─────────────────────────────────────────────────────────┤
│               GR4 GPU Abstraction Layer                  │
│  ┌───────────┐  ┌──────────┐  ┌───────────────────┐    │
│  │  Kernel   │  │ Device   │  │  GPU Sub-graph    │    │
│  │ Generator │  │ Memory   │  │  Scheduler        │    │
│  │(processOne│  │ Manager  │  │  (event-based     │    │
│  │ → kernel) │  │ (USM)    │  │   ordering)       │    │
│  └─────┬─────┘  └────┬─────┘  └────────┬──────────┘    │
├────────┼──────────────┼─────────────────┼───────────────┤
│        │   GPU Backend Abstraction      │                │
│  ┌─────▼──────────────▼─────────────────▼──────────┐    │
│  │              DeviceContext                        │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │    │
│  │  │  SYCL    │ │  CUDA    │ │   ROCm/HIP       │ │    │
│  │  │ Backend  │ │ Backend  │ │   Backend         │ │    │
│  │  │(Adaptive │ │ (future) │ │   (future)        │ │    │
│  │  │  Cpp)    │ │          │ │                   │ │    │
│  │  └──────────┘ └──────────┘ └──────────────────┘ │    │
│  └──────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                   GR4 Core (unmodified)                   │
│   Block<T>  ·  CircularBuffer  ·  Scheduler  ·  Tags    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decision: Device Edges (not CircularBuffer)

CPU-side `CircularBuffer` uses atomic counters for lock-free single-writer/multi-reader coordination between concurrent
threads. This mechanism is fundamentally incompatible with GPU execution:

- `std::atomic` in USM shared memory does not have well-defined cross-device semantics. SYCL 2020 provides
  `sycl::atomic_ref` for device-side atomics, but memory ordering guarantees between host and device are weaker and
  implementation-dependent.
- Even on hardware with coherent shared memory (Grace-Hopper, MI300), fine-grained atomic coordination between a
  CPU-side scheduler thread and a GPU kernel would be unreliable and a performance disaster due to coherence traffic.
- USM shared allocators were investigated and the atomic counters proved to be the limiting factor.

**Key insight**: Within a GPU sub-graph, lock-free coordination is unnecessary. GPU blocks are submitted as kernels to a
SYCL queue — ordering is either implicit (in-order queue) or explicit (SYCL events as dependencies). A device-side "
edge" between two GPU blocks is just a USM device allocation that kernel N writes and kernel N+1 reads, with
synchronization provided by the queue/event mechanism.

Therefore:

- **CPU edges**: `CircularBuffer<T>` as today (host memory, atomic counters, lock-free). **Completely unmodified.**
- **Host→Device transition**: `HostToDevice<T>` block bulk-copies from `CircularBuffer` into a device-resident USM
  allocation. This is where adaptive chunking lives.
- **Device edges**: `DeviceBuffer<T>` — simple double-buffered device USM allocations. No atomics. Synchronization via
  SYCL event dependencies managed by the GPU sub-graph scheduler.
- **Device→Host transition**: `DeviceToHost<T>` block copies results back into a `CircularBuffer` for downstream CPU
  blocks.

The "edge" concept is preserved — device edges still connect ports between blocks — but the backing implementation is
fundamentally different from `CircularBuffer`. Existing `CircularBuffer` code is completely unaffected (zero cost for
CPU-only graphs).

**Open question (spike)**: Could a USM shared allocator for `CircularBuffer` work for simple SISO Case A blocks,
avoiding the need for explicit `HostToDevice` blocks? Worth a spike to test atomic counter behaviour on shared memory.
If it works, it simplifies Case A. This does not affect the overall architecture.

---

## 4. Key Abstractions

### 4.1 DeviceContext

The vendor-agnostic wrapper around a compute device and its command queue. Inspired by the gpuAgnosticFunctor pattern: a
thin struct that forwards operations to the active backend.

```cpp
namespace gr::gpu {

enum class Backend { sycl, cuda, rocm, cpu_fallback };

// Compile-time backend selection via build configuration
#if defined(GR_HAS_SYCL)
inline constexpr Backend kActiveBackend = Backend::sycl;
#elif defined(GR_HAS_CUDA)
inline constexpr Backend kActiveBackend = Backend::cuda;
#elif defined(GR_HAS_ROCM)
inline constexpr Backend kActiveBackend = Backend::rocm;
#else
inline constexpr Backend kActiveBackend = Backend::cpu_fallback;
#endif

class DeviceContext {
public:
    // Construction: picks the best available device
    explicit DeviceContext(Backend backend = kActiveBackend);

    // Queue/stream management
    void submit(auto&& kernel);          // Submit a kernel for execution
    void wait();                          // Wait for all submitted work
    void synchronize_event(Event& evt);   // Wait for a specific event

    // Memory management (USM-style)
    template<typename T>
    T* allocate_device(std::size_t count);

    template<typename T>
    T* allocate_shared(std::size_t count);  // accessible from host and device

    template<typename T>
    void deallocate(T* ptr);

    // Transfers
    template<typename T>
    Event copy_host_to_device(const T* host, T* device, std::size_t count);

    template<typename T>
    Event copy_device_to_host(const T* device, T* host, std::size_t count);

    // Backend-specific access (for advanced users)
    // Guarded by #if GR_HAS_SYCL / GR_HAS_CUDA etc.
    auto& native_queue();  // sycl::queue& or cudaStream_t etc.
};

} // namespace gr::gpu
```

### 4.2 DeviceBuffer

Device-resident buffer for GPU sub-graph edges. **Not** a `CircularBuffer` — synchronisation is via SYCL events, not
atomics.

```cpp
namespace gr::gpu {

template<typename T>
class DeviceBuffer {
    T*          _data      = nullptr;
    std::size_t _capacity  = 0;
    std::size_t _size      = 0;      // current valid sample count
    Event       _ready;               // signals when data is ready to read
    DeviceContext* _ctx    = nullptr;

public:
    DeviceBuffer(DeviceContext& ctx, std::size_t capacity);
    ~DeviceBuffer();

    T*          data()      { return _data; }
    std::size_t size()      { return _size; }
    std::size_t capacity()  { return _capacity; }

    void set_size(std::size_t n)   { _size = n; }
    void set_ready(Event evt)      { _ready = std::move(evt); }
    Event ready() const            { return _ready; }
};

} // namespace gr::gpu
```

### 4.3 Vendor-Agnostic Parallel Execution

The core abstraction: execute a functor over N work items, dispatched to whichever backend is active.

```cpp
namespace gr::gpu {

// --- Parallel execution primitive ---
template<typename Func>
void parallel_for(DeviceContext& ctx, std::size_t count, Func&& f) {
#if defined(GR_HAS_SYCL)
    ctx.native_queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> idx) {
            f(idx[0]);
        });
    });
#elif defined(GR_HAS_CUDA)
    // CUDA kernel launch wrapper
    cuda_parallel_for<<<grid, block>>>(count, std::forward<Func>(f));
#elif defined(GR_HAS_ROCM)
    // HIP kernel launch wrapper
    hip_parallel_for<<<grid, block>>>(count, std::forward<Func>(f));
#else
    // CPU fallback: plain loop
    for (std::size_t i = 0; i < count; ++i) {
        f(i);
    }
#endif
}

// --- Memory operations ---
template<typename T>
T* allocate_device(DeviceContext& ctx, std::size_t count) {
#if defined(GR_HAS_SYCL)
    return sycl::malloc_device<T>(count, ctx.native_queue());
#elif defined(GR_HAS_CUDA)
    T* ptr; cudaMalloc(&ptr, count * sizeof(T)); return ptr;
#elif defined(GR_HAS_ROCM)
    T* ptr; hipMalloc(&ptr, count * sizeof(T)); return ptr;
#else
    return new T[count];  // CPU fallback
#endif
}

// deallocate, copy_h2d, copy_d2h follow the same pattern

} // namespace gr::gpu
```

### 4.4 GpuCapable Concept and Block Traits

Blocks opt into GPU execution via a trait. The framework detects at compile time whether a block's `processOne` or
`processBulk` can be mapped to a GPU kernel.

```cpp
namespace gr::gpu {

// Trait: block declares itself as GPU-capable
template<typename TBlock>
concept GpuCapableBlock = requires {
    { TBlock::gpu_capable } -> std::convertible_to<bool>;
} && TBlock::gpu_capable;

// Execution policy NTTP for block instantiation
enum class ExecutionTarget { cpu, gpu, auto_select };

} // namespace gr::gpu
```

Usage at block definition — **only addition is `gpu_capable = true`**, no changes to block logic:

```cpp
template<typename T>
struct MultiplyConst : gr::Block<MultiplyConst<T>> {
    using Description = Doc<R"(Multiplies input by a constant factor.)">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    float          factor = 1.0f;

    static constexpr bool gpu_capable = true;

    GR_MAKE_REFLECTABLE(MultiplyConst, in, out, factor);

    [[nodiscard]] constexpr T processOne(T input) const noexcept {
        return input * static_cast<T>(factor);
    }
};
```

### 4.5 Transition Blocks

Explicit blocks that mark host↔device boundaries:

```cpp
template<typename T>
struct HostToDevice : gr::Block<HostToDevice<T>> {
    gr::PortIn<T>  in;   // CPU-side CircularBuffer edge
    gr::PortOut<T> out;  // device-side DeviceBuffer edge

    // Chunking configuration (user-tunable, auto-adjusted)
    std::size_t min_chunk_size  = 1024;
    std::size_t max_chunk_size  = 65536;
    std::size_t chunk_size      = 4096;  // current adaptive value

    static constexpr bool gpu_capable = true;

    GR_MAKE_REFLECTABLE(HostToDevice, in, out,
                        min_chunk_size, max_chunk_size, chunk_size);

    gr::work::Status processBulk(
        const gr::ConsumableSpan auto& inSpan,
        gr::PublishableSpan auto& outSpan);
};

template<typename T>
struct DeviceToHost : gr::Block<DeviceToHost<T>> {
    gr::PortIn<T>  in;   // device-side DeviceBuffer edge
    gr::PortOut<T> out;  // CPU-side CircularBuffer edge

    static constexpr bool gpu_capable = true;

    GR_MAKE_REFLECTABLE(DeviceToHost, in, out);

    gr::work::Status processBulk(
        const gr::ConsumableSpan auto& inSpan,
        gr::PublishableSpan auto& outSpan);
};
```

---

## 5. Graph Construction API

### 5.1 Case A: Standalone GPU Block

```cpp
gr::Graph flow;
auto& src       = flow.emplaceBlock<SignalSource<float>>();
auto& h2d       = flow.emplaceBlock<gpu::HostToDevice<float>>();
auto& gpuMul    = flow.emplaceBlock<MultiplyConst<float>>({{"factor", 2.0f}});
auto& d2h       = flow.emplaceBlock<gpu::DeviceToHost<float>>();
auto& sink      = flow.emplaceBlock<DataSink<float>>();

flow.connect(src, "out", h2d, "in");
flow.connect(h2d, "out", gpuMul, "in");   // device edge
flow.connect(gpuMul, "out", d2h, "in");   // device edge
flow.connect(d2h, "out", sink, "in");
```

### 5.2 Case B: GPU Sub-graph

```cpp
auto gpuCtx = gpu::DeviceContext{};

gr::Graph gpuGraph;  // sub-graph, all blocks GPU
auto& fir    = gpuGraph.emplaceBlock<FirFilter<float>>({{"taps", taps}});
auto& mul    = gpuGraph.emplaceBlock<MultiplyConst<float>>({{"factor", 0.5f}});
gpuGraph.connect(fir, "out", mul, "in");   // device edge (no host roundtrip)

gr::Graph mainFlow;
auto& src   = mainFlow.emplaceBlock<SignalSource<float>>();
auto& sink  = mainFlow.emplaceBlock<DataSink<float>>();
// addGpuSubGraph inserts HostToDevice/DeviceToHost automatically
mainFlow.addGpuSubGraph(gpuCtx, gpuGraph, src, "out", sink, "in");
```

---

## 6. Kernel Generation from processOne

### 6.1 Single Block Kernel Wrapping

For a block with `processOne`, the framework generates:

```cpp
// Pseudocode of the generated kernel dispatch
template<typename TBlock>
void launch_processOne_kernel(
    DeviceContext& ctx,
    const typename TBlock::input_type* __restrict__ in,
    typename TBlock::output_type* __restrict__ out,
    std::size_t count,
    DeviceBlockState<TBlock>& device_state)
{
    parallel_for(ctx, count, [=](std::size_t i) {
        // device_state contains mirrored settings (e.g., factor)
        out[i] = TBlock::apply(in[i], device_state);
    });
}
```

The `TBlock::apply` is a static device-compatible function extracted from `processOne`. For blocks where `processOne` is
`const` and uses only reflected member fields, this extraction is automatic via compile-time reflection.

### 6.2 Block State Mirroring

Block member fields (settings) are mirrored to device-accessible memory:

```cpp
template<typename TBlock>
struct DeviceBlockState {
    // Generated via compile-time reflection over GR_MAKE_REFLECTABLE fields.
    // Contains only the fields needed by processOne/processBulk.
    // Example for MultiplyConst: just `float factor;`
    // Excludes port definitions and non-data members.
};

// Sync host → device when settings change (detected via tag processing)
template<typename TBlock>
void sync_state_to_device(
    DeviceContext& ctx,
    const TBlock& host_block,
    DeviceBlockState<TBlock>* device_state);
```

### 6.3 Automatic Kernel Fusion (Phase 1.5)

When multiple consecutive `processOne` blocks form a chain within a GPU sub-graph:

```
BlockA::processOne → BlockB::processOne → BlockC::processOne
```

The framework fuses them into a single kernel:

```cpp
parallel_for(ctx, count, [=](std::size_t i) {
    auto a_out = BlockA::apply(in[i], a_state);
    auto b_out = BlockB::apply(a_out, b_state);
    out[i]     = BlockC::apply(b_out, c_state);
});
```

**Fusion rules:**

- Only `processOne` blocks are fusable (element-wise, no cross-element dependencies).
- Fusion breaks at any `processBulk` block (which becomes a separate kernel launch).
- Settings for each block are namespaced in the fused state struct (e.g., `a_state.factor`, `b_state.factor`). This
  handles blocks with same-named member fields correctly.
- Fusion breaks at tag boundaries that affect any block in the chain.
- **Phase 1 baseline**: no fusion, separate kernel per block. **Phase 1.5**: automatic fusion for `processOne` chains.

---

## 7. GPU Sub-graph Scheduler

### 7.1 Relationship to Parent Scheduler

The GPU sub-graph scheduler runs as a managed sub-graph within GR4's existing scheduler hierarchy. This leverages the
existing managed/unmanaged sub-graph distinction already present in GR4.

```
Main Scheduler (CPU)
  ├── CPU Block A
  ├── CPU Block B
  ├── GPU Sub-graph Scheduler  ← triggered when HostToDevice has data
  │     ├── kernel launch: FirFilter
  │     ├── kernel launch: MultiplyConst
  │     └── (results ready → DeviceToHost fires)
  ├── CPU Block C
  └── ...
```

The GPU sub-graph scheduler is triggered from the parent scheduler (which flags new work/data) but runs largely
independently, processing work/samples in its own time-scale/scheduling paradigm.

### 7.2 Execution Flow

1. **Parent scheduler** runs `HostToDevice` block. It copies a chunk from `CircularBuffer` to a `DeviceBuffer` and
   signals the GPU sub-graph.
2. **GPU sub-graph scheduler** iterates over blocks in topological order:
    - For each block: check if input `DeviceBuffer` has a ready event.
    - Submit the kernel, passing input/output `DeviceBuffer` pointers.
    - Record the completion event on the output `DeviceBuffer`.
3. When the last GPU block completes, `DeviceToHost` copies results to its output `CircularBuffer`.
4. **Parent scheduler** sees data in the output `CircularBuffer` and schedules downstream CPU blocks.

### 7.3 Tag Handling in GPU Sub-graphs

```
incoming samples: [ s0 s1 s2 s3 | TAG(factor=3.0) | s4 s5 s6 s7 s8 s9 ... ]
                  ← chunk 1 →                       ← chunk 2 →
```

1. `HostToDevice` scans incoming tags for keys matching reflected fields of any block in the sub-graph. This is
   straightforward because those tags have keys that match compile-time reflectable member fields of the blocks.
2. If a settings-affecting tag is found at sample index N:
    - Chunk 1: samples 0..N-1, launched with current settings.
    - Settings update: `sync_state_to_device(...)` for affected block(s).
    - Chunk 2: samples N..end, launched with new settings.
3. Tags that are purely informational (no matching keys in any block) are forwarded without splitting. This minimises
   the stop/start overhead.

For **Case B composite blocks**: tags propagate between GPU blocks without roundtripping to the host. The sub-graph
chain is only interrupted if there are tags where one of the blocks needs to take action.

---

## 8. Adaptive Chunking

### 8.1 Algorithm

The `HostToDevice` block maintains a throughput estimator:

```
chunk_size ∈ [min_chunk_size, max_chunk_size]

For each transfer+compute cycle:
  measured_throughput = chunk_size / elapsed_time

  if measured_throughput > previous_throughput * 1.05:
      chunk_size = min(chunk_size * 2, max_chunk_size)     // scale up
  elif measured_throughput < previous_throughput * 0.9:
      chunk_size = max(chunk_size / 2, min_chunk_size)     // scale down
  // else: keep current size (stable region)
```

The user provides `min_chunk_size` and `max_chunk_size` as block settings. The framework adapts within these bounds.

### 8.2 Latency vs. Throughput

- Small chunks → lower latency, higher overhead ratio.
- Large chunks → higher throughput, higher latency.
- Tags that require work-splitting naturally limit chunk sizes.

---

## 9. Vanilla GPU FFT

### 9.1 Algorithm

Radix-2 Cooley-Tukey decimation-in-frequency, implemented as a sequence of SYCL `parallel_for` kernel launches (one per
butterfly stage).

```
FFT size N = 2^k

For stage s = 0 .. k-1:
    parallel_for(N/2 work items):
        Compute butterfly for this stage
        Twiddle factor: W_N^{bit-reversed index}

Final: bit-reversal permutation (single parallel_for)
```

### 9.2 Interface

```cpp
namespace gr::gpu {

class GpuFft {
    DeviceContext& _ctx;
    std::size_t    _fft_size;
    // Pre-computed twiddle factors on device
    std::complex<float>* _d_twiddles = nullptr;

public:
    GpuFft(DeviceContext& ctx, std::size_t fft_size);
    ~GpuFft();

    // In-place forward FFT
    Event forward(std::complex<float>* d_data, std::size_t count);
    // In-place inverse FFT
    Event inverse(std::complex<float>* d_data, std::size_t count);
    // Batched: count / fft_size independent FFTs
    Event forward_batch(std::complex<float>* d_data, std::size_t count);
};

} // namespace gr::gpu
```

### 9.3 Scope

- Power-of-2 sizes only: 1024, 2048, 4096, 8192, 16384, 32768, 65536.
- `std::complex<float>` (float32) only.
- Numerically correct (validated against CPU reference).
- Performance target: within ~5× of cuFFT for the supported sizes.
- Used internally by `FirFilter` for overlap-save frequency-domain convolution.
- No external library dependencies.

---

## 10. Extension Point for Advanced Users

Users who need vendor-specific code can override `processGpu`:

```cpp
template<typename T>
struct AdvancedFirFilter : gr::Block<AdvancedFirFilter<T>> {
    // ... ports, settings, GR_MAKE_REFLECTABLE ...

    static constexpr bool gpu_capable = true;

    // Standard CPU path
    gr::work::Status processBulk(
        const gr::ConsumableSpan auto& in, gr::PublishableSpan auto& out);

    // Custom GPU path — takes priority over auto-generated kernel
    gr::work::Status processGpu(
        gpu::DeviceContext& ctx,
        const T* __restrict__ d_in, std::size_t in_count,
        T* __restrict__ d_out, std::size_t out_capacity)
    {
#if defined(GR_HAS_SYCL)
        // Custom SYCL kernel
        ctx.native_queue().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{in_count}, [=](sycl::id<1> i) {
                // user's custom SYCL code
            });
        });
#elif defined(GR_HAS_CUDA)
        // Custom CUDA kernel launch
        my_fir_cuda_kernel<<<grid, block>>>(d_in, d_out, in_count);
#endif
        return gr::work::Status::OK;
    }
};
```

The `invokeWork(...)` helper checks for `processGpu` first (when running on a GPU context), then falls back to wrapping
`processOne`/`processBulk`.

### Compile Guards

```cpp
// Top-level detection (set by CMake based on available backends)
// CMakeLists.txt: option(GR_ENABLE_GPU "Enable GPU support" OFF)
//   → sets GR_HAS_SYCL, GR_HAS_CUDA, GR_HAS_ROCM as appropriate

#if defined(GR_HAS_SYCL) || defined(GR_HAS_CUDA) || defined(GR_HAS_ROCM)
  #define GR_HAS_GPU 1
#else
  #define GR_HAS_GPU 0
#endif

// Usage in user blocks:
#if GR_HAS_GPU
    gr::work::Status processGpu(gpu::DeviceContext& ctx, ...);
#endif
```

All GPU-related headers are guarded so that GCC and Emscripten builds compile cleanly with GPU features absent.

---

## 11. Motivating Use-Cases

### 11.1 Case A: GPU-accelerated FIR Filter (standalone)

**Scenario**: A wideband SDR receiver digitizes at 100 MS/s. A large FIR filter (512+ taps) decimates and channelizes
the signal. On CPU, this is the bottleneck.

**Graph**:

```
SignalSource → HostToDevice → FirFilter(GPU) → DeviceToHost → Demodulator(CPU) → Sink
```

**Why GPU helps**: FIR filtering via overlap-save in the frequency domain requires FFTs and element-wise
multiplication — massively parallel, data-independent operations. A 4096-tap FIR filter on a 65536-sample chunk requires
2 FFTs + 1 pointwise multiply, all of which saturate GPU compute.

**Implementation**: `FirFilter` uses `processBulk` with a custom `processGpu` that calls the vanilla GPU FFT and
performs the overlap-save bookkeeping.

### 11.2 Case A: GPU MultiplyConst (processOne proof-of-concept)

**Scenario**: Simplest possible GPU block. Validates the `processOne → kernel` machinery.

**Graph**:

```
SignalSource → HostToDevice → MultiplyConst(GPU) → DeviceToHost → Sink
```

**Why it matters**: This is the "hello world" of GPU integration. If this works cleanly, the entire `processOne`
auto-kernelization pathway is validated.

### 11.3 Case B: GPU Signal Conditioning Chain

**Scenario**: A chain of simple DSP operations: DC removal → gain → IQ imbalance correction → frequency shift. Each is a
`processOne` block.

**Graph**:

```
Source → HostToDevice → [DcRemoval → Gain → IqCorrection → FreqShift](GPU sub-graph) → DeviceToHost → Sink
```

**Why GPU helps**: Individually, each block is too trivial for GPU. But fused into a single kernel, the chain processes
millions of samples per kernel launch with minimal overhead. The fusion turns 4 separate memory-bandwidth-limited
operations into 1 compute-limited operation.

### 11.4 Case B: Polyphase Channelizer

**Scenario**: A polyphase channelizer splits a wideband signal into N sub-channels. Each sub-channel is filtered and
gain-adjusted.

**Graph**:

```
Source → HostToDevice → [PolyphaseFilter → FFT → MultiplyConst → ...](GPU sub-graph) → DeviceToHost → per-channel Sinks
```

**Why a sub-graph**: The intermediate data between the polyphase filter, FFT, and per-channel gain is large (N ×
chunk_size). Keeping it on-device avoids N round-trips through PCIe.

**Fusion opportunity**: The per-channel `MultiplyConst` blocks are pure `processOne` — they can be fused into the tail
of the FFT kernel or run as a single vectorized kernel.

---

## 12. Phased Implementation Plan

### Phase 1.0 — Foundation

**Goal**: GPU abstraction layer compiles, allocates device memory, runs a trivial kernel on AdaptiveCpp CPU fallback.

| Step  | Task                                                                         | Manual | Claude Code |
|-------|------------------------------------------------------------------------------|--------|-------------|
| 1.0.1 | CMake infrastructure: `GR_ENABLE_GPU`, detect AdaptiveCpp, set `GR_HAS_SYCL` | 1d     | 0.5d        |
| 1.0.2 | `DeviceContext` class (SYCL backend + CPU fallback)                          | 2d     | 1d          |
| 1.0.3 | `DeviceBuffer<T>` with USM device allocation                                 | 1d     | 0.5d        |
| 1.0.4 | `parallel_for` abstraction (SYCL + CPU fallback)                             | 1d     | 0.5d        |
| 1.0.5 | Memory transfer wrappers (H2D, D2H)                                          | 1d     | 0.5d        |
| 1.0.6 | Unit tests: allocate, transfer, parallel_for on CPU fallback                 | 2d     | 1d          |
|       | **Subtotal**                                                                 | **8d** | **4d**      |

**Deliverable**: `gr::gpu::DeviceContext` and `gr::gpu::DeviceBuffer` compile and pass tests with AdaptiveCpp CPU
backend.

### Phase 1.1 — Standalone GPU Blocks (Case A)

**Goal**: `MultiplyConst<float>` runs on GPU via `processOne` auto-kernelization.

| Step  | Task                                                             | Manual  | Claude Code |
|-------|------------------------------------------------------------------|---------|-------------|
| 1.1.1 | `GpuCapableBlock` concept + `gpu_capable` trait                  | 1d      | 0.5d        |
| 1.1.2 | `DeviceBlockState<TBlock>` generation via reflection             | 3d      | 1.5d        |
| 1.1.3 | `processOne → parallel_for` kernel wrapper in `invokeWork`       | 3d      | 1.5d        |
| 1.1.4 | `HostToDevice<T>` transition block (fixed chunk size initially)  | 2d      | 1d          |
| 1.1.5 | `DeviceToHost<T>` transition block                               | 1d      | 0.5d        |
| 1.1.6 | `sync_state_to_device` for settings updates                      | 1d      | 0.5d        |
| 1.1.7 | Integration test: `Source → H2D → MultiplyConst → D2H → Sink`    | 2d      | 1d          |
| 1.1.8 | `processBulk → kernel` wrapper (for blocks that use processBulk) | 2d      | 1d          |
|       | **Subtotal**                                                     | **15d** | **7.5d**    |

**Deliverable**: End-to-end test passes: CPU source → GPU multiply → CPU sink, verified on AdaptiveCpp CPU fallback
and (if available) on an NVIDIA GPU.

**Risk note**: This is the highest-risk phase. The `processOne` → kernel mapping must interact correctly with the CRTP
`Block<T>`, compile-time reflection, and AdaptiveCpp's SSCP kernel extraction. Recommend a standalone spike before
committing to the full architecture.

### Phase 1.2 — Composable GPU Sub-graphs (Case B)

**Goal**: Two+ GPU blocks in a sub-graph, data stays on device between them.

| Step  | Task                                                                     | Manual  | Claude Code |
|-------|--------------------------------------------------------------------------|---------|-------------|
| 1.2.1 | Device edge type: `DeviceBuffer`-backed port connections                 | 3d      | 1.5d        |
| 1.2.2 | GPU sub-graph scheduler (event-based kernel ordering)                    | 4d      | 2d          |
| 1.2.3 | `addGpuSubGraph(...)` API on `Graph` (auto-inserts H2D/D2H)              | 2d      | 1d          |
| 1.2.4 | Tag boundary detection across sub-graph blocks                           | 2d      | 1d          |
| 1.2.5 | Work-splitting at tag boundaries + state re-sync                         | 2d      | 1d          |
| 1.2.6 | Integration test: `Source → [MultiplyConst → MultiplyConst](GPU) → Sink` | 2d      | 1d          |
|       | **Subtotal**                                                             | **15d** | **7.5d**    |

**Deliverable**: GPU sub-graph with two blocks, data stays on device, tags handled correctly. Data does NOT roundtrip to
host between GPU blocks.

### Phase 1.3 — Adaptive Chunking

**Goal**: `HostToDevice` automatically finds a good chunk size.

| Step  | Task                                                        | Manual | Claude Code |
|-------|-------------------------------------------------------------|--------|-------------|
| 1.3.1 | Throughput estimator in `HostToDevice`                      | 1d     | 0.5d        |
| 1.3.2 | Adaptive chunk size adjustment (double/halve within bounds) | 1d     | 0.5d        |
| 1.3.3 | Integration with tag-based work splitting                   | 1d     | 0.5d        |
| 1.3.4 | Benchmark: adaptive vs fixed chunk sizes                    | 1d     | 0.5d        |
|       | **Subtotal**                                                | **4d** | **2d**      |

### Phase 1.4 — FIR Filter and Vanilla FFT

**Goal**: A practically useful GPU block. Validates the whole stack under realistic DSP load.

| Step  | Task                                                                   | Manual  | Claude Code |
|-------|------------------------------------------------------------------------|---------|-------------|
| 1.4.1 | Vanilla radix-2 FFT in SYCL (forward, inverse, batched)                | 4d      | 2d          |
| 1.4.2 | FFT numerical validation against CPU reference (use existing SIMD-FFT) | 1d      | 0.5d        |
| 1.4.3 | `FirFilter<float>` GPU block (overlap-save, `processGpu`)              | 3d      | 1.5d        |
| 1.4.4 | Integration test: `Source → H2D → FirFilter(GPU) → D2H → Sink`         | 1d      | 0.5d        |
| 1.4.5 | Benchmark: CPU SIMD FIR vs GPU FIR at various filter/chunk sizes       | 2d      | 1d          |
|       | **Subtotal**                                                           | **11d** | **5.5d**    |

**Deliverable**: GPU FIR filter produces correct output. Benchmark identifies the crossover point where GPU becomes
faster than CPU.

### Phase 1.5 — Kernel Fusion (Optimisation)

**Goal**: Consecutive `processOne` blocks in a GPU sub-graph are fused into one kernel.

| Step  | Task                                                                         | Manual | Claude Code |
|-------|------------------------------------------------------------------------------|--------|-------------|
| 1.5.1 | Detect fusable `processOne` chains in sub-graph topology                     | 2d     | 1d          |
| 1.5.2 | Generate fused kernel (compile-time chain composition)                       | 4d     | 2d          |
| 1.5.3 | Fused `DeviceBlockState` with per-block namespacing (`a.param0`, `b.param0`) | 2d     | 1d          |
| 1.5.4 | Benchmark: fused vs unfused 4-block chain                                    | 1d     | 0.5d        |
|       | **Subtotal**                                                                 | **9d** | **4.5d**    |

### Phase 1.6 — Extension Points and Polish

| Step  | Task                                                         | Manual | Claude Code |
|-------|--------------------------------------------------------------|--------|-------------|
| 1.6.1 | `processGpu(...)` override mechanism in `invokeWork`         | 2d     | 1d          |
| 1.6.2 | `std::complex<float>` validation across pipeline (SSCP flow) | 1d     | 0.5d        |
| 1.6.3 | Compile guard validation: GCC, Emscripten clean builds       | 1d     | 0.5d        |
| 1.6.4 | Block author documentation (how to write a GPU block)        | 2d     | 1.5d        |
| 1.6.5 | CI integration for AdaptiveCpp CPU fallback tests            | 1d     | 0.5d        |
|       | **Subtotal**                                                 | **7d** | **4d**      |

### Summary

| Phase                       | Manual  | Claude Code Assisted |
|-----------------------------|---------|----------------------|
| 1.0 Foundation              | 8d      | 4d                   |
| 1.1 Standalone GPU (Case A) | 15d     | 7.5d                 |
| 1.2 Sub-graphs (Case B)     | 15d     | 7.5d                 |
| 1.3 Adaptive Chunking       | 4d      | 2d                   |
| 1.4 FIR + FFT               | 11d     | 5.5d                 |
| 1.5 Kernel Fusion           | 9d      | 4.5d                 |
| 1.6 Polish                  | 7d      | 4d                   |
| **Total**                   | **69d** | **35d**              |

At ~60% allocation: **~12 weeks (Claude Code assisted)** or **~23 weeks (manual)**.

**Recommended order**: 1.0 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6

(Adaptive chunking before FIR because it's simpler and immediately useful for benchmarking.)

---

## 13. Open Questions and Risks

### 13.1 Open Questions (to revisit during implementation)

1. **CircularBuffer allocator for USM shared**: Could a USM shared allocator for `CircularBuffer` work for simple SISO
   Case A blocks (avoiding the need for explicit `HostToDevice` blocks)? Worth a spike to test atomic counter behaviour
   on shared memory. If it works, it's a simpler path for Case A.

2. **SYCL event overhead**: For sub-graphs with many small blocks, per-kernel event tracking may dominate. Profiling
   needed to determine if in-order queue (implicit ordering) is sufficient.

3. **AdaptiveCpp SSCP and `std::complex`**: Needs validation that `std::complex<float>` member functions (
   multiplication, addition) compile correctly through the SSCP flow for all target backends.

4. **Reflection depth for DeviceBlockState**: How deep does the reflection need to go? Simple scalar settings are
   straightforward. What about `std::vector<float>` taps in a FIR filter? These need device-side copies of
   variable-length data — will require special handling (device-side copy + pointer in DeviceBlockState).

### 13.2 Risks

| Risk                                                                  | Impact                         | Mitigation                                                                                                                       |
|-----------------------------------------------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| AdaptiveCpp SSCP does not handle GR4's heavy template metaprogramming | High — kernels won't compile   | Early spike in Phase 1.0: compile a trivial `processOne` block through SSCP before committing to architecture                    |
| Kernel fusion requires constexpr evaluation beyond what SSCP supports | Medium — fusion deferred       | Phase 1.5 is optional; separate kernels (Phase 1.2) work without fusion                                                          |
| Atomic counters in USM shared memory don't work reliably              | Low — already mitigated        | Design uses `DeviceBuffer` (no atomics) for device edges; `CircularBuffer` unchanged                                             |
| Vanilla FFT is >5× slower than cuFFT, making GPU FIR uncompetitive    | Medium — FFT use-case weakened | The FIR filter is one use-case; `processOne` chains (channelizer, signal conditioning) don't need FFT and still benefit from GPU |
| Tag splitting causes excessive small kernel launches                  | Medium — throughput loss       | Batch informational tags; only split on settings-affecting tags (detectable at compile time via reflection)                      |

---

## 14. File/Directory Structure (proposed)

```
gnuradio4/
├── core/
│   └── include/gnuradio-4.0/
│       ├── Block.hpp                    (existing, minor additions for gpu traits)
│       ├── gpu/
│       │   ├── GpuConfig.hpp            (compile guards, backend detection)
│       │   ├── DeviceContext.hpp         (vendor-agnostic device wrapper)
│       │   ├── DeviceBuffer.hpp         (device-resident edge buffer)
│       │   ├── DeviceBlockState.hpp     (reflection-based state mirroring)
│       │   ├── KernelGenerator.hpp      (processOne → kernel mapping)
│       │   ├── KernelFusion.hpp         (Phase 1.5: chain fusion)
│       │   ├── GpuSubGraphScheduler.hpp (event-based kernel ordering)
│       │   ├── ParallelFor.hpp          (vendor-agnostic parallel execution)
│       │   └── GpuFft.hpp              (vanilla radix-2 FFT)
│       └── ...
├── blocks/
│   └── gpu/
│       ├── HostToDevice.hpp
│       └── DeviceToHost.hpp
├── core/test/
│   └── gpu/
│       ├── qa_DeviceContext.cpp
│       ├── qa_DeviceBuffer.cpp
│       ├── qa_KernelGenerator.cpp
│       ├── qa_GpuSubGraph.cpp
│       ├── qa_GpuFft.cpp
│       └── qa_GpuFirFilter.cpp
└── cmake/
    └── FindAdaptiveCpp.cmake
```

---

## 15. Glossary

| Term               | Definition                                                                                                                                                                              |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SSCP               | Single-Source, Single Compiler Pass — AdaptiveCpp's generic compilation flow. Parses code once, embeds backend-independent LLVM IR, JIT-compiles to target at runtime.                  |
| USM                | Unified Shared Memory — SYCL memory model where host and device share a pointer address space. Three tiers: device (GPU only), shared (both), host (CPU only, device-accessible).       |
| H2D / D2H          | Host-to-Device / Device-to-Host memory transfer                                                                                                                                         |
| DeviceBuffer       | GR4's device-resident buffer type for GPU sub-graph edges. Not CircularBuffer — uses event-based sync instead of atomics.                                                               |
| DeviceContext      | GR4's vendor-agnostic wrapper around a GPU device and its command queue. Dispatches to SYCL, CUDA, ROCm, or CPU fallback.                                                               |
| Kernel fusion      | Combining multiple `processOne` blocks into a single GPU kernel launch to reduce memory bandwidth pressure and kernel launch overhead.                                                  |
| Tag boundary       | A sample position in the stream where a Tag requires a settings change in one or more blocks. GPU work is split at these boundaries.                                                    |
| gpuAgnosticFunctor | Pattern by Nick Karpowicz (LightwaveExplorer) where a functor is dispatched to CUDA, SYCL, or CPU via compile-time backend selection. Inspiration for GR4's `parallel_for` abstraction. |
