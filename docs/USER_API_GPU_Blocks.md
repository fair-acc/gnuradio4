# Writing blocks that run on a GPU or other accelerator

A block runs on an accelerator when its `compute_domain` names one. Nothing else about the block changes: you write
ordinary C++, with ordinary settings, and one processing function. This document describes the three ways to write
that function, what a kernel may and may not touch, and how array settings reach the device.

Runnable versions of every example below live in `core/test/qa_DeviceBlockStyles.cpp`.

## Compute domains

A domain is `kind[:backend[:deviceIndex]]`. The **kind** says where the memory lives; the **backend** says what
executes the work.

| domain                                        | memory                           | executes on                            |
| --------------------------------------------- | -------------------------------- | -------------------------------------- |
| `host` (default), `default_cpu`, `default_io` | host                             | the CPU, through the normal block path |
| `host:sycl`                                   | host-resident, device-accessible | the SYCL CPU device                    |
| `gpu:sycl`, `gpu:sycl:1`                      | device (USM)                     | the first / second SYCL GPU            |

An unrecognised kind falls back to plain host, so a typo can never silently promote a block onto a device. A domain
that no registered backend serves warns once and runs on the CPU.

Call `gr::device::registerSyclRuntime()` once before building the graph. It enumerates the SYCL devices and publishes
a domain per device kind that exists — so a machine with no GPU publishes `host:sycl` but not `gpu:sycl`. A domain
nobody publishes warns once and runs on the CPU.

## Style 1 — `processOne`: let the framework write the kernel

If your block already has a `const noexcept processOne`, it is a kernel. The framework supplies the parallelism; the
block _is_ the functor.

```cpp
struct Gain : gr::Block<Gain> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    gr::Annotated<float, "gain"> gain = 1.f;
    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};
```

Set `compute_domain` to `gpu:sycl` and it runs on the GPU, unchanged: you supply the body, the framework writes the
parallel launch around it.

`const` and `noexcept` are the contract, not decoration. They are what makes the block safe to copy into device memory
and read from many work items at once.

## Style 1c — a `const processBulk` the framework runs as a kernel body

Between the two: a `processBulk` that takes **views** rather than port spans, and is `const`.

```cpp
[[nodiscard]] gr::work::Status processBulk(gr::InputViewLike auto& in, gr::OutputViewLike auto& out) const noexcept {
    for (std::size_t i = 0UZ; i < in.size(); ++i) {
        out[i] = in[i] * gain;
    }
    return gr::work::Status::OK;
}
```

The framework moves the block to the device and calls this on the device copy, so the same rules as style 1 apply:
`const`, `DeviceRelocatable`, no tags, no mutable state. What it buys over `processOne` is the whole span at once —
useful when a sample's output depends on its neighbours.

`InputViewLike` / `OutputViewLike` carry no `consume`/`publish`/tag members at all, by design: a kernel body cannot
reach the host ring, so the framework does that accounting. Constrain the signature to the *view* concepts, not to
`InputSpanLike`/`OutputSpanLike` — a block written against the span concepts does not match this tier and quietly
takes the CPU path instead.

Single input and single output only.

## Style 1s — a `processBulk` over port spans, run on the device as one work item

When a block genuinely needs `consume`/`publish` — a decimator, an interpolator, an IIR, a state machine — constrain
it to the *span* concepts instead:

```cpp
[[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& in, gr::OutputSpanLike auto& out) {
    const std::size_t nConsumed = std::min(in.size(), out.size() / 2UZ);
    // ... write 2 * nConsumed outputs ...
    std::ignore = in.consume(nConsumed);
    out.publish(2UZ * nConsumed);
    return gr::work::Status::OK;
}
```

This is not parallelised — it runs as a single work item, and the point is residency: a sequential body stays on the
device between its neighbours instead of round-tripping through the host. The two spans are sized independently, so
the body may read one number of samples and write another; the counts it records are replayed onto the real spans
once the kernel finishes. A body that asks for nothing consumes and publishes what was available, exactly as on the
host. Input tags are readable here, and a tag may be published if its payload is built in place (see above).

## Style 2 — `processBulk_sycl`: the expert extension point

This is _not_ a kernel body. It runs on the host thread and hands you the queue and the spans, so you can submit your
own kernels, chain them with events, use local memory, or call a vendor library. `gr::blocks::fft::FFT`'s multi-stage Stockham chain
lives here.

```cpp
[[nodiscard]] gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& in, OutputSpanLike auto& out) {
    // full control: submit kernels, chain events, publish tags after queue.wait()
}
```

Because it runs on the host, it is the only style that may publish tags or touch block state.

## Settings on the device

A member of a device-eligible block must be one of:

- a **fundamental or trivially copyable** type — carried verbatim into the device copy;
- a **pmr container** of trivially copyable elements (`std::pmr::vector<float>`, `gr::Tensor<float>`) — the framework
  re-seats its storage onto the device's memory during `init()`, so the kernel indexes the same buffer the host owns;
- a **port** — a kernel never touches ports; their data arrives as `processOne` arguments.

Write `std::pmr::vector<float> taps;`, not `std::vector<float> taps;`. A plain `std::vector` keeps its data on the host
heap, and the device copy would carry a host pointer. The block is then not device-eligible and the dispatch says which
member is to blame:

```
device dispatch: member 'taps' cannot be relocated to device memory
                 (use a fundamental, trivially copyable, or pmr type); running processOne on the CPU
```

Strings are **host-only**. `std::pmr::string` stores short values inside the object, so a copy in device memory would
point back at the host original. A block with a string setting can still run on the CPU; it just is not device-eligible.

Settings assignment keeps the device seat: once `init()` has re-seated `taps`, a later `settings().set(...)` reallocates
through the same device resource. You do not have to think about it.

## What a kernel may not do

`processOne` runs on the device, so it may only read its own settings.

- **No tags.** `publishTag` and `mergedInputTag()` are non-const, and a kernel body is `const`, so the compiler already
  stops you. Emit tags from `processBulk` or `processBulk_sycl`, which run on the host.
- **No mutable state.** A `mutable` member written by a `const processOne` is written into the _device copy_ and thrown
  away. No type trait can see this before C++26, so the framework probes the body once per settings change, in every
  build, and fails the dispatch with a clear message naming what it saw. Do not use `mutable` in a device-eligible
  block. Settings held in pmr storage are probed too — they share their bytes with the device copy, so writes that
  reach them are not lost and are not reported. The probe declines only for a setting seated on a device resource,
  whose bytes the host may not read.
- **No host pointers.** A raw pointer member is a host address and means nothing on the device; such a block is not
  eligible.

## What stays on the device, and what does not

Residency follows the **edge**: an edge whose source and destination declare the same device `compute_domain` never
crosses to the host, so its buffer is device memory. That covers a linear chain (`FFT -> multiply -> iFFT`) and
equally a **fan-out** — one output port feeding several device consumers shares one device ring with several
readers, with no host round trip. Only an edge with the host on one side is staged through host memory.

**The one limitation:** a `SubGraph`'s exported boundary port does not carry its member's `compute_domain` outward,
so the parent edge is an ordinary host edge and two groups cannot be chained device-to-device. Within a group, and
in a flat graph, device-to-device is the normal case.

## Building

The SYCL backend is compiled in when AdaptiveCpp is the compiler. Without it the seam compiles to nothing and
CPU-only graphs are bit-for-bit unchanged.

`_GLIBCXX_DEBUG` must not be defined when a device backend is compiled in: it changes container layout, which would
silently invalidate the block's device copy. The build stops with an explanatory `#error`.
