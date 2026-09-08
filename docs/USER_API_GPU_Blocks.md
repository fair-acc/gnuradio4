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
reach the host ring, so the framework does that accounting. Constrain the signature to the _view_ concepts, not to
`InputSpanLike`/`OutputSpanLike` — a block written against the span concepts matches the span tier below instead,
which runs as one work item.

Single input and single output only.

## Style 1s — a `processBulk` over port spans, run on the device as one work item

When a block genuinely needs `consume`/`publish` — a decimator, an interpolator, an IIR, a state machine — constrain
it to the _span_ concepts instead:

```cpp
[[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& in, gr::OutputSpanLike auto& out) const {
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

### Where a stateful block keeps its state

The body is `const`, which is the whole contract: **reflected members are the host's** — they are the settings, the
host owns them, and a kernel cannot write them. State the **kernel** owns is declared `mutable` and left out of
`GR_MAKE_REFLECTABLE`:

```cpp
struct OnePole : gr::Block<OnePole> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    gr::Annotated<float, "alpha"> alpha = 0.2f;
    GR_MAKE_REFLECTABLE(OnePole, in, out, alpha);

    mutable float _previous = 0.f; // the kernel's own, between dispatches

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& in, gr::OutputSpanLike auto& out) const {
        const std::size_t n = std::min(in.size(), out.size());
        for (std::size_t i = 0UZ; i < n; ++i) {
            _previous = alpha * in[i] + (1.f - alpha) * _previous;
            out[i]    = _previous;
        }
        std::ignore = in.consume(n);
        out.publish(n);
        return gr::work::Status::OK;
    }
};
```

Such a member lives in the block's device mirror and is **not** reset between dispatches, so a delay line survives
the chunk boundary. It is re-seeded when the block enters `INITIALISED`, so a stopped and restarted graph does not
inherit the previous run's history. Nothing copies it back to the host: it is device-private, and the host does not
see it while the graph runs — which is the point, not a limitation.

Two rules follow from the bytes being copied verbatim:

- it must be **trivially copyable and own no host storage**. `HistoryBuffer<float, 16>` qualifies — the
  fixed-capacity form is backed by a `std::array`. `HistoryBuffer<float>` does not: the dynamic-extent form owns a
  `std::vector` whose pointer a kernel would follow back to host memory. C++23 cannot check this for members the
  reflection macro was not given, so it is on you.
- **do not split state across both kinds.** A pmr container's _contents_ are shared with the device, so writes
  through it are seen on both sides — but the bookkeeping inside the container object is not. A ring whose samples
  live in a shared container and whose read and write positions live in a reflected member will drift out of step
  with itself.

Array settings — filter taps, window coefficients — are the mirror image: put them in a `std::pmr::vector` and
reflect them. The framework re-seats their storage onto device memory, so the kernel reads them in place, and a
settings change that resizes them reaches the kernel on the next dispatch.

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

### A hatch on a windowed block gets the whole span

If your block declares a window with `Resampling<>`/`Stride<>` _and_ defines `processBulk_sycl`, the hatch wins:
it is offered before the framework's window tier, and it receives the entire batched span rather than one window.
That is deliberate — submitting one kernel for every frame in the span is the cost the batching exists to avoid —
but it means the hatch has to walk the frames itself.

Do not work the frame arithmetic out by hand. Ask for it:

```cpp
#include <gnuradio-4.0/device/WindowGeometry.hpp>

const gr::device::WindowGeometry frames = gr::device::windowGeometry(*this, in.size(), out.size());
if (frames.nWindows == 0UZ) {
    std::ignore = in.consume(0UZ);
    out.publish(0UZ);
    return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
}
// frame f reads in[f * frames.hop] for frames.inChunk samples, writes out[f * frames.outChunk]
```

`windowGeometry` answers from the window the block already declared, and it is the same function the framework's
window tier uses, so the two cannot drift apart. It is also the arithmetic this repository has already got wrong
once, in the FFT, which is why it is a function and not an example to copy.

## Settings on the device

A member of a device-eligible block must be one of:

- a **fundamental or trivially copyable** type — carried verbatim into the device copy;
- a **pmr container** of trivially copyable elements (`std::pmr::vector<float>`, `gr::Tensor<float>`) — the framework
  re-seats its storage onto the device's memory during `init()`, so the kernel indexes the same buffer the host owns;
- a **port** — a kernel never touches ports; their data arrives as `processOne` arguments.

Write `std::pmr::vector<float> taps;`, not `std::vector<float> taps;`. A plain `std::vector` keeps its data on the host
heap, and the device copy would carry a host pointer. The block is then not device-eligible, and rather than running it
somewhere it did not ask to run, the dispatch **refuses** and names the member to blame:

```
device dispatch refused: member 'taps' cannot be relocated to device memory
                         (use a fundamental, trivially copyable, or pmr type)
```

The refusal stops the graph. That is deliberate: the same body on the CPU computes exactly the numbers the kernel would
have, so substituting it silently would hide the misconfiguration behind a correct-looking answer.

Strings are **host-only**. `std::pmr::string` stores short values inside the object, so a copy in device memory would
point back at the host original. A block with a string setting runs perfectly well on `host`; it simply cannot be given
a device domain.

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

The SYCL backend is compiled in when AdaptiveCpp is the compiler. Without it the seam compiles to nothing, so
nothing here changes what a CPU-only graph computes.

One host-side change does ride along, independent of the backend: a block declaring `Stride<>` is now handed as
many windows as its span holds rather than one per call. It produces and consumes exactly what it did before, but
a body written to assume a single window per call now sees only the first -- loop over the span, or ask
`windowGeometry` for the frame count.

`_GLIBCXX_DEBUG` must not be defined when a device backend is compiled in: it changes container layout, which would
silently invalidate the block's device copy. The build stops with an explanatory `#error`.
