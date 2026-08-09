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

A third kind sits between the two, and it is the one that surprises people: **state the kernel owns whose meaning
depends on a setting**. A filter's delay line is the example — it is device-private, so it is `mutable` and
unreflected, but it only means anything for the coefficients it was accumulated with. Change those at run time and
the mirror keeps the state while the reflected coefficients are replaced under it, so the next dispatch runs old
state through new settings. The host path does not show this, because there the redesign usually constructs fresh
state; only the device path keeps it.

The fix is to make the dependency explicit rather than hope. Carry a counter the host bumps whenever it rewrites
those settings, reflect it so the mirror receives it, and have the body compare it against a `mutable` copy:

```cpp
gr::Size_t         _design_epoch = 0U;   // reflected, bumped by settingsChanged; the leading underscore keeps it
                                         // off the settings surface
mutable gr::Size_t _stateEpoch   = 0U;   // device-private: the epoch the state in hand belongs to

// at the top of the body
if (_stateEpoch != _design_epoch) {
    _state.fill({});
    _stateEpoch = _design_epoch;
}
```

This behaves identically on the host, so there is one code path and one place to get it right. `gr::filter::BasicFilter`
does exactly this.

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

## Style 2 — `processBulkDevice`: the expert extension point

This is _not_ a kernel body. It runs on the host thread and hands you the queue and the spans, so you can submit your
own kernels, chain them with events, use local memory, or call a vendor library. `gr::blocks::fft::FFT`'s multi-stage Stockham chain
lives here.

```cpp
[[nodiscard]] gr::work::Status processBulkDevice(gr::device::DeviceContext& ctx, InputSpanLike auto& in, OutputSpanLike auto& out) {
    // full control: submit kernels, chain events, publish tags after queue.wait()
}
```

Because it runs on the host, it is the only style that may publish tags or touch block state.

### A hatch on a windowed block gets the whole span

If your block declares a window with `Resampling<>`/`Stride<>` _and_ defines `processBulkDevice`, the hatch wins:
it is offered before the framework's window tier, and it receives the entire batched span rather than one window.
That is deliberate — submitting one kernel for every frame in the span is the cost the batching exists to avoid —
but it means the hatch has to walk the frames itself.

Do not work the frame arithmetic out by hand. Ask for it:

```cpp
#include <gnuradio-4.0/WindowGeometry.hpp>

const gr::WindowGeometry frames = gr::windowGeometry(*this, in.size(), out.size());
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

**To tell the rest of the graph something, publish a tag rather than change your own settings.** A tag is read
downstream and already works from a kernel; a setting written on the device is written into the device copy, which no
path brings back. The few blocks that really must change their own state keep it in `mutable` members of a span-form
body — a single work item, so there is an ordering to rely on — never in a parallelised one.

## What a kernel may not do

These are the rules for a body the framework parallelises — `processOne` (style 1) and the view-form `processBulk`
(style 1c). Every work item shares one device copy of the block, which is what makes the first two rules bite. The
span form (style 1s) runs as a **single** work item and is exempt from both; `processBulkDevice` runs on the host
thread and is exempt from all three.

- **No tags.** `publishTag` and `mergedInputTag()` are non-const, and a parallelised body is `const`, so the compiler
  already stops you. Emit tags from the span form or from `processBulkDevice`.
- **No mutable state.** This is a contract, not a checked condition: a `mutable` member written by a parallelised
  `const` body is written into the _device copy_ and thrown away, silently — and with N work items sharing one copy
  there is no ordering to salvage. No type trait can see it before C++26 and the framework does not go looking. A
  setting held in pmr storage is the one safe case: it shares its bytes with the device copy, so a write through it
  is not lost. (The span form's single work item is exactly why `mutable` state is the documented way to keep
  per-block state there — see `OnePole` above.)
- **No host pointers.** A raw pointer member is a host address and means nothing on the device; such a block is not
  eligible. This one applies to every style.

### Tags a kernel emits, and tags it may not change

A span-form body may **publish** a tag from a kernel — build the payload as a `gr::pmt::ValueMapView` in
pre-reserved storage and the host replays it onto the real output span. It may **read** its input tags, which the
framework stages into device-reachable slots before the launch.

It may **not modify a forwarded tag in place**: input tags reach a kernel as `std::span<const gr::Tag>`. A body that
must change a tag's values republishes it with the new ones and consumes the original. This is the one case where a
device body differs from the host body it was written as, and it is deliberate — mutable staged input tags would
have to be replayed back over the host ring, which buys one pattern a second write-back path.

### The one hazard the framework cannot see for you

A `processBulk(InputSpanLike auto&, OutputSpanLike auto&)` body runs as a kernel, and an **owning** tag payload built
inside it (`publishTag(property_map{…}, i)`) is not merely wrong there — the SSCP JIT cannot build that
kernel at all, and AdaptiveCpp then marks the whole device context poisoned for the rest of the process, taking every
later block with it. The failure is a property of the code in the body, not of whether that line ever runs.

The dispatcher does what it can: at a host boundary it runs the body once on a throwaway bit-copy of the block and
refuses the dispatch if a tag was attempted. **That check is not available on a device-resident edge**, because the
spans then point at memory the host thread must not read — and a device-resident edge is precisely what this style is
for. Measured on `qa_DeviceDspChain`: half the span-tier dispatches are on device-resident spans and take no check at
all. Nor can the shape be refused at compile time — `gr::OutputSpanLike` requires the owning `publishTag` overload, so
removing it would disqualify every span body rather than the tag-publishing ones.

**So treat it as a rule you keep, not a rule the framework enforces.** Build the payload as a `gr::pmt::ValueMapView`
in pre-reserved storage (`ValueMapView::formatAt` and `try_emplace` are device-callable) — see `qa_DeviceSpans.cpp`'s
`writeTriggerContract`. That form is copied into a tag slot and replayed by the host, allocates nothing, and never
reaches the failing path.

## What stays on the device, and what does not

Residency follows the **edge**: an edge whose source and destination declare the same device `compute_domain` never
crosses to the host, so its buffer is device memory. That covers a linear chain (`FFT -> multiply -> iFFT`) and
equally a **fan-out** — one output port feeding several device consumers shares one device ring with several
readers, with no host round trip. Only an edge with the host on one side is staged through host memory.

A **group** is no exception: `makeSubGraph` gives the group block its members' domain, so an edge between two device
groups is a device edge like any other and the chain does not surface at the boundary between them.

## Three ways across the host/device boundary

They are not alternatives to choose between once; they answer different questions, and they compose.

**Residency by edge domain — the default, and what most graphs want.** Give each block a `compute_domain` and the
edges follow: an edge whose two ends name the same device domain never comes back to the host. Nothing is declared
about the boundary itself; it falls where the domains stop matching. Reach for anything else only when this cannot
express what you need.

**`HostToDevice` / `DeviceToHost` — when the boundary must be explicit.** Registered blocks
(`blocks/basic/.../TransferBlocks.hpp`) that move a stream across in one bulk copy. Use them when you want the
transfer to be a visible node in the graph -- because you are measuring it, because a tool downstream reasons about
it, or because the block on one side cannot carry a `compute_domain` of its own. `qa_HostToDevice` covers them.

**`makeDeviceSubGraph` — when a whole group crosses together.** Hoists one shared `compute_domain` onto a set of
blocks so the group is placed as a unit rather than block by block (`blocks/basic/.../DeviceSubGraph.hpp`,
`qa_SubGraphVertical`). The group block carries that domain outward, so two such groups chain device-to-device with
no host hop between them.

### What deferral assumes about the platform

A device chain returns from a dispatch once its kernel is _enqueued_, not once it has run. While that kernel runs,
the host still touches shared (managed) USM belonging to the same block — the tag ring it publishes into, and the
block's own mirror when a setting changes. Both are ordered where it matters: a settings apply waits for the kernel
first, and the samples themselves live in device-only memory the host never reads.

What is assumed is that concurrent access to _shared_ USM, from the host and the device at once, is allowed at all.
On Linux with a Pascal-or-later NVIDIA device it is (`concurrentManagedAccess`); on a platform without it, such an
access faults rather than racing. If you are bringing up a device where that does not hold, deferral is the first
thing to turn off — every span the framework defers on is device-only, so the rule to relax is `isDeviceOnly`, not
the dispatch itself.

## Where a device starts to pay

Measured with `core/benchmarks/bm_DeviceDispatch`, sweeping the chunk one dispatch is handed from the frame sizes
filters actually run at up to the canonical FFT sizes (`bm_FFT_backends.cpp`). 4 194 304 samples per row, best of 5,
six pinned cores. Three blocks bracket the shapes: a multiply is one flop per sample, a polynomial 128, and an IIR
biquad is strictly sequential.

**MSample/s against the chunk:**

| block, domain          |   16 |   64 |   256 |  1024 |  4096 | 16384 | 65536 |
| ---------------------- | ---: | ---: | ----: | ----: | ----: | ----: | ----: |
| multiply `host`        | 24.3 | 64.8 | 111.8 | 136.5 | 142.6 | 143.7 | 143.2 |
| multiply `host:sycl`   |  0.6 |  2.2 |   8.3 |  25.7 |  64.3 | 104.7 | 125.8 |
| multiply `gpu:sycl`    |  0.3 |  1.2 |   4.5 |  16.3 |  46.7 |  87.7 | 113.0 |
| polynomial `host`      | 14.8 | 24.4 |  29.4 |  31.0 |  31.3 |  31.4 |  31.4 |
| polynomial `host:sycl` |  0.6 |  2.1 |   7.9 |  25.5 |  59.8 |  95.6 | 114.2 |
| polynomial `gpu:sycl`  |  0.3 |  1.2 |   4.5 |  16.3 |  46.8 |  87.7 | 113.2 |
| IIR `host`             | 21.9 | 50.0 |  74.2 |  84.6 |  87.3 |  87.7 |  87.7 |
| IIR `host:sycl`        |  0.6 |  2.3 |   8.6 |  25.9 |  53.2 |  72.6 |  79.7 |
| IIR `gpu:sycl`         |  0.2 |  0.8 |   2.6 |   5.0 |   6.7 |   7.3 |   7.6 |

**Cross-over.** Only the arithmetic-bound block ever crosses: **both device domains overtake the host between 1024
and 4096 samples per dispatch** for 128 flops/sample. The multiply never crosses — the host stays ~13 % ahead of the
best device row even at 65536 — and the sequential IIR never crosses either, by a wide margin.

The host rows are flat above ~1024; the device rows climb all the way, because a dispatch costs the same whether it
carries 16 samples or 65536. So the cross-over is set by how much arithmetic each sample carries, not by anything
the framework does. Below ~256 samples per dispatch a device is an order of magnitude _slower_ whatever the block.

**`gpu:sycl` returns 113 MS/s for the 1-flop block and 113 MS/s for the 128-flop block.** Identical — and
`host:sycl`, which crosses no bus at all, returns the same. Neither the arithmetic nor the transfer is the ceiling
here; both rows are climbing towards the host-side cost of the graph itself. The next section takes that apart.

### What actually bounds these numbers

Not the PCIe bus, and not the arithmetic. The evidence is in the table above: **`host:sycl` has no bus to cross and
returns the same ~113 MS/s as `gpu:sycl`.** A transfer cannot be the ceiling when a domain with no transfer sits on
the same number. Nor can the kernel: the 1-flop and the 128-flop blocks both return 113.

What they share is the **host side of the graph**. The `host` multiply row — a nearly free operation between a
source and a sink — tops out at 143 MS/s, i.e. **7 ns per sample** spent getting each sample through a block
boundary and into a sink that inspects it. Every device row is climbing towards that same floor, and none can pass
it. So the device figures here measure how quickly a dispatch can _reach_ the framework's per-sample cost, not what
the device could do.

The second cost, visible only at small chunks, is **fixed per dispatch**: from the 16-sample column, ~53 µs for
`gpu:sycl` and ~27 µs for `host:sycl`. At a 64 Ki chunk that is ~11 % of the call; at 256 samples it is everything.
It is dominated by the per-call shared-USM allocation and free for sample staging (a `sycl::free` is an implicit
device synchronisation on CUDA) plus the boundary waits — not by the bus, which at 8 B/sample and ~25 GB/s would
account for ~21 µs of a 512 KiB transfer.

**So, in order of what is worth attacking:**

1. the framework's ~7 ns/sample at a block boundary — a core concern, not a device one;
2. the ~53 µs fixed cost of a boundary dispatch, most of it a per-call allocate/free pair that could be hoisted the
   way the control area already is;
3. the bus, which is nowhere near the limit at these sizes.

For the interior of a device chain none of this applies: those hops neither stage nor synchronise, and cost about
**3 µs per dispatch** — see below.

### One stage against eight — where the ordering inverts

Same sweep, same 128-flop block, but a chain of 8 device stages beside the single stage. MSample/s:

| domain / stages |   16 |  256 | 1024 | 4096 | 16384 | 65536 | retained at 65536 |
| --------------- | ---: | ---: | ---: | ---: | ----: | ----: | ----------------: |
| `host` / 1      | 15.0 | 29.7 | 31.3 | 31.5 |  31.7 |  31.7 |                 — |
| `host` / 8      |  3.1 |  4.7 |  4.8 |  4.8 |   4.8 |   4.8 |          **15 %** |
| `host:sycl` / 1 |  0.6 |  8.4 | 25.6 | 59.1 |  97.0 | 116.9 |                 — |
| `host:sycl` / 8 |  0.1 |  1.0 |  4.0 | 11.5 |  27.2 |  43.6 |          **37 %** |
| `gpu:sycl` / 1  |  0.3 |  4.5 | 16.1 | 46.2 |  89.3 | 116.6 |                 — |
| `gpu:sycl` / 8  |  0.1 |  1.5 |  5.9 | 19.9 |  52.4 |  94.6 |          **81 %** |

Eight times the work for 1.23× the time on `gpu:sycl`. The host keeps 15 % of its single-stage rate, `host:sycl`
37 %, the GPU **81 %** — because only the GPU chain's interior hops are device-resident and therefore deferred.

**And the ordering inverts.** For one stage, `host:sycl` (116.9) edges out `gpu:sycl` (116.6) and the plain host is
far behind on this block. For eight stages, `gpu:sycl` (94.6) is more than twice `host:sycl` (43.6) and twenty times
the host. A single-block comparison is therefore the wrong way to decide whether a GPU is worth it: the property
being bought is residency across a chain, and it does not show up until there is a chain.

Below ~1024 samples per dispatch none of this helps — the fixed per-dispatch cost dominates and the host wins
outright at every chain length.

### What a chain costs, once it is resident

Same benchmark, `runChainOfLength`: N device stages in a row, so N−2 of them are interior — device memory on both
sides — and their kernels are enqueued without ever being awaited. 1 048 576 samples, 64 KiB chunk.

| domain      |     N=1 |     N=2 |      N=4 |      N=8 | per added stage |
| ----------- | ------: | ------: | -------: | -------: | --------------: |
| `host`      | 33.9 ms | 60.1 ms | 112.8 ms | 218.0 ms |         26.3 ms |
| `host:sycl` | 14.6 ms | 14.9 ms |  20.4 ms |  30.8 ms |          2.3 ms |
| `gpu:sycl`  | 14.1 ms | 14.1 ms |  13.0 ms |  14.4 ms | flat, see below |

**A chain of eight device stages costs what a chain of one costs** on `gpu:sycl` — 14.1 → 14.4 ms, inside the
run-to-run spread of 1–2 ms. With deferral disabled the same row rises to 17.0 ms. The host row grows linearly at
26 ms per stage, as it must.

Do not read a "ms per added stage" figure off the GPU row: across repeated runs it lands anywhere between 0.04 and
0.46 ms, because it is a small difference divided by seven. The claim that survives repetition is the shape — _flat
to N=8_ — and the exact version of it is the barrier count asserted in `qa_SubGraphVertical`, which is
independent of chain length and fails when deferral is forced off.

`host:sycl` pays 2.3 ms per added stage — but that is the polynomial _running_ on six CPU cores, not
synchronisation: 2.3 ms over 112 dispatches is ~144 µs of kernel, which is what that kernel costs on this CPU. The
GPU row avoids it by actually being faster at the arithmetic once the samples are already there, which is the whole
case for keeping a chain resident.

## When the device is not there

A compute domain is a preference by default. If the named device is not served -- no driver, wrong image, a
build without the backend -- the block runs on the host, says so once, and the graph proceeds. The same happens
when the block itself offers no device path for its types.

Spell the domain with a trailing `!` to make it a requirement instead:

```cpp
flow.emplaceBlock<MyBlock<float>>({{"compute_domain", "gpu:sycl!"}});
```

Now anything that would put the block back on the host stops the graph with a named error, whether the device is
absent or the block cannot be dispatched to it. The marker is part of the spelling, not a separate setting, and
it never reaches the registry: `gpu:sycl!` and `gpu:sycl` resolve to the same device.

Use it wherever running on the host would be wrong rather than merely slower -- a chain with a deadline, or a
test that means to prove a device ran. Leave it off for a graph that should stay portable across machines.

## Building

The SYCL backend is compiled in when AdaptiveCpp is the compiler. Without it the seam compiles to nothing, so
nothing here changes what a CPU-only graph computes.

A body that declares overlapping windows is handed exactly one window per call, on the host as on a device, so the
same body computes the same thing on either. See `Stride` in `USER_API_Block_Development.md`.

`_GLIBCXX_DEBUG` must not be defined when a device backend is compiled in: it changes container layout, which would
silently invalidate the block's device copy. The build stops with an explanatory `#error`.
