# Block Connection & Merge API — User-API Reference

GR4 provides three ways to wire blocks together, each trading flexibility for performance:

1. **Runtime connection** — port names as strings, fully dynamic (plugins, GRC, Python, RPC)
2. **Compile-time connection** — port names as template parameters, validated at compile time (C++, embedded)
3. **Merge API** — fuse block types into a single zero-overhead block (maximum performance, C++, embedded)

The first two connection APIs return `std::expected<void, Error>` — no exceptions are thrown.

## TL;DR

```cpp
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/BlockMerging.hpp>  // only for Merge / FeedbackMerge / SplitMergeCombine
#include <gnuradio-4.0/Scheduler.hpp>

gr::Graph graph;
auto& src    = graph.emplaceBlock<gr::testing::NullSource<float>>();
auto& filter = graph.emplaceBlock<MyFilter<float>>();
auto& sink   = graph.emplaceBlock<gr::testing::NullSink<float>>();

// 1) runtime connect — port names resolved at runtime (plugins, GRC, Python)
// 2) compile-time connect — port names checked at compile time (C++, embedded)
// chain connections monadically — short-circuits on first error
auto result = graph.connect(src, "out", filter, "in")                        // runtime
    .and_then([&] { return graph.connect<"out", "in">(filter, sink); });     // compile-time
if (!result) { std::println(stderr, "{}", result.error().message); }

// 3) compile-time merge — fuse block types into a single zero-overhead block
using Chain = gr::Merge<ScaleA<float>, "out", ScaleB<float>, "in">;
auto& merged = graph.emplaceBlock<Chain>();
graph.connect<"out", "in">(src, merged)
    .and_then([&] { return graph.connect<"out", "in">(merged, sink); });

// run the graph
gr::scheduler::Simple sched{std::move(graph)};
sched.runAndWait();
```

---

## 1. Runtime connection API

Use when block types are only known at runtime — dynamically loaded plugins, GRC flowgraph
files, Python bindings, or RPC-driven topology changes. Port names are resolved at runtime;
type mismatches produce runtime errors.

```cpp
graph.connect(source, "out", sink, "in")                                          // block references
    .and_then([&] { return graph.connect(source, "out#0", sink, "in#2"); })       // port collections
    .and_then([&] { return graph.connect(source, "out", sink, "in",
                                         {.minBufferSize = 4096UZ}); });          // with edge parameters

// with shared_ptr<BlockModel> (from plugin registry)
graph.connect(srcModel, PortDefinition("out"), sinkModel, PortDefinition("in"));
```

---

## 2. Compile-time connection API

Preferred for C++ and embedded use. Template string parameters let the compiler validate
port existence and type compatibility — typos and mismatches become compile errors.

```cpp
graph.connect<"out", "in">(source, filter)
    .and_then([&] { return graph.connect<"out", "in">(filter, sink, {
        .minBufferSize = 8192UZ, .weight = 10, .name = "main data path"
    }); })
    .and_then([&] { return graph.connect<"out#0", "in#1">(multiOut, multiIn); }); // port arrays
```

---

## 3. Compile-time merge API

The merge API fuses block _types_ into a single block type at compile time. The resulting
block has zero runtime buffer overhead between the merged stages — the compiler can inline,
vectorise, and register-allocate the entire chain.

Merge types live in `<gnuradio-4.0/BlockMerging.hpp>` (explicit include required):

```cpp
#include <gnuradio-4.0/BlockMerging.hpp>
```

### `Merge` — linear chain fusion

Connects an output port of the left block to an input port of the right block:

```cpp
// by port name
using Chain = gr::Merge<Scale<float, 2>, "out", Scale<float, 3>, "in">;

// by port index
using Chain = gr::MergeByIndex<Scale<float, 2>, 0, Scale<float, 3>, 0>;

// use like any other block
auto& chain = graph.emplaceBlock<Chain>();
```

The merged block exposes all ports from both sub-blocks **except** the connected
pair. Settings are forwarded via nested property_map keys `"leftBlock"` and `"rightBlock"`.

### `SplitMergeCombine` — fan-out topology

Splits the input to N parallel paths and sums the results:

```
in ──┬── Path0 ──┬── (+) ── out
     ├── Path1 ──┤
     └── Path2 ──┘
```

```cpp
// two parallel paths — implicit identity-copy splitter, sum combiner
using FanOut = gr::SplitMergeCombine<Scale<float, 2>, Scale<float, 3>>;

// three or more paths
using FanOut3 = gr::SplitMergeCombine<PathA, PathB, PathC>;

// with per-path output signs (negate path 1 before summation)
using Diff = gr::SplitMergeCombine<gr::OutputSigns<+1.0f, -1.0f>, ScaleA, ScaleB>;
```

Each path must be a 1-in/1-out block. The merged block has a single `in` and
single `out` port. Access sub-blocks via `block.path<I>()`.

### `FeedbackMerge` — closed-loop feedback

Creates a feedback loop where the forward block's output is fed through a
feedback block and returned to one of the forward block's inputs:

```
            Forward
            adder       ┌────────────────────> out
            ┌────┐      │
───── in1 ─>┤    │      │     Feedback
            │    ├─out─>┤     scale
     ┌─in2─>┤    │      │     ┌────┐
     │      └────┘      └─in─>┤    ├─out──┐
     │                        └────┘      │
     └────────────────<───────────────────┘
```

```cpp
// by port name — forward output, feedback output, feedback-input on forward block
using IIR = gr::FeedbackMerge<Adder<float>, "out", Scale<float, alpha>, "out", "in2">;

// by port index
using IIR = gr::FeedbackMergeByIndex<Adder<float>, 0, Scale<float, alpha>, 0, 1>;
```

The merged block exposes all forward-block inputs **except** the feedback input
port, plus all forward-block outputs.

### `FeedbackMergeWithTap` — feedback with tap output

Like `FeedbackMerge`, but also exposes the feedback signal as a `splitOut` port:

```cpp
using Tapped = gr::FeedbackMergeWithTap<Adder<float>, "out", Scale<float, alpha>, "out", "in2">;
// extra port: tapped.splitOut
```

### Composing merge types

Merge types are blocks — they can be composed with each other and connected
via the runtime graph API:

```cpp
// IIR low-pass decomposed: FeedbackMerge wrapping a SplitMergeCombine
using IIR = gr::FeedbackMerge<
    Adder<float>, "out",
    gr::SplitMergeCombine<gr::OutputSigns<+1.0f, -1.0f>, Scale<float, 1>, Scale<float, alpha>>,
    "out", "in2">;

auto& filter = graph.emplaceBlock<IIR>();
graph.connect<"out", "in">(source, filter)
    .and_then([&] { return graph.connect<"out", "in">(filter, sink); });
```

### Settings forwarding

Merged blocks accept nested settings via sub-block keys:

```cpp
// MergeByIndex: keys "leftBlock" and "rightBlock"
chain.settings().set({{"leftBlock", property_map{{"gain", 2.0f}}},
                      {"rightBlock", property_map{{"gain", 3.0f}}}});

// SplitMergeCombine: keys "path0", "path1", ...
fanout.settings().set({{"path0", property_map{{"gain", 2.0f}}}});

// FeedbackMerge: keys "forward" and "feedback"
iir.settings().set({{"feedback", property_map{{"gain", 0.95f}}}});
```

---

## Performance comparison

The three API levels trade flexibility for performance. The benchmark below
(GCC 15, `-O3`, single-threaded, `bm_MergeApi`) illustrates the differences:

```
┌─────────────────────────────benchmark:──────────────────────────────┬──────┬─#N──┬──mean──┬─median─┬─ops/s─┐
│ runtime   src->copy->sink                                           │ PASS │ 10  │   6 ms │   6 ms │  162M │
│ runtime   src->mult->div->add->sink - float                         │ PASS │ 10  │  12 ms │  12 ms │ 87.0M │
│ runtime   src->(mult->div->add)^10->sink - float                    │ PASS │ 10  │  96 ms │  91 ms │ 10.4M │
│ runtime   IIR low-pass (feedback) - float                           │ PASS │     │ 101 ms │        │  994k │
├─────────────────────────────────────────────────────────────────────┼──────┼─────┼────────┼────────┼───────┤
│ merged    src->sink                                                 │ PASS │ 10  │   3 ms │   2 ms │  381M │
│ merged    src->mult->div->add->sink - float                         │ PASS │ 10  │   5 ms │   5 ms │  187M │
│ merged    src->(mult->div->add)^10->sink - float                    │ PASS │ 10  │   8 ms │   7 ms │  133M │
│ merged    IIR low-pass (FeedbackMerge) - float                      │ PASS │ 10  │   9 ms │   9 ms │  113M │
├─────────────────────────────────────────────────────────────────────┼──────┼─────┼────────┼────────┼───────┤
│ constexpr src->sink                                                 │ PASS │ 10  │  39 us │  39 us │ 25.4G │
│ constexpr src->mult->div->add->sink - float                         │ PASS │ 10  │  39 us │  39 us │ 25.4G │
│ constexpr src->(mult->div->add)^10->sink - float                    │ PASS │ 10  │ 351 us │ 349 us │  2.9G │
│ constexpr IIR low-pass (FeedbackMerge) - float                      │ PASS │ 10  │ 153 us │ 152 us │  656M │
└─────────────────────────────────────────────────────────────────────┴──────┴─────┴────────┴────────┴───────┘
```

Key takeaways:

- **Runtime → merged (scheduler)**: ~2-10x speedup by eliminating inter-block buffer copies
  while keeping a normal scheduler loop. For feedback topologies the gain is ~100x
  (994k → 113M ops/s) because `FeedbackMerge` avoids the 1-sample-per-scheduler-cycle
  bottleneck of runtime feedback loops.
- **Merged (scheduler) → merged (direct)**: another ~60-100x by bypassing the scheduler
  entirely via `loop_over_processOne()` — this matches hand-written scalar performance
  and enables full SIMD vectorisation.
- **IIR feedback**: the runtime graph processes one sample per scheduler cycle (the feedback
  path limits available data to 1 sample at a time). The merge API eliminates this by fusing
  the feedback loop into a single `processOne` call.

---

## `EdgeParameters`

All connection methods accept an optional `EdgeParameters` struct:

```cpp
struct EdgeParameters {
    std::size_t                minBufferSize = undefined_size;                    // minimum stream buffer size
    std::int32_t               weight        = 0;                                // scheduling weight/priority
    std::string                name          = "unnamed edge";                   // human-readable edge label
    std::pmr::memory_resource* dataResource  = std::pmr::get_default_resource(); // PMR allocator for stream buffer
    std::pmr::memory_resource* tagResource   = std::pmr::get_default_resource(); // PMR allocator for tag buffer
    ComputeDomain              domain        = ComputeDomain::host();            // compute domain (reserved)
};
```

### Custom buffer allocators (PMR)

The `dataResource` and `tagResource` fields control where the framework allocates
the circular stream and tag buffers for an edge. This is important for:

- **Embedded systems** — placing buffers in specific memory regions (SRAM, DRAM, TCM)
- **Memory optimisation** — using arena allocators to avoid per-edge heap allocations
- **Heterogeneous computing** — co-locating buffers with accelerator memory (CPU↔GPU
  shared memory, CUDA managed memory, SYCL USM) to avoid explicit data transfers

```cpp
// arena allocator — all buffers from a single 1 MB pool
std::pmr::monotonic_buffer_resource arena(1 << 20);
graph.connect<"out", "in">(source, sink, {
    .minBufferSize = 4096UZ,
    .dataResource  = &arena
});

// separate resources for stream data and tags
std::pmr::monotonic_buffer_resource dataPool(1 << 20);
std::pmr::monotonic_buffer_resource tagPool(1 << 16);
graph.connect<"out", "in">(source, sink, {
    .dataResource = &dataPool,
    .tagResource  = &tagPool
});

// GPU shared memory (hypothetical — resource provided by accelerator backend)
auto* gpuMemory = getGpuSharedResource();
graph.connect<"out", "in">(gpuSource, gpuSink, {
    .minBufferSize = 8192UZ,
    .dataResource  = gpuMemory
});
```

When left at the default (`std::pmr::get_default_resource()`), the framework uses
its standard double-mapped circular buffer allocator on platforms with POSIX mmap
support, falling back to the default PMR resource otherwise.

---

## Error handling

All connection APIs return `std::expected<void, gr::Error>`:

```cpp
struct Error {
    std::string          message;
    std::source_location sourceLocation;
};
```

### Common failure modes

| Error                   | Cause                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| Block not in graph      | Connecting blocks not added via `emplaceBlock`                     |
| Port name not found     | Typo or wrong port name (runtime only — compile-time catches this) |
| Port type mismatch      | Source and destination port value types differ                     |
| Port index out of range | `"out#5"` when only 3 ports exist                                  |

The compile-time API (`connect<"out", "in">`) catches port name typos and type
mismatches at compile time — prefer it for C++ code to eliminate an entire class of
errors before the programme runs.

### Error handling patterns

**Monadic chaining** — short-circuits on first error, no intermediate variables:

```cpp
graph.connect<"out", "in">(src, filter)
    .and_then([&] { return graph.connect<"out", "in">(filter, sink); })
    .and_then([&] { return graph.connect<"out", "in">(sink, next); });

// with inline error logging
graph.connect<"out", "in">(src, filter)
    .and_then([&] { return graph.connect<"out", "in">(filter, sink); })
    .or_else([](gr::Error e) -> std::expected<void, gr::Error> {
        std::println(stderr, "{}", e.message);
        return std::unexpected(e);
    });
```

**Inline `if` check** — explicit per-connection handling:

```cpp
if (auto r = graph.connect<"out", "in">(source, sink); !r) {
    std::println(stderr, "connection failed: {}", r.error().message);
}
```

**Propagate via `std::expected`** — for graph-builder functions:

```cpp
auto buildGraph() -> std::expected<gr::Graph, gr::Error> {
    gr::Graph graph;
    auto& src  = graph.emplaceBlock<gr::testing::NullSource<float>>();
    auto& sink = graph.emplaceBlock<gr::testing::NullSink<float>>();
    if (auto r = graph.connect<"out", "in">(src, sink); !r) { return std::unexpected(r.error()); }
    return graph;
}
```

**Helper lambda** — for tests/benchmarks where failure is fatal:

```cpp
auto ok = [](std::expected<void, gr::Error> r) {
    if (!r) { throw std::runtime_error(std::format("connect: {}", r.error().message)); }
};
ok(graph.connect<"out", "in">(source, sink));
```

**Collect all errors** — attempt all connections, report failures at end:

```cpp
std::vector<gr::Error> errors;
for (auto r : {graph.connect<"out","in">(a, b), graph.connect<"out","in">(b, c)}) {
    if (!r) { errors.push_back(r.error()); }
}
```

**Intentional discard** — for best-effort operations (e.g. cleanup):

```cpp
std::ignore = port.disconnect();
```

---

## Quick reference

| API                                        | Use when                        | Port validation | Overhead           |
| ------------------------------------------ | ------------------------------- | --------------- | ------------------ |
| `graph.connect(a, "out", b, "in")`         | Plugins, GRC, Python, RPC       | Runtime         | inter-block buffer |
| `graph.connect<"out","in">(a, b)`          | C++, embedded (types known)     | Compile-time    | inter-block buffer |
| `Merge<A,"out",B,"in">`                    | Fuse two blocks (zero overhead) | Compile-time    | none               |
| `SplitMergeCombine<P0, P1, ...>`           | Fan-out + sum (parallel paths)  | Compile-time    | none               |
| `FeedbackMerge<Fwd,"out",Fb,"out","fbIn">` | Closed-loop (IIR, PLL, AGC)     | Compile-time    | none               |
