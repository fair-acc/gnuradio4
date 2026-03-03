# Block Connection API - Runtime Graph Connectivity

## Overview

The connection API supports both compile-time type-safe connections (when
topologies are known at compile-time) and runtime dynamic connections (when
topologies are defined and can change during run-time).

**Key characteristics:**

- **Type-safe when possible** — compile-time validation of port names and types
  for compile-time-known blocks
- **Runtime flexibility** — string-based connections for dynamic block
  instantiation
- **Error reporting** — uses `std::expected` for recoverable errors instead of
  exceptions
- **Buffer management** — configurable buffer sizes with sensible defaults

## Connection Methods

## Compile-Time Connection using port names

Use template parameters to specify port names. The compiler validates port
existence and type compatibility.

### Syntax

```cpp
graph.connect<"sourcePortName", "destinationPortName">(sourceBlock, destinationBlock);
graph.connect<"sourcePortName", "destinationPortName">(sourceBlock, destinationBlock, parameters);
```

### Example: Basic Connection

```cpp
auto& source = graph.emplaceBlock<gr::testing::NullSource<float>>();
auto& sink   = graph.emplaceBlock<gr::testing::NullSink<float>>();

// connect source.out to sink.in
auto result = graph.connect<"out", "in">(source, sink);
```

### Example: With Edge Parameters

```cpp
auto& source = graph.emplaceBlock<gr::testing::NullSource<float>>();
auto& sink   = graph.emplaceBlock<gr::testing::NullSink<float>>();

// specify buffer size and edge name
auto result = graph.connect<"out", "in">(source, sink, {
    .minBufferSize = 8192,
    .weight = 10,
    .name = "edge name"
});
```

### Port Name Syntax

For **simple ports** (single port):

```cpp
graph.connect<"out", "in">(source, sink);
```

For **port arrays** (compile-time size, e.g., `std::array<PortOut<T>, N>`):

```cpp
// use "#index" suffix to select specific port
graph.connect<"out#0", "in#1">(source, sink);
```

### Advantages

- **Compile-time validation** — typos in port names cause compile errors
- **Type checking** — port type mismatches caught at compile time
- **Performance** — zero runtime overhead for port lookup

## Runtime Connection using port names

Required when block types and flowgraph topologies are not known at compile time
(plugins, GRC, dynamically loaded blocks) or when using dynamic collections of
ports (`std::vector<PortIn>`).

### Syntax

```cpp
graph.connect(sourceBlock, "sourcePortName", destinationBlock, "destinationPortName");
graph.connect(sourceBlock, "sourcePortName", destinationBlock, "destinationPortName", parameters);
```

### Port String Syntax

For **simple ports**:

```cpp
graph.connect(source, "out", sink, "in");
```

For **port collections**:

```cpp
// use "#index" suffix (index validated at runtime)
graph.connect(source, "out#1", sink, "in#0");
```

### Advantages

- **Runtime flexibility** — connect blocks from plugins, load topologies from
  GRC files

## Edge Parameters

Connection methods accept an optional `EdgeParameters` struct:

```cpp
struct EdgeParameters {
    std::size_t  minBufferSize = undefined_size;  // minimum buffer size (bytes or samples)
    std::int32_t weight        = 0;               // scheduling weight/priority
    std::string  name          = "unnamed edge";  // human-readable edge identifier
};
```

# Block Merging — Compile-Time Composition for High-Performance Chains

## Overview

The Merge API provides compile-time block composition primitives that fuse
multiple blocks into a single optimized unit. These eliminate dynamic runtime
buffer allocations and enable aggressive compiler optimizations including
inlining, loop fusion, and vectorization.

**Key characteristics:**

- **Compile-time fusion** — blocks are combined at compile time, eliminating
  runtime buffer overhead
- **Zero-copy data flow** — intermediate results pass directly between blocks
  via registers or stack
- **SIMD-ready** — merged blocks preserve and propagate SIMD capabilities from
  constituent blocks
- **Type-safe connections** — port connections are validated at compile time
- **Orders of magnitude faster** — see benchmarks for performance comparisons

## Merge Types

There are three merge primitives:

| Type                                                   | Purpose                   | Output Ports                 |
| ------------------------------------------------------ | ------------------------- | ---------------------------- |
| `Merge` / `MergeByIndex`                               | Linear chain fusion       | Forward block outputs        |
| `FeedbackMerge` / `FeedbackMergeByIndex`               | Feedback loops (IIR-like) | Forward block outputs        |
| `FeedbackMergeWithTap` / `FeedbackMergeWithTapByIndex` | Feedback with tap output  | Forward outputs + `splitOut` |

Each type has two variants:

- **Named variant** (`Merge`, `FeedbackMerge`, `FeedbackMergeWithTap`) — uses
  port names as template parameters
- **Indexed variant** (`MergeByIndex`, etc.) — uses port indices as template
  parameters

## `Merge` — Linear Block Fusion

Combines two blocks by connecting an output port of the first block to an input
port of the second block.

### Syntax

```cpp
// by port name
using CustomBlock = Merge<SourceBlock, "outputPortName", SinkBlock, "inputPortName">;

// by port index
using CustomBlock = MergeByIndex<SourceBlock, outputPortIndex, SinkBlock, inputPortIndex>;
```

### Example: Scale and Add Chain

```cpp
#include <gnuradio-4.0/Graph.hpp>

template<typename T, T factor>
struct Scale : gr::Block<Scale<T, factor>> {
    ...
};

template<typename T>
struct Add : gr::Block<Add<T>> {
    ...
};

// create a merged block: scale input by 2, then add with second input
using ScaleThenAdd = gr::Merge<Scale<float, 2>, "out", Add<float>, "in1">;
```

---

## `FeedbackMerge` — merging with a Feedback Loop

Creates a feedback loop where the forward block's output is fed through a
feedback block and returned to one of the forward block's inputs.

### Syntax

```cpp
// by port name
FeedbackMerge<ForwardBlock, "forwardOutputPort",
              FeedbackBlock, "feedbackOutputPort",
              "forwardFeedbackInputPort">

// by port index
FeedbackMergeByIndex<ForwardBlock, forwardOutputPortIndex,
                     FeedbackBlock, feedbackOutputPortIndex,
                     forwardFeedbackInputPortIndex>
```

### Signal Flow

```
          Forward
          adder       ┌─────────────────────> out of FeedbackMerge
          ┌────┐      │
─────in1─>┤    │      │      Feedback
          │    ├─out─>┤      scale
   ┌─in2─>┤    │      │      ┌────┐
   │      └────┘      └──in─>┤    ├─out───┐
   │                         └────┘       │
   └──────────────────<───────────────────┘
```

### Example: Delayed Feedback Sum (IIR)

```cpp
template<typename T>
struct Adder : gr::Block<Adder<T>> {
    ...
};

template<std::size_t N>
struct Delay : gr::Block<Delay<N>> {
    ...
};

using DelayedSum = gr::FeedbackMerge<
    Adder<float>, "out",      // forward: adder output
    Delay<2>, "out",          // feedback: delay output
    "in2"                     // feedback connects to adder.in2
>;
```

### Constraints

- **Feedback block must have exactly one input port**
- **Forward block must have at least two input ports** (at least one for
  external input, one for feedback)
- Feedback state is zero-initialized

### Port Signature

The merged block exposes:

- **Input ports**: all forward block inputs **except** the feedback input port
- **Output ports**: all forward block outputs (unchanged)

---

## `FeedbackMergeWithTap` — Feedback with Tap Output

Like `FeedbackMerge`, but also exposes the feedback signal as a separate output
port named `splitOut`. This allows monitoring or tapping the feedback path while
maintaining the feedback connection.

### Syntax

```cpp
// by port name
FeedbackMergeWithTap<ForwardBlock, "forwardOutputPort",
           FeedbackBlock, "feedbackOutputPort",
           "forwardFeedbackInputPort">

// by port index
FeedbackMergeWithTapByIndex<ForwardBlock, forwardOutputPortIndex,
                  FeedbackBlock, feedbackOutputPortIndex,
                  forwardFeedbackInputPortIndex>
```

### Signal Flow

```
          Forward
          adder       ┌──────────────────────> out of FeedbackMergeWithTap
          ┌────┐      │
─────in1─>┤    │      │      Feedback
          │    ├─out─>┤      scale       ┌--─> splitOut
   ┌─in2─>┤    │      │      ┌────┐      │
   │      └────┘      └──in─>┤    ├─out─>┤
   │                         └────┘      │
   └──────────────────<──────────────────┘
```

### Use Cases

- **Debugging feedback loops** — inspect feedback signal values
- **Monitoring filter state** — observe internal state evolution
- **Parallel processing** — feed both output and feedback to different
  downstream blocks

### Port Signature

The merged block exposes:

- **Input ports**: all forward block inputs **except** the feedback input port
- **Output ports**: all forward block outputs **plus** `splitOut` (feedback
  signal)

---

## Performance Considerations

### Optimization Benefits

Merged blocks enable:

- **Inlining** — function calls eliminated
- **Loop fusion** — single loop instead of multiple passes
- **Register allocation** — intermediate values stay in registers
- **SIMD vectorization** — compiler can vectorize the entire chain
- **Constant propagation** — compile-time constants folded

### Trade-offs

**Advantages:**

- Extreme performance gains
- Zero runtime buffer allocation
- Compile-time type safety

**Disadvantages:**

- Increased compilation time
- Larger binary size (template instantiation)
- No runtime reconfiguration
- Limited introspection and debugging
