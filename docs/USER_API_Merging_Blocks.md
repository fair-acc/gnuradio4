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

| Type                                     | Purpose                   | Output Ports                 |
| ---------------------------------------- | ------------------------- | ---------------------------- |
| `Merge` / `MergeByIndex`                 | Linear chain fusion       | Forward block outputs        |
| `FeedbackMerge` / `FeedbackMergeByIndex` | Feedback loops (IIR-like) | Forward block outputs        |
| `SplitMerge` / `SplitMergeByIndex`       | Feedback with tap output  | Forward outputs + `splitOut` |

Each type has two variants:

- **Named variant** (`Merge`, `FeedbackMerge`, `SplitMerge`) — uses port names
  as template parameters
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
          block          *------------------> output
          +----+        /
------in1-|    |       /     Feedback
          |    |-out--*      block
    *-in2-|    |       \     +----+
   /      +----+        *-in-|    |-out--*
   |                         +----+       \
   \______________________________________/
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

## `SplitMerge` — Feedback with Tap Output

Like `FeedbackMerge`, but also exposes the feedback signal as a separate output
port named `splitOut`. This allows monitoring or tapping the feedback path while
maintaining the feedback connection.

### Syntax

```cpp
// by port name
SplitMerge<ForwardBlock, "forwardOutputPort",
           FeedbackBlock, "feedbackOutputPort",
           "forwardFeedbackInputPort">

// by port index
SplitMergeByIndex<ForwardBlock, forwardOutputPortIndex,
                  FeedbackBlock, feedbackOutputPortIndex,
                  forwardFeedbackInputPortIndex>
```

### Signal Flow

```
          Forward
          block          *------------------> output
          +----+        /
------in1-|    |       /     Feedback
          |    |-out--*      block
    *-in2-|    |       \     +----+
   /      +----+        *-in-|    |-out--*---------> splitOut
   |                         +----+       \
   \______________________________________/
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
