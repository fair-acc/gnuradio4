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
