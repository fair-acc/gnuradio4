# GR4 Block Development — User-API Reference

GR4 blocks are `struct`s that inherit from `gr::Block<Derived>` via CRTP.
Settings are (optionally) declared as `Annotated<T, "name", ...>` fields and exposed
through compile-time reflection via `GR_MAKE_REFLECTABLE`. Blocks implement exactly one
processing function — `processOne` (per-sample), `processBulk` (span-based), or `work(..)` (advanced, from scratch).

Focus on **your algorithm** — the framework handles scheduling, buffer management, tag
propagation, SIMD vectorisation, and settings synchronisation. Lean on `std::ranges`,
`std::algorithms`, and value semantics; avoid manual loops and raw pointers.

## TL;DR — minimal block

```cpp
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

// marker macro — parsed at build time by `parse_registrations`
// syntax: GR_REGISTER_BLOCK(fully::qualified::Name, [T], [type1, type2, ...])
//   [T]              — template parameter placeholder expanded over the type list
//   [float, double]  — types to instantiate the block for
GR_REGISTER_BLOCK(example::Gain, [T], [float, double])

namespace example {

template<typename T>
struct Gain : gr::Block<Gain<T>> {
    using Description = Doc<R""(Multiplies input samples by a constant gain factor.)"">;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    gr::Annotated<float, "gain", gr::Doc<"linear gain">, gr::Unit<"dB">> gain = 1.0f;

    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    // t_or_simd<T> accepts both scalar T and SIMD vector types — the framework
    // automatically vectorises processOne calls when the method is const & noexcept
    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a * static_cast<T>(gain);
    }
};

} // namespace example
```

## Block anatomy — canonical member order

Members **must** appear in this order (blank lines separate groups):

```cpp
template<typename T>
struct MyBlock : gr::Block<MyBlock<T>> {
    // 1. type aliases & nested types
    using Description = Doc<R""(one-line purpose)"">;
    using value_type  = T;

    // 2. ports
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    // 3. settings & public fields
    gr::Annotated<float, "sample_rate", gr::Doc<"signal sample rate">, gr::Unit<"Hz">> sample_rate = 1.f;

    // 4. GR_MAKE_REFLECTABLE
    GR_MAKE_REFLECTABLE(MyBlock, in, out, sample_rate);

    // 5. private state (prefixed _)
    float _cachedCoeff = 0.f;

    // 6. lifecycle methods: start(), stop(), reset(), pause(), resume()
    // 7. processing: processOne() XOR processBulk()
    // 8. settingsChanged(...)
    // 9. helper methods
};
```

> **No custom constructors or destructors.** `Block<T>` provides a carefully sequenced
> constructor that initialises the CRTP chain, reflection metadata, settings pipeline,
> and port buffers. Custom constructors bypass this and break settings deserialisation,
> `property_map` initialisation (via `emplaceBlock<T>({...})`), and lifecycle contracts.
> Use default field initialisers for state, and `settingsChanged()` or `start()` for
> derived initialisation logic.

## Ports

### Static ports

```cpp
gr::PortIn<float>  in;       // single input
gr::PortOut<float> out;      // single output
```

### Dynamic ports

Resize in `settingsChanged` when the count setting changes:

```cpp
std::vector<gr::PortIn<T, gr::Async>>  inputs{};
std::vector<gr::PortOut<T, gr::Async>> outputs{};

gr::Annotated<gr::Size_t, "n_inputs", gr::Visible, gr::Limits<1U, 32U>> n_inputs = 2U;

void settingsChanged(const gr::property_map& /*old*/, const gr::property_map& newSettings) {
    if (newSettings.contains("n_inputs")) {
        inputs.resize(n_inputs);
    }
}
```

### Port attributes

| Attribute                       | Effect                                               |
| ------------------------------- | ---------------------------------------------------- |
| _(none)_                        | synchronous, required (default)                      |
| `gr::Async`                     | rate-decoupled; excluded from synchronous scheduling |
| `gr::Optional`                  | may be left unconnected                              |
| `gr::RequiredSamples<min, max>` | constrain min/max samples per work call              |

### Port span concepts (for `processBulk`)

| Concept              | Used for                                               |
| -------------------- | ------------------------------------------------------ |
| `gr::InputSpanLike`  | input spans — `consume(n)`, `tags()`, `rawTags`        |
| `gr::OutputSpanLike` | output spans — `publish(n)`, `publishTag(map, offset)` |

Reference: [`Selector.hpp`](../blocks/basic/include/gnuradio-4.0/basic/Selector.hpp) (dynamic ports), [`Soapy.hpp`](../blocks/soapy/include/gnuradio-4.0/soapy/Soapy.hpp) (conditional port count)

## Settings & `Annotated<T, ...>`

### Plain fields vs. `Annotated<T, ...>`

Any reflected field (listed in `GR_MAKE_REFLECTABLE`) is a setting — `Annotated<>` is
optional but recommended:

```cpp
// plain field — works, but no metadata for UIs, documentation generators, or unit validation
float sample_rate = 48000.f;

// annotated — carries description, unit, visibility, and value-range limits
gr::Annotated<float, "sample_rate", gr::Doc<"signal sample rate">, gr::Unit<"Hz">,
              gr::Visible, gr::Limits<1.f, 1e9f>> sample_rate = 48000.f;
```

Use `Annotated<>` when a setting should appear in auto-generated runtime documentation,
UI settings panels, or when you want automatic quantity/unit annotation and range validation.
Plain fields suffice for internal parameters that are not user-facing.

### Full syntax

```cpp
gr::Annotated<T, "snake_case_name", Doc<"...">, Unit<"[V]">, Visible, Limits<min, max>>
```

| Annotation         | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| `Doc<"...">`       | human-readable description                      |
| `Unit<"[V]">`      | physical SI unit (ISO 80000-1:2022)             |
| `Visible`          | show in UI settings panels                      |
| `Limits<min, max>` | value range validation (compile-time constants) |

### Shorthand alias

Blocks with many settings often define a local alias:

```cpp
template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
using A = gr::Annotated<U, description, Arguments...>;

A<gr::Size_t, "n_inputs", gr::Visible, gr::Limits<1U, 32U>> n_inputs = 0U;
```

### Supported types

Scalars (`float`, `double`, `int`, ...), `std::string`, enums, `std::vector<T>`,
`std::array<T, N>`, `gr::Tensor<T>`, `std::complex<T>`, `gr::UncertainValue<T>`.

### When to use `std::vector<T>` vs. `std::array<T, N>` vs. `gr::Tensor<T>`

| Type               | Use case                                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `std::vector<T>`   | variable-length 1D data — filter coefficients, lookup tables, any list whose size changes at runtime     |
| `std::array<T, N>` | fixed-length 1D data — compile-time-known small buffers (e.g. 3-element RGB, fixed-order filter)         |
| `gr::Tensor<T>`    | multi-dimensional data — matrices, spectral maps, anything with rank > 1 or mixed static/dynamic extents |

For simple 1D settings, prefer `std::vector<T>` (runtime-sized) or `std::array<T, N>`
(fixed-size). Reserve `Tensor<T>` for genuinely multi-dimensional data.

### `GR_MAKE_REFLECTABLE`

Lists the block type followed by **all** reflected members (ports and settings).
The macro enables compile-time reflection, serialisation, and the settings pipeline.

```cpp
GR_MAKE_REFLECTABLE(MyBlock, in, out, sample_rate, threshold);
```

### `property_map` wire format

```cpp
using gr::property_map = gr::pmt::Value::Map;  // std::pmr::map<std::pmr::string, pmt::Value>
```

Settings are read and written as `property_map` key–value pairs where keys match
the `snake_case` setting name and values are `pmt::Value`.

Reference: [`Soapy.hpp`](../blocks/soapy/include/gnuradio-4.0/soapy/Soapy.hpp) (rich annotations), [`Rotator.hpp`](../blocks/math/include/gnuradio-4.0/math/Rotator.hpp) (XOR constraints)

## Processing functions

Implement exactly one: `processOne`, `processBulk`, or — for full control —
override `work()` directly (see [Advanced: custom `work()` function](#advanced-custom-work-function)).

### `processOne` — per-sample (preferred for 1:1 transforms)

```cpp
// 1:1 transform
template<gr::meta::t_or_simd<T> V>
[[nodiscard]] constexpr V processOne(const V& a) const noexcept {
    return a * static_cast<T>(gain);
}

// source (no input args)
[[nodiscard]] constexpr T processOne() noexcept {
    return _nextSample++;
}

// sink (void return)
constexpr void processOne(const T& input) noexcept {
    _buffer.push_back(input);
}
```

### `processBulk` — span-based (resampling, variable-rate, tag access)

```cpp
[[nodiscard]] constexpr gr::work::Status processBulk(
        std::span<const T> input, std::span<T> output) noexcept {
    std::ranges::copy(input, output.begin());
    return gr::work::Status::OK;
}

// with InputSpanLike / OutputSpanLike for tag access and manual publish/consume
gr::work::Status processBulk(gr::InputSpanLike auto& inSpan,
                             gr::OutputSpanLike auto& outSpan) {
    const auto nSamples = std::min(inSpan.size(), outSpan.size());
    std::ranges::copy_n(inSpan.begin(), nSamples, outSpan.begin());
    outSpan.publish(nSamples); // optional, if ommitted then all samples are consumed/published
    return gr::work::Status::OK;
}

// source (output only)
gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
    std::ranges::fill(outSpan, T{});
    outSpan.publish(outSpan.size());
    return gr::work::Status::OK;
}

// multi-input (dynamic ports)
template<gr::InputSpanLike TInSpan>
gr::work::Status processBulk(std::span<TInSpan>& ins, gr::OutputSpanLike auto& out) {
    // ins[0], ins[1], ...
    return gr::work::Status::OK;
}
```

### SIMD-aware `processOne`

When `processOne` is `const` and `noexcept`, the framework automatically vectorises it
using `vir::simd` / `std::simd`. The concept `gr::meta::t_or_simd<T>` accepts both
scalar `T` and SIMD vector types — no special code is needed:

```cpp
template<gr::meta::t_or_simd<T> V>
[[nodiscard]] constexpr V processOne(const V& a) const noexcept {
    if constexpr (gr::meta::any_simd<V>) {
        // optional: SIMD-specific path (rarely needed)
    } else {
        // scalar path
    }
    return a * static_cast<T>(gain);  // works for both scalar and SIMD
}
```

The framework determines the optimal SIMD width at compile time, packs input samples
into SIMD registers, calls `processOne` with the widened type, and scatters results
back to the output buffer — all transparently. A non-`const` or non-`noexcept`
`processOne` falls back to scalar-only evaluation.

### `work::Status`

| Value                            | Meaning                               |
| -------------------------------- | ------------------------------------- |
| `OK` (0)                         | processed successfully                |
| `DONE` (-1)                      | block finished; flowgraph should stop |
| `INSUFFICIENT_INPUT_ITEMS` (-2)  | need more input samples               |
| `INSUFFICIENT_OUTPUT_ITEMS` (-3) | need a larger output buffer           |
| `ERROR` (-100)                   | error occurred                        |

Reference: [`NullSources.hpp`](../blocks/testing/include/gnuradio-4.0/testing/NullSources.hpp) (source), [`CommonBlocks.hpp`](../blocks/basic/include/gnuradio-4.0/basic/CommonBlocks.hpp) (1:1), [`time_domain_filter.hpp`](../blocks/filter/include/gnuradio-4.0/filter/time_domain_filter.hpp) (history)

## Resampling & decimation/interpolation

Resampling is a **first-class citizen** in GR4. Declare the ratio as a `Block` template
argument and the framework handles chunked scheduling, buffer sizing, and automatic
`sample_rate` tag adjustment.

### Declaring resampling

```cpp
// default: 1:1 fixed during compile-time (isConst = true)

// decimation 8:1 — for every 8 input samples, produce 1 output
template<typename T>
struct Decimator : gr::Block<Decimator<T>, gr::Resampling<8U, 1U, true>> { ... };

// interpolation 1:2 — for every 1 input sample, produce 2 outputs
template<typename T>
struct Upsampler : gr::Block<Upsampler<T>, gr::Resampling<1U, 2U, true>> { ... };

// runtime-configurable ratio (isConst = false)
template<typename T>
struct FlexResampler : gr::Block<FlexResampler<T>, gr::Resampling<10U, 1U>> { ... };
```

| Parameter                                | Default           | Purpose                                       |
| ---------------------------------------- | ----------------- | --------------------------------------------- |
| `Resampling<inChunk, outChunk, isConst>` | `<1U, 1U, false>` | decimation / interpolation ratio              |
| `Stride<N, isConst>`                     | `<0, false>`      | sample spacing between calls (0 = no overlap) |

### How it works

For `Resampling<N, M>`:

- the scheduler provides exactly `k * N` input samples and `k * M` output samples to `processBulk`
- settings `input_chunk_size` and `output_chunk_size` are auto-created on the block
- when `isConst = false` (default), these settings can be changed at runtime
- **automatic `sample_rate` tag adjustment**: when the framework forwards a `sample_rate`
  tag through a resampling block, it multiplies by `output_chunk_size / input_chunk_size`

### Stride (overlapping windows)

`Stride<N>` advances the input pointer by `N` samples between `processBulk` calls
instead of consuming the full input span. This enables overlapping-window processing
(e.g. FFT with 50 % overlap):

```cpp
template<typename T>
struct WindowedFFT : gr::Block<WindowedFFT<T>, gr::Stride<512U>> { ... };
```

Reference: [`ConverterBlocks.hpp`](../blocks/basic/include/gnuradio-4.0/basic/ConverterBlocks.hpp) (`Resampling<1,2>` and `<2,1>`), [`time_domain_filter.hpp`](../blocks/filter/include/gnuradio-4.0/filter/time_domain_filter.hpp) (filter with history)

## Lifecycle methods

### State machine

The full state diagram including transition functions
(see also [`LifeCycle.hpp`](../core/include/gnuradio-4.0/LifeCycle.hpp)):

```
                Block<T>()              can be reached from
                   │                   anywhere and anytime.
              ┌─────┴────┐                   ┌────┴────┐
 ┌────────────┤   IDLE   │                   │  ERROR  │
 │            └────┬─────┘                   └────┬────┘
 │                 │ init()                       │ reset()
 │                 v                              │
 │         ┌───────┴───────┐                      │
 ├<────────┤  INITIALISED  ├<─────────────────────┤
 │         └───────┬───────┘                      │
 │                 │ start()                      │
 │                 v                              │
 │   stop() ┌──────┴──────┐                      │  ╓
 │ ┌────────┤   RUNNING   ├<──────────┐          │  ║
 │ │        └─────┬───────┘           │          │  ║  isActive(state) → true
 │ │              │ pause()           │          │  ║
 │ │              v                   │ resume() │  ║
 │ │    ┌─────────┴─────────┐   ┌─────┴─────┐   │  ║
 │ │    │  REQUESTED_PAUSE  ├──>┤  PAUSED   │   │  ║
 │ │    └──────────┬────────┘   └─────┬─────┘   │  ╙
 │ │               │ stop()           │ stop()  │
 │ │               v                  │         │  ╓
 │ │     ┌─────────┴────────┐         │         │  ║
 │ └────>┤  REQUESTED_STOP  ├<────────┘         │  ║
 │       └────────┬─────────┘                   │  ║  isShuttingDown(state) → true
 │                │                             │  ║
 │                v                             │  ║
 │          ┌─────┴─────┐ reset()               │  ║
 └─────────>│  STOPPED  ├──────────────────────>┘  ║
            └─────┬─────┘                          ╙
                  │
                  v
              ~Block<T>()
```

### Optional lifecycle overrides

All lifecycle methods are **optional** — implement only what your block needs:

```cpp
void start()  { /* acquire resources */ }
void stop()   { /* release resources */ }
void reset()  { /* clear internal state, return to INITIALISED */ }
void pause()  { /* suspend activity */ }
void resume() { /* resume after pause */ }
```

### Runtime control

```cpp
this->requestStop();                          // signal end from within processOne/processBulk
gr::lifecycle::isActive(this->state());       // true for RUNNING, REQUESTED_PAUSE, PAUSED
gr::lifecycle::isShuttingDown(this->state()); // true for REQUESTED_STOP, STOPPED
```

Reference: [`Soapy.hpp`](../blocks/soapy/include/gnuradio-4.0/soapy/Soapy.hpp) (start/stop with hardware), [`NullSources.hpp`](../blocks/testing/include/gnuradio-4.0/testing/NullSources.hpp) (`requestStop`)

## Settings change callback

Called automatically when settings are updated, **before** the next processing call.

### 2-argument variant

```cpp
void settingsChanged(const gr::property_map& oldSettings,
                     const gr::property_map& newSettings) {
    if (newSettings.contains("sample_rate") || newSettings.contains("frequency")) {
        _coeff = computeCoefficient(sample_rate, frequency);
    }
}
```

### 3-argument variant (modify downstream tags)

```cpp
void settingsChanged(const gr::property_map& oldSettings,
                     const gr::property_map& newSettings,
                     gr::property_map& forwardSettings) {
    if (newSettings.contains("sample_rate")) {
        forwardSettings["sample_rate"] = sample_rate.value;
    }
}
```

### Common patterns

**Dynamic port resize:**

```cpp
if (newSettings.contains("n_inputs"))  { inputs.resize(n_inputs); }
if (newSettings.contains("n_outputs")) { outputs.resize(n_outputs); }
```

**XOR constraint (mutually exclusive settings):**

```cpp
if (newSettings.contains("frequency_shift") && !newSettings.contains("phase_increment")) {
    phase_increment = 2.f * std::numbers::pi_v<float> * frequency_shift / sample_rate;
} else if (!newSettings.contains("frequency_shift") && newSettings.contains("phase_increment")) {
    frequency_shift = phase_increment / (2.f * std::numbers::pi_v<float>) * sample_rate;
}
```

Reference: [`Rotator.hpp`](../blocks/math/include/gnuradio-4.0/math/Rotator.hpp) (XOR), [`Selector.hpp`](../blocks/basic/include/gnuradio-4.0/basic/Selector.hpp) (port resize + validation)

## Tags & streaming metadata

A tag is a `{index, property_map}` pair attached to a specific sample position.

### Tag access in `processOne`

```cpp
[[nodiscard]] constexpr T processOne(const T& input) noexcept {
    if (this->inputTagsPresent()) {
        const gr::Tag& tag = this->mergedInputTag();
        // read tag.map entries
    }
    this->publishTag({{"trigger_name", "my_event"}}, 0UZ);  // publish at current sample
    return input;
}
```

### Tag access in `processBulk`

```cpp
gr::work::Status processBulk(gr::InputSpanLike auto& inSpan,
                             gr::OutputSpanLike auto& outSpan) {
    // iterate over tags within current chunk
    for (const auto& [normalisedIndex, tagMap] : inSpan.tags()) {
        outSpan.publishTag(tagMap.get(), static_cast<std::size_t>(normalisedIndex));
    }

    // or access raw tags directly
    for (const auto& tag : inSpan.rawTags) { /* tag.index, tag.map */ }

    std::ranges::copy(inSpan, outSpan.begin());
    outSpan.publish(inSpan.size());
    return gr::work::Status::OK;
}
```

### Tag forwarding policy

By default, tags from all input ports are merged and forwarded to all output ports automatically.
To handle tags manually, disable default forwarding:

```cpp
template<typename T>
struct MyBlock : gr::Block<MyBlock<T>, gr::NoDefaultTagForwarding> { ... };
```

### End-of-stream

```cpp
this->publishEoS();  // convenience: publishes {gr::tag::END_OF_STREAM, true}
```

### Standard tag keys

| Key                             | Type          | Description                        |
| ------------------------------- | ------------- | ---------------------------------- |
| `"sample_rate"`                 | `float`       | signal sample rate [Hz]            |
| `"signal_name"`                 | `std::string` | signal name                        |
| `"signal_quantity"`             | `std::string` | physical quantity (e.g. "voltage") |
| `"signal_unit"`                 | `std::string` | SI unit (e.g. "[V]")               |
| `"signal_min"` / `"signal_max"` | `float`       | expected value range               |
| `"trigger_name"`                | `std::string` | trigger identifier                 |
| `"trigger_time"`                | `uint64_t`    | UTC timestamp [ns]                 |
| `"trigger_offset"`              | `float`       | sample delay w.r.t. trigger [s]    |
| `"context"`                     | `std::string` | multiplexing context key           |
| `"end_of_stream"`               | `bool`        | end-of-stream marker               |
| `"reset_default"`               | `bool`        | reset block to stored defaults     |
| `"store_default"`               | `bool`        | store current settings as defaults |

Reference: [`TagMonitors.hpp`](../blocks/testing/include/gnuradio-4.0/testing/TagMonitors.hpp) (TagSource, TagMonitor), [`ClockSource.hpp`](../blocks/testing/include/gnuradio-4.0/testing/ClockSource.hpp)

## Graph & connections

```cpp
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

gr::Graph graph;

// emplace blocks with optional initial settings
auto& src  = graph.emplaceBlock<example::Source<float>>({{"n_samples_max", gr::Size_t(1024)}});
auto& gain = graph.emplaceBlock<example::Gain<float>>({{"gain", 2.0f}});
auto& sink = graph.emplaceBlock<example::Sink<float>>();

// connect by port name
graph.connect<"out">(src).to<"in">(gain);
graph.connect<"out">(gain).to<"in">(sink);

// run
gr::scheduler::Simple sched;
sched.exchange(std::move(graph));
sched.runAndWait();
```

## Block registration

`GR_REGISTER_BLOCK` is a marker macro parsed at build time by `parse_registrations`.
It **must** appear inside the namespace, immediately **before** the block's `template`
declaration. It does not generate code itself — the build tool scans for it and
generates the required template instantiations and registry entries.

### Syntax

```
GR_REGISTER_BLOCK(BlockType)
GR_REGISTER_BLOCK(BlockType, [T], [type1, type2, ...])
GR_REGISTER_BLOCK(BlockType, ([T], [U]), [types for T], [types for U])
GR_REGISTER_BLOCK(BlockType, ([T], 3UZ), [types for T])
GR_REGISTER_BLOCK("custom::name", BlockType, ([T], policy<[T]>), [types for T])
```

### Variants

```cpp
// non-template block
GR_REGISTER_BLOCK(my::BlockNoTemplates)

// single type parameter — [T] is expanded over [float, double]
GR_REGISTER_BLOCK(my::Gain, [T], [float, double])

// multiple type parameters — each [T], [U] has its own type list
GR_REGISTER_BLOCK(my::Converter, ([T], [U]), [float, double], [int, long])

// extra non-type template arguments (here: 3UZ)
GR_REGISTER_BLOCK("gr::electrical::ThreePhasePower", gr::electrical::PowerMetrics,
                   ([T], 3UZ), [float, double])

// custom name + policy template argument — [T] inside policy is also expanded
GR_REGISTER_BLOCK("gr::blocks::math::AddConst", gr::blocks::math::MathOpImpl,
                   ([T], std::plus<[T]>), [float, double, std::complex<float>])
```

| Syntax element          | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `[T]`, `[U]`            | type parameters to expand                                    |
| `3UZ`, `std::plus<[T]>` | fixed non-type/type template arguments                       |
| `[float, double]`       | type list for expansion of corresponding `[T]`               |
| `"custom::name"`        | optional custom registered name (default: deduced from type) |

Reference: [`CommonBlocks.hpp`](../blocks/basic/include/gnuradio-4.0/basic/CommonBlocks.hpp), [`Math.hpp`](../blocks/math/include/gnuradio-4.0/math/Math.hpp), [`PowerEstimators.hpp`](../blocks/electrical/include/gnuradio-4.0/electrical/PowerEstimators.hpp), [`Soapy.hpp`](../blocks/soapy/include/gnuradio-4.0/soapy/Soapy.hpp)

## Dos and don'ts

| Do                                                   | Don't                                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| use `struct` (default)                               | use `class` without a genuine invariant                                                                |
| `snake_case` for reflected settings                  | `camelCase` for settings                                                                               |
| `lowerCamelCase` for methods                         | `snake_case` for methods                                                                               |
| `Annotated<T, "name", ...>` for user-facing settings | bare fields for user-facing settings (no UI metadata)                                                  |
| `[[nodiscard]]`, `constexpr`, `noexcept`             | omit attributes on processing functions                                                                |
| `std::expected` / `std::optional` for errors         | throw exceptions in library code                                                                       |
| `vir::simd` / `std::simd` for SIMD                   | raw compiler intrinsics                                                                                |
| `gr::pmt::Value` for type-erased values              | `std::variant` for wire-format data                                                                    |
| default field initialisers + `settingsChanged()`     | custom constructors/destructors (break CRTP init, settings pipeline, `emplaceBlock` property_map init) |
| `std::ranges` / `std::algorithms` for transforms     | manual `for` loops when an algorithm exists                                                            |
| lean on the framework — focus on your algorithm      | re-implement scheduling, buffering, or tag forwarding                                                  |
| verify APIs against actual headers                   | assume/hallucinate API methods                                                                         |

## Advanced: custom `work()` function

For blocks that need full control over buffer acquisition, tag handling, and
publish/consume semantics, override `work()` directly. This bypasses the framework's
`processOne`/`processBulk` dispatch entirely — you are responsible for everything.

### Signature

```cpp
gr::work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max()) noexcept {
    // ... custom implementation ...
    return { requestedWork, performedWork, gr::work::Status::OK };
}
```

### `work::Result`

```cpp
struct work::Result {
    std::size_t requested_work = std::numeric_limits<std::size_t>::max();
    std::size_t performed_work = 0;
    Status      status         = Status::OK;
};
```

### Direct port buffer access

Ports expose low-level reader/writer handles for direct buffer and tag manipulation:

```cpp
// input port — reading samples and tags
auto& reader    = in.streamReader();     // CircularBuffer reader
auto& tagReader = in.tagReader();        // tag ring-buffer reader

// output port — writing samples and tags
auto& writer    = out.streamWriter();    // CircularBuffer writer
auto& tagWriter = out.tagWriter();       // tag ring-buffer writer
```

_N.B. the reader/writer handler should be initialised and kept before the first call to `work()`.
The single producer buffer are not thread-safe and you should need to access only one at a time._

### Example: custom `work()` with manual tag handling

```cpp
gr::work::Result work(std::size_t requestedWork) noexcept {
    auto& reader = in.streamReader();
    auto& writer = out.streamWriter();

    const auto nAvailable = std::min(reader.available(), requestedWork);
    if (nAvailable == 0UZ) {
        return { requestedWork, 0UZ, gr::work::Status::INSUFFICIENT_INPUT_ITEMS };
    }

    // acquire input and output spans
    const auto inData  = reader.get(nAvailable);
    auto       outData = writer.tryReserve(nAvailable);
    if (outData.size() < nAvailable) {
        return { requestedWork, 0UZ, gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS };
    }

    // process samples
    std::ranges::transform(inData, outData.begin(), [this](const auto& s) {
        return s * static_cast<T>(gain);
    });

    // manual tag forwarding
    const auto tags = in.tagReader().get();
    for (const auto& tag : tags) {
        if (tag.index >= reader.position() && tag.index < reader.position() + nAvailable) {
            out.publishTag(tag.map, tag.index - reader.position());
        }
    }
    in.tagReader().consume(tags.size());

    // publish output and consume input
    outData.publish(nAvailable);
    std::ignore = inData.consume(nAvailable);

    return { requestedWork, nAvailable, gr::work::Status::OK };
}
```

> **Prefer `processOne` or `processBulk`** unless you have a genuine need for
> manual buffer control. The framework's default `work()` implementation handles
> settings application, tag merging, SIMD vectorisation, resampling alignment, and
> lifecycle state checks — all of which you must replicate or forgo when overriding.

Reference: [`Block.hpp`](../core/include/gnuradio-4.0/Block.hpp) (`workInternal` implementation), [`Port.hpp`](../core/include/gnuradio-4.0/Port.hpp) (`streamReader`/`streamWriter`/`tagReader`/`tagWriter`)

## See also

- [`USER_API_Tensor_Value.md`](USER_API_Tensor_Value.md) — `pmt::Value` and `Tensor<T>`
- [`USER_API_handling_blocking_blocks.md`](USER_API_handling_blocking_blocks.md) — `BlockingSync<T>` pattern
- [`USER_API_Drawable_UI.md`](USER_API_Drawable_UI.md) — `Drawable<>` UI blocks
- [`Block.hpp`](../core/include/gnuradio-4.0/Block.hpp) — `Block<Derived>` base class
- [`Port.hpp`](../core/include/gnuradio-4.0/Port.hpp) — port types and concepts
- [`annotated.hpp`](../core/include/gnuradio-4.0/annotated.hpp) — `Annotated<T>`, `Limits`, `Doc`, `Unit`, `Visible`
- [`Tag.hpp`](../core/include/gnuradio-4.0/Tag.hpp) — `Tag` struct and standard keys
- [`LifeCycle.hpp`](../core/include/gnuradio-4.0/LifeCycle.hpp) — state machine (canonical diagram)
- [`CommonBlocks.hpp`](../blocks/basic/include/gnuradio-4.0/basic/CommonBlocks.hpp) — simple reference blocks
- [`Selector.hpp`](../blocks/basic/include/gnuradio-4.0/basic/Selector.hpp) — dynamic ports, tag routing
- [`TagMonitors.hpp`](../blocks/testing/include/gnuradio-4.0/testing/TagMonitors.hpp) — tag handling patterns
- [`NullSources.hpp`](../blocks/testing/include/gnuradio-4.0/testing/NullSources.hpp) — source/sink patterns
- [`ConverterBlocks.hpp`](../blocks/basic/include/gnuradio-4.0/basic/ConverterBlocks.hpp) — resampling examples
