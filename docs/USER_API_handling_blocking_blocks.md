# `gr::BlockingSync<T>` User-API Reference

## Overview

`BlockingSync<T>` is a CRTP mixin for blocks requiring **wall-clock synchronised timing** or rely on **external blocking
threads** - clock sources, signal generators, and hardware interface blocks.

> **Note:** GR4 uses work-scheduling where blocks should **never block** inside `processBulk()`/`processOne()`.
> `BlockingSync` is an exception for the rare cases requiring real-time synchronisation. Most blocks will not need this.

> **Deprecation:** `BlockingSync<T>` replaces the previous `BlockingIO<>` template parameter.

## Operating Modes

| Mode                        | Condition                                           | Behaviour                            |
| --------------------------- | --------------------------------------------------- | ------------------------------------ |
| **Clock-connected**         | `clk_in` connected                                  | Scheduler-driven, 1 output per input |
| **Free-running (internal)** | `clk_in` not connected, `use_internal_thread=true`  | GR4 timer thread wakes scheduler     |
| **Free-running (external)** | `clk_in` not connected, `use_internal_thread=false` | On-demand / BYO thread               |

## Quick Start

### Source Block (no inputs)

```cpp
#include <gnuradio-4.0/BlockingSync.hpp>

GR_REGISTER_BLOCK(MyClockSource)

struct MyClockSource : gr::Block<MyClockSource>, gr::BlockingSync<MyClockSource> {
    gr::PortOut<std::uint8_t> clk;

    gr::Annotated<float, "sample_rate">        sample_rate         = 1000.f; // mandatory field
    gr::Annotated<gr::Size_t, "chunk_size">    chunk_size          = 100;    // optional
    gr::Annotated<bool, "use_internal_thread"> use_internal_thread = true;   // optional

    GR_MAKE_REFLECTABLE(MyClockSource, clk, sample_rate, chunk_size, use_internal_thread);

    void start() { this->blockingSyncStart(); }
    void stop()  { this->blockingSyncStop(); }

    gr::work::Status processBulk(gr::OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(output);

        if (nSamples == 0) {
            output.publish(0);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            output[i] = generateSample();
        }

        output.publish(nSamples);
        return gr::work::Status::OK;
    }
};
```

### Generator Block (optional clock input)

```cpp
template<typename T>
struct MyGenerator : gr::Block<MyGenerator<T>>, gr::BlockingSync<MyGenerator<T>> {
    gr::PortIn<std::uint8_t, gr::Optional> clk_in;
    gr::PortOut<T>                         out;

    gr::Annotated<float, "sample_rate">     sample_rate = 1000.f;
    gr::Annotated<gr::Size_t, "chunk_size"> chunk_size  = 100;

    GR_MAKE_REFLECTABLE(MyGenerator, clk_in, out, sample_rate, chunk_size);

    void start() { this->blockingSyncStart(); }
    void stop()  { this->blockingSyncStop(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        // syncSamples handles mode detection internally:
        // - clock-driven (clk_in connected): returns min(input.size(), output.size())
        // - free-running (clk_in disconnected): returns wall-clock based sample count
        const auto nSamples = this->syncSamples(input, output);

        if (nSamples == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            output[i] = generateSample();
        }

        // Consume clock samples only in clock-driven mode
        std::ignore = input.consume(this->isFreeRunning() ? 0 : nSamples);
        output.publish(nSamples);
        return gr::work::Status::OK;
    }
};
```

## API Reference

### Required Members (in derived class)

| Member        | Type    | Description              |
| ------------- | ------- | ------------------------ |
| `sample_rate` | `float` | Output sample rate in Hz |

### Optional Members

| Member                | Type                          | Default          | Description          |
| --------------------- | ----------------------------- | ---------------- | -------------------- |
| `chunk_size`          | `gr::Size_t`                  | `sample_rate/10` | Samples per update   |
| `use_internal_thread` | `bool`                        | `true`           | Use GR4 timer thread |
| `clk_in` or `clk`     | `PortIn<uint8_t[, Optional]>` | -                | Clock input port     |

### Methods

```cpp
// Lifecycle - call from your start()/stop()
void blockingSyncStart();
void blockingSyncStop();

// Runtime checks
bool isFreeRunning() const;        // true if clock port unconnected
bool isUsingInternalThread() const; // true if internal timer used

// Sample calculation (call in processBulk)
// For blocks with optional clock input:
std::size_t syncSamples(TInput& input, TOutput& output, bool dropIfBehind = false);
// For pure source blocks (no clock input):
std::size_t syncSamples(TOutput& output, bool dropIfBehind = false);
// Direct size overload (for testing or when span size is known):
std::size_t syncSamples(std::size_t maxSamples, bool dropIfBehind = false);

// Time tracking
TimePoint blockingSyncStartTime() const;
TimePoint blockingSyncLastUpdateTime() const;

// Manual timing control (for HW drivers)
void blockingSyncResetTiming();
```

### The `syncSamples()` Method

The `syncSamples()` method is the primary API for determining how many samples to produce:

**Two-argument overload** (for blocks with optional clock input):

```cpp
const auto nSamples = this->syncSamples(input, output);
```

- If clock port is connected: returns `min(input.size(), output.size())`
- If clock port is disconnected: returns wall-clock based sample count

**One-argument overload** (for pure source blocks):

```cpp
const auto nSamples = this->syncSamples(output);
```

- Always uses wall-clock timing

**The `dropIfBehind` parameter:**

- `false` (default): Advances time by exact sample duration, allowing catch-up if behind schedule
- `true`: Advances time to now, dropping samples to prevent unbounded latency

## Internal Thread Control

```cpp
// Default: internal timer thread (consistent wall-clock timing)
struct MySource : Block<MySource>, BlockingSync<MySource> {
    Annotated<float, "sample_rate"> sample_rate = 1000.f;
    // use_internal_thread defaults to true
};

// Disable internal thread (on-demand or HW-driven)
struct MyHWSource : Block<MyHWSource>, BlockingSync<MyHWSource> {
    Annotated<float, "sample_rate">        sample_rate         = 1000.f;
    Annotated<bool, "use_internal_thread"> use_internal_thread = false;
};
```

**`use_internal_thread = true`:**

- GR4 spawns timer thread from IO thread-pool
- Wakes scheduler every `chunk_size / sample_rate` seconds
- Consistent output rate regardless of downstream load

**`use_internal_thread = false`:**

- No thread spawned by BlockingSync
- Samples calculated based on elapsed time when `processBulk` called
- Use for HW drivers (BYO thread) or testing

## Common Patterns

### Real-time dropping (live signals)

```cpp
// Skip ahead if falling behind (e.g., audio visualisation)
const auto nSamples = this->syncSamples(output, true);  // dropIfBehind = true
```

### Hardware driver integration

```cpp
struct MySDRSource : Block<MySDRSource>, BlockingSync<MySDRSource> {
    Annotated<float, "sample_rate">        sample_rate         = 1e6f;
    Annotated<bool, "use_internal_thread"> use_internal_thread = false;  // HW has own thread

    std::atomic<bool> dataReady{false};
    RingBuffer<T>     hwBuffer;

    void start() { this->blockingSyncStart(); /* init HW */ }
    // ... see lifecycle state machine for other available methods
    void stop()  { this->blockingSyncStop(); /* close HW */ }

    // called by HW driver thread
    void hwCallback(std::span<const T> samples) {
        hwBuffer.push(samples);
        dataReady = true;
        this->progress->incrementAndGet();  // wake scheduler
        this->progress->notify_all();
    }

    work::Status processBulk(OutputSpanLike auto& output) {
        if (!dataReady.exchange(false)) {
            output.publish(0);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        std::size_t nSamples = hwBuffer.pop(output);
        output.publish(nSamples);
        return nSamples > 0UZ ? gr::work::Status::OK : gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
    }
};
```

### Pause handling

`BlockingSync` automatically maintains phase continuity across pause/resume cycles.

## See Also

- `gr::basic::ClockSource` - Reference source implementation
- `gr::basic::SignalGenerator` - Generator with optional clock
- `gr::basic::FunctionGenerator` - Function generator with optional clock
