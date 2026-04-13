# Tag mechanics — User-API Reference

Tags are key-value metadata attached to specific sample positions in a data stream. They carry
signal descriptions (`sample_rate`, `signal_name`, ...), trigger events, context switches, and
user-defined metadata. Tags propagate through the graph alongside samples — the framework handles
forwarding, settings synchronisation, and position tracking automatically unless the block opts out.

Tag keys follow the [SigMF](https://sigmf.org/) standard where applicable to simplify import/export
of annotated signal data.

## TL;DR

- **processBulk**: read tags via `input.tags()` on the InputSpan. Publish via
  `outSpan.publishTag(map, offset)`. The framework auto-forwards standard tag keys.
- **processOne**: check `inputTagsPresent()`, read via `mergedInputTag()` (returns `const Tag&`, once
  per work call). Publish via `this->publishTag(map)` — the framework places it at the correct sample
  offset.
- **Forwarding policies**: default (forward), `BackwardTagPropagation`, or `NoTagPropagation`.
  Override `forwardTags()` for full custom control.
- Tags are **never merged** across different positions or sources. No implicit data loss.
- **Contexts** (`context` + `time` keys) switch entire settings presets per sample — a pipeline can
  coherently shift between different operating modes.

---

## Tag model

A `Tag` is `{ std::size_t index; property_map map; }` where `index` is the absolute sample position
and `map` is a `std::pmr::unordered_map<std::pmr::string, pmt::Value>`.

Tags live in circular buffers alongside (but separate from) the sample data. Each port has its own
tag buffer. When a block publishes a tag, it goes into the output port's tag buffer. The downstream
block reads it from its input port's tag buffer.

```
  tags:     ▽              ▼                    ▽  ▼        ▽
            │              │                    │  │        │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
            │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
           ─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─
  index:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

  ▽ = signal/settings tag (sample_rate, signal_name, ...)
  ▼ = trigger tag (trigger_name, trigger_time, ...)
```

Multiple tags can exist at the same sample position (e.g., a signal description and a trigger event
at sample 0):

```
  tags:     ▽▼                              two tags at sample 0
            ││
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬─
            │  │  │  │  │  │  │  │
           ─┴──┴──┴──┴──┴──┴──┴──┴─
            0  1  2  3  4  5  6  7
```

Standard tag keys are defined in `gr::tag::` (`SAMPLE_RATE`, `SIGNAL_NAME`, ...). Any string key is
valid — standard keys get automatic forwarding and settings synchronisation.

---

## Reading tags

### processBulk

The InputSpan provides direct access to tags as a lazy view — no allocation:

```cpp
work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
    for (const auto& [relIndex, tagMapRef] : input.tags()) {
        // relIndex: position relative to span start (ptrdiff_t, can be negative)
        // tagMapRef: std::reference_wrapper<const property_map>
        const property_map& map = tagMapRef.get();
    }
    return work::Status::OK;
}
```

Use `input.tags(n)` to limit to tags within the first `n` samples. For raw access (manual
consumption): `input.rawTags` is the underlying `ReaderSpan<Tag>`.

### processOne

processOne processes one sample per work call when a tag is present. Use the pre-computed merged tag:

```cpp
T processOne(T input) noexcept {
    if (this->inputTagsPresent()) {
        const Tag& tag = this->mergedInputTag();
        // tag.map contains all keys from all sync input ports at relIndex 0
    }
    return input * gain;
}
```

`inputTagsPresent()` is a trivial `bool` check — zero cost per sample. `mergedInputTag()` clears
the flag so subsequent calls in the same work call see stale data. This matches the
one-tag-per-work-call contract for processOne.

`mergedInputTag()` has a `requires` clause — it only compiles for processOne blocks. processBulk
blocks must use `input.tags()`.

---

## Publishing tags

### processBulk

Publish directly on the OutputSpan:

```cpp
work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
    output.publishTag({{"trigger_name", "peak_detected"}}, 42); // at sample offset 42
    return work::Status::OK;
}
```

### processOne

Use `this->publishTag()` — the framework determines the correct sample offset:

```cpp
T processOne(T input) noexcept {
    if (input > threshold) {
        this->publishTag({{"trigger_name", "threshold_crossed"}});
    }
    return input;
}
```

When processOne publishes a tag, the dispatch loop breaks after the current sample. The next work
call continues from the following sample. This ensures tags are positioned at the exact sample where
`publishTag` was called.

---

## Automatic tag forwarding

The framework forwards tags from input to output automatically. Three built-in policies are selected
via CRTP arguments. For full custom control, override `forwardTags()` (see below).

### Default (forward tag forwarding)

```cpp
struct MyBlock : gr::Block<MyBlock> { ... };
```

The default policy breaks chunks at tag boundaries and forwards only the tag at position 0 of each
chunk. Tags at later positions within the chunk are NOT consumed — they carry over to the next chunk
where they appear at position 0 (shifted forward). Standard tag keys are forwarded with value
substitution (the block's current value replaces the tag value if modified).

```
  input:    ▽           ▼              ▽
            │           │              │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
            0  1  2  3  4  5  6  7  8  9  10

  chunk 1:  ├───────────┤                          processedIn=4  (▽ at 0, next tag at 4)
  chunk 2:              ├──────────────┤           processedIn=5  (▼ at 4, next tag at 9)
  chunk 3:                             ├─────┤     processedIn=2  (▽ at 9, end of data)

  output:   ▽           ▼              ▽           tags forwarded at chunk start (offset 0)
```

Each chunk starts at a tag and extends to the next tag (or end of available data). The tag is always
at relIndex 0 of its chunk — at most one tagged sample per chunk, the rest are tag-free. This
applies equally to processOne and processBulk blocks.

Tags NOT at position 0 of their chunk are unconsumed and shift forward:

```
  input:    ▽     ▼  ▽                         tags at 0, 2, 3
            │     │  │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬─
            0  1  2  3  4  5  6  7

  chunk 1:  ├─────┤                            processedIn=2  (▽ at 0, next tag at 2)
                                               only tag at 0 consumed → ▽(3) stays in buffer
  chunk 2:        ├──┤                         processedIn=1  (▼ at 2, next tag at 3)
  chunk 3:           ├──────────────┤          processedIn=5  (▽ at 3, no more tags)

  output:   ▽     ▼  ▽                         tags forwarded at position 0 of their chunk
```

### Forward tag propagation (`ForwardTagPropagation`)

```cpp
struct FFTBlock : gr::Block<FFTBlock, gr::ForwardTagPropagation, gr::Resampling<1024, 1024>> { ... };
```

For fixed-chunk blocks (FFT, filter banks): chunk size is NOT broken by tag positions. Tags at
position 0 are forwarded immediately. Tags at other positions within the chunk are NOT consumed —
they carry forward to the next chunk where they appear at position 0.

```
  input:    ▽        ▼                         tags at 0 and 3
            │        │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─     input_chunk_size = 10
            0  1  2  3  4  5  6  7  8  9

  chunk:    ├─────────────────────────────┤     processedIn=10  (chunk NOT broken at tag 3)
                                                ▽ at 0 forwarded, ▼ at 3 carries to next chunk
```

### Backward tag propagation (`BackwardTagPropagation`)

```cpp
struct Decimator : gr::Block<Decimator, gr::BackwardTagPropagation> { ... };
```

ALL tags within the entire input chunk are forwarded, mapped to output position 0. All tags are
consumed per chunk. Used by decimation blocks where multiple input samples map to one output sample.

```
  input:    ▽     ▼  ▽
            │     │  │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─     input_chunk_size = 10
            0  1  2  3  4  5  6  7  8  9

  chunk:    ├─────────────────────────────┤     processedIn=10, all tags in window
                           │
                      decimate ×10
                           │
                           ▼

  output:   ▽▼▽                                all three tags at output position 0
            │││
  samples: ─┬──┬─
            0
```

### Merge tag propagation (`MergeTagPropagation`)

```cpp
struct MyBlock : gr::Block<MyBlock, gr::MergeTagPropagation> { ... };
```

All auto-forward keys from all input tags are merged into a single output tag at position 0.
This mimics the legacy forwarding behaviour. Last key wins when tags have overlapping keys.

### No tag propagation (`NoTagPropagation`)

```cpp
struct MyBlock : gr::Block<MyBlock, gr::NoTagPropagation> { ... };
```

The framework does not forward any tags. The block handles tag propagation entirely in `processBulk`
or via a custom `forwardTags()` override.

### Custom tag forwarding — `forwardTags()`

For full control, override `forwardTags()` in your block. The framework calls it instead of the
default forwarding logic:

```cpp
struct MyBlock : gr::Block<MyBlock> {
    template<typename TInputSpans, typename TOutputSpans>
    void forwardTags(TInputSpans& inputSpans, TOutputSpans& outputSpans, std::size_t processedIn) {
        for_each_reader_span([&](auto& in) {
            for (const auto& [relIndex, tagMapRef] : in.tags()) {
                property_map modified = tagMapRef.get();
                modified["my_key"] = "my_value";
                for_each_writer_span([&](auto& out) {
                    out.publishTag(modified, 0);
                }, outputSpans);
            }
        }, inputSpans);
    }
};
```

`forwardTags` is detected via `requires` at compile time — no registration needed. The CRTP policy
tags only affect the default implementation; a user `forwardTags` replaces it entirely.

---

## Multi-input deduplication

For blocks with multiple input ports, identical tags from fan-out (same index and same map content)
are deduplicated automatically by the default forwarding logic.

```
  upstream block
       │
       ├──────────┬──────────┐    fan-out: same tag on both connections
       ▽          ▽          │
  ┌──────────┬──────────┐    │
  │  in[0]   │  in[1]   │    │
  │  ▽ A     │  ▽ A     │    │    tag A appears on both ports (duplicate)
  │  ▼ B     │          │    │    tag B only on port 0 (unique)
  └──────────┴──────────┘    │
       │                     │
  dedup: A forwarded once, B forwarded once
       │
       ▽
  ┌──────────┐
  │  out[0]  │
  │  ▽ A     │    1× A (deduped) + 1× B (unique) = 2 output tags
  │  ▼ B     │
  └──────────┘
```

The dedup uses a stack-local array (capacity 8) — no heap allocation. The comparison early-returns
on first key-value mismatch. If more than 8 unique tags appear in a single work call, tags beyond
the capacity are forwarded without dedup tracking (redundant but correct — no data loss).

---

## Settings synchronisation

Tags drive automatic settings updates. When a tag arrives with a key matching a block setting (e.g.,
`sample_rate`), the framework calls `settings().autoUpdate(tag)` which stages the new value. The
block's `settingsChanged(old, new)` callback fires before the next processing call.

```
  upstream                  downstream
  ┌──────────┐              ┌──────────┐
  │ TagSource│──▽───────────│ Gain     │
  │          │  tag:        │          │
  │          │  {sr: 48k}   │ sr=48000 │    ← auto-updated from tag
  └──────────┘              └──────────┘
```

The set of auto-forwarded keys is `settings().autoForwardParameters()` — by default the standard tag
keys (`kDefaultTags`). Blocks can add custom keys:

```cpp
void start() {
    settings().autoForwardParameters().insert("my_custom_key");
}
```

### Init-time settings forwarding

When a block is constructed with explicit settings (e.g.,
`emplaceBlock<Gain>({{"sample_rate", 48000.f}})`), the changed settings are re-staged and forwarded
as a tag in the first work call. This ensures downstream blocks receive the initial configuration
without requiring an explicit tag from the source.

### `sample_rate` and decimation/interpolation

`sample_rate` receives special treatment in blocks with
`input_chunk_size != output_chunk_size`. When the framework forwards `sample_rate`, it automatically
adjusts the value based on the chunk ratio:

```
  forwarded sample_rate = input sample_rate × (output_chunk_size / input_chunk_size)
```

Example — a 10× decimator:

```
  input:    ▽ sample_rate = 10000 Hz
            │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─    input_chunk_size = 10
            │  │  │  │  │  │  │  │  │  │
           ─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─
                           │
                      decimate ×10
                           │
                           ▼
  output:   ▽ sample_rate = 1000 Hz           10000 × (1/10) = 1000
            │
  samples: ─┬──┬─                             output_chunk_size = 1
            │  │
           ─┴──┴─
```

This happens in `applyStagedParameters()` — the block author does not need to handle it manually.
The decimation block only needs to set `input_chunk_size` in `settingsChanged`:

```cpp
void settingsChanged(const property_map&, const property_map&) {
    this->input_chunk_size = decim;
}
```

---

## Contexts — modal settings switching

The `context` and `time` tag keys enable coherent pipeline-wide settings switching. A context is a
named operating mode (e.g., `"calibration"`, `"measurement"`, `"beam_A"`) with an associated UTC
timestamp.

### How contexts work

1. **Store settings for a context**: call `settings().set(params, ctx)` with a
   `SettingsCtx{.time = ..., .context = "measurement"}`. The settings are stored (not yet applied)
   and associated with that context.

2. **Activate via tag**: when a tag with `context` and/or `time` keys arrives at a block,
   `autoUpdate` calls `activateContext(ctx)`. This looks up the best-matching stored settings for
   that context and stages them.

3. **Pipeline coherence**: the same context tag propagates through the graph. Each block activates
   its own stored settings for that context — the entire pipeline switches coherently at the tagged
   sample position.

```
  timing source                     block A                block B
  ┌───────────┐                 ┌────────────┐         ┌────────────┐
  │           │──▽──────────────│ ctx="cal"  │──▽──────│ ctx="cal"  │
  │  tag:     │   propagates    │ gain=0.5   │         │ filter=LP  │
  │  {ctx:    │   through       │ (stored    │         │ (stored    │
  │   "cal"}  │   graph         │  for "cal")│         │  for "cal")│
  └───────────┘                 └────────────┘         └────────────┘

  Before tag:  gain=1.0, filter=BP      ← "default" context active
  After tag:   gain=0.5, filter=LP      ← "cal" context activated simultaneously
```

This mechanism is very powerful for:

- **Beam switching** in accelerator control (different optics settings per beam)
- **Calibration sequences** (switch to calibration mode, then back)
- **Multi-mission instruments** (telescope switching between targets)

The context string is arbitrary — the user defines the operating modes. Settings for each context
are stored independently and activated on demand.

### `context` vs `time`

- `context` (string): the mode name — selects WHICH settings to apply
- `time` (uint64, ns UTC): WHEN the settings should take effect — enables sample-accurate switching
  synchronised across the pipeline

Both are optional in a tag. A tag with only `context` activates immediately. A tag with both uses the
timestamp for precise scheduling.

---

## Trigger tags

Trigger tags mark events in the data stream — zero crossings, threshold exceedances, external timing
events, etc. They use a standard key triplet (plus one optional key):

| Key                 | Type            | Required | Description                                               |
| ------------------- | --------------- | -------- | --------------------------------------------------------- |
| `trigger_name`      | `std::string`   | yes      | event identifier (e.g., `"zero_crossing"`, `"beam_in"`)   |
| `trigger_time`      | `uint64_t` (ns) | yes      | UTC timestamp of the event                                |
| `trigger_offset`    | `float` (s)     | yes      | sub-sample time offset (compensating analog group delays) |
| `trigger_meta_info` | `property_map`  | optional | additional key-value pairs (detector ID, amplitude, ...)  |

**These keys should always appear together** (except `trigger_meta_info` which is optional). A
trigger tag without `trigger_time` or `trigger_offset` is incomplete.

```
  trigger tag at sample 5:
  {
      "trigger_name":      "beam_extraction",
      "trigger_time":      1720000000000000000,    // UTC ns
      "trigger_offset":    -0.000023,              // 23 µs before this sample
      "trigger_meta_info": {
          "beam_id":    "PS_B1",
          "intensity":  2.3e12
      }
  }
```

Trigger tags are auto-forwarded like any standard tag key. Downstream blocks (e.g.,
`StreamToDataSet`) use the trigger matcher API to start/stop data acquisition windows based on
trigger patterns.

---

## Standard tag keys — `kDefaultTags`

All standard keys use the `gr:` prefix internally (e.g., `gr:sample_rate`) but are referenced by
short key in code (`"sample_rate"`). Keys are aligned with the
[SigMF](https://sigmf.org/) standard for interoperability.

### Signal description

| Key               | Type          | Unit | Purpose                                                               |
| ----------------- | ------------- | ---- | --------------------------------------------------------------------- |
| `sample_rate`     | `float`       | Hz   | sample rate of the stream. Auto-adjusted by decimation/interpolation. |
| `signal_name`     | `std::string` |      | human-readable signal name (e.g., `"RF_pickup_H"`)                    |
| `num_channels`    | `gr::Size_t`  |      | interleaved channel count (for multi-channel streams)                 |
| `signal_quantity` | `std::string` |      | physical quantity (e.g., `"voltage"`, `"current"`)                    |
| `signal_unit`     | `std::string` |      | SI unit (e.g., `"V"`, `"A"`)                                          |
| `signal_min`      | `float`       | a.u. | physical minimum (e.g., DAQ range lower bound)                        |
| `signal_max`      | `float`       | a.u. | physical maximum (e.g., DAQ range upper bound)                        |

### Data quality

| Key                 | Type         | Purpose                                                       |
| ------------------- | ------------ | ------------------------------------------------------------- |
| `n_dropped_samples` | `gr::Size_t` | number of samples lost before this point (DAQ overflow, etc.) |

### Triggers

| Key                 | Type           | Unit | Purpose                    |
| ------------------- | -------------- | ---- | -------------------------- |
| `trigger_name`      | `std::string`  |      | event identifier           |
| `trigger_time`      | `uint64_t`     | ns   | UTC timestamp of the event |
| `trigger_offset`    | `float`        | s    | sub-sample time correction |
| `trigger_meta_info` | `property_map` |      | additional event metadata  |

### Context switching

| Key       | Type            | Purpose                                                      |
| --------- | --------------- | ------------------------------------------------------------ |
| `context` | `std::string`   | multiplexing key — selects which stored settings to activate |
| `time`    | `uint64_t` (ns) | UTC timestamp for sample-accurate context activation         |

### Lifecycle

| Key             | Type   | Purpose                                                      |
| --------------- | ------ | ------------------------------------------------------------ |
| `reset_default` | `bool` | reset block settings to stored defaults                      |
| `store_default` | `bool` | store current settings as new defaults                       |
| `end_of_stream` | `bool` | signals end of data — downstream blocks transition to `DONE` |

---

## Performance notes

- `inputTagsPresent()` is a `bool` check — zero cost in the no-tag path.
- `input.tags()` is a lazy view over the tag buffer — no allocation.
- `settings().get()` (for value substitution during forwarding) is allocated lazily via
  `std::optional` — only when an auto-forward key is actually found in a tag.
- The processOne dispatch loop checks `_outputTagPending` (a `bool`) per sample —
  branch-predicted, no overhead when no tags are published.
- Single-input blocks skip multi-port dedup entirely (compile-time `if constexpr`).
- Tag-heavy workloads (tag every sample) are dominated by `property_map` allocation, not by the
  forwarding logic.
