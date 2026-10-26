# Tag mechanics — user-API reference

Tags are key-value metadata attached to specific sample positions in a data stream.
They carry signal descriptions (`sample_rate`, `signal_name`, ...), trigger events, context switches and user-defined metadata.
Tags travel through the graph alongside the samples;
the framework forwards them, keeps the blocks' settings in step with them and tracks their positions, unless a block opts out.

Tag keys follow the [SigMF](https://sigmf.org/) standard where applicable, to simplify the import and export of annotated signal data.

## Summary

- **Forward-all**: every input tag is forwarded with all its keys, on the sample it marks;
  a key naming one of the block's settings carries the block's value, and the rate keys follow resampling.
  A propagation policy changes where a tag goes, not what it holds.
- **processBulk**: read tags via `input.tags()` on the InputSpan, publish via `output.publishTag(map, offset)`.
- **processOne**: check `inputTagsPresent()`, read via `mergedInputTag()` (`const Tag&`, once per work call), publish via `this->publishTag(map)`; the framework places it on the current sample.
- **Policies**: default, `ForwardTagPropagation`, `BackwardTagPropagation`, `MergeTagPropagation`, `NoTagPropagation`;
  custom forwarding is `NoTagPropagation` plus `forwardDelayedTags` / `forwardedTag` / `forwardInputTags` in `processBulk`.
- **Canonical settings**: a setting named like a canonical key (`sample_rate`, `frequency`, ...) follows that tag and is announced downstream as that tag;
  declare it through its alias (`gr::SampleRate sample_rate;`), which fixes type, unit and documentation; another type fails to compile.
- **Rates**: a resampling block rescales `gr:sample_rate` (and `gr:frame_rate`) by its chunk ratio; a framing block keeps `gr:sample_rate` and rescales `gr:frame_rate`.
- **Contexts** (`gr:context`, timed by `gr:trigger_time` or `gr:ctx_time`) switch entire settings presets at a sample, so a pipeline shifts coherently between operating modes.

---

## Tag model

A `Tag` is `{ std::size_t index; ValueMapView map; }`, where `index` is the absolute sample position and `map` is a **non-owning view** onto a packed key→value blob.
The in-memory layout is the wire format, so a tag can be handed to a device through unified shared memory (USM) or to another process through inter-process communication (IPC) as it is.
`Tag` is trivially copyable: it is a descriptor, not a container.

Each output port owns one tag buffer (`ChunkBuffer<Tag>`) beside its sample buffer, and every input connected to it reads that buffer through its own reader.
Publishing a tag copies its payload into the output's tag buffer; the downstream block reads a view of it.
A `Tag` obtained from an input span is therefore only valid **while that span is alive**:
to keep a payload beyond the work call, copy it into an owning `property_map` (`gr::pmt::ValueMap`).

```text
  tags:     ▽              ▼                    ▽  ▼        ▽
            │              │                    │  │        │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
            │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
           ─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─
  index:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

  ▽ = signal/settings tag (sample_rate, signal_name, ...)
  ▼ = trigger tag (trigger_name, trigger_time, ...)
```

Several tags can sit on the same sample (e.g. a signal description and a trigger event at sample 0, `▽▼`).

### Tag keys: canonical, setting commands and user data

> [!WARNING]
> A tag whose key matches a setting name reconfigures every downstream block that has that setting.
> `name`, `trigger_name` and `n` are such keys: a tag carrying `name` renames each downstream block, and `n` changes the copy count of every downstream `Repeat`.
> Give custom tags a namespaced key that matches no setting (`myexperiment:run_id`), or put the data under `gr:user_data`.

A tag map holds three kinds of root key:

| kind            | example                             | defined by                    | at each downstream block                                                        |
| --------------- | ----------------------------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| canonical key   | `gr:sample_rate`, `gr:trigger_name` | `gr::tag::` in `Tag.hpp` (37) | forwarded; applied to the block's setting of the same short name, if it has one |
| setting command | `delay`, `n_taps`                   | the blocks' public settings   | forwarded; applied by **every** block that has a setting of that name           |
| user data       | anything under `gr:user_data`       | the user                      | forwarded; applied only by a block that declares a `user_data` setting          |

- Emit canonical keys through their definition (`gr::tag::SAMPLE_RATE(48000.f)`, `gr::tag::TRIGGER_NAME.key()`), never as a hand-written string.
- Put metadata that is not meant to change a setting under `gr:user_data`, and only what a downstream consumer reads.
- Do not invent `gr:` keys: a block strips the prefix before matching its settings, so `gr:delay` acts like `delay`.

```cpp
const gr::property_map runInfo{{"run", 42}, {"operator", "night shift"}};
out.publishTag(gr::property_map{{gr::tag::TRIGGER_NAME.key(), "shot"}, {gr::tag::USER_DATA.key(), runInfo}}, 0UZ);
```

### Setting aliases: one per canonical key

Each canonical key has one setting alias, its key in UpperCamelCase (`gr::SampleRate`, `gr::Frequency`, `gr::SignalName`, ...; full list in [Standard tag keys](#standard-tag-keys--kdefaulttags)).
The alias is `gr::SettingOf<decltype(gr::tag::KEY), extras…>::type`, an `Annotated<T, "key", Unit<…>, Doc<…>, extras…>` whose type, unit and documentation are the tag's, so each is defined once:

```cpp
struct MySource : gr::Block<MySource> {
    gr::PortOut<float> out;

    gr::SampleRate sample_rate = 48000.f; // float, Hz, Visible, Limits<0, max>
    gr::Frequency  frequency   = 107.0e6; // double, Hz, Visible
    gr::SignalName signal_name;           // std::pmr::string

    GR_MAKE_REFLECTABLE(MySource, out, sample_rate, frequency, signal_name);

    [[nodiscard]] constexpr float processOne() const noexcept { return 0.f; }
};
```

- `gr::Visible` only on `SampleRate`, `FrameRate` and `Frequency`;
- `SampleRate`, `FrameRate` and `EstSampleRate` refuse negative values; their limits allow 0, the convention for "not yet known";
- the string keys (`signal_name`, `signal_quantity`, `signal_unit`, `trigger_name`, `context`) are `std::string` on the wire and `std::pmr::string` as settings, so a block that keeps its settings on a memory resource needs no hand-written declaration.

**Compile-time rule**: a setting named like a canonical key holds the alias' value type, or a `std::vector` of it (one per channel); anything else fails to compile (`gr::tag::hasCanonicalSettingType<T>(name)`, asserted for every setting):

```cpp
float       frequency = 1000.f; // error: a setting named like a canonical tag key must have the tag's value type, or a std::vector of it
std::string signal_name;        // error: the setting type is std::pmr::string
```

A deliberate exception — hardware ranges, a strictly positive rate where a block divides by it, per-channel values — is a hand-written `Annotated` with the canonical type and unit and its own limits:

```cpp
gr::Annotated<float, "sample_rate", gr::Unit<"Hz">, gr::Visible, gr::Limits<225e3f, 3.2e6f>, gr::Doc<"ADC sample rate">> sample_rate = 2.048e6f;

gr::Annotated<float, "sample_rate", gr::Unit<"Hz">, gr::Doc<"places the bins">,
              gr::Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate = 1.f;

gr::Annotated<std::vector<double>, "frequency", gr::Unit<"Hz">, gr::Visible, gr::Doc<"per-channel centre frequency">> frequency;
```

---

## Reading tags

### processBulk

The InputSpan provides direct access to tags as a lazy view, without allocation:

```cpp
gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
    for (const auto& [relIndex, tagMapRef] : input.tags()) { // relIndex: std::ptrdiff_t, relative to the span start, can be negative
        const gr::pmt::ValueMapView& map = tagMapRef.get(); // valid for this work call only
    }
    return gr::work::Status::OK;
}
```

`input.tags(n)` limits the view to tags before local index `n`.
`input.rawTags()` yields the non-owning `Tag` descriptors over the ring slots, valid only within the span, for device consumers.

Binding a view to an owning `property_map` (`const gr::property_map& map = tagMapRef.get();`) compiles, since the converting constructor is implicit, but allocates and copies the blob per tag.
Take `const ValueMapView&` unless you intend to own the payload.

### processOne

A tag starts a chunk, so in processOne it belongs to the first sample of the work call.
Read it through the pre-computed merged tag:

```cpp
T processOne(T input) noexcept {
    if (this->inputTagsPresent()) {
        const gr::Tag& tag = this->mergedInputTag();
        // … tag.map
    }
    return input * gain;
}
```

`mergedInputTag()` views the tags of all sync, connected input ports on the first sample, merged (last port wins per key) for a block with several inputs; the merged payload lives in the block, so the view stays valid.
`inputTagsPresent()` is a `bool` check; `mergedInputTag()` clears it, so later samples of the same work call see no tag.
`mergedInputTag()` compiles only for processOne blocks; a processBulk block uses `input.tags()`.

Limits, as the code stands:

- only tags exactly on the first sample (relIndex 0) are shown: a tag deferred from an earlier sample (negative relIndex, e.g. one inside a minimum chunk) is not;
- a single-input block with two tags on the same sample shows none (`inputTagsPresent()` stays `false`).

Both kinds are still applied to the settings and forwarded.

---

## Publishing tags

### processBulk

Publish on the OutputSpan, offsets ascending per output:

```cpp
gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
    output.publishTag(gr::property_map{{gr::tag::TRIGGER_NAME.key(), "peak_detected"}}, 42UZ); // on output sample 42
    return gr::work::Status::OK;
}
```

### processOne

`this->publishTag()` places the tag on the current sample:

```cpp
T processOne(T input) noexcept {
    if (input > threshold) {
        this->publishTag(gr::property_map{{gr::tag::TRIGGER_NAME.key(), "threshold_crossed"}});
    }
    return input;
}
```

The dispatch loop stops after the sample on which a tag was published, and the next work call continues with the following one, so the tag lands on the sample where `publishTag` was called.

---

## Automatic tag forwarding

The framework forwards every tag of each sync, connected input to every connected, single-producer output: all its keys, with or without the `gr:` prefix, on the sample it marks, or at the start of a later chunk where the policy or the chunk size defers it (forward-all).
Asynchronous and unconnected inputs are not forwarded; a tag for an unconnected output is dropped; an output written by several producers receives no forwarded tags.
Tags are forwarded one by one, never merged across positions.
On the way, three things change:

- a key naming one of the block's settings carries the block's current value;
- `gr:sample_rate` and `gr:frame_rate` follow the resampling ratio ([below](#resampling-and-framing));
- where the tag is rebuilt for either reason, a `gr:frame_rate` equal to `gr:sample_rate` is dropped.

A block with `Stride<>` forwards the tags of the samples a stride skips at offset 0 of its next output.
The CRTP policy arguments change where tags go:

| policy                   | chunk ends          | forwarded per work call                      |
| ------------------------ | ------------------- | -------------------------------------------- |
| default                  | before the next tag | the tag at position 0, on position 0         |
| `ForwardTagPropagation`  | regardless of tags  | the tag at position 0; later ones wait       |
| `BackwardTagPropagation` | before the next tag | every tag of the chunk, on output position 0 |
| `MergeTagPropagation`    | before the next tag | the tags at position 0, merged into one      |
| `NoTagPropagation`       | before the next tag | none; the block publishes its own            |

### Default

```cpp
struct MyBlock : gr::Block<MyBlock> { ... };
```

The default policy breaks chunks at tag boundaries: each chunk starts at a tag and extends to the next one (or the end of the available data), so at most one tagged sample per chunk, at relIndex 0.
This applies equally to processOne and processBulk blocks.
A tag that is not at position 0 is not consumed: it starts the next chunk.

```text
  input:    ▽     ▼  ▽                         tags at 0, 2, 3
            │     │  │
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬─
            0  1  2  3  4  5  6  7

  chunk 1:  ├─────┤                            processedIn=2  (▽ at 0, next tag at 2)
  chunk 2:        ├──┤                         processedIn=1  (▼ at 2, next tag at 3)
  chunk 3:           ├──────────────┤          processedIn=5  (▽ at 3, no more tags)

  output:   ▽     ▼  ▽                         each tag on the sample it marked
```

A chunk is never shorter than `input_chunk_size` or the sync inputs' `min_samples`:
a tag closer to the chunk start moves to position 0 of the next chunk, later than the sample it marked.

### Forward tag propagation (`ForwardTagPropagation`)

```cpp
struct FFTBlock : gr::Block<FFTBlock, gr::ForwardTagPropagation, gr::Resampling<1024, 1024>> { ... };
```

For fixed-chunk blocks (FFT, filter banks): tag positions do not break chunks.
A tag at position 0 is forwarded at once; a tag elsewhere in the chunk is not consumed and moves to position 0 of the next chunk.

```text
  input:    ▽        ▼                         tags at 0 and 3, input_chunk_size = 10
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
            0  1  2  3  4  5  6  7  8  9
  chunk:    ├─────────────────────────────┤    processedIn=10: ▽ forwarded, ▼ to the next chunk's start
```

### Backward tag propagation (`BackwardTagPropagation`)

```cpp
template<typename T>
struct Decimator : gr::Block<Decimator<T>, gr::BackwardTagPropagation, gr::Resampling<10, 1>> { ... };
```

Every tag within the input chunk is forwarded on output position 0 and consumed.
Used by decimating blocks, where many input samples map to one output sample.

```text
  input:    ▽     ▼  ▽                         input_chunk_size = 10
  samples: ─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─
            0  1  2  3  4  5  6  7  8  9
                      decimate ×10
  output:   ▽▼▽                                all three tags at output position 0
```

### Merge tag propagation (`MergeTagPropagation`)

```cpp
struct MyBlock : gr::Block<MyBlock, gr::MergeTagPropagation> { ... };
```

All keys of the tags at position 0, over all inputs, merge into a single output tag at position 0, the last port winning per key.
This mimics the legacy forwarding behaviour.
Only this policy and `mergedInputTag()` merge the tags of several inputs.

### No tag propagation (`NoTagPropagation`)

```cpp
struct MyBlock : gr::Block<MyBlock, gr::NoTagPropagation> { ... };
```

The framework forwards no input tags; only the tags the block publishes itself, and the `gr:end_of_stream` the framework emits when the block stops, go downstream.
Input tags still apply to the block's settings, but a change of a setting is not announced downstream.

### Custom forwarding: `NoTagPropagation` and `processBulk`

A block that moves, routes or rewrites tags (a delay, a per-port route, an index remap, a run-time switch) takes `NoTagPropagation` and forwards from a `processBulk` over tag-capable spans (`InputSpanLike` / `OutputSpanLike`), once per work call, from its host and device overloads alike:

- `forwardedTag(tagMap, emit)`: one tag as default forwarding emits it (setting values substituted, rates rescaled); `emit` receives the input blob itself when nothing changes, a rebuilt `property_map` otherwise;
- `forwardDelayedTags(pending, input, outputs, n, delay)`: for a 1:1 block whose output sample `i + delay` carries input sample `i`; each tag waits in `pending` (`gr::DelayedTags`) across calls, and one whose sample is gone is erased; `n` is the number of samples this call processed;
- `forwardInputTags(inputs, outputs, nIn)`: the default forwarding, to switch on or off at run time; `nIn` is the number of input samples this call consumed.

`outputs` (and `inputs`) are tuples of spans, `std::tie(out)` or `std::tie(realOut, imagOut)`.

```cpp
template<typename T>
struct MyDelay : gr::Block<MyDelay<T>, gr::NoTagPropagation> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    gr::Annotated<gr::Size_t, "delay", gr::Doc<"output sample n + delay carries input sample n">> delay = 4U;

    GR_MAKE_REFLECTABLE(MyDelay, in, out, delay);

    gr::DelayedTags _delayedTags;

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        this->forwardDelayedTags(_delayedTags, input, std::tie(output), output.size(), delay.value);
        // … the samples, delayed by `delay`
        return gr::work::Status::OK;
    }
};
```

A tag of any other shape is read with `input.tags()` and published with `output.publishTag(map, offset)`, through `forwardedTag` where settings and rates should apply as usual (`Mux` places each input's tags on their muxed output sample this way).

### Multi-input deduplication

For blocks with several input ports, identical tags from a fan-out (same position and same map content) are forwarded once:

```text
  upstream ──┬── in[0]: ▽A ▼B ──┐
             └── in[1]: ▽A    ──┴── block ── out[0]: ▽A ▼B     A once (duplicate), B once (unique)
```

The dedup uses a stack-local array of 8 entries, without heap allocation;
beyond 8 distinct tags in one work call, further tags are forwarded without dedup tracking (redundant but correct, no loss).
A port collection (`std::vector` or `std::array` of ports) counts as one port: neither the dedup nor `mergedInputTag()` works across its elements.

---

## Settings synchronisation

Tags drive settings updates.
When a tag arrives with a key matching a setting that still follows tags (e.g. `sample_rate`), `settings().autoUpdate(tagMap)` stages the new value, and the block's `settingsChanged(old, new)` runs before the next processing call.
A value equal to the active one is not staged again, so a constant setting tag costs no reconfiguration.

```text
  upstream                  downstream
  ┌──────────┐              ┌──────────┐
  │ TagSource│──▽───────────│ Gain     │
  │          │  tag:        │          │
  │          │  {sr: 48k}   │ sr=48000 │    ← auto-updated from tag
  └──────────┘              └──────────┘
```

A changed setting is announced downstream as a tag only for the canonical keys (with the `gr:` prefix); other settings stay local.
Settings given at construction are announced in the block's first work call, so downstream blocks receive the initial configuration without a tag from the source.
A tag arriving on an input travels on whole, whatever its keys.

---

## Resampling and framing

`gr:sample_rate` is the rate at which the signal was sampled; only a block that resamples the signal changes it.
A block with a `Resampling<>` control rescales the rate keys it forwards — whether forwarded from an input or announced from its own settings — by

```text
  forwarded rate = input rate × output_chunk_size / hop        hop = stride if Stride<> is set and stride ≠ 0, else input_chunk_size
```

whenever `hop ≠ output_chunk_size` (`resamplingHop()`, `rescalesRates()` in `Block.hpp`);
a sliding window reads more than it consumes, so `Resampling<8, 8>` with `Stride<4>` turns 1000 Hz into 2000 Hz.
Any numeric rate is rescaled and goes out as a `float`; the block's own settings keep the input rate.
`Stride<>` without `Resampling<>` never rescales.

A block that packs samples into frames — a stream into tensors, M streams into [M, N] frames — or unpacks them is not resampling and declares `gr::Framing`:
its forwarded `gr:sample_rate` stays as it is, and `gr:frame_rate` — frames (items) per second on the edge — follows the chunk ratio instead (from `gr:sample_rate` where no frame rate arrives).
Without `Framing`, both rate keys are rescaled.
Where the tag is rebuilt, a `gr:frame_rate` equal to `gr:sample_rate` is dropped, so unpacking back into a stream removes it.

```text
  stream ▽ sample_rate 48000
    ├─ decimate 4:1   Resampling<4, 1>               ─▶ ▽ sample_rate 12000
    └─ frame 1024     Resampling<1024, 1>, Framing   ─▶ ▽ sample_rate 48000, frame_rate 46.875   (48000 × 1/1024)
         └─ unframe   Resampling<1, 1024>, Framing   ─▶ ▽ sample_rate 48000                      (46.875 × 1024 = 48000: dropped)
```

A decimating block only sets its chunk sizes, e.g. in `settingsChanged`:

```cpp
template<typename T>
struct Decimator : gr::Block<Decimator<T>, gr::BackwardTagPropagation, gr::Resampling<10, 1>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    gr::Annotated<gr::Size_t, "decim", gr::Doc<"input samples per output sample">> decim = 10U;

    GR_MAKE_REFLECTABLE(Decimator, in, out, decim);

    void settingsChanged(const gr::property_map&, const gr::property_map&) { this->input_chunk_size = decim.value; }
    // … processBulk
};
```

A numerical block takes the signal's rate from `gr:sample_rate` (bin spacing fₛ/N, lag step 1/fₛ) and a time per item from `gr:frame_rate`, or from `gr:sample_rate` where no frame rate arrives.

---

## Contexts — modal settings switching

The `gr:context` and `gr:trigger_time` keys switch settings coherently across a pipeline.
A context is a named operating mode (e.g. `"calibration"`, `"measurement"`, `"beam_A"`) with a UTC time-stamp.

1. **Store settings for a context**: `settings().set(params, gr::SettingsCtx{.time = ..., .context = "measurement"})` stores the settings (not yet applied) for that context.
2. **Activate via tag**: when a tag with a `gr:context` key arrives, `autoUpdate` calls `activateContext(ctx)`, which looks up the best-matching stored settings for that context and stages them.
3. **Pipeline coherence**: the same tag propagates through the graph, and each block activates its own stored settings for that context, so the entire pipeline switches at the tagged sample.

```text
  timing source                     block A                block B
  ┌───────────┐                 ┌────────────┐         ┌────────────┐
  │           │──▽──────────────│ ctx="cal"  │──▽──────│ ctx="cal"  │
  │  tag:     │   propagates    │ gain=0.5   │         │ filter=LP  │
  │  {ctx:    │   through       │ (stored    │         │ (stored    │
  │   "cal"}  │   graph         │  for "cal")│         │  for "cal")│
  └───────────┘                 └────────────┘         └────────────┘

  Before tag:  gain=1.0, filter=BP      ← default context "" active
  After tag:   gain=0.5, filter=LP      ← "cal" context activated simultaneously
```

Typical uses: beam switching in accelerator control (optics settings per beam), calibration sequences, multi-mission instruments switching between targets.
The context string is arbitrary; the settings of each context are stored independently and activated on demand.

- `gr:context` (string): the mode name, which settings to apply;
- `gr:trigger_time` (`std::uint64_t`, ns UTC) in the same tag, else `gr:ctx_time`: when they take effect, for a switch synchronised across the pipeline;
  without either, the host's `std::chrono::system_clock::now()` when the tag is applied.

A message or a flow-graph file names the time `ctx_time`.

---

## Trigger tags

Trigger tags mark events in the data stream: zero crossings, threshold exceedances, external timing events.
A trigger tag carries `trigger_name`, `trigger_time` and `trigger_offset` together, optionally `trigger_meta_info` and the window keys (`trigger_time_error`, `trigger_stop_time`, `trigger_pre`, `trigger_post`); types and units are in the [table below](#standard-tag-keys--kdefaulttags).

```text
  trigger tag at sample 5:
  {
      "gr:trigger_name":      "beam_extraction",
      "gr:trigger_time":      1720000000000000000,
      "gr:trigger_offset":    0.4e-9,           // sub-ns residual: the event is at trigger_time + trigger_offset
      "gr:trigger_meta_info": {
          "beam_id":    "PS_B1",
          "intensity":  2.3e12
      }
  }
```

`trigger_time + trigger_offset` is the trigger's moment, and its tag sits on the first sample at or after it.
The full rules, and how tags compare with events and messages as transports, are in [Triggers](USER_API_Triggers.md).

Trigger tags are forwarded like any tag.
Downstream blocks (e.g. `StreamToDataSet`) use the trigger matcher API to start and stop acquisition windows on trigger patterns.

---

## Pitfalls

- **A setting named like a canonical key is not private.**
  It is overwritten by every upstream tag of that key and announced downstream as that tag on every change.
  A value with another meaning under that name (an offset under `frequency`) is either overwritten by the band centre an upstream source announces or, set by hand, announced downstream as the band centre.
  Give a non-canonical quantity its own name (`signal_label`, `offset_frequency`).
- **A setting set by hand stops following tags.**
  A value given at construction (`graph.emplaceBlock<gr::blocks::math::Rotator<std::complex<float>>>({{"sample_rate", 48000.f}})`) or through `settings().set(...)` clears that setting's auto-update in its context (a settings message, staged through `setStaged`, does not):
  later upstream tags of that key no longer change it, and the block forwards its own value in their place.
  `settings().autoUpdateParameters()` lists the settings that still follow tags.
- **A setting command travels on.**
  A root key named like a setting (`size`, `delay`, `n_taps`, ...) reconfigures every downstream block that has that setting, not only the next one, and arrives there with the value the last block applied.
  `TagToEvent` (`blocks/trigger`) routes the tags its `filter` names to an event port unchanged, or every tag without one, which shows what travels past a point.
- **Narrowing.**
  A tag value of another type than the setting is converted as a settings message is (`gr::pmt::convert_safely`): an integer only if its bits fit the mantissa, but a `double` into a `float` only checked for range, so its precision is lost without a warning.
  The compile-time rule closes this on the declaring side: a `float frequency` would hold 107.1 MHz only to about 8 Hz and forward a `float`, so it fails to compile.
  A publisher that puts a `double` under `gr:sample_rate` is still narrowed at the receiver:
  publish through `gr::tag::SAMPLE_RATE(value)`, which accepts only the key's type; `gr::tag::put(map, ...)` and a hand-built `property_map` do not check.
  A value that cannot be converted is dropped with a `gr::log` warning on every drop, and the forwarded tag carries the value the block holds.
- **A value outside `Limits` is refused, but still announced.**
  The block keeps its old value and reports the refusal on stderr (`Failed to validate field …`);
  the refused value is nevertheless announced downstream as the setting's tag.
- **String settings are `std::pmr::string`.**
  `gr::tag::SIGNAL_NAME(...)` takes a `std::string` only, so publish a `gr::SignalName` setting as `gr::tag::SIGNAL_NAME(std::string(signal_name.value))`.
- **One setting, one key.**
  `sample_rate` and `gr:sample_rate` are two keys naming one setting; the later one in the map wins unless it equals the block's current value.
  Never carry both.
- **Conflicting tags at one sample.**
  Tags at the same sample on two inputs are applied in port order, so the last port's value wins; forwarding passes an identical tag once and conflicting ones both.
- **`reset_default` overrides what it arrives with.**
  `reset_default` or `store_default` set to `true`, in a tag or a settings message, resets to or stores the defaults without being a setting; values staged together with a reset are dropped, a store runs after them.
- **processOne publishes once per work call.**
  Each `this->publishTag()` ends the work call after the current sample; several publishes on one sample merge into one tag (last key wins), and the offset argument is ignored.
- **A work call that publishes no samples drops its tags.**
  Every tag published in a call that produces no output samples is discarded; input tags not consumed come back with the next call.
- **Offsets are not checked in Release.**
  Ascending order per output is checked only in a Debug build (it aborts); the offset range is checked in neither.
- **Tensor axes travel unchanged.**
  Default forwarding passes the `axis_*` keys as they are; a block that changes a tensor's dimensions must drop or rewrite them itself (`NoTagPropagation` plus `forwardedTag`).
- **Keep tags minimal.**
  Every block checks every root key of every tag against its settings, a canonical key by its compact id, any other key by name; a nested map under `gr:user_data` is one key however much it holds.
  The block's active settings are read only when a key names one of its settings.
  Every publish copies the blob into the output's tag buffer;
  in addition, the tag is rebuilt (a second copy, into a `property_map`) where a key names a setting with a value other than the block's, where a resampler rescales a rate key, under `MergeTagPropagation`, and for delayed and stride-skipped tags.
- **A tag per sample costs.**
  The default policy ends a chunk at each tag, so a tag on every sample means work calls of one sample;
  without tags, the processOne dispatch loop costs one predicted branch per sample.
  `bm_TagForwarding` (instructions per sample, four pass-through blocks, GCC 15 Release, 2026-10-19): 31.5 without tags, 1 400 with a tag every 16 samples.
  Reading through `ValueMapView` keeps the read path allocation-free.
- **A full tag buffer drops tags.**
  The drop is logged as `output tag dropped` at the 1st, 2nd, 4th, 8th, ... drop in the process; a dropped `gr:end_of_stream` leaves the receiver running.

---

## Standard tag keys — `kDefaultTags`

A canonical key carries the `gr:` prefix in a tag map (`gr::tag::SAMPLE_RATE.key()`, `"gr:sample_rate"`);
its short form (`gr::tag::SAMPLE_RATE.shortKey()`, `"sample_rate"`) is the name of the block setting it updates.
`kDefaultTags` lists the 37 short forms; each has a compact id (`gr::pmt::keys::kCanonical`, `ValueMap.hpp`), and a `static_assert` ties name, unit and wire type of both lists together.
The meaning is the tag's `description()`, also the `Doc<>` of its alias.

| key                  | `gr::tag::`          | alias `gr::`      | type                               | unit | meaning                                                                       |
| -------------------- | -------------------- | ----------------- | ---------------------------------- | ---- | ----------------------------------------------------------------------------- |
| `sample_rate`        | `SAMPLE_RATE`        | `SampleRate`      | `float`                            | Hz   | sampling rate fₛ: samples per second on this edge                             |
| `frame_rate`         | `FRAME_RATE`         | `FrameRate`       | `float`                            | Hz   | frames (items) per second on an edge, where it differs from the sampling rate |
| `frequency`          | `FREQUENCY`          | `Frequency`       | `double`                           | Hz   | signal or band centre frequency                                               |
| `bandwidth`          | `BANDWIDTH`          |                   | `double`                           | Hz   | occupied bandwidth of the signal                                              |
| `frequency_min`      | `FREQUENCY_MIN`      |                   | `double`                           | Hz   | lower edge of the occupied band                                               |
| `frequency_max`      | `FREQUENCY_MAX`      |                   | `double`                           | Hz   | upper edge of the occupied band                                               |
| `est_sample_rate`    | `EST_SAMPLE_RATE`    | `EstSampleRate`   | `float`                            | Hz   | measured sampling rate                                                        |
| `est_frame_rate`     | `EST_FRAME_RATE`     |                   | `float`                            | Hz   | measured frame rate                                                           |
| `signal_name`        | `SIGNAL_NAME`        | `SignalName`      | `std::string` → `std::pmr::string` |      | signal name                                                                   |
| `num_channels`       | `NUM_CHANNELS`       | `NumChannels`     | `gr::Size_t`                       |      | number of interleaved channels                                                |
| `signal_quantity`    | `SIGNAL_QUANTITY`    | `SignalQuantity`  | `std::string` → `std::pmr::string` |      | physical quantity of the signal, e.g. voltage (ISO 80000-1)                   |
| `signal_unit`        | `SIGNAL_UNIT`        | `SignalUnit`      | `std::string` → `std::pmr::string` |      | SI unit of the signal, e.g. V                                                 |
| `signal_min`         | `SIGNAL_MIN`         | `SignalMin`       | `float`                            | a.u. | lower physical limit of the signal, e.g. the DAQ range                        |
| `signal_max`         | `SIGNAL_MAX`         | `SignalMax`       | `float`                            | a.u. | upper physical limit of the signal, e.g. the DAQ range                        |
| `n_dropped_samples`  | `N_DROPPED_SAMPLES`  |                   | `gr::Size_t`                       |      | number of dropped samples                                                     |
| `rx_overflow`        | `RX_OVERFLOW`        |                   | `bool`                             |      | receiver overflow indicator                                                   |
| `trigger_name`       | `TRIGGER_NAME`       | `TriggerName`     | `std::string` → `std::pmr::string` |      | trigger name                                                                  |
| `trigger_time`       | `TRIGGER_TIME`       | `TriggerTime`     | `std::uint64_t`                    | ns   | UTC time-stamp of the trigger                                                 |
| `trigger_time_error` | `TRIGGER_TIME_ERROR` |                   | `std::uint64_t`                    | ns   | 1σ uncertainty of trigger_time                                                |
| `trigger_offset`     | `TRIGGER_OFFSET`     | `TriggerOffset`   | `float`                            | s    | sample delay relative to the trigger, e.g. compensating analogue group delays |
| `trigger_meta_info`  | `TRIGGER_META_INFO`  | `TriggerMetaInfo` | `property_map`                     |      | map with additional trigger information                                       |
| `trigger_stop_time`  | `TRIGGER_STOP_TIME`  |                   | `std::uint64_t`                    | ns   | absolute UTC end of the window                                                |
| `trigger_pre`        | `TRIGGER_PRE`        | `TriggerPre`      | `std::uint64_t`                    | ns   | window extension before trigger_time                                          |
| `trigger_post`       | `TRIGGER_POST`       | `TriggerPost`     | `std::uint64_t`                    | ns   | window extension after trigger_stop_time                                      |
| `acquisition_flags`  | `ACQUISITION_FLAGS`  |                   | `property_map`                     |      | flag name → bool, inside trigger_meta_info; empty = clean                     |
| `user_data`          | `USER_DATA`          |                   | `property_map`                     |      | opaque user-provided metadata, not interpreted as settings                    |
| `context`            | `CONTEXT`            | `Context`         | `std::string` → `std::pmr::string` |      | multiplexing key selecting a settings context                                 |
| `ctx_time`           | `CONTEXT_TIME`       |                   | `std::uint64_t`                    | ns   | UTC time at which the context applies                                         |
| `local_time`         | `LOCAL_TIME`         |                   | `std::uint64_t`                    | ns   | host UTC time-stamp inside trigger_meta_info, for metering propagation delays |
| `reset_default`      | `RESET_DEFAULTS`     |                   | `bool`                             |      | reset block state to the stored default                                       |
| `store_default`      | `STORE_DEFAULTS`     |                   | `bool`                             |      | store block settings as the default                                           |
| `end_of_stream`      | `END_OF_STREAM`      |                   | `bool`                             |      | end of stream: the receiver changes to DONE                                   |
| `axis_origin`        | `AXIS_ORIGIN`        |                   | `Tensor<double>` [D]               |      | per tensor dimension: the coordinate of cell 0, in its axis_unit              |
| `axis_step`          | `AXIS_STEP`          |                   | `Tensor<double>` [D]               |      | per tensor dimension: the coordinate step from cell to cell                   |
| `axis_periodic`      | `AXIS_PERIODIC`      |                   | `Tensor<bool>` [D]                 |      | per tensor dimension: its two ends are neighbours (a full turn)               |
| `axis_unit`          | `AXIS_UNIT`          | `AxisUnit`        | `Tensor<pmt::Value>` [D]           |      | per tensor dimension: the unit of its coordinates, e.g. rad or m              |
| `axis_name`          | `AXIS_NAME`          | `AxisName`        | `Tensor<pmt::Value>` [D]           |      | per tensor dimension: what it spans, e.g. azimuth or x                        |

`std::string` → `std::pmr::string`: a `std::string` on the wire, a `std::pmr::string` as a setting.
The `property_map` and `Tensor` keys travel as an untyped `gr::pmt::Value`, read through the `Value` API.
`property_map` is `gr::pmt::ValueMap`; `gr::Size_t` is `std::uint32_t`.

### Tensor axes

The `axis_*` keys give a tensor's dimensions in physical units, one entry per dimension: cell `i` of dimension `d` lies at `axis_origin[d] + i * axis_step[d]`.
A block that publishes a map over a scan grid (directions in rad, positions in m) sets them.

```cpp
const gr::property_map axes{{gr::tag::AXIS_ORIGIN.key(), gr::Tensor<double>{-std::numbers::pi}},
                            {gr::tag::AXIS_STEP.key(), gr::Tensor<double>{2.0 * std::numbers::pi / 360.0}},
                            {gr::tag::AXIS_PERIODIC.key(), gr::Tensor<bool>(gr::data_from, {true})},
                            {gr::tag::AXIS_UNIT.key(), gr::Tensor<gr::pmt::Value>{gr::pmt::Value{"rad"}}},
                            {gr::tag::AXIS_NAME.key(), gr::Tensor<gr::pmt::Value>{gr::pmt::Value{"azimuth"}}}};
outSpan.publishTag(axes, 0UZ); // a [360] azimuth map over a full turn
```
