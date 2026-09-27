# Trigger and event blocks

`gr::blocks::trigger` decides **when** a stream may pass, not what it carries.

```cpp
auto& source = fixture.emplace<MarbleSource<float>>({{"script", "a b T:c d e f |"}, {"sample_values", values}, {"sample_tags", tags}});
auto& gate   = fixture.emplace<Gate<float>>({{"mode", "once"}, {"open_filter", "start"}, {"n_open", 3U}});
auto& sink   = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
expect(fixture.connect<"out", "in">(source, gate).has_value());
expect(fixture.connect<"out", "in">(gate, sink).has_value());
expect(fixture.run().has_value());
expect(eq(sink.script(), "T:c d e |")); // three samples from the trigger, then suppressed, then end of stream
```

```text
in   ─a──b──T:c──d──e──f─▶
     ┌────────────────┐
     │ Gate(once, 3)  │
     └────────────────┘
out  ───────T:c──d──e────│
```

**The condition is data, not code.** A filter is a string compiled once in `settingsChanged` into a trivially copyable
`MatchState`, so the same match runs on the host and inside a SYCL kernel. Nothing in the decision path allocates,
throws, or needs a runtime lambda.

## Two carriers, one meaning

| carrier             | what it is                                       | when                                                 |
| ------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **tag** on a sample | in-band, sample-exact, ordered with the data     | the condition was found in the samples               |
| **event** on a bus  | `EventPortIn`/`Out`: async `property_map` stream | the condition comes from elsewhere, or is a decision |

An event port is a **bus**: many-to-many, ordered by claim rather than by time. Blocks drain it on sight and work on
their own copy. An injected event applies at the **first sample of the current work call** — span-granular by
construction, so no marble equality is claimed for it. Canonical keys only: `trigger_name`, `trigger_time` (whole UTC
ns), `trigger_offset` (sub-ns remainder), `trigger_time_error`, `context`, `trigger_meta_info`.

## Pedigree

- **Reactive streams** — Rx (Meijer, Microsoft, 2009) → [ReactiveX](https://reactivex.io) →
  [RxCpp](https://github.com/ReactiveX/RxCpp). Diagrams and normative behaviour: [RxMarbles](https://rxmarbles.com),
  data from [staltz/rxmarbles](https://github.com/staltz/rxmarbles) (`src/data/*-examples.js`).
- **Composable C++ pipelines** — Niebler's [range-v3](https://github.com/ericniebler/range-v3) → P0896 →
  C++20 `<ranges>`. The trigger family is the same idea over a _timed_ stream.
- **Instrumentation vocabulary** — IEEE Std 1057-2017 (digitizing waveform recorders: trigger point, pre/post
  trigger), IEEE Std 181-2011 (pulse terminology: the conditions `ValueTrigger` implements), IEEE Std 1588-2019 (PTP:
  where `trigger_time` comes from).
- **Scope of the catalogue** — fair-acc/gnuradio4 issue 161.

## The catalogue

Each block's header carries its own marble diagram. `·` marks no RxMarbles equivalent.

**Bridges and test sources**

| block                  | Rx  | emits                                                    |
| ---------------------- | --- | -------------------------------------------------------- |
| `TagToMessage<T>`      | ·   | an event per accepted tag, stream untouched              |
| `MessageToTag<T>`      | ·   | a stream tag per event that asks for one                 |
| `MarbleSource/Sink<T>` | ·   | plays / spells back a marble script, terminator included |

**Triggers — what in the samples is worth a decision**

| block                       | Rx  | emits                                                                    |
| --------------------------- | --- | ------------------------------------------------------------------------ |
| `SchmittTrigger<T, Method>` | ·   | an edge with hysteresis, interpolated to a fraction of a sample          |
| `ValueTrigger<T, Cond>`     | ·   | `level`, `window`, `pulse_width`, `runt`, `slew`, `slew_rate`, `dropout` |
| `PatternTrigger<T>`         | ·   | a pattern of `1/0/X` across N channels at one instant                    |
| `SerialPatternTrigger<T>`   | ·   | the same pattern across _time_, sampled on a clock's edges               |
| `SetupHoldTrigger<T>`       | ·   | a data change too close to a clock edge — the fault, not the signal      |

**Event algebra**

| block             | Rx                                      | emits                                                                       |
| ----------------- | --------------------------------------- | --------------------------------------------------------------------------- |
| `EventFilter`     | [filter](https://rxmarbles.com/#filter) | the events a filter accepts, optionally renamed                             |
| `Count`           | [count](https://rxmarbles.com/#count)   | every nth, or once at the nth, plus `rate`                                  |
| `Sequence`        | ·                                       | a trigger inside an armed window; a window that expired                     |
| `Coincidence`     | ·                                       | N conditions within ±Δt: `any`/`all`/`at_least`/`exactly`/`exclusive`, veto |
| `TimeInterval<T>` | ·                                       | Δt as a sample: `to_reference`/`nearest`/`paired`/`to_previous`/`tie`       |
| `EventBuilder<T>` | [zip](https://rxmarbles.com/#zip)       | fragments matched by timestamp or id, as one `DataSet`                      |

**Gating, routing, rate**

| block                     | Rx                                                                                           | emits                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `Gate<T>`                 | [takeUntil](https://rxmarbles.com/#takeUntil), [skipUntil](https://rxmarbles.com/#skipUntil) | the stream while a trigger says it may — six modes     |
| `TakeN<T>`                | [take](https://rxmarbles.com/#take)                                                          | n samples from the trigger                             |
| `SkipN<T>`                | [skip](https://rxmarbles.com/#skip)                                                          | all but n samples from the trigger                     |
| `SampleFilter<T>`         | [filter](https://rxmarbles.com/#filter)                                                      | the samples that satisfy a comparison or an expression |
| `TakeWhile<T>`            | [takeWhile](https://rxmarbles.com/#takeWhile)                                                | until the first failure, then nothing                  |
| `SkipWhile<T>`            | [skipWhile](https://rxmarbles.com/#skipWhile)                                                | from the first failure, then everything                |
| `ElementAt<T>`            | [elementAt](https://rxmarbles.com/#elementAt)                                                | item n, of the stream or of every segment              |
| `SampleAndHold<T>`        | [sample](https://rxmarbles.com/#sample)                                                      | the last matched sample, held                          |
| `Distinct<T>`             | [distinct](https://rxmarbles.com/#distinct)                                                  | each value once, within a bounded memory               |
| `DistinctUntilChanged<T>` | [distinctUntilChanged](https://rxmarbles.com/#distinctUntilChanged)                          | a sample only where it changed                         |
| `Debounce<T>`             | [debounce](https://rxmarbles.com/#debounce)                                                  | one trigger per burst, the last of it                  |
| `Throttle<T>`             | [throttle](https://rxmarbles.com/#throttle)                                                  | the first of a burst, then a dead time                 |
| `DelayWhen<T>`            | [delayWhen](https://rxmarbles.com/#delayWhen)                                                | each item after its own delay                          |
| `Repeat<T>`               | [repeat](https://rxmarbles.com/#repeat)                                                      | a captured segment again, n times                      |
| `TakeLast<T>`             | [takeLast](https://rxmarbles.com/#takeLast)                                                  | the last n samples of a segment                        |
| `SkipLast<T>`             | [skipLast](https://rxmarbles.com/#skipLast)                                                  | a segment without its last n samples                   |
| `Pairwise<T>`             | [pairwise](https://rxmarbles.com/#pairwise)                                                  | each sample beside its predecessor, on two outputs     |
| `SequenceEqual<T>`        | [sequenceEqual](https://rxmarbles.com/#sequenceEqual)                                        | one verdict per segment                                |
| `Demux<T>`                | ·                                                                                            | the stream routed by its context tag                   |
| `Mux<T>`                  | [merge](https://rxmarbles.com/#merge)                                                        | the states put back into one stream, by context        |
| `TriggerWatchdog<T>`      | ·                                                                                            | an outage and a recovery; can tell a gate to close     |

**Windows and reductions**

| block                            | Rx                                                                                                                  | emits                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `BufferCount<T>`                 | [bufferCount](https://rxmarbles.com/#bufferCount)                                                                   | `Tensor<T>` per n samples, overlap via `n_skip`    |
| `BufferTime<T>`                  | [bufferTime](https://rxmarbles.com/#bufferTime)                                                                     | `DataSet<T>` per duration                          |
| `BufferToggle<T>`                | [bufferToggle](https://rxmarbles.com/#bufferToggle)                                                                 | `DataSet<T>` a trigger opens                       |
| `BufferWhen<T>`                  | [bufferWhen](https://rxmarbles.com/#bufferWhen)                                                                     | `DataSet<T>` between triggers                      |
| `Accumulate<T, Accumulation>`    | [last](https://rxmarbles.com/#last), [reduce](https://rxmarbles.com/#reduce), [every](https://rxmarbles.com/#every) | one value per segment                              |
| `Scan<T>`                        | [scan](https://rxmarbles.com/#scan)                                                                                 | the running value, on every sample                 |
| `MultiChannelRecorder<T>`        | ·                                                                                                                   | the same window from every channel on one decision |
| `gr::blocks::math::Histogram<T>` | ·                                                                                                                   | the distribution and its figures                   |

**Algorithms, no ports** — `TimeBase` (when a sample happened), `EventStore` (a drained bus), `SegmentCollector` /
`WindowCollector` (the samples a window needs), `gr::algorithm::HistogramAccumulator`,
`gr::algorithm::SchmittTrigger` (the hysteretic comparator every edge trigger uses; device-capable).

## Worked examples

**Gate a stream on an external timing event** — no tags in the stream at all:

```cpp
auto& gate = graph.emplaceBlock<Gate<float>>({{"mode", "toggle"}, {"open_filter", "CMD_BP_START"}, {"close_filter", "CMD_BP_END"}});
graph.connect(timingReceiver, gr::PortDefinition{"evtOut"}, gate, gr::PortDefinition{"evtIn"});
graph.connect<"out", "in">(adc, gate);
```

**Four channels, one coincidence, one snapshot** (use case #1, `qa_UseCase1_MultiChannelTrigger.cpp`):

```cpp
auto& edge0     = graph.emplaceBlock<SchmittTrigger<float>>({{"threshold", 0.5f}, {"offset", 2.5f}});
auto& together  = graph.emplaceBlock<Coincidence>({{"filters", std::vector<std::string>{"ch0", "ch1", "ch2", "ch3"}},
                                                   {"logic", "all"}, {"window", 50e-6}, {"holdoff", 1e-3}});
auto& recorder  = graph.emplaceBlock<MultiChannelRecorder<float>>({{"n_inputs", 4U}, {"n_pre", 32U}, {"n_post", 96U}});
graph.connect(edge0, gr::PortDefinition{"evtOut"}, together, gr::PortDefinition{"evtIn"});   // and ch1..ch3
graph.connect(together, gr::PortDefinition{"evtOut"}, recorder, gr::PortDefinition{"evtIn"});
graph.connect(adc0, gr::PortDefinition{"out"}, recorder, gr::PortDefinition{"in#0"});        // and in#1..in#3
```

**Jitter, end to end** — timing becomes a signal, then a distribution (`qa_TimeInterval.cpp`):

```cpp
auto& period = graph.emplaceBlock<TimeInterval<double>>({{"mode", "to_previous"}});
auto& spread = graph.emplaceBlock<gr::blocks::math::Histogram<double>>({{"bin_min", 0.95e-3}, {"bin_max", 1.05e-3}, {"n_bins", 20U}});
graph.connect(tickSource, gr::PortDefinition{"evtOut"}, period, gr::PortDefinition{"evtIn"});
graph.connect<"out", "in">(period, spread);
// spread.mean, spread.stddev, and a DataSet an ImChart draws as bars
```

**Clean up a noisy trigger, then limit its rate**:

```cpp
auto& clean   = graph.emplaceBlock<Debounce<float>>({{"filter", "edge"}, {"n_samples", 50U}});   // one per burst
auto& limited = graph.emplaceBlock<Throttle<float>>({{"filter", "edge"}, {"timeout", 0.1f}, {"sample_rate", 1e6f}});
```

**A window per machine cycle, reduced to one figure**:

```cpp
auto& perCycle = graph.emplaceBlock<BufferWhen<float>>({{"filter", "CMD_BP_START"}});             // DataSet per cycle
auto& peak     = graph.emplaceBlock<Accumulate<float, Accumulation::maximum>>({{"segment_filter", "CMD_BP_START"}});
```

## Conventions that hold everywhere

- **Durations are samples xor seconds.** `n_*_samples` and `*_seconds`/`timeout`; zero disables either, seconds need a
  known rate (two `trigger_time` anchors, else `sample_rate`), and whichever limit is met first expires.
- **Zero is never "unset" for a count.** `take(0)` emits nothing, `elementAt(0)` is the first sample. Where a bound is
  optional it is spelled optional, and a count that cannot mean zero refuses it and keeps the previous value.
- **Nothing waits on a wall clock.** Quiet periods, inhibits, holdoffs and cadences count the stream's own samples;
  event timeouts use the times the events carry. `flush_after` (own steady clock, off by default) is the only exception,
  for a group nothing else will close.
- **Bounded state that runs out stops claiming to know**, counts it, and emits one coalesced error event —
  `Distinct` forwards everything, `DelayWhen` reverts to arrival order, `SequenceEqual` withholds its verdict,
  `Repeat` passes the segment through once.
- **Every block reflects what it did**: `n_passed`, `n_suppressed`, `n_refused`, `n_dropped`, `n_overflow`,
  `n_events_dropped`, `is_open`, … A deliberate loss is always visible as a counter.
- **An event output claims several slots per work call** (`streamSlotsPerPublish`); the framework default of one
  silently truncates a block with two things to say.
- **`bool` is `std::uint8_t`** on a stream: GR4 cannot instantiate a `bool` ring. Non-zero is true.
- **A block that needs a window declares it** (`in.min_samples`), so the scheduler cannot cut inside it.

## Testing

`qa_<Block>.cpp` or `qa_<Group>.cpp`, Boost.UT. Every block's test **draws what it did** — `gr::testing::MarbleDiagram` prints rows on a
shared time axis in RxMarbles' style, `TriggerTest.hpp` shows what survived where it came from, and `ImChart` covers
windows and distributions.

`TriggerTest.hpp` holds twenty acceptance configurations to **the same answer however the stream is cut**, over eight tag placements
(first sample, last, consecutive, two on one sample, one the filter ignores, none, a trigger with no time, a single
sample), plus unconnected ports and a settings change between runs.

Device coverage runs from `main()` through `"…"_domain_test = body | kAllDomains`, sweeping `host`, `host:sycl` and
`gpu:sycl` wherever served. AdaptiveCpp does not run namespace-scope static initialisers in a translation unit holding a
kernel, so a `boost::ut` suite object there never registers — hence `main()`.

## Out of reach, and why

`mergeMap`, `concatMap`, `switchMap` in their selector forms instantiate an inner stream per item: runtime subgraph
creation. The `window*` family emits `Observable<Observable<T>>`, for which there is no GR4 type — `buffer*` is the
substitute and the `window*` names stay unimplemented rather than aliased. `race` cancels its losers, and there is no
upstream cancellation. Rx's `repeat` re-subscribes a completed source; `Repeat` repeats a segment instead. Resilience4j
(`CircuitBreaker`, `RateLimiter`, `Retry`, `Bulkhead`, `Cache`) is policy over a stream, not triggers: its own checkpoint.

Two framework gaps the family works around, written up as issues in the branch's review §10: a block whose inputs are
all asynchronous never reaches its epilogue, and the end of a stream is not observable from inside processing.
