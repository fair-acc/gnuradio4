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

**The condition is data, not code.**
A filter is a string compiled once in `settingsChanged` into a trivially copyable `MatchState`, so the same match runs on the host and inside a SYCL kernel.
Nothing in the decision path allocates, throws, or needs a runtime lambda.

## Two carriers, one meaning

| carrier             | what it is                                       | when                                                 |
| ------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **tag** on a sample | in-band, sample-exact, ordered with the data     | the condition was found in the samples               |
| **event** on a bus  | `EventPortIn`/`Out`: async `property_map` stream | the condition comes from elsewhere, or is a decision |

An event port is a **bus**: many-to-many, ordered by claim rather than by time.
Blocks drain it on sight and work on their own copy.
An injected event applies at the **first sample of the current work call** — span-granular by construction, so no marble equality is claimed for it.
Canonical keys only: `trigger_name`, `trigger_time` (whole UTC ns), `trigger_offset` (sub-ns remainder), `trigger_time_error`, `context`, `trigger_meta_info`; a window adds `trigger_stop_time` (UTC ns), `trigger_pre` and `trigger_post` (ns), and `acquisition_flags` inside `trigger_meta_info`.

Code outside `work()` -- a UI control's `draw()`, an I/O callback, a timer -- emits with `gr::emit(evtOut, event)` (or `gr::emit<cmd>(msgOut, data, endpoint, serviceName)` for a message) from any thread, one writer per call, then calls `notifyProgress()` so a parked scheduler wakes at once.
A full or unconnected event port refuses the event instead of blocking.
A UI "tag" is an event; a `MessageToTag` turns it into a stream tag.

## Time

`trigger_time` is UTC in whole nanoseconds everywhere.
A source converts before it emits -- PTP, White Rabbit, GPS and TAI clocks, leap seconds included -- so blocks never mix time scales.
An event raised by a UI or an I/O callback is stamped with the host's UTC and SHOULD carry `trigger_time_error`, since a host clock is not a timing receiver.
Within a stream, the most recent dated tag anchors the count of samples at the stream's `sample_rate`; a recorder places a decision only once its samples have arrived, by whatever anchor holds then, so the result does not depend on where the stream is cut.

## One grammar

A **filter** (`open_filter`, `filter`, ...) names triggers and contexts: `name`, `name/context`, or a list.
A **predicate** (`SampleFilter`, `TakeWhile`, `PredicateSwitch`, the expression blocks) is an exprtk expression over `x` (the sample) and, through `gr::algorithm::ExpressionSymbols`, the block's own settings by their snake_case name and the fields of the tags seen so far by their short key (`sample_rate` reads `gr:sample_rate`).
These are the names a later rule language binds to.

## Pedigree

- **Reactive streams** — Rx (Meijer, Microsoft, 2009) → [ReactiveX](https://reactivex.io) → [RxCpp](https://github.com/ReactiveX/RxCpp).
  Diagrams and normative behaviour: [RxMarbles](https://rxmarbles.com), data from [staltz/rxmarbles](https://github.com/staltz/rxmarbles) (`src/data/*-examples.js`).
- **Composable C++ pipelines** — Niebler's [range-v3](https://github.com/ericniebler/range-v3) → P0896 → C++20 `<ranges>`.
  The trigger family is the same idea over a _timed_ stream.
- **Instrumentation vocabulary** — IEEE Std 1057-2017 (digitizing waveform recorders: trigger point, pre/post trigger), IEEE Std 181-2011 (pulse terminology: the conditions `ValueTrigger` implements), IEEE Std 1588-2019 (PTP: where `trigger_time` comes from).
- **Scope of the catalogue** — fair-acc/gnuradio4 issue 161.

## The catalogue

Each block's header carries its own marble diagram.
`·` marks no RxMarbles equivalent.

**Bridges and test sources**

| block             | Rx  | emits                                                       |
| ----------------- | --- | ----------------------------------------------------------- |
| `TagToEvent<T>`   | ·   | an event per accepted tag or matching key, stream untouched |
| `MessageToTag<T>` | ·   | a stream tag per event that asks for one                    |

**Triggers — what in the samples is worth a decision**

| block                       | Rx  | emits                                                                    |
| --------------------------- | --- | ------------------------------------------------------------------------ |
| `SchmittTrigger<T, Method>` | ·   | an edge with hysteresis, interpolated; as a tag and on `evtOut`          |
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
| `Demux<T>`                | ·                                                                                            | the stream routed by its context tag, or `select_key`  |
| `PredicateSwitch<T>`      | ·                                                                                            | each sample to the first predicate that holds          |
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
| `MultiChannelRecorder<T>`        | ·                                                                                                                   | a window in time, from every channel, as a DataSet |
| `gr::blocks::math::Histogram<T>` | ·                                                                                                                   | the distribution and its figures                   |

**Algorithms, no ports** — `TimeBase` (when a sample happened), `EventStore` (a drained bus), `WindowCollector` (the samples a window needs, history included), `gr::algorithm::HistogramAccumulator`, `gr::trigger::SchmittTrigger` (the hysteretic comparator every edge trigger uses; device-capable), `gr::algorithm::ExpressionSymbols` (the identifiers an exprtk expression may use).

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
auto& recorder  = graph.emplaceBlock<MultiChannelRecorder<float>>({{"n_inputs", 4U}, {"trigger_pre", 32'000U}, {"trigger_post", 96'000U}}); // ns
graph.connect(edge0, gr::PortDefinition{"evtOut"}, together, gr::PortDefinition{"evtIn"});   // and ch1..ch3
graph.connect(together, gr::PortDefinition{"evtOut"}, recorder, gr::PortDefinition{"evtIn"});
graph.connect(adc0, gr::PortDefinition{"out"}, recorder, gr::PortDefinition{"in#0"});        // and in#1..in#3
```

**An event recorder**: an edge on one channel freezes a window of another (use case #2, `qa_UseCase2_MultiRateReplay.cpp`):

```cpp
auto& edge     = graph.emplaceBlock<SchmittTrigger<float>>({{"threshold", 1.f}, {"offset", 9.f}, {"sample_rate", 1e3f}});
auto& recorder = graph.emplaceBlock<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"trigger_pre", 1'000'000U}, {"trigger_post", 1'000'000U}});
graph.connect(edge, gr::PortDefinition{"evtOut"}, recorder, gr::PortDefinition{"evtIn"});
graph.connect(otherAdc, gr::PortDefinition{"out"}, recorder, gr::PortDefinition{"in#0"});
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

- **Durations are samples xor seconds.**
  `n_*_samples` and `*_seconds`/`timeout`; zero disables either, seconds need a known rate (the nominal `sample_rate`, from the setting or a tag), and whichever limit is met first expires.
- **Zero is never "unset" for a count.**
  `take(0)` emits nothing, `elementAt(0)` is the first sample.
  Where a bound is optional it is spelled optional, and a count that cannot mean zero refuses it and keeps the previous value.
- **Nothing waits on a wall clock.**
  Quiet periods, inhibits, holdoffs and cadences count the stream's own samples; event timeouts use the times the events carry.
  A dated event that matches nothing is a heartbeat: it advances event time and closes what it has passed in `Coincidence` and `Sequence`, so a group or window nothing else would close needs only a periodic event — a `ClockSource`'s timing tags through `TagToEvent`, say.
  A group still open when the run ends is counted in `n_pending`, not judged.
  `EventBuilder` judges an incomplete event when a later fragment passes its `timeout`, or, keyed by `event_id`, when `max_pending` is reached.
- **Bounded state that runs out stops claiming to know**, counts it, and emits one coalesced error event — `Distinct` forwards everything, `DelayWhen` reverts to arrival order, `SequenceEqual` withholds its verdict, `Repeat` passes the segment through once.
- **Every block reflects what it did**: `n_passed`, `n_suppressed`, `n_refused`, `n_dropped`, `n_overflow`, `n_events_dropped`, `is_open`, … A deliberate loss is always visible as a counter.
- **An event output claims several slots per work call** (`streamSlotsPerPublish`); the framework default of one silently truncates a block with two things to say.
- **`bool` is `std::uint8_t`** on a stream: GR4 cannot instantiate a `bool` ring.
  Non-zero is true.
- **A block that needs a window declares it** (`in.min_samples`), so the scheduler cannot cut inside it.

## Device placement

These run on the device their `compute_domain` names and answer as on the host, which `qa_TriggerDevice` checks on every served domain: `SampleAndHold`, `SchmittTrigger`, `ValueTrigger`, `SerialPatternTrigger`, `SetupHoldTrigger`, `TriggerWatchdog`, `Gate`, `TakeN`, `SkipN`, `TakeWhile`, `SkipWhile`, `ElementAt`, `TakeLast`, `SkipLast`, `SampleFilter`, `Scan`, `Pairwise`, `Distinct`, `DistinctUntilChanged`, `Debounce`, `Throttle`, `DelayWhen`, `Repeat`, `SequenceEqual`, `Coincidence`, `EventFilter`, `Count` and `TimeInterval`.
An event stream between two of them on one device stays in device memory, with no host hop, as long as no host block reads or writes it.
A kernel cannot log, so their warnings and counters are host-path figures.
State a kernel keeps has fixed capacity -- `slew_samples`, a serial pattern, a tail's `n` (64), `hold_samples` (63), `Distinct`'s `max_values` (1024), `DelayWhen`'s `max_pending` (1024), `Repeat`'s `capacity` (4096), `Coincidence`'s conditions (8) and open contributions (64), `TimeInterval`'s waiting results (256) -- and a larger setting is clamped with a warning.
Events are written in place: a call stops at the first one its event output has no slot for, so a full output holds the stream back instead of losing events, while an unconnected one holds nothing back.
A block whose present settings need the host (an ExprTk `expression`, an `EventFilter` rename, a `Coincidence` condition with a context or a range, since a kernel matches trigger names by hash) refuses a device and says why.

The rest stay on the host: `PatternTrigger`, `Demux`, `Mux` and `PredicateSwitch` have dynamic port collections (and `PredicateSwitch` ExprTk); `TagToEvent` and `MessageToTag` carry whole tags, which can exceed the 1 KiB a device event slot holds; `Accumulate`, `Sequence` and `EventBuilder` keep growing queues; `MultiChannelRecorder` and the `Buffer*` blocks emit owning `DataSet`s.

## Testing

`qa_<Block>.cpp` or `qa_<Group>.cpp`, Boost.UT.
`test/Marble.hpp`'s `MarbleSource/Sink<T>` play and spell back a marble script, terminator included.
Every block's test **draws what it did** — `gr::testing::MarbleDiagram` prints rows on a shared time axis in RxMarbles' style, `TriggerTest.hpp` shows what survived where it came from, and `ImChart` covers windows and distributions.

`TriggerTest.hpp` holds twenty acceptance configurations to **the same answer however the stream is cut**, over eight tag placements (first sample, last, consecutive, two on one sample, one the filter ignores, none, a trigger with no time, a single sample), plus unconnected ports and a settings change between runs.

Device coverage runs from `main()` through `"…"_domain_test = body | kAllDomains`, sweeping `host`, `host:sycl` and `gpu:sycl` wherever served.
AdaptiveCpp does not run namespace-scope static initialisers in a translation unit holding a kernel, so a `boost::ut` suite object there never registers — hence `main()`.

## Out of reach, and why

`mergeMap`, `concatMap`, `switchMap` in their selector forms instantiate an inner stream per item: runtime subgraph creation.
The `window*` family emits `Observable<Observable<T>>`, for which there is no GR4 type — `buffer*` is the substitute and the `window*` names stay unimplemented rather than aliased.
`race` cancels its losers, and there is no upstream cancellation.
Rx's `repeat` re-subscribes a completed source; `Repeat` repeats a segment instead.
Resilience4j (`CircuitBreaker`, `RateLimiter`, `Retry`, `Bulkhead`, `Cache`) is policy over a stream, not triggers: its own checkpoint.

Two framework gaps the family works around, written up as issues in the branch's review §10: a block whose inputs are all asynchronous never reaches its epilogue, and the end of a stream is not observable from inside processing.
