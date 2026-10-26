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

The contract (the trigger map, tags, events and messages, time, the filter grammar, a composed example) is in [Triggers](../../docs/USER_API_Triggers.md).

## The catalogue

Each block's header carries its own marble diagram.
`·` marks no RxMarbles equivalent.

**Bridges**

| block           | Rx  | emits                                                                                        |
| --------------- | --- | -------------------------------------------------------------------------------------------- |
| `TagToEvent<T>` | ·   | the tags a filter names (all without one) as events, stream untouched; `delay` re-dates them |
| `EventToTag<T>` | ·   | a stream tag per event that asks for one, on the sample its time names; `delay` re-dates it  |

**Triggers — what in the samples is worth a decision**

| block                       | Rx  | emits                                                                                                                                                                           |
| --------------------------- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SchmittTrigger<T, Method>` | ·   | an edge with hysteresis, interpolated; as a tag and on `evtOut`; `trigger_name_rising`/`_falling` name the polarities                                                           |
| `ValueTrigger<T, Cond>`     | ·   | `level` (`LevelTrigger`, the default edge detector), `window`, `pulseWidth`, `runt`, `transition`, `slewRate`, `levelTimeout`, `edgeTimeout`, `interval`, each its own block id |
| `PatternTrigger<T>`         | ·   | a pattern of `1/0/X` across N channels at one instant, `all` or `any` (three-valued)                                                                                            |
| `SerialPatternTrigger<T>`   | ·   | the same pattern across _time_, sampled on a clock's edges                                                                                                                      |
| `SetupHoldTrigger<T>`       | ·   | a data change too close to a clock edge — the fault, not the signal                                                                                                             |

**Event algebra**

| block             | Rx                                      | emits                                                                                             |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `EventFilter`     | [filter](https://rxmarbles.com/#filter) | the events a filter names                                                                         |
| `Count`           | [count](https://rxmarbles.com/#count)   | every nth, once at the nth, or n within an interval; plus `rate`                                  |
| `Then`            | ·                                       | B within [min, max) after A, one per A, and an A that waited in vain; without A, a gap between Bs |
| `While<T>`        | ·                                       | an event, if a line was at the required level at its time                                         |
| `Coincidence`     | ·                                       | N conditions within ±Δt: `any`/`all`/`atLeast`/`exactly`/`exclusive`, veto                        |
| `StateLogic`      | ·                                       | entry and exit of `allOf`/`anyOf`/`notOf` over interval states, three-valued                      |
| `EventStatistics` | ·                                       | running mean, standard deviations, min, max, count, last of an event field (module `measurement`) |
| `TimeInterval<T>` | ·                                       | Δt as a sample: `toReference`/`nearest`/`paired`/`toPrevious`/`tie`                               |
| `EventBuilder<T>` | [zip](https://rxmarbles.com/#zip)       | fragments matched by timestamp or id, as one `DataSet`                                            |

**Gating, routing, rate**

| block                                  | Rx                                                                                           | emits                                                                             |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `Gate<T>`                              | [takeUntil](https://rxmarbles.com/#takeUntil), [skipUntil](https://rxmarbles.com/#skipUntil) | the stream while a trigger says it may — six modes                                |
| `TakeN<T>`                             | [take](https://rxmarbles.com/#take)                                                          | n samples from the trigger                                                        |
| `SkipN<T>`                             | [skip](https://rxmarbles.com/#skip)                                                          | all but n samples from the trigger                                                |
| `SampleFilter<T>`                      | [filter](https://rxmarbles.com/#filter)                                                      | the samples that satisfy a comparison or an expression                            |
| `TakeWhile<T>`                         | [takeWhile](https://rxmarbles.com/#takeWhile)                                                | until the first failure, then nothing                                             |
| `SkipWhile<T>`                         | [skipWhile](https://rxmarbles.com/#skipWhile)                                                | from the first failure, then everything                                           |
| `ElementAt<T>`                         | [elementAt](https://rxmarbles.com/#elementAt)                                                | item n, of the stream or of every segment                                         |
| `SampleAndHold<T>`                     | [sample](https://rxmarbles.com/#sample)                                                      | the last matched sample, held                                                     |
| `Distinct<T>`                          | [distinct](https://rxmarbles.com/#distinct)                                                  | each value once, within a bounded memory                                          |
| `DistinctUntilChanged<T>`              | [distinctUntilChanged](https://rxmarbles.com/#distinctUntilChanged)                          | a sample only where it changed                                                    |
| `Debounce<T>`                          | [debounce](https://rxmarbles.com/#debounce)                                                  | one trigger per burst, the last of it                                             |
| `Throttle<T>`                          | [throttle](https://rxmarbles.com/#throttle)                                                  | the first of a burst, then a dead time                                            |
| `DelayWhen<T>`                         | [delayWhen](https://rxmarbles.com/#delayWhen)                                                | each item after its own delay                                                     |
| `Repeat<T>`                            | [repeat](https://rxmarbles.com/#repeat)                                                      | a captured segment again, n times                                                 |
| `TakeLast<T>`                          | [takeLast](https://rxmarbles.com/#takeLast)                                                  | the last n samples of a segment                                                   |
| `SkipLast<T>`                          | [skipLast](https://rxmarbles.com/#skipLast)                                                  | a segment without its last n samples                                              |
| `Pairwise<T>`                          | [pairwise](https://rxmarbles.com/#pairwise)                                                  | each sample beside its predecessor, on two outputs                                |
| `SequenceEqual<T>`                     | [sequenceEqual](https://rxmarbles.com/#sequenceEqual)                                        | one verdict per segment                                                           |
| `basic::Demux<T, 0, Routing::context>` | ·                                                                                            | the stream routed by its context tag, or `select_key`                             |
| `PredicateSwitch<T>`                   | ·                                                                                            | each sample to the first predicate that holds                                     |
| `basic::Mux<T, 0, Routing::context>`   | [merge](https://rxmarbles.com/#merge)                                                        | the states put back into one stream, by context                                   |
| `TriggerWatchdog<T>`                   | ·                                                                                            | an outage and a recovery, or an absence after an anchor; can tell a gate to close |

**Windows and reductions**

| block                            | Rx                                                                                                                  | emits                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `BufferCount<T>`                 | [bufferCount](https://rxmarbles.com/#bufferCount)                                                                   | `Tensor<T>` per n samples, overlap via `n_skip`                                   |
| `BufferTime<T>`                  | [bufferTime](https://rxmarbles.com/#bufferTime)                                                                     | `DataSet<T>` per duration                                                         |
| `BufferToggle<T>`                | [bufferToggle](https://rxmarbles.com/#bufferToggle)                                                                 | `DataSet<T>` a trigger opens                                                      |
| `BufferWhen<T>`                  | [bufferWhen](https://rxmarbles.com/#bufferWhen)                                                                     | `DataSet<T>` between triggers                                                     |
| `Accumulate<T, Accumulation>`    | [last](https://rxmarbles.com/#last), [reduce](https://rxmarbles.com/#reduce), [every](https://rxmarbles.com/#every) | one value per segment; registered as `Last`, `Reduce`, `Every`, `Any`             |
| `Scan<T>`                        | [scan](https://rxmarbles.com/#scan)                                                                                 | the running value, on every sample                                                |
| `MultiChannelRecorder<T>`        | ·                                                                                                                   | a window `[anchor - n_pre, anchor + n_post)` from every channel, as one `DataSet` |
| `gr::blocks::math::Histogram<T>` | ·                                                                                                                   | the distribution and its figures                                                  |

**Algorithms, no ports** — `TimeBase` (when a sample happened), `EventStore` (a drained bus), `WindowCollector` (the samples a window needs, history included), `gr::algorithm::HistogramAccumulator`, `gr::trigger::SchmittTrigger` (the hysteretic comparator every edge trigger uses; device-capable), `gr::algorithm::ExpressionSymbols` (the identifiers an exprtk expression may use).

Measurements over the gates these blocks open (amplitude, top and base, rise and fall time, widths, duty cycle, overshoot, area, standard deviations, period, frequency, delay and phase) are `blocks/measurement`: `Measure<T, Quantity>` (one block per quantity), `MeasurementCondition` and `EventStatistics`.
Runnable use cases are in `test/qa_UseCases.cpp`: four channels, one coincidence, one snapshot; and an edge on one channel recording another.

## Device placement

These run on the device their `compute_domain` names and answer as on the host (`qa_TriggerDevice`, every served domain):
`SchmittTrigger`, `ValueTrigger`, `SerialPatternTrigger`, `SetupHoldTrigger`, `TriggerWatchdog`, `TagToEvent`,
`EventFilter`, `Count`, `Then`, `While`, `Coincidence`, `StateLogic`, `TimeInterval`, `Accumulate`,
`Gate`, `TakeN`, `SkipN`, `TakeWhile`, `SkipWhile`, `ElementAt`, `TakeLast`, `SkipLast`, `SampleFilter`, `SampleAndHold`, `Scan`, `Pairwise`,
`Distinct`, `DistinctUntilChanged`, `Debounce`, `Throttle`, `DelayWhen`, `Repeat`, `SequenceEqual`,
and in `blocks/measurement` `MeasurementCondition` and `EventStatistics`.

These stay on the host: `PatternTrigger`, `PredicateSwitch`, `EventToTag`, `EventBuilder`, `MultiChannelRecorder`, the `Buffer*` blocks, `Measure`, the context `Mux`/`Demux`, `StreamToDataSet` and `StreamFilter`.

An event stream between two device blocks on one device stays in device memory as long as no host block reads or writes it.
A kernel's state has a fixed capacity, named in its setting's `Doc` or `gr::Limits<>`; a value outside a `gr::Limits<>` is refused.
A block whose present settings need the host refuses a device and says why.
On a device `TagToEvent` never sees a tag larger than an event slot (1 KiB): the dispatcher drops it with a warning.

## Out of reach

- `mergeMap`, `concatMap`, `switchMap` (selector forms): an inner stream per item is runtime subgraph creation.
- `window*`: GR4 has no type for a stream of streams; `buffer*` is the substitute.
- `race`: cancelling the losers needs upstream cancellation, which GR4 does not have.
- `repeat`: re-subscribing a completed source has no GR4 meaning; `Repeat` repeats a segment instead.

## Acknowledgement

The operators and their marble diagrams follow ReactiveX ([reactivex.io](https://reactivex.io), [RxMarbles](https://rxmarbles.com)); composable pipelines follow range-v3 and C++20 `<ranges>`.
The trigger and pulse vocabulary follows IEEE Std 181-2011, IEEE Std 1057-2017 and IEEE Std 1588-2019.
The catalogue's scope is fair-acc/gnuradio4 issue 161.
