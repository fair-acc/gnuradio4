# Triggers

A trigger names a moment in a stream: a threshold crossing, a completed pulse, a timing event, a measurement.
GR4 carries every trigger as one small `property_map` with the same root keys, whichever of the three transports moves it: a tag on a sample, an event on an event port, or a message on a message port.
The transports differ in latency and accuracy, not in meaning: a map built for one is valid on the others.

## TL;DR: the invariants

1. **The root tuple.** `trigger_name` (string), `trigger_time` (`uint64_t`, UTC ns since 1970), `trigger_offset` (`float`, s) are always sent together; `trigger_meta_info` (`property_map`) is optional.
2. **One trigger, one moment.** `trigger_time + trigger_offset` is the moment the trigger names: `trigger_offset` holds the sub-ns residual, or a small correction such as an analogue group delay, where the float's resolution is no issue.
3. **A tag sits on the first sample at or after its moment**, so its event lies less than one sample before the sample it is attached to.
4. **The root holds only canonical keys:** the tuple, `context`, `trigger_time_error`, and the window keys `trigger_stop_time`, `trigger_pre`, `trigger_post`.
   Everything specific to a block (a start tuple, counts, a measured value, a state) lives in `trigger_meta_info`, which is present only when it holds such content.
5. **A delayed decision is dated when it is decided.**
   A pulse is known only at its trailing edge, a timeout only at its deadline: that is its `trigger_time`.
   The moment it is about (the leading edge, the start of an episode) is a full trigger tuple inside `trigger_meta_info`, under the same canonical key names.
   A duration is never stored; it is the root time minus that one.
6. **Operators compare and place by the root time by default.**
   `Then`, `While`, `Coincidence` and `MultiChannelRecorder` take `reference = decision` (the root `trigger_time`) or `reference = start` (the start tuple in `trigger_meta_info`, the decision time where an event carries none).
   Any other consumer acts on the leading edge by composition with the trigger of that edge.
7. **Recoverable errors are messages, not events.**
   A refused setting, an undated or dropped event or an overflow is a warning in the log and a non-fatal `Notify` on the block's `msgOut`, carrying `{"error": reason}`; `evtOut` carries functional events only.
   A value outside a setting's `gr::Limits<>` and a filter that does not compile are refused and reported the same way.
   The `Error` kind is kept for a block that cannot continue.
8. **Degrade, never drop.**
   The root tuple of an event always stands; where `trigger_meta_info` does not fit the event slot (1 KiB), its keys are left out from the end, in priority order, instead of the event being lost.
9. **Every deliberate loss is a counter** (`n_passed`, `n_suppressed`, `n_events_dropped`, ...).
   Counters are lifetime statistics: a settings change never zeroes them, the scheduler's reset does.
10. **Zero is never "unset" for a count:** `ElementAt`'s `n = 0` is the first sample; a count that cannot mean zero (`Gate`'s `n_open`) refuses it.
11. **`bool` is `std::uint8_t` on a stream**, non-zero true: GR4 cannot instantiate a `bool` ring.
12. **An event output claims several slots per work call** (`streamSlotsPerPublish`): the default of one for a multi-producer port truncates a block with two things to say.

```text
  a positive pulse of 3 samples, PulseWidthTrigger, tag at sample 9:
  {
      trigger_name:       "pulse",
      trigger_time:       t(8.4),            // the trailing edge, decided here
      trigger_offset:     <sub-ns residual>,
      trigger_time_error: <1σ, ns>,
      context:            "",
      trigger_meta_info: {
          trigger_name:   "RISING",          // the leading edge, as its own trigger
          trigger_time:   t(5.2),
          trigger_offset: <sub-ns residual>
      }
  }
  width = t(8.4) - t(5.2)
```

## The three transports

| transport | carrier                                        | latency and accuracy                                                              | ordering                                           | combining                                                    | typical use                                            |
| --------- | ---------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| tag       | `publishTag` on a stream sample                | in band: travels with its sample, sample-exact, sub-ns with `trigger_offset`      | the stream's order                                 | through the stream: each block forwards or acts on it        | gating, recording, measuring around the sample itself  |
| event     | `gr::EventPortOut` → `gr::EventPortIn`         | out of band: the same or the next work call; the time is carried, not the arrival | by `trigger_time`, within a block's `max_lateness` | cheap: many producers on one port, operators without samples | trigger logic: coincidences, sequences, counts, vetoes |
| message   | `gr::MsgPortOut` → `gr::MsgPortIn`, a `Notify` | best effort: the next message pass, dropped when the port is full                 | arrival order, none guaranteed                     | in user code                                                 | UI, logging, automation, remote clients                |

A tag is the only transport that is exact without a time base: its position is its time.
An event or a message is exact only through `trigger_time`, so its consumer needs a dated stream to place it on (see [Time](#time)).

### Tags

```cpp
gr::work::Status processBulk(gr::InputSpanLike auto& in, gr::OutputSpanLike auto& out) {
    // ... a decision at sample i, the event 0.3 samples before it:
    gr::property_map trigger{{std::string(gr::tag::kTriggerName.key()), std::string("beam_in")}, {std::string(gr::tag::kTriggerTime.key()), timeNs}, {std::string(gr::tag::kTriggerOffset.key()), residualSeconds}};
    out.publishTag(trigger, i);
    // ...
}
```

Read them through `in.rawTags()` (or `mergedInputTag()` in `processOne`); see [Tag mechanics](USER_API_Tag_mechanics.md).

### Events

An event port carries `property_map_view`s; producers emit without allocating, so a kernel body can emit too.

```cpp
gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}}; // up to 8 events per work call
gr::EventPortIn  evtIn;

// producing: build in place, `nKeys` the root keys the builder writes
std::size_t nEmitted = 0UZ;
if (gr::emit(evtOutSpan, nEmitted, 6U, [&](gr::property_map_view& event) {
        return event.try_emplace(gr::tag::kTriggerName.key(), std::string_view{"beam_in"}) && event.try_emplace(gr::tag::kTriggerTime.key(), timeNs)
            && event.try_emplace(gr::tag::kTriggerOffset.key(), 0.f);
    })) {
    ++nEmitted;
}
evtOutSpan.publish(nEmitted);

// consuming
for (const gr::property_map_view& event : evtInSpan) {
    if (const auto stamp = event.get_if<std::uint64_t>(gr::tag::kTriggerTime.key())) { /* ... */ }
}
std::ignore = evtInSpan.consume(evtInSpan.size());
```

`gr::emit(port, event)` puts an event on a multi-producer port from any thread (a UI's `draw()`, an I/O callback, a timer) and never blocks: a full or unconnected port refuses the event.
Call `notifyProgress()` afterwards so a parked scheduler wakes at once.
A UI "tag" is an event; an `EventToTag` turns it into a stream tag.
Connect event ports like any other: `graph.connect<"evtOut", "evtIn">(detector, operator)`; several producers may share one `evtIn`.

### Messages

```cpp
this->emitMessage("/triggers", trigger); // a Notify on the block's built-in msgOut; dropped when full

void processMessages(gr::MsgPortIn& /*port*/, std::span<const gr::Message> messages) {
    for (const gr::Message& message : messages) {
        if (message.data && message.data->contains("error")) { /* a refusal or loss the block reports, and went on */ }
        else if (message.data) { /* the same trigger map: message.data->... */ }
    }
}
```

No block bridges messages and events: a UI or an automation hook receives triggers in its own message handler.
A refusal made while the initial settings are applied is held and sent when the block starts.
A device kernel cannot send one and counts instead; a counter whose `Doc` says _host path only_ is not advanced inside a kernel.

### Placement

`MultiChannelRecorder`, `StreamToDataSet`, `Measure`, `While`, `Gate` and `EventToTag` place an event on the sample its `trigger_time` names, and count what they cannot place.
`BufferToggle`, `BufferWhen`, `ElementAt`, `TakeWhile`, `SkipWhile` and the context `Mux`/`Demux` are best effort: an event applies at the first sample of the work call that finds it.
`Gate` places on the host; with `evtIn` connected it refuses a device.
`TakeN` and `SkipN` have no event input and count samples.

## Time

`trigger_time` is UTC in whole nanoseconds everywhere.
A source converts before it emits (PTP, White Rabbit, GPS and TAI clocks, leap seconds included) so blocks never mix time scales [3].
An event raised by a UI or an I/O callback is stamped with the host's UTC and should carry `trigger_time_error`, since a host clock is not a timing receiver.

Within a stream, a block's `TimeBase` adopts the first dated tag and counts samples at the stream's `sample_rate` from `trigger_time + trigger_offset`.
It re-anchors only where a later trigger disagrees with that count by a sample or more (a time jump), so a trigger derived from the same time changes nothing; a change of `sample_rate` mid-stream re-anchors nothing.
An undated stream is dated by its source (`SoapySource` and `RTL2832Source` publish `trigger_time`) or by a `ClockSource` with `start_time` set: its first sample carries `trigger_time`, it never reads the host clock, and with `use_internal_thread = false` it runs on a device.
An undated trigger is otherwise passed undated; blocks that cannot place it count it (`n_undated`, `n_unplaceable`), and none dates it by a guess.

Nothing waits on a wall clock: holdoffs, inhibits and cadences count the stream's samples or the events' times.
A dated event that matches nothing is a heartbeat: it advances event time and closes what it has passed, so a deadline nothing else would close needs only a periodic event (a `ClockSource`'s tags through `TagToEvent`).
Deadlines are half-open, `[start, deadline)`.
Events of one time are taken in the order `(trigger_time, trigger_offset, trigger_name, context)`.
`Coincidence`, `Then`, `Count`, `TimeInterval`, `EventStatistics`, `MultiChannelRecorder` and `Measure` hold an event until the latest `trigger_time` seen has passed its own by `max_lateness`, the longest one producer's events reach them after another's of the same time.
On a device, `Then` has no such verdict: its timeout is decided at the next dated event.
What is still open when the run ends is counted in `n_pending`, not judged: the framework gives a block with event inputs only no end-of-stream signal.

## Filters

Every block that selects triggers takes the same filter (`open_filter`, `filter`, `arm_filter`, ...), matched on `trigger_name` and `context`:

| filter                        | selects                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `beam_in`                     | that name                                                                                   |
| `beam_in/ring_a`              | that name in that context                                                                   |
| `beam_*`                      | every name the `*` completes                                                                |
| `!noise`, `edge/!calibration` | every name, or context, the rest does not match                                             |
| `[start, stop]`               | an interval: its start and its stop                                                         |
| `[start, ^stop]`              | an interval that closes on the first tag after the run of `stop`s; a `^` comes before a `!` |

An empty filter selects everything in `TagToEvent` and `EventFilter`.
A filter compiles once, in `settingsChanged`, into a trivially copyable state, so the same match runs on the host and in a kernel.
A **predicate** (`SampleFilter`, `TakeWhile`, `SkipWhile`, `PredicateSwitch`) is an ExprTk expression over `x`, the sample, and, through `gr::algorithm::ExpressionSymbols`, the block's settings by their `snake_case` name and the fields of the tags seen so far by their short key (`sample_rate` reads `gr:sample_rate`); an expression keeps a block on the host.

## A composed example

A rising edge followed within 5 µs by a pulse longer than 2 µs records the channel from 100 samples before the edge; every edge is measured for its rise time.

```text
  adc ─▶ LevelTrigger ─▶ PulseWidthTrigger ─┬─▶ MultiChannelRecorder ─▶ DataSet
           │ "up"          │ "long"         └─▶ RiseTime ─▶ value, as an event and a tag
           │               ▼
           ├─────────────▶ Then ── "up_then_long" ─▶ MultiChannelRecorder
           └──────────────────────────────── "up" ─▶ RiseTime
```

```cpp
using namespace gr::blocks::trigger;
using gr::blocks::measurement::Measure, gr::blocks::measurement::Quantity;
auto& up       = graph.emplaceBlock<ValueTrigger<float, ValueCondition::level>>({{"threshold", 0.5f}, {"hysteresis", 0.1f}, {"edge", "rising"}, {"trigger_name", "up"}, {"sample_rate", 1e6f}});
auto& pulse    = graph.emplaceBlock<ValueTrigger<float, ValueCondition::pulseWidth>>({{"threshold", 0.5f}, {"edge", "rising"}, {"duration_condition", "greater"}, {"duration_min", 2e-6}, {"duration_unit", "seconds"}, {"trigger_name", "long"}, {"sample_rate", 1e6f}});
auto& then     = graph.emplaceBlock<Then>({{"arm_filter", "up"}, {"trigger_filter", "long"}, {"max_delay", 5e-6}, {"trigger_name", "up_then_long"}});
auto& recorder = graph.emplaceBlock<MultiChannelRecorder<float>>({{"n_inputs", 1U}, {"open_filter", "up_then_long"}, {"reference", "start"}, {"n_pre", 100U}, {"n_post", 400U}});
auto& rise     = graph.emplaceBlock<Measure<float, Quantity::riseTime>>({{"open_filter", "up"}, {"n_pre", 8U}, {"n_post", 32U}, {"sample_rate", 1e6f}});
```

`Then` reports B dated as B, with A's tuple in `trigger_meta_info`, and an A that waited in vain as a `timeout_name` event dated at its deadline.
With `reference = start` the recorder anchors at that A, so its window `[A - n_pre, A + n_post)` holds the edge [2]; a window reaching before the first held sample is cut and flagged `incomplete`, never padded.
`Measure` (one block per quantity: `RiseTime`, `Rms`, `PositiveWidth`, ...) reports each value as a trigger decided at its gate's last sample, the gate's start in `trigger_meta_info` with the value and unit [1].
`StreamToDataSet` (or `StreamFilter` for a stream out) is the single-channel alternative to the recorder, cut between a start and a stop tag (`filter = "[start, stop]"`).
`Then` with no `arm_filter` watches the gaps between B events instead: a gap longer than `max_delay` is one timeout (`report = sequence|timeout|both`).
Runnable versions are in `blocks/trigger/test/qa_UseCases.cpp` and `blocks/measurement/test/qa_Measure.cpp`.

## Acknowledgement

[1] IEEE Std 181-2011, _IEEE Standard for Transitions, Pulses, and Related Waveforms_: the pulse conditions and measurements.
[2] IEEE Std 1057-2017, _IEEE Standard for Digitizing Waveform Recorders_: trigger point, pre- and post-trigger.
[3] IEEE Std 1588-2019, _IEEE Standard for a Precision Clock Synchronization Protocol for Networked Measurement and Control Systems_: the time `trigger_time` comes from.
