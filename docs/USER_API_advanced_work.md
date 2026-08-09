# Advanced work dispatch — User-API Reference

Most blocks implement `processOne` or `processBulk` and never touch the work dispatch.
This document is for advanced users who need full control over the processing pipeline.

## TL;DR — contract

- Override `work()`. `Block<T>` provides it as a member template, so a plain `work()` in a
  derived block hides it and takes over the whole pipeline.
- `work()` must not block — no I/O, no unbounded allocation, no locks.
- `work()` must consume inputs (`consumeReaders()`) and publish outputs (`publishSamples()`)
  — or delegate both to `finaliseIO()`.
- `work()` must call `progress->incrementAndGet()` when work was performed. The scheduler
  uses this to detect forward progress and avoid starvation.
- `work()` must return `work::Result{requestedWork, performedWork, status}`.
- On `DONE`: call `publishEoS()` before returning — downstream blocks need the EOS tag.

## `work::Status` and `work::Result`

```cpp
enum class work::Status {
    ERROR                     = -100,  // error in the work function
    INSUFFICIENT_OUTPUT_ITEMS = -3,    // output buffer too small
    INSUFFICIENT_INPUT_ITEMS  = -2,    // input buffer too small
    DONE                      = -1,    // block completed, flowgraph should terminate
    OK                        = 0,     // success, return values valid
};

struct work::Result {
    std::size_t requested_work;   // echo of the scheduler's request (pass through)
    std::size_t performed_work;   // how much work was done (not necessarily samples)
    Status      status;
};
```

The scheduler calls `work(requestedWork)` with a limit. The block echoes `requestedWork`
in the result and reports how much work was actually performed. `requested_work` and
`performed_work` are abstract work units — not necessarily samples. The scheduler uses
them to regulate how much work a block should do per invocation (e.g. based on how long
the previous call took). The only requirement is an affine relationship between
`requested_work` and `performed_work`.

`INSUFFICIENT_INPUT_ITEMS` / `INSUFFICIENT_OUTPUT_ITEMS` are not errors — they tell the
scheduler to retry later when more data is available. The framework zeroes `performed_work`
for these via `work::sanitiseProcessStatus()`.

## Flowgraph termination

A flowgraph terminates through two propagation paths:

### Forward propagation (source → sink)

1. A source block produces an **EOS (end-of-stream) tag** (`gr::tag::END_OF_STREAM`)
   on its output.
2. Downstream blocks detect the EOS tag in `getNextTagAndEosPosition()`.
3. Each block transitions: `RUNNING` → `REQUESTED_STOP` → `STOPPED`.
4. On the next `work()` call, the lifecycle branch completes the transition and returns `DONE`.
5. Before returning `DONE`, each block calls `publishEoS()` to propagate the tag downstream.
6. The process repeats until all blocks in the chain have stopped.

### Backward propagation (sink → source)

1. A block with `disconnect_on_done = true` (default) checks whether all its **non-optional**
   output ports have connected downstream blocks (`hasNoDownStreamConnectedChildren()`).
2. If all mandatory downstream consumers are gone, the block calls `requestStop()`.
3. On the next `work()` call, the block transitions to `STOPPED` and calls
   `disconnectFromUpStreamParents()` — which disconnects its input ports.
4. This triggers backward propagation: the upstream block now has no downstream children,
   so it stops too.
5. The process repeats until it reaches the source.

### Interaction between both paths

In a typical flowgraph, forward propagation (EOS tag) drives the shutdown.
Backward propagation handles the case where a downstream block is removed or disconnected
at runtime. Both paths converge at the same lifecycle transitions.

```
Source ──EOS tag──▶ Block ──EOS tag──▶ Sink
                                         │
                              disconnect_on_done
                                         │
Source ◀──disconnect── Block ◀──disconnect─┘
```

## When to override `work()`

Override only when the standard dispatch is insufficient:

- device dispatch (GPU/FPGA) with custom buffer management
- multi-rate blocks with non-standard resampling logic
- blocks that need to interleave computation with async I/O

For everything else, `processOne` / `processBulk` is simpler and sufficient.

## Where `work()` sits

```
BlockModel::work()  ← virtual, called by the scheduler via type-erased BlockWrapper
  └─ Block<Derived>::work()  ← CRTP-resolved, two `requires`-constrained overloads:
                               one for NormalBlock carrying the pipeline below,
                               one for every other block::Category returning OK
       └─ dispatchProcessing()  ← which KIND of body runs
            └─ dispatchProcessOne()  ← how a processOne body runs
```

A user block that needs custom dispatch declares its own `work()`, which hides both base
overloads, and reuses the composable helper methods from `Block<T>` listed below.

## Pipeline phases

The `NormalBlock` `work()` overload orchestrates these phases, each available as a method:

```
[state check]  — one acquire load; the branch below is skipped while RUNNING
    │  settles REQUESTED_STOP → STOPPED; ends the call on ERROR (ERROR) or STOPPED (DONE)
    v
[disconnect_on_done]  — if nothing downstream is connected → requestStop()
    v
applyChangedSettings()  — commits staged settings, stages forward params
    v
computeSampleLimits(requestedWork) → SampleLimits
    │  reads port caches, tag positions, resampling, chunk limits
    v
[skip-before handling]  — stride: consume samples before the processing window
    v
[EOS check]  — if end-of-stream, publish EOS and return DONE
    v
prepareStreams(...)  — create InputSpanLike / OutputSpanLike
    v
applyInputTagsAndSettings(inputSpans, processedIn, hasAnyTag)
    │  merge input tags, auto-apply settings
    v
dispatchProcessing(inputSpans, outputSpans, processedIn, processedOut) → Status
    │  device bulk tier / processBulk / dispatchProcessOne
    │      dispatchProcessOne → device per-sample tier / SIMD / pure / non-const
    v
work::sanitiseProcessStatus(status, processedIn, processedOut)
    │  if ERROR or INSUFFICIENT → zero both counts
    v
finaliseIO(inputSpans, outputSpans, status, processedIn, processedOut, resampledIn)
    │  publish tags + samples, consume inputs, handle EOS
    v
work::computePerformedWork(status, processedIn, processedOut, isSource)
    v
progress->incrementAndGet()  — mandatory when performedWork > 0
```

## Example: custom `work()` override

```cpp
template<typename T>
struct MyDeviceBlock : gr::Block<MyDeviceBlock<T>> {
    // ... ports, settings ...

    work::Result work(std::size_t requestedWork = std::numeric_limits<std::size_t>::max(), gr::device::DeviceContext& computeBackend = gr::device::hostBackend()) noexcept {
        using enum gr::work::Status;
        std::ignore = computeBackend;

        // read the state once: a RUNNING block skips the whole lifecycle branch. An override owns this
        // itself -- `Block<T>` keeps it as a local lambda rather than a reusable member.
        if (const gr::lifecycle::State state = this->state(); state != gr::lifecycle::State::RUNNING) {
            if (state == gr::lifecycle::State::ERROR) {
                return {requestedWork, 0UZ, ERROR};
            }
            if (state == gr::lifecycle::State::REQUESTED_STOP) {
                std::ignore = this->changeStateTo(gr::lifecycle::State::STOPPED);
            }
            if (this->state() == gr::lifecycle::State::STOPPED) {
                this->disconnectFromUpStreamParents();
                return {requestedWork, 0UZ, DONE};
            }
        }

        auto limits = this->computeSampleLimits(requestedWork);
        if (limits.resampledIn == 0 && limits.resampledOut == 0) {
            return {requestedWork, 0UZ, limits.resampledStatus};
        }

        std::size_t processedIn  = limits.resampledIn;
        std::size_t processedOut = limits.resampledOut;

        auto inputSpans  = this->prepareStreams(
            this->template inputPorts<PortType::STREAM>(&this->self()), processedIn);
        auto outputSpans = this->prepareStreams(
            this->template outputPorts<PortType::STREAM>(&this->self()), processedOut);

        this->applyInputTagsAndSettings(inputSpans, processedIn, limits.hasAnyTag);

        // custom device dispatch instead of standard dispatchProcessing()
        work::Status status = myGpuKernel(inputSpans, outputSpans);

        work::sanitiseProcessStatus(status, processedIn, processedOut);
        this->finaliseIO(inputSpans, outputSpans, status, processedIn, processedOut,
            limits.resampledIn);

        constexpr bool kIsSource =
            gr::traits::block::stream_input_port_types<MyDeviceBlock>::size.value == 0;
        std::size_t performedWork =
            work::computePerformedWork(status, processedIn, processedOut, kIsSource);
        if (performedWork > 0UZ) {
            this->progress->incrementAndGet();
        }
        return {requestedWork, performedWork, status};
    }
};
```

## `SampleLimits` struct

Returned by `computeSampleLimits()`:

```cpp
struct SampleLimits {
    std::size_t  resampledIn{}, resampledOut{}, inputSkipBefore{};
    work::Status resampledStatus = work::Status::OK;
    bool         hasTag{}, hasAnyTag{}, asyncEoS{}, isEosPresent{};
    bool         hasAsyncIn{}, hasAsyncOut{};
};
```

## Utility functions

Defined in `gr::work` namespace:

| Function                                          | What it does                                 |
| ------------------------------------------------- | -------------------------------------------- |
| `sanitiseProcessStatus(status, in, out)`          | if ERROR/INSUFFICIENT → zero both counts     |
| `computePerformedWork(status, in, out, isSource)` | work accounting for scheduler prioritisation |

## Full contract

| Requirement                                  | Why                                                  |
| -------------------------------------------- | ---------------------------------------------------- |
| return `work::Result`                        | scheduler reads `performed_work` and `status`        |
| echo `requestedWork` in result               | scheduler compares requested vs performed work units |
| consume + publish or `finaliseIO()`          | back-pressure propagation, buffer management         |
| `progress->incrementAndGet()` when work done | scheduler starvation detection                       |
| `publishEoS()` before returning `DONE`       | forward-propagate shutdown to downstream             |
| no blocking calls                            | real-time thread, deterministic execution            |
| no unbounded allocation                      | real-time thread                                     |
| handle `DONE` from `dispatchProcessing()`    | user's `processBulk` may signal completion           |
