# GR4 Logging — User-API Reference

GR4 logging is a lightweight diagnostic channel for fatal failures, recoverable errors, warnings, and debug-build trace records.
The implementation is compatible with use on hosted (CPU), MCU and GPU targets.

## TL;DR

```cpp
#include <gnuradio-4.0/Logger.hpp>
#include <tuple>

gr::log::warning("dropped {} samples", dropped);
gr::log::error("pool '{}' not found", name);
gr::log::failure("device initialisation failed");

gr::log::info("block {} started", blockName); // debug builds only
gr::log::debug("state={}", state);            // debug builds only
gr::log::trace("chunk={}", chunk);            // debug builds only

gr::log::fatal("invalid invariant: {}", reason); // throws on hosted targets, aborts without exceptions
```

## Log levels

Level policy is fixed at compile time. There is no runtime severity threshold.

- `fatal`, `failure`, `error`, and `warning` are always compiled in and always emitted. These are
  the production-facing levels: true failures, dangerous fallbacks, data loss, and resource
  exhaustion.
- `info`, `debug`, and `trace` compile to no-ops unless `gr::meta::kDebugBuild` is `true`
  (debug builds only). They are development diagnostics, not routine production observability.

This is a fault/telemetry channel, not an operational logging/debugging framework: either there is a warning or an
error, or there is not.

## Compile-time and runtime format strings

```cpp
gr::log::warning("rate={} Hz", rate); // compile-time checked format string

std::string fmt = "{}: {}";
gr::log::error(gr::log::runtime(fmt), key, value); // explicit runtime format string
```

## Default backend

Hosted builds use the default console logger. It is write-through: each record is written to
the console as it is published, so no explicit pump is needed and a `fatal` record is visible
even if the process aborts immediately after. `fatal`, `error`, and `warning` records go to
stderr; `failure`, `info`, `debug`, and `trace` go to stdout by default.

`gr::log::flush()` returns the number of records flushed. The default backend does not need it;
it exists for backends that deliberately defer console I/O, e.g. a user-supplied backend that
enqueues records and notifies a dedicated consumer thread.

`MinSizeRel` / `EMBEDDED` builds default to `gr::log::HistoryLoggerBackend`: no console
output, no heap-backed queue, and the most recent 20 records retained in-place. The store is
bounded, non-blocking and best-effort (see below).

Hosted console routing is compile-time configurable with
`GR_LOG_CONSOLE_SPLIT_STDOUT_STDERR`: `1` keeps the split above, `0` sends all console records to stderr.

## History backend

```cpp
gr::log::HistoryLoggerBackend history;
auto* previous = gr::log::setBackend(&history);

gr::log::warning("boot step {}", step);

struct Snapshot {
    std::array<gr::log::LogRecord, gr::log::HistoryLoggerBackend::kCapacity> records{};
    std::size_t count{};
};

auto collect = [](const gr::log::LogRecord& record, void* user) noexcept {
    auto& snapshot = *static_cast<Snapshot*>(user);
    if (snapshot.count < snapshot.records.size()) {
        snapshot.records[snapshot.count++] = record;
    }
};

Snapshot snapshot;
std::ignore = history.snapshot(collect, &snapshot); // keep records available
std::ignore = history.drain(collect, &snapshot);    // mark current records consumed

std::ignore = gr::log::setBackend(previous);
```

MCU platform code can periodically call `snapshot()` or `drain()` and stream records to
UART/TTY using its own framing, interrupt, DMA, or polling policy. Core only stores and
exposes records.

The history backend is bounded, non-blocking and best-effort. `publish()` never blocks and
never allocates; if it finds the backend busy (a concurrent `publish`, `snapshot`, `drain`, or
`clear`), the record is dropped and counted rather than stored. `droppedBefore` on a retained
record reports a lower bound of records lost before that entry, covering both capacity
overwrites and contention drops. `clear()` is likewise best-effort and no-ops while busy.

## Device logging (GPU / accelerator kernels)

A kernel logs with the same call syntax as the host. It captures a `gr::log::DeviceLogger` — a
trivially copyable handle over a slab of device-shared memory that the host allocated:

```cpp
#include <gnuradio-4.0/device/DeviceLoggerBackend.hpp>

gr::device::DeviceLoggerBackend logBackend(context); // context is a gr::device::DeviceContext&
gr::log::setBackend(&logBackend);

struct Kernel {
    gr::log::DeviceLogger log;
    void operator()(std::size_t i) const { log.warning("processed {} items", i); }
};

context.parallelFor(n, Kernel{.log = logBackend.deviceLogger()});
context.wait();                 // the device produces; the host drains only after this barrier
std::ignore = gr::log::flush(); // decode, render, merge into the host backend
```

The kernel never allocates and never blocks. It claims a slot with one atomic bump, writes the record
as a `gr::pmt::ValueMap` (format string, arguments, level, `file:line`), and drops the record if the
slab is full. The **host** renders the text with real `std::format`, so format specifications such as
`{:.2f}` behave exactly as they do on the host:

```cpp
log.warning("v={:.2f} ok={}", 1.5, true); // -> "v=1.50 ok=true"
```

Call-site file and line are captured automatically; the timestamp is applied by the host when the
record is merged.

Device argument types are `bool`, any arithmetic type, and `std::string_view`. A `const char*`
argument is rejected at compile time — it would be a device pointer the host cannot dereference. At
most `gr::log::kDeviceLogMaxArgs` arguments are carried per record.

`droppedDeviceRecords()` counts records lost to a full slab, `truncatedDeviceRecords()` counts records
published with fields that did not fit the slot. Host-side `gr::log::*` calls are unaffected: they go
to the host backend directly and never touch the slab.

`gr::log::StaticDeviceLogSlab<slots>` offers the same transport over static storage, for targets
without an allocator.
