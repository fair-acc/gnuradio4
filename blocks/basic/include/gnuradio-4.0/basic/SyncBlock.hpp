#ifndef GNURADIO_SYNC_BLOCK_HPP
#define GNURADIO_SYNC_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>

namespace gr::basic {

using namespace gr;

using SyncBlockDoc = Doc<R""(
### Important Prerequisites
The sample rate must be the same across all input ports.

For the examples provided, consider the following:
1) Samples (`s1, s2, ...`) represent samples where the index indicates their sequence of arrival.
2) Samples for different ports may vary (`s1` for `port1` may may not be equal to `s1` for `port2`)
3) Illustrated with examples for two inputs (`in1`, `in2`) and two outputs (`out1`, `out2`) for simplicity:
```text
       +-------------------------------+
in1 +-+|                               |+-+ out1
    +-+|                               |+-+
       |          SyncBlock            |
in2 +-+|                               |+-+ out2
    +-+|                               |+-+
       +-------------------------------+
```

### Implementing `SyncBlock`: Key Scenarios
When designing a `SyncBlock`, consider the following four scenarios:

#### 1. Time Shift at Start
Devices may initiate data transmission at different times, leading to a time shift in data arrival.

Example:
```text
                 T1                         T1,TD1
                  ↓                            ↓
in1: s2-s3-s4-s5-s6      →  out1: s2-s3-s4-s5-s6

                    T1                        T1
                     ↓                         ↓
in2: s1-s2-s3-s4-s5-s6   →  out2: s2-s3-s4-s5-s6
```
Both ports align with synchronization tag `T1` at sample `s6`.
Samples received before `T1` are also included in the output, resulting in `out1` and `out2` both outputting `s2-s3-s4-s5-s6`.
Note that the synchronization tags are forwarded further.
In addition a desync tag (`TD1`) is published for `out1` which includes information about number of dropped samples.

#### 2. Clock Drift
This scenario accounts for a clock drift - minor timing discrepancy, such as several PPM (±1 sample per 100,000 samples).

Example:
```text
     T1          T2               T1          T2
      ↓           ↓                ↓           ↓
in1: s1-s2-s3-s4-s5      →  out1: s1-s2-s3-s4-s5

     T1             T2            T1         T2,TD2
      ↓              ↓             ↓           ↓
in2: s1-s2-s3-s4-s5-s6   →  out2: s1-s2-s3-s4-s6
```
Post-synchronization with `T1`, the `SyncBlock` adjusts to align the samples, dropping any excess samples.
The `out1` is `s1-s2-s3-s4-s5`, the `out2` is `s1-s2-s3-s4-s6`. `s5` sample is skipped in the examples.
In addition a desync tag (`TD2`) is published for port `out2` which includes information about number of dropped samples.

#### 3a. Time-Out: no data
This situation involves a significant delay in data reception (> time-out).

```text
     T1                  T2                T1         TD1      T2
      ↓                   ↓                 ↓          ↓        ↓
in1: s1-s2-....-s5-s6-s7-s8-s9    →  out1: s1-s2   →  s5-s6-s7-s8-s9

     T1                   T2               T1         TD1      T2
      ↓                    ↓                ↓          ↓        ↓
in2: s1-s2-s3-s4-s5-s6-s7-s8-s9   →  out2: s1-s2   →  s5-s6-s7-s8-s9
```

New synchronization occurs with `s8`, including prior samples (`s5-s6-s7`) in the output after the desync period.
`SyncBlock` publishes desynchronization tag (`TD1`) for all ports to indicate data stream desynchronization.

#### 3b. Time-Out: no tag
```text
     T1                T3                  T1         TD1        T3
      ↓                 ↓                   ↓         ↓           ↓
in1: s1-s2-s3-s4-s5-s6-s7-s8-s9   →  out1: s1-s2  →  s3-s4-s5-s6-s7-s8-s9

     T1    T2          T3                  T1        T2          T3
      ↓     ↓           ↓                   ↓         ↓           ↓
in2: s1-s2-s3-s4-s5-s6-s7-s8-s9   →  out2: s1-s2  →  s3-s4-s5-s6-s7-s8-s9
```
If port `in2` receives `T2` but port `in1` did not before a timeout, the `SyncBlock` awaits the next sync tag to attempt data alignment.
The prior samples to `T3` (`s3-s4-s5-s6`) will be also included.

#### 4. Zero padding (optional)
similar to 3a, this situation involves a significant delay in data reception (> time-out),
but in this case `SyncBlock` publishes "zeros" after timeout.

```text
     T1              T2                    T1        TD1       T2
      ↓               ↓                     ↓        ↓          ↓
in1: s1-s2-...-s6-s7-s8-s9         → out1: s1-s2  →  0-0-0-0-0-s8-s9

     T1                   T2              T1                        T2
      ↓                    ↓               ↓                         ↓
in2: s1-s2-s3-s4-s5-s6-s7-s8-s9   → out2: s1-s2  →   s3-s4-s5-s6-s7-s8-s9
```
`SyncBlock` publishes desynchronization tag (`TD1`) with the first "zero" sample.
New synchronization occurs with `s8`, prior samples (`s6-s7`) are NOT included to the output for zero padding.

)"">;

template<typename T>
struct SyncBlock : public gr::Block<SyncBlock<T>, SyncBlockDoc> {
    std::vector<gr::PortIn<T, gr::Async>> inputs;
    std::vector<gr::PortOut<T>>           outputs;

    // settings
    Annotated<gr::Size_t, "n_ports", gr::Visible, gr::Doc<"variable number of in/out ports">, gr::Limits<1U, 32U>> n_ports     = 0U;
    Annotated<gr::Size_t, "buffer_size", Doc<"Size of history buffer">>                                            buffer_size = 65536;

    std::vector<HistoryBuffer<T>> _inBuffer{};

    struct SyncTagData {
        std::uint64_t          time;
        Tag::signed_index_type index;
    };
    std::vector<std::queue<SyncTagData>> _syncTagData;

    int processBulkCounter = 0;

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("n_ports") && oldSettings.at("n_ports") != newSettings.at("n_ports")) {
            // if one of the port is already connected and n_ports was changed then throw
            bool inConnected  = std::any_of(inputs.begin(), inputs.end(), [](const auto& port) { return port.isConnected(); });
            bool outConnected = std::any_of(outputs.begin(), outputs.end(), [](const auto& port) { return port.isConnected(); });
            if (inConnected || outConnected) {
                // TODO: for the moment keep both: throw exception and emit message
                using namespace gr::message;
                std::string messageError{"Number of input/output ports cannot be changed after Graph initialization."};
                // this->emitMessage(this->msgOut, { { key::Kind, kind::Error }, { key::ErrorInfo, messageError } });
                throw std::range_error(messageError);
            }
            fmt::print("{}: configuration changed: n_ports {} -> {}\n", this->name, oldSettings.at("n_ports"), newSettings.at("n_ports"));
            inputs.resize(n_ports);
            outputs.resize(n_ports);
            _inBuffer.resize(n_ports);
            for (std::size_t i = 0; i < _inBuffer.size(); i++) {
                _inBuffer[i].reset();
                _inBuffer[i] = HistoryBuffer<T>(buffer_size);
            }
        }

        if (newSettings.contains("buffer_size")) {
            for (std::size_t i = 0; i < _inBuffer.size(); i++) {
                auto newBuffer = HistoryBuffer<T>(buffer_size);
                newBuffer.push_back_bulk(_inBuffer[i]);
                _inBuffer[i] = std::move(newBuffer);
            }
        }
    }

    template<ConsumablePortSpan TInput, PublishableSpan TOutput>
    gr::work::Status processBulk(const std::span<TInput>& ins, std::span<TOutput>& outs) {
        fmt::println("SyncBlock::processBulk ins.size:{}, outs.size:{}, processBulkCounter:{}", ins.size(), outs.size(), processBulkCounter++);
        std::size_t nPorts = ins.size();
        for (std::size_t i = 0; i < nPorts; i++) {
            fmt::println("SyncBlock::processBulk ins[{}].size:{}, outs[{}].size:{}, ins.tags.size():{}", i, ins[i].size(), i, outs[i].size(), ins[i].tags.size());
            ins[i].consume(ins[i].size());
            outs[i].publish(ins[i].size());
        }

        return gr::work::Status::OK;
    }

private:
    template<ConsumablePortSpan TInput>
    void processInputTags(const std::span<TInput>& ins) {
        for (std::size_t i = 0; i < ins.size(); i++) {
            for (const Tag& tag : ins[i].tags) {
                if (tag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
                    const pmtv::pmt& pmtTimeUtcNs = tag.map.at(std::string(gr::tag::TRIGGER_TIME.shortKey()));
                    if (std::holds_alternative<uint64_t>(pmtTimeUtcNs)) {
                    }
                }
            }
        }
    }
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::basic::SyncBlock<T>), inputs, outputs, n_ports);

#endif // GNURADIO_SYNC_BLOCK_HPP
