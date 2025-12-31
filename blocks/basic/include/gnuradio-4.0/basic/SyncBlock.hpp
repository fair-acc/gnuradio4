#ifndef GNURADIO_SYNC_BLOCK_HPP
#define GNURADIO_SYNC_BLOCK_HPP

#include <gnuradio-4.0/Block.hpp>

namespace gr::basic {

using namespace gr;

GR_REGISTER_BLOCK(gr::basic::SyncBlock, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct SyncBlock : Block<SyncBlock<T>, NoDefaultTagForwarding> {
    using Description = Doc<R""(@brief SyncBlock synchronises data streams across multiple inputs.

The SyncBlock addresses key scenarios including initial time shifts, clock drift, and significant delays.
It synchronises input samples based on synchronization tags and publishes desynchronization tags with
information about number of samples that were dropped.

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

#### 4. Zero padding (optional) -> timeout and zero padding to be discussed see https://github.com/fair-acc/gnuradio4/issues/466
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

Note: We assume that desynchronization should not exceed the buffer size of the SyncBlock; if it does, the samples will be dropped.
)"">;

    std::vector<gr::PortIn<T, gr::Async>> inputs;
    std::vector<gr::PortOut<T>>           outputs;

    // settings
    Annotated<gr::Size_t, "n_ports", gr::Visible, gr::Doc<"variable number of in/out ports">, gr::Limits<1U, 32U>> n_ports          = 0U;
    Annotated<gr::Size_t, "max_history_size", Doc<"Max size of history">>                                          max_history_size = 32000U; // should be less than actual buffer size (better < 80%)
    Annotated<std::string, "filter", Doc<"trigger name filter">>                                                   filter           = "";
    Annotated<std::uint64_t, "tolerance", Doc<"trigger time tolerance [ns]">>                                      tolerance        = 5ULL;

    GR_MAKE_REFLECTABLE(SyncBlock, inputs, outputs, n_ports, max_history_size, filter, tolerance);

    bool                     _isStreamSynchronized = false;
    std::vector<std::size_t> _nDroppedSamples{}; // number of dropped samples, to be sent with desynchronized tag

    int _processBulkCounter = 0;

    struct SyncData {
        std::size_t index; // sync index
        std::size_t nPre;  // number of pre samples, sample with sync index is not included
        std::size_t nPost; // number of post samples, sample with sync index is not included
    };

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("n_ports") && oldSettings.at("n_ports") != newSettings.at("n_ports")) {
            // if one of the port is already connected and n_ports was changed then throw
            bool inConnected  = std::any_of(inputs.begin(), inputs.end(), [](const auto& port) { return port.isConnected(); });
            bool outConnected = std::any_of(outputs.begin(), outputs.end(), [](const auto& port) { return port.isConnected(); });
            if (inConnected || outConnected) {
                throw gr::exception("Number of input/output ports cannot be changed after Graph initialization.");
            }
            std::print("{}: configuration changed: n_ports {} -> {}\n", this->name, oldSettings.at("n_ports"), newSettings.at("n_ports"));
            inputs.resize(n_ports);
            outputs.resize(n_ports);
            _nDroppedSamples.resize(n_ports, 0UZ);
        }

        if (newSettings.contains("max_history_size")) {
            // should be less than actual buffer size (better < 80%)
            // The logic has to be implemented when the new Port API is available
        }
    }

    template<InputSpanLike TInput, OutputSpanLike TOutput>
    gr::work::Status processBulk(const std::span<TInput>& ins, std::span<TOutput>& outs) {
        std::size_t nPorts = ins.size();

        const std::vector<SyncData> syncData = synchronize(ins);
        const bool                  canSync  = syncData.size() == nPorts;

        if (canSync) {
            const std::size_t minPre            = std::ranges::min(syncData | std::views::transform(&SyncData::nPre));
            const std::size_t minPost           = std::ranges::min(syncData | std::views::transform(&SyncData::nPost));
            const std::size_t minSamplesOut     = std::ranges::min(outs | std::views::transform([&](const auto& out) { return out.size(); }));
            const std::size_t nSamplesToPublish = std::min(minPre + 1 + minPost, minSamplesOut);

            for (std::size_t i = 0UZ; i < nPorts; i++) {
                const std::size_t nSamplesToDrop    = syncData[i].index - minPre;
                const std::size_t nSamplesToConsume = nSamplesToDrop + nSamplesToPublish;

                std::ranges::copy_n(std::next(ins[i].begin(), static_cast<std::ptrdiff_t>(nSamplesToDrop)), static_cast<std::ptrdiff_t>(nSamplesToPublish), outs[i].begin());
                const std::size_t totalDroppedSamples = _nDroppedSamples[i] + nSamplesToDrop;

                publishDroppedSamplesTagIfNotZero(outs[i], totalDroppedSamples);
                publishInputTags(ins[i], outs[i], nSamplesToDrop, nSamplesToPublish);
                _nDroppedSamples[i] = 0UZ;

                std::ignore = ins[i].consume(nSamplesToConsume);
                ins[i].consumeTags(nSamplesToConsume);
                outs[i].publish(nSamplesToPublish);
            }
            _isStreamSynchronized = true;
        } else {
            const std::size_t minSamplesBeforeSyncTag = std::ranges::min(ins | std::views::transform([&](const auto& in) { return getNSamplesBeforeSyncTag(in); }));
            const std::size_t minSamplesOut           = std::ranges::min(outs | std::views::transform([&](const auto& out) { return out.size(); }));
            const std::size_t nSamplesToCopy          = std::min(minSamplesBeforeSyncTag, minSamplesOut);
            if (_isStreamSynchronized && nSamplesToCopy > 0UZ) { // all streams are in sync -> write sample before first Sync tag
                for (std::size_t i = 0; i < nPorts; i++) {
                    std::ranges::copy_n(ins[i].begin(), static_cast<std::ptrdiff_t>(nSamplesToCopy), outs[i].begin());

                    publishDroppedSamplesTagIfNotZero(outs[i], _nDroppedSamples[i]);
                    publishInputTags(ins[i], outs[i], 0UZ, nSamplesToCopy);

                    _nDroppedSamples[i] = 0UZ;

                    std::ignore = ins[i].consume(nSamplesToCopy);
                    ins[i].consumeTags(nSamplesToCopy);
                    outs[i].publish(nSamplesToCopy);
                }
            } else { // streams are NOT in sync -> check back pressure and drop samples if needed
                for (std::size_t i = 0; i < nPorts; i++) {
                    const std::size_t nSamplesToDrop = ins[i].size() < static_cast<std::size_t>(max_history_size) ? 0UZ : ins[i].size() - static_cast<std::size_t>(max_history_size);
                    if (nSamplesToDrop != 0UZ) {
                        std::ignore = ins[i].consume(nSamplesToDrop);
                        ins[i].consumeTags(nSamplesToDrop);

                        _nDroppedSamples[i] += nSamplesToDrop;
                        outs[i].publish(0UZ);
                        _isStreamSynchronized = false;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < nPorts; i++) {
            if (!ins[i].isConsumeRequested()) {
                std::ignore = ins[i].consume(0UZ);
                ins[i].consumeTags(0UZ);
                outs[i].publish(0UZ);
            }
        }
        return gr::work::Status::OK;
    }

    constexpr void publishDroppedSamplesTagIfNotZero(OutputSpanLike auto& out, std::size_t nDroppedSamples) {
        if (nDroppedSamples > 0UZ) {
            out.publishTag(property_map{{gr::tag::N_DROPPED_SAMPLES.shortKey(), gr::pmt::Value(nDroppedSamples)}}, 0UZ);
        }
    }

    constexpr void publishInputTags(InputSpanLike auto& in, OutputSpanLike auto& out, std::size_t nDroppedSamples, std::size_t nSamplesToPublish) {
        for (const auto& tag : in.rawTags) {
            const std::size_t relativeTagIndex = getRelativeTagIndex(in, tag);
            if (relativeTagIndex != std::numeric_limits<std::size_t>::max() && relativeTagIndex >= nDroppedSamples && relativeTagIndex < nSamplesToPublish + nDroppedSamples) {
                out.publishTag(tag.map, relativeTagIndex - nDroppedSamples);
            }
        }
    }

    template<InputSpanLike TInput>
    [[nodiscard]] constexpr std::vector<SyncData> synchronize(const std::span<TInput>& ins) {
        const std::uint64_t syncTime = findSyncTime(ins);
        if (syncTime == std::numeric_limits<std::uint64_t>::max()) {
            return {};
        }

        std::vector<SyncData> syncData;
        syncData.reserve(ins.size());
        for (const auto& inSpan : ins) {
            for (const auto& tag : inSpan.rawTags) {
                const std::size_t relativeTagIndex = getRelativeTagIndex(inSpan, tag);
                if (isTimeDifferenceWithinTolerance(getTime(tag), syncTime) && relativeTagIndex != std::numeric_limits<std::size_t>::max()) {
                    const std::size_t nPre  = getAvailablePreSamples(inSpan, relativeTagIndex);
                    const std::size_t nPost = getAvailablePostSamples(inSpan, relativeTagIndex);
                    syncData.push_back({relativeTagIndex, nPre, nPost});
                    break;
                }
            }
        }

        assert(ins.size() == syncData.size());

        return syncData;
    }

    template<InputSpanLike TInput>
    [[nodiscard]] constexpr std::uint64_t findSyncTime(const std::span<TInput>& ins) {
        std::vector<std::vector<std::uint64_t>> syncTimesPerInSpan; // Collect sync times for each input span
        syncTimesPerInSpan.reserve(ins.size());
        std::set<std::uint64_t> allSyncTimes; // Collect all unique sync times, ordered
        for (const auto& inSpan : ins) {
            std::vector<std::uint64_t> syncTimes;
            for (const auto& tag : inSpan.rawTags) {
                const std::size_t relativeTagIndex = getRelativeTagIndex(inSpan, tag);
                if (isSyncTag(tag) && relativeTagIndex < inSpan.size() && relativeTagIndex != std::numeric_limits<std::size_t>::max()) {
                    const std::uint64_t time = getTime(tag);
                    syncTimes.push_back(time);
                    allSyncTimes.insert(time);
                }
            }
            syncTimesPerInSpan.push_back(std::move(syncTimes));
        }

        // Find the earliest sync time present in all input spans within tolerance
        auto findInAllIt = std::ranges::find_if(allSyncTimes, [&](const auto& curTime) { //
            return std::ranges::all_of(syncTimesPerInSpan, [&](const auto& syncTimes) {  //
                return std::ranges::any_of(syncTimes, [&](const auto& t) { return isTimeDifferenceWithinTolerance(curTime, t); });
            });
        });

        return findInAllIt != allSyncTimes.end() ? *findInAllIt : std::numeric_limits<std::uint64_t>::max();
    }

    [[nodiscard]] constexpr std::size_t getAvailablePreSamples(const InputSpanLike auto& in, std::size_t syncIndex) {
        // Check if there’s an earlier sync tag that cannot be synchronized; if so, calculate available samples only up to that tag.
        const auto foundTag = std::ranges::find_if(in.rawTags, [&](const auto& tag) { return getRelativeTagIndex(in, tag) < syncIndex && isSyncTag(tag); });
        if (foundTag != in.rawTags.end()) {
            return syncIndex - getRelativeTagIndex(in, *foundTag) - 1;
        }
        return syncIndex;
    }

    [[nodiscard]] constexpr std::size_t getAvailablePostSamples(const InputSpanLike auto& in, std::size_t syncIndex) {
        // Check if there’s a later sync tag; if so, calculate available samples only up to that tag.
        auto foundTag = std::ranges::find_if(in.rawTags, [&](const auto& tag) {
            const std::size_t relativeTagIndex = getRelativeTagIndex(in, tag);
            return relativeTagIndex > syncIndex && relativeTagIndex < in.size() && isSyncTag(tag);
        });
        if (foundTag != in.rawTags.end()) {
            return getRelativeTagIndex(in, *foundTag) - syncIndex - 1;
        }
        return in.size() - syncIndex - 1;
    }

    [[nodiscard]] std::size_t getNSamplesBeforeSyncTag(const InputSpanLike auto& in) const {
        for (const auto& tag : in.rawTags) {
            const std::size_t relativeTagIndex = getRelativeTagIndex(in, tag);
            if (isSyncTag(tag) && relativeTagIndex != std::numeric_limits<std::size_t>::max()) {
                return std::min(relativeTagIndex, in.size()); // tags are ordered, return distance to the first sync tag
            }
        }
        return in.size();
    }

    [[nodiscard]] constexpr bool isTimeDifferenceWithinTolerance(std::uint64_t t1, std::uint64_t t2) { return ((t1 > t2) ? t1 - t2 : t2 - t1) < tolerance; }

    [[nodiscard]] constexpr bool isSyncTag(const gr::Tag& tag) const {
        const std::string keyTriggerName = gr::tag::TRIGGER_NAME.shortKey();
        const std::string keyTriggerTime = gr::tag::TRIGGER_TIME.shortKey();

        const auto itName = tag.map.find(keyTriggerName);
        if (itName == tag.map.end() || //
            (!filter->empty() && itName->second.value_or(std::string_view{}) != filter)) {
            return false;
        }
        const auto itTime = tag.map.find(keyTriggerTime);
        if (itTime == tag.map.end() || !itTime->second.holds<std::uint64_t>()) {
            return false;
        }
        return true;
    }

    [[nodiscard]] constexpr std::uint64_t getTime(const gr::Tag& tag) const {
        const std::string keyTriggerTime = gr::tag::TRIGGER_TIME.shortKey();

        auto it = tag.map.find(keyTriggerTime);
        if (it == tag.map.end()) {
            return 0ULL;
        }

        auto ptr = it->second.get_if<std::uint64_t>();
        if (ptr == nullptr) {
            return 0ULL;
        }
        return *ptr;
    }

    [[nodiscard]] constexpr std::size_t getRelativeTagIndex(const InputSpanLike auto& in, const Tag& tag) const { //
        return tag.index >= in.streamIndex ? tag.index - in.streamIndex : std::numeric_limits<std::size_t>::max();
    }
};

} // namespace gr::basic

#endif // GNURADIO_SYNC_BLOCK_HPP
