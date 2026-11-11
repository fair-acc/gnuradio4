#ifndef GNURADIO_BASIC_STREAM_TO_TENSOR_HPP
#define GNURADIO_BASIC_STREAM_TO_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/WindowGeometry.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

namespace gr::basic {

/** the first window of `size` samples, one every `stride`, that holds sample i or, when i falls between windows, starts after it */
[[nodiscard]] constexpr std::size_t firstWindowHolding(std::size_t i, std::size_t size, std::size_t stride) noexcept { return i < size ? 0UZ : (i - size) / stride + 1UZ; }

/** [ns] at `sampleRate` [Hz]; unchanged at an unknown rate (0), saturated at the ends of the uint64 range */
[[nodiscard]] inline std::uint64_t timeOfFrameStart(std::uint64_t sampleTimeNs, std::ptrdiff_t samplesAfterFrameStart, float sampleRate) noexcept {
    if (!(sampleRate > 0.f)) {
        return sampleTimeNs;
    }
    constexpr std::uint64_t kLatestNs = std::numeric_limits<std::uint64_t>::max();
    const double            shiftNs   = std::round(static_cast<double>(samplesAfterFrameStart) * 1e9 / static_cast<double>(sampleRate));
    if (shiftNs >= static_cast<double>(sampleTimeNs)) {
        return 0U;
    }
    if (-shiftNs >= static_cast<double>(kLatestNs - sampleTimeNs)) {
        return kLatestNs;
    }
    return shiftNs >= 0. ? sampleTimeNs - static_cast<std::uint64_t>(shiftNs) : sampleTimeNs + static_cast<std::uint64_t>(-shiftNs);
}

namespace detail {

[[nodiscard]] std::size_t completeWindows(const auto& block, std::size_t available, std::size_t outputSpace) {
    const WindowGeometry geometry = windowGeometry(block, available, outputSpace);
    return geometry.hop == 0UZ ? std::min(available, outputSpace) : geometry.nWindows;
}

/**
 * calls `visit(window, relativeIndex, tagMap)` for each tag of `input` at or after sample `coveredUntil` with the first window holding its sample -- or, for a skipped
 * sample, the next one: one of the `nWindows` emitted, or `nWindows` for a consumed sample after them, whose window a later call emits;
 * a tag on a sample not yet consumed waits for that call
 */
void forEachWindowTag(auto& input, std::size_t nWindows, std::size_t nConsumed, std::size_t size, std::size_t hop, std::size_t coveredUntil, auto&& visit) {
    for (const auto& [relativeIndex, tagMap] : input.tags()) {
        const std::size_t sample = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(input.streamIndex) + relativeIndex);
        if (sample < coveredUntil || nWindows == 0UZ) {
            continue;
        }
        const std::size_t window = firstWindowHolding(static_cast<std::size_t>(relativeIndex), size, hop);
        if (window < nWindows || static_cast<std::size_t>(relativeIndex) < nConsumed) {
            visit(std::min(window, nWindows), static_cast<std::ptrdiff_t>(relativeIndex), tagMap.get());
        }
    }
}

/** the input sample index up to which the tags of `nTensors` tensors starting at `streamIndex` have gone out */
[[nodiscard]] constexpr std::size_t coveredAfter(std::size_t streamIndex, std::size_t nTensors, std::size_t size, std::size_t hop) noexcept { return streamIndex + (nTensors - 1UZ) * hop + std::max(size, hop); }
} // namespace detail

GR_REGISTER_BLOCK(gr::basic::StreamsToTensor, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (nPorts <= detail::kMaxPorts)
struct StreamsToTensor : Block<StreamsToTensor<T, nPorts>, NoTagPropagation, Resampling<1U, 1U, false>, Stride<0U, false>, Framing> {
    using Description = Doc<R""(@brief one tensor per frame of the M inputs, input m in row m (GR3: Streams to Vector):
a frame is `size` samples of every input, one every `stride` samples -- of shape [M] for `size` 1, [M,N] channel-major for `size` N;
`stride` 0 is `size`: back to back; smaller overlaps, larger skips.

The frames arrive at the inputs' rate divided by the stride, their `gr:frame_rate`; `gr:sample_rate` stays the signal's; trailing samples short of a frame stay behind, and a last frame short of its stride is not
emitted where its storage is device memory.
A tag on any input goes to the first frame holding its sample, or the next one when the sample is skipped; the same tag on several inputs goes once.
A frame carries the time of its first sample: a forwarded `gr:trigger_time` moves by its sample's distance from the frame's first sample at the inputs' `gr:sample_rate`, and stays while that rate is unknown.
`nPorts = 0` has `n_ports` inputs, fixed once connected; a skip reads the whole stride, so `max(size, stride)` must fit the input buffers.

Gated, with `evtIn` connected at start (host only): each event passing `filter`, dated on input 0's time base,
opens one frame of `size` samples from `n_pre` before its sample, tagged with the event dated at the frame's first sample and the `sample_rate`.
An event within `holdoff_samples` of the last frame's event opens none.
An event may arrive up to the longer of `kLateSamples` and `kLateSeconds` after its sample; later, undated and unplaceable ones are counted.)"">;

    static constexpr std::size_t kLateSamples = 4096UZ;
    static constexpr double      kLateSeconds = 0.5; /// at a known sample rate: a trigger behind a matched filter lags it by the filter's length

    gr::EventPortIn                    evtIn;
    detail::PortsOf<PortIn<T>, nPorts> in;
    TensorPortOut<T>                   out;

    Annotated<gr::Size_t, "n_ports", Doc<"inputs; nPorts when that is not 0">, Limits<1U, static_cast<gr::Size_t>(detail::kMaxPorts)>> n_ports = nPorts == 0UZ ? 2U : static_cast<gr::Size_t>(nPorts);
    Annotated<gr::Size_t, "size", Doc<"samples per input in a frame; 1 gives frames of shape [M]">, Limits<1U, detail::kMaxRunLength>> size    = 1U;
    Annotated<std::string, "filter", Doc<"gated: the trigger filter an event must pass; empty: every event">>                          filter;
    Annotated<gr::Size_t, "n_pre", Doc<"gated: samples of a frame before its event's sample">, Limits<0U, detail::kMaxRunLength>>      n_pre           = 0U;
    Annotated<gr::Size_t, "holdoff_samples", Doc<"gated: samples after a frame's event in which no further one opens, 0 = none">>      holdoff_samples = 0U;

    gr::Size_t _eventsRefused = 0U; /// gated: undated, unplaceable or too late to open a frame

    GR_MAKE_REFLECTABLE(StreamsToTensor, evtIn, in, out, n_ports, size, filter, n_pre, holdoff_samples, _eventsRefused);

    struct GatedState {
        bool                                             active = false; /// at start(): a finished event source disconnects evtIn
        gr::blocks::trigger::TimeBase                    time;
        gr::blocks::trigger::EventStore                  events;
        gr::blocks::trigger::detail::MatchState          opens{};
        bool                                             filtered = false;
        std::vector<std::vector<T>>                      held; /// per input, the samples from `heldStart` on
        std::size_t                                      heldStart = 0UZ;
        std::size_t                                      deafUntil = 0UZ;
        std::deque<std::pair<std::size_t, property_map>> pending; /// frames to emit: first sample, the event that opened it
    };
    GatedState _gated;

    std::size_t               _tagsForwardedUntilSample = 0UZ;
    std::vector<property_map> _tagsOfSkippedSamples;
    float                     _sampleRate = 0.f; // [Hz]; 0: not yet known

    void start() {
        resizePorts();
        _tagsForwardedUntilSample = 0UZ;
        _tagsOfSkippedSamples.clear();
        _sampleRate     = 0.f;
        _eventsRefused  = 0U;
        _gated          = GatedState{};
        _gated.active   = evtIn.isConnected();
        _gated.filtered = gr::blocks::trigger::detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _gated.opens);
        if (_gated.active) {
            this->input_chunk_size  = 1U;
            this->output_chunk_size = 1U;
        }
    }

    template<InputSpanLike TIn>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& evtSpan, const std::span<TIn>& inputs, OutputSpanLike auto& output) {
        if (_gated.active) {
            return gatedFrames(evtSpan, inputs, output);
        }
        const std::size_t hop      = this->resamplingHop();
        const std::size_t nFitting = detail::completeWindows(*this, shortestInput(inputs), output.size());
        std::size_t       nFrames  = 0UZ;
        for (; nFrames < nFitting; ++nFrames) {
            const auto frame = reserveFrame(output, nFrames, inputs.size());
            if (!frame) {
                break;
            }
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                std::copy_n(inputs[port].data() + nFrames * hop, size.value, frame->data() + port * size.value);
            }
        }
        return finishFrames(inputs, output, nFrames, nFrames * hop);
    }

    template<InputSpanLike TIn>
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& /*evtSpan*/, std::vector<TIn>& inputs, OutputSpanLike auto& output) {
        if (_gated.active) [[unlikely]] {
            this->emitErrorMessage("processBulk()", "gated framing (evtIn connected) runs on the host only");
            this->requestStop();
            return gr::work::Status::ERROR;
        }
        const std::size_t n        = size.value;
        const std::size_t hop      = this->resamplingHop();
        const std::size_t ports    = inputs.size();
        const std::size_t nFitting = detail::completeWindows(*this, shortestInput(inputs), output.size());
        std::size_t       nFrames  = 0UZ;
        while (nFrames < nFitting && reserveFrame(output, nFrames, ports)) {
            ++nFrames;
        }
        const std::array<const T*, detail::kMaxPorts> sources = detail::dataOf<const T*>(inputs);
        gr::device::parallelFor(ctx, nFrames * ports, [sources, frames = output.data(), n, hop, ports] GR_DEVICE_LAMBDA(std::size_t i) {
            const std::size_t frame = i / ports;
            const std::size_t port  = i % ports;
            for (std::size_t j = 0UZ; j < n; ++j) {
                frames[frame].data()[port * n + j] = sources[port][frame * hop + j];
            }
        });
        return finishFrames(std::span(inputs), output, nFrames, nFrames * hop);
    }

    template<InputSpanLike TIn>
    gr::work::Status processEpilogue(InputSpanLike auto& /*evtSpan*/, const std::span<TIn>& inputs, OutputSpanLike auto& output) {
        const std::size_t available = shortestInput(inputs);
        if (_gated.active || available < size.value || output.size() == 0UZ || !output.blobsHostAddressable()) {
            return gr::work::Status::OK;
        }
        const auto frame = reserveFrame(output, 0UZ, inputs.size());
        if (!frame) {
            return gr::work::Status::OK;
        }
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            std::copy_n(inputs[port].data(), size.value, frame->data() + port * size.value);
        }
        return finishFrames(inputs, output, 1UZ, std::numeric_limits<std::size_t>::max());
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        detail::refuseNewPortCount<nPorts>(*this, newSettings, in);
        resizePorts();
        if (newSettings.contains("stride") && this->stride.value > detail::kMaxRunLength) {
            const gr::Size_t previous = oldSettings.value_or<gr::Size_t>("stride", 0U);
            this->emitErrorMessage("settingsChanged()", std::format("stride {} exceeds {}: keeps {}", this->stride.value, detail::kMaxRunLength, previous));
            this->stride = previous;
        }
        if (newSettings.contains("filter")) {
            _gated.filtered = gr::blocks::trigger::detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _gated.opens);
        }
        const std::size_t chunk = _gated.active || evtIn.isConnected() ? 1UZ : std::max<std::size_t>(size.value, this->stride.value);
        const std::string what  = std::format("size {} or stride {}", size.value, this->stride.value);
        if (!std::ranges::all_of(in, [&](const auto& port) { return detail::chunkFitsBuffer(*this, port, chunk, what); })) {
            return;
        }
        this->input_chunk_size  = static_cast<gr::Size_t>(chunk);
        this->output_chunk_size = 1U;
    }

    /** frame `index`, uninitialised, of shape [M] or [M,N]; none, with the error reported and the graph stopped, when its storage is exhausted */
    [[nodiscard]] std::optional<std::span<T>> reserveFrame(auto& output, std::size_t index, std::size_t ports) {
        const std::array<std::size_t, 1> rowExtents{ports};
        const std::array<std::size_t, 2> frameExtents{ports, size.value};
        const auto                       extents = size.value == 1U ? std::span<const std::size_t>(rowExtents) : std::span<const std::size_t>(frameExtents);
        auto                             frame   = gr::emit(*this, output, index, extents);
        if (!frame) [[unlikely]] {
            return std::nullopt;
        }
        return *frame;
    }

    /** forwards the frame tags of every input in frame order, the same tag on several inputs once, then consumes and publishes */
    template<InputSpanLike TIn>
    gr::work::Status finishFrames(const std::span<TIn>& inputs, auto& output, std::size_t nFrames, std::size_t nConsumed) {
        std::vector<std::pair<std::size_t, property_map>> tags;
        if (nFrames > 0UZ) {
            for (property_map& held : _tagsOfSkippedSamples) {
                tags.emplace_back(0UZ, std::move(held));
            }
            _tagsOfSkippedSamples.clear();
        }
        const std::size_t limit        = nConsumed == std::numeric_limits<std::size_t>::max() ? 0UZ : nConsumed;
        const std::size_t hop          = this->resamplingHop();
        const auto        datedToFrame = [&](property_map& tag, std::size_t frame, std::ptrdiff_t relativeIndex) {
            const float* const wireRate = tag.get_if<float>(gr::tag::kSampleRate.key());
            const float* const rate     = wireRate != nullptr ? wireRate : tag.get_if<float>(gr::tag::kSampleRate.shortKey());
            if (rate != nullptr) {
                _sampleRate = *rate;
            }
            if (const std::uint64_t* const sampleTimeNs = tag.get_if<std::uint64_t>(gr::tag::kTriggerTime.key())) {
                const std::ptrdiff_t samplesAfterFrameStart = relativeIndex - static_cast<std::ptrdiff_t>(frame * hop);
                const std::uint64_t  frameTimeNs            = timeOfFrameStart(*sampleTimeNs, samplesAfterFrameStart, _sampleRate);
                tag.insert_or_assign(gr::tag::kTriggerTime.key(), frameTimeNs);
            }
        };
        for (auto& input : inputs) {
            detail::forEachWindowTag(input, nFrames, limit, size.value, hop, _tagsForwardedUntilSample, [&](std::size_t frame, std::ptrdiff_t relativeIndex, const auto& tagMap) { //
                this->forwardedTag(tagMap, [&](const auto& forwarded) { datedToFrame(tags.emplace_back(frame, property_map(forwarded)).second, frame, relativeIndex); });
            });
        }
        std::ranges::stable_sort(tags, {}, &std::pair<std::size_t, property_map>::first);
        auto outputs = std::tie(output);
        for (std::size_t i = 0UZ; i < tags.size(); ++i) {
            const bool repeated = std::ranges::any_of(tags | std::views::take(i), [&](const auto& earlier) { return earlier.first == tags[i].first && earlier.second == tags[i].second; });
            if (repeated) {
                continue;
            }
            if (tags[i].first < nFrames) {
                this->publishTagToSingleProducerOutputs(outputs, tags[i].second, tags[i].first);
            } else {
                _tagsOfSkippedSamples.push_back(tags[i].second);
            }
        }
        if (nFrames > 0UZ) {
            _tagsForwardedUntilSample = detail::coveredAfter(inputs[0].streamIndex, nFrames, size.value, hop);
        }
        for (auto& input : inputs) {
            std::ignore = input.consume(std::min(nConsumed, input.size()));
        }
        output.publish(nFrames);
        return nFrames == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    /** the gated path: holds the inputs' recent samples, places the events on them and emits a frame once its samples are all held */
    template<InputSpanLike TIn>
    gr::work::Status gatedFrames(InputSpanLike auto& evtSpan, const std::span<TIn>& inputs, OutputSpanLike auto& output) {
        GatedState&       state       = _gated;
        const std::size_t n           = shortestInput(inputs);
        const std::size_t streamIndex = inputs[0].streamIndex;
        if (state.held.size() != inputs.size()) {
            state.held.assign(inputs.size(), {});
            state.heldStart = streamIndex;
        }
        for (const auto& [relativeIndex, tagMap] : inputs[0].tags()) {
            std::ignore = state.time.adopt(tagMap.get(), streamIndex + (relativeIndex < 0 ? 0UZ : static_cast<std::size_t>(relativeIndex)));
        }
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            state.held[port].insert(state.held[port].end(), inputs[port].begin(), std::next(inputs[port].begin(), static_cast<std::ptrdiff_t>(n)));
        }
        const std::size_t heldEnd = state.heldStart + state.held[0].size();

        state.events.drain(evtSpan);
        const std::size_t ready   = state.events.ready();
        std::size_t       handled = 0UZ;
        for (const gr::blocks::trigger::StoredEvent& stored : state.events.ordered() | std::views::take(ready)) {
            const property_map_view          event{stored.event};
            const std::optional<std::size_t> at = stored.dated() ? state.time.indexAt(*stored.at) : std::nullopt;
            if (at && *at >= heldEnd) {
                break; // its sample has not arrived yet
            }
            ++handled;
            if (!gr::blocks::trigger::detail::acceptsTrigger(state.opens, state.filtered, event)) {
                continue;
            }
            const std::size_t start = at ? *at - std::min<std::size_t>(*at, n_pre.value) : 0UZ;
            if (!at || start < state.heldStart) {
                _eventsRefused = _eventsRefused + 1U;
                continue;
            }
            if (*at < state.deafUntil) {
                continue;
            }
            state.deafUntil = *at + std::max<std::size_t>(holdoff_samples.value, 1UZ);
            property_map frameTag(stored.event);
            if (const std::optional<std::uint64_t> firstSampleNs = state.time.at(start)) {
                frameTag.insert_or_assign(std::string(gr::tag::kTriggerTime.key()), *firstSampleNs);
                frameTag.insert_or_assign(std::string(gr::tag::kTriggerOffset.key()), 0.f);
            }
            frameTag.insert_or_assign(std::string(gr::tag::kSampleRate.key()), static_cast<float>(state.time.rate()));
            state.pending.emplace_back(start, std::move(frameTag));
        }
        state.events.retire(handled);

        std::size_t nFrames = 0UZ;
        auto        outputs = std::tie(output);
        while (!state.pending.empty() && nFrames < output.size() && state.pending.front().first + size.value <= heldEnd) {
            const auto frame = reserveFrame(output, nFrames, inputs.size());
            if (!frame) {
                break;
            }
            const std::size_t offset = state.pending.front().first - state.heldStart;
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                std::copy_n(state.held[port].begin() + static_cast<std::ptrdiff_t>(offset), size.value, frame->data() + port * size.value);
            }
            this->publishTagToSingleProducerOutputs(outputs, state.pending.front().second, nFrames);
            state.pending.pop_front();
            ++nFrames;
        }

        const std::size_t lateSamples = std::max(kLateSamples, static_cast<std::size_t>(std::llround(kLateSeconds * state.time.rate())));
        const std::size_t lateReach   = heldEnd - std::min(heldEnd - state.heldStart, lateSamples + static_cast<std::size_t>(n_pre.value));
        const std::size_t keepFrom    = state.pending.empty() ? lateReach : std::min(lateReach, state.pending.front().first);
        for (std::vector<T>& held : state.held) {
            held.erase(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(keepFrom - state.heldStart));
        }
        state.heldStart = keepFrom;
        for (auto& input : inputs) {
            std::ignore = input.consume(n);
        }
        output.publish(nFrames);
        return n == 0UZ && nFrames == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    [[nodiscard]] static std::size_t shortestInput(const auto& inputs) noexcept {
        return std::ranges::min(inputs | std::views::transform([](const auto& input) { return input.size(); }));
    }

    void resizePorts() {
        if constexpr (nPorts == 0UZ) {
            in.resize(n_ports.value);
        }
    }
};

GR_REGISTER_BLOCK(gr::basic::TensorToStreams, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (nPorts <= detail::kMaxPorts)
struct TensorToStreams : Block<TensorToStreams<T, nPorts>, NoTagPropagation, Resampling<1U, 1U, false>> {
    using Description = Doc<R""(@brief row p of each tensor to output p (GR3: Vector to Streams): a tensor of shape [K] gives one sample per output,
one of shape [K,N] N samples per output, at the signal's `gr:sample_rate`; N may change from tensor to tensor.

A tensor of rank other than 1 or 2, with K other than the number of outputs, or wider than an output buffer stops the stream with an error,
after the tensors before it; the output chunk grows to the widest tensor seen.
A tag on a tensor goes to every output at the tensor's first sample, each output being one channel of the same moment;
its `frame_rate` is dropped, since the outputs are streams again.
`nPorts = 0` has `n_ports` outputs, fixed once connected.)"">;

    struct Fit {
        std::size_t nIn;
        std::size_t nOut;
        bool        stop;
    };

    TensorPortIn<T>                     in;
    detail::PortsOf<PortOut<T>, nPorts> out;

    Annotated<gr::Size_t, "n_ports", Doc<"outputs; nPorts when that is not 0">, Limits<1U, static_cast<gr::Size_t>(detail::kMaxPorts)>> n_ports = nPorts == 0UZ ? 2U : static_cast<gr::Size_t>(nPorts);

    GR_MAKE_REFLECTABLE(TensorToStreams, in, out, n_ports);

    void start() { resizePorts(); }

    template<OutputSpanLike TOut>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, const std::span<TOut>& outputs) {
        const Fit   fit    = fittingTensors(input, outputs);
        std::size_t offset = 0UZ;
        for (std::size_t k = 0UZ; k < fit.nIn; ++k) {
            const std::size_t n = samplesPerOutput(input[k]);
            for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                std::copy_n(input[k].data() + port * n, n, outputs[port].data() + offset);
            }
            offset += n;
        }
        return consumeAndPublish(input, outputs, fit);
    }

    template<OutputSpanLike TOut>
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, std::vector<TOut>& outputs) {
        const Fit                               fit   = fittingTensors(input, outputs);
        const std::array<T*, detail::kMaxPorts> sinks = detail::dataOf<T*>(outputs);
        gr::device::parallelFor(ctx, outputs.size(), [sinks, tensors = input.data(), nIn = fit.nIn] GR_DEVICE_LAMBDA(std::size_t port) {
            std::size_t offset = 0UZ;
            for (std::size_t k = 0UZ; k < nIn; ++k) {
                const std::size_t n = samplesPerOutput(tensors[k]);
                for (std::size_t j = 0UZ; j < n; ++j) {
                    sinks[port][offset + j] = tensors[k].data()[port * n + j];
                }
                offset += n;
            }
        });
        return consumeAndPublish(input, std::span(outputs), fit);
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        detail::refuseNewPortCount<nPorts>(*this, newSettings, out);
        resizePorts();
    }

    [[nodiscard]] static constexpr std::size_t samplesPerOutput(const TensorView<T>& tensor) noexcept { return tensor.rank() == 2UZ ? tensor.extent(1UZ) : 1UZ; }

    [[nodiscard]] bool takesShape(const TensorView<T>& tensor, std::size_t ports) const noexcept {
        const bool rank1or2        = tensor.rank() == 1UZ || tensor.rank() == 2UZ;
        const auto narrowestBuffer = std::ranges::min(out | std::views::transform([](const auto& port) { return port.bufferSize(); }));
        return rank1or2 && tensor.extent(0UZ) == ports && samplesPerOutput(tensor) <= narrowestBuffer;
    }

    [[nodiscard]] Fit fittingTensors(auto& input, const auto& outputs) const {
        const std::size_t space = std::ranges::min(outputs | std::views::transform([](const auto& output) { return output.size(); }));
        Fit               fit{0UZ, 0UZ, false};
        for (; fit.nIn < input.size(); ++fit.nIn) {
            fit.stop = !takesShape(input[fit.nIn], outputs.size());
            if (fit.stop || fit.nOut + samplesPerOutput(input[fit.nIn]) > space) {
                break;
            }
            fit.nOut += samplesPerOutput(input[fit.nIn]);
        }
        return fit;
    }

    /** forwards the tags of the tensors that went out, consumes and publishes them; a tensor that did not fit grows the output chunk for the next call */
    gr::work::Status consumeAndPublish(auto& input, auto outputs, const Fit& fit) {
        const std::string_view frameRateKey{gr::tag::kFrameRate.key()};
        const auto             publishToEveryOutput = [&](const auto& tag) {
            for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                auto portOutput = std::tie(outputs[port]);
                this->publishTagToSingleProducerOutputs(portOutput, tag, 0UZ);
            }
        };
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            if (relativeIndex != 0 || fit.nIn == 0UZ) {
                continue;
            }
            const auto& tag      = tagMap.get();
            const auto  ofTensor = [&](std::string_view key) { return key == frameRateKey; };
            if (std::ranges::any_of(tag.keys(), ofTensor)) {
                property_map unframed;
                for (const auto& [key, value] : tag) {
                    if (!ofTensor(key)) {
                        unframed.insert_or_assign(key, value);
                    }
                }
                publishToEveryOutput(unframed);
            } else {
                publishToEveryOutput(tag);
            }
        }
        std::ignore = input.consume(fit.nIn);
        for (auto& output : outputs) {
            output.publish(fit.nOut);
        }
        if (fit.stop) [[unlikely]] {
            const TensorView<T>& tensor = input[fit.nIn];
            this->emitErrorMessage("processBulk()", std::format("tensor {} has shape {}, which none of the {} outputs of buffer {} takes", input.streamIndex + fit.nIn, tensor.extents(), outputs.size(), out[0].bufferSize()));
            this->requestStop();
            return gr::work::Status::OK;
        }
        if (fit.nIn < input.size() && samplesPerOutput(input[fit.nIn]) > this->output_chunk_size) {
            this->output_chunk_size = static_cast<gr::Size_t>(samplesPerOutput(input[fit.nIn]));
        }
        if (fit.nIn > 0UZ) {
            return gr::work::Status::OK;
        }
        return input.size() == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
    }

    void resizePorts() {
        if constexpr (nPorts == 0UZ) {
            out.resize(n_ports.value);
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_STREAM_TO_TENSOR_HPP
