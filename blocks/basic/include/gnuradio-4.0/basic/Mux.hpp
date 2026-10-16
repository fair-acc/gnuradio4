#ifndef GNURADIO_BASIC_MUX_HPP
#define GNURADIO_BASIC_MUX_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::basic {

namespace detail {
[[nodiscard]] inline bool validLengths(const Tensor<gr::Size_t>& lengths, std::size_t nPorts) {
    const bool sized = nPorts == 0UZ ? !lengths.empty() && lengths.size() <= kMaxPorts : lengths.size() == nPorts;
    return sized && std::ranges::all_of(lengths, [](gr::Size_t length) { return length >= 1U && length <= kMaxRunLength; });
}

[[nodiscard]] inline Tensor<gr::Size_t> unitRuns(std::size_t nPorts) { return Tensor<gr::Size_t>(gr::data_from, std::vector<gr::Size_t>(nPorts == 0UZ ? 2UZ : nPorts, 1U)); }

/** true when `lengths` holds one run of 1..kMaxRunLength per port and, once a port is connected, as many as there are ports; else the previous setting is kept */
template<std::size_t nPorts>
[[nodiscard]] bool acceptLengths(auto& block, const auto& ports, const property_map& oldSettings, std::string_view what) {
    const bool connected = std::ranges::any_of(ports, [](const auto& port) { return port.isConnected(); });
    if (validLengths(block.lengths.value, nPorts) && !(connected && block.lengths.value.size() != ports.size())) {
        return true;
    }
    const std::span refused(block.lengths.value.data(), block.lengths.value.size());
    block.emitErrorMessage("settingsChanged()", std::format("lengths {} refused: one run of 1..{} per {}, fixed once connected", refused, kMaxRunLength, what));
    block.lengths = oldSettings.value_or<Tensor<gr::Size_t>>("lengths", unitRuns(nPorts));
    return false;
}

/** `forwarded` with its `sample_rate` replaced by `rate`, or without it when `rate` is 0; a mux or demux announces the rate of each output itself */
template<typename TBlock>
void publishWithRate(auto& outputSpans, const auto& forwarded, std::size_t offset, float rate) {
    const std::string_view rateKey{gr::tag::SAMPLE_RATE.key()};
    if (!forwarded.contains(rateKey)) {
        TBlock::publishTagToSingleProducerOutputs(outputSpans, forwarded, offset);
        return;
    }
    property_map withRate(forwarded);
    if (rate > 0.f) {
        withRate.insert_or_assign(std::string(rateKey), rate);
    } else {
        withRate.erase(rateKey);
    }
    if (!withRate.empty()) {
        TBlock::publishTagToSingleProducerOutputs(outputSpans, withRate, offset);
    }
}
} // namespace detail

/** one turn of a mux: run p, of port p, starts at runStart[p]; runStart[nRuns] is the turn's length */
struct TurnLayout {
    std::array<std::size_t, detail::kMaxPorts + 1UZ> runStart{};
    std::size_t                                      nRuns = 0UZ;

    [[nodiscard]] constexpr std::size_t length(std::size_t port) const noexcept { return runStart[port + 1UZ] - runStart[port]; }
    [[nodiscard]] constexpr std::size_t turn() const noexcept { return runStart[nRuns]; }

    /** where sample k of port `port` goes in the muxed stream */
    [[nodiscard]] constexpr std::size_t muxedIndex(std::size_t port, std::size_t k) const noexcept { return k / length(port) * turn() + runStart[port] + k % length(port); }

    [[nodiscard]] constexpr std::size_t portOf(std::size_t i) const noexcept {
        const std::size_t inTurn = i % turn();
        std::size_t       port   = 0UZ;
        while (runStart[port + 1UZ] <= inTurn) {
            ++port;
        }
        return port;
    }

    /** the index of muxed sample i in its port, portOf(i): the inverse of `muxedIndex` */
    [[nodiscard]] constexpr std::size_t indexInPort(std::size_t i, std::size_t port) const noexcept { return i / turn() * length(port) + i % turn() - runStart[port]; }

    /** how many of port `port`'s samples come before muxed sample i */
    [[nodiscard]] constexpr std::size_t portSamplesBefore(std::size_t i, std::size_t port) const noexcept {
        const std::size_t inTurn = i % turn();
        const std::size_t inRun  = inTurn < runStart[port] ? 0UZ : (inTurn < runStart[port + 1UZ] ? inTurn - runStart[port] : length(port));
        return i / turn() * length(port) + inRun;
    }
};

[[nodiscard]] constexpr TurnLayout turnLayoutOf(const auto& lengths) noexcept {
    TurnLayout layout;
    for (const auto length : lengths) {
        layout.runStart[layout.nRuns + 1UZ] = layout.runStart[layout.nRuns] + static_cast<std::size_t>(length);
        ++layout.nRuns;
    }
    return layout;
}

GR_REGISTER_BLOCK(gr::basic::Mux, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (nPorts <= detail::kMaxPorts)
struct Mux : Block<Mux<T, nPorts>, NoTagPropagation> {
    using Description = Doc<R""(@brief takes turns of `lengths[p]` samples from each input p (GR3: Interleave, Stream Mux);
equal lengths interleave, unequal ones put, for example, a header before each payload.

The inputs run at rates in proportion to their lengths; the output's is input 0's times sum(lengths) / lengths[0].
A tag lands on its sample's place in the output, its `sample_rate` replaced by the output's (input 0) or dropped (other inputs).
Only whole turns go out: the stream ends when an ended input cannot fill its next run.)"">;

    detail::PortsOf<PortIn<T, Async, Optional>, nPorts> in;
    PortOut<T>                                          out;

    Annotated<Tensor<gr::Size_t>, "lengths", Doc<"samples per run of each input, 1..2^24; sets the inputs when nPorts is 0">> lengths = detail::unitRuns(nPorts);

    GR_MAKE_REFLECTABLE(Mux, in, out, lengths);

    /** input 0's; 0: unknown */
    float _inputSampleRate = 0.f;
    bool  _announceRate    = false;

    void start() {
        applyLengths();
        _inputSampleRate = 0.f;
        if (std::ranges::any_of(in, [](const auto& port) { return !port.isConnected(); })) {
            this->emitErrorMessage("start()", "every input needs a connection: an unconnected one would stall its run forever");
            this->requestStop();
        }
    }

    template<InputSpanLike TIn>
    [[nodiscard]] gr::work::Status processBulk(const std::span<TIn>& inputs, OutputSpanLike auto& output) {
        const TurnLayout  layout = turnLayoutOf(lengths.value);
        const std::size_t nTurns = wholeTurns(inputs, output, layout);
        for (std::size_t turn = 0UZ; turn < nTurns; ++turn) {
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                std::copy_n(inputs[port].data() + turn * layout.length(port), layout.length(port), output.data() + turn * layout.turn() + layout.runStart[port]);
            }
        }
        return publishTurns(inputs, output, layout, nTurns);
    }

    template<InputSpanLike TIn>
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, std::vector<TIn>& inputs, OutputSpanLike auto& output) {
        const std::array<const T*, detail::kMaxPorts> sources = detail::dataOf<const T*>(inputs);
        const TurnLayout                              layout  = turnLayoutOf(lengths.value);
        const std::size_t                             nTurns  = wholeTurns(inputs, output, layout);
        gr::device::parallelFor(ctx, nTurns * layout.turn(), [sources, muxed = output.data(), layout] GR_DEVICE_LAMBDA(std::size_t i) {
            const std::size_t port = layout.portOf(i);
            muxed[i]               = sources[port][layout.indexInPort(i, port)];
        });
        return publishTurns(inputs, output, layout, nTurns);
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("lengths") && detail::acceptLengths<nPorts>(*this, in, oldSettings, "input")) {
            const std::size_t turn = turnLayoutOf(lengths.value).turn();
            if (detail::chunkFitsBuffer(*this, out, turn, std::format("a turn of {} samples", turn))) {
                applyLengths();
                _announceRate = _inputSampleRate > 0.f;
            }
        }
    }

    [[nodiscard]] static std::size_t wholeTurns(const auto& inputs, const auto& output, const TurnLayout& layout) noexcept {
        std::size_t nTurns = output.size() / layout.turn();
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            nTurns = std::min(nTurns, inputs[port].size() / layout.length(port));
        }
        return nTurns;
    }

    [[nodiscard]] gr::work::Status publishTurns(auto& inputs, OutputSpanLike auto& output, const TurnLayout& layout, std::size_t nTurns) {
        const double rateFactor = static_cast<double>(layout.turn()) / static_cast<double>(layout.length(0UZ));
        const auto   outputRate = [rateFactor](float inputRate) { return static_cast<float>(static_cast<double>(inputRate) * rateFactor); };
        if (_announceRate) {
            output.publishTag(detail::sampleRateTag(outputRate(_inputSampleRate)), 0UZ);
            _announceRate = false;
        }
        auto                                       outputs = std::tie(output);
        std::array<std::size_t, detail::kMaxPorts> tagCursor{};
        while (true) { // the inputs' tags merged in output order
            std::size_t next      = inputs.size();
            std::size_t nextIndex = std::numeric_limits<std::size_t>::max();
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                auto tags = inputs[port].tags();
                for (; tagCursor[port] < std::ranges::size(tags); ++tagCursor[port]) {
                    const auto [relativeIndex, tagMap] = tags[static_cast<std::ptrdiff_t>(tagCursor[port])];
                    if (relativeIndex >= 0) {
                        const auto k = static_cast<std::size_t>(relativeIndex);
                        if (k < nTurns * layout.length(port) && layout.muxedIndex(port, k) < nextIndex) {
                            next      = port;
                            nextIndex = layout.muxedIndex(port, k);
                        }
                        break;
                    }
                }
            }
            if (next == inputs.size()) {
                break;
            }
            const auto [relativeIndex, tagMap] = inputs[next].tags()[static_cast<std::ptrdiff_t>(tagCursor[next]++)];
            this->forwardedTag(tagMap.get(), [&](const auto& forwarded) {
                if (next == 0UZ && forwarded.contains(std::string_view{gr::tag::SAMPLE_RATE.key()})) {
                    _inputSampleRate = forwarded.template value_or<float>(gr::tag::SAMPLE_RATE.key(), 0.f);
                }
                detail::publishWithRate<Mux>(outputs, forwarded, nextIndex, next == 0UZ ? outputRate(_inputSampleRate) : 0.f);
            });
        }
        bool ended = false;
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            const std::size_t taken = nTurns * layout.length(port);
            std::ignore             = inputs[port].consume(taken);
            ended                   = ended || (inputs[port].size() - taken < layout.length(port) && gr::endsStreamAt(in[port], inputs[port].streamIndex + inputs[port].size()));
        }
        output.publish(nTurns * layout.turn());
        if (ended) {
            return gr::work::Status::DONE;
        }
        return nTurns == 0UZ ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::OK;
    }

    void applyLengths() {
        if constexpr (nPorts == 0UZ) {
            in.resize(lengths.value.size());
        }
        out.min_samples = turnLayoutOf(lengths.value).turn();
    }
};

GR_REGISTER_BLOCK(gr::basic::Demux, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (nPorts <= detail::kMaxPorts)
struct Demux : Block<Demux<T, nPorts>, NoTagPropagation> {
    using Description = Doc<R""(@brief deals turns of `lengths[p]` samples to each output p (GR3: Deinterleave, Stream Demux);
equal lengths deinterleave, unequal ones split, for example, a header from each payload.

Output p's rate is the input's times lengths[p] / sum(lengths).
A tag goes with its sample; its `sample_rate`, replaced by each output's, goes to all of them.
Only whole turns are dealt: a partial turn at the end of the stream stays behind.)"">;

    PortIn<T>                           in;
    detail::PortsOf<PortOut<T>, nPorts> out;

    Annotated<Tensor<gr::Size_t>, "lengths", Doc<"samples per run to each output, 1..2^24; sets the outputs when nPorts is 0">> lengths = detail::unitRuns(nPorts);

    GR_MAKE_REFLECTABLE(Demux, in, out, lengths);

    /** 0: unknown */
    float _inputSampleRate = 0.f;
    bool  _announceRate    = false;

    void start() {
        applyLengths();
        _inputSampleRate = 0.f;
    }

    template<OutputSpanLike TOut>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, const std::span<TOut>& outputs) {
        const TurnLayout  layout = turnLayoutOf(lengths.value);
        const std::size_t nTurns = wholeTurns(input, outputs, layout);
        for (std::size_t turn = 0UZ; turn < nTurns; ++turn) {
            for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                std::copy_n(input.data() + turn * layout.turn() + layout.runStart[port], layout.length(port), outputs[port].data() + turn * layout.length(port));
            }
        }
        return publishTurns(input, outputs, layout, nTurns);
    }

    template<OutputSpanLike TOut>
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, std::vector<TOut>& outputs) {
        const std::array<T*, detail::kMaxPorts> sinks  = detail::dataOf<T*>(outputs);
        const TurnLayout                        layout = turnLayoutOf(lengths.value);
        const std::size_t                       nTurns = wholeTurns(input, outputs, layout);
        gr::device::parallelFor(ctx, nTurns * layout.turn(), [sinks, muxed = input.data(), layout] GR_DEVICE_LAMBDA(std::size_t i) {
            const std::size_t port                   = layout.portOf(i);
            sinks[port][layout.indexInPort(i, port)] = muxed[i];
        });
        return publishTurns(input, outputs, layout, nTurns);
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("lengths") && detail::acceptLengths<nPorts>(*this, out, oldSettings, "output")) {
            const std::size_t turn = turnLayoutOf(lengths.value).turn();
            if (detail::chunkFitsBuffer(*this, in, turn, std::format("a turn of {} samples", turn))) {
                applyLengths();
                _announceRate = _inputSampleRate > 0.f;
            }
        }
    }

    [[nodiscard]] static std::size_t wholeTurns(const auto& input, const auto& outputs, const TurnLayout& layout) noexcept {
        std::size_t nTurns = input.size() / layout.turn();
        for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
            nTurns = std::min(nTurns, outputs[port].size() / layout.length(port));
        }
        return nTurns;
    }

    [[nodiscard]] gr::work::Status publishTurns(InputSpanLike auto& input, auto& outputs, const TurnLayout& layout, std::size_t nTurns) {
        const double runRate    = 1. / static_cast<double>(layout.turn());
        const auto   outputRate = [&](std::size_t port) { return static_cast<float>(static_cast<double>(_inputSampleRate) * runRate * static_cast<double>(layout.length(port))); };
        if (_announceRate) {
            for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                outputs[port].publishTag(detail::sampleRateTag(outputRate(port)), 0UZ);
            }
            _announceRate = false;
        }
        const std::size_t taken = nTurns * layout.turn();
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            const auto i = static_cast<std::size_t>(relativeIndex);
            if (relativeIndex < 0 || i >= taken) {
                continue;
            }
            const std::size_t samplePort = layout.portOf(i);
            this->forwardedTag(tagMap.get(), [&](const auto& forwarded) {
                if (!forwarded.contains(std::string_view{gr::tag::SAMPLE_RATE.key()})) {
                    auto sampleOutput = std::tie(outputs[samplePort]);
                    detail::publishWithRate<Demux>(sampleOutput, forwarded, layout.portSamplesBefore(i, samplePort), 0.f);
                    return;
                }
                _inputSampleRate = forwarded.template value_or<float>(gr::tag::SAMPLE_RATE.key(), 0.f);
                for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                    const std::size_t offset = layout.portSamplesBefore(i, port);
                    if (port == samplePort) {
                        auto sampleOutput = std::tie(outputs[port]);
                        detail::publishWithRate<Demux>(sampleOutput, forwarded, offset, outputRate(port));
                    } else {
                        outputs[port].publishTag(detail::sampleRateTag(outputRate(port)), offset);
                    }
                }
            });
        }
        std::ignore = input.consume(taken);
        for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
            outputs[port].publish(nTurns * layout.length(port));
        }
        if (input.size() - taken < layout.turn() && gr::endsStreamAt(in, input.streamIndex + input.size())) {
            return gr::work::Status::DONE;
        }
        return nTurns == 0UZ ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }

    void applyLengths() {
        if constexpr (nPorts == 0UZ) {
            out.resize(lengths.value.size());
        }
        in.min_samples = turnLayoutOf(lengths.value).turn(); // else the framework cuts the input at a tag short of a turn
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_MUX_HPP
