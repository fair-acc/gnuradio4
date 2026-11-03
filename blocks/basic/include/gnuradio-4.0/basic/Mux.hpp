#ifndef GNURADIO_BASIC_MUX_HPP
#define GNURADIO_BASIC_MUX_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::basic {

namespace detail {
template<typename T>
inline constexpr bool kIsEvent = std::same_as<T, property_map> || std::same_as<T, property_map_view>;

template<typename T>
using MuxIn = std::conditional_t<kIsEvent<T>, EventPortIn, PortIn<T, Async, Optional>>;
template<typename T>
using DemuxIn = std::conditional_t<kIsEvent<T>, EventPortIn, PortIn<T>>;
template<typename T>
using MuxOut = std::conditional_t<kIsEvent<T>, EventPortOut, PortOut<T>>;

/** an event output claims this many slots per work call, so it is the largest turn of events */
inline constexpr PortConfig kEventPortConfig{.streamSlotsPerPublish = 64UZ};

template<typename TPort, std::size_t nPorts>
[[nodiscard]] PortsOf<TPort, nPorts> configuredPorts() {
    if constexpr (nPorts == 0UZ) {
        return {};
    } else {
        return [&]<std::size_t... index>(std::index_sequence<index...>) { return PortsOf<TPort, nPorts>{((void)index, TPort(kEventPortConfig))...}; }(std::make_index_sequence<nPorts>());
    }
}

void resizePorts(auto& ports, std::size_t nPorts) {
    while (ports.size() > nPorts) {
        ports.pop_back();
    }
    while (ports.size() < nPorts) {
        ports.emplace_back(kEventPortConfig);
    }
}

/** an empty event is padding: a writer of a shared event ring committed slots it claimed and did not fill; a turn counts only the others */
inline constexpr auto isEvent = [](const property_map_view& event) { return !event.empty(); };

template<typename TSpan>
inline constexpr bool kHoldsEvents = std::same_as<typename std::remove_cvref_t<TSpan>::value_type, property_map_view>;

[[nodiscard]] std::size_t itemsBefore(const auto& span, std::size_t end) {
    if constexpr (kHoldsEvents<decltype(span)>) {
        return static_cast<std::size_t>(std::ranges::count_if(span | std::views::take(end), isEvent));
    } else {
        return end;
    }
}

[[nodiscard]] std::size_t itemCount(const auto& span) { return itemsBefore(span, span.size()); }

/** whether the stream on `port` has ended: an end-of-stream tag at the end of `span`, or, on a multi-producer event input, no writer left */
[[nodiscard]] bool streamEnded(auto& port, const auto& span) {
    if constexpr (std::remove_cvref_t<decltype(port)>::kIsMultiProducer) {
        return port.nWriters() == 0UZ;
    } else {
        return gr::endsStreamAt(port, span.streamIndex + span.size());
    }
}

/** how much of `span` to consume once its first `nItems` items are taken: up to the last of them */
[[nodiscard]] std::size_t spanIndexAfter(const auto& span, std::size_t nItems) {
    if constexpr (kHoldsEvents<decltype(span)>) {
        std::size_t index = 0UZ;
        for (std::size_t taken = 0UZ; taken < nItems; ++index) {
            taken += isEvent(span[index]) ? 1UZ : 0UZ;
        }
        return index;
    } else {
        return nItems;
    }
}

/** an event's payload goes into the output's own slot, as a view into the input dangles once consumed; false: refused, reported, and the block asked to stop */
[[nodiscard]] bool emitOrStop(auto& block, auto& outSpan, std::size_t index, const property_map_view& event) noexcept {
    if (const auto emitted = gr::emit(outSpan, index, event); !emitted) [[unlikely]] {
        gr::detail::stopOnError(block, index, emitted.error());
        return false;
    }
    return true;
}

[[nodiscard]] inline bool validLengths(const Tensor<gr::Size_t>& lengths, std::size_t nPorts) {
    const bool sized = nPorts == 0UZ ? !lengths.empty() && lengths.size() <= kMaxPorts : lengths.size() == nPorts;
    return sized && std::ranges::all_of(lengths, [](gr::Size_t length) { return length >= 1U && length <= kMaxRunLength; });
}

[[nodiscard]] inline Tensor<gr::Size_t> unitRuns(std::size_t nPorts) { return Tensor<gr::Size_t>(gr::data_from, std::vector<gr::Size_t>(nPorts == 0UZ ? 2UZ : nPorts, 1U)); }

/** true when `lengths` holds one run of 1..kMaxRunLength per port, a turn of events no longer than an event output's claim, and, once a port is connected, as many runs as there are ports; else the previous setting is kept */
template<typename T, std::size_t nPorts>
[[nodiscard]] bool acceptLengths(auto& block, const auto& ports, const property_map& oldSettings, std::string_view what) {
    const bool        connected = std::ranges::any_of(ports, [](const auto& port) { return port.isConnected(); });
    const std::size_t turn      = std::ranges::fold_left(block.lengths.value, 0UZ, std::plus{});
    const std::size_t maxTurn   = kIsEvent<T> ? kEventPortConfig.streamSlotsPerPublish : std::numeric_limits<std::size_t>::max();
    if (validLengths(block.lengths.value, nPorts) && turn <= maxTurn && !(connected && block.lengths.value.size() != ports.size())) {
        return true;
    }
    const std::span refused(block.lengths.value.data(), block.lengths.value.size());
    block.emitErrorMessage("settingsChanged()", std::format("lengths {} refused: one run of 1..{} per {}, a turn of at most {} events, fixed once connected", refused, kMaxRunLength, what, kEventPortConfig.streamSlotsPerPublish));
    block.lengths = oldSettings.value_or<Tensor<gr::Size_t>>("lengths", unitRuns(nPorts));
    return false;
}

/** `forwarded` with its `sample_rate` replaced by `rate`, or without it when `rate` is 0; a mux or demux announces the rate of each output itself */
void publishWithRate(auto& outSpan, const auto& forwarded, std::size_t offset, float rate) {
    const std::string_view rateKey{gr::tag::kSampleRate.key()};
    if (!forwarded.contains(rateKey)) {
        outSpan.publishTag(forwarded, offset);
        return;
    }
    property_map withRate(forwarded);
    if (rate > 0.f) {
        withRate.insert_or_assign(std::string(rateKey), rate);
    } else {
        withRate.erase(rateKey);
    }
    if (!withRate.empty()) {
        outSpan.publishTag(withRate, offset);
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

enum class Routing : std::uint8_t { turns, context };

GR_REGISTER_BLOCK(gr::basic::Mux, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double>, gr::property_map ])
GR_REGISTER_BLOCK(gr::basic::Mux, ([T], 0UZ, gr::basic::Routing::context), [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ, Routing routing = Routing::turns>
requires(std::is_arithmetic_v<T> || meta::complex_like<T> || detail::kIsEvent<T>) && (nPorts <= detail::kMaxPorts) && (routing == Routing::turns || nPorts == 0UZ)
struct Mux : Block<Mux<T, nPorts, routing>, NoTagPropagation> {
    using Description = Doc<R""(@brief takes turns of `lengths[p]` samples from each input p (GR3: Interleave, Stream Mux);
equal lengths interleave, unequal ones put, for example, a header before each payload.

The inputs run at rates in proportion to their lengths; the output's is input 0's times sum(lengths) / lengths[0].
A tag lands on its sample's place in the output, its `sample_rate` replaced by the output's (input 0) or dropped (other inputs).
Only whole turns go out: the stream ends when an ended input cannot fill its next run.
For `property_map` (or `property_map_view`) the ports are event ports and a turn counts events; host only.)"">;

    detail::PortsOf<detail::MuxIn<T>, nPorts> in;
    detail::MuxOut<T>                         out{detail::kEventPortConfig};

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
        if constexpr (detail::kIsEvent<T>) {
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                std::size_t k = 0UZ;
                for (const property_map_view& event : inputs[port] | std::views::filter(detail::isEvent) | std::views::take(nTurns * layout.length(port))) {
                    if (!detail::emitOrStop(*this, output, layout.muxedIndex(port, k++), event)) {
                        return gr::work::Status::ERROR;
                    }
                }
            }
        } else {
            for (std::size_t turn = 0UZ; turn < nTurns; ++turn) {
                for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                    std::copy_n(inputs[port].data() + turn * layout.length(port), layout.length(port), output.data() + turn * layout.turn() + layout.runStart[port]);
                }
            }
        }
        return publishTurns(inputs, output, layout, nTurns);
    }

    template<InputSpanLike TIn>
    requires(!detail::kIsEvent<T>)
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
        if (newSettings.contains("lengths") && detail::acceptLengths<T, nPorts>(*this, in, oldSettings, "input")) {
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
            nTurns = std::min(nTurns, detail::itemCount(inputs[port]) / layout.length(port));
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
        std::array<std::size_t, detail::kMaxPorts> tagCursor{};
        while (true) { // the inputs' tags merged in output order
            std::size_t next      = inputs.size();
            std::size_t nextIndex = std::numeric_limits<std::size_t>::max();
            for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
                auto tags = inputs[port].tags();
                for (; tagCursor[port] < std::ranges::size(tags); ++tagCursor[port]) {
                    const auto [relativeIndex, tagMap] = tags[static_cast<std::ptrdiff_t>(tagCursor[port])];
                    if (relativeIndex >= 0) {
                        const std::size_t k = detail::itemsBefore(inputs[port], static_cast<std::size_t>(relativeIndex));
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
                if (next == 0UZ && forwarded.contains(std::string_view{gr::tag::kSampleRate.key()})) {
                    _inputSampleRate = forwarded.template value_or<float>(gr::tag::kSampleRate.key(), 0.f);
                }
                detail::publishWithRate(output, forwarded, nextIndex, next == 0UZ ? outputRate(_inputSampleRate) : 0.f);
            });
        }
        bool ended = false;
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            const std::size_t taken = nTurns * layout.length(port);
            std::ignore             = inputs[port].consume(detail::spanIndexAfter(inputs[port], taken));
            ended                   = ended || (detail::itemCount(inputs[port]) - taken < layout.length(port) && detail::streamEnded(in[port], inputs[port]));
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

template<typename T>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>)
struct Mux<T, 0UZ, Routing::context> : gr::Block<Mux<T, 0UZ, Routing::context>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief forward the input its context names, and put the states back into one stream [merge, by context]

    in#0    ─a─a─a────────────▶    merge, but by context
    in#1    ────────b─b─b─────▶    contexts = RAMP, FLATTOP
    evtIn   ─RAMP───FLATTOP───▶
    out     ─a─a─a──b─b─b─────▶

 [1] example: https://rxmarbles.com/#merge
 [2] detailed documentation: https://reactivex.io/documentation/operators/merge.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                       evtIn;
    std::vector<gr::PortIn<T, gr::Async>> in;
    gr::PortOut<T>                        out;

    A<std::vector<std::string>, "contexts", Doc<"one context per input, in input order">> contexts;
    A<bool, "back pressure", Doc<"true: do not consume from un-selected inputs">>         back_pressure = false;

    A<gr::Size_t, "n switches", Doc<"context changes acted on">>              n_switches  = 0U;
    A<gr::Size_t, "n dropped", Doc<"samples read from an un-selected input">> n_dropped   = 0U;
    A<gr::Size_t, "n unmatched", Doc<"contexts naming no input">>             n_unmatched = 0U;

    GR_MAKE_REFLECTABLE(Mux, evtIn, in, out, contexts, back_pressure, n_switches, n_dropped, n_unmatched);

    constexpr static std::size_t kNothingSelected = std::numeric_limits<std::size_t>::max();

    std::size_t _selected = kNothingSelected;

    void start() { _selected = kNothingSelected; }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("contexts") && oldSettings.find_value("contexts") != newSettings.find_value("contexts")) {
            in.resize(contexts.value.size());
            _selected = kNothingSelected;
        }
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, std::span<TInput>& ins, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty()) {
                select(event);
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        const std::size_t forwarded = _selected < ins.size() ? forward(ins[_selected], outSpan) : 0UZ;
        for (std::size_t channel = 0UZ; channel < ins.size(); ++channel) {
            if (channel != _selected) {
                discard(ins[channel]);
            }
        }

        outSpan.publish(forwarded);
        return gr::work::Status::OK;
    }

private:
    void select(const gr::property_map_view& event) {
        const auto named = event.template get_if<std::string_view>(std::string_view{gr::tag::kContext.key()});
        if (!named) {
            return;
        }
        const auto found = std::ranges::find(contexts.value, *named);
        if (found == contexts.value.end()) {
            n_unmatched = n_unmatched + 1U;
            return;
        }
        const std::size_t wanted = static_cast<std::size_t>(std::ranges::distance(contexts.value.begin(), found));
        if (wanted != _selected) {
            _selected  = wanted;
            n_switches = n_switches + 1U;
        }
    }

    [[nodiscard]] std::size_t forward(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t nSamples = std::min(inSpan.size(), outSpan.size());
        std::ranges::copy(inSpan | std::views::take(nSamples), outSpan.begin());
        for (const auto& tag : inSpan.rawTags()) {
            if (tag.index >= inSpan.streamIndex && tag.index - inSpan.streamIndex < nSamples) {
                outSpan.publishTag(tag.map, tag.index - inSpan.streamIndex);
            }
        }
        std::ignore = inSpan.consume(nSamples);
        return nSamples;
    }

    void discard(gr::InputSpanLike auto& inSpan) {
        if (back_pressure) {
            std::ignore = inSpan.consume(0UZ);
            return;
        }
        n_dropped   = n_dropped + static_cast<gr::Size_t>(inSpan.size());
        std::ignore = inSpan.consume(inSpan.size());
    }
};

GR_REGISTER_BLOCK(gr::basic::Demux, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double>, gr::property_map ])
GR_REGISTER_BLOCK(gr::basic::Demux, ([T], 0UZ, gr::basic::Routing::context), [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ, Routing routing = Routing::turns>
requires(std::is_arithmetic_v<T> || meta::complex_like<T> || detail::kIsEvent<T>) && (nPorts <= detail::kMaxPorts) && (routing == Routing::turns || nPorts == 0UZ)
struct Demux : Block<Demux<T, nPorts, routing>, NoTagPropagation> {
    using Description = Doc<R""(@brief deals turns of `lengths[p]` samples to each output p (GR3: Deinterleave, Stream Demux);
equal lengths deinterleave, unequal ones split, for example, a header from each payload.

Output p's rate is the input's times lengths[p] / sum(lengths).
A tag goes with its sample; its `sample_rate`, replaced by each output's, goes to all of them.
Only whole turns are dealt: a partial turn at the end of the stream stays behind.
For `property_map` (or `property_map_view`) the ports are event ports and a turn counts events, dealt round-robin; host only.)"">;

    detail::DemuxIn<T>                         in;
    detail::PortsOf<detail::MuxOut<T>, nPorts> out = detail::configuredPorts<detail::MuxOut<T>, nPorts>();

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
        if constexpr (detail::kIsEvent<T>) {
            std::size_t i = 0UZ;
            for (const property_map_view& event : input | std::views::filter(detail::isEvent) | std::views::take(nTurns * layout.turn())) {
                const std::size_t port = layout.portOf(i);
                if (!detail::emitOrStop(*this, outputs[port], layout.indexInPort(i++, port), event)) {
                    return gr::work::Status::ERROR;
                }
            }
        } else {
            for (std::size_t turn = 0UZ; turn < nTurns; ++turn) {
                for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                    std::copy_n(input.data() + turn * layout.turn() + layout.runStart[port], layout.length(port), outputs[port].data() + turn * layout.length(port));
                }
            }
        }
        return publishTurns(input, outputs, layout, nTurns);
    }

    template<OutputSpanLike TOut>
    requires(!detail::kIsEvent<T>)
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
        if (newSettings.contains("lengths") && detail::acceptLengths<T, nPorts>(*this, out, oldSettings, "output")) {
            const std::size_t turn = turnLayoutOf(lengths.value).turn();
            if (detail::chunkFitsBuffer(*this, in, turn, std::format("a turn of {} samples", turn))) {
                applyLengths();
                _announceRate = _inputSampleRate > 0.f;
            }
        }
    }

    [[nodiscard]] static std::size_t wholeTurns(const auto& input, const auto& outputs, const TurnLayout& layout) noexcept {
        std::size_t nTurns = detail::itemCount(input) / layout.turn();
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
            if (relativeIndex < 0) {
                continue;
            }
            const std::size_t i = detail::itemsBefore(input, static_cast<std::size_t>(relativeIndex));
            if (i >= taken) {
                continue;
            }
            const std::size_t samplePort = layout.portOf(i);
            this->forwardedTag(tagMap.get(), [&](const auto& forwarded) {
                if (!forwarded.contains(std::string_view{gr::tag::kSampleRate.key()})) {
                    detail::publishWithRate(outputs[samplePort], forwarded, layout.portSamplesBefore(i, samplePort), 0.f);
                    return;
                }
                _inputSampleRate = forwarded.template value_or<float>(gr::tag::kSampleRate.key(), 0.f);
                for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
                    const std::size_t offset = layout.portSamplesBefore(i, port);
                    if (port == samplePort) {
                        detail::publishWithRate(outputs[port], forwarded, offset, outputRate(port));
                    } else {
                        outputs[port].publishTag(detail::sampleRateTag(outputRate(port)), offset);
                    }
                }
            });
        }
        std::ignore = input.consume(detail::spanIndexAfter(input, taken));
        for (std::size_t port = 0UZ; port < outputs.size(); ++port) {
            outputs[port].publish(nTurns * layout.length(port));
        }
        if (detail::itemCount(input) - taken < layout.turn() && detail::streamEnded(in, input)) {
            return gr::work::Status::DONE;
        }
        return nTurns == 0UZ ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }

    void applyLengths() {
        if constexpr (nPorts == 0UZ) {
            detail::resizePorts(out, lengths.value.size());
        }
        in.min_samples = turnLayoutOf(lengths.value).turn(); // else the framework cuts the input at a tag short of a turn
    }
};

template<typename T>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>)
struct Demux<T, 0UZ, Routing::context> : gr::Block<Demux<T, 0UZ, Routing::context>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief route a stream to the output its context tag (or any `select_key`) names, sample-exact

    in      ─a─a─R:a─a─a─F:b─b─▶    (no RxMarbles equivalent: Rx routes by predicate, not by tag)
    out#0   ──────R:a─a─a──────▶    contexts = RAMP, FLATTOP
    out#1   ─────────────F:b─b─▶
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    gr::PortIn<T>                          in;
    std::vector<gr::PortOut<T, gr::Async>> out;

    A<std::vector<std::string>, "contexts", Doc<"one context per output, in output order">>   contexts;
    A<std::string, "select key", Doc<"tag or event key whose string value names the output">> select_key = std::string(gr::tag::kContext.key());

    A<gr::Size_t, "n switches", Doc<"context changes acted on">>                     n_switches  = 0U;
    A<gr::Size_t, "n unmatched", Doc<"samples with no output, or before the first">> n_unmatched = 0U;

    GR_MAKE_REFLECTABLE(Demux, evtIn, in, out, contexts, select_key, n_switches, n_unmatched);

    std::size_t                                           _streamIndex = 0UZ;
    std::size_t                                           _selected    = kNothingSelected;
    std::vector<std::pair<std::size_t, gr::property_map>> _switches;
    std::vector<std::size_t>                              _published;

    constexpr static std::size_t kNothingSelected = std::numeric_limits<std::size_t>::max();

    void start() {
        _streamIndex = 0UZ;
        _selected    = kNothingSelected;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("contexts") && oldSettings.find_value("contexts") != newSettings.find_value("contexts")) {
            out.resize(contexts.value.size());
            _selected = kNothingSelected;
        }
    }

    template<gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, std::span<TOutput>& outs) {
        gather(evtSpan, inSpan);
        _published.assign(outs.size(), 0UZ);

        std::size_t consumed   = 0UZ;
        std::size_t nextSwitch = 0UZ;
        while (consumed < inSpan.size()) {
            const std::size_t absolute = _streamIndex + consumed;
            while (nextSwitch < _switches.size() && _switches[nextSwitch].first <= absolute) {
                select(_switches[nextSwitch].second, outs);
                ++nextSwitch;
            }

            const std::size_t until = nextSwitch < _switches.size() ? _switches[nextSwitch].first : _streamIndex + inSpan.size();
            std::size_t       run   = until - absolute;
            if (_selected >= outs.size()) {
                n_unmatched = n_unmatched + static_cast<gr::Size_t>(run);
                consumed += run;
                continue;
            }
            run = std::min(run, outs[_selected].size() - _published[_selected]);
            if (run == 0UZ) {
                break;
            }
            std::ranges::copy(inSpan | std::views::drop(consumed) | std::views::take(run), outs[_selected].begin() + static_cast<std::ptrdiff_t>(_published[_selected]));
            _published[_selected] += run;
            consumed += run;
        }

        for (std::size_t channel = 0UZ; channel < outs.size(); ++channel) {
            outs[channel].publish(_published[channel]);
        }
        _streamIndex += consumed;
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void gather(gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan) {
        _switches.clear();
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && event.contains(std::string_view{select_key.value})) {
                _switches.emplace_back(_streamIndex, gr::property_map{event});
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        for (const auto& tag : inSpan.rawTags()) {
            if (tag.map.contains(std::string_view{select_key.value})) {
                _switches.emplace_back(_streamIndex + (tag.index - inSpan.streamIndex), gr::property_map{gr::property_map_view{tag.map}});
            }
        }
        std::ranges::stable_sort(_switches, {}, [](const auto& entry) { return entry.first; });
    }

    template<gr::OutputSpanLike TOutput>
    void select(const gr::property_map& switching, std::span<TOutput>& outs) {
        const gr::property_map_view view{switching};
        const auto                  named = view.template get_if<std::string_view>(std::string_view{select_key.value});
        const std::size_t           found = named.has_value() ? indexOf(*named) : kNothingSelected;
        if (found != _selected) {
            n_switches = n_switches + 1U;
        }
        _selected = found;
        if (_selected < outs.size()) {
            outs[_selected].publishTag(switching, _published[_selected]);
        }
    }

    [[nodiscard]] std::size_t indexOf(std::string_view context) const {
        const auto found = std::ranges::find(contexts.value, context);
        return found == contexts.value.end() ? kNothingSelected : static_cast<std::size_t>(std::ranges::distance(contexts.value.begin(), found));
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_MUX_HPP
