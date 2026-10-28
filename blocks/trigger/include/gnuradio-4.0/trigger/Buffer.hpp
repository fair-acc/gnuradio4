#ifndef GNURADIO_TRIGGER_BUFFER_HPP
#define GNURADIO_TRIGGER_BUFFER_HPP

#include <algorithm>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

namespace gr::blocks::trigger {

enum class BufferMode : std::uint8_t { count, time, toggle, when };

GR_REGISTER_BLOCK("gr::blocks::trigger::BufferCount", gr::blocks::trigger::Buffer, ([T], gr::blocks::trigger::BufferMode::count), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::BufferTime", gr::blocks::trigger::Buffer, ([T], gr::blocks::trigger::BufferMode::time), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::BufferToggle", gr::blocks::trigger::Buffer, ([T], gr::blocks::trigger::BufferMode::toggle), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::trigger::BufferWhen", gr::blocks::trigger::Buffer, ([T], gr::blocks::trigger::BufferMode::when), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

namespace detail {

template<typename T>
[[nodiscard]] gr::Tensor<T> asTensor(std::vector<T>&& window) {
    return gr::Tensor<T>(gr::data_from, std::move(window));
}

template<typename T>
[[nodiscard]] gr::DataSet<T> asDataSet(std::vector<T>&& window, std::size_t firstSample, double rate, std::string_view signalName) {
    gr::DataSet<T> dataSet;
    dataSet.timestamp = 0;
    dataSet.axis_names.emplace_back(rate > 0. ? "time" : "sample");
    dataSet.axis_units.emplace_back(rate > 0. ? "s" : "");
    dataSet.axis_values.resize(1UZ);
    dataSet.axis_values[0].reserve(window.size());
    for (std::size_t i = 0UZ; i < window.size(); ++i) {
        const double position = static_cast<double>(firstSample + i);
        dataSet.axis_values[0].emplace_back(static_cast<T>(rate > 0. ? position / rate : position));
    }
    dataSet.extents.emplace_back(static_cast<std::int32_t>(window.size()));
    dataSet.signal_names.emplace_back(signalName);
    dataSet.signal_quantities.emplace_back("");
    dataSet.signal_units.emplace_back("");
    dataSet.signal_values.assign(window.begin(), window.end());
    dataSet.signal_ranges.resize(1UZ);
    dataSet.meta_information.resize(1UZ);
    dataSet.meta_information[0].insert_or_assign(std::string_view{"first_sample"}, static_cast<std::uint64_t>(firstSample));
    dataSet.timing_events.resize(1UZ);
    return dataSet;
}

} // namespace detail

template<BufferMode kMode>
using BufferDescription = std::conditional_t<kMode == BufferMode::count, Doc<R"(@brief gather a fixed number of samples into a window, with an optional overlap [bufferCount]

    in   ─1──2──3──4──5──6──7──8─▶     bufferCount
    out  ──────[1,2,3]───[4,5,6]────▶  n_count = 3, n_skip = 0

 [1] example: https://rxmarbles.com/#bufferCount
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffercount.html
)">,
    std::conditional_t<kMode == BufferMode::time, Doc<R"(@brief gather the samples of a duration into a window, with an optional creation interval [bufferTime]

 [1] example: https://rxmarbles.com/#bufferTime
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffertime.html
)">,
        std::conditional_t<kMode == BufferMode::toggle, Doc<R"(@brief open a window on a trigger and close it after the duration that trigger carries [bufferToggle]

 [1] example: https://rxmarbles.com/#bufferToggle
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffertoggle.html
)">,
            Doc<R"(@brief close a window on each trigger and open the next, so the windows tile the stream [bufferWhen]

    evtIn  ──────T────────T──▶     bufferWhen
    in     ─1─2─3─4─5─6─7─8──▶
    out    ─────[1,2]────[3..6]─▶

 [1] example: https://rxmarbles.com/#bufferWhen
 [2] detailed documentation: https://reactivex.io/documentation/operators/bufferwhen.html
)">>>>;

/** one window collector behind the four Rx buffer operators; each name sets only the settings its mode reads */
template<typename T, BufferMode kMode>
struct Buffer : gr::Block<Buffer<T, kMode>, gr::NoTagPropagation> {
    using Description = BufferDescription<kMode>;
    using Window      = std::conditional_t<kMode == BufferMode::count, gr::Tensor<T>, gr::DataSet<T>>;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                evtIn;
    gr::PortIn<T>                  in;
    gr::PortOut<Window, gr::Async> out;

    A<gr::Size_t, "n count", Doc<"count: samples per window">, gr::Limits<1U, std::numeric_limits<gr::Size_t>::max()>>                                                n_count   = 1U;
    A<gr::Size_t, "n skip", Doc<"count: samples between window starts, 0 = back to back, below n_count overlaps">>                                                    n_skip    = 0U;
    A<gr::Size_t, "n samples", Doc<"time, toggle: window length in samples, 0 = from 'timeout'; toggle: a trigger's closing_samples or closing_seconds override it">> n_samples = 0U;
    A<float, "timeout", Doc<"time, toggle: s, window length; 0 = from 'n_samples'">>                                                                                  timeout   = 0.f;
    A<gr::Size_t, "n every", Doc<"time: samples between window starts, 0 = back to back">>                                                                            n_every   = 0U;
    A<std::pmr::string, "opening filter", Doc<"toggle: filter opening a window, empty = every trigger">>                                                              opening_filter;
    A<std::pmr::string, "filter", Doc<"when: filter closing a window, empty = every trigger">>                                                                        filter;
    gr::SampleRate                                                                                                                                                    sample_rate = 0.f;
    gr::SignalName                                                                                                                                                    signal_name = std::pmr::string("window");
    A<gr::Size_t, "max open", Doc<"windows open at once before one is refused">>                                                                                      max_open    = 16U;

    A<gr::Size_t, "n windows">                                                           n_windows  = 0U;
    A<gr::Size_t, "n refused", Doc<"windows an already-full collector would not open">>  n_refused  = 0U;
    A<gr::Size_t, "n openings", Doc<"toggle: triggers that opened a window">>            n_openings = 0U;
    A<gr::Size_t, "n empty", Doc<"when: windows two triggers on one sample left empty">> n_empty    = 0U;

    GR_MAKE_REFLECTABLE(Buffer, evtIn, in, out, n_count, n_skip, n_samples, timeout, n_every, opening_filter, filter, sample_rate, signal_name, max_open, n_windows, n_refused, n_openings, n_empty);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    WindowCollector<T>  _windows;
    TimeBase            _time;
    MatchState          _accept{};
    bool                _filtered    = false;
    std::size_t         _nextOpening = 0UZ;
    detail::ErrorMemory _deferredError{};

    void start() {
        detail::sendDeferredError(*this);
        _windows.reset();
        _windows.maxOpen = kMode == BufferMode::when ? 2UZ : std::max<std::size_t>(max_open.value, 1UZ);
        _nextOpening     = 0UZ;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
        if constexpr (kMode == BufferMode::when) {
            std::ignore = _windows.openHere();
        }
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if constexpr (kMode == BufferMode::toggle || kMode == BufferMode::when) {
            _filtered = detail::compileOptionalFilter(*this, kMode == BufferMode::toggle ? opening_filter.value : filter.value, std::string_view{}, _accept);
        }
        _time.setRate(static_cast<double>(sample_rate));
        if constexpr (kMode != BufferMode::when) {
            _windows.maxOpen = std::max<std::size_t>(max_open.value, 1UZ);
        }
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        if constexpr (kMode == BufferMode::toggle || kMode == BufferMode::when) {
            for (const gr::property_map_view& event : evtSpan) {
                if (!event.empty() && detail::acceptsTrigger(_accept, _filtered, event)) {
                    actOn(event);
                }
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        if constexpr (kMode == BufferMode::count || kMode == BufferMode::time) {
            for (const auto& tag : inSpan.rawTags()) {
                _time.adopt(gr::property_map_view{tag.map}, _windows.streamIndex + (tag.index - inSpan.streamIndex));
            }
            const std::size_t length = kMode == BufferMode::count ? static_cast<std::size_t>(n_count) : resolveDuration(_time, n_samples, timeout).value_or(0UZ);
            const std::size_t stride = kMode == BufferMode::count ? (n_skip == 0U ? length : static_cast<std::size_t>(n_skip)) : (n_every == 0U ? length : static_cast<std::size_t>(n_every));
            while (length > 0UZ && _nextOpening < _windows.streamIndex + inSpan.size() && _windows.open(_nextOpening, _nextOpening + length)) {
                _nextOpening += stride;
            }
            if (length > 0UZ) {
                _windows.push(std::span<const T>{inSpan.begin(), inSpan.size()});
            }
        } else {
            std::size_t pushed = 0UZ;
            for (const auto& tag : inSpan.rawTags()) {
                const gr::property_map_view carried{tag.map};
                const std::size_t           at = tag.index - inSpan.streamIndex;
                _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(pushed), at - pushed});
                pushed = at;
                _time.adopt(carried, _windows.streamIndex);
                if (detail::acceptsTrigger(_accept, _filtered, carried)) {
                    actOn(carried);
                }
            }
            _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(pushed), inSpan.size() - pushed});
        }
        n_refused = static_cast<gr::Size_t>(_windows.refused);

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size()) {
            auto window = _windows.take();
            if (!window) {
                break;
            }
            if (window->samples.empty()) {
                n_empty = n_empty + 1U;
            }
            if constexpr (kMode == BufferMode::count) {
                outSpan[emitted++] = detail::asTensor(std::move(window->samples));
            } else {
                const std::size_t length = window->samples.size();
                outSpan[emitted++]       = detail::asDataSet(std::move(window->samples), _windows.streamIndex - length, _time.rate(), signal_name.value);
            }
            n_windows = n_windows + 1U;
        }
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    /** toggle: open a window as long as the trigger says, else the settings; when: close the current window, open the next */
    void actOn(const gr::property_map_view& trigger) {
        if constexpr (kMode == BufferMode::when) {
            _windows.closeOldest();
            std::ignore = _windows.openHere();
        } else {
            std::size_t length = resolveDuration(_time, n_samples, timeout).value_or(0UZ);
            if (const auto meta = trigger.template get_if<gr::property_map>(std::string_view{gr::tag::kTriggerMetaInfo.key()})) {
                const gr::property_map_view details{*meta};
                if (const auto samples = details.template get_if<std::uint64_t>(std::string_view{"closing_samples"})) {
                    length = static_cast<std::size_t>(*samples);
                } else if (const auto seconds = details.template get_if<double>(std::string_view{"closing_seconds"})) {
                    length = _time.samples(*seconds).value_or(length);
                }
            }
            if (length > 0UZ && _windows.openHere(_windows.streamIndex + length)) {
                n_openings = n_openings + 1U;
            }
        }
    }
};

template<typename T>
using BufferCount = Buffer<T, BufferMode::count>;
template<typename T>
using BufferTime = Buffer<T, BufferMode::time>;
template<typename T>
using BufferToggle = Buffer<T, BufferMode::toggle>;
template<typename T>
using BufferWhen = Buffer<T, BufferMode::when>;

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_BUFFER_HPP
