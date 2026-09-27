#ifndef GNURADIO_TRIGGER_BUFFER_HPP
#define GNURADIO_TRIGGER_BUFFER_HPP

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

GR_REGISTER_BLOCK(gr::blocks::trigger::BufferCount, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::BufferTime, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::BufferToggle, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK(gr::blocks::trigger::BufferWhen, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

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

template<typename T>
struct BufferCount : gr::Block<BufferCount<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief gather a fixed number of samples into a window, with an optional overlap [bufferCount]

    in   ─1──2──3──4──5──6──7──8─▶     bufferCount
    out  ──────[1,2,3]───[4,5,6]────▶  n_count = 3, n_skip = 0

`n_count` sets the window length; `n_skip` sets the distance from one window's start to the next, 0 meaning back to back so
every sample belongs to exactly one window, and a value below `n_count` overlaps them.

 [1] example: https://rxmarbles.com/#bufferCount
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffercount.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>                         in;
    gr::PortOut<gr::Tensor<T>, gr::Async> out;

    A<gr::Size_t, "n count", Doc<"samples per window, at least 1">>              n_count  = 1U;
    A<gr::Size_t, "n skip", Doc<"window stride, samples">>                       n_skip   = 0U;
    A<gr::Size_t, "max open", Doc<"windows open at once before one is refused">> max_open = 16U;

    A<gr::Size_t, "n windows">                                                          n_windows = 0U;
    A<gr::Size_t, "n refused", Doc<"windows an already-full collector would not open">> n_refused = 0U;

    GR_MAKE_REFLECTABLE(BufferCount, in, out, n_count, n_skip, max_open, n_windows, n_refused);

    WindowCollector<T> _windows;
    std::size_t        _nextOpening = 0UZ;

    void start() {
        _windows.reset();
        _windows.maxOpen = max_open;
        _nextOpening     = 0UZ;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n_count == 0U) {
            const gr::Size_t previous = oldSettings.value_or<gr::Size_t>(std::string("n_count"), 1U);
            gr::log::warning("BufferCount: 'n_count' = 0 refused (a window of no samples is not a window); keeping {}", previous == 0U ? 1U : previous);
            n_count = previous == 0U ? 1U : previous;
        }
        _windows.maxOpen = max_open == 0U ? 1UZ : max_open.value;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const std::size_t stride = n_skip == 0U ? n_count.value : n_skip.value;
        const std::size_t end    = _windows.streamIndex + inSpan.size();
        while (_nextOpening < end) {
            if (!_windows.open(_nextOpening, _nextOpening + n_count)) {
                break;
            }
            _nextOpening += stride;
        }
        _windows.push(std::span<const T>{inSpan.begin(), inSpan.size()});
        n_refused = static_cast<gr::Size_t>(_windows.refused);

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size()) {
            auto window = _windows.take();
            if (!window) {
                break;
            }
            outSpan[emitted++] = detail::asTensor(std::move(*window));
            n_windows          = n_windows + 1U;
        }
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct BufferTime : gr::Block<BufferTime<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief gather the samples of a duration into a window, with an optional creation interval [bufferTime]

The window's length is a duration rather than a count, which is what an operator asks for -- "a window per millisecond" -- and
what stays right when the sample rate changes.

 [1] example: https://rxmarbles.com/#bufferTime
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffertime.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>                          in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;

    A<gr::Size_t, "n samples", Doc<"window length in samples, 0 = take it from 'timeout'">> n_samples   = 0U;
    A<float, "timeout", Doc<"s, window length; 0 = from 'n_samples'">>                      timeout     = 0.f;
    A<gr::Size_t, "n every", Doc<"samples between window starts, 0 = back to back">>        n_every     = 0U;
    A<float, "sample rate", Doc<"Hz, converts settings, dates axis; 0 = unknown">>          sample_rate = 0.f;
    A<std::pmr::string, "signal name">                                                      signal_name = std::pmr::string("window");
    A<gr::Size_t, "max open", Doc<"windows open at once before one is refused">>            max_open    = 16U;

    A<gr::Size_t, "n windows">                                                          n_windows = 0U;
    A<gr::Size_t, "n refused", Doc<"windows an already-full collector would not open">> n_refused = 0U;

    GR_MAKE_REFLECTABLE(BufferTime, in, out, n_samples, timeout, n_every, sample_rate, signal_name, max_open, n_windows, n_refused);

    WindowCollector<T> _windows;
    TimeBase           _time;
    std::size_t        _nextOpening = 0UZ;

    void start() {
        _windows.reset();
        _windows.maxOpen = max_open;
        _nextOpening     = 0UZ;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _time.setRate(static_cast<double>(sample_rate));
        _windows.maxOpen = max_open == 0U ? 1UZ : max_open.value;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        for (const auto& tag : inSpan.rawTags()) {
            _time.adopt(gr::property_map_view{tag.map}, _windows.streamIndex + (tag.index - inSpan.streamIndex));
        }
        const std::size_t length = resolveDuration(_time, n_samples, timeout).value_or(0UZ);
        if (length == 0UZ) {
            if (!inSpan.consume(inSpan.size())) {
                return gr::work::Status::ERROR;
            }
            outSpan.publish(0UZ);
            return gr::work::Status::OK;
        }
        const std::size_t stride = n_every == 0U ? length : n_every.value;

        openThrough(length, stride, _windows.streamIndex + inSpan.size());
        _windows.push(std::span<const T>{inSpan.begin(), inSpan.size()});
        n_refused = static_cast<gr::Size_t>(_windows.refused);

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size()) {
            auto window = _windows.take();
            if (!window) {
                break;
            }
            const std::size_t length_ = window->size();
            outSpan[emitted++]        = detail::asDataSet(std::move(*window), _windows.streamIndex - length_, _time.rate(), signal_name.value);
            n_windows                 = n_windows + 1U;
        }
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void openThrough(std::size_t length, std::size_t stride, std::size_t end) {
        while (_nextOpening < end) {
            if (!_windows.open(_nextOpening, _nextOpening + length)) {
                break;
            }
            _nextOpening += stride;
        }
    }
};

template<typename T>
struct BufferToggle : gr::Block<BufferToggle<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief open a window on a trigger and close it after the duration that trigger carries [bufferToggle]

Unlike a count or a periodic duration, this window is placed by the machine rather than by the clock: an event says "from here",
and how long it lasts belongs to that event, not to the data.

 [1] example: https://rxmarbles.com/#bufferToggle
 [2] detailed documentation: https://reactivex.io/documentation/operators/buffertoggle.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    gr::PortIn<T>                          in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;

    A<std::pmr::string, "opening filter", Doc<"filter opening a window, empty = every trigger">> opening_filter;
    A<gr::Size_t, "n samples", Doc<"window length, 0 = from 'timeout'">>                         n_samples   = 0U;
    A<float, "timeout", Doc<"s, window length where the opening event does not say">>            timeout     = 0.f;
    A<float, "sample rate", Doc<"Hz, converts settings, dates axis; 0 = unknown">>               sample_rate = 0.f;
    A<std::pmr::string, "signal name">                                                           signal_name = std::pmr::string("window");
    A<gr::Size_t, "max open", Doc<"windows open at once before one is refused">>                 max_open    = 16U;

    A<gr::Size_t, "n windows">                                                          n_windows  = 0U;
    A<gr::Size_t, "n openings", Doc<"triggers that opened a window">>                   n_openings = 0U;
    A<gr::Size_t, "n refused", Doc<"windows an already-full collector would not open">> n_refused  = 0U;

    GR_MAKE_REFLECTABLE(BufferToggle, evtIn, in, out, opening_filter, n_samples, timeout, sample_rate, signal_name, max_open, n_windows, n_openings, n_refused);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    WindowCollector<T> _windows;
    TimeBase           _time;
    MatchState         _opens{};
    bool               _filtered = false;

    void start() {
        _windows.reset();
        _windows.maxOpen = max_open;
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(opening_filter.value, std::string_view{}, _opens, "BufferToggle");
        _time.setRate(static_cast<double>(sample_rate));
        _windows.maxOpen = max_open == 0U ? 1UZ : max_open.value;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && accepts(event)) {
                openFor(event, _windows.streamIndex);
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t consumed = 0UZ;
        for (const auto& tag : inSpan.rawTags()) {
            const gr::property_map_view carried{tag.map};
            const std::size_t           at = tag.index - inSpan.streamIndex;
            _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(consumed), at - consumed});
            consumed = at;
            _time.adopt(carried, _windows.streamIndex);
            if (accepts(carried)) {
                openFor(carried, _windows.streamIndex);
            }
        }
        _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(consumed), inSpan.size() - consumed});
        n_refused = static_cast<gr::Size_t>(_windows.refused);

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size()) {
            auto window = _windows.take();
            if (!window) {
                break;
            }
            const std::size_t length = window->size();
            outSpan[emitted++]       = detail::asDataSet(std::move(*window), _windows.streamIndex - length, _time.rate(), signal_name.value);
            n_windows                = n_windows + 1U;
        }
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool accepts(const gr::property_map_view& candidate) { return detail::acceptsTrigger(_opens, _filtered, candidate); }

    void openFor(const gr::property_map_view& opening, std::size_t at) {
        std::size_t length = resolveDuration(_time, n_samples, timeout).value_or(0UZ);
        if (const auto meta = opening.template get_if<gr::property_map>(std::string_view{gr::tag::TRIGGER_META_INFO.key()})) {
            const gr::property_map_view details{*meta};
            if (const auto samples = details.template get_if<std::uint64_t>(std::string_view{"closing_samples"})) {
                length = static_cast<std::size_t>(*samples);
            } else if (const auto seconds = details.template get_if<double>(std::string_view{"closing_seconds"})) {
                length = _time.samples(*seconds).value_or(length);
            }
        }
        if (length == 0UZ) {
            return;
        }
        if (_windows.open(at, at + length)) {
            n_openings = n_openings + 1U;
        }
    }
};

template<typename T>
struct BufferWhen : gr::Block<BufferWhen<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief close a window on each trigger and open the next, so the windows tile the stream [bufferWhen]

    evtIn  ──────T────────T──▶     bufferWhen
    in     ─1─2─3─4─5─6─7─8──▶
    out    ─────[1,2]────[3..6]─▶

Nothing says in advance how long a window will be -- a cycle boundary, an operator's mark, the next zero crossing.

 [1] example: https://rxmarbles.com/#bufferWhen
 [2] detailed documentation: https://reactivex.io/documentation/operators/bufferwhen.html
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    gr::PortIn<T>                          in;
    gr::PortOut<gr::DataSet<T>, gr::Async> out;

    A<std::pmr::string, "filter", Doc<"filter closing a window, empty = every trigger">> filter;
    A<float, "sample rate", Doc<"Hz, dates the axis of the window, 0 = sample indices">> sample_rate = 0.f;
    A<std::pmr::string, "signal name">                                                   signal_name = std::pmr::string("window");

    A<gr::Size_t, "n windows">                                                     n_windows = 0U;
    A<gr::Size_t, "n empty", Doc<"windows two triggers on one sample left empty">> n_empty   = 0U;

    GR_MAKE_REFLECTABLE(BufferWhen, evtIn, in, out, filter, sample_rate, signal_name, n_windows, n_empty);

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    WindowCollector<T> _windows;
    TimeBase           _time;
    MatchState         _closes{};
    bool               _filtered = false;

    void start() {
        _windows.reset();
        _windows.maxOpen = 2UZ;
        std::ignore      = _windows.openHere();
        _time.reset();
        _time.setRate(static_cast<double>(sample_rate));
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered = detail::compileOptionalFilter(filter.value, std::string_view{}, _closes, "BufferWhen");
        _time.setRate(static_cast<double>(sample_rate));
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && accepts(event)) {
                closeHere();
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        std::size_t consumed = 0UZ;
        for (const auto& tag : inSpan.rawTags()) {
            const gr::property_map_view carried{tag.map};
            const std::size_t           at = tag.index - inSpan.streamIndex;
            _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(consumed), at - consumed});
            consumed = at;
            _time.adopt(carried, _windows.streamIndex);
            if (accepts(carried)) {
                closeHere();
            }
        }
        _windows.push(std::span<const T>{inSpan.begin() + static_cast<std::ptrdiff_t>(consumed), inSpan.size() - consumed});

        std::size_t emitted = 0UZ;
        while (emitted < outSpan.size()) {
            auto window = _windows.take();
            if (!window) {
                break;
            }
            if (window->empty()) {
                n_empty = n_empty + 1U;
            }
            const std::size_t length = window->size();
            outSpan[emitted++]       = detail::asDataSet(std::move(*window), _windows.streamIndex - length, _time.rate(), signal_name.value);
            n_windows                = n_windows + 1U;
        }
        outSpan.publish(emitted);
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    [[nodiscard]] bool accepts(const gr::property_map_view& candidate) { return detail::acceptsTrigger(_closes, _filtered, candidate); }

    void closeHere() {
        _windows.closeOldest();
        std::ignore = _windows.openHere();
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_BUFFER_HPP
