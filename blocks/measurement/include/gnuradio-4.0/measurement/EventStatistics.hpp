#ifndef GNURADIO_MEASUREMENT_EVENTSTATISTICS_HPP
#define GNURADIO_MEASUREMENT_EVENTSTATISTICS_HPP

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/PmtTypeHelpers.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::measurement {

GR_REGISTER_BLOCK(gr::blocks::measurement::EventStatistics)

struct EventStatistics : gr::Block<EventStatistics, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief running mean, standard deviations, minimum, maximum, count and last value of a numeric event field

    evtIn   ──v=2──v=4──v=9──▶     the field `value` of each matching event
    evtOut  ──────────────────▶    one event per input: count, mean, sd_sample, sd_population, min, max, last

The report is in the new event's `trigger_meta_info`, dated as the event that caused it.
`sd_sample` (divisor n - 1) is absent for a single value, `sd_population` is zero.

 [1] B. P. Welford, "Note on a method for calculating corrected sums of squares and products", Technometrics 4(3), 1962
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "filter", Doc<"trigger filter of the events that count; empty = all">>                                                                                             filter;
    A<std::pmr::string, "field", Doc<"name of the numeric entry to accumulate, in trigger_meta_info else at the root; a kernel reads a real or an integer, not a number written as text">> field        = std::pmr::string("value");
    A<double, "max lateness", Doc<"s, how long after its trigger_time an event may arrive; earlier ones wait as long">>                                                                    max_lateness = 0.;
    gr::TriggerName                                                                                                                                                                        trigger_name;
    mutable A<gr::Size_t, "n accumulated", Doc<"values folded into the statistics">>                                                                                                       n_accumulated    = 0U;
    mutable A<gr::Size_t, "n skipped", Doc<"matching events without a numeric field">>                                                                                                     n_skipped        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>                                                                                                     n_events_dropped = 0U;

    using MatchState = gr::trigger::BasicTriggerNameCtxMatcher::MatchState;

    MatchState _accepts{};
    bool       _filtered       = false;
    gr::Size_t _settings_epoch = 0U;

    GR_MAKE_REFLECTABLE(EventStatistics, evtIn, evtOut, filter, field, max_lateness, trigger_name, n_accumulated, n_skipped, n_events_dropped, _accepts, _filtered, _settings_epoch);

    constexpr static std::uint32_t kReportKeys = 8U;

    mutable MatchState                      _accepting{};
    mutable gr::Size_t                      _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    mutable gr::Size_t                      _n          = 0U; /// values folded, the Welford count
    mutable double                          _mean       = 0.;
    mutable double                          _sumSquares = 0.; /// of the deviations from the running mean
    mutable double                          _minimum    = std::numeric_limits<double>::infinity();
    mutable double                          _maximum    = -std::numeric_limits<double>::infinity();
    mutable double                          _last       = 0.;
    mutable gr::blocks::trigger::EventStore _events;

    void start() {
        gr::blocks::trigger::detail::nameTriggerAfterBlock(*this);
        _stateEpoch = std::numeric_limits<gr::Size_t>::max();
    }

    void reset() {
        _events.clear();
        _n               = 0U;
        _mean            = 0.;
        _sumSquares      = 0.;
        _minimum         = std::numeric_limits<double>::infinity();
        _maximum         = -std::numeric_limits<double>::infinity();
        _last            = 0.;
        n_accumulated    = 0U;
        n_skipped        = 0U;
        n_events_dropped = 0U;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _filtered        = gr::blocks::trigger::detail::compileOptionalFilter(*this, filter.value, std::string_view{}, _accepts);
        _events.lateness = gr::blocks::trigger::detail::nanoseconds(max_lateness.value);
        _settings_epoch  = _settings_epoch + 1U;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        if (_stateEpoch != _settings_epoch) { // a new filter, not a new run: the statistics carry on
            _accepting  = _accepts;
            _stateEpoch = _settings_epoch;
        }
        std::size_t emitted = 0UZ;
        std::ignore         = gr::blocks::trigger::forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            if (gr::blocks::trigger::detail::acceptsTrigger(_accepting, _filtered, event)) {
                accumulate<kOnHost>(event, evtOutSpan, emitted);
            }
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = static_cast<gr::Size_t>(_events.dropped);
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }

private:
    template<bool onHost>
    [[nodiscard]] std::optional<double> numberIn(const gr::property_map_view& map) const {
        const std::string_view key{field.value};
        if constexpr (onHost) {
            const auto found = map.find_value(key);
            if (!found) {
                return std::nullopt;
            }
            const auto converted = gr::pmt::convert_safely<double>(*found);
            return converted ? std::optional{*converted} : std::nullopt;
        } else { // convert_safely's numeric rules, without the Value it needs: an integer beyond double's mantissa is refused
            const auto read = [&]<typename TNumber>() -> std::optional<double> {
                const auto found = map.template get_if<TNumber>(key);
                if (!found) {
                    return std::nullopt;
                }
                if constexpr (std::is_integral_v<TNumber> && std::numeric_limits<TNumber>::digits > std::numeric_limits<double>::digits) {
                    const std::uint64_t magnitude = [&]() -> std::uint64_t {
                        if constexpr (std::is_signed_v<TNumber>) {
                            return *found < 0 ? 0ULL - static_cast<std::uint64_t>(*found) : static_cast<std::uint64_t>(*found);
                        } else {
                            return *found;
                        }
                    }();
                    if (std::bit_width(magnitude) > std::numeric_limits<double>::digits) {
                        return std::nullopt;
                    }
                }
                return static_cast<double>(*found);
            };
            std::optional<double>            number;
            const auto                       readFirst = [&]<typename... TNumber>() { return ((number = read.template operator()<TNumber>()).has_value() || ...); };
            std::ignore                                = readFirst.template operator()<double, float, std::int64_t, std::uint64_t, std::int32_t, std::uint32_t, std::int16_t, std::uint16_t, std::int8_t, std::uint8_t>();
            return number;
        }
    }

    template<bool onHost>
    void accumulate(const gr::property_map_view& event, auto& evtOutSpan, std::size_t& emitted) const {
        const auto details = event.template get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
        const auto value   = (details ? numberIn<onHost>(*details) : std::nullopt).or_else([&] { return numberIn<onHost>(event); });
        if (!value || !std::isfinite(*value)) {
            n_skipped = n_skipped + 1U;
            return;
        }
        _n                 = _n + 1U;
        n_accumulated      = _n;
        const double count = static_cast<double>(_n);
        const double shift = *value - _mean; // Welford's update; with no value yet it starts from 0 exactly
        _mean += shift / count;
        _sumSquares += shift * (*value - _mean);
        _minimum = std::min(_minimum, *value);
        _maximum = std::max(_maximum, *value);
        _last    = *value;

        const auto writeReport = [&](gr::property_map_view& result) {
            if (!gr::blocks::trigger::detail::writeEvent(result, trigger_name.value, event)) {
                return false;
            }
            constexpr std::uint32_t kFiguresBytes = (kReportKeys - 1U) * gr::pmt::alignToRecord(gr::pmt::kRecHeaderBytes + static_cast<std::uint32_t>(sizeof(double)));
            const std::uint32_t     fieldBytes    = gr::pmt::alignToRecord(gr::pmt::kRecHeaderBytes + static_cast<std::uint32_t>(field.value.size()) + 1U);
            gr::property_map_view   report        = result.try_emplace_map(gr::tag::TRIGGER_META_INFO.key(), kReportKeys, kFiguresBytes + fieldBytes);
            const bool              withField     = !report.blob().empty();
            if (!withField) { // degrade, never drop: the field name is the first to go
                report = result.try_emplace_map(gr::tag::TRIGGER_META_INFO.key(), kReportKeys - 1U, kFiguresBytes);
            }
            const auto put      = [&report](std::string_view key, const auto& figure) { return report.try_emplace(key, figure); };
            const bool moments  = !report.blob().empty() && put("count", _n) && put("mean", _mean) && put("sd_population", std::sqrt(_sumSquares / count));
            const bool spread   = moments && (_n < 2U || put("sd_sample", std::sqrt(_sumSquares / (count - 1.))));
            const bool extremes = spread && put("min", _minimum) && put("max", _maximum) && put("last", _last);
            std::ignore         = extremes && withField && put("field", std::string_view{field.value});
            return true;
        };
        if (gr::emit(evtOutSpan, emitted, gr::blocks::trigger::detail::kEventKeys, writeReport).has_value()) {
            ++emitted;
        }
    }
};

} // namespace gr::blocks::measurement

#endif // GNURADIO_MEASUREMENT_EVENTSTATISTICS_HPP
