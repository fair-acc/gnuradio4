#ifndef GNURADIO_MEASUREMENT_MEASUREMENTCONDITION_HPP
#define GNURADIO_MEASUREMENT_MEASUREMENTCONDITION_HPP

#include <format>
#include <limits>
#include <string>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/device/DeviceSpans.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/SampleTest.hpp>

namespace gr::blocks::measurement {

GR_REGISTER_BLOCK(gr::blocks::measurement::MeasurementCondition)

struct MeasurementCondition : gr::Block<MeasurementCondition, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief pass the measurement results whose value meets a comparison

    evtIn   ─R(3)──R(7)──R(✗)──R(5)─▶   a result's trigger_meta_info.value
    evtOut  ───────R(7)─────────R(5)─▶   comparison = greater, lower = 4

The passed event keeps the result's times, so it refers to the gate that was measured.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn  evtIn;
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::pmr::string, "comparison", Doc<"greater|greaterEqual|less|lessEqual|equal|notEqual|inside|outside">> comparison       = std::pmr::string("greater");
    A<double, "lower", Doc<"read by greater, greaterEqual, equal, notEqual, inside and outside">>               lower            = 0.;
    A<double, "upper", Doc<"read by less, lessEqual, inside and outside">>                                      upper            = 0.;
    mutable A<gr::Size_t, "n passed", Doc<"results that met the comparison">>                                   n_passed         = 0U;
    mutable A<gr::Size_t, "n failed", Doc<"results that did not">>                                              n_failed         = 0U;
    mutable A<gr::Size_t, "n invalid", Doc<"results without a value, which neither pass nor fail">>             n_invalid        = 0U;
    mutable A<gr::Size_t, "n events dropped", Doc<"events the store had no room for">>                          n_events_dropped = 0U;

    gr::blocks::trigger::Comparison _comparison = gr::blocks::trigger::Comparison::greater;

    GR_MAKE_REFLECTABLE(MeasurementCondition, evtIn, evtOut, comparison, lower, upper, n_passed, n_failed, n_invalid, n_events_dropped, _comparison);

    mutable gr::blocks::trigger::EventStore  _events;
    gr::blocks::trigger::detail::ErrorMemory _deferredError{};

    void start() { gr::blocks::trigger::detail::sendDeferredError(*this); }

    void reset() {
        _events.clear();
        n_passed         = 0U;
        n_failed         = 0U;
        n_invalid        = 0U;
        n_events_dropped = 0U;
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        if (const auto named = gr::blocks::trigger::parseComparison(comparison.value)) {
            _comparison = *named;
        } else {
            const std::string message = std::format("comparison '{}' refused (not a comparison name); keeping '{}'", comparison.value, gr::meta::enumName(_comparison).value_or(""));
            gr::blocks::trigger::detail::reportError(*this, "settingsChanged()", message);
            comparison = std::pmr::string(gr::meta::enumName(_comparison).value_or(""));
        }
        const bool range = _comparison == gr::blocks::trigger::Comparison::inside || _comparison == gr::blocks::trigger::Comparison::outside;
        if (range && lower.value > upper.value) {
            const std::string message = std::format("range [{}, {}] is inverted: nothing passes until it is set right", lower.value, upper.value);
            gr::blocks::trigger::detail::reportError(*this, "settingsChanged()", message);
        }
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& evtOutSpan) const {
        constexpr bool kOnHost = !gr::device::kIsDeviceSpan<std::remove_cvref_t<decltype(evtSpan)>>;
        std::size_t    emitted = 0UZ;
        std::ignore            = gr::blocks::trigger::forEachEvent<kOnHost>(_events, evtSpan, kOnHost ? std::numeric_limits<std::size_t>::max() : evtOutSpan.size(), [&](const gr::property_map_view& event) {
            if (emitted >= evtOutSpan.size()) {
                return false;
            }
            const auto meta  = event.template get_if<gr::property_map_view>(gr::tag::TRIGGER_META_INFO.key());
            const auto value = meta ? meta->get_if<double>(std::string_view{"value"}) : nullptr;
            if (value == nullptr) {
                n_invalid = n_invalid + 1U;
                return true;
            }
            if (!gr::blocks::trigger::holds(_comparison, *value, lower.value, upper.value)) {
                n_failed = n_failed + 1U;
                return true;
            }
            if (gr::emit(evtOutSpan, emitted, event).has_value()) {
                ++emitted;
                n_passed = n_passed + 1U;
            }
            return true;
        });
        if constexpr (kOnHost) {
            n_events_dropped = static_cast<gr::Size_t>(_events.dropped);
        }
        evtOutSpan.publish(emitted);
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::measurement

#endif // GNURADIO_MEASUREMENT_MEASUREMENTCONDITION_HPP
