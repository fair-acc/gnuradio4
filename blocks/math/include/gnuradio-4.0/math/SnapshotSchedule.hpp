#ifndef GNURADIO_MATH_SNAPSHOT_SCHEDULE_HPP
#define GNURADIO_MATH_SNAPSHOT_SCHEDULE_HPP

#include <chrono>
#include <cstddef>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>

namespace gr::blocks::math {

inline constexpr gr::Size_t kDefaultBins = 100U;

/**
 * When an accumulating block publishes a snapshot: on an event its trigger filter accepts (any event while none is named), after `nSamples` entries, or `timeout` s after the last snapshot -- whichever comes first.
 * The block owns the settings (`trigger_name`, `n_samples`, `timeout`) and passes them in; this holds the state between work calls.
 *
 *   if (const auto set = _snapshot.setFilter(trigger_name.value); !set) { keepPreviousSetting(*this, trigger_name, "trigger_name", oldSettings, std::string(), set.error()); }
 */
struct SnapshotSchedule {
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState filter{};
    bool                                                filtered         = false;
    bool                                                asked            = false;
    std::size_t                                         sinceLastPublish = 0UZ;
    std::chrono::steady_clock::time_point               lastPublish;

    /** an empty name accepts any event; a name that does not compile is refused with the reason, and then any event is accepted */
    [[nodiscard]] std::expected<void, std::string> setFilter(std::string_view triggerName) {
        filtered = !triggerName.empty();
        if (!filtered) {
            return {};
        }
        const auto compiled = gr::trigger::BasicTriggerNameCtxMatcher::compile(triggerName);
        if (!compiled) {
            filtered = false;
            return std::unexpected(compiled.error().message);
        }
        filter = *compiled;
        return {};
    }

    void noteEvent(const gr::property_map_view& event) { asked = asked || (!event.empty() && (!filtered || gr::trigger::BasicTriggerNameCtxMatcher::match(filter, event) == gr::trigger::MatchResult::Matching)); }

    void noteEvents(auto& evtSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            noteEvent(event);
        }
        std::ignore = evtSpan.consume(evtSpan.size());
    }

    [[nodiscard]] bool due(std::chrono::steady_clock::time_point now, std::size_t nSamples, double timeout) const {
        if (asked) {
            return true;
        }
        if (nSamples > 0UZ && sinceLastPublish >= nSamples) {
            return true;
        }
        return timeout > 0. && std::chrono::duration<double>(now - lastPublish).count() >= timeout;
    }

    void published(std::chrono::steady_clock::time_point now) {
        asked            = false;
        sinceLastPublish = 0UZ;
        lastPublish      = now;
    }
};

/** refuses the value just set on `setting`: restores the one in `oldSettings` (else `fallback`) and tells the operator why */
template<typename TBlock>
void keepPreviousSetting(TBlock& block, auto& setting, std::string_view key, const gr::property_map& oldSettings, auto fallback, std::string_view reason) {
    const auto previous = oldSettings.value_or<std::remove_cvref_t<decltype(fallback)>>(std::string(key), fallback);
    block.emitErrorMessage("settingsChanged()", std::format("'{}' refused ({}): keeps {}", key, reason, previous));
    setting = previous;
}

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_SNAPSHOT_SCHEDULE_HPP
