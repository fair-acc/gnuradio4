#ifndef GNURADIO_TRIGGER_TIMEBASE_HPP
#define GNURADIO_TRIGGER_TIMEBASE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::trigger {

/**
 * Stream time from an anchor, shared by every block that has to date a sample.
 *
 * A `trigger_time` tag pins an absolute time to a stream index; `trigger_offset` shifts it. Two such anchors give
 * the sample rate outright, which no setting can contradict; one anchor converts through the `sample_rate` setting;
 * with neither, the base stays unarmed and the block reports rather than inventing a time.
 *
 * Times are computed from the index, never accumulated per sample, so a rate that does not divide a nanosecond
 * cannot drift.
 *
 * @code
 * TimeBase time;
 * time.setRate(sample_rate);
 * time.adopt(tagMap, streamIndex);                  // on every tag carrying a trigger_time
 * if (const auto stamp = time.at(streamIndex + i)) { ... }
 * @endcode
 */
enum class AnchorChange : std::uint8_t { first, forward, backward };

struct TimeBase {
    std::optional<std::uint64_t> anchor;             // ns, shifted by trigger_offset
    std::size_t                  anchorIndex  = 0UZ; // the sample the anchor dates
    double                       anchoredRate = 0.;  // Hz, from two anchors
    double                       settingRate  = 0.;  // Hz, from the sample_rate setting

    void reset() noexcept {
        anchor       = std::nullopt;
        anchoredRate = 0.;
    }

    constexpr void setRate(double hz) noexcept { settingRate = hz > 0. ? hz : 0.; }

    AnchorChange anchorAt(std::uint64_t timeNs, float offsetSeconds, std::size_t streamIndex) noexcept {
        const std::uint64_t dated  = timeNs + static_cast<std::uint64_t>(std::llround(static_cast<double>(offsetSeconds) * 1e9));
        const AnchorChange  change = !anchor ? AnchorChange::first : (dated < *anchor ? AnchorChange::backward : AnchorChange::forward);
        if (anchor && streamIndex > anchorIndex && dated > *anchor) {
            anchoredRate = static_cast<double>(streamIndex - anchorIndex) * 1e9 / static_cast<double>(dated - *anchor);
        }
        anchor      = dated;
        anchorIndex = streamIndex;
        return change;
    }

    std::optional<AnchorChange> adopt(const gr::property_map_view& tagMap, std::size_t streamIndex) {
        if (const auto rate = tagMap.template get_if<float>(std::string_view{"sample_rate"}); rate && *rate > 0.f) {
            settingRate = static_cast<double>(*rate);
        }
        const auto stamp = tagMap.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        if (!stamp) {
            return std::nullopt;
        }
        const auto offset = tagMap.template get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        return anchorAt(*stamp, offset ? *offset : 0.f, streamIndex);
    }

    [[nodiscard]] constexpr double rate() const noexcept { return anchoredRate > 0. ? anchoredRate : settingRate; }

    [[nodiscard]] constexpr bool measuresDuration() const noexcept { return rate() > 0.; }
    [[nodiscard]] constexpr bool datesSamples() const noexcept { return anchor.has_value() && rate() > 0.; }

    [[nodiscard]] constexpr std::optional<double> seconds(std::size_t nSamples) const noexcept {
        if (!measuresDuration()) {
            return std::nullopt;
        }
        return static_cast<double>(nSamples) / rate();
    }

    [[nodiscard]] constexpr std::optional<std::size_t> samples(double durationSeconds) const noexcept {
        if (!measuresDuration() || durationSeconds < 0.) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::llround(durationSeconds * rate()));
    }

    [[nodiscard]] std::optional<std::size_t> indexAt(std::uint64_t timeNs) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const double elapsed = (static_cast<double>(timeNs) - static_cast<double>(*anchor)) * 1e-9 * rate();
        const auto   index   = static_cast<std::ptrdiff_t>(anchorIndex) + static_cast<std::ptrdiff_t>(std::llround(elapsed));
        return index < 0 ? std::nullopt : std::optional{static_cast<std::size_t>(index)};
    }

    [[nodiscard]] std::optional<std::uint64_t> at(std::size_t streamIndex) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const double elapsed = (static_cast<double>(streamIndex) - static_cast<double>(anchorIndex)) * 1e9 / rate();
        const auto   shift   = static_cast<std::int64_t>(std::llround(elapsed));
        return static_cast<std::uint64_t>(static_cast<std::int64_t>(*anchor) + shift);
    }
};

[[nodiscard]] inline std::optional<std::size_t> resolveDuration(const TimeBase& time, gr::Size_t samples, float seconds) {
    const std::optional<std::size_t> bySamples = samples != 0U ? std::optional{static_cast<std::size_t>(samples)} : std::nullopt;
    const std::optional<std::size_t> bySeconds = seconds > 0.f ? time.samples(static_cast<double>(seconds)) : std::nullopt;
    if (bySamples && bySeconds) {
        return std::min(*bySamples, *bySeconds);
    }
    return bySamples ? bySamples : bySeconds;
}

[[nodiscard]] inline bool withinDuration(const TimeBase& time, std::size_t span, gr::Size_t minSamples, float minSeconds, gr::Size_t maxSamples, float maxSeconds) {
    const auto low  = resolveDuration(time, minSamples, minSeconds);
    const auto high = resolveDuration(time, maxSamples, maxSeconds);
    return (!low || span >= *low) && (!high || span <= *high);
}

[[nodiscard]] inline gr::property_map datedTrigger(const TimeBase& time, std::size_t sampleIndex, UncertainValue<float> subSample, std::string_view name, std::string_view context) {
    gr::property_map tagMap{{std::string(gr::tag::TRIGGER_NAME.key()), std::string(name)}, {std::string(gr::tag::CONTEXT.key()), std::string(context)}};

    const auto stamp = time.at(sampleIndex);
    if (!stamp) {
        return tagMap;
    }
    const double nsPerSample = 1e9 / time.rate();
    const double subSampleNs = static_cast<double>(gr::value(subSample)) * nsPerSample;
    const auto   wholeNs     = static_cast<std::int64_t>(std::floor(subSampleNs));
    const double residualNs  = subSampleNs - static_cast<double>(wholeNs);

    tagMap[std::string(gr::tag::TRIGGER_TIME.key())]       = static_cast<std::uint64_t>(static_cast<std::int64_t>(*stamp) + wholeNs);
    tagMap[std::string(gr::tag::TRIGGER_OFFSET.key())]     = static_cast<float>(residualNs * 1e-9); // sub-nanosecond remainder, in seconds
    tagMap[std::string(gr::tag::TRIGGER_TIME_ERROR.key())] = static_cast<std::uint64_t>(std::llround(static_cast<double>(gr::uncertainty(subSample)) * nsPerSample));
    return tagMap;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TIMEBASE_HPP
