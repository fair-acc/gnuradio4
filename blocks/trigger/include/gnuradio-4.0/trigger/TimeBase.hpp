#ifndef GNURADIO_TRIGGER_TIMEBASE_HPP
#define GNURADIO_TRIGGER_TIMEBASE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::trigger {

enum class AnchorChange : std::uint8_t { first, forward, backward };

/**
 * Stream time from the latest `trigger_time` anchor (shifted by `trigger_offset`) at the nominal `sample_rate`, computed from the index so no rate can drift; unarmed, reporting rather than inventing a time, without an anchor or a rate.
 *
 *   TimeBase time; time.setRate(sample_rate); time.adopt(tagMap, streamIndex); if (const auto stamp = time.at(streamIndex + i)) { ... }
 */
struct TimeBase {
    /** ns, shifted by trigger_offset */
    std::optional<std::uint64_t> anchorTimeNs;
    std::size_t                  anchorIndex = 0UZ;
    /** Hz, nominal: the sample_rate setting or tag */
    double nominalRateHz = 0.;

    void reset() noexcept { anchorTimeNs = std::nullopt; }

    constexpr void setRate(double hz) noexcept { nominalRateHz = hz > 0. ? hz : 0.; }

    AnchorChange anchorAt(std::uint64_t timeNs, float offsetSeconds, std::size_t streamIndex) noexcept {
        const std::uint64_t dated  = timeNs + static_cast<std::uint64_t>(static_cast<std::int64_t>(std::round(static_cast<double>(offsetSeconds) * 1e9)));
        const AnchorChange  change = !anchorTimeNs ? AnchorChange::first : (dated < *anchorTimeNs ? AnchorChange::backward : AnchorChange::forward);
        anchorTimeNs               = dated;
        anchorIndex                = streamIndex;
        return change;
    }

    std::optional<AnchorChange> adopt(const gr::property_map_view& tagMap, std::size_t streamIndex) {
        const float* const wireRate = tagMap.template get_if<float>(gr::tag::SAMPLE_RATE.key());
        const float* const rate     = wireRate != nullptr ? wireRate : tagMap.template get_if<float>(gr::tag::SAMPLE_RATE.shortKey());
        if (rate != nullptr && *rate > 0.f) {
            nominalRateHz = static_cast<double>(*rate);
        }
        const auto stamp = tagMap.template get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        if (!stamp) {
            return std::nullopt;
        }
        const auto offset = tagMap.template get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        return anchorAt(*stamp, offset ? *offset : 0.f, streamIndex);
    }

    [[nodiscard]] constexpr double rate() const noexcept { return nominalRateHz; }

    [[nodiscard]] constexpr bool measuresDuration() const noexcept { return rate() > 0.; }
    [[nodiscard]] constexpr bool datesSamples() const noexcept { return anchorTimeNs.has_value() && rate() > 0.; }

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
        return static_cast<std::size_t>(std::round(durationSeconds * rate()));
    }

    [[nodiscard]] std::optional<std::size_t> indexAt(std::uint64_t timeNs) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const auto   sinceAnchorNs = static_cast<std::int64_t>(timeNs - *anchorTimeNs);
        const double elapsed       = static_cast<double>(sinceAnchorNs) * 1e-9 * rate();
        const auto   index         = static_cast<std::ptrdiff_t>(anchorIndex) + static_cast<std::ptrdiff_t>(std::round(elapsed));
        return index < 0 ? std::nullopt : std::optional{static_cast<std::size_t>(index)};
    }

    [[nodiscard]] std::optional<std::uint64_t> at(std::size_t streamIndex) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const double elapsed = (static_cast<double>(streamIndex) - static_cast<double>(anchorIndex)) * 1e9 / rate();
        const auto   shift   = static_cast<std::int64_t>(std::round(elapsed));
        return static_cast<std::uint64_t>(static_cast<std::int64_t>(*anchorTimeNs) + shift);
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

/**
 * writes a dated trigger into `tagMap` without allocating, so a kernel body can build it too; false when a field did
 * not fit the map's capacity
 */
[[nodiscard]] inline bool writeDatedTrigger(gr::property_map_view& tagMap, const TimeBase& time, std::size_t sampleIndex, UncertainValue<float> subSample, std::string_view name, std::string_view context) noexcept {
    const bool named = tagMap.try_emplace(gr::tag::TRIGGER_NAME.key(), name) && tagMap.try_emplace(gr::tag::CONTEXT.key(), context);
    const auto stamp = time.at(sampleIndex);
    if (!stamp) {
        return named;
    }
    const double nsPerSample = 1e9 / time.rate();
    const double subSampleNs = static_cast<double>(gr::value(subSample)) * nsPerSample;
    const auto   wholeNs     = static_cast<std::int64_t>(std::floor(subSampleNs));
    const double residualNs  = subSampleNs - static_cast<double>(wholeNs);
    return named && tagMap.try_emplace(gr::tag::TRIGGER_TIME.key(), static_cast<std::uint64_t>(static_cast<std::int64_t>(*stamp) + wholeNs)) //
           && tagMap.try_emplace(gr::tag::TRIGGER_OFFSET.key(), static_cast<float>(residualNs * 1e-9))                                       //
           && tagMap.try_emplace(gr::tag::TRIGGER_TIME_ERROR.key(), static_cast<std::uint64_t>(std::round(static_cast<double>(gr::uncertainty(subSample)) * nsPerSample)));
}

inline constexpr std::uint32_t kDatedTriggerKeys = 5U;

[[nodiscard]] inline gr::property_map datedTrigger(const TimeBase& time, std::size_t sampleIndex, UncertainValue<float> subSample, std::string_view name, std::string_view context) {
    gr::property_map tagMap;
    tagMap.reserve(kDatedTriggerKeys, static_cast<std::uint32_t>(256UZ + name.size() + context.size()));
    [[maybe_unused]] const bool complete = writeDatedTrigger(tagMap, time, sampleIndex, subSample, name, context);
    assert(complete && "the reservation holds every field");
    return tagMap;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TIMEBASE_HPP
