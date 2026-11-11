#ifndef GNURADIO_TRIGGER_TIMEBASE_HPP
#define GNURADIO_TRIGGER_TIMEBASE_HPP

#include <algorithm>
#include <array>
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
 * Stream time from the anchor in effect at a sample -- a tag's `trigger_time` shifted by `trigger_offset` -- at the nominal `sample_rate`, computed from the index so no rate can drift; unarmed, reporting rather than inventing a time, without an anchor or a rate.
 * A sample before the latest anchor is dated by the anchor it followed, of the last `kSupersededAnchors` replaced; an older one by the oldest kept.
 * A trigger dates its event, which lies up to a sample before the sample its tag sits on; so once anchored, a tag moves the
 * anchor only where it disagrees with the running time by a sample or more (a jump), and a trigger derived from this same
 * time upstream changes nothing.
 *
 *   TimeBase time; time.setRate(sample_rate); time.adopt(tagMap, streamIndex); if (const auto stamp = time.at(streamIndex + i)) { ... }
 */
struct TimeBase {
    struct Anchor {
        std::uint64_t timeNs = 0U;
        std::size_t   index  = 0UZ;
    };
    static constexpr std::size_t kSupersededAnchors = 16UZ;

    /** ns, shifted by trigger_offset */
    std::optional<std::uint64_t> anchorTimeNs;
    std::size_t                  anchorIndex = 0UZ;
    /** Hz, nominal: the sample_rate setting or tag */
    double nominalRateHz = 0.;

    std::array<Anchor, kSupersededAnchors> _superseded{}; /// oldest first
    std::size_t                            _nSuperseded = 0UZ;

    void reset() noexcept {
        anchorTimeNs = std::nullopt;
        _nSuperseded = 0UZ;
    }

    constexpr void setRate(double hz) noexcept { nominalRateHz = hz > 0. ? hz : 0.; }

    AnchorChange anchorAt(std::uint64_t timeNs, float offsetSeconds, std::size_t streamIndex) noexcept {
        const std::uint64_t dated  = timeNs + static_cast<std::uint64_t>(static_cast<std::int64_t>(std::round(static_cast<double>(offsetSeconds) * 1e9)));
        const AnchorChange  change = !anchorTimeNs ? AnchorChange::first : (dated < *anchorTimeNs ? AnchorChange::backward : AnchorChange::forward);
        if (anchorTimeNs && anchorIndex < streamIndex) {
            if (_nSuperseded == kSupersededAnchors) {
                std::shift_left(_superseded.begin(), _superseded.end(), 1);
                --_nSuperseded;
            }
            _superseded[_nSuperseded++] = Anchor{.timeNs = *anchorTimeNs, .index = anchorIndex};
        }
        anchorTimeNs = dated;
        anchorIndex  = streamIndex;
        return change;
    }

    std::optional<AnchorChange> adopt(const gr::property_map_view& tagMap, std::size_t streamIndex) {
        const float* const wireRate = tagMap.template get_if<float>(gr::tag::kSampleRate.key());
        const float* const rate     = wireRate != nullptr ? wireRate : tagMap.template get_if<float>(gr::tag::kSampleRate.shortKey());
        if (rate != nullptr && *rate > 0.f) {
            nominalRateHz = static_cast<double>(*rate);
        }
        const auto stamp = tagMap.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!stamp) {
            return std::nullopt;
        }
        const auto  offset        = tagMap.template get_if<float>(gr::tag::kTriggerOffset.key());
        const float offsetSeconds = offset ? *offset : 0.f;
        if (const auto running = at(streamIndex)) {
            const double eventNs = static_cast<double>(*stamp) + static_cast<double>(offsetSeconds) * 1e9;
            if (std::abs(static_cast<double>(*running) - eventNs) < 1e9 / nominalRateHz) {
                return std::nullopt;
            }
        }
        return anchorAt(*stamp, offsetSeconds, streamIndex);
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

    /**
     * the first sample at or after `timeNs`, the one a trigger's tag of that moment sits on; a whole-ns stamp within a nanosecond of a sample is on it;
     * a time skipped by a forward jump is on the jump's sample
     */
    [[nodiscard]] std::optional<std::size_t> indexAt(std::uint64_t timeNs) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const auto indexBy = [&](const Anchor& anchor) {
            const auto   sinceAnchorNs = static_cast<std::int64_t>(timeNs - anchor.timeNs);
            const double elapsed       = static_cast<double>(sinceAnchorNs) * 1e-9 * rate();
            const double stampSamples  = std::min(1e-9 * rate(), 0.5); // a whole-ns stamp is on its sample, but never shifts a time by a whole sample (>= 1 GS/s)
            return static_cast<std::ptrdiff_t>(anchor.index) + static_cast<std::ptrdiff_t>(std::ceil(elapsed - stampSamples));
        };
        Anchor         anchor{.timeNs = *anchorTimeNs, .index = anchorIndex};
        std::ptrdiff_t index = indexBy(anchor);
        for (std::size_t k = _nSuperseded; k > 0UZ && index < static_cast<std::ptrdiff_t>(anchor.index); --k) {
            const auto followingIndex = static_cast<std::ptrdiff_t>(anchor.index);
            anchor                    = _superseded[k - 1UZ];
            index                     = std::min(indexBy(anchor), followingIndex);
        }
        return index < 0 ? std::nullopt : std::optional{static_cast<std::size_t>(index)};
    }

    [[nodiscard]] std::optional<std::uint64_t> at(std::size_t streamIndex) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        Anchor anchor{.timeNs = *anchorTimeNs, .index = anchorIndex};
        for (std::size_t k = _nSuperseded; k > 0UZ && streamIndex < anchor.index; --k) {
            anchor = _superseded[k - 1UZ];
        }
        const double elapsed = (static_cast<double>(streamIndex) - static_cast<double>(anchor.index)) * 1e9 / rate();
        const auto   shift   = static_cast<std::int64_t>(std::round(elapsed));
        return static_cast<std::uint64_t>(static_cast<std::int64_t>(anchor.timeNs) + shift);
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

/** `trigger_time` and `trigger_offset` of the moment `samples` from the sample `sampleIndex` (negative: earlier), the offset holding the sub-ns residual; false when undated or a field did not fit */
[[nodiscard]] inline bool writeTriggerTime(gr::property_map_view& tagMap, const TimeBase& time, std::size_t sampleIndex, double samples) noexcept {
    const auto stamp = time.at(sampleIndex);
    if (!stamp) {
        return false;
    }
    const double shiftNs    = samples * 1e9 / time.rate();
    const double wholeNs    = std::floor(shiftNs);
    const double residualNs = shiftNs - wholeNs;
    return tagMap.try_emplace(gr::tag::kTriggerTime.key(), static_cast<std::uint64_t>(static_cast<std::int64_t>(*stamp) + static_cast<std::int64_t>(wholeNs))) //
           && tagMap.try_emplace(gr::tag::kTriggerOffset.key(), static_cast<float>(residualNs * 1e-9));
}

/**
 * writes a dated trigger into `tagMap` without allocating, so a kernel body can build it too: `trigger_time + trigger_offset`
 * is the event, `eventSamples` from the sample the tag sits on (in (-1, 0]: the tag sits on the first sample at or after it);
 * false when a field did not fit
 */
[[nodiscard]] inline bool writeDatedTrigger(gr::property_map_view& tagMap, const TimeBase& time, std::size_t sampleIndex, double eventSamples, float errorSamples, std::string_view name, std::string_view context) noexcept {
    const bool named = tagMap.try_emplace(gr::tag::kTriggerName.key(), name) && tagMap.try_emplace(gr::tag::kContext.key(), context);
    if (!time.datesSamples()) {
        return named;
    }
    const auto errorNs = static_cast<std::uint64_t>(std::round(static_cast<double>(errorSamples) * 1e9 / time.rate())); // not llround: AdaptiveCpp's SSCP device library has none
    return named && writeTriggerTime(tagMap, time, sampleIndex, eventSamples) && tagMap.try_emplace(gr::tag::kTriggerTimeError.key(), errorNs);
}

inline constexpr std::uint32_t kDatedTriggerKeys = 5U;

[[nodiscard]] inline gr::property_map datedTrigger(const TimeBase& time, std::size_t sampleIndex, double eventSamples, float errorSamples, std::string_view name, std::string_view context) {
    gr::property_map tagMap;
    tagMap.reserve(kDatedTriggerKeys, static_cast<std::uint32_t>(256UZ + name.size() + context.size()));
    [[maybe_unused]] const bool complete = writeDatedTrigger(tagMap, time, sampleIndex, eventSamples, errorSamples, name, context);
    assert(complete && "the reservation holds every field");
    return tagMap;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TIMEBASE_HPP
