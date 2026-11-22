#ifndef GNURADIO_TRIGGER_TIMEBASE_HPP
#define GNURADIO_TRIGGER_TIMEBASE_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::trigger {

enum class AnchorChange : std::uint8_t { first, forward, backward };

namespace detail {
/// floor(a·b / c) and its remainder, exact without a 128-bit type (wasm32, SYCL device code and WGSL have none); the quotient must fit 64 bits
[[nodiscard]] constexpr std::pair<std::uint64_t, std::uint64_t> mulDiv(std::uint64_t a, std::uint64_t b, std::uint64_t c) noexcept {
    constexpr std::uint64_t kLowHalf = 0xFFFF'FFFFU;
    const std::uint64_t     lowLow   = (a & kLowHalf) * (b & kLowHalf);
    const std::uint64_t     lowHigh  = (a & kLowHalf) * (b >> 32U);
    const std::uint64_t     highLow  = (a >> 32U) * (b & kLowHalf);
    const std::uint64_t     middle   = (lowLow >> 32U) + (lowHigh & kLowHalf) + (highLow & kLowHalf);
    const std::uint64_t     high     = ((a >> 32U) * (b >> 32U)) + (lowHigh >> 32U) + (highLow >> 32U) + (middle >> 32U);
    const std::uint64_t     low      = (middle << 32U) | (lowLow & kLowHalf);
    std::uint64_t           quotient = 0U;
    std::uint64_t           rest     = high; // < c, as the quotient fits 64 bits
    for (std::uint32_t bit = 64U; bit-- > 0U;) {
        const bool wraps = (rest >> 63U) != 0U;
        rest             = (rest << 1U) | ((low >> bit) & 1U);
        quotient <<= 1U;
        if (wraps || rest >= c) {
            rest -= c;
            quotient |= 1U;
        }
    }
    return {quotient, rest};
}

struct ExactFloat {
    std::uint64_t mantissa = 0U;
    std::int32_t  exponent = 0;
    bool          negative = false;
};

[[nodiscard]] constexpr ExactFloat exactly(float value) noexcept {
    const auto          bits     = std::bit_cast<std::uint32_t>(value);
    const auto          biased   = static_cast<std::int32_t>((bits >> 23U) & 0xFFU);
    const std::uint64_t fraction = bits & 0x7F'FFFFU;
    return {.mantissa = biased == 0 ? fraction : (fraction | 0x80'0000U), .exponent = std::max(biased, 1) - 150, .negative = (bits >> 31U) != 0U};
}

/// round(a · b) of two floats, exact as their mantissas multiply within 48 bits, half away from zero as `std::round`; |a · b| < 2^63
[[nodiscard]] constexpr std::int64_t roundedProduct(float a, float b) noexcept {
    const ExactFloat    x         = exactly(a);
    const ExactFloat    y         = exactly(b);
    const std::uint64_t product   = x.mantissa * y.mantissa;
    const std::int32_t  exponent  = x.exponent + y.exponent;
    const auto          magnitude = [&]() -> std::uint64_t {
        if (product == 0U || exponent <= -64) {
            return 0U;
        }
        if (exponent >= 0) {
            return product << static_cast<std::uint32_t>(exponent);
        }
        const auto shift = static_cast<std::uint32_t>(-exponent);
        return (product + (std::uint64_t{1} << (shift - 1U))) >> shift;
    }();
    return x.negative != y.negative ? -static_cast<std::int64_t>(magnitude) : static_cast<std::int64_t>(magnitude);
}

[[nodiscard]] constexpr std::uint64_t magnitude(std::int64_t value) noexcept { return value < 0 ? 0U - static_cast<std::uint64_t>(value) : static_cast<std::uint64_t>(value); }
} // namespace detail

/**
 * Stream time from the anchor in effect at a sample -- a tag's `trigger_time` shifted by `trigger_offset` -- at the nominal `sample_rate`, computed from the index so no rate can drift; unarmed, reporting rather than inventing a time, without an anchor or a rate.
 * Time is whole nanoseconds, exact for rates in [2^-11, 2^41) Hz: a float rate is m·2^e, so `_spanSamples` samples last exactly `_spanNs` ns, and every target and GPU computes the same stamps; a higher rate is no rate.
 * A sample before the latest anchor is dated by the anchor it followed, of the last `kSupersededAnchors` replaced; an older one by the oldest kept.
 * A trigger dates its event, up to a sample before its tag's sample, so once anchored a tag moves the anchor only where it disagrees with the running time by a sample or more (a jump); a trigger derived upstream from this same time changes nothing.
 *
 *   TimeBase time; time.setRate(sample_rate); time.adopt(tagMap, streamIndex); if (const auto stamp = time.at(streamIndex + i)) { ... }
 */
struct TimeBase {
    struct Anchor {
        std::uint64_t timeNs = 0U;
        std::size_t   index  = 0UZ;
    };
    /// |samples| sample periods: `quotient + remainder / divisor` ns
    struct ExactNs {
        std::uint64_t quotient  = 0U;
        std::uint64_t remainder = 0U;
        std::uint64_t divisor   = 1U;
        bool          negative  = false;
    };
    struct FlooredNs {
        std::int64_t ns         = 0;
        float        residualNs = 0.f; /// [0, 1)
    };
    static constexpr std::size_t   kSupersededAnchors = 16UZ;
    static constexpr std::uint32_t kMaxFractionBits   = 32U;     /// of a sample; finer fractions are truncated
    static constexpr float         kMaxRateHz         = 0x1p41f; /// exclusive: above it a span overflows 64 bits
    static constexpr float         kMaxOffsetSeconds  = 0x1p33f; /// exclusive: 2^33 s in ns stays below 2^63

    /** ns, shifted by trigger_offset */
    std::optional<std::uint64_t> anchorTimeNs;
    std::size_t                  anchorIndex = 0UZ;
    /** Hz, nominal: the sample_rate setting or tag */
    float nominalRateHz = 0.f;

    std::array<Anchor, kSupersededAnchors> _superseded{}; /// oldest first
    std::size_t                            _nSuperseded = 0UZ;
    std::uint64_t                          _spanSamples = 0U; /// 0 without a rate
    std::uint64_t                          _spanNs      = 0U;

    void reset() noexcept {
        anchorTimeNs = std::nullopt;
        _nSuperseded = 0UZ;
    }

    constexpr void setRate(float hz) noexcept {
        nominalRateHz                         = hz > 0.f && hz < kMaxRateHz ? hz : 0.f;
        const detail::ExactFloat rate         = detail::exactly(nominalRateHz);
        const auto               zeros        = rate.mantissa == 0U ? 0 : std::countr_zero(rate.mantissa);
        const std::uint64_t      odd          = rate.mantissa >> zeros;
        const std::int32_t       twos         = rate.exponent + zeros;
        const std::int32_t       shared       = std::clamp(twos, 0, 9); // 1e9 = 2^9 · 5^9: shared powers of two cancel
        const auto               fractionBits = static_cast<std::uint32_t>(std::max(-twos, 0));
        const std::uint32_t      dropped      = fractionBits - std::min(fractionBits, 34U); // below 2^-11 Hz, 1e9 · 2^fractionBits would exceed 64 bits
        _spanSamples                          = dropped >= 24U ? 0U : (odd >> dropped) << static_cast<std::uint32_t>(std::max(twos, 0) - shared);
        _spanNs                               = (std::uint64_t{1'000'000'000U} >> static_cast<std::uint32_t>(shared)) << (fractionBits - dropped);
    }

    [[nodiscard]] static constexpr std::int64_t offsetNs(float seconds) noexcept { return seconds > -kMaxOffsetSeconds && seconds < kMaxOffsetSeconds ? detail::roundedProduct(seconds, 1e9f) : 0; }

    AnchorChange anchorAt(std::uint64_t timeNs, float offsetSeconds, std::size_t streamIndex) noexcept {
        const std::uint64_t dated  = timeNs + static_cast<std::uint64_t>(offsetNs(offsetSeconds));
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
            setRate(*rate);
        }
        const auto stamp = tagMap.template get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        if (!stamp) {
            return std::nullopt;
        }
        const auto  offset        = tagMap.template get_if<float>(gr::tag::kTriggerOffset.key());
        const float offsetSeconds = offset ? *offset : 0.f;
        if (const auto running = at(streamIndex)) {
            const std::uint64_t eventNs = *stamp + static_cast<std::uint64_t>(offsetNs(offsetSeconds));
            const std::uint64_t apartNs = *running > eventNs ? *running - eventNs : eventNs - *running;
            if (apartNs < _spanNs && detail::mulDiv(apartNs, _spanSamples, _spanNs).first == 0U) { // less than one sample period
                return std::nullopt;
            }
        }
        return anchorAt(*stamp, offsetSeconds, streamIndex);
    }

    [[nodiscard]] constexpr float rate() const noexcept { return nominalRateHz; }

    [[nodiscard]] constexpr bool measuresDuration() const noexcept { return _spanSamples > 0U; }
    [[nodiscard]] constexpr bool datesSamples() const noexcept { return anchorTimeNs.has_value() && measuresDuration(); }

    [[nodiscard]] constexpr std::optional<double> seconds(std::size_t nSamples) const noexcept {
        if (!measuresDuration()) {
            return std::nullopt;
        }
        return static_cast<double>(nSamples) / static_cast<double>(rate());
    }

    /// as `round(durationSeconds · rate)` in double, which the product of two floats is exactly
    [[nodiscard]] constexpr std::optional<std::size_t> samples(float durationSeconds) const noexcept {
        if (!measuresDuration() || durationSeconds < 0.f) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(detail::roundedProduct(durationSeconds, rate()));
    }

    [[nodiscard]] constexpr std::optional<std::size_t> samples(double durationSeconds) const noexcept {
        if (!measuresDuration() || durationSeconds < 0.) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::round(durationSeconds * static_cast<double>(rate())));
    }

    /// `wholeSamples + fractionSamples` sample periods, exactly to 2^-kMaxFractionBits of a sample; needs a rate
    [[nodiscard]] constexpr ExactNs exactNs(std::int64_t wholeSamples, float fractionSamples) const noexcept {
        struct Count {
            std::int64_t  units        = 0; /// of 2^-fractionBits samples
            std::uint32_t fractionBits = 0U;
        };
        const Count count = [&]() -> Count {
            const detail::ExactFloat fraction = detail::exactly(fractionSamples);
            if (fraction.mantissa == 0U) {
                return {.units = wholeSamples, .fractionBits = 0U};
            }
            const auto          zeros     = std::countr_zero(fraction.mantissa);
            const std::uint64_t odd       = fraction.mantissa >> zeros;
            const std::int32_t  twos      = fraction.exponent + zeros;
            const auto          signedOdd = [&](std::uint64_t value) { return fraction.negative ? -static_cast<std::int64_t>(value) : static_cast<std::int64_t>(value); };
            if (twos >= 0) {
                return {.units = wholeSamples + signedOdd(odd << static_cast<std::uint32_t>(twos)), .fractionBits = 0U};
            }
            const auto          room = 62U - std::min(static_cast<std::uint32_t>(std::bit_width(detail::magnitude(wholeSamples))), 62U); // so that `units` fits 63 bits
            const std::uint32_t bits = std::min({static_cast<std::uint32_t>(-twos), kMaxFractionBits, room});
            const std::uint32_t lost = static_cast<std::uint32_t>(-twos) - bits;
            const std::uint64_t kept = lost >= 64U ? 0U : odd >> lost;
            return {.units = wholeSamples * (std::int64_t{1} << bits) + signedOdd(kept), .fractionBits = bits};
        }();
        const std::uint64_t divisor      = _spanSamples << count.fractionBits;
        const auto [quotient, remainder] = detail::mulDiv(detail::magnitude(count.units), _spanNs, divisor);
        return {.quotient = quotient, .remainder = remainder, .divisor = divisor, .negative = count.units < 0};
    }

    /// half away from zero, as `std::round`
    [[nodiscard]] static constexpr std::int64_t rounded(const ExactNs& ns) noexcept {
        const std::uint64_t magnitude = ns.quotient + (ns.remainder >= ns.divisor - ns.remainder ? 1U : 0U);
        return ns.negative ? -static_cast<std::int64_t>(magnitude) : static_cast<std::int64_t>(magnitude);
    }

    [[nodiscard]] static constexpr FlooredNs floored(const ExactNs& ns) noexcept {
        const bool          below = ns.negative && ns.remainder != 0U;
        const std::uint64_t rest  = below ? ns.divisor - ns.remainder : ns.remainder;
        const auto          whole = static_cast<std::int64_t>(ns.quotient + (below ? 1U : 0U));
        const auto          units = static_cast<std::uint32_t>(detail::mulDiv(rest, std::uint64_t{1} << 24U, ns.divisor).first); // the residual to 24 bits, truncated alike everywhere
        return {.ns = ns.negative ? -whole : whole, .residualNs = static_cast<float>(units) * 0x1p-24f};
    }

    /// the whole samples from an anchor to the first sample at or after `sinceAnchorNs`: a whole-ns stamp within a nanosecond, and at most half a sample, after a sample is on it
    [[nodiscard]] constexpr std::int64_t samplesOnOrAfter(std::int64_t sinceAnchorNs) const noexcept {
        if (2U * _spanSamples <= _spanNs) { // ceil((ns - 1) · rate): up to 0.5 GS/s a nanosecond is at most half a sample
            const std::int64_t shifted = sinceAnchorNs - 1;
            const auto [samples, rest] = detail::mulDiv(detail::magnitude(shifted), _spanSamples, _spanNs);
            const auto wholeAfter      = static_cast<std::int64_t>(samples + (rest != 0U ? 1U : 0U));
            return shifted >= 0 ? wholeAfter : -static_cast<std::int64_t>(samples);
        }
        const auto [samples, rest] = detail::mulDiv(detail::magnitude(sinceAnchorNs), _spanSamples, _spanNs); // ceil(ns · rate - 1/2)
        const bool pastHalf        = sinceAnchorNs >= 0 ? rest > _spanNs - rest : rest >= _spanNs - rest;
        const auto whole           = static_cast<std::int64_t>(samples + (pastHalf ? 1U : 0U));
        return sinceAnchorNs >= 0 ? whole : -whole;
    }

    /**
     * the first sample at or after `timeNs`, the one a trigger's tag of that moment sits on; a whole-ns stamp within a nanosecond of a sample is on it;
     * a time skipped by a forward jump is on the jump's sample
     */
    [[nodiscard]] std::optional<std::size_t> indexAt(std::uint64_t timeNs) const noexcept {
        if (!datesSamples()) {
            return std::nullopt;
        }
        const auto   indexBy = [&](const Anchor& anchor) { return static_cast<std::int64_t>(anchor.index) + samplesOnOrAfter(static_cast<std::int64_t>(timeNs - anchor.timeNs)); };
        Anchor       anchor{.timeNs = *anchorTimeNs, .index = anchorIndex};
        std::int64_t index = indexBy(anchor);
        for (std::size_t k = _nSuperseded; k > 0UZ && index < static_cast<std::int64_t>(anchor.index); --k) {
            const auto followingIndex = static_cast<std::int64_t>(anchor.index);
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
        const std::int64_t elapsed = static_cast<std::int64_t>(streamIndex) - static_cast<std::int64_t>(anchor.index);
        return anchor.timeNs + static_cast<std::uint64_t>(rounded(exactNs(elapsed, 0.f)));
    }
};

[[nodiscard]] inline std::optional<std::size_t> resolveDuration(const TimeBase& time, gr::Size_t samples, float seconds) {
    const std::optional<std::size_t> bySamples = samples != 0U ? std::optional{static_cast<std::size_t>(samples)} : std::nullopt;
    const std::optional<std::size_t> bySeconds = seconds > 0.f ? time.samples(seconds) : std::nullopt;
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
 * `trigger_time` and `trigger_offset` of the moment `wholeSamples + fractionSamples` from the sample `sampleIndex` (negative: earlier), counted from that sample's stamp,
 * the offset holding the sub-ns residual; false when undated or a field did not fit
 */
[[nodiscard]] inline bool writeTriggerTime(gr::property_map_view& tagMap, const TimeBase& time, std::size_t sampleIndex, std::int64_t wholeSamples, float fractionSamples) noexcept {
    const auto stamp = time.at(sampleIndex);
    if (!stamp) {
        return false;
    }
    const TimeBase::FlooredNs shift = TimeBase::floored(time.exactNs(wholeSamples, fractionSamples));
    return tagMap.try_emplace(gr::tag::kTriggerTime.key(), *stamp + static_cast<std::uint64_t>(shift.ns)) //
           && tagMap.try_emplace(gr::tag::kTriggerOffset.key(), shift.residualNs * 1e-9f);
}

/**
 * writes a dated trigger into `tagMap` without allocating, so a kernel body can build it too: `trigger_time + trigger_offset`
 * is the event, `eventSamples` from the sample the tag sits on (in (-1, 0]: the tag sits on the first sample at or after it);
 * false when a field did not fit
 */
[[nodiscard]] inline bool writeDatedTrigger(gr::property_map_view& tagMap, const TimeBase& time, std::size_t sampleIndex, float eventSamples, float errorSamples, std::string_view name, std::string_view context) noexcept {
    const bool named = tagMap.try_emplace(gr::tag::kTriggerName.key(), name) && tagMap.try_emplace(gr::tag::kContext.key(), context);
    if (!time.datesSamples()) {
        return named;
    }
    const auto errorNs = static_cast<std::uint64_t>(TimeBase::rounded(time.exactNs(0, errorSamples)));
    return named && writeTriggerTime(tagMap, time, sampleIndex, 0, eventSamples) && tagMap.try_emplace(gr::tag::kTriggerTimeError.key(), errorNs);
}

inline constexpr std::uint32_t kDatedTriggerKeys = 5U;

[[nodiscard]] inline gr::property_map datedTrigger(const TimeBase& time, std::size_t sampleIndex, float eventSamples, float errorSamples, std::string_view name, std::string_view context) {
    gr::property_map tagMap;
    tagMap.reserve(kDatedTriggerKeys, static_cast<std::uint32_t>(256UZ + name.size() + context.size()));
    [[maybe_unused]] const bool complete = writeDatedTrigger(tagMap, time, sampleIndex, eventSamples, errorSamples, name, context);
    assert(complete && "the reservation holds every field");
    return tagMap;
}

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_TIMEBASE_HPP
