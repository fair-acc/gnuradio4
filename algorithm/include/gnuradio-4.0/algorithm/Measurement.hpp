#ifndef GNURADIO_ALGORITHM_MEASUREMENT_HPP
#define GNURADIO_ALGORITHM_MEASUREMENT_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>

namespace gr::algorithm::measurement {

/**
 * Oscilloscope measurements over a gate of samples (IEEE 181-2011), each one function with an analytic ground truth.
 * Amplitudes are in the samples' unit, times in samples (a caller multiplies by the sample interval), ratios in percent.
 * A gate that cannot carry the quantity says why instead of returning a number: no NaN, no zero stands in for a result.
 *   const auto rise = transitionTime(std::span{samples}, true); if (rise) { seconds = *rise / sampleRate; }
 */
enum class Invalid : std::uint8_t { tooFewSamples, nonFinite, noTwoStates, noTransition };

using Measured = std::expected<double, Invalid>;

/** IEEE 181: the reference levels of a pulse as fractions of its amplitude, low and high for transitions, mid for widths */
struct ReferenceLevels {
    double low  = 0.1;
    double mid  = 0.5;
    double high = 0.9;
};

inline constexpr double      kStateProminence = 2.;    /// a histogram mode holds at least this many times the mean of the other occupied bins of its half
inline constexpr std::size_t kHistogramBins   = 100UZ; /// IEEE 181's histogram resolution between the gate's extremes

struct TopBase {
    double top;
    double base;

    [[nodiscard]] constexpr double amplitude() const noexcept { return top - base; }
    [[nodiscard]] constexpr double level(double fraction) const noexcept { return base + fraction * amplitude(); }
};

namespace detail {
template<typename T>
[[nodiscard]] std::expected<void, Invalid> usable(std::span<const T> samples, std::size_t minimum = 1UZ) {
    if (samples.size() < minimum) {
        return std::unexpected(Invalid::tooFewSamples);
    }
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::ranges::all_of(samples, [](T x) { return std::isfinite(x); })) {
            return std::unexpected(Invalid::nonFinite);
        }
    }
    return {};
}

/** the crossings of `level`, in fractional samples, with the trigger's own boundary interpolation (zero hysteresis) */
template<typename T>
[[nodiscard]] std::vector<std::pair<double, bool>> crossings(std::span<const T> samples, double level) {
    gr::trigger::SchmittTrigger<double, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, 2UZ> comparator({.level = level, .hysteresis = 0.});
    std::vector<std::pair<double, bool>>                                                                   found; /// position, rising
    for (std::size_t i = 0UZ; i < samples.size(); ++i) {
        if (const auto edge = comparator.processOne(static_cast<double>(samples[i])); edge != gr::trigger::EdgeDetection::NONE) {
            found.emplace_back(static_cast<double>(i) + static_cast<double>(gr::value(comparator.lastEdgePosition)), edge == gr::trigger::EdgeDetection::RISING);
        }
    }
    return found;
}

[[nodiscard]] inline std::optional<double> firstAfter(const std::vector<std::pair<double, bool>>& found, double after, bool rising) {
    const auto it = std::ranges::find_if(found, [&](const auto& crossing) { return crossing.first > after && crossing.second == rising; });
    return it == found.end() ? std::nullopt : std::optional{it->first};
}
} // namespace detail

template<typename T>
[[nodiscard]] Measured minimum(std::span<const T> samples) {
    return detail::usable(samples).transform([&] { return static_cast<double>(std::ranges::min(samples)); });
}

template<typename T>
[[nodiscard]] Measured maximum(std::span<const T> samples) {
    return detail::usable(samples).transform([&] { return static_cast<double>(std::ranges::max(samples)); });
}

template<typename T>
[[nodiscard]] Measured peakToPeak(std::span<const T> samples) {
    return detail::usable(samples).transform([&] {
        const auto [low, high] = std::ranges::minmax(samples);
        return static_cast<double>(high) - static_cast<double>(low);
    });
}

template<typename T>
[[nodiscard]] Measured mean(std::span<const T> samples) {
    return detail::usable(samples).transform([&] { return std::ranges::fold_left(samples | std::views::transform([](T x) { return static_cast<double>(x); }), 0., std::plus{}) / static_cast<double>(samples.size()); });
}

template<typename T>
[[nodiscard]] Measured rms(std::span<const T> samples) {
    const auto squared = [](T x) { return static_cast<double>(x) * static_cast<double>(x); };
    return detail::usable(samples).transform([&] { return std::sqrt(std::ranges::fold_left(samples | std::views::transform(squared), 0., std::plus{}) / static_cast<double>(samples.size())); });
}

namespace detail {
template<typename T>
[[nodiscard]] double squaredDeviations(std::span<const T> samples, double average) {
    return std::ranges::fold_left(samples | std::views::transform([average](T x) { return (static_cast<double>(x) - average) * (static_cast<double>(x) - average); }), 0., std::plus{});
}
} // namespace detail

/** the population standard deviation, divisor n: the RMS once the gate's mean is removed (the AC RMS), two-pass so a large offset does not cancel it */
template<typename T>
[[nodiscard]] Measured populationStdDev(std::span<const T> samples) {
    return mean(samples).transform([&](double average) { return std::sqrt(detail::squaredDeviations(samples, average) / static_cast<double>(samples.size())); });
}

/** the sample standard deviation, divisor n - 1: a gate of one sample has none */
template<typename T>
[[nodiscard]] Measured sampleStdDev(std::span<const T> samples) {
    return detail::usable(samples, 2UZ).and_then([&]() -> Measured { return std::sqrt(detail::squaredDeviations(samples, *mean(samples)) / static_cast<double>(samples.size() - 1UZ)); });
}

template<typename T>
[[nodiscard]] Measured median(std::span<const T> samples) {
    return detail::usable(samples).transform([&] {
        std::vector<double> sorted(samples.begin(), samples.end());
        const std::size_t   half = sorted.size() / 2UZ;
        std::ranges::nth_element(sorted, sorted.begin() + static_cast<std::ptrdiff_t>(half));
        if (sorted.size() % 2UZ == 1UZ) {
            return sorted[half];
        }
        const double upper = sorted[half];
        return 0.5 * (*std::ranges::max_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(half)) + upper);
    });
}

/**
 * IEEE 181-2011 §5.2, histogram mode method: the samples are binned between their extremes, base is the mean of the
 * fullest bin in the lower half, top of the fullest bin in the upper half; a gate without two states (a constant, a ramp) has neither
 */
template<typename T>
[[nodiscard]] std::expected<TopBase, Invalid> topBase(std::span<const T> samples) {
    if (const auto ok = detail::usable(samples, 2UZ); !ok) {
        return std::unexpected(ok.error());
    }
    const auto [lowest, highest] = std::ranges::minmax(samples);
    const double low             = static_cast<double>(lowest);
    const double span            = static_cast<double>(highest) - low;
    if (span <= 0.) {
        return std::unexpected(Invalid::noTwoStates);
    }
    std::vector<std::size_t> counts(kHistogramBins, 0UZ);
    std::vector<double>      sums(kHistogramBins, 0.);
    for (const T x : samples) {
        const auto bin = std::min(kHistogramBins - 1UZ, static_cast<std::size_t>((static_cast<double>(x) - low) / span * static_cast<double>(kHistogramBins)));
        ++counts[bin];
        sums[bin] += static_cast<double>(x);
    }
    const auto fullest = [&](std::size_t from, std::size_t to) {
        const auto it = std::max_element(counts.begin() + static_cast<std::ptrdiff_t>(from), counts.begin() + static_cast<std::ptrdiff_t>(to));
        return static_cast<std::size_t>(std::distance(counts.begin(), it));
    };
    const std::size_t lower = fullest(0UZ, kHistogramBins / 2UZ);
    const std::size_t upper = fullest(kHistogramBins / 2UZ, kHistogramBins);
    if (counts[lower] == 0UZ || counts[upper] == 0UZ) {
        return std::unexpected(Invalid::noTwoStates);
    }
    // a state is a mode that towers over the other occupied bins of its half; a ramp's bins all hold alike
    const auto stands = [&](std::size_t from, std::size_t to, std::size_t mode) {
        std::size_t others = 0UZ;
        std::size_t held   = 0UZ;
        for (std::size_t bin = from; bin < to; ++bin) {
            if (bin != mode && counts[bin] > 0UZ) {
                ++others;
                held += counts[bin];
            }
        }
        return others == 0UZ || static_cast<double>(counts[mode]) >= kStateProminence * static_cast<double>(held) / static_cast<double>(others);
    };
    if (!stands(0UZ, kHistogramBins / 2UZ, lower) || !stands(kHistogramBins / 2UZ, kHistogramBins, upper)) {
        return std::unexpected(Invalid::noTwoStates);
    }
    return TopBase{.top = sums[upper] / static_cast<double>(counts[upper]), .base = sums[lower] / static_cast<double>(counts[lower])};
}

template<typename T>
[[nodiscard]] Measured top(std::span<const T> samples) {
    return topBase(samples).transform([](const TopBase& levels) { return levels.top; });
}

template<typename T>
[[nodiscard]] Measured base(std::span<const T> samples) {
    return topBase(samples).transform([](const TopBase& levels) { return levels.base; });
}

template<typename T>
[[nodiscard]] Measured amplitude(std::span<const T> samples) {
    return topBase(samples).transform([](const TopBase& levels) { return levels.amplitude(); });
}

/** samples from the first `low` to the next `high` reference-level crossing on a rising transition (`rising`), else the mirror */
template<typename T>
[[nodiscard]] Measured transitionTime(std::span<const T> samples, bool rising, ReferenceLevels levels = {}) {
    return topBase(samples).and_then([&](const TopBase& states) -> Measured {
        const auto  lowCrossings  = detail::crossings(samples, states.level(levels.low));
        const auto  highCrossings = detail::crossings(samples, states.level(levels.high));
        const auto& from          = rising ? lowCrossings : highCrossings;
        const auto& to            = rising ? highCrossings : lowCrossings;
        const auto  start         = detail::firstAfter(from, -1., rising);
        const auto  stop          = start ? detail::firstAfter(to, *start, rising) : std::nullopt;
        if (!stop) {
            return std::unexpected(Invalid::noTransition);
        }
        return *stop - *start;
    });
}

/** from the first mid-level crossing of the given polarity to the next one of the other: a positive pulse starts rising */
template<typename T>
[[nodiscard]] Measured width(std::span<const T> samples, bool positive, ReferenceLevels levels = {}) {
    return topBase(samples).and_then([&](const TopBase& states) -> Measured {
        const auto found = detail::crossings(samples, states.level(levels.mid));
        const auto start = detail::firstAfter(found, -1., positive);
        const auto stop  = start ? detail::firstAfter(found, *start, !positive) : std::nullopt;
        if (!stop) {
            return std::unexpected(Invalid::noTransition);
        }
        return *stop - *start;
    });
}

/** positive width over the period between the first two rising mid-level crossings, in percent */
template<typename T>
[[nodiscard]] Measured dutyCycle(std::span<const T> samples, ReferenceLevels levels = {}) {
    return topBase(samples).and_then([&](const TopBase& states) -> Measured {
        const auto found  = detail::crossings(samples, states.level(levels.mid));
        const auto first  = detail::firstAfter(found, -1., true);
        const auto fall   = first ? detail::firstAfter(found, *first, false) : std::nullopt;
        const auto second = first ? detail::firstAfter(found, *first, true) : std::nullopt;
        if (!fall || !second) {
            return std::unexpected(Invalid::noTransition);
        }
        return 100. * (*fall - *first) / (*second - *first);
    });
}

/** samples between the first two rising mid-level crossings */
template<typename T>
[[nodiscard]] Measured period(std::span<const T> samples, ReferenceLevels levels = {}) {
    return topBase(samples).and_then([&](const TopBase& states) -> Measured {
        const auto found  = detail::crossings(samples, states.level(levels.mid));
        const auto first  = detail::firstAfter(found, -1., true);
        const auto second = first ? detail::firstAfter(found, *first, true) : std::nullopt;
        if (!second) {
            return std::unexpected(Invalid::noTransition);
        }
        return *second - *first;
    });
}

/** cycles per sample, the inverse of `period`; a caller multiplies by the sample rate */
template<typename T>
[[nodiscard]] Measured frequency(std::span<const T> samples, ReferenceLevels levels = {}) {
    return period(samples, levels).transform([](double samplesPerCycle) { return 1. / samplesPerCycle; });
}

/** samples from the first rising mid-level crossing of `reference` to that of `delayed`, each about its own top and base: positive when `delayed` lags */
template<typename T>
[[nodiscard]] Measured delay(std::span<const T> reference, std::span<const T> delayed, ReferenceLevels levels = {}) {
    const auto firstRising = [&](std::span<const T> samples) -> Measured {
        return topBase(samples).and_then([&](const TopBase& states) -> Measured {
            const auto start = detail::firstAfter(detail::crossings(samples, states.level(levels.mid)), -1., true);
            return start ? Measured{*start} : std::unexpected(Invalid::noTransition);
        });
    };
    return firstRising(reference).and_then([&](double from) { return firstRising(delayed).transform([from](double to) { return to - from; }); });
}

/** degrees `delayed` lags `reference`, in (-180, 180], the delay over the reference's period */
template<typename T>
[[nodiscard]] Measured phase(std::span<const T> reference, std::span<const T> delayed, ReferenceLevels levels = {}) {
    return delay(reference, delayed, levels).and_then([&](double lag) -> Measured {
        return period(reference, levels).transform([&](double samplesPerCycle) {
            const double degrees = std::remainder(360. * lag / samplesPerCycle, 360.);
            return degrees <= -180. ? degrees + 360. : degrees;
        });
    });
}

/** the excursion beyond top, in percent of the amplitude */
template<typename T>
[[nodiscard]] Measured overshoot(std::span<const T> samples) {
    return topBase(samples).transform([&](const TopBase& states) { return 100. * (static_cast<double>(std::ranges::max(samples)) - states.top) / states.amplitude(); });
}

/** the excursion below base, in percent of the amplitude */
template<typename T>
[[nodiscard]] Measured undershoot(std::span<const T> samples) {
    return topBase(samples).transform([&](const TopBase& states) { return 100. * (states.base - static_cast<double>(std::ranges::min(samples))) / states.amplitude(); });
}

/** trapezoidal integral in sample-units times samples; a caller multiplies by the sample interval */
template<typename T>
[[nodiscard]] Measured area(std::span<const T> samples) {
    return detail::usable(samples, 2UZ).transform([&] {
        const double ends = 0.5 * (static_cast<double>(samples.front()) + static_cast<double>(samples.back()));
        return std::ranges::fold_left(samples | std::views::transform([](T x) { return static_cast<double>(x); }), 0., std::plus{}) - ends;
    });
}

} // namespace gr::algorithm::measurement

#endif // GNURADIO_ALGORITHM_MEASUREMENT_HPP
