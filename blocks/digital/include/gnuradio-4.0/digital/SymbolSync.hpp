#ifndef GNURADIO_DIGITAL_SYMBOLSYNC_HPP
#define GNURADIO_DIGITAL_SYMBOLSYNC_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/HermiteResampler.hpp>
#include <gnuradio-4.0/algorithm/filter/RaisedCosine.hpp>
#include <gnuradio-4.0/algorithm/signal/PhaseLockedLoopCore.hpp>

namespace gr::digital {

GR_REGISTER_BLOCK(gr::digital::SymbolSync, [T], [ float, double ])

template<std::floating_point T>
struct SymbolSync : Block<SymbolSync<T>, NoTagPropagation> {
    using Description = Doc<R""(@brief Symbol synchroniser: picks one complex sample per symbol from the matched-filter output, at the symbol instants.

A Gardner timing error detector steers a second-order loop that sets the sampling instants and the symbol period, so it follows the transmitter's symbol clock within rate_tolerance.
The nominal samples_per_symbol need not be an integer: values between input samples come from cubic interpolation, which is accurate from about 4 samples per symbol.
It works before carrier recovery, and the input amplitude does not matter.
Each input tag moves to the first output symbol at or after it, with sample_rate divided by samples_per_symbol.

 * F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled receivers", IEEE Trans. Commun., vol. 34, no. 5, pp. 423-429, 1986.
 * F. M. Gardner, "Interpolation in digital modems - Part I: Fundamentals", IEEE Trans. Commun., vol. 41, no. 3, pp. 501-507, 1993.
 * M. Rice, "Digital Communications: A Discrete-Time Approach". Upper Saddle River, NJ: Pearson Prentice Hall, 2009, ch. 8 and app. C.)"">;

    using Interpolator = gr::algorithm::filter::HermiteResampler<std::complex<T>>;

    static constexpr float  kMaxSamplesPerSymbol  = 1024.f;
    static constexpr float  kMaxLoopBandwidth     = 0.1f;
    static constexpr float  kMaxRateTolerance     = 0.1f;
    static constexpr double kMaxTimingKick        = 0.5;
    static constexpr double kMaxTimingError       = 2.;
    static constexpr double kSymbolPowerAveraging = 0.01;

    PortIn<std::complex<T>>  in;
    PortOut<std::complex<T>> out;

    Annotated<float, "samples per symbol", Visible, Doc<"nominal input samples per symbol">>                        samples_per_symbol = 4.f;
    Annotated<float, "excess bandwidth", Visible, Doc<"roll-off of the raised-cosine pulse at the input">>          excess_bandwidth   = 0.35f;
    Annotated<float, "loop bandwidth", Doc<"noise bandwidth of the timing loop as a fraction of the symbol rate">>  loop_bandwidth     = 0.002f;
    Annotated<float, "rate tolerance", Doc<"largest relative deviation of the symbol rate from its nominal value">> rate_tolerance     = 0.005f;

    GR_MAKE_REFLECTABLE(SymbolSync, in, out, samples_per_symbol, excess_bandwidth, loop_bandwidth, rate_tolerance);

    double                         _proportionalGain{};
    double                         _integralGain{};
    double                         _minPeriod{};
    double                         _maxPeriod{};
    double                         _period{};
    double                         _nextSymbolTime{};
    double                         _meanSymbolPower{};
    double                         _forwardedTagsBefore{};
    std::optional<std::complex<T>> _previousSymbol;
    std::vector<double>            _symbolTimes;

    void start() {
        updateFromSettings();
        reset();
    }

    void reset() {
        _period              = static_cast<double>(samples_per_symbol.value);
        _nextSymbolTime      = _period;
        _meanSymbolPower     = 0.;
        _forwardedTagsBefore = 0.;
        _previousSymbol.reset();
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::span<const std::complex<T>> window{input.data(), input.size()};
        const std::span<std::complex<T>>       symbols{output.data(), output.size()};
        const double                           nextTagTime   = timeOfNextTag(input);
        const auto                             beforeNextTag = [this, nextTagTime] { return _symbolTimes.empty() || _symbolTimes.back() < nextTagTime; };
        _symbolTimes.clear();
        while (_symbolTimes.size() < symbols.size() && static_cast<std::size_t>(_nextSymbolTime) + 2UZ < window.size() && beforeNextTag()) {
            const double          symbolTime  = _nextSymbolTime;
            const std::complex<T> symbol      = interpolateAt(window, symbolTime);
            const std::complex<T> midpoint    = interpolateAt(window, symbolTime - _period / 2.);
            const double          timingError = timingErrorAt(midpoint, symbol);
            _period                           = std::clamp(_period + _integralGain * timingError, _minPeriod, _maxPeriod);
            _nextSymbolTime                   = symbolTime + _period + std::clamp(_proportionalGain * timingError, -kMaxTimingKick * _period, kMaxTimingKick * _period);
            _previousSymbol                   = symbol;
            symbols[_symbolTimes.size()]      = symbol;
            _symbolTimes.push_back(symbolTime);
        }
        if (_symbolTimes.empty()) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return symbols.empty() ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const double      firstKeptSample = std::min(_symbolTimes.back(), _nextSymbolTime - _period / 2. - 1.);
        const std::size_t consumed        = firstKeptSample > 0. ? static_cast<std::size_t>(firstKeptSample) : 0UZ;
        forwardTagsOntoSymbols(input, output);
        input.consumeTags(1UZ);
        _forwardedTagsBefore = std::floor(_symbolTimes.back()) + 1. - static_cast<double>(consumed);
        _nextSymbolTime -= static_cast<double>(consumed);
        std::ignore = input.consume(consumed);
        output.publish(_symbolTimes.size());
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validSamplesPerSymbol = std::isfinite(samples_per_symbol.value) && samples_per_symbol.value >= 2.f && samples_per_symbol.value <= kMaxSamplesPerSymbol;
        const bool validExcessBandwidth  = std::isfinite(excess_bandwidth.value) && excess_bandwidth.value > 0.f && excess_bandwidth.value <= 1.f;
        const bool validLoopBandwidth    = std::isfinite(loop_bandwidth.value) && loop_bandwidth.value > 0.f && loop_bandwidth.value <= kMaxLoopBandwidth;
        const bool validRateTolerance    = std::isfinite(rate_tolerance.value) && rate_tolerance.value >= 0.f && rate_tolerance.value <= kMaxRateTolerance;
        if (!validSamplesPerSymbol || !validExcessBandwidth || !validLoopBandwidth || !validRateTolerance) {
            this->emitErrorMessage("settingsChanged()", std::format("samples_per_symbol {} must be within [2, {}], excess_bandwidth {} within (0, 1], loop_bandwidth {} within (0, {}] and rate_tolerance {} within [0, {}], keeping the previous settings", samples_per_symbol.value, kMaxSamplesPerSymbol, excess_bandwidth.value, loop_bandwidth.value, kMaxLoopBandwidth, rate_tolerance.value, kMaxRateTolerance));
            samples_per_symbol = oldSettings.value_or<float>("samples_per_symbol", samples_per_symbol.value);
            excess_bandwidth   = oldSettings.value_or<float>("excess_bandwidth", excess_bandwidth.value);
            loop_bandwidth     = oldSettings.value_or<float>("loop_bandwidth", loop_bandwidth.value);
            rate_tolerance     = oldSettings.value_or<float>("rate_tolerance", rate_tolerance.value);
            return;
        }
        _period *= static_cast<double>(samples_per_symbol.value) / static_cast<double>(oldSettings.value_or<float>("samples_per_symbol", samples_per_symbol.value));
        updateFromSettings();
    }

    void updateFromSettings() {
        const double                                     nominalPeriod       = static_cast<double>(samples_per_symbol.value);
        const gr::signal::PhaseLockedLoopCore::LoopGains gains               = gr::signal::PhaseLockedLoopCore::loopGains(static_cast<double>(loop_bandwidth.value));
        const double                                     samplesPerUnitError = nominalPeriod / gardnerDetectorGain(static_cast<double>(excess_bandwidth.value));
        _proportionalGain                                                    = gains.proportional * samplesPerUnitError;
        _integralGain                                                        = gains.integral * samplesPerUnitError;
        _minPeriod                                                           = nominalPeriod * (1. - static_cast<double>(rate_tolerance.value));
        _maxPeriod                                                           = nominalPeriod * (1. + static_cast<double>(rate_tolerance.value));
        _period                                                              = std::clamp(_period, _minPeriod, _maxPeriod);
        _nextSymbolTime                                                      = std::clamp(_nextSymbolTime, _period / 2. + 1., (1. + kMaxTimingKick) * _period + 1.);
        in.min_samples                                                       = static_cast<gr::Size_t>(std::ceil((1. + kMaxTimingKick) * _maxPeriod)) + 4U;
    }

    [[nodiscard]] static double gardnerDetectorGain(double excessBandwidth) noexcept {
        constexpr int    kPulseSpanInSymbols = 64;
        constexpr double kTimingStep         = 1e-4;
        const auto       pulse               = [excessBandwidth](double timeInSymbols) { return gr::algorithm::filter::raisedCosine(timeInSymbols, excessBandwidth); };
        const auto       expectedError       = [&pulse](double timingOffset) {
            double sum = 0.;
            for (int symbol = -kPulseSpanInSymbols; symbol <= kPulseSpanInSymbols; ++symbol) {
                const double symbolTime = timingOffset - static_cast<double>(symbol);
                sum += pulse(symbolTime - 0.5) * (pulse(symbolTime - 1.) - pulse(symbolTime));
            }
            return sum;
        };
        return (expectedError(-kTimingStep) - expectedError(kTimingStep)) / (2. * kTimingStep);
    }

    [[nodiscard]] double timingErrorAt(std::complex<T> midpoint, std::complex<T> symbol) noexcept {
        const double symbolPower = static_cast<double>(std::norm(symbol));
        if (std::isfinite(symbolPower)) {
            _meanSymbolPower += (_meanSymbolPower > 0. ? kSymbolPowerAveraging : 1.) * (symbolPower - _meanSymbolPower);
        }
        const std::complex<T> previous = _previousSymbol.value_or(symbol);
        const double          error    = static_cast<double>(std::real(std::conj(midpoint) * (previous - symbol))) / _meanSymbolPower;
        return std::isfinite(error) ? std::clamp(error, -kMaxTimingError, kMaxTimingError) : 0.;
    }

    [[nodiscard]] static std::complex<T> interpolateAt(std::span<const std::complex<T>> window, double time) noexcept {
        const auto index = static_cast<std::size_t>(time);
        return Interpolator::sampleAt(window, index, static_cast<T>(time - static_cast<double>(index)));
    }

    [[nodiscard]] static double timeOfNextTag(InputSpanLike auto& input) {
        auto       tags    = input.tags();
        const auto nextTag = std::ranges::find_if(tags, [](const auto& tag) { return tag.first > 0; });
        return nextTag == std::ranges::end(tags) ? std::numeric_limits<double>::infinity() : static_cast<double>((*nextTag).first);
    }

    void forwardTagsOntoSymbols(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            const double tagTime                 = static_cast<double>(relativeIndex);
            const bool   forwardedInPreviousCall = tagTime < _forwardedTagsBefore;
            if (forwardedInPreviousCall || tagTime > _symbolTimes.back()) {
                continue;
            }
            property_map symbolTag;
            for (const auto& [key, value] : tagMap.get()) {
                const bool    isSampleRate = gr::tag::settingsKey(std::string_view{key}) == gr::tag::kSampleRate.shortKey();
                const float*  floatRate    = isSampleRate ? value.template get_if<float>() : nullptr;
                const double* doubleRate   = isSampleRate ? value.template get_if<double>() : nullptr;
                if (floatRate != nullptr) {
                    symbolTag.insert_or_assign(key, *floatRate / samples_per_symbol.value);
                } else if (doubleRate != nullptr) {
                    symbolTag.insert_or_assign(key, *doubleRate / static_cast<double>(samples_per_symbol.value));
                } else {
                    symbolTag.insert_or_assign(key, value);
                }
            }
            const auto firstSymbolReaching = std::ranges::lower_bound(_symbolTimes, tagTime);
            output.publishTag(symbolTag, static_cast<std::size_t>(std::distance(_symbolTimes.begin(), firstSymbolReaching)));
        }
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_SYMBOLSYNC_HPP
