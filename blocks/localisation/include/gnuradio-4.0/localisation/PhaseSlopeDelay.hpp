#ifndef GNURADIO_LOCALISATION_PHASE_SLOPE_DELAY_HPP
#define GNURADIO_LOCALISATION_PHASE_SLOPE_DELAY_HPP

#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>

namespace gr::localisation {

template<std::floating_point T>
struct PhaseSlopeEstimate {
    T delaySeconds           = std::numeric_limits<T>::quiet_NaN();
    T fitErrorRad            = std::numeric_limits<T>::infinity();
    T bandwidthHz            = T{};
    T ambiguityPeriodSeconds = T{}; // 0 unless the estimate rests on one bin
};

// cross-phase follows phase(f) = -2*pi*f*delay + intercept; each usable bin's phase is unwrapped against the previous usable bin's
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN PhaseSlopeEstimate<T> estimatePhaseSlope(std::span<const std::complex<T>> spectrum, double binHz, double fMin, double fMax) noexcept {
    constexpr double kTwoPi    = 2.0 * std::numbers::pi;
    double           weightSum = 0.0, frequencySum = 0.0, phaseSum = 0.0, frequencySquaredSum = 0.0, crossSum = 0.0, phaseSquaredSum = 0.0;
    double           previousRaw = 0.0, phase = 0.0, first = 0.0, last = 0.0;
    std::size_t      used = 0UZ;
    for (std::size_t bin = 0UZ; bin <= spectrum.size() / 2UZ && bin < spectrum.size(); ++bin) {
        const double frequency = static_cast<double>(bin) * binHz;
        const double re        = static_cast<double>(spectrum[bin].real());
        const double im        = static_cast<double>(spectrum[bin].imag());
        const double weight    = re * re + im * im;
        if (frequency < fMin || frequency > fMax || !(weight > 0.0) || !std::isfinite(weight)) {
            continue;
        }
        const double raw = std::atan2(im, re);
        phase            = used == 0UZ ? raw : phase + std::remainder(raw - previousRaw, kTwoPi);
        previousRaw      = raw;
        if (used++ == 0UZ) {
            first = frequency;
        }
        last = frequency;
        weightSum += weight;
        frequencySum += weight * frequency;
        phaseSum += weight * phase;
        frequencySquaredSum += weight * frequency * frequency;
        crossSum += weight * frequency * phase;
        phaseSquaredSum += weight * phase * phase;
    }
    PhaseSlopeEstimate<T> result;
    if (used == 1UZ && first > 0.0) {
        result.delaySeconds           = static_cast<T>(-phase / (kTwoPi * first));
        result.fitErrorRad            = T{};
        result.ambiguityPeriodSeconds = static_cast<T>(1.0 / first);
    } else if (used >= 2UZ) {
        const double variance   = frequencySquaredSum - frequencySum * frequencySum / weightSum;
        const double covariance = crossSum - frequencySum * phaseSum / weightSum;
        if (variance > 0.0) {
            const double slope    = covariance / variance;
            const double residual = phaseSquaredSum - phaseSum * phaseSum / weightSum - slope * covariance;
            result.delaySeconds   = static_cast<T>(-slope / kTwoPi);
            result.fitErrorRad    = static_cast<T>(std::sqrt(std::max(0.0, residual / weightSum)));
            result.bandwidthHz    = static_cast<T>(last - first);
        }
    }
    return result;
}

GR_REGISTER_BLOCK(gr::localisation::PhaseSlopeDelay, [T], [ float, double ])

template<std::floating_point T>
struct PhaseSlopeDelay : Block<PhaseSlopeDelay<T>> {
    using Description = Doc<"Delay per pair from the weighted cross-phase slope: [pairs,3] {seconds, RMS fit error rad, bandwidth Hz}; a one-bin estimate tags its ambiguity period.">;

    static constexpr std::size_t kEstimateColumns = 3UZ;

    TensorPortIn<std::complex<T>> in;
    TensorPortOut<T>              out;

    Annotated<float, "f_min", Unit<"Hz">, Doc<"lowest frequency used">, Limits<0.f, std::numeric_limits<float>::max()>>                                                                 f_min       = 0.f;
    Annotated<float, "f_max", Unit<"Hz">, Doc<"highest frequency used, at most Nyquist">, Limits<0.f, std::numeric_limits<float>::max()>>                                               f_max       = std::numeric_limits<float>::max();
    Annotated<float, "sample_rate", Unit<"Hz">, Doc<"places the band; forwarded from upstream">, Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate = 1.f;

    GR_MAKE_REFLECTABLE(PhaseSlopeDelay, in, out, f_min, f_max, sample_rate);

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return processBulk(gr::device::hostBackend(), input, output); }

    work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count   = std::min(input.size(), output.size());
        std::size_t       emitted = 0UZ;
        for (; emitted < count; ++emitted) {
            const auto& frame = input[emitted];
            if (frame.rank() != 2UZ || frame.extent(0UZ) == 0UZ || frame.extent(1UZ) == 0UZ) {
                this->emitErrorMessage("processBulk()", "expected a nonempty [pairs,bins] cross-spectrum");
                this->requestStop();
                break;
            }
            const std::size_t pairs = frame.extent(0UZ);
            const std::size_t bins  = frame.extent(1UZ);
            auto              rows  = gr::emit(*this, output, emitted, std::array<std::size_t, 2UZ>{pairs, kEstimateColumns});
            if (!rows) {
                break;
            }
            auto                    ambiguityStorage = ctx.allocateShared<T>(pairs);
            const gr::on_scope_exit release{[&] {
                if (!ctx.peekDeviceError()) {
                    ctx.deallocate(ambiguityStorage);
                }
            }};
            T*                      ambiguityPeriods = ambiguityStorage.template devicePointer<T>();
            if (ambiguityPeriods == nullptr) {
                this->emitErrorMessage("processBulk()", "could not allocate ambiguity periods");
                this->requestStop();
                break;
            }
            const std::complex<T>* spectra   = frame.data();
            T*                     estimates = rows->data();
            const double           binHz     = static_cast<double>(sample_rate) / static_cast<double>(bins);
            const double           low       = static_cast<double>(f_min);
            const double           high      = std::min(static_cast<double>(f_max), static_cast<double>(sample_rate) / 2.0);
            gr::device::parallelFor(ctx, pairs, [=] GR_DEVICE_LAMBDA(std::size_t pair) {
                const auto estimate                      = estimatePhaseSlope<T>({spectra + pair * bins, bins}, binHz, low, high);
                estimates[kEstimateColumns * pair]       = estimate.delaySeconds;
                estimates[kEstimateColumns * pair + 1UZ] = estimate.fitErrorRad;
                estimates[kEstimateColumns * pair + 2UZ] = estimate.bandwidthHz;
                ambiguityPeriods[pair]                   = estimate.ambiguityPeriodSeconds;
            });
            if (ctx.pollDeviceError()) {
                break;
            }
            const std::span<const T> periods(ambiguityPeriods, pairs);
            if (std::ranges::any_of(periods, [](T period) { return period > T{}; })) {
                output.publishTag(property_map{{"delay_ambiguity_period", Tensor<T>(std::vector<T>(periods.begin(), periods.end()))}}, emitted);
            }
        }
        std::ignore = input.consume(emitted);
        output.publish(emitted);
        return work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        if (f_min > f_max) {
            this->emitErrorMessage("settingsChanged()", std::format("band [{}, {}] Hz is inverted: keeps the previous band", f_min.value, f_max.value));
            f_min = oldSettings.value_or<float>("f_min", 0.f);
            f_max = oldSettings.value_or<float>("f_max", std::numeric_limits<float>::max());
        }
    }
};

} // namespace gr::localisation
#endif // GNURADIO_LOCALISATION_PHASE_SLOPE_DELAY_HPP
