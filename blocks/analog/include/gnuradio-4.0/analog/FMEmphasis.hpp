#ifndef GNURADIO_ANALOG_FMEMPHASIS_HPP
#define GNURADIO_ANALOG_FMEMPHASIS_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::analog {

GR_REGISTER_BLOCK(gr::analog::FMPreemphasis, [T], [ float, double ])

template<std::floating_point T>
struct FMPreemphasis : Block<FMPreemphasis<T>> {
    using Description = Doc<R""(@brief FM pre-emphasis: boosts the high audio frequencies before FMModulator.

It is a first-order filter with unit gain at DC; above 1 / (2 pi time_constant), its gain rises by 20 dB per decade.
FMDeemphasis with the same time_constant undoes it exactly.
Broadcast FM uses 50 µs, or 75 µs in the Americas and South Korea; 0 passes the input through.

 * ITU-R, "Transmission standards for FM sound broadcasting at VHF", Recommendation ITU-R BS.450-4, 2019.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    gr::SampleRate                                                                                             sample_rate   = 240'000.f;
    Annotated<float, "time constant", Unit<"s">, Visible, Doc<"RC time constant, 0 passes the input through">> time_constant = 50e-6f;

    GR_MAKE_REFLECTABLE(FMPreemphasis, in, out, sample_rate, time_constant);

    T _differenceGain{};
    T _previousInput{};

    void start() noexcept {
        updateDifferenceGain();
        reset();
    }

    void reset() noexcept { _previousInput = T{0}; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        T previousInput = _previousInput;
        std::ranges::transform(input, output.begin(), [&previousInput, differenceGain = _differenceGain](T sample) {
            const T emphasised = sample + differenceGain * (sample - previousInput);
            previousInput      = sample;
            return emphasised;
        });
        _previousInput = previousInput;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate   = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validTimeConstant = std::isfinite(time_constant.value) && time_constant.value >= 0.f;
        if (!validSampleRate || !validTimeConstant) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz must be finite and positive and time_constant {} s finite and non-negative, keeping the previous settings", sample_rate.value, time_constant.value));
            sample_rate   = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            time_constant = oldSettings.value_or<float>("time_constant", time_constant.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateDifferenceGain();
    }

    void updateDifferenceGain() noexcept {
        const double samplesPerTimeConstant = static_cast<double>(sample_rate.value) * static_cast<double>(time_constant.value);
        _differenceGain                     = samplesPerTimeConstant > 0. ? static_cast<T>(1. / std::expm1(1. / samplesPerTimeConstant)) : T{0};
    }
};

GR_REGISTER_BLOCK(gr::analog::FMDeemphasis, [T], [ float, double ])

template<std::floating_point T>
struct FMDeemphasis : Block<FMDeemphasis<T>> {
    using Description = Doc<R""(@brief FM de-emphasis: first-order RC low-pass that undoes FMPreemphasis and reduces the high-frequency noise after FMDemodulator.

Its impulse response is that of an analogue RC low-pass, sampled and scaled to unit gain at DC.
Broadcast FM uses 50 µs, or 75 µs in the Americas and South Korea; 0 passes the input through.
For stereo, use one on each output of FMStereoDecoder.

 * ITU-R, "Transmission standards for FM sound broadcasting at VHF", Recommendation ITU-R BS.450-4, 2019.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    gr::SampleRate                                                                                             sample_rate   = 240'000.f;
    Annotated<float, "time constant", Unit<"s">, Visible, Doc<"RC time constant, 0 passes the input through">> time_constant = 50e-6f;

    GR_MAKE_REFLECTABLE(FMDeemphasis, in, out, sample_rate, time_constant);

    T _smoothingFactor{};
    T _output{};

    void start() noexcept {
        updateSmoothingFactor();
        reset();
    }

    void reset() noexcept { _output = T{0}; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        T filtered = _output;
        std::ranges::transform(input, output.begin(), [&filtered, smoothingFactor = _smoothingFactor](T sample) {
            const T next = filtered + smoothingFactor * (sample - filtered);
            if (std::isfinite(next)) [[likely]] {
                filtered = next;
            }
            return filtered;
        });
        _output = filtered;
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/, property_map& forwardSettings) {
        const bool validSampleRate   = std::isfinite(sample_rate.value) && sample_rate.value > 0.f;
        const bool validTimeConstant = std::isfinite(time_constant.value) && time_constant.value >= 0.f;
        if (!validSampleRate || !validTimeConstant) {
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} Hz must be finite and positive and time_constant {} s finite and non-negative, keeping the previous settings", sample_rate.value, time_constant.value));
            sample_rate   = oldSettings.value_or<float>("sample_rate", sample_rate.value);
            time_constant = oldSettings.value_or<float>("time_constant", time_constant.value);
            forwardSettings.erase("sample_rate");
            return;
        }
        updateSmoothingFactor();
    }

    void updateSmoothingFactor() noexcept {
        const double samplesPerTimeConstant = static_cast<double>(sample_rate.value) * static_cast<double>(time_constant.value);
        _smoothingFactor                    = samplesPerTimeConstant > 0. ? static_cast<T>(-std::expm1(-1. / samplesPerTimeConstant)) : T{1};
    }
};

} // namespace gr::analog

#endif // GNURADIO_ANALOG_FMEMPHASIS_HPP
