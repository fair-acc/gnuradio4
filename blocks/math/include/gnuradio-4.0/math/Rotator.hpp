#ifndef GNURADIO_ROTATOR_HPP
#define GNURADIO_ROTATOR_HPP

#include <cmath>
#include <complex>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <numbers>

namespace gr::blocks::math {

GR_REGISTER_BLOCK(gr::blocks::math::Rotator, [T], [ std::complex<float>, std::complex<double> ])

template<pmtv::Complex T>
struct Rotator : gr::Block<Rotator<T>> {
    using value_type  = typename T::value_type;
    using Description = Doc<R""(
@brief Rotator block shifts complex input samples by a given incremental phase every sample,
       thus effectively performing a frequency translation.

This block supports either `phase_increment` in radians per sample (x) or relative `frequency_shift` in Hz for a
given 'sample_rate' in Hz (N.B sample_rate is normalised to '1' by default).
 )"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>                           sample_rate     = 1.f;
    Annotated<float, "frequency shift", Doc<"rel. frequency shift">, Unit<"Hz">>                     frequency_shift = 0.0f;
    Annotated<value_type, "phase_increment", Unit<"rad">, Doc<"how many radians to add per sample">> phase_increment{0};
    Annotated<value_type, "initial_phase", Unit<"rad">, Doc<"starting offset for each new chunk">>   initial_phase{0};

    value_type _accumulated_phase{0};

    GR_MAKE_REFLECTABLE(Rotator, in, out, sample_rate, frequency_shift, initial_phase, phase_increment);

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("frequency_shift") && !newSettings.contains("phase_increment")) {
            phase_increment = value_type(2) * static_cast<value_type>(std::numbers::pi_v<float> * frequency_shift / sample_rate);
        } else if (!newSettings.contains("frequency_shift") && newSettings.contains("phase_increment")) {
            frequency_shift = static_cast<float>(phase_increment / (value_type(2) * std::numbers::pi_v<value_type>)) * sample_rate;
        } else if (newSettings.contains("frequency_shift") && newSettings.contains("phase_increment")) {
            throw gr::exception(std::format("cannot set both 'frequency_shift' and 'phase_increment' in new setting (XOR): {}", newSettings));
        }
        _accumulated_phase = initial_phase;
    }

    [[nodiscard]] constexpr T processOne(const T& inSample) noexcept {
        _accumulated_phase += phase_increment;
        // optional: wrap angle if too large
        if (_accumulated_phase > value_type(2) * std::numbers::pi_v<value_type>) {
            _accumulated_phase -= value_type(2) * std::numbers::pi_v<value_type>;
        } else if (_accumulated_phase < value_type(0)) {
            _accumulated_phase += value_type(2) * std::numbers::pi_v<value_type>;
        }

        return inSample * std::complex<value_type>(std::cos(_accumulated_phase), std::sin(_accumulated_phase));
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_ROTATOR_HPP
