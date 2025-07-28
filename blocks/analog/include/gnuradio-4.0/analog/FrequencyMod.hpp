#ifndef GNURADIO_ANALOG_FREQUENCYMOD_HPP
#define GNURADIO_ANALOG_FREQUENCYMOD_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <complex>
#include <numbers>

namespace gr::blocks::analog {

template<typename T>
struct FrequencyMod;                                   

template<>
struct FrequencyMod<float> : Block<FrequencyMod<float>>
{
    using Description = Doc<"Frequency‑modulator (float → complex<float>)">;

    PortIn<float>                 in;
    PortOut<std::complex<float>>  out;

    Annotated<float,
              "sensitivity",
              Visible,
              Doc<"Phase increment [rad/sample] per input‑unit">>
        sensitivity = std::numbers::pi_v<float> / 4.0f;   // π/4 rad/sample

    GR_MAKE_REFLECTABLE(FrequencyMod, in, out, sensitivity);

    float _phase = 0.0f;
    void  start() { _phase = 0.0f; }

    inline std::complex<float> to_polar(float ph) const
    {
        return { std::cos(ph), std::sin(ph) };
    }

    work::Status processOne(float x, std::complex<float>& y)
    {
        _phase += x * sensitivity;
        /* keep phase in [-π, π] to avoid float overflow */
        constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
        if (_phase >  std::numbers::pi_v<float>)  _phase -= two_pi;
        if (_phase < -std::numbers::pi_v<float>)  _phase += two_pi;

        y = to_polar(_phase);
        return work::Status::OK;
    }

    template<InputSpanLike  InSpan,
             OutputSpanLike OutSpan>
    work::Status processBulk(const InSpan& xs, OutSpan& ys)
    {
        if (xs.empty())
            return work::Status::DONE;

        const std::size_t n = std::min(xs.size(), ys.size());
        for (std::size_t i = 0; i < n; ++i)
            processOne(xs[i], ys[i]);

        ys.publish(n);
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::analog::FrequencyMod",gr::blocks::analog::FrequencyMod,([T]),[ float ])

} // namespace gr::blocks::analog
#endif /* GNURADIO_ANALOG_FREQUENCYMOD_HPP */
