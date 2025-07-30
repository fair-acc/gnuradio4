#ifndef INCLUDED_ANALOG_AGC_HPP
#define INCLUDED_ANALOG_AGC_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <cmath>
#include <complex>

namespace gr::blocks::analog {

template<typename T, bool IsFloat = std::is_floating_point_v<T>>
struct Agc : Block<Agc<T, IsFloat>>
{
    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "rate",      Visible> rate = 1.0e-4f;
    Annotated<float, "reference", Visible> ref  = 1.0f;
    Annotated<float, "gain",      Visible> gain = 1.0f;
    Annotated<float, "max_gain",  Visible> gmax = 0.0f;   // 0 ⇒ unlimited

    GR_MAKE_REFLECTABLE(Agc, in, out, rate, ref, gain, gmax);

    template<InputSpanLike InSpan, OutputSpanLike OutSpan>
    work::Status processBulk(const InSpan& xs, OutSpan& ys)
    {
        const std::size_t n = std::min(xs.size(), ys.size());
        float g = gain;
        const float r = rate, R = ref, M = gmax;

        for (std::size_t i = 0; i < n; ++i) {
            const auto x = xs[i];
            const auto y = static_cast<T>(x * g);              // apply current gain

            const float amp = IsFloat ? std::fabs(y)
                                      : std::abs(y);          // magnitude of *output*
            g += (R - amp) * r;                                // adapt afterwards
            if (M > 0.f && g > M) g = M;

            ys[i] = y;
        }
        gain = g;
        ys.publish(n);
        return work::Status::OK;
    }
};

using AgcCC = Agc<std::complex<float>, false>;
using AgcFF = Agc<float, true>;

GR_REGISTER_BLOCK("gr::blocks::analog::AgcCC", gr::blocks::analog::AgcCC)
GR_REGISTER_BLOCK("gr::blocks::analog::AgcFF", gr::blocks::analog::AgcFF)

} // namespace gr::blocks::analog
#endif /* INCLUDED_ANALOG_AGC_HPP */
