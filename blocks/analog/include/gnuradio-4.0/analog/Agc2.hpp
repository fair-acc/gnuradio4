#ifndef INCLUDED_ANALOG_AGC2_HPP
#define INCLUDED_ANALOG_AGC2_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <algorithm>
#include <cmath>
#include <complex>

namespace gr::blocks::analog {

template<typename T, bool IsFloat = std::is_floating_point_v<T>>
struct Agc2 : Block<Agc2<T, IsFloat>>
{
    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "attack_rate", Visible> attack_rate = 1.0e-1f;
    Annotated<float, "decay_rate",  Visible> decay_rate  = 1.0e-2f;
    Annotated<float, "reference",   Visible> ref         = 1.0f;
    Annotated<float, "gain",        Visible> gain        = 1.0f;
    Annotated<float, "max_gain",    Visible> gmax        = 0.0f;   // 0 ⇒ unlimited

    GR_MAKE_REFLECTABLE(Agc2, in, out,
                        attack_rate, decay_rate,
                        ref, gain, gmax);

    template<InputSpanLike InSpan, OutputSpanLike OutSpan>
    work::Status processBulk(const InSpan& xs, OutSpan& ys)
    {
        const std::size_t N = std::min(xs.size(), ys.size());
        float g       = gain;
        const float R = ref,
                    A = attack_rate,
                    D = decay_rate,
                    M = gmax;

        for (std::size_t i = 0; i < N; ++i) {
            const auto x = xs[i];
            const auto y = static_cast<T>(x * g);         // apply current gain

            const float amp  = IsFloat ? std::fabs(y) : std::abs(y);
            const float rate = (std::fabs(amp - R) > g) ? A : D;   // attack vs decay
            g -= (amp - R) * rate;

            if (g < 1.0e-5f) g = 1.0e-5f;                // avoid blow‑ups
            if (M > 0.f && g > M) g = M;

            ys[i] = y;
        }
        gain = g;
        ys.publish(N);
        return work::Status::OK;
    }
};

using Agc2CC = Agc2<std::complex<float>, false>;
using Agc2FF = Agc2<float, true>;

GR_REGISTER_BLOCK("gr::blocks::analog::Agc2CC", gr::blocks::analog::Agc2CC)
GR_REGISTER_BLOCK("gr::blocks::analog::Agc2FF", gr::blocks::analog::Agc2FF)

} // namespace gr::blocks::analog
#endif /* INCLUDED_ANALOG_AGC2_HPP */
