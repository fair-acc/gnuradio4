#ifndef GNURADIO_ANALOG_QUADRATUREDEMOD_HPP
#define GNURADIO_ANALOG_QUADRATUREDEMOD_HPP
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <complex>
#include <numbers>

namespace gr::blocks::analog {

template<typename T> struct QuadratureDemod;               

template<>
struct QuadratureDemod<std::complex<float>>
    : Block<QuadratureDemod<std::complex<float>>>
{
    using Description =
        Doc<"Quadrature FM demodulator (complex<float> → float)">;

    PortIn<std::complex<float>> in;
    PortOut<float>              out;

    Annotated<float,"gain",Visible,
              Doc<"Output scale (rad → user units)">>
        gain = 1.0f;

    GR_MAKE_REFLECTABLE(QuadratureDemod,in,out,gain);

    std::complex<float> _prev{1.0f,0.0f};
    bool                _have_prev{false};

    void start() { _have_prev = false; _prev = {1.0f,0.0f}; }

    work::Status processOne(std::complex<float> x, float& y)
    {
        if(!_have_prev){                   // first sample ⇒ y = 0
            y = 0.0f;
            _prev = x;
            _have_prev = true;
        } else {
            y     = gain * std::arg(x * std::conj(_prev));
            _prev = x;
        }
        return work::Status::OK;
    }

    template<InputSpanLike InSpan, OutputSpanLike OutSpan>
    work::Status processBulk(const InSpan& xs, OutSpan& ys)
    {
        const std::size_t n = std::min(xs.size(), ys.size());
        for(std::size_t i=0;i<n;++i) processOne(xs[i], ys[i]);
        ys.publish(n);
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::analog::QuadratureDemod",gr::blocks::analog::QuadratureDemod,([T]),[ std::complex<float> ])

} // namespace gr::blocks::analog
#endif /* GNURADIO_ANALOG_QUADRATUREDEMOD_HPP */
