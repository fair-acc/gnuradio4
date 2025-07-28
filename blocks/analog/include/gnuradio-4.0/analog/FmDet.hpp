#ifndef GNURADIO_ANALOG_FMDET_HPP
#define GNURADIO_ANALOG_FMDET_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <complex>
#include <numbers>

namespace gr::blocks::analog {

template<typename T> struct FmDet;            

template<>
struct FmDet<std::complex<float>> : Block<FmDet<std::complex<float>>>
{
    using Description = Doc<"IQ slope detector (complex → float)">;

    PortIn<std::complex<float>> in;
    PortOut<float>              out;

    Annotated<float,"samplerate",Visible> samplerate = 1.0f;    
    Annotated<float,"freq_low",  Visible> f_low      = -1.0f;
    Annotated<float,"freq_high", Visible> f_high     =  1.0f;
    Annotated<float,"scale",     Visible> scl        =  1.0f;   
    GR_MAKE_REFLECTABLE(FmDet,in,out,samplerate,f_low,f_high,scl);

    std::complex<float> _prev  = {1.0f, 0.0f};   // initial phase = 0
    float               _bias  = 0.0f;

    void recompute_bias()
    {
        const float hi = f_high, lo = f_low;
        _bias = (hi != lo) ? 0.5f * scl * (hi + lo) / (hi - lo) : 0.0f;
    }

    void  set_scale(float s)                { scl = s;  recompute_bias(); }
    float scale() const                     { return scl; }

    void  set_freq_range(float lo,float hi) { f_low = lo; f_high = hi; recompute_bias(); }
    float freq_low()  const                 { return f_low;  }
    float freq_high() const                 { return f_high; }
    float freq()      const                 { return 0.0f;   }   // legacy stub
    float bias()      const                 { return _bias;  }

    void start() { _prev = {1.0f,0.0f}; recompute_bias(); }

    work::Status processOne(const std::complex<float>& x, float& y)
    {
        const std::complex<float> prod = x * std::conj(_prev);
        _prev = x;
        y = scl * std::arg(prod) - _bias;
        return work::Status::OK;
    }

    template<InputSpanLike InSpan, OutputSpanLike OutSpan>
    work::Status processBulk(const InSpan& xs, OutSpan& ys)
    {
        const std::size_t n = std::min(xs.size(), ys.size());
        for(std::size_t i = 0; i < n; ++i) processOne(xs[i], ys[i]);
        ys.publish(n);
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::analog::FmDet",gr::blocks::analog::FmDet,([T]),[ std::complex<float> ])

} // namespace gr::blocks::analog
#endif /* GNURADIO_ANALOG_FMDET_HPP */
