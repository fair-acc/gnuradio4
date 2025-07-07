#ifndef GNURADIO_ANALOG_HPP
#define GNURADIO_ANALOG_HPP
/* --------------------------------------------------------------------- *
 *  GNU Radio 4 – “analog” starter block-family
 * --------------------------------------------------------------------- */
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <type_traits>

namespace gr::blocks::analog {

/* helper: how many samples may we still emit? ------------------------- */
[[nodiscard]] constexpr std::size_t
quota_left(gr::Size_t generated, gr::Size_t limit, std::size_t want)
{
    if (limit == 0UZ) return want;           // unlimited
    const auto remaining = limit - generated;
    return std::min<std::size_t>(want,
                                 static_cast<std::size_t>(remaining));
}

/* ===================================================================== */
/* 1. SigSource                                                          */
/* ===================================================================== */
enum class Waveform { SIN, COS, TRI, SQR };

template<typename T>
requires(std::is_same_v<T,float> || std::is_same_v<T,double> ||
         std::is_same_v<T,std::complex<float>> ||
         std::is_same_v<T,std::complex<double>>)
struct SigSource : Block<SigSource<T>>
{
    using Description = Doc<"Signal source (sin, cos, tri, sqr)">;

    PortOut<T> out;

    Annotated<Waveform,"wf", Visible> wf   = Waveform::SIN;
    Annotated<double,  "fs", Visible> fs   = 32'000.0;   // S/s
    Annotated<double,  "f0", Visible> f0   = 1'000.0;    // Hz
    Annotated<double,  "amp",Visible> amp  = 1.0;
    Annotated<double,  "offs",Visible> offs = 0.0;
    Annotated<double,  "phi0",Visible> phi0 = 0.0;
    Annotated<gr::Size_t,"n_samples_max",Visible,
              Doc<"stop after this many samples (0 = unlimited)">>
                                            n_samples_max = 0UZ;

    GR_MAKE_REFLECTABLE(SigSource,
                        out,wf,fs,f0,amp,offs,phi0,n_samples_max);

    /* state ----------------------------------------------------------- */
    double  _phase     = 0.0;
    Size_t  _generated = 0;

    void start()
    {
        _phase     = phi0;
        _generated = 0;
    }

    /* single-sample helper (nice for tests, reused in bulk) ----------- */
    work::Status processOne(T& y)
    {
        using namespace std::numbers;
        const double dphi = 2.0 * pi * f0 / fs;

        double v = 0.0;
        switch (wf) {
            case Waveform::SIN: v = std::sin(_phase); break;
            case Waveform::COS: v = std::cos(_phase); break;
            case Waveform::TRI: v = 2.0/pi * std::asin(std::sin(_phase)); break;
            case Waveform::SQR: v = (_phase < pi) ? 1.0 : -1.0; break;
        }

        if constexpr (std::is_floating_point_v<T>)
            y = static_cast<T>(offs + amp*v);
        else
            y = static_cast<T>(offs + amp*v);          // complex ctor

        _phase = std::fmod(_phase + dphi, 2.0*pi);

        if (n_samples_max.value>0 && ++_generated >= n_samples_max.value)
            this->requestStop();
        return work::Status::OK;
    }

    template<OutputSpanLike Span>
    work::Status processBulk(Span& outs)
    {
        const std::size_t want = quota_left(_generated,
                                            n_samples_max.value,
                                            outs.size());

        for (std::size_t i = 0; i < want; ++i)
            processOne(outs[i]);

        outs.publish(want);
        return work::Status::OK;
    }
};

/* ===================================================================== */
/* 2. NoiseSource                                                        */
/* ===================================================================== */
enum class NoiseType { GAUSSIAN, UNIFORM };

template<typename T>
requires(std::is_same_v<T,float> || std::is_same_v<T,double> ||
         std::is_same_v<T,std::complex<float>> ||
         std::is_same_v<T,std::complex<double>>)
struct NoiseSource : Block<NoiseSource<T>>
{
    using Description = Doc<"White-noise source (Gaussian / Uniform)">;

    PortOut<T> out;

    Annotated<NoiseType,"kind", Visible> kind = NoiseType::GAUSSIAN;
    Annotated<double,   "amp",  Visible> amp  = 1.0;
    Annotated<uint64_t, "seed", Visible> seed = 0;
    Annotated<gr::Size_t,"n_samples_max", Visible,
              Doc<"stop after this many samples (0 = unlimited)">>
                                            n_samples_max = 0UZ;

    GR_MAKE_REFLECTABLE(NoiseSource,out,kind,amp,seed,n_samples_max);

    std::mt19937_64 _rng;
    Size_t          _generated = 0;

    void start()
    {
        _generated = 0;
        _rng.seed(seed.value ? seed.value : std::random_device{}());
    }

    work::Status processOne(T& y)
    {
        auto make = [this](auto&& dist){
            if constexpr (std::is_floating_point_v<T>)
                return static_cast<T>(dist(_rng));
            else
                return T(static_cast<typename T::value_type>(dist(_rng)),
                         static_cast<typename T::value_type>(dist(_rng)));
        };

        switch (kind) {
            case NoiseType::GAUSSIAN:
                y = make(std::normal_distribution<double>{0.0, amp});
                break;
            case NoiseType::UNIFORM:
                y = make(std::uniform_real_distribution<double>{-amp, amp});
                break;
        }

        if (n_samples_max.value>0 && ++_generated >= n_samples_max.value)
            this->requestStop();
        return work::Status::OK;
    }

    template<OutputSpanLike Span>
    work::Status processBulk(Span& outs)
    {
        const std::size_t want = quota_left(_generated,
                                            n_samples_max.value,
                                            outs.size());

        for (std::size_t i = 0; i < want; ++i)
            processOne(outs[i]);

        outs.publish(want);
        return work::Status::OK;
    }
};

/* ===================================================================== */
/* 3. AGC stub (gain = 1)                                                */
/* ===================================================================== */
template<typename T>
struct AGC : Block<AGC<T>>
{
    using Description = Doc<"Automatic gain control – placeholder">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(AGC,in,out);

    template<InputSpanLike In, OutputSpanLike Out>
    work::Status processBulk(const In& ins, Out& outs)
    {
        std::copy(ins.begin(), ins.end(), outs.begin());
        outs.publish(ins.size());
        return work::Status::OK;
    }
};

/* ===================================================================== */
/* 4. FreqMod / FreqDemod stubs                                          */
/* ===================================================================== */
template<typename T> struct FreqMod;            // undefined for others

template<>
struct FreqMod<float> : Block<FreqMod<float>>
{
    PortIn<float>                in;
    PortOut<std::complex<float>> out;
    GR_MAKE_REFLECTABLE(FreqMod,in,out);

    template<InputSpanLike In, OutputSpanLike Out>
    work::Status processBulk(const In&, Out& outs)
    {
        std::fill(outs.begin(), outs.end(),
                  std::complex<float>{0.0f,0.0f});
        outs.publish(outs.size());
        return work::Status::OK;
    }
};

template<typename T> struct FreqDemod;          // undefined for others

template<>
struct FreqDemod<std::complex<float>>
      : Block<FreqDemod<std::complex<float>>>
{
    PortIn<std::complex<float>> in;
    PortOut<float>              out;
    GR_MAKE_REFLECTABLE(FreqDemod,in,out);

    template<InputSpanLike In, OutputSpanLike Out>
    work::Status processBulk(const In&, Out& outs)
    {
        std::fill(outs.begin(), outs.end(), 0.0f);
        outs.publish(outs.size());
        return work::Status::OK;
    }
};

/* ===================================================================== */
/* 5. PLLCarrierTracking stub                                            */
/* ===================================================================== */
template<typename T>
requires std::is_same_v<T,std::complex<float>>
struct PLLCarrierTracking : Block<PLLCarrierTracking<T>>
{
    PortIn<T>  in;
    PortOut<T> out;
    GR_MAKE_REFLECTABLE(PLLCarrierTracking,in,out);

    template<InputSpanLike In, OutputSpanLike Out>
    work::Status processBulk(const In& ins, Out& outs)
    {
        std::copy(ins.begin(), ins.end(), outs.begin());
        outs.publish(ins.size());
        return work::Status::OK;
    }
};

/* ======================================================================= */
/*  Registration macros (single-line form)                                 */
/* ======================================================================= */
GR_REGISTER_BLOCK("gr::blocks::analog::SigSource",gr::blocks::analog::SigSource,([T]),[ float,std::complex<float>,double,std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::analog::NoiseSource",gr::blocks::analog::NoiseSource,([T]),[ float,std::complex<float>,double,std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::analog::AGC",gr::blocks::analog::AGC,([T]),[ float,std::complex<float>,double,std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::analog::FreqMod",gr::blocks::analog::FreqMod,([T]),[ float ])
GR_REGISTER_BLOCK("gr::blocks::analog::FreqDemod",gr::blocks::analog::FreqDemod,([T]),[ std::complex<float> ])
GR_REGISTER_BLOCK("gr::blocks::analog::PLLCarrierTracking",gr::blocks::analog::PLLCarrierTracking,([T]),[ std::complex<float> ])


} // namespace gr::blocks::analog

#endif /* GNURADIO_ANALOG_HPP */
