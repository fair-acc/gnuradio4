#ifndef GNURADIO_ANALOG_HPP
#define GNURADIO_ANALOG_HPP
/*  ----------------------------------------------------------------------
 *  GNU Radio 4 – “analog” block-family (very small starter subset)
 *
 *  Blocks:
 *    1) SigSource<T>         – sine / cosine / triangle / square
 *    2) NoiseSource<T>       – Gaussian or uniform white noise
 *    3) AGC<T>               – stub, gain = 1
 *    4) FreqMod<T>, FreqDemod<T>  (float ↔ complex<float>) – stubs
 *    5) PLLCarrierTracking   – stub pass-through
 *  -------------------------------------------------------------------- */
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <algorithm>    // std::copy
#include <cmath>        // std::sin, std::cos, …
#include <complex>
#include <numbers>
#include <random>
#include <type_traits>

namespace gr::blocks::analog {

/* ======================================================================= */
/* 1. Signal generator – SigSource                                          */
/* ======================================================================= */
enum class Waveform { SIN, COS, TRI, SQR };

template<typename T>
requires(std::is_same_v<T, float> || std::is_same_v<T, double> ||
         std::is_same_v<T, std::complex<float>> ||
         std::is_same_v<T, std::complex<double>>)
struct SigSource : Block<SigSource<T>>
{
    using Description = Doc<"Classic signal source (sin, cos, tri, sqr)">;

    /* ● ports ---------------------------------------------------------- */
    PortOut<T> out;

    /* ● run-time settings --------------------------------------------- */
    Annotated<Waveform, "wf",  Visible>            wf   = Waveform::SIN;
    Annotated<double,   "fs",  Visible>            fs   = 32'000.0;   // S/s
    Annotated<double,   "f0",  Visible>            f0   = 1'000.0;    // Hz
    Annotated<double,   "amp", Visible>            amp  = 1.0;
    Annotated<double,   "offs",Visible>            offs = 0.0;
    Annotated<double,   "phi0",Visible>            phi0 = 0.0;        // rad
    Annotated<gr::Size_t,
              "n_samples_max",
              Visible,
              Doc<"stop after this many samples (0 = unlimited)">>
                                                n_samples_max = 0UZ;

    GR_MAKE_REFLECTABLE(SigSource,
                        out, wf, fs, f0, amp, offs, phi0, n_samples_max);

    /* ● internal state ------------------------------------------------- */
    double      _phase     = 0.0;
    gr::Size_t  _generated = 0;

    void start()
    {
        _phase     = phi0;
        _generated = 0;
    }

    template<OutputSpanLike TSpan>
    work::Status processBulk(TSpan& outs)
    {
        const double dphi = 2.0 * std::numbers::pi * f0 / fs;

        /* respect n_samples_max -------------------------------------- */
        std::size_t quota = outs.size();
        if (n_samples_max.value > 0 && _generated + quota > n_samples_max.value)
            quota = static_cast<std::size_t>(n_samples_max.value - _generated);

        /* generate samples ------------------------------------------- */
        auto it = outs.begin();
        for (std::size_t i = 0; i < quota; ++i, ++it, ++_generated) {
            double val = 0.0;
            switch (wf) {
                case Waveform::SIN: val = std::sin(_phase); break;
                case Waveform::COS: val = std::cos(_phase); break;
                case Waveform::TRI: val = 2.0 / std::numbers::pi
                                                * std::asin(std::sin(_phase)); break;
                case Waveform::SQR: val = (_phase < std::numbers::pi) ? 1.0 : -1.0; break;
            }
            if constexpr (std::is_floating_point_v<T>)
                *it = static_cast<T>(offs + amp * val);
            else
                *it = static_cast<T>(offs + amp * val);   // complex ctor

            _phase = std::fmod(_phase + dphi, 2.0 * std::numbers::pi);
        }
        outs.publish(quota);

        if (n_samples_max.value > 0 && _generated >= n_samples_max.value)
            this->requestStop();

        return work::Status::OK;
    }
};

/* ======================================================================= */
/* 2. NoiseSource                                                           */
/* ======================================================================= */
enum class NoiseType { GAUSSIAN, UNIFORM };

template<typename T>
requires(std::is_same_v<T, float> || std::is_same_v<T, double> ||
         std::is_same_v<T, std::complex<float>> ||
         std::is_same_v<T, std::complex<double>>)
struct NoiseSource : Block<NoiseSource<T>>
{
    using Description = Doc<"White noise source (Gaussian / Uniform)">;

    PortOut<T> out;

    Annotated<NoiseType, "kind", Visible>          kind = NoiseType::GAUSSIAN;
    Annotated<double,    "amp",  Visible>          amp  = 1.0;
    Annotated<uint64_t,  "seed", Visible>          seed = 0;
    Annotated<gr::Size_t,
              "n_samples_max",
              Visible,
              Doc<"stop after this many samples (0 = unlimited)">>
                                               n_samples_max = 0UZ;

    GR_MAKE_REFLECTABLE(NoiseSource,
                        out, kind, amp, seed, n_samples_max);

    std::mt19937_64 _rng;
    gr::Size_t      _generated = 0;

    void start()
    {
        _generated = 0;
        _rng.seed(seed.value ? static_cast<std::mt19937_64::result_type>(seed.value)
                             : std::random_device{}());
    }

    template<OutputSpanLike TSpan>
    work::Status processBulk(TSpan& outs)
    {
        std::size_t quota = outs.size();
        if (n_samples_max.value > 0 && _generated + quota > n_samples_max.value)
            quota = static_cast<std::size_t>(n_samples_max.value - _generated);

        switch (kind) {
            case NoiseType::GAUSSIAN: {
                std::normal_distribution<double> N(0.0, amp);
                for (std::size_t i = 0; i < quota; ++i, ++_generated)
                    if constexpr (std::is_floating_point_v<T>)
                        outs[i] = static_cast<T>(N(_rng));
                    else
                        outs[i] = {static_cast<typename T::value_type>(N(_rng)),
                                   static_cast<typename T::value_type>(N(_rng))};
                break;
            }
            case NoiseType::UNIFORM: {
                std::uniform_real_distribution<double> U(-amp, amp);
                for (std::size_t i = 0; i < quota; ++i, ++_generated)
                    if constexpr (std::is_floating_point_v<T>)
                        outs[i] = static_cast<T>(U(_rng));
                    else
                        outs[i] = {static_cast<typename T::value_type>(U(_rng)),
                                   static_cast<typename T::value_type>(U(_rng))};
                break;
            }
        }
        outs.publish(quota);

        if (n_samples_max.value > 0 && _generated >= n_samples_max.value)
            this->requestStop();

        return work::Status::OK;
    }
};

/* ======================================================================= */
/* 3. AGC stub (gain = 1)                                                   */
/* ======================================================================= */
template<typename T>
struct AGC : Block<AGC<T>>
{
    using Description = Doc<"Automatic gain control – placeholder (gain = 1)">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(AGC, in, out);

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(const TIn& ins, TOut& outs)
    {
        std::copy(ins.begin(), ins.end(), outs.begin());
        return work::Status::OK;
    }
};

/* ======================================================================= */
/* 4. FreqMod / FreqDemod – minimal stubs                                   */
/* ======================================================================= */
template<typename T> struct FreqMod;   // undefined for non-float

template<> struct FreqMod<float> : Block<FreqMod<float>>
{
    PortIn<float>                in;
    PortOut<std::complex<float>> out;
    GR_MAKE_REFLECTABLE(FreqMod, in, out);

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(const TIn&, TOut& outs)
    {
        std::fill(outs.begin(), outs.end(), std::complex<float>(0.0f, 0.0f));
        return work::Status::OK;
    }
};

template<typename T> struct FreqDemod; // undefined for non-complex

template<> struct FreqDemod<std::complex<float>>
       : Block<FreqDemod<std::complex<float>>>
{
    PortIn<std::complex<float>> in;
    PortOut<float>              out;
    GR_MAKE_REFLECTABLE(FreqDemod, in, out);

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(const TIn&, TOut& outs)
    {
        std::fill(outs.begin(), outs.end(), 0.0f);
        return work::Status::OK;
    }
};

/* ======================================================================= */
/* 5. PLLCarrierTracking – placeholder                                      */
/* ======================================================================= */
template<typename T>
requires std::is_same_v<T, std::complex<float>>
struct PLLCarrierTracking : Block<PLLCarrierTracking<T>>
{
    PortIn<T>  in;
    PortOut<T> out;
    GR_MAKE_REFLECTABLE(PLLCarrierTracking, in, out);

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(const TIn& ins, TOut& outs)
    {
        std::copy(ins.begin(), ins.end(), outs.begin());
        return work::Status::OK;
    }
};

/* ======================================================================= */
/*  Registration macros                                                     */
/* ======================================================================= */
GR_REGISTER_BLOCK("gr::blocks::analog::SigSource",
                  gr::blocks::analog::SigSource,
                  ([T]),
                  [ float, std::complex<float>, double, std::complex<double> ])

GR_REGISTER_BLOCK("gr::blocks::analog::NoiseSource",
                  gr::blocks::analog::NoiseSource,
                  ([T]),
                  [ float, std::complex<float>, double, std::complex<double> ])

GR_REGISTER_BLOCK("gr::blocks::analog::AGC",
                  gr::blocks::analog::AGC,
                  ([T]),
                  [ float, std::complex<float>, double, std::complex<double> ])

GR_REGISTER_BLOCK("gr::blocks::analog::FreqMod",
                  gr::blocks::analog::FreqMod,
                  ([T]), [ float ])

GR_REGISTER_BLOCK("gr::blocks::analog::FreqDemod",
                  gr::blocks::analog::FreqDemod,
                  ([T]), [ std::complex<float> ])

GR_REGISTER_BLOCK("gr::blocks::analog::PLLCarrierTracking",
                  gr::blocks::analog::PLLCarrierTracking,
                  ([T]), [ std::complex<float> ])

}  // namespace gr::blocks::analog
  
#endif  /* GNURADIO_ANALOG_HPP */
