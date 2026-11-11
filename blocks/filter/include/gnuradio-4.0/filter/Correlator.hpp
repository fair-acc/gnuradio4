#ifndef GNURADIO_BLOCKS_CORRELATOR_HPP
#define GNURADIO_BLOCKS_CORRELATOR_HPP

#include <memory_resource>

#include <cmath>
#include <complex>
#include <format>
#include <functional>
#include <ranges>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::Correlator, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires(std::floating_point<T> || gr::meta::complex_like<T>)
struct Correlator : Block<Correlator<T>, Resampling<>, Stride<>> {
    using Description = Doc<R""(cross-correlation against a stored reference, one lag per output sample; the reference is conjugated, so a complex
signal correlated against itself peaks real and positive.

An empty `reference` with a `duration` above 0 is the linear chirp SignalGenerator emits for the same f_start, f_stop, duration, taper and
sample_rate (amplitude 1, phase 0), so emitter and matched filter cannot drift apart; a non-empty `reference` set by the user wins.
`coefficient` divides each lag by |reference| |window|, which puts its magnitude in [0, 1] whatever the signal level; a silent window gives 0.

Every lag is independent, so the block states its window and the framework may run the lags concurrently on whichever
domain it is placed.

 * G. L. Turin, "An introduction to matched filters", IRE Trans. Inf. Theory, vol. 6, no. 3, pp. 311-329, 1960.)"">;

    using Real                              = decltype(std::abs(T{}));
    static constexpr std::size_t kMaxLength = 1048576UZ;

    PortIn<T>  in;
    PortOut<T> out;

    std::pmr::vector<T>                                                                                                      reference{};
    Annotated<gr::Size_t, "lags", Limits<1UZ, kMaxLength>>                                                                   lags        = 64U;
    Annotated<double, "f_start", Unit<"Hz">, Doc<"chirp reference start frequency">>                                         f_start     = gr::signal::ChirpParameters{}.startHz;
    Annotated<double, "f_stop", Unit<"Hz">, Doc<"chirp reference stop frequency">>                                           f_stop      = gr::signal::ChirpParameters{}.stopHz;
    Annotated<double, "duration", Unit<"s">, Doc<"chirp reference duration; 0: no chirp">>                                   duration    = 0.0;
    Annotated<double, "taper", Doc<"Tukey fraction of the chirp reference, 0 to 1">, Limits<0.0, 1.0>>                       taper       = gr::signal::ChirpParameters{}.tukeyFraction;
    gr::SampleRate                                                                                                           sample_rate = 1000.f;
    Annotated<bool, "coefficient", Doc<"true: each lag the correlation coefficient, |c| <= 1; false: the raw sum">, Visible> coefficient = false;

    std::pmr::vector<T> _chirpReference{}; /// used while `reference` is empty; reflected, so re-seated with the block

    GR_MAKE_REFLECTABLE(Correlator, in, out, reference, lags, f_start, f_stop, duration, taper, sample_rate, coefficient, _chirpReference);

    /** |reference in use|, for the coefficient */
    Real _referenceNorm = Real{0};

    /// R_xy(l) = Σ conj(y[k]) x[l+k) -- the conjugate is on the reference, which is the convention that makes a
    /// signal's autocorrelation peak on the real axis instead of rotating with the signal's own phase
    [[nodiscard]] static constexpr T conjugated(T value) noexcept {
        if constexpr (gr::meta::complex_like<T>) {
            return std::conj(value);
        } else {
            return value;
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const double nChirp = duration.value * static_cast<double>(sample_rate.value);
        if (duration.value <= 0.0) {
            _chirpReference.clear();
        } else if (nChirp < 1.0 || nChirp > static_cast<double>(kMaxLength)) [[unlikely]] {
            const std::string message = std::format("a chirp of {} s at {} Hz is {} samples, outside 1..{}: no chirp reference", duration.value, sample_rate.value, nChirp, kMaxLength);
            this->emitErrorMessage("settingsChanged()", message);
            _chirpReference.clear();
        } else {
            gr::signal::SignalGeneratorCore<T> chirp;
            chirp.configure(gr::signal::SignalType::LinearChirp, 0.0, sample_rate.value, 0.f, 1.f, 0.f, 0U);
            chirp.chirp = {.startHz = f_start.value, .stopHz = f_stop.value, .durationSeconds = duration.value, .repeat = false, .periodSeconds = 0.0, .tukeyFraction = taper.value};
            _chirpReference.resize(static_cast<std::size_t>(std::llround(nChirp)));
            chirp.fill(std::span<T>(_chirpReference));
        }
        const double energy = std::ranges::fold_left(referenceInUse() | std::views::transform([](const T& value) { return static_cast<double>(squaredMagnitude(value)); }), 0.0, std::plus{});
        _referenceNorm      = static_cast<Real>(std::sqrt(energy));

        const std::size_t length = std::max(std::size_t{1}, referenceInUse().size());
        this->output_chunk_size  = lags;
        this->input_chunk_size   = static_cast<gr::Size_t>(static_cast<std::size_t>(lags) + length - 1UZ);
        this->stride             = lags;
    }

    [[nodiscard]] static constexpr Real squaredMagnitude(T value) noexcept {
        if constexpr (gr::meta::complex_like<T>) {
            return value.real() * value.real() + value.imag() * value.imag();
        } else {
            return value * value;
        }
    }

    [[nodiscard]] const std::pmr::vector<T>& referenceInUse() const noexcept { return reference.empty() ? _chirpReference : reference; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) const noexcept {
        const std::pmr::vector<T>& used       = referenceInUse();
        const std::size_t          length     = used.size();
        const bool                 normalised = coefficient.value;
        for (std::size_t lag = 0UZ; lag < output.size(); ++lag) {
            Real windowEnergy{0};
            for (std::size_t k = 0UZ; normalised && k < length; ++k) {
                windowEnergy += squaredMagnitude(input[lag + k]);
            }
            const Real scale  = _referenceNorm * std::sqrt(windowEnergy);
            const Real factor = !normalised ? Real{1} : (scale > Real{0} ? Real{1} / scale : Real{0});
            if constexpr (gr::meta::complex_like<T>) {
                // accumulated in gr::complex rather than std::complex: the latter's operator* lowers to the
                // libgcc helper `__mulsc3`, which a device JIT cannot resolve, and the kernel fails to link
                gr::complex<Real> sum{};
                for (std::size_t k = 0UZ; k < length; ++k) {
                    const gr::complex<Real> conjugatedReference{used[k].real(), -used[k].imag()};
                    sum += conjugatedReference * gr::complex<Real>{input[lag + k].real(), input[lag + k].imag()};
                }
                output[lag] = T{sum.real() * factor, sum.imag() * factor};
            } else {
                T sum{};
                for (std::size_t k = 0UZ; k < length; ++k) {
                    sum += conjugated(used[k]) * input[lag + k];
                }
                output[lag] = sum * factor;
            }
        }
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_CORRELATOR_HPP
