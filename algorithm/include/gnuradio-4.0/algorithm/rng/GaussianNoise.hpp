#ifndef GNURADIO_ALGORITHM_GAUSSIAN_NOISE_HPP
#define GNURADIO_ALGORITHM_GAUSSIAN_NOISE_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <numbers>
#include <span>
#include <utility>

#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace gr::rng {

/**
 * @brief Generates Gaussian-distributed samples via the Marsaglia polar method.
 *
 * Produces pairs of independent N(0,1) variates per rejection cycle, caching the spare.
 * complexSample() returns Option B convention: nI, nQ ~ N(0, 1/2) so E[|n|^2] = 1.
 * fill()/fillComplex() use local PRNG state for bulk throughput.
 * @see Marsaglia polar method: https://en.wikipedia.org/wiki/Marsaglia_polar_method
 */
template<std::floating_point F>
struct GaussianNoise {
    Xoshiro256pp& _rng;
    F             _spare{};
    bool          _hasSpare = false;

    explicit constexpr GaussianNoise(Xoshiro256pp& rng) noexcept : _rng(rng) {}

    void reset() noexcept { _hasSpare = false; }

    constexpr F operator()() noexcept {
        if (_hasSpare) {
            _hasSpare = false;
            return _spare;
        }

        F u, v, s;
        do {
            u = _rng.uniformM11<F>();
            v = _rng.uniformM11<F>();
            s = u * u + v * v;
        } while (s >= F(1) || s == F(0));

        const F factor = std::sqrt(F(-2) * std::log(s) / s);
        _spare         = v * factor;
        _hasSpare      = true;
        return u * factor;
    }

    constexpr std::complex<F> complexSample() noexcept {
        constexpr F scale = F(1) / std::numbers::sqrt2_v<F>;
        return {(*this)() * scale, (*this)() * scale};
    }

    void fill(std::span<F> out, F amplitude = F(1), F offset = F(0)) noexcept {
        auto s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
        bool hasSpare = false;
        F    spare{};
        for (auto& sample : out) {
            if (hasSpare) {
                hasSpare = false;
                sample   = amplitude * spare + offset;
                continue;
            }
            F u, v, s;
            do {
                u = Xoshiro256pp::toUniformM11<F>(Xoshiro256pp::next(s0, s1, s2, s3));
                v = Xoshiro256pp::toUniformM11<F>(Xoshiro256pp::next(s0, s1, s2, s3));
                s = u * u + v * v;
            } while (s >= F(1) || s == F(0));
            const F factor = std::sqrt(F(-2) * std::log(s) / s);
            spare          = v * factor;
            hasSpare       = true;
            sample         = amplitude * (u * factor) + offset;
        }
        _rng._state[0] = s0;
        _rng._state[1] = s1;
        _rng._state[2] = s2;
        _rng._state[3] = s3;
        _hasSpare      = hasSpare;
        _spare         = spare;
    }

    void fillComplex(std::span<std::complex<F>> out, F amplitude = F(1), F offset = F(0)) noexcept {
        constexpr F scale = F(1) / std::numbers::sqrt2_v<F>;
        auto        s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
        for (auto& sample : out) {
            const auto [g1, g2] = polarPair(s0, s1, s2, s3);
            sample              = {amplitude * (g1 * scale) + offset, amplitude * (g2 * scale)};
        }
        _rng._state[0] = s0;
        _rng._state[1] = s1;
        _rng._state[2] = s2;
        _rng._state[3] = s3;
        _hasSpare      = false;
    }

private:
    static std::pair<F, F> polarPair(std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) noexcept {
        F u, v, s;
        do {
            u = Xoshiro256pp::toUniformM11<F>(Xoshiro256pp::next(s0, s1, s2, s3));
            v = Xoshiro256pp::toUniformM11<F>(Xoshiro256pp::next(s0, s1, s2, s3));
            s = u * u + v * v;
        } while (s >= F(1) || s == F(0));
        const F factor = std::sqrt(F(-2) * std::log(s) / s);
        return {u * factor, v * factor};
    }
};

} // namespace gr::rng

#endif // GNURADIO_ALGORITHM_GAUSSIAN_NOISE_HPP
