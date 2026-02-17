#ifndef GNURADIO_ALGORITHM_XOSHIRO256PP_HPP
#define GNURADIO_ALGORITHM_XOSHIRO256PP_HPP

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>

namespace gr::rng {

/**
 * @brief Fast non-cryptographic PRNG for DSP/simulation with uniform and triangular (cheap semi-Gaussian) noise helpers.
 *
 * Engine: xoshiro256++ (256-bit state, high throughput). Seeded via SplitMix64 to avoid the all-zero fixed point.
 * triangularM11 computes (u1 + u2 - 1) — a symmetric Irwin-Hall(n=2) distribution on [-1, +1), mean = 0.
 * @see D. Blackman, S. Vigna, "Scrambled Linear Pseudorandom Number Generators", arXiv:1805.01407
 * @see reference implementation: https://prng.di.unimi.it/
 * @see Irwin-Hall distribution: https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution
 */
struct Xoshiro256pp {
    using result_type = std::uint64_t;

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

    std::uint64_t _state[4]{};

    constexpr Xoshiro256pp() noexcept { seed(0); }
    explicit constexpr Xoshiro256pp(std::uint64_t seedValue) noexcept { seed(seedValue); }

    constexpr void seed(std::uint64_t seedValue) noexcept {
        std::uint64_t sm = seedValue;
        _state[0]        = splitMix64(sm);
        _state[1]        = splitMix64(sm);
        _state[2]        = splitMix64(sm);
        _state[3]        = splitMix64(sm);
    }

    static constexpr result_type next(std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) noexcept {
        const result_type   result = std::rotl(s0 + s3, 23) + s0;
        const std::uint64_t t      = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = std::rotl(s3, 45);
        return result;
    }

    constexpr result_type operator()() noexcept { return next(_state[0], _state[1], _state[2], _state[3]); }

    template<std::floating_point F>
    static constexpr F toUniform01(result_type raw) noexcept {
        if constexpr (std::same_as<F, float>) {
            return static_cast<F>(raw >> 40) * F(0x1.0p-24);
        } else {
            return static_cast<F>(raw >> 11) * F(0x1.0p-53);
        }
    }

    template<std::floating_point F>
    static constexpr F toUniformM11(result_type raw) noexcept {
        return F(2) * toUniform01<F>(raw) - F(1);
    }

    template<std::floating_point F>
    constexpr F uniform01() noexcept {
        return toUniform01<F>(operator()());
    }

    template<std::floating_point F>
    constexpr F uniformM11() noexcept {
        return toUniformM11<F>(operator()());
    }

    template<std::floating_point F>
    constexpr F triangularM11() noexcept {
        return uniform01<F>() + uniform01<F>() - F(1);
    }

    void fillRaw(std::span<std::uint64_t> out) noexcept {
        auto s0 = _state[0], s1 = _state[1], s2 = _state[2], s3 = _state[3];
        for (auto& v : out) {
            v = next(s0, s1, s2, s3);
        }
        _state[0] = s0;
        _state[1] = s1;
        _state[2] = s2;
        _state[3] = s3;
    }

private:
    static constexpr std::uint64_t splitMix64(std::uint64_t& x) noexcept {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

static_assert(std::uniform_random_bit_generator<Xoshiro256pp>);

} // namespace gr::rng

#endif // GNURADIO_ALGORITHM_XOSHIRO256PP_HPP
