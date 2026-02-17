#ifndef GNURADIO_ALGORITHM_NOISE_GENERATOR_HPP
#define GNURADIO_ALGORITHM_NOISE_GENERATOR_HPP

#include <complex>
#include <concepts>
#include <cstdint>
#include <span>

#include <gnuradio-4.0/algorithm/rng/GaussianNoise.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace gr::signal {

enum class NoiseType : int { Uniform, Triangular, Gaussian };

/**
 * @brief Stateful noise source producing Uniform/Triangular/Gaussian samples.
 *
 * output = A * noise + O.  Base noise ranges: Uniform [-1,+1), Triangular [-1,+1), Gaussian N(0,1).
 * Complex output: Uniform/Triangular use independent real+imag components;
 * Gaussian uses Option B (nI, nQ ~ N(0, 1/2), E[|n|^2] = 1 before amplitude scaling).
 */
template<std::floating_point F>
struct NoiseGenerator {
    NoiseType                 _type      = NoiseType::Uniform;
    F                         _amplitude = F(1);
    F                         _offset    = F(0);
    gr::rng::Xoshiro256pp     _rng;
    gr::rng::GaussianNoise<F> _gauss{_rng};

    NoiseGenerator() = default;
    NoiseGenerator(const NoiseGenerator& other) noexcept : _type(other._type), _amplitude(other._amplitude), _offset(other._offset), _rng(other._rng), _gauss(_rng) {
        _gauss._hasSpare = other._gauss._hasSpare;
        _gauss._spare    = other._gauss._spare;
    }
    NoiseGenerator(NoiseGenerator&& other) noexcept : _type(other._type), _amplitude(other._amplitude), _offset(other._offset), _rng(std::move(other._rng)), _gauss(_rng) {
        _gauss._hasSpare = other._gauss._hasSpare;
        _gauss._spare    = other._gauss._spare;
    }
    NoiseGenerator& operator=(const NoiseGenerator& other) noexcept {
        if (this != &other) {
            _type            = other._type;
            _amplitude       = other._amplitude;
            _offset          = other._offset;
            _rng             = other._rng;
            _gauss           = gr::rng::GaussianNoise<F>(_rng);
            _gauss._hasSpare = other._gauss._hasSpare;
            _gauss._spare    = other._gauss._spare;
        }
        return *this;
    }
    NoiseGenerator& operator=(NoiseGenerator&& other) noexcept {
        if (this != &other) {
            _type            = other._type;
            _amplitude       = other._amplitude;
            _offset          = other._offset;
            _rng             = std::move(other._rng);
            _gauss           = gr::rng::GaussianNoise<F>(_rng);
            _gauss._hasSpare = other._gauss._hasSpare;
            _gauss._spare    = other._gauss._spare;
        }
        return *this;
    }

    void configure(NoiseType type, F amplitude, F offset, std::uint64_t seed) noexcept {
        _type      = type;
        _amplitude = amplitude;
        _offset    = offset;
        _rng.seed(seed);
        _gauss.reset();
    }

    void reset(std::uint64_t seed) noexcept {
        _rng.seed(seed);
        _gauss.reset();
    }

    [[nodiscard]] constexpr F generateSample() noexcept { return _amplitude * rawSample() + _offset; }

    void fill(std::span<F> out) noexcept {
        const F amp = _amplitude;
        const F off = _offset;
        switch (_type) {
        case NoiseType::Uniform: {
            auto s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
            for (auto& s : out) {
                s = amp * gr::rng::Xoshiro256pp::toUniformM11<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3)) + off;
            }
            _rng._state[0] = s0;
            _rng._state[1] = s1;
            _rng._state[2] = s2;
            _rng._state[3] = s3;
            return;
        }
        case NoiseType::Triangular: {
            auto s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
            for (auto& s : out) {
                const F u1 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                const F u2 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                s          = amp * (u1 + u2 - F(1)) + off;
            }
            _rng._state[0] = s0;
            _rng._state[1] = s1;
            _rng._state[2] = s2;
            _rng._state[3] = s3;
            return;
        }
        case NoiseType::Gaussian: _gauss.fill(out, amp, off); return;
        }
    }

    [[nodiscard]] constexpr std::complex<F> generateComplexSample() noexcept {
        if (_type == NoiseType::Gaussian) {
            const auto raw = _gauss.complexSample();
            return {_amplitude * raw.real() + _offset, _amplitude * raw.imag()};
        }
        return {_amplitude * rawSample() + _offset, _amplitude * rawSample()};
    }

    void fillComplex(std::span<std::complex<F>> out) noexcept {
        const F amp = _amplitude;
        const F off = _offset;
        switch (_type) {
        case NoiseType::Uniform: {
            auto s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
            for (auto& s : out) {
                const F re = amp * gr::rng::Xoshiro256pp::toUniformM11<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3)) + off;
                const F im = amp * gr::rng::Xoshiro256pp::toUniformM11<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                s          = {re, im};
            }
            _rng._state[0] = s0;
            _rng._state[1] = s1;
            _rng._state[2] = s2;
            _rng._state[3] = s3;
            return;
        }
        case NoiseType::Triangular: {
            auto s0 = _rng._state[0], s1 = _rng._state[1], s2 = _rng._state[2], s3 = _rng._state[3];
            for (auto& s : out) {
                const F u1 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                const F u2 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                const F u3 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                const F u4 = gr::rng::Xoshiro256pp::toUniform01<F>(gr::rng::Xoshiro256pp::next(s0, s1, s2, s3));
                s          = {amp * (u1 + u2 - F(1)) + off, amp * (u3 + u4 - F(1))};
            }
            _rng._state[0] = s0;
            _rng._state[1] = s1;
            _rng._state[2] = s2;
            _rng._state[3] = s3;
            return;
        }
        case NoiseType::Gaussian: _gauss.fillComplex(out, amp, off); return;
        }
    }

private:
    [[nodiscard]] constexpr F rawSample() noexcept {
        switch (_type) {
        case NoiseType::Uniform: return _rng.uniformM11<F>();
        case NoiseType::Triangular: return _rng.triangularM11<F>();
        case NoiseType::Gaussian: return _gauss();
        }
        return F(0);
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_NOISE_GENERATOR_HPP
