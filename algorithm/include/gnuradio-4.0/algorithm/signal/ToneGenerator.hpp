#ifndef GNURADIO_ALGORITHM_TONE_GENERATOR_HPP
#define GNURADIO_ALGORITHM_TONE_GENERATOR_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <numbers>
#include <span>

namespace gr::signal {

enum class ToneType : int { Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos };

/**
 * @brief Stateful oscillator producing Const/Sin/Cos/Square/Saw/Triangle/FastSin/FastCos waveforms.
 *
 * Sin/Cos call std::sin/std::cos per sample (high precision, no drift).
 * FastSin/FastCos use a recursive phasor rotation (one complex multiply per sample,
 * ~10x faster, with periodic renormalization every 65536 samples to bound drift).
 * output(t) = A * waveform(2*pi*f*t + phase) + O.
 * frequency <= 0 is coerced to Const at configure time.
 */
template<std::floating_point F>
struct ToneGenerator {
    ToneType _type      = ToneType::Sin;
    F        _frequency = F(1);
    F        _amplitude = F(1);
    F        _offset    = F(0);
    F        _phase     = F(0);

    F _currentTime   = F(0);
    F _timeTick      = F(0);
    F _omega         = F(0); // 2*pi*f, precomputed in configure()
    F _phaseInCycles = F(0); // phase / (2*pi), precomputed in configure()

    // Recursive phasor state for FastSin/FastCos
    std::complex<F> _phasor{F(1), F(0)};
    std::complex<F> _rotation{F(1), F(0)};
    std::size_t     _sampleCount = 0;

    void configure(ToneType type, F frequency, F sampleRate, F phase, F amplitude, F offset) noexcept {
        constexpr F pi2 = F(2) * std::numbers::pi_v<F>;
        _frequency      = frequency;
        _amplitude      = amplitude;
        _offset         = offset;
        _phase          = phase;
        _timeTick       = F(1) / sampleRate;
        _type           = (frequency <= F(0) && type != ToneType::Const) ? ToneType::Const : type;
        _omega          = pi2 * _frequency;
        _phaseInCycles  = _phase / pi2;
        initPhasor();
    }

    void reset() noexcept {
        _currentTime = F(0);
        initPhasor();
    }

    [[nodiscard]] constexpr F generateSample() noexcept {
        const F value = computeSample();
        advanceState();
        return value;
    }

    void fill(std::span<F> out) noexcept {
        switch (_type) {
        case ToneType::FastSin: fillPhasor(out, [](F /*pr*/, F pi) { return pi; }); return;
        case ToneType::FastCos: fillPhasor(out, [](F pr, F /*pi*/) { return pr; }); return;
        default:
            for (auto& sample : out) {
                sample = generateSample();
            }
            return;
        }
    }

    [[nodiscard]] constexpr std::complex<F> generateComplexSample() noexcept {
        const F         theta = _omega * _currentTime + _phase;
        std::complex<F> result;

        switch (_type) {
        case ToneType::Sin: {
            result = std::complex<F>(_amplitude * std::sin(theta) + _offset, -_amplitude * std::cos(theta));
            break;
        }
        case ToneType::Cos: {
            result = std::complex<F>(_amplitude * std::cos(theta) + _offset, _amplitude * std::sin(theta));
            break;
        }
        case ToneType::FastSin:
            // analytic signal of sin: {sin(θ), -cos(θ)} from phasor = exp(jθ)
            result = std::complex<F>(_amplitude * _phasor.imag() + _offset, -_amplitude * _phasor.real());
            break;
        case ToneType::FastCos:
            // analytic signal of cos: {cos(θ), sin(θ)} from phasor = exp(jθ)
            result = std::complex<F>(_amplitude * _phasor.real() + _offset, _amplitude * _phasor.imag());
            break;
        default: result = std::complex<F>(computeSample(), F(0)); break;
        }
        advanceState();
        return result;
    }

    void fillComplex(std::span<std::complex<F>> out) noexcept {
        switch (_type) {
        case ToneType::FastSin:
            // analytic signal of sin: {sin(θ), -cos(θ)}
            fillPhasorComplex(out, [](F pr, F pi) { return std::complex<F>(pi, -pr); });
            return;
        case ToneType::FastCos:
            // analytic signal of cos: {cos(θ), sin(θ)}
            fillPhasorComplex(out, [](F pr, F pi) { return std::complex<F>(pr, pi); });
            return;
        default:
            for (auto& sample : out) {
                sample = generateComplexSample();
            }
            return;
        }
    }

private:
    template<typename ExtractComponent>
    void fillPhasor(std::span<F> out, ExtractComponent extract) noexcept {
        const F    rr = _rotation.real(), ri = _rotation.imag();
        const F    amp = _amplitude, off = _offset;
        const auto n = out.size();

        // K=2 interleaved phasors: two independent FMA chains execute in parallel,
        // breaking the 8-cycle serial dependency of a single phasor rotation
        F       prA = _phasor.real(), piA = _phasor.imag();
        F       prB = prA * rr - piA * ri, piB = prA * ri + piA * rr;
        const F rr2 = rr * rr - ri * ri, ri2 = F(2) * rr * ri;

        const std::size_t nPairs = n / 2;
        std::size_t       i      = 0;
        for (std::size_t p = 0; p < nPairs; ++p, i += 2) {
            out[i]        = amp * extract(prA, piA) + off;
            out[i + 1]    = amp * extract(prB, piB) + off;
            const F newRA = prA * rr2 - piA * ri2;
            const F newIA = prA * ri2 + piA * rr2;
            const F newRB = prB * rr2 - piB * ri2;
            const F newIB = prB * ri2 + piB * rr2;
            prA           = newRA;
            piA           = newIA;
            prB           = newRB;
            piB           = newIB;
            if ((p & 0x7FFF) == 0x7FFF) {
                F invMag = F(1) / std::sqrt(prA * prA + piA * piA);
                prA *= invMag;
                piA *= invMag;
                invMag = F(1) / std::sqrt(prB * prB + piB * piB);
                prB *= invMag;
                piB *= invMag;
            }
        }
        if (n & 1) {
            out[i]     = amp * extract(prA, piA) + off;
            const F nr = prA * rr - piA * ri;
            const F ni = prA * ri + piA * rr;
            prA        = nr;
            piA        = ni;
        }
        _phasor = {prA, piA};
        _sampleCount += n;
        _currentTime += _timeTick * static_cast<F>(n);
    }

    template<typename ExtractComponent>
    void fillPhasorComplex(std::span<std::complex<F>> out, ExtractComponent extract) noexcept {
        const F    rr = _rotation.real(), ri = _rotation.imag();
        const F    amp = _amplitude, off = _offset;
        const auto n = out.size();

        F       prA = _phasor.real(), piA = _phasor.imag();
        F       prB = prA * rr - piA * ri, piB = prA * ri + piA * rr;
        const F rr2 = rr * rr - ri * ri, ri2 = F(2) * rr * ri;

        const std::size_t nPairs = n / 2;
        std::size_t       i      = 0;
        for (std::size_t p = 0; p < nPairs; ++p, i += 2) {
            const auto rawA = extract(prA, piA);
            const auto rawB = extract(prB, piB);
            out[i]          = std::complex<F>(amp * rawA.real() + off, amp * rawA.imag());
            out[i + 1]      = std::complex<F>(amp * rawB.real() + off, amp * rawB.imag());
            const F newRA   = prA * rr2 - piA * ri2;
            const F newIA   = prA * ri2 + piA * rr2;
            const F newRB   = prB * rr2 - piB * ri2;
            const F newIB   = prB * ri2 + piB * rr2;
            prA             = newRA;
            piA             = newIA;
            prB             = newRB;
            piB             = newIB;
            if ((p & 0x7FFF) == 0x7FFF) {
                F invMag = F(1) / std::sqrt(prA * prA + piA * piA);
                prA *= invMag;
                piA *= invMag;
                invMag = F(1) / std::sqrt(prB * prB + piB * piB);
                prB *= invMag;
                piB *= invMag;
            }
        }
        if (n & 1) {
            const auto raw = extract(prA, piA);
            out[i]         = std::complex<F>(amp * raw.real() + off, amp * raw.imag());
            const F nr     = prA * rr - piA * ri;
            const F ni     = prA * ri + piA * rr;
            prA            = nr;
            piA            = ni;
        }
        _phasor = {prA, piA};
        _sampleCount += n;
        _currentTime += _timeTick * static_cast<F>(n);
    }

    void initPhasor() noexcept {
        constexpr F pi2  = F(2) * std::numbers::pi_v<F>;
        const F     dphi = pi2 * _frequency * _timeTick;
        _rotation        = std::complex<F>(std::cos(dphi), std::sin(dphi));
        _phasor          = std::complex<F>(std::cos(_phase), std::sin(_phase));
        _sampleCount     = 0;
    }

    constexpr void advanceState() noexcept {
        _currentTime += _timeTick;
        if (_type == ToneType::FastSin || _type == ToneType::FastCos) {
            _phasor *= _rotation;
            if ((++_sampleCount & 0xFFFF) == 0) {
                const F invMag = F(1) / std::abs(_phasor);
                _phasor        = {_phasor.real() * invMag, _phasor.imag() * invMag};
            }
        }
    }

    [[nodiscard]] constexpr F computeSample() const noexcept {
        const F theta = _omega * _currentTime + _phase;
        const F cycle = _frequency * _currentTime + _phaseInCycles;

        switch (_type) {
        case ToneType::Sin: return _amplitude * std::sin(theta) + _offset;
        case ToneType::Cos: return _amplitude * std::cos(theta) + _offset;
        case ToneType::FastSin: return _amplitude * _phasor.imag() + _offset;
        case ToneType::FastCos: return _amplitude * _phasor.real() + _offset;
        case ToneType::Const: return _amplitude + _offset;
        case ToneType::Square: {
            const F frac = cycle - std::floor(cycle);
            return (frac < F(0.5)) ? _amplitude + _offset : -_amplitude + _offset;
        }
        case ToneType::Saw: return _amplitude * (F(2) * (cycle - std::floor(cycle + F(0.5)))) + _offset;
        case ToneType::Triangle: {
            return _amplitude * (F(4) * std::abs(cycle - std::floor(cycle + F(0.75)) + F(0.25)) - F(1)) + _offset;
        }
        }
        return F(0);
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_TONE_GENERATOR_HPP
