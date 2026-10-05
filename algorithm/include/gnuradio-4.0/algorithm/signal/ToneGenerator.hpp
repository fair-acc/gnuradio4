#ifndef GNURADIO_ALGORITHM_TONE_GENERATOR_HPP
#define GNURADIO_ALGORITHM_TONE_GENERATOR_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <numbers>
#include <span>
#include <type_traits>

namespace gr::signal {

enum class ToneType : int { Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos };

/**
 * @brief Stateful oscillator producing Const/Sin/Cos/Square/Saw/Triangle/FastSin/FastCos waveforms.
 *
 * Sin/Cos call std::sin/std::cos per sample (high precision, no drift).
 * FastSin/FastCos use a recursive phasor rotation (one complex multiply per sample,
 * ~10x faster, re-anchored to the accumulated phase every 1024 (float) or 65536 (double) samples to bound drift).
 * output = A * waveform(2*pi*cycles + phase) + O, with cycles += f/fs per sample (double, wrapped to [0, 1)):
 * a frequency change keeps the phase continuous, a phase change steps it by the difference.
 * frequency <= 0 is coerced to Const at configure time.
 */
template<std::floating_point F>
struct ToneGenerator {
    ToneType _type      = ToneType::Sin;
    F        _frequency = F(1);
    F        _amplitude = F(1);
    F        _offset    = F(0);
    F        _phase     = F(0);

    double _cycles          = 0.;
    double _cyclesPerSample = 0.;
    double _phaseInCycles   = 0.;

    // Recursive phasor state for FastSin/FastCos
    std::complex<F> _phasor{F(1), F(0)};
    std::complex<F> _rotation{F(1), F(0)};
    std::size_t     _sampleCount = 0;

    void configure(ToneType type, F frequency, F sampleRate, F phase, F amplitude, F offset) noexcept {
        _frequency       = frequency;
        _amplitude       = amplitude;
        _offset          = offset;
        _phase           = phase;
        _type            = (frequency <= F(0) && type != ToneType::Const) ? ToneType::Const : type;
        _cyclesPerSample = static_cast<double>(frequency) / static_cast<double>(sampleRate);
        _phaseInCycles   = static_cast<double>(phase) / kTwoPi;
        initPhasor();
    }

    void reset() noexcept {
        _cycles = 0.;
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
        const F         theta = thetaAt(_cycles);
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
    static constexpr double      kTwoPi              = 2. * std::numbers::pi;
    static constexpr std::size_t kAnchorIntervalMask = std::is_same_v<F, float> ? 0x3FFUZ : 0xFFFFUZ; // float phasor rounding drifts ~1e-8 rad/sample

    [[nodiscard]] static constexpr double wrapped(double cycles) noexcept { return cycles - std::floor(cycles); }

    [[nodiscard]] constexpr double cyclesAfter(std::size_t nSamples) const noexcept { return wrapped(_cycles + static_cast<double>(nSamples) * _cyclesPerSample); }

    [[nodiscard]] constexpr F thetaAt(double cycles) const noexcept { return static_cast<F>(kTwoPi * wrapped(cycles + _phaseInCycles)); }

    [[nodiscard]] std::complex<F> phasorAt(double cycles) const noexcept {
        const double theta = kTwoPi * wrapped(cycles + _phaseInCycles);
        return {static_cast<F>(std::cos(theta)), static_cast<F>(std::sin(theta))};
    }

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
            if ((p & (kAnchorIntervalMask >> 1U)) == (kAnchorIntervalMask >> 1U)) {
                const std::complex<F> anchorA = phasorAt(cyclesAfter(i + 2));
                const std::complex<F> anchorB = phasorAt(cyclesAfter(i + 3));
                prA                           = anchorA.real();
                piA                           = anchorA.imag();
                prB                           = anchorB.real();
                piB                           = anchorB.imag();
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
        _cycles = cyclesAfter(n);
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
            if ((p & (kAnchorIntervalMask >> 1U)) == (kAnchorIntervalMask >> 1U)) {
                const std::complex<F> anchorA = phasorAt(cyclesAfter(i + 2));
                const std::complex<F> anchorB = phasorAt(cyclesAfter(i + 3));
                prA                           = anchorA.real();
                piA                           = anchorA.imag();
                prB                           = anchorB.real();
                piB                           = anchorB.imag();
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
        _cycles = cyclesAfter(n);
    }

    void initPhasor() noexcept {
        const double dphi = kTwoPi * _cyclesPerSample;
        _rotation         = std::complex<F>(static_cast<F>(std::cos(dphi)), static_cast<F>(std::sin(dphi)));
        _phasor           = phasorAt(_cycles);
        _sampleCount      = 0;
    }

    constexpr void advanceState() noexcept {
        _cycles = cyclesAfter(1UZ);
        if (_type == ToneType::FastSin || _type == ToneType::FastCos) {
            _phasor *= _rotation;
            if ((++_sampleCount & kAnchorIntervalMask) == 0) {
                _phasor = phasorAt(_cycles);
            }
        }
    }

    [[nodiscard]] constexpr F computeSample() const noexcept {
        const F theta = thetaAt(_cycles);
        const F cycle = static_cast<F>(wrapped(_cycles + _phaseInCycles));

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
