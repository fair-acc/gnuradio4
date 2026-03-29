#ifndef GNURADIO_BURST_TAPER_HPP
#define GNURADIO_BURST_TAPER_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <expected>
#include <functional>
#include <numbers>
#include <ranges>
#include <source_location>
#include <span>
#include <vector>

#include <gnuradio-4.0/Message.hpp>

namespace gr::algorithm {

enum class TaperType {
    None,
    Linear,
    RaisedCosine, // param = power exponent (default 1)
    Tukey,        // param = roll-off ratio (default 0.5)
    Gaussian,     // param = sigma (default 0.4)
    Mushroom,     // zero-integral polynomial (C1)
    MushroomSine, // zero-integral sinusoidal (C∞)
};

/**
 * @brief Burst taper generator and real-time envelope for pulsed RF applications.
 *
 * Mushroom/MushroomSine have zero-integral property for RF cavity phase preservation.
 *   Mushroom:     g(u) = -12u² + 28u³ - 15u⁴
 *   MushroomSine: g(u) = (1-cos(πu))/2 - 3π/8·sin³(πu)
 * Origin: J. Tückmantel, R.J. Steinhagen (CERN)
 */
template<std::floating_point T = float, std::size_t N = std::dynamic_extent, typename Allocator = std::allocator<T>>
struct BurstTaper {
    enum class Phase : std::uint8_t { Off, RampUp, On, RampDown };

    using CoeffStorage = std::conditional_t<N == std::dynamic_extent, std::vector<T, Allocator>, std::array<T, N>>;

    TaperType    _taperType    = TaperType::Linear;
    T            _rampTime     = T{0}; // [s]
    T            _sampleRate   = T{1}; // [Hz]
    T            _shapeParam   = T{0};
    Phase        _phase        = Phase::Off;
    std::size_t  _rampPosition = 0UZ;
    bool         _targetOn     = false;
    CoeffStorage _riseCoefficients{};
    CoeffStorage _fallCoefficients{};

    BurstTaper() = default;

    explicit BurstTaper(TaperType type, T rampTimeSec = T{0}, T sampleRateHz = T{1}, T param = T{0}) {
        if (rampTimeSec >= T{0} && sampleRateHz > T{0}) {
            _taperType  = type;
            _rampTime   = rampTimeSec;
            _sampleRate = sampleRateHz;
            _shapeParam = param;
            buildCoefficients();
        }
    }

    [[nodiscard]] std::expected<void, gr::Error> configure(TaperType type, T rampTimeSec, T sampleRateHz, T param = T{0}, std::source_location location = std::source_location::current()) {
        if (rampTimeSec < T{0}) {
            return std::unexpected(gr::Error(std::format("rampTime must be non-negative, got {}", rampTimeSec), location));
        }
        if (sampleRateHz <= T{0}) {
            return std::unexpected(gr::Error(std::format("sampleRate must be positive, got {}", sampleRateHz), location));
        }
        if constexpr (N != std::dynamic_extent) {
            auto nSamples = static_cast<std::size_t>(std::round(rampTimeSec * sampleRateHz));
            if (nSamples != N) {
                return std::unexpected(gr::Error(std::format("fixed ramp length {} does not match rampTime*sampleRate = {}", N, nSamples), location));
            }
        }
        _taperType  = type;
        _rampTime   = rampTimeSec;
        _sampleRate = sampleRateHz;
        _shapeParam = param;
        reset();
        buildCoefficients();
        return {};
    }

    constexpr void reset() noexcept {
        _phase        = Phase::Off;
        _rampPosition = 0UZ;
        _targetOn     = false;
    }

    [[nodiscard]] bool setTarget(bool on, bool force = false) noexcept {
        _targetOn      = on;
        auto startRamp = [this](Phase rampPhase) {
            if (rampLength() == 0UZ) {
                _phase = (rampPhase == Phase::RampUp) ? Phase::On : Phase::Off;
            } else {
                _phase        = rampPhase;
                _rampPosition = 0UZ;
            }
        };

        switch (_phase) {
        case Phase::Off:
            if (on) {
                startRamp(Phase::RampUp);
                return true;
            }
            return false;
        case Phase::On:
            if (!on) {
                startRamp(Phase::RampDown);
                return true;
            }
            return false;
        case Phase::RampUp:
            if (!on && force) {
                _rampPosition = std::min(_rampPosition, rampLength());
                _rampPosition = rampLength() - _rampPosition;
                _phase        = Phase::RampDown;
                return true;
            }
            return !on;
        case Phase::RampDown:
            if (on && force) {
                _rampPosition = std::min(_rampPosition, rampLength());
                _rampPosition = rampLength() - _rampPosition;
                _phase        = Phase::RampUp;
                return true;
            }
            return on;
        }
        return false;
    }

    [[nodiscard]] constexpr Phase       phase() const noexcept { return _phase; }
    [[nodiscard]] constexpr bool        targetOn() const noexcept { return _targetOn; }
    [[nodiscard]] constexpr bool        isOn() const noexcept { return _phase == Phase::On; }
    [[nodiscard]] constexpr bool        isOff() const noexcept { return _phase == Phase::Off; }
    [[nodiscard]] constexpr bool        isRamping() const noexcept { return _phase == Phase::RampUp || _phase == Phase::RampDown; }
    [[nodiscard]] constexpr std::size_t rampLength() const noexcept { return _riseCoefficients.size(); }
    [[nodiscard]] constexpr std::size_t remainingSamples() const noexcept { return isRamping() ? (rampLength() - std::min(_rampPosition, rampLength())) : 0UZ; }
    [[nodiscard]] constexpr TaperType   type() const noexcept { return _taperType; }
    [[nodiscard]] constexpr T           rampTime() const noexcept { return _rampTime; }
    [[nodiscard]] constexpr T           sampleRate() const noexcept { return _sampleRate; }

    [[nodiscard]] constexpr T processOne() noexcept {
        switch (_phase) {
        case Phase::Off: return T{0};
        case Phase::On: return T{1};
        case Phase::RampUp:
        case Phase::RampDown: {
            const auto len = rampLength();
            if (len == 0UZ || _rampPosition >= len) {
                transitionAfterRamp();
                return (_phase == Phase::On) ? T{1} : T{0};
            }
            T val = (_phase == Phase::RampUp) ? _riseCoefficients[_rampPosition] : _fallCoefficients[_rampPosition];
            _rampPosition++;
            if (_rampPosition >= len) {
                transitionAfterRamp();
            }
            return val;
        }
        }
        return T{0};
    }

    void applyTo(std::span<const T> in, std::span<T> out) noexcept { applyBulk(in.data(), out.data(), std::min(in.size(), out.size()), false); }

    void applyInPlace(std::span<T> samples) noexcept { applyBulk(samples.data(), samples.data(), samples.size(), true); }

    static void generateEdge(TaperType type, std::span<T> out, bool rising = true, T param = T{0}) {
        const auto nSamples = out.size();
        if (nSamples == 0UZ) {
            return;
        }
        const T nInv    = (nSamples > 1UZ) ? T{1} / static_cast<T>(nSamples - 1UZ) : T{1};
        auto    indices = std::views::iota(0UZ, nSamples);
        if (rising) {
            std::ranges::transform(indices, out.begin(), [=](std::size_t i) { return computeRise(type, static_cast<T>(i) * nInv, param); });
        } else {
            std::ranges::transform(indices, out.begin(), [=](std::size_t i) { return computeRise(type, T{1} - static_cast<T>(i) * nInv, param); });
        }
    }

    static std::vector<T> generateEdge(TaperType type, std::size_t nSamples, bool rising = true, T param = T{0}) {
        std::vector<T> edge(nSamples);
        generateEdge(type, std::span(edge), rising, param);
        return edge;
    }

    static void generateTaper(TaperType type, std::span<T> out, std::size_t nRise, std::size_t nFlat, std::size_t nFall, T param = T{0}) {
        assert(out.size() >= nRise + nFlat + nFall);
        generateEdge(type, out.subspan(0, nRise), true, param);
        std::ranges::fill(out.subspan(nRise, nFlat), T{1});
        generateEdge(type, out.subspan(nRise + nFlat, nFall), false, param);
    }

    static std::vector<T> generateTaper(TaperType type, std::size_t nRise, std::size_t nFlat, std::size_t nFall, T param = T{0}) {
        std::vector<T> taper(nRise + nFlat + nFall);
        generateTaper(type, std::span(taper), nRise, nFlat, nFall, param);
        return taper;
    }

private:
    void applyBulk(const T* in, T* out, std::size_t n, bool inPlace) noexcept {
        std::size_t i = 0UZ;
        while (i < n) {
            switch (_phase) {
            case Phase::Off: std::fill_n(out + i, n - i, T{0}); return;
            case Phase::On:
                if (!inPlace) {
                    std::copy_n(in + i, n - i, out + i);
                }
                return;
            case Phase::RampUp: i += applyRamp(in, out, i, n, _riseCoefficients.data()); break;
            case Phase::RampDown: i += applyRamp(in, out, i, n, _fallCoefficients.data()); break;
            }
        }
    }

    std::size_t applyRamp(const T* in, T* out, std::size_t offset, std::size_t n, const T* coeffs) noexcept {
        const auto count = std::min(rampLength() - _rampPosition, n - offset);
        std::ranges::transform(std::span(in + offset, count), std::span(coeffs + _rampPosition, count), out + offset, std::multiplies{});
        _rampPosition += count;
        if (_rampPosition >= rampLength()) {
            transitionAfterRamp();
            if (_phase == Phase::Off) {
                std::fill_n(out + offset + count, n - offset - count, T{0});
            }
        }
        return count;
    }

    constexpr void transitionAfterRamp() noexcept {
        if (_phase == Phase::RampUp) {
            _phase = _targetOn ? Phase::On : Phase::RampDown;
        } else {
            _phase = _targetOn ? Phase::RampUp : Phase::Off;
        }
        _rampPosition = 0UZ;
    }

    void buildCoefficients() {
        if constexpr (N == std::dynamic_extent) {
            auto nSamples = static_cast<std::size_t>(std::round(_rampTime * _sampleRate));
            _riseCoefficients.resize(nSamples);
            _fallCoefficients.resize(nSamples);
        }
        generateEdge(_taperType, std::span(_riseCoefficients), true, _shapeParam);
        generateEdge(_taperType, std::span(_fallCoefficients), false, _shapeParam);
    }

    static T computeRise(TaperType type, T u, T param) {
        using std::numbers::pi_v;
        switch (type) {
        case TaperType::None: return T{1};
        case TaperType::Linear: return u;
        case TaperType::RaisedCosine: {
            T base = (T{1} - std::cos(pi_v<T> * u)) / T{2};
            return (param > T{0} && param != T{1}) ? std::pow(base, param) : base;
        }
        case TaperType::Tukey: {
            T alpha = (param > T{0}) ? param : T{0.5};
            return (u < alpha / T{2}) ? (T{1} - std::cos(T{2} * pi_v<T> * u / alpha)) / T{2} : T{1};
        }
        case TaperType::Gaussian: {
            T sigma  = (param > T{0}) ? param : T{2} / T{5};
            T raw    = std::exp(T{-0.5} * ((u - T{1}) / sigma) * ((u - T{1}) / sigma));
            T rawAt0 = std::exp(T{-0.5} / (sigma * sigma));
            return (raw - rawAt0) / (T{1} - rawAt0);
        }
        case TaperType::Mushroom: {
            T u2 = u * u;
            return u2 * (T{-12} + u * (T{28} - T{15} * u));
        }
        case TaperType::MushroomSine: {
            T sinPu = std::sin(pi_v<T> * u);
            return (T{1} - std::cos(pi_v<T> * u)) / T{2} - (T{3} * pi_v<T> / T{8}) * sinPu * sinPu * sinPu;
        }
        }
        return u;
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_BURST_TAPER_HPP
