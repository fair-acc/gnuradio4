#ifndef GNURADIO_DIGITAL_MANCHESTERCODING_HPP
#define GNURADIO_DIGITAL_MANCHESTERCODING_HPP

#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::digital {

GR_REGISTER_BLOCK(gr::digital::ManchesterEncoder)

struct ManchesterEncoder : Block<ManchesterEncoder, Resampling<1U, 2U, true>> {
    using Description = Doc<R""(@brief Manchester (biphase-level) encoder: turns each bit into two half-bits, 1 into (1, 0) and 0 into (0, 1), at twice the sample rate.

Sent as BPSK, every bit changes sign in its middle, so the signal has no power at the carrier, as the Radio Data System (RDS) requires.
ManchesterDecoder undoes it.)"">;

    PortIn<std::uint8_t>  in;
    PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(ManchesterEncoder, in, out);

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& bits, OutputViewLike auto& halfBits) const noexcept {
        assert(halfBits.size() == 2UZ * bits.size());
        for (std::size_t n = 0UZ; n < bits.size(); ++n) {
            const bool bit          = bits[n] != 0U;
            halfBits[2UZ * n]       = bit;
            halfBits[2UZ * n + 1UZ] = !bit;
        }
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::digital::ManchesterDecoder, [T], [ float, double ])

template<std::floating_point T>
struct ManchesterDecoder : Block<ManchesterDecoder<T>, Resampling<2U, 1U, true>> {
    using Description = Doc<R""(@brief Manchester (biphase-level) decoder: turns each pair of BPSK symbols into one bit, at half the sample rate.

It undoes ManchesterEncoder followed by BPSK modulation: a bit is 1 when the real part of its first symbol is larger than that of its second.
It finds by itself which two symbols form a bit: in the right pairs the two halves cancel, so it compares the mean of |Re(a) + Re(b)| over about the last 32 bits for both ways of pairing.
When the mean of the other pairing falls below half of the current one, it switches, which loses or repeats one bit.)"">;

    static constexpr T kMismatchAveraging = T{1} / T{32};
    static constexpr T kSwitchRatio       = T(0.5);

    PortIn<std::complex<T>> in;
    PortOut<std::uint8_t>   out;

    GR_MAKE_REFLECTABLE(ManchesterDecoder, in, out);

    T    _previousSymbol{};
    T    _alignedMismatch{};
    T    _lateMismatch{};
    bool _pairsStartLate = false;

    void reset() noexcept {
        _previousSymbol  = T{0};
        _alignedMismatch = T{0};
        _lateMismatch    = T{0};
        _pairsStartLate  = false;
    }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& symbols, OutputViewLike auto& bits) noexcept {
        assert(symbols.size() == 2UZ * bits.size());
        for (std::size_t n = 0UZ; n < bits.size(); ++n) {
            const T previous = _previousSymbol;
            const T first    = symbols[2UZ * n].real();
            const T second   = symbols[2UZ * n + 1UZ].real();
            updatePairing(std::abs(first + second), std::abs(previous + first));
            bits[n]         = _pairsStartLate ? previous > first : first > second;
            _previousSymbol = second;
        }
        return gr::work::Status::OK;
    }

    void updatePairing(T alignedMismatch, T lateMismatch) noexcept {
        if (!std::isfinite(alignedMismatch) || !std::isfinite(lateMismatch)) {
            return;
        }
        _alignedMismatch += kMismatchAveraging * (alignedMismatch - _alignedMismatch);
        _lateMismatch += kMismatchAveraging * (lateMismatch - _lateMismatch);
        const T currentMismatch = _pairsStartLate ? _lateMismatch : _alignedMismatch;
        const T otherMismatch   = _pairsStartLate ? _alignedMismatch : _lateMismatch;
        if (otherMismatch < kSwitchRatio * currentMismatch) {
            _pairsStartLate = !_pairsStartLate;
        }
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_MANCHESTERCODING_HPP
