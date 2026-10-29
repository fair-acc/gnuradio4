#ifndef GNURADIO_DIGITAL_CONSTELLATION_HPP
#define GNURADIO_DIGITAL_CONSTELLATION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/RaisedCosine.hpp>

namespace gr::digital {

/**
 * @brief Gray-mapped symbol alphabets with unit mean energy, shared by ConstellationModulator, CostasLoop and ConstellationDemapper.
 * A label holds the bits of a symbol, the first bit sent as the most significant; BPSK puts its bit on the real axis, and PSK8 puts the label m XOR (m >> 1) at exp(j m pi / 4).
 * QPSK and QAM16 are square grids: the first half of the label picks the in-phase level and the second half the quadrature level, each Gray-coded from the lowest level, so neighbouring points differ in one bit.
 * The nearest point is the maximum-likelihood decision for equally likely symbols in white Gaussian noise.
 *
 * J. G. Proakis and M. Salehi, "Digital Communications", 5th ed. New York: McGraw-Hill, 2008, ch. 3 and 4.
 */
enum class Constellation { BPSK, QPSK, PSK8, QAM16 };

[[nodiscard]] constexpr std::size_t bitsPerSymbol(Constellation constellation) noexcept {
    switch (constellation) {
    case Constellation::BPSK: return 1UZ;
    case Constellation::QPSK: return 2UZ;
    case Constellation::PSK8: return 3UZ;
    case Constellation::QAM16: return 4UZ;
    }
    std::unreachable();
}

namespace detail {

[[nodiscard]] constexpr std::size_t grayToBinary(std::size_t grayCode) noexcept {
    std::size_t binary = 0UZ;
    for (; grayCode != 0UZ; grayCode >>= 1UZ) {
        binary ^= grayCode;
    }
    return binary;
}

[[nodiscard]] constexpr double grayCodedLevel(std::size_t grayCode, std::size_t bitsPerAxis) noexcept {
    const std::size_t nLevels = 1UZ << bitsPerAxis;
    return 2. * static_cast<double>(grayToBinary(grayCode)) - static_cast<double>(nLevels - 1UZ);
}

[[nodiscard]] inline std::complex<double> squareGridPoint(std::size_t label, std::size_t bitsPerAxis) noexcept {
    const std::size_t          nLevels    = 1UZ << bitsPerAxis;
    const double               meanEnergy = 2. * static_cast<double>(nLevels * nLevels - 1UZ) / 3.;
    const std::complex<double> point{grayCodedLevel(label >> bitsPerAxis, bitsPerAxis), grayCodedLevel(label & (nLevels - 1UZ), bitsPerAxis)};
    return point / std::sqrt(meanEnergy);
}

[[nodiscard]] inline std::complex<double> constellationPoint(Constellation constellation, std::size_t label) noexcept {
    switch (constellation) {
    case Constellation::BPSK: return {grayCodedLevel(label, 1UZ), 0.};
    case Constellation::QPSK: return squareGridPoint(label, 1UZ);
    case Constellation::PSK8: return std::polar(1., std::numbers::pi / 4. * static_cast<double>(grayToBinary(label)));
    case Constellation::QAM16: return squareGridPoint(label, 2UZ);
    }
    std::unreachable();
}

} // namespace detail

template<std::floating_point T>
[[nodiscard]] std::vector<std::complex<T>> constellationPoints(Constellation constellation) {
    std::vector<std::complex<T>> points(1UZ << bitsPerSymbol(constellation));
    std::ranges::transform(std::views::iota(0UZ, points.size()), points.begin(), [constellation](std::size_t label) { return std::complex<T>(detail::constellationPoint(constellation, label)); });
    return points;
}

template<std::floating_point T>
[[nodiscard]] std::size_t nearestLabel(std::span<const std::complex<T>> points, std::complex<T> symbol) noexcept {
    if (!std::isfinite(symbol.real()) || !std::isfinite(symbol.imag())) {
        return 0UZ;
    }
    const auto decisionMetric = [symbol](std::complex<T> point) { return std::real(symbol * std::conj(point)) - std::norm(point) / T{2}; };
    return static_cast<std::size_t>(std::distance(points.begin(), std::ranges::max_element(points, {}, decisionMetric)));
}

[[nodiscard]] constexpr std::size_t packLabel(std::span<const std::uint8_t> bits) noexcept {
    return std::accumulate(bits.begin(), bits.end(), 0UZ, [](std::size_t label, std::uint8_t bit) { return (label << 1UZ) | (bit != 0U ? 1UZ : 0UZ); });
}

constexpr void unpackLabel(std::size_t label, std::span<std::uint8_t> bits) noexcept {
    for (std::size_t i = 0UZ; i < bits.size(); ++i) {
        bits[i] = static_cast<std::uint8_t>((label >> (bits.size() - 1UZ - i)) & 1UZ);
    }
}

GR_REGISTER_BLOCK(gr::digital::ConstellationModulator, [T], [ float, double ])

template<std::floating_point T>
struct ConstellationModulator : Block<ConstellationModulator<T>, Resampling<1U, 4U, false>> {
    using Description = Doc<R""(@brief Constellation modulator: maps bits to constellation points and sends each point as a root-raised-cosine pulse in complex baseband.

A symbol carries 1, 2, 3 or 4 bits for binary, quadrature or 8-point phase-shift keying (BPSK, QPSK, PSK8) or 16-point quadrature amplitude modulation (QAM16); the first bit is the most significant.
The points are Gray-mapped, so neighbouring points differ in one bit, and have unit mean energy; BPSK sends 0 as -1 and 1 as +1.
The output has samples_per_symbol samples per symbol, a delay of filter_span / 2 symbols and a bandwidth of (1 + excess_bandwidth) times the symbol rate.
At the receiver, RootRaisedCosineFilter with the same settings is the matched filter.

 * J. G. Proakis and M. Salehi, "Digital Communications", 5th ed. New York: McGraw-Hill, 2008, ch. 3 and 9.)"">;

    static constexpr gr::Size_t kMaxSamplesPerSymbol = 1024U;
    static constexpr gr::Size_t kMaxFilterSpan       = 1024U;

    PortIn<std::uint8_t>     in;
    PortOut<std::complex<T>> out;

    Annotated<Constellation, "constellation", Visible, Doc<"symbol alphabet and its bit mapping">> constellation      = Constellation::BPSK;
    Annotated<gr::Size_t, "samples per symbol", Visible, Doc<"output samples per symbol">>         samples_per_symbol = 4U;
    Annotated<float, "excess bandwidth", Visible, Doc<"roll-off of the root-raised-cosine pulse">> excess_bandwidth   = 0.35f;
    Annotated<gr::Size_t, "filter span", Doc<"pulse length in symbols">>                           filter_span        = 10U;

    GR_MAKE_REFLECTABLE(ConstellationModulator, in, out, constellation, samples_per_symbol, excess_bandwidth, filter_span);

    std::vector<std::complex<T>> _points;
    std::vector<T>               _pulse;
    std::vector<std::complex<T>> _recentSymbols;

    void start() {
        updateFromSettings();
        reset();
    }

    void reset() { std::ranges::fill(_recentSymbols, std::complex<T>{}); }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& bits, OutputViewLike auto& output) noexcept {
        const std::size_t bitsInSymbol                 = bitsPerSymbol(constellation.value);
        const std::size_t samplesPerSymbol             = samples_per_symbol.value;
        const std::size_t nSymbols                     = bits.size() / bitsInSymbol;
        const bool        chunkSizedBeforeFormatChange = nSymbols * bitsInSymbol != bits.size() || nSymbols * samplesPerSymbol != output.size();
        if (chunkSizedBeforeFormatChange) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        for (std::size_t n = 0UZ; n < nSymbols; ++n) {
            std::shift_right(_recentSymbols.begin(), _recentSymbols.end(), 1);
            _recentSymbols.front() = _points[packLabel(std::span(bits).subspan(n * bitsInSymbol, bitsInSymbol))];
            for (std::size_t phase = 0UZ; phase < samplesPerSymbol; ++phase) {
                std::complex<T> sample{};
                for (std::size_t age = 0UZ; age * samplesPerSymbol + phase < _pulse.size(); ++age) {
                    sample += _recentSymbols[age] * _pulse[age * samplesPerSymbol + phase];
                }
                output[n * samplesPerSymbol + phase] = sample;
            }
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validSamplesPerSymbol = samples_per_symbol.value >= 2U && samples_per_symbol.value <= kMaxSamplesPerSymbol;
        const bool validFilterSpan       = filter_span.value >= 1U && filter_span.value <= kMaxFilterSpan;
        const bool validExcessBandwidth  = std::isfinite(excess_bandwidth.value) && excess_bandwidth.value > 0.f && excess_bandwidth.value <= 1.f;
        if (!validSamplesPerSymbol || !validFilterSpan || !validExcessBandwidth) {
            this->emitErrorMessage("settingsChanged()", std::format("samples_per_symbol {} must be within [2, {}], filter_span {} within [1, {}] and excess_bandwidth {} within (0, 1], keeping the previous settings", samples_per_symbol.value, kMaxSamplesPerSymbol, filter_span.value, kMaxFilterSpan, excess_bandwidth.value));
            constellation      = gr::meta::parseEnum<Constellation>(oldSettings.value_or<std::string_view>("constellation", "")).value_or(constellation.value);
            samples_per_symbol = oldSettings.value_or<gr::Size_t>("samples_per_symbol", samples_per_symbol.value);
            filter_span        = oldSettings.value_or<gr::Size_t>("filter_span", filter_span.value);
            excess_bandwidth   = oldSettings.value_or<float>("excess_bandwidth", excess_bandwidth.value);
        }
        updateFromSettings();
    }

    void updateFromSettings() {
        _points = constellationPoints<T>(constellation.value);
        _pulse  = gr::algorithm::filter::rootRaisedCosineTaps<T>(samples_per_symbol.value, static_cast<double>(excess_bandwidth.value), filter_span.value);
        _recentSymbols.resize(filter_span.value + 1UZ, std::complex<T>{});
        this->input_chunk_size  = static_cast<gr::Size_t>(bitsPerSymbol(constellation.value));
        this->output_chunk_size = samples_per_symbol.value;
    }
};

GR_REGISTER_BLOCK(gr::digital::ConstellationDemapper, [T], [ float, double ])

template<std::floating_point T>
struct ConstellationDemapper : Block<ConstellationDemapper<T>, Resampling<1U, 1U, false>> {
    using Description = Doc<R""(@brief Constellation demapper: turns each symbol into the bits of the nearest constellation point, undoing the mapping of ConstellationModulator.

In white Gaussian noise, the nearest point is the most likely one (hard decision).
The input must be one sample per symbol with the carrier removed, as SymbolSync and CostasLoop deliver.
QAM16 also needs unit mean power, for example from an AutomaticGainControl after SymbolSync.
Carrier recovery may lock with the constellation turned onto itself, for BPSK by half a turn, which inverts all bits; for BPSK, a DifferentialDecoder after this block removes that.

 * J. G. Proakis and M. Salehi, "Digital Communications", 5th ed. New York: McGraw-Hill, 2008, ch. 4.)"">;

    PortIn<std::complex<T>> in;
    PortOut<std::uint8_t>   out;

    Annotated<Constellation, "constellation", Visible, Doc<"symbol alphabet and its bit mapping">> constellation = Constellation::BPSK;

    GR_MAKE_REFLECTABLE(ConstellationDemapper, in, out, constellation);

    std::vector<std::complex<T>> _points;

    void start() { updateFromSettings(); }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& symbols, OutputViewLike auto& bits) noexcept {
        const std::size_t bitsInSymbol                 = bitsPerSymbol(constellation.value);
        const bool        chunkSizedBeforeFormatChange = bits.size() != symbols.size() * bitsInSymbol;
        if (chunkSizedBeforeFormatChange) {
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        for (std::size_t n = 0UZ; n < symbols.size(); ++n) {
            unpackLabel(nearestLabel<T>(_points, symbols[n]), std::span(bits).subspan(n * bitsInSymbol, bitsInSymbol));
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { updateFromSettings(); }

    void updateFromSettings() {
        _points                 = constellationPoints<T>(constellation.value);
        this->input_chunk_size  = 1U;
        this->output_chunk_size = static_cast<gr::Size_t>(bitsPerSymbol(constellation.value));
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_CONSTELLATION_HPP
