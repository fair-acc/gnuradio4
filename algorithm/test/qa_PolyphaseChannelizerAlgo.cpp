#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using Channelizer = gr::algorithm::filter::PolyphaseChannelizer<float>;
using Complex     = std::complex<float>;

/// every channel of every output set, through the two passes the algorithm documents
[[nodiscard]] std::vector<Complex> channelise(std::span<const Complex> input, std::span<const float> phases, std::size_t phaseLen, std::size_t nChannels, std::size_t nOutputs) {
    const auto           table = Channelizer::designTwiddles(nChannels);
    std::vector<Complex> twiddles(nChannels);
    for (std::size_t j = 0UZ; j < nChannels; ++j) {
        twiddles[j] = Complex{table[j].real(), table[j].imag()};
    }
    std::vector<Complex> arms(nChannels);
    std::vector<Complex> out(nOutputs * nChannels);
    for (std::size_t m = 0UZ; m < nOutputs; ++m) {
        for (std::size_t p = 0UZ; p < nChannels; ++p) {
            arms[p] = Channelizer::armOf<Complex>(input, phases, phaseLen, nChannels, m, p);
        }
        for (std::size_t k = 0UZ; k < nChannels; ++k) {
            out[m * nChannels + k] = Channelizer::channelAt(arms.data(), twiddles.data(), nChannels, k);
        }
    }
    return out;
}

/// a complex exponential at the centre of channel `k`: one cycle per nChannels samples, k times over
[[nodiscard]] std::vector<Complex> channelTone(std::size_t nSamples, std::size_t nChannels, std::size_t k) {
    std::vector<Complex> x(nSamples);
    for (std::size_t n = 0UZ; n < nSamples; ++n) {
        const float angle = 2.f * std::numbers::pi_v<float> * static_cast<float>(k) * static_cast<float>(n) / static_cast<float>(nChannels);
        x[n]              = Complex{std::cos(angle), std::sin(angle)};
    }
    return x;
}

/// index of the channel carrying the most energy in the last output set
[[nodiscard]] std::size_t strongestChannel(std::span<const Complex> out, std::size_t nChannels) {
    const std::size_t last  = out.size() / nChannels - 1UZ;
    std::size_t       best  = 0UZ;
    float             bestM = -1.f;
    for (std::size_t k = 0UZ; k < nChannels; ++k) {
        if (const float mag = std::abs(out[last * nChannels + k]); mag > bestM) {
            bestM = mag;
            best  = k;
        }
    }
    return best;
}
} // namespace

const boost::ut::suite<"PolyphaseChannelizer"> channelizerTests = [] {
    using namespace boost::ut;
    using gr::test::eq;

    "the prototype has unity gain at DC"_test = [] {
        constexpr std::size_t kChannels = 8UZ;
        const auto            prototype = Channelizer::designPrototype(64UZ, kChannels);
        const float           sum       = std::accumulate(prototype.begin(), prototype.end(), 0.f);
        expect(approx(sum, 1.f, 1e-5f)) << std::format("prototype sums to {}, so a constant would change level", sum);
    };

    "each phase carries an equal share"_test = [] {
        constexpr std::size_t kChannels = 8UZ;
        const auto            prototype = Channelizer::designPrototype(64UZ, kChannels);
        const auto            phases    = Channelizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Channelizer::phaseLength(prototype.size(), kChannels);
        expect(eq(phases.size(), kChannels * phaseLen));
        for (std::size_t p = 0UZ; p < kChannels; ++p) {
            const float sum = std::accumulate(phases.begin() + static_cast<std::ptrdiff_t>(p * phaseLen), phases.begin() + static_cast<std::ptrdiff_t>((p + 1UZ) * phaseLen), 0.f);
            expect(approx(sum, 1.f, 1e-5f)) << std::format("phase {} sums to {}: unequal phases beat at the output rate", p, sum);
        }
    };

    "a tone at a channel centre lands in that channel"_test = [] {
        constexpr std::size_t kChannels = 8UZ;
        constexpr std::size_t kOutputs  = 16UZ;
        const auto            prototype = Channelizer::designPrototype(64UZ, kChannels);
        const auto            phases    = Channelizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Channelizer::phaseLength(prototype.size(), kChannels);
        const std::size_t     nInput    = Channelizer::windowLength(kOutputs, kChannels, phaseLen);

        for (std::size_t k = 0UZ; k < kChannels; ++k) {
            const auto        input = channelTone(nInput, kChannels, k);
            const auto        out   = channelise(input, phases, phaseLen, kChannels, kOutputs);
            const std::size_t got   = strongestChannel(out, kChannels);
            expect(eq(got, k)) << std::format("a tone at the centre of channel {} came out strongest in channel {}", k, got);
        }
    };

    "the other channels are suppressed"_test = [] {
        constexpr std::size_t kChannels = 8UZ;
        constexpr std::size_t kOutputs  = 16UZ;
        const auto            prototype = Channelizer::designPrototype(128UZ, kChannels);
        const auto            phases    = Channelizer::decompose(prototype, kChannels);
        const std::size_t     phaseLen  = Channelizer::phaseLength(prototype.size(), kChannels);
        const std::size_t     nInput    = Channelizer::windowLength(kOutputs, kChannels, phaseLen);

        constexpr std::size_t kTone = 3UZ;
        const auto            input = channelTone(nInput, kChannels, kTone);
        const auto            out   = channelise(input, phases, phaseLen, kChannels, kOutputs);

        const std::size_t last   = kOutputs - 1UZ;
        const float       signal = std::abs(out[last * kChannels + kTone]);
        for (std::size_t k = 0UZ; k < kChannels; ++k) {
            if (k == kTone) {
                continue;
            }
            const float leak = std::abs(out[last * kChannels + k]);
            expect(lt(leak, 0.1f * signal)) << std::format("channel {} holds {:.4f} against the wanted {:.4f}", k, leak, signal);
        }
    };
};

int main() { /* not needed for UT */ }
