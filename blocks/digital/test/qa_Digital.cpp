#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/filter/RaisedCosine.hpp>
#include <gnuradio-4.0/digital/Constellation.hpp>
#include <gnuradio-4.0/digital/CostasLoop.hpp>
#include <gnuradio-4.0/digital/DifferentialCoding.hpp>
#include <gnuradio-4.0/digital/ManchesterCoding.hpp>
#include <gnuradio-4.0/digital/RDSDecoder.hpp>
#include <gnuradio-4.0/digital/RootRaisedCosineFilter.hpp>
#include <gnuradio-4.0/digital/SymbolSync.hpp>
#include <gnuradio-4.0/math/Rotator.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {

using gr::digital::Constellation;

constexpr std::array kConstellations{Constellation::BPSK, Constellation::QPSK, Constellation::PSK8, Constellation::QAM16};

constexpr double      kSampleRate     = 240'000.;
constexpr double      kCarrierOffset  = 2'000.;
constexpr double      kCarrierPhase   = 0.7;
constexpr std::size_t kLockingSymbols = 24'000UZ;

struct Carrier {
    double offset;
    double phase;
};

constexpr Carrier kOffsetCarrier{kCarrierOffset, kCarrierPhase};

constexpr float kDefaultExcessBandwidth = 0.35f;

constexpr float      kInfinity = std::numeric_limits<float>::infinity();
constexpr float      kNaN      = std::numeric_limits<float>::quiet_NaN();
constexpr std::array kInvalidForPositive{0.f, -1.f, kInfinity, kNaN};
constexpr std::array kInvalidForFinite{kInfinity, -kInfinity, kNaN};

template<typename T>
constexpr T kTolerance = std::same_as<T, float> ? T(1e-5) : T(1e-12);

[[nodiscard]] constexpr double phaseStep(double frequency) { return 2. * std::numbers::pi * frequency / kSampleRate; }

template<typename T>
[[nodiscard]] std::complex<T> phasor(double phase) {
    return std::polar(T{1}, static_cast<T>(std::remainder(phase, 2. * std::numbers::pi)));
}

template<typename T>
[[nodiscard]] T distance(std::complex<T> actual, std::complex<T> expected) {
    return std::abs(actual - expected);
}

[[nodiscard]] std::vector<std::uint8_t> randomBits(std::size_t nBits) {
    std::mt19937              generator{42U};
    std::vector<std::uint8_t> bits(nBits);
    std::ranges::generate(bits, [&generator] { return static_cast<std::uint8_t>(generator() & 1U); });
    return bits;
}

[[nodiscard]] std::vector<std::uint8_t> differentiallyEncoded(std::span<const std::uint8_t> bits) {
    std::vector<std::uint8_t> encoded(bits.size());
    std::inclusive_scan(bits.begin(), bits.end(), encoded.begin(), std::bit_xor<>{});
    return encoded;
}

[[nodiscard]] constexpr double bpskSymbol(std::uint8_t bit) { return bit == 0U ? -1. : 1.; }

[[nodiscard]] std::vector<double> randomBpskSymbols(std::size_t nSymbols) {
    const std::vector<std::uint8_t> bits = randomBits(nSymbols);
    std::vector<double>             symbols(nSymbols);
    std::ranges::transform(bits, symbols.begin(), bpskSymbol);
    return symbols;
}

[[nodiscard]] std::vector<std::complex<double>> bpskSymbols(std::span<const std::uint8_t> bits) {
    std::vector<std::complex<double>> symbols(bits.size());
    std::ranges::transform(bits, symbols.begin(), bpskSymbol);
    return symbols;
}

[[nodiscard]] std::string settingName(Constellation constellation) { return std::string(gr::meta::enumName(constellation).value_or("")); }

[[nodiscard]] std::vector<std::complex<double>> expectedPoints(Constellation constellation) {
    constexpr double                  kHalfSqrt2 = std::numbers::sqrt2 / 2.;
    constexpr std::array              kPsk8EighthTurnsByLabel{0., 1., 3., 2., 7., 6., 4., 5.};
    constexpr std::array              kQam16LevelsByGrayCode{-3., -1., 3., 1.};
    std::vector<std::complex<double>> points;
    switch (constellation) {
    case Constellation::BPSK: return {{-1., 0.}, {1., 0.}};
    case Constellation::QPSK: return {{-kHalfSqrt2, -kHalfSqrt2}, {-kHalfSqrt2, kHalfSqrt2}, {kHalfSqrt2, -kHalfSqrt2}, {kHalfSqrt2, kHalfSqrt2}};
    case Constellation::PSK8: std::ranges::transform(kPsk8EighthTurnsByLabel, std::back_inserter(points), [](double eighthTurns) { return std::polar(1., std::numbers::pi / 4. * eighthTurns); }); break;
    case Constellation::QAM16:
        for (const double inPhase : kQam16LevelsByGrayCode) {
            for (const double quadrature : kQam16LevelsByGrayCode) {
                points.emplace_back(inPhase / std::sqrt(10.), quadrature / std::sqrt(10.));
            }
        }
        break;
    }
    return points;
}

[[nodiscard]] std::vector<std::uint8_t> labelBits(std::size_t label, std::size_t bitsInSymbol) {
    std::vector<std::uint8_t> bits(bitsInSymbol);
    for (std::size_t i = 0UZ; i < bitsInSymbol; ++i) {
        bits[i] = static_cast<std::uint8_t>((label >> (bitsInSymbol - 1UZ - i)) & 1UZ);
    }
    return bits;
}

[[nodiscard]] std::vector<std::uint8_t> allLabelBits(Constellation constellation) {
    const std::size_t         bitsInSymbol = gr::digital::bitsPerSymbol(constellation);
    std::vector<std::uint8_t> bits;
    for (std::size_t label = 0UZ; label < (1UZ << bitsInSymbol); ++label) {
        std::ranges::copy(labelBits(label, bitsInSymbol), std::back_inserter(bits));
    }
    return bits;
}

[[nodiscard]] std::vector<std::size_t> labelsOf(std::span<const std::uint8_t> bits, std::size_t bitsInSymbol) {
    std::vector<std::size_t> labels(bits.size() / bitsInSymbol);
    for (std::size_t n = 0UZ; n < labels.size(); ++n) {
        labels[n] = gr::digital::packLabel(bits.subspan(n * bitsInSymbol, bitsInSymbol));
    }
    return labels;
}

[[nodiscard]] std::vector<std::size_t> randomLabels(std::size_t nSymbols, std::size_t nLabels) {
    std::mt19937             generator{42U};
    std::vector<std::size_t> labels(nSymbols);
    std::ranges::generate(labels, [&generator, nLabels] { return generator() % nLabels; });
    return labels;
}

[[nodiscard]] std::vector<std::complex<double>> shapedReference(std::span<const std::complex<double>> symbols, std::size_t samplesPerSymbol, double excessBandwidth, std::size_t filterSpan) {
    const std::vector<double>         pulse = gr::algorithm::filter::rootRaisedCosineTaps<double>(samplesPerSymbol, excessBandwidth, filterSpan);
    std::vector<std::complex<double>> shaped(symbols.size() * samplesPerSymbol);
    for (std::size_t n = 0UZ; n < symbols.size(); ++n) {
        for (std::size_t k = 0UZ; k < pulse.size() && n * samplesPerSymbol + k < shaped.size(); ++k) {
            shaped[n * samplesPerSymbol + k] += symbols[n] * pulse[k];
        }
    }
    return shaped;
}

[[nodiscard]] std::vector<std::complex<float>> matchedFilterOutput(const std::vector<double>& symbols, double samplesPerSymbol, double clockOffset, double timingOffset, std::complex<double> carrier) {
    constexpr std::ptrdiff_t         kPulseSpanInSymbols = 16;
    const auto                       nSamples            = static_cast<std::size_t>(static_cast<double>(symbols.size() - static_cast<std::size_t>(2 * kPulseSpanInSymbols)) * samplesPerSymbol);
    const auto                       lastSymbol          = static_cast<std::ptrdiff_t>(symbols.size()) - 1;
    std::vector<std::complex<float>> samples(nSamples);
    for (std::size_t i = 0UZ; i < nSamples; ++i) {
        const double         time    = static_cast<double>(i) / samplesPerSymbol * (1. + clockOffset) + timingOffset;
        const std::ptrdiff_t nearest = static_cast<std::ptrdiff_t>(time);
        double               value   = 0.;
        for (std::ptrdiff_t k = std::max<std::ptrdiff_t>(0, nearest - kPulseSpanInSymbols); k <= std::min(lastSymbol, nearest + kPulseSpanInSymbols); ++k) {
            value += symbols[static_cast<std::size_t>(k)] * gr::algorithm::filter::raisedCosine(time - static_cast<double>(k), 0.35);
        }
        samples[i] = std::complex<float>(carrier * value);
    }
    return samples;
}

template<typename TBlock>
[[nodiscard]] gr::work::Status processViews(TBlock& block, auto input, auto... outputs) {
    return block.processBulk(input, outputs...);
}

template<typename T>
[[nodiscard]] std::vector<std::complex<T>> modulate(gr::digital::ConstellationModulator<T>& block, std::span<const std::uint8_t> bits) {
    std::vector<std::complex<T>> shaped(bits.size() / gr::digital::bitsPerSymbol(block.constellation.value) * block.samples_per_symbol.value);
    boost::ut::expect(processViews(block, bits, std::span(shaped)) == gr::work::Status::OK);
    return shaped;
}

template<typename T>
[[nodiscard]] std::vector<std::uint8_t> demap(gr::digital::ConstellationDemapper<T>& block, const std::vector<std::complex<T>>& symbols) {
    std::vector<std::uint8_t> bits(symbols.size() * gr::digital::bitsPerSymbol(block.constellation.value));
    boost::ut::expect(processViews(block, std::span(symbols), std::span(bits)) == gr::work::Status::OK);
    return bits;
}

template<typename TBlock>
using InputType = typename decltype(TBlock::in)::value_type;

template<typename TBlock>
using OutputTypes = typename gr::traits::block::stream_output_port_types<TBlock>::template apply<std::tuple>;

template<typename TPort>
using PlainSpan = std::conditional_t<TPort::kIsInput, std::span<const typename TPort::value_type>, std::span<typename TPort::value_type>>;

template<typename TBlock>
concept ProcessesSamples = !TBlock::ResamplingControl::kEnabled && gr::traits::block::can_processBulk_helper<TBlock, PlainSpan>;

template<ProcessesSamples TBlock>
[[nodiscard]] auto processSample(TBlock& block, InputType<TBlock> sample) {
    const std::span<const InputType<TBlock>> input{&sample, 1UZ};
    OutputTypes<TBlock>                      outputs{};
    std::apply([&block, &input](auto&... output) { boost::ut::expect(processViews(block, input, std::span{&output, 1UZ}...) == gr::work::Status::OK); }, outputs);
    if constexpr (std::tuple_size_v<OutputTypes<TBlock>> == 1UZ) {
        return std::get<0>(outputs);
    } else {
        return outputs;
    }
}

template<ProcessesSamples TBlock>
[[nodiscard]] auto processEach(TBlock& block, const std::vector<InputType<TBlock>>& inputs) {
    std::vector<std::tuple_element_t<0UZ, OutputTypes<TBlock>>> outputs(inputs.size());
    boost::ut::expect(processViews(block, std::span(inputs), std::span(outputs)) == gr::work::Status::OK);
    return outputs;
}

template<typename T>
void expectImpulseResponse(gr::digital::RootRaisedCosineFilter<T>& block, const std::vector<T>& taps) {
    using namespace boost::ut;
    std::vector<std::complex<T>> impulse(taps.size() + 8UZ);
    impulse.front()                             = {T{0}, T{2}};
    const std::vector<std::complex<T>> response = processEach(block, impulse);
    for (std::size_t i = 0UZ; i < response.size(); ++i) {
        const T expected = i < taps.size() ? T{2} * taps[i] : T{0};
        expect(approx(distance(response[i], std::complex<T>{T{0}, expected}), T{0}, kTolerance<T>)) << "i=" << i;
    }
}

template<typename TTags>
[[nodiscard]] bool carriesSampleRate(const TTags& tags, float sampleRate) {
    constexpr float kRelativeTolerance = 1e-6f;
    return std::ranges::any_of(tags, [sampleRate](const auto& receivedTag) { return std::abs(receivedTag.map.template value_or<float>(gr::tag::kSampleRate.key(), 0.f) - sampleRate) <= kRelativeTolerance * sampleRate; });
}

template<typename T>
[[nodiscard]] std::complex<T> onCarrier(std::complex<double> point, std::size_t n, double amplitude, Carrier carrier) {
    return std::complex<T>(amplitude * point) * phasor<T>(carrier.phase + phaseStep(carrier.offset) * static_cast<double>(n));
}

template<typename TBlock>
void startOutsideScheduler(TBlock& block) {
    block.settings().init();
    std::ignore = block.settings().applyStagedParameters();
    if constexpr (requires { block.start(); }) {
        block.start();
    }
}

template<typename TBlock>
void applySettings(TBlock& block, const gr::property_map& settings) {
    boost::ut::expect(block.settings().set(settings).empty());
    boost::ut::expect(block.settings().activateContext() != std::nullopt);
    std::ignore = block.settings().applyStagedParameters();
}

template<typename TBlock>
void expectRejected(const gr::property_map& update) {
    using namespace boost::ut;
    using TInput = InputType<TBlock>;
    gr::MsgPortIn errors;
    TBlock        block;
    TBlock        unchanged;
    startOutsideScheduler(block);
    startOutsideScheduler(unchanged);
    expect(block.msgOut.connect(errors).has_value()) << fatal;
    const gr::property_map settingsBefore = block.settings().get();
    applySettings(block, update);
    expect(eq(errors.streamReader().available(), 1UZ)) << "one error reported";
    expect(block.settings().get() == settingsBefore) << "previous settings kept";
    if constexpr (ProcessesSamples<TBlock>) {
        for (const TInput sample : {TInput{1}, TInput{-1}}) {
            expect(processSample(block, sample) == processSample(unchanged, sample)) << "previous processing kept";
        }
    }
}

constexpr std::uint16_t    kProgrammeIdentification  = 0x1234U;
constexpr std::uint8_t     kProgrammeType            = 10U;
constexpr std::string_view kProgrammeServiceName     = "GR4 RDS ";
constexpr std::string_view kRadioText                = "Radio Data System test";
constexpr std::size_t      kRdsGroupBits             = 4UZ * gr::digital::rds::kBlockBits;
constexpr std::uint32_t    kNoAlternativeFrequencies = 0xE0CDU;

void appendBlock(std::vector<std::uint8_t>& bits, std::uint32_t information, gr::digital::rds::Offset offset) {
    const std::uint32_t block = gr::digital::rds::encodeBlock(static_cast<std::uint16_t>(information), offset);
    for (std::size_t bit = gr::digital::rds::kBlockBits; bit-- > 0UZ;) {
        bits.push_back(static_cast<std::uint8_t>((block >> bit) & 1U));
    }
}

void appendGroup(std::vector<std::uint8_t>& bits, std::uint32_t groupType, std::uint32_t groupSpecificBits, std::uint32_t blockC, std::uint32_t blockD, std::uint16_t programmeIdentification = kProgrammeIdentification) {
    using gr::digital::rds::Offset;
    appendBlock(bits, programmeIdentification, Offset::A);
    appendBlock(bits, (groupType << 12U) | (static_cast<std::uint32_t>(kProgrammeType) << 5U) | groupSpecificBits, Offset::B);
    appendBlock(bits, blockC, Offset::C);
    appendBlock(bits, blockD, Offset::D);
}

[[nodiscard]] std::uint32_t characterPair(std::string_view text, std::size_t index) { return (static_cast<std::uint32_t>(static_cast<unsigned char>(text[index])) << 8U) | static_cast<std::uint32_t>(static_cast<unsigned char>(text[index + 1UZ])); }

[[nodiscard]] std::vector<std::uint8_t> rdsBits(std::string_view programmeServiceName, std::string_view radioText, std::uint32_t textFlag) {
    std::vector<std::uint8_t> bits;
    for (std::uint32_t segment = 0U; segment < 4U; ++segment) {
        appendGroup(bits, 0U, segment, kNoAlternativeFrequencies, characterPair(programmeServiceName, 2UZ * segment));
    }
    std::string text = std::string(radioText) + '\r';
    text.resize((text.size() + 3UZ) / 4UZ * 4UZ, ' ');
    for (std::uint32_t segment = 0U; 4UZ * segment < text.size(); ++segment) {
        appendGroup(bits, 2U, (textFlag << 4U) | segment, characterPair(text, 4UZ * segment), characterPair(text, 4UZ * segment + 2UZ));
    }
    return bits;
}

template<typename TTags>
[[nodiscard]] std::vector<gr::property_map> stationTags(const TTags& tags) {
    std::vector<gr::property_map> stations;
    for (const auto& receivedTag : tags) {
        if (receivedTag.map.contains(gr::digital::RDSDecoder::kProgrammeIdentificationKey)) {
            stations.push_back(receivedTag.map);
        }
    }
    return stations;
}

[[nodiscard]] std::vector<gr::property_map> decodeRds(const std::vector<std::uint8_t>& bits) {
    using namespace boost::ut;
    using namespace gr::testing;
    gr::Graph flow;
    auto&     source  = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", static_cast<gr::Size_t>(bits.size())}, {"values", bits}, {"mark_tag", false}});
    auto&     decoder = flow.emplaceBlock<gr::digital::RDSDecoder>();
    auto&     sink    = flow.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>();
    expect(flow.connect<"out", "in">(source, decoder).has_value()) << fatal;
    expect(flow.connect<"out", "in">(decoder, sink).has_value()) << fatal;
    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value()) << fatal;
    expect(sched.runAndWait().has_value());
    expect(sink._samples == bits) << "bits pass unchanged";
    return stationTags(sink._tags);
}

void expectStation(const std::vector<gr::property_map>& stations, std::string_view radioText) {
    using namespace boost::ut;
    using gr::digital::RDSDecoder;
    expect(!stations.empty()) << fatal << "station data decoded";
    expect(std::ranges::all_of(stations, [](const gr::property_map& station) { return station.value_or<std::uint16_t>(RDSDecoder::kProgrammeIdentificationKey, 0U) == kProgrammeIdentification && station.value_or<std::uint8_t>(RDSDecoder::kProgrammeTypeKey, 0U) == kProgrammeType; })) << "PI code and PTY in every tag";
    expect(stations.back().value_or<std::string_view>(RDSDecoder::kProgrammeServiceNameKey, "") == std::string_view{"GR4 RDS"}) << "PS name without the trailing space";
    expect(stations.back().value_or<std::string_view>(RDSDecoder::kRadioTextKey, "") == radioText) << "radio text";
}

constexpr auto kTypes = std::tuple<float, double>{};

} // namespace

const auto constellationTests = [] {
    using namespace boost::ut;
    using namespace gr::digital;

    "points"_test = []<typename T> {
        for (const Constellation constellation : kConstellations) {
            const std::vector<std::complex<T>>      points   = constellationPoints<T>(constellation);
            const std::vector<std::complex<double>> expected = expectedPoints(constellation);
            expect(eq(points.size(), 1UZ << bitsPerSymbol(constellation))) << fatal;
            expect(eq(points.size(), expected.size())) << fatal;
            for (std::size_t label = 0UZ; label < points.size(); ++label) {
                expect(approx(distance(points[label], std::complex<T>(expected[label])), T{0}, kTolerance<T>)) << settingName(constellation) << " label=" << label;
            }
            const double meanEnergy = std::transform_reduce(points.begin(), points.end(), 0., std::plus<>{}, [](std::complex<T> point) { return static_cast<double>(std::norm(point)); }) / static_cast<double>(points.size());
            expect(approx(meanEnergy, 1., 1e-6)) << "unit mean energy, " << settingName(constellation);
        }
    } | kTypes;

    "Gray mapping"_test = [] {
        for (const Constellation constellation : kConstellations) {
            const std::vector<std::complex<double>> points = constellationPoints<double>(constellation);
            for (std::size_t label = 0UZ; label < points.size(); ++label) {
                const auto   distanceFrom    = [&](std::size_t other) { return std::abs(points[other] - points[label]); };
                auto         others          = std::views::iota(0UZ, points.size()) | std::views::filter([label](std::size_t other) { return other != label; });
                const double nearestDistance = std::ranges::min(others | std::views::transform(distanceFrom));
                for (const std::size_t neighbour : others | std::views::filter([&](std::size_t other) { return distanceFrom(other) < nearestDistance + 1e-9; })) {
                    expect(eq(std::popcount(label ^ neighbour), 1)) << "nearest neighbours differ in one bit, " << settingName(constellation) << " labels " << label << " and " << neighbour;
                }
            }
        }
    };

    "decision"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        for (const Constellation constellation : kConstellations) {
            const std::vector<std::complex<T>> points          = constellationPoints<T>(constellation);
            T                                  minimumDistance = Limits::max();
            for (std::size_t a = 0UZ; a < points.size(); ++a) {
                for (std::size_t b = a + 1UZ; b < points.size(); ++b) {
                    minimumDistance = std::min(minimumDistance, distance(points[a], points[b]));
                }
            }
            for (std::size_t label = 0UZ; label < points.size(); ++label) {
                for (std::size_t eighthTurns = 0UZ; eighthTurns < 8UZ; ++eighthTurns) {
                    const std::complex<T> symbol = points[label] + std::polar(T(0.49) * minimumDistance, static_cast<T>(std::numbers::pi / 4. * static_cast<double>(eighthTurns)));
                    expect(eq(nearestLabel<T>(points, symbol), label)) << "within half the minimum distance, " << settingName(constellation) << " label=" << label;
                }
                if (constellation != Constellation::QAM16) {
                    for (const T amplitude : {T(1e-3), T(1e3)}) {
                        expect(eq(nearestLabel<T>(points, amplitude * points[label]), label)) << "phase-shift keying ignores the amplitude, " << settingName(constellation);
                    }
                }
            }
            for (const std::complex<T> unusable : {std::complex<T>{Limits::quiet_NaN(), T{0}}, std::complex<T>{T{1}, Limits::infinity()}}) {
                expect(eq(nearestLabel<T>(points, unusable), 0UZ)) << "label 0 for a symbol that is not finite";
            }
        }
        expect(eq(nearestLabel<T>(constellationPoints<T>(Constellation::BPSK), std::complex<T>{T{0}, T{1}}), 0UZ)) << "the lower label at equal distance";
    } | kTypes;
};

const auto costasLoopTests = [] {
    using namespace boost::ut;
    using gr::digital::CostasLoop;

    "carrier offset"_test = []<typename T> {
        struct Lock {
            Constellation constellation;
            Carrier       carrier;
        };
        for (const Lock lock : {Lock{Constellation::BPSK, kOffsetCarrier}, Lock{Constellation::QPSK, {500., kCarrierPhase}}, Lock{Constellation::PSK8, {500., kCarrierPhase}}, Lock{Constellation::QAM16, {50., 0.3}}}) {
            const std::vector<std::complex<double>> points     = expectedPoints(lock.constellation);
            const std::vector<std::size_t>          labels     = randomLabels(kLockingSymbols + 64UZ, points.size());
            const std::vector<double>               amplitudes = lock.constellation == Constellation::QAM16 ? std::vector{1.} : std::vector{1e-3, 1e3};
            for (const double amplitude : amplitudes) {
                CostasLoop<T> block({{"constellation", settingName(lock.constellation)}});
                startOutsideScheduler(block);
                for (std::size_t n = 0UZ; n < kLockingSymbols; ++n) {
                    std::ignore = processSample(block, onCarrier<T>(points[labels[n]], n, amplitude, lock.carrier));
                }
                std::complex<double> rotation{};
                for (std::size_t n = kLockingSymbols; n < labels.size(); ++n) {
                    const auto [symbolEstimate, frequency]  = processSample(block, onCarrier<T>(points[labels[n]], n, amplitude, lock.carrier));
                    const std::complex<double> unitEstimate = std::complex<double>(symbolEstimate) / amplitude;
                    if (n == kLockingSymbols) {
                        rotation = unitEstimate / points[labels[n]];
                    }
                    expect(approx(std::abs(unitEstimate - points[labels[n]] * rotation), 0., static_cast<double>(kTolerance<T>))) << "points with one rotation, " << settingName(lock.constellation) << " amplitude=" << amplitude << " n=" << n;
                    expect(approx(frequency, static_cast<T>(lock.carrier.offset), T(1e-3))) << "loop frequency";
                }
            }
        }
    } | kTypes;

    "zero and non-finite input"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        CostasLoop<T> block({{"min_frequency", 100.f}, {"max_frequency", 300.f}});
        startOutsideScheduler(block);
        std::size_t n = 0UZ;
        for (const std::complex<T> unusable : {std::complex<T>{}, std::complex<T>{Limits::quiet_NaN(), T{0}}, std::complex<T>{Limits::infinity(), T{0}}, std::complex<T>{T{0}, -Limits::infinity()}}) {
            expect(approx(std::get<1>(processSample(block, unusable)), T{200}, T(1e-3))) << "frequency unchanged, n=" << n;
            ++n;
        }
        expect(approx(distance(std::get<0>(processSample(block, {T{1}, T{0}})), phasor<T>(-phaseStep(200.) * static_cast<double>(n))), T{0}, kTolerance<T>)) << "loop ran on at the start frequency";
    } | kTypes;

    "settings change"_test = []<typename T> {
        CostasLoop<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"min_frequency", 500.f}, {"max_frequency", 600.f}});
        expect(approx(std::get<1>(processSample(block, {T{1}, T{0}})), T{500}, T(1e-3))) << "loop frequency moved into the new range";

        applySettings(block, {{"constellation", settingName(Constellation::QPSK)}, {"min_frequency", 0.f}, {"max_frequency", 0.f}});
        std::complex<T> symbolEstimate;
        for (std::size_t n = 0UZ; n < 4'000UZ; ++n) {
            symbolEstimate = std::get<0>(processSample(block, phasor<T>(std::numbers::pi / 4. + 0.3)));
        }
        expect(approx(distance(symbolEstimate, phasor<T>(std::numbers::pi / 4.)), T{0}, kTolerance<T>)) << "locked onto the nearest QPSK point";
    } | kTypes;

    "range beyond sample_rate / 2"_test = []<typename T> {
        CostasLoop<T> block({{"sample_rate", 1'000.f}, {"min_frequency", 200.f}});
        startOutsideScheduler(block);
        expect(approx(std::get<1>(processSample(block, {T{1}, T{0}})), T{350}, T(1e-3))) << "range from 200 Hz to 10 kHz limited to 500 Hz, starting in the middle";
    } | kTypes;

    "reset"_test = []<typename T> {
        const std::vector<double> symbols = randomBpskSymbols(1'000UZ);
        CostasLoop<T>             block;
        startOutsideScheduler(block);
        for (std::size_t n = 0UZ; n < symbols.size(); ++n) {
            std::ignore = processSample(block, onCarrier<T>(symbols[n], n, 1., kOffsetCarrier));
        }
        block.reset();
        const auto [symbolEstimate, frequency] = processSample(block, {T{1}, T{0}});
        expect(approx(distance(symbolEstimate, std::complex<T>{T{1}, T{0}}), T{0}, kTolerance<T>)) << "zero phase";
        expect(approx(frequency, T{0}, T(1e-3))) << "middle of the frequency range";
    } | kTypes;

    "invalid settings"_test = [] {
        expectRejected<CostasLoop<float>>({{"sample_rate", 0.f}});
        for (const float invalid : kInvalidForPositive) {
            expectRejected<CostasLoop<float>>({{"sample_rate", 480'000.f}, {"loop_bandwidth", invalid}});
        }
        for (const float invalid : kInvalidForFinite) {
            expectRejected<CostasLoop<float>>({{"sample_rate", 480'000.f}, {"min_frequency", invalid}});
            expectRejected<CostasLoop<float>>({{"sample_rate", 480'000.f}, {"max_frequency", invalid}});
        }
        expectRejected<CostasLoop<float>>({{"min_frequency", 20'000.f}});
        expectRejected<CostasLoop<float>>({{"constellation", settingName(Constellation::QPSK)}, {"sample_rate", 0.f}});
    };
};

const auto constellationModulatorTests = [] {
    using namespace boost::ut;
    using gr::digital::ConstellationModulator;

    "pulse shaping"_test = []<typename T> {
        struct Shape {
            gr::Size_t samplesPerSymbol;
            gr::Size_t filterSpan;
            float      excessBandwidth;
        };
        const std::vector<std::uint8_t> bits = randomBits(200UZ);
        for (const Shape shape : {Shape{4U, 10U, 0.35f}, Shape{8U, 6U, 1.f}}) {
            ConstellationModulator<T> block({{"samples_per_symbol", shape.samplesPerSymbol}, {"filter_span", shape.filterSpan}, {"excess_bandwidth", shape.excessBandwidth}});
            startOutsideScheduler(block);
            const std::vector<std::complex<T>>      shaped   = modulate(block, bits);
            const std::vector<std::complex<double>> expected = shapedReference(bpskSymbols(bits), shape.samplesPerSymbol, static_cast<double>(shape.excessBandwidth), shape.filterSpan);
            expect(eq(shaped.size(), expected.size())) << fatal;
            for (std::size_t i = 0UZ; i < shaped.size(); ++i) {
                expect(approx(distance(shaped[i], std::complex<T>(expected[i])), T{0}, kTolerance<T>)) << "samples_per_symbol=" << shape.samplesPerSymbol << " i=" << i;
            }
        }
    } | kTypes;

    "mapping"_test = []<typename T> {
        for (const Constellation constellation : kConstellations) {
            ConstellationModulator<T> block({{"constellation", settingName(constellation)}});
            startOutsideScheduler(block);
            const std::vector<std::complex<T>>      shaped   = modulate(block, allLabelBits(constellation));
            const std::vector<std::complex<double>> expected = shapedReference(expectedPoints(constellation), 4UZ, static_cast<double>(kDefaultExcessBandwidth), 10UZ);
            expect(eq(shaped.size(), expected.size())) << fatal;
            for (std::size_t i = 0UZ; i < shaped.size(); ++i) {
                expect(approx(distance(shaped[i], std::complex<T>(expected[i])), T{0}, kTolerance<T>)) << settingName(constellation) << " i=" << i;
            }
        }
    } | kTypes;

    "chunks"_test = []<typename T> {
        const std::vector<std::uint8_t> bits = randomBits(200UZ);
        ConstellationModulator<T>       whole;
        ConstellationModulator<T>       chunked;
        startOutsideScheduler(whole);
        startOutsideScheduler(chunked);
        const std::vector<std::complex<T>> expected = modulate(whole, bits);
        std::vector<std::complex<T>>       shaped   = modulate(chunked, std::span(bits).first(37UZ));
        const std::vector<std::complex<T>> rest     = modulate(chunked, std::span(bits).subspan(37UZ));
        shaped.insert(shaped.end(), rest.begin(), rest.end());
        expect(shaped == expected) << "continuous across calls";
    } | kTypes;

    "settings change"_test = []<typename T> {
        const std::vector<std::uint8_t> bits = randomBits(50UZ);
        ConstellationModulator<T>       block;
        startOutsideScheduler(block);
        applySettings(block, {{"samples_per_symbol", gr::Size_t{8U}}});
        expect(eq(block.output_chunk_size.value, gr::Size_t{8U})) << "eight output samples per symbol";
        block.reset();
        const std::vector<std::complex<T>>      shaped   = modulate(block, bits);
        const std::vector<std::complex<double>> expected = shapedReference(bpskSymbols(bits), 8UZ, static_cast<double>(kDefaultExcessBandwidth), 10UZ);
        for (std::size_t i = 0UZ; i < shaped.size(); ++i) {
            expect(approx(distance(shaped[i], std::complex<T>(expected[i])), T{0}, kTolerance<T>)) << "i=" << i;
        }
        applySettings(block, {{"constellation", settingName(Constellation::QAM16)}});
        expect(eq(block.input_chunk_size.value, gr::Size_t{4U})) << "four input bits per symbol";
    } | kTypes;

    "reset"_test = []<typename T> {
        const std::vector<std::uint8_t> bits = randomBits(50UZ);
        ConstellationModulator<T>       used;
        ConstellationModulator<T>       fresh;
        startOutsideScheduler(used);
        startOutsideScheduler(fresh);
        std::ignore = modulate(used, bits);
        used.reset();
        expect(modulate(used, bits) == modulate(fresh, bits)) << "empty pulse history";
    } | kTypes;

    "invalid settings"_test = [] {
        for (const gr::Size_t invalid : {gr::Size_t{0U}, gr::Size_t{1U}, gr::Size_t{1'025U}}) {
            expectRejected<ConstellationModulator<float>>({{"samples_per_symbol", invalid}});
        }
        for (const gr::Size_t invalid : {gr::Size_t{0U}, gr::Size_t{1'025U}}) {
            expectRejected<ConstellationModulator<float>>({{"filter_span", invalid}});
        }
        for (const float invalid : {0.f, -0.1f, 1.5f, kInfinity, kNaN}) {
            expectRejected<ConstellationModulator<float>>({{"excess_bandwidth", invalid}});
        }
        expectRejected<ConstellationModulator<float>>({{"constellation", settingName(Constellation::QPSK)}, {"samples_per_symbol", gr::Size_t{1U}}});
    };

    "settings tag"_test = [] {
        using namespace gr::testing;
        constexpr gr::Size_t            kBits  = 1'000U;
        const std::vector<std::uint8_t> values = randomBits(64UZ);
        gr::Graph                       flow;
        auto&                           source    = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", kBits}, {"values", values}, {"mark_tag", false}});
        auto&                           modulator = flow.emplaceBlock<ConstellationModulator<float>>();
        auto&                           sink      = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>();
        source._tags.emplace_back(0UZ, gr::property_map{{"samples_per_symbol", gr::Size_t{8U}}});
        expect(flow.connect<"out", "in">(source, modulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(modulator, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());
        const std::vector<std::uint8_t>         sent     = std::views::iota(0UZ, std::size_t{kBits}) | std::views::transform([&values](std::size_t n) { return values[n % values.size()]; }) | std::ranges::to<std::vector>();
        const std::vector<std::complex<double>> expected = shapedReference(bpskSymbols(sent), 8UZ, static_cast<double>(kDefaultExcessBandwidth), 10UZ);
        expect(eq(sink._samples.size(), expected.size())) << fatal << "eight samples per bit from the first bit on";
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(approx(distance(sink._samples[i], std::complex<float>(expected[i])), 0.f, kTolerance<float>)) << "i=" << i;
        }
    };

    "in a graph"_test = [] {
        using namespace gr::testing;
        constexpr gr::Size_t kBits = 1'000U;
        for (const Constellation constellation : {Constellation::BPSK, Constellation::QPSK}) {
            const std::size_t bitsInSymbol = gr::digital::bitsPerSymbol(constellation);
            gr::Graph         flow;
            auto&             source    = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", kBits}, {"values", randomBits(64UZ)}, {"mark_tag", false}});
            auto&             modulator = flow.emplaceBlock<ConstellationModulator<float>>({{"constellation", settingName(constellation)}});
            auto&             sink      = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}, {"log_tags", true}});
            source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), 1'000.f}});

            expect(flow.connect<"out", "in">(source, modulator).has_value()) << fatal;
            expect(flow.connect<"out", "in">(modulator, sink).has_value()) << fatal;
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value()) << fatal;
            expect(sched.runAndWait().has_value());

            expect(eq(sink._samples.size(), 4UZ * kBits / bitsInSymbol)) << "four samples per symbol, " << settingName(constellation);
            const auto rateTag = std::ranges::find_if(sink._tags, [](const auto& receivedTag) { return receivedTag.map.contains(gr::tag::kSampleRate.key()); });
            expect(rateTag != sink._tags.end()) << fatal;
            expect(approx(rateTag->map.value_or<float>(gr::tag::kSampleRate.key(), 0.f), 4'000.f / static_cast<float>(bitsInSymbol), 1e-3f)) << "sample rate scaled by samples_per_symbol over the bits per symbol";
        }
    };
};

const auto symbolSyncTests = [] {
    using namespace boost::ut;
    using gr::digital::SymbolSync;

    "clock and timing offset"_test = [] {
        using namespace gr::testing;
        struct Link {
            float                samplesPerSymbol;
            double               clockOffset;
            std::complex<double> carrier;
        };
        constexpr std::size_t     kSettledSymbols = 2'000UZ;
        const std::vector<double> symbols         = randomBpskSymbols(6'000UZ);
        for (const Link link : {Link{4.f, 200e-6, std::polar(1e-3, kCarrierPhase)}, Link{4.37f, -300e-6, {1., 0.}}}) {
            const std::vector<std::complex<float>> received = matchedFilterOutput(symbols, static_cast<double>(link.samplesPerSymbol), link.clockOffset, 0.3, link.carrier);

            gr::Graph flow;
            auto&     source = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", static_cast<gr::Size_t>(received.size())}, {"values", received}, {"mark_tag", false}});
            auto&     sync   = flow.emplaceBlock<SymbolSync<float>>({{"samples_per_symbol", link.samplesPerSymbol}});
            auto&     sink   = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), 240'000.f}});
            expect(flow.connect<"out", "in">(source, sync).has_value()) << fatal;
            expect(flow.connect<"out", "in">(sync, sink).has_value()) << fatal;
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value()) << fatal;
            expect(sched.runAndWait().has_value());

            const std::size_t nSymbols = sink._samples.size();
            expect(gt(nSymbols, kSettledSymbols + 1'000UZ)) << fatal << "one output per symbol";
            const auto worstErrorForLag = [&](std::ptrdiff_t lag) {
                double worst = 0.;
                for (std::size_t k = kSettledSymbols; k < nSymbols; ++k) {
                    const std::complex<double> estimate = std::complex<double>(sink._samples[k]) / link.carrier;
                    worst                               = std::max(worst, std::abs(estimate - symbols[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(k) + lag)]));
                }
                return worst;
            };
            const double bestWorstError = std::min({worstErrorForLag(0), worstErrorForLag(1), worstErrorForLag(2)});
            expect(lt(bestWorstError, 0.05)) << "symbols recovered at samples_per_symbol=" << link.samplesPerSymbol;

            expect(carriesSampleRate(sink._tags, 240'000.f / link.samplesPerSymbol)) << "output sample rate is the symbol rate";
            expect(!carriesSampleRate(sink._tags, 240'000.f)) << "input sample rate not passed on";
        }
    };

    "tags and samples_per_symbol change"_test = [] {
        using namespace gr::testing;
        static constexpr std::string_view      kMarkerKey      = "marker";
        constexpr std::size_t                  kMarkerIndex    = 2'000UZ;
        constexpr std::size_t                  kCheckedSymbols = 1'000UZ;
        const std::vector<std::complex<float>> fourPerSymbol   = matchedFilterOutput(randomBpskSymbols(3'000UZ), 4., 0., 0.3, {1., 0.});
        const std::vector<std::complex<float>> eightPerSymbol  = matchedFilterOutput(randomBpskSymbols(6'000UZ), 8., 0., 0.3, {1., 0.});
        std::vector<std::complex<float>>       received        = fourPerSymbol;
        received.insert(received.end(), eightPerSymbol.begin(), eightPerSymbol.end());

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", static_cast<gr::Size_t>(received.size())}, {"values", received}, {"mark_tag", false}});
        auto&     sync   = flow.emplaceBlock<SymbolSync<float>>();
        auto&     sink   = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>();
        source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), 240'000.}});
        source._tags.emplace_back(kMarkerIndex, gr::property_map{{kMarkerKey, true}});
        source._tags.emplace_back(fourPerSymbol.size(), gr::property_map{{"samples_per_symbol", 8.f}});
        expect(flow.connect<"out", "in">(source, sync).has_value()) << fatal;
        expect(flow.connect<"out", "in">(sync, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        const std::vector<std::size_t> markerIndices = sink._tags | std::views::filter([](const auto& receivedTag) { return receivedTag.map.contains(kMarkerKey); }) | std::views::transform([](const auto& receivedTag) { return receivedTag.index; }) | std::ranges::to<std::vector>();
        expect(eq(markerIndices.size(), 1UZ)) << fatal << "each tag passed on once";
        expect(approx(static_cast<double>(markerIndices.front()), static_cast<double>(kMarkerIndex) / 4., 2.)) << "on the symbol at the tag position";
        expect(std::ranges::any_of(sink._tags, [](const auto& receivedTag) { return std::abs(receivedTag.map.template value_or<double>(gr::tag::kSampleRate.key(), 0.) - 60'000.) < 1e-6; })) << "a double sample_rate is scaled too";
        expect(gt(sink._samples.size(), kCheckedSymbols)) << fatal;
        expect(std::ranges::all_of(std::span(sink._samples).last(kCheckedSymbols), [](std::complex<float> symbol) { return std::abs(std::abs(symbol.real()) - 1.f) < 0.05f && std::abs(symbol.imag()) < 0.05f; })) << "symbol instants found again after the change";
    };

    "tag positions"_test = [] {
        using namespace gr::testing;
        static constexpr std::string_view kMarkerKey = "marker";
        struct Placement {
            std::size_t inputIndex;
            std::size_t symbolIndex;
        };
        constexpr std::array kPlacements{Placement{8UZ, 1UZ}, Placement{9UZ, 2UZ}, Placement{100UZ, 24UZ}};

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", gr::Size_t{400U}}, {"values", std::vector{std::complex<float>{1.f, 0.f}}}, {"mark_tag", false}});
        auto&     sync   = flow.emplaceBlock<SymbolSync<float>>();
        auto&     sink   = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>();
        for (const Placement placement : kPlacements) {
            source._tags.emplace_back(placement.inputIndex, gr::property_map{{kMarkerKey, true}});
        }
        expect(flow.connect<"out", "in">(source, sync).has_value()) << fatal;
        expect(flow.connect<"out", "in">(sync, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        const std::vector<std::size_t> markerIndices = sink._tags | std::views::filter([](const auto& receivedTag) { return receivedTag.map.contains(kMarkerKey); }) | std::views::transform([](const auto& receivedTag) { return receivedTag.index; }) | std::ranges::to<std::vector>();
        const std::vector<std::size_t> expected      = kPlacements | std::views::transform(&Placement::symbolIndex) | std::ranges::to<std::vector>();
        expect(markerIndices == expected) << "constant input puts the symbols at 4, 8, 12, ...; each tag on the first symbol at or after it";
    };

    "settings change"_test = [] {
        SymbolSync<float> block;
        startOutsideScheduler(block);
        applySettings(block, {{"samples_per_symbol", 8.f}});
        expect(eq(block.in.min_samples, static_cast<gr::Size_t>(std::ceil(1.5 * 8. * 1.005)) + 4U)) << "input window follows the longest symbol period";
        applySettings(block, {{"loop_bandwidth", 0.1f}, {"rate_tolerance", 0.1f}});
        expect(eq(block.in.min_samples, static_cast<gr::Size_t>(std::ceil(1.5 * 8. * 1.1)) + 4U)) << "largest loop_bandwidth and rate_tolerance allowed";
    };

    "invalid settings"_test = [] {
        for (const float invalid : {1.9f, 1'025.f, kInfinity, kNaN}) {
            expectRejected<SymbolSync<float>>({{"samples_per_symbol", invalid}});
        }
        for (const float invalid : {0.f, -0.1f, 1.5f, kNaN}) {
            expectRejected<SymbolSync<float>>({{"excess_bandwidth", invalid}});
        }
        for (const float invalid : {0.f, -0.01f, 0.2f, kNaN}) {
            expectRejected<SymbolSync<float>>({{"loop_bandwidth", invalid}});
        }
        for (const float invalid : {-0.01f, 0.2f, kInfinity, kNaN}) {
            expectRejected<SymbolSync<float>>({{"rate_tolerance", invalid}});
        }
    };
};

const auto rootRaisedCosineFilterTests = [] {
    using namespace boost::ut;
    using gr::algorithm::filter::rootRaisedCosineTaps;
    using gr::digital::RootRaisedCosineFilter;

    "impulse response"_test = []<typename T> {
        RootRaisedCosineFilter<T> block({{"samples_per_symbol", gr::Size_t{8U}}, {"filter_span", gr::Size_t{6U}}, {"excess_bandwidth", 0.5f}});
        startOutsideScheduler(block);
        expectImpulseResponse(block, rootRaisedCosineTaps<T>(8UZ, 0.5, 6UZ));
    } | kTypes;

    "chunked input"_test = []<typename T> {
        constexpr std::array      kChunkSizes{1UZ, 3UZ, 40UZ, 41UZ, 100UZ};
        const std::vector<T>      taps = rootRaisedCosineTaps<T>(4UZ, static_cast<double>(kDefaultExcessBandwidth), 10UZ);
        RootRaisedCosineFilter<T> block;
        startOutsideScheduler(block);
        std::vector<std::complex<T>> input(std::accumulate(kChunkSizes.begin(), kChunkSizes.end(), 0UZ));
        for (std::size_t n = 0UZ; n < input.size(); ++n) {
            input[n] = phasor<T>(0.3 * static_cast<double>(n));
        }
        std::vector<std::complex<T>> output;
        std::size_t                  first = 0UZ;
        for (const std::size_t chunkSize : kChunkSizes) {
            const std::span<const std::complex<T>> chunk = std::span(input).subspan(first, chunkSize);
            std::ranges::copy(processEach(block, std::vector<std::complex<T>>(chunk.begin(), chunk.end())), std::back_inserter(output));
            first += chunkSize;
        }
        for (std::size_t n = 0UZ; n < output.size(); ++n) {
            std::complex<T> expected{};
            for (std::size_t k = 0UZ; k < std::min(taps.size(), n + 1UZ); ++k) {
                expected += taps[k] * input[n - k];
            }
            expect(approx(distance(output[n], expected), T{0}, kTolerance<T>)) << "n=" << n;
        }
    } | kTypes;

    "matched to modulator"_test = []<typename T> {
        constexpr std::size_t kSamplesPerSymbol       = 4UZ;
        constexpr std::size_t kFilterSpan             = 10UZ;
        constexpr std::size_t kSymbols                = 200UZ;
        constexpr T           kTruncationInterference = T(0.025);
        for (const Constellation constellation : kConstellations) {
            const std::vector<std::complex<double>> points       = expectedPoints(constellation);
            const std::size_t                       bitsInSymbol = gr::digital::bitsPerSymbol(constellation);
            const std::vector<std::uint8_t>         bits         = randomBits(kSymbols * bitsInSymbol);
            const std::vector<std::size_t>          labels       = labelsOf(bits, bitsInSymbol);
            gr::digital::ConstellationModulator<T>  modulator({{"constellation", settingName(constellation)}});
            RootRaisedCosineFilter<T>               matchedFilter;
            startOutsideScheduler(modulator);
            startOutsideScheduler(matchedFilter);
            const std::vector<std::complex<T>> filtered = processEach(matchedFilter, modulate(modulator, bits));
            for (std::size_t n = kFilterSpan; n + kFilterSpan < kSymbols; ++n) {
                expect(approx(distance(filtered[(n + kFilterSpan) * kSamplesPerSymbol], std::complex<T>(points[labels[n]])), T{0}, kTruncationInterference)) << settingName(constellation) << " n=" << n;
            }
        }
    } | kTypes;

    "settings change"_test = []<typename T> {
        RootRaisedCosineFilter<T> block;
        startOutsideScheduler(block);
        applySettings(block, {{"samples_per_symbol", gr::Size_t{8U}}, {"excess_bandwidth", 1.f}});
        expectImpulseResponse(block, rootRaisedCosineTaps<T>(8UZ, 1., 10UZ));
    } | kTypes;

    "reset"_test = []<typename T> {
        RootRaisedCosineFilter<T> block;
        startOutsideScheduler(block);
        std::ignore = processEach(block, std::vector<std::complex<T>>(50UZ, std::complex<T>{T{1}, T{-1}}));
        block.reset();
        expectImpulseResponse(block, rootRaisedCosineTaps<T>(4UZ, static_cast<double>(kDefaultExcessBandwidth), 10UZ));
    } | kTypes;

    "invalid settings"_test = [] {
        for (const gr::Size_t invalid : {gr::Size_t{0U}, gr::Size_t{1U}, gr::Size_t{1'025U}}) {
            expectRejected<RootRaisedCosineFilter<float>>({{"samples_per_symbol", invalid}});
        }
        for (const gr::Size_t invalid : {gr::Size_t{0U}, gr::Size_t{1'025U}}) {
            expectRejected<RootRaisedCosineFilter<float>>({{"filter_span", invalid}});
        }
        for (const float invalid : {0.f, -0.1f, 1.5f, kInfinity, kNaN}) {
            expectRejected<RootRaisedCosineFilter<float>>({{"excess_bandwidth", invalid}});
        }
    };
};

const auto constellationDemapperTests = [] {
    using namespace boost::ut;
    using gr::digital::ConstellationDemapper;

    "hard decision"_test = []<typename T> {
        using Limits = std::numeric_limits<T>;
        struct Decision {
            std::complex<T> symbol;
            std::uint8_t    bit;
        };
        ConstellationDemapper<T> block;
        startOutsideScheduler(block);
        for (const Decision decision : {Decision{{T{1}, T{0}}, 1U}, Decision{{T{-1}, T{0}}, 0U}, Decision{{T(1e-6), T{-5}}, 1U}, Decision{{T(-1e-6), T{5}}, 0U}, Decision{{T{0}, T{1}}, 0U}, Decision{{Limits::quiet_NaN(), T{0}}, 0U}, Decision{{Limits::infinity(), T{0}}, 0U}}) {
            expect(demap(block, {decision.symbol}) == std::vector<std::uint8_t>{decision.bit}) << "symbol " << decision.symbol.real() << " + " << decision.symbol.imag() << "j";
        }
    } | kTypes;

    "mapping"_test = []<typename T> {
        const std::complex<double> distortion = std::polar(0.9, 0.05);
        for (const Constellation constellation : kConstellations) {
            std::vector<std::complex<T>> symbols;
            std::ranges::transform(expectedPoints(constellation), std::back_inserter(symbols), [distortion](std::complex<double> point) { return std::complex<T>(point * distortion); });
            ConstellationDemapper<T> block({{"constellation", settingName(constellation)}});
            startOutsideScheduler(block);
            expect(demap(block, symbols) == allLabelBits(constellation)) << "label bits, most significant first, " << settingName(constellation);
        }
    } | kTypes;

    "settings change"_test = [] {
        ConstellationDemapper<float> block;
        startOutsideScheduler(block);
        applySettings(block, {{"constellation", settingName(Constellation::QAM16)}});
        expect(eq(block.output_chunk_size.value, gr::Size_t{4U})) << "four output bits per symbol";
        expect(demap(block, {std::complex<float>{3.f, 1.f} / std::sqrt(10.f)}) == std::vector<std::uint8_t>{1U, 0U, 1U, 1U}) << "in-phase +3 as 10, quadrature +1 as 11";
    };

    "settings tag"_test = [] {
        using namespace gr::testing;
        constexpr std::size_t                   kSymbols = 1'000UZ;
        const std::vector<std::uint8_t>         bits     = randomBits(2UZ * kSymbols);
        const std::vector<std::complex<double>> points   = expectedPoints(Constellation::QPSK);
        std::vector<std::complex<float>>        symbols;
        std::ranges::transform(labelsOf(bits, 2UZ), std::back_inserter(symbols), [&points](std::size_t label) { return std::complex<float>(points[label]); });

        gr::Graph flow;
        auto&     source   = flow.emplaceBlock<TagSource<std::complex<float>>>({{"n_samples_max", static_cast<gr::Size_t>(kSymbols)}, {"values", symbols}, {"mark_tag", false}});
        auto&     demapper = flow.emplaceBlock<ConstellationDemapper<float>>();
        auto&     sink     = flow.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>();
        source._tags.emplace_back(0UZ, gr::property_map{{"constellation", settingName(Constellation::QPSK)}});
        expect(flow.connect<"out", "in">(source, demapper).has_value()) << fatal;
        expect(flow.connect<"out", "in">(demapper, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());
        expect(sink._samples == bits) << "two bits per symbol from the first symbol on";
    };
};

const auto differentialEncoderTests = [] {
    using namespace boost::ut;
    using gr::digital::DifferentialEncoder;

    "encoding"_test = [] {
        DifferentialEncoder block;
        expect(processEach(block, std::vector<std::uint8_t>{1U, 0U, 1U, 1U, 0U, 0U, 1U, 7U}) == std::vector<std::uint8_t>{1U, 1U, 0U, 1U, 1U, 1U, 0U, 1U}) << "a 1 flips the output, any non-zero input counts as 1";
    };

    "reset"_test = [] {
        DifferentialEncoder block;
        std::ignore = processSample(block, 1U);
        block.reset();
        expect(processSample(block, 0U) == std::uint8_t{0U}) << "output starts from 0";
    };
};

const auto differentialDecoderTests = [] {
    using namespace boost::ut;
    using gr::digital::DifferentialDecoder;

    "decoding"_test = [] {
        DifferentialDecoder block;
        expect(processEach(block, std::vector<std::uint8_t>{1U, 7U, 0U, 1U, 1U, 1U, 0U, 1U}) == std::vector<std::uint8_t>{1U, 0U, 1U, 1U, 0U, 0U, 1U, 1U}) << "1 where the input changes, any non-zero input counts as 1";
    };

    "inverted input"_test = [] {
        const std::vector<std::uint8_t> bits = randomBits(200UZ);
        std::vector<std::uint8_t>       invertedBits(bits.size());
        std::ranges::transform(bits, invertedBits.begin(), [](std::uint8_t bit) { return bit == 0U ? std::uint8_t{1U} : std::uint8_t{0U}; });
        DifferentialDecoder             block;
        DifferentialDecoder             invertedBlock;
        const std::vector<std::uint8_t> decoded         = processEach(block, bits);
        const std::vector<std::uint8_t> decodedInverted = processEach(invertedBlock, invertedBits);
        expect(decoded.front() != decodedInverted.front()) << "first bit differs";
        expect(std::ranges::equal(std::span(decoded).subspan(1UZ), std::span(decodedInverted).subspan(1UZ))) << "all later bits agree";
    };

    "reset"_test = [] {
        DifferentialDecoder block;
        std::ignore = processSample(block, 1U);
        block.reset();
        expect(processSample(block, 1U) == std::uint8_t{1U}) << "previous input is 0";
    };
};

const auto manchesterEncoderTests = [] {
    using namespace boost::ut;
    using gr::digital::ManchesterEncoder;

    "encoding"_test = [] {
        ManchesterEncoder               block;
        const std::vector<std::uint8_t> bits{1U, 0U, 7U};
        std::vector<std::uint8_t>       halfBits(2UZ * bits.size());
        expect(processViews(block, std::span(bits), std::span(halfBits)) == gr::work::Status::OK);
        expect(halfBits == std::vector<std::uint8_t>{1U, 0U, 0U, 1U, 1U, 0U}) << "1 as (1, 0) and 0 as (0, 1), any non-zero input counts as 1";
    };
};

const auto manchesterDecoderTests = [] {
    using namespace boost::ut;
    using gr::digital::ManchesterDecoder;

    "pairing"_test = []<typename T> {
        constexpr std::size_t           kSettledBits = 64UZ;
        const std::vector<std::uint8_t> bits         = randomBits(400UZ);
        std::vector<std::complex<T>>    symbols{std::complex<T>{T(0.3), T{0}}};
        for (const std::uint8_t bit : bits) {
            const T firstHalf = bit == 0U ? T{-1} : T{1};
            symbols.push_back({firstHalf, T{0}});
            symbols.push_back({-firstHalf, T{0}});
        }
        symbols.push_back({});
        ManchesterDecoder<T>      block;
        std::vector<std::uint8_t> decoded(symbols.size() / 2UZ);
        expect(processViews(block, std::span(symbols), std::span(decoded)) == gr::work::Status::OK);
        expect(std::ranges::equal(std::span(decoded).subspan(kSettledBits + 1UZ), std::span(bits).subspan(kSettledBits))) << "finds the pairing that starts one symbol late";
    } | kTypes;
};

const auto rdsDecoderTests = [] {
    using namespace boost::ut;
    using gr::digital::RDSDecoder;

    "station data"_test = [] {
        constexpr std::size_t           kPrefixBits = 13UZ;
        const std::vector<std::uint8_t> cycle       = rdsBits(kProgrammeServiceName, kRadioText, 0U);
        std::vector<std::uint8_t>       bits        = randomBits(kPrefixBits);
        for (std::size_t repeat = 0UZ; repeat < 2UZ; ++repeat) {
            bits.insert(bits.end(), cycle.begin(), cycle.end());
        }
        const std::size_t corruptedBit = kPrefixBits + 4UZ * kRdsGroupBits + 2UZ * gr::digital::rds::kBlockBits + 5UZ;
        bits[corruptedBit]             = static_cast<std::uint8_t>(bits[corruptedBit] ^ 1U);

        const std::vector<gr::property_map> stations = decodeRds(bits);
        expectStation(stations, kRadioText);
        expect(std::ranges::all_of(stations, [](const gr::property_map& station) { return !station.contains(RDSDecoder::kRadioTextKey) || station.value_or<std::string_view>(RDSDecoder::kRadioTextKey, "") == kRadioText; })) << "no radio text from the corrupted block";
    };

    "chance checkword matches"_test = [] {
        using gr::digital::rds::Offset;
        constexpr std::array<std::uint32_t, 4UZ> kGroupWithChanceMatches{0x16DBU, 0x0650U, kNoAlternativeFrequencies, 0x5131U};
        constexpr std::array                     kOffsets{Offset::A, Offset::B, Offset::C, Offset::D};
        std::vector<std::uint8_t>                bits;
        for (std::size_t repeat = 0UZ; repeat < 4UZ; ++repeat) {
            for (std::size_t position = 0UZ; position < kOffsets.size(); ++position) {
                appendBlock(bits, kGroupWithChanceMatches[position], kOffsets[position]);
            }
        }
        const std::vector<gr::property_map> stations = decodeRds(bits);
        expect(!stations.empty()) << fatal << "synchronised although a chance checkword match lies between any two blocks";
        expect(eq(stations.front().value_or<std::uint16_t>(RDSDecoder::kProgrammeIdentificationKey, 0U), static_cast<std::uint16_t>(kGroupWithChanceMatches[0]))) << "PI code from block A";
    };

    "checkword of the standard"_test = [] { expect(eq(gr::digital::rds::cyclicRedundancy(std::uint16_t{0x8000U}), 0b00'0111'0111U)) << "first row of the generator matrix in IEC 62106"; };

    "programme change"_test = [] {
        constexpr std::uint16_t   kNextProgrammeIdentification = 0x5678U;
        std::vector<std::uint8_t> bits                         = rdsBits(kProgrammeServiceName, kRadioText, 0U);
        for (std::uint32_t segment = 0U; segment < 2U; ++segment) {
            appendGroup(bits, 0U, segment, kNoAlternativeFrequencies, characterPair("NEXT STA", 2UZ * segment), kNextProgrammeIdentification);
        }
        const std::vector<gr::property_map> stations = decodeRds(bits);
        expect(!stations.empty()) << fatal;
        expect(stations.back().value_or<std::uint16_t>(RDSDecoder::kProgrammeIdentificationKey, 0U) == kNextProgrammeIdentification) << "PI code of the new station";
        expect(!stations.back().contains(RDSDecoder::kProgrammeServiceNameKey) && !stations.back().contains(RDSDecoder::kRadioTextKey)) << "no PS name or radio text of the previous station";
    };

    "radio text change"_test = [] {
        constexpr std::string_view      kNextRadioText = "Next title";
        std::vector<std::uint8_t>       bits           = rdsBits(kProgrammeServiceName, kRadioText, 0U);
        const std::vector<std::uint8_t> nextBits       = rdsBits(kProgrammeServiceName, kNextRadioText, 1U);
        bits.insert(bits.end(), nextBits.begin(), nextBits.end());
        expectStation(decodeRds(bits), kNextRadioText);
    };
};

const auto digitalTests = [] {
    using namespace boost::ut;
    using namespace gr::digital;

    "differential round trip"_test = [] {
        const std::vector<std::uint8_t> bits = randomBits(1'000UZ);
        DifferentialEncoder             encoder;
        DifferentialDecoder             decoder;
        const std::vector<std::uint8_t> encoded = processEach(encoder, bits);
        expect(encoded == differentiallyEncoded(bits)) << "running XOR";
        expect(processEach(decoder, encoded) == bits) << "decoder inverts encoder";
    };

    "BPSK link"_test = [] {
        using namespace gr::testing;
        using gr::blocks::math::Rotator;
        struct Link {
            float carrierPhase;
            float carrierOffset;
            float receiverSamplesPerSymbol;
        };
        constexpr gr::Size_t            kBits        = 4'000U;
        constexpr std::size_t           kSettledBits = 500UZ;
        constexpr std::size_t           kMaxDelay    = 20UZ;
        const std::vector<std::uint8_t> bits         = randomBits(kBits);
        const std::vector<std::uint8_t> sent         = differentiallyEncoded(bits);
        std::vector<bool>               lockedInverted;
        for (const Link link : {Link{0.7f, 600.f, 4.004f}, Link{2.5f, -600.f, 3.996f}}) {
            gr::Graph flow;
            auto&     source        = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", kBits}, {"values", bits}, {"mark_tag", false}});
            auto&     encoder       = flow.emplaceBlock<DifferentialEncoder>();
            auto&     modulator     = flow.emplaceBlock<ConstellationModulator<float>>();
            auto&     carrier       = flow.emplaceBlock<Rotator<std::complex<float>>>({{"sample_rate", 240'000.f}, {"frequency_shift", link.carrierOffset}, {"initial_phase", link.carrierPhase}});
            auto&     matchedFilter = flow.emplaceBlock<RootRaisedCosineFilter<float>>();
            auto&     sync          = flow.emplaceBlock<SymbolSync<float>>({{"samples_per_symbol", link.receiverSamplesPerSymbol}});
            auto&     costas        = flow.emplaceBlock<CostasLoop<float>>({{"loop_bandwidth", 600.f}, {"min_frequency", -5'000.f}, {"max_frequency", 5'000.f}});
            auto&     demapper      = flow.emplaceBlock<ConstellationDemapper<float>>();
            auto&     decoder       = flow.emplaceBlock<DifferentialDecoder>();
            auto&     symbolSink    = flow.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>();
            auto&     bitSink       = flow.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>();
            source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), 60'000.f}});
            expect(flow.connect<"out", "in">(source, encoder).has_value()) << fatal;
            expect(flow.connect<"out", "in">(encoder, modulator).has_value()) << fatal;
            expect(flow.connect<"out", "in">(modulator, carrier).has_value()) << fatal;
            expect(flow.connect<"out", "in">(carrier, matchedFilter).has_value()) << fatal;
            expect(flow.connect<"out", "in">(matchedFilter, sync).has_value()) << fatal;
            expect(flow.connect<"out", "in">(sync, costas).has_value()) << fatal;
            expect(flow.connect<"out", "in">(costas, symbolSink).has_value()) << fatal;
            expect(flow.connect<"out", "in">(costas, demapper).has_value()) << fatal;
            expect(flow.connect<"out", "in">(demapper, decoder).has_value()) << fatal;
            expect(flow.connect<"out", "in">(decoder, bitSink).has_value()) << fatal;
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value()) << fatal;
            expect(sched.runAndWait().has_value());

            const auto& received = bitSink._samples;
            expect(gt(received.size(), kSettledBits + kMaxDelay + 3'000UZ)) << fatal << "one bit per symbol";
            const auto bitErrors = [&](std::size_t delay) {
                std::size_t errors = 0UZ;
                for (std::size_t k = kSettledBits + delay; k < std::min(received.size(), bits.size() + delay); ++k) {
                    errors += received[k] != bits[k - delay] ? 1UZ : 0UZ;
                }
                return errors;
            };
            const std::size_t delay = std::ranges::min(std::views::iota(0UZ, kMaxDelay), {}, bitErrors);
            expect(eq(bitErrors(delay), 0UZ)) << "no bit errors after the first " << kSettledBits << " bits, carrier phase " << link.carrierPhase;

            const auto& symbols     = symbolSink._samples;
            double      correlation = 0.;
            for (std::size_t k = kSettledBits + delay; k < std::min(symbols.size(), sent.size() + delay); ++k) {
                correlation += static_cast<double>(symbols[k].real()) * bpskSymbol(sent[k - delay]);
            }
            lockedInverted.push_back(correlation < 0.);

            expect(carriesSampleRate(bitSink._tags, 240'000.f / link.receiverSamplesPerSymbol)) << "receiver symbol rate at the end of the chain";
            expect(!carriesSampleRate(bitSink._tags, 240'000.f)) << "no sample rate left from before SymbolSync";
        }
        expect(lockedInverted == std::vector<bool>{false, true}) << "the second link locks with inverted symbols and still decodes";
    };

    "constellation links"_test = [] {
        using namespace gr::testing;
        using gr::blocks::math::Rotator;
        struct Link {
            Constellation constellation;
            float         carrierPhase;
            float         carrierOffset;
        };
        constexpr std::size_t kSymbols        = 4'000UZ;
        constexpr std::size_t kSettledSymbols = 1'000UZ;
        constexpr std::size_t kMaxDelay       = 20UZ;
        constexpr std::size_t kEighthTurns    = 8UZ;
        for (const Link link : {Link{Constellation::QPSK, 0.4f, 300.f}, Link{Constellation::PSK8, 1.f, -200.f}, Link{Constellation::QAM16, 0.3f, 50.f}}) {
            const std::size_t               bitsInSymbol = bitsPerSymbol(link.constellation);
            const float                     bitRate      = 60'000.f * static_cast<float>(bitsInSymbol);
            const std::vector<std::uint8_t> bits         = randomBits(kSymbols * bitsInSymbol);
            const gr::property_map          constellationSetting{{"constellation", settingName(link.constellation)}};
            gr::Graph                       flow;
            auto&                           source        = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", static_cast<gr::Size_t>(bits.size())}, {"values", bits}, {"mark_tag", false}});
            auto&                           modulator     = flow.emplaceBlock<ConstellationModulator<float>>(constellationSetting);
            auto&                           carrier       = flow.emplaceBlock<Rotator<std::complex<float>>>({{"sample_rate", 240'000.f}, {"frequency_shift", link.carrierOffset}, {"initial_phase", link.carrierPhase}});
            auto&                           matchedFilter = flow.emplaceBlock<RootRaisedCosineFilter<float>>();
            auto&                           sync          = flow.emplaceBlock<SymbolSync<float>>();
            auto&                           costas        = flow.emplaceBlock<CostasLoop<float>>({{"constellation", settingName(link.constellation)}, {"loop_bandwidth", 600.f}, {"min_frequency", -5'000.f}, {"max_frequency", 5'000.f}});
            auto&                           demapper      = flow.emplaceBlock<ConstellationDemapper<float>>(constellationSetting);
            auto&                           bitSink       = flow.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>();
            source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), bitRate}});
            expect(flow.connect<"out", "in">(source, modulator).has_value()) << fatal;
            expect(flow.connect<"out", "in">(modulator, carrier).has_value()) << fatal;
            expect(flow.connect<"out", "in">(carrier, matchedFilter).has_value()) << fatal;
            expect(flow.connect<"out", "in">(matchedFilter, sync).has_value()) << fatal;
            expect(flow.connect<"out", "in">(sync, costas).has_value()) << fatal;
            expect(flow.connect<"out", "in">(costas, demapper).has_value()) << fatal;
            expect(flow.connect<"out", "in">(demapper, bitSink).has_value()) << fatal;
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value()) << fatal;
            expect(sched.runAndWait().has_value());

            const std::vector<std::complex<double>> points   = constellationPoints<double>(link.constellation);
            const std::vector<std::size_t>          sent     = labelsOf(bits, bitsInSymbol);
            const std::vector<std::size_t>          received = labelsOf(bitSink._samples, bitsInSymbol);
            expect(gt(received.size(), kSettledSymbols + kMaxDelay + 2'000UZ)) << fatal << "one label per symbol";
            const auto symbolErrors = [&](std::size_t delay, std::size_t eighthTurns) {
                const std::complex<double> rotation = std::polar(1., std::numbers::pi / 4. * static_cast<double>(eighthTurns));
                std::size_t                errors   = 0UZ;
                for (std::size_t k = kSettledSymbols + delay; k < std::min(received.size(), sent.size() + delay); ++k) {
                    errors += received[k] != nearestLabel<double>(points, points[sent[k - delay]] * rotation) ? 1UZ : 0UZ;
                }
                return errors;
            };
            std::size_t fewestErrors = std::numeric_limits<std::size_t>::max();
            for (std::size_t delay = 0UZ; delay < kMaxDelay; ++delay) {
                for (std::size_t eighthTurns = 0UZ; eighthTurns < kEighthTurns; ++eighthTurns) {
                    fewestErrors = std::min(fewestErrors, symbolErrors(delay, eighthTurns));
                }
            }
            expect(eq(fewestErrors, 0UZ)) << "no symbol errors after the first " << kSettledSymbols << " symbols for the best delay and rotation, " << settingName(link.constellation);
            expect(carriesSampleRate(bitSink._tags, bitRate)) << "bit rate at the end of the chain";
        }
    };

    "RDS link"_test = [] {
        using namespace gr::testing;
        using gr::blocks::math::Rotator;
        constexpr gr::Size_t            kSamplesPerHalfBit = 8U;
        const std::vector<std::uint8_t> cycle              = rdsBits(kProgrammeServiceName, kRadioText, 0U);
        std::vector<std::uint8_t>       bits;
        for (std::size_t repeat = 0UZ; repeat < 5UZ; ++repeat) {
            bits.insert(bits.end(), cycle.begin(), cycle.end());
        }

        gr::Graph flow;
        auto&     source              = flow.emplaceBlock<TagSource<std::uint8_t>>({{"n_samples_max", static_cast<gr::Size_t>(bits.size())}, {"values", bits}, {"mark_tag", false}});
        auto&     differentialEncoder = flow.emplaceBlock<DifferentialEncoder>();
        auto&     manchesterEncoder   = flow.emplaceBlock<ManchesterEncoder>();
        auto&     modulator           = flow.emplaceBlock<ConstellationModulator<float>>({{"samples_per_symbol", kSamplesPerHalfBit}, {"excess_bandwidth", 1.f}});
        auto&     carrier             = flow.emplaceBlock<Rotator<std::complex<float>>>({{"sample_rate", 19'000.f}, {"frequency_shift", 10.f}, {"initial_phase", 0.7f}});
        auto&     matchedFilter       = flow.emplaceBlock<RootRaisedCosineFilter<float>>({{"samples_per_symbol", kSamplesPerHalfBit}, {"excess_bandwidth", 1.f}});
        auto&     sync                = flow.emplaceBlock<SymbolSync<float>>({{"samples_per_symbol", 8.008f}, {"excess_bandwidth", 1.f}});
        auto&     costas              = flow.emplaceBlock<CostasLoop<float>>({{"loop_bandwidth", 20.f}, {"min_frequency", -100.f}, {"max_frequency", 100.f}});
        auto&     manchesterDecoder   = flow.emplaceBlock<ManchesterDecoder<float>>();
        auto&     differentialDecoder = flow.emplaceBlock<DifferentialDecoder>();
        auto&     rdsDecoder          = flow.emplaceBlock<RDSDecoder>();
        auto&     sink                = flow.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>();
        auto&     frequencySink       = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>();
        source._tags.emplace_back(0UZ, gr::property_map{{gr::tag::kSampleRate.key(), 1'187.5f}});
        expect(flow.connect<"out", "in">(source, differentialEncoder).has_value()) << fatal;
        expect(flow.connect<"out", "in">(differentialEncoder, manchesterEncoder).has_value()) << fatal;
        expect(flow.connect<"out", "in">(manchesterEncoder, modulator).has_value()) << fatal;
        expect(flow.connect<"out", "in">(modulator, carrier).has_value()) << fatal;
        expect(flow.connect<"out", "in">(carrier, matchedFilter).has_value()) << fatal;
        expect(flow.connect<"out", "in">(matchedFilter, sync).has_value()) << fatal;
        expect(flow.connect<"out", "in">(sync, costas).has_value()) << fatal;
        expect(flow.connect<"out", "in">(costas, manchesterDecoder).has_value()) << fatal;
        expect(flow.connect<"frequency", "in">(costas, frequencySink).has_value()) << fatal;
        expect(flow.connect<"out", "in">(manchesterDecoder, differentialDecoder).has_value()) << fatal;
        expect(flow.connect<"out", "in">(differentialDecoder, rdsDecoder).has_value()) << fatal;
        expect(flow.connect<"out", "in">(rdsDecoder, sink).has_value()) << fatal;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());

        expectStation(stationTags(sink._tags), kRadioText);
        expect(!frequencySink._samples.empty()) << fatal;
        expect(approx(frequencySink._samples.back(), 10.f, 0.1f)) << "Costas loop at the carrier offset";
    };
};

const auto domainTests = [] {
    using namespace boost::ut;
    using gr::digital::ManchesterEncoder;
    using gr::testing::operator""_domain_test;

    "Manchester encoder"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::vector<std::uint8_t> bits    = randomBits(4'096UZ);
        auto&                           source  = gr::testing::sourceOf(fixture, bits);
        auto&                           encoder = fixture.template emplace<ManchesterEncoder>({{"gr:compute_domain", std::string(domain) + "!"}});
        auto&                           sink    = fixture.template emplace<gr::testing::ProbeSink<std::uint8_t>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, encoder).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(encoder, sink).has_value()) << fatal;
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        expect(eq(sink._samples.size(), 2UZ * bits.size())) << fatal;
        const auto isBiphase = [&](std::size_t n) { return sink._samples[2UZ * n] == bits[n] && sink._samples[2UZ * n + 1UZ] == static_cast<std::uint8_t>(bits[n] == 0U); };
        expect(std::ranges::all_of(std::views::iota(0UZ, bits.size()), isBiphase)) << domain << ": bit b gives (b, 1 - b)";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

int main() {
    using namespace boost::ut;
    "Constellation"_test           = constellationTests;
    "CostasLoop"_test              = costasLoopTests;
    "ConstellationModulator"_test  = constellationModulatorTests;
    "SymbolSync"_test              = symbolSyncTests;
    "RootRaisedCosineFilter"_test  = rootRaisedCosineFilterTests;
    "ConstellationDemapper"_test   = constellationDemapperTests;
    "DifferentialEncoder"_test     = differentialEncoderTests;
    "DifferentialDecoder"_test     = differentialDecoderTests;
    "ManchesterEncoder"_test       = manchesterEncoderTests;
    "ManchesterDecoder"_test       = manchesterDecoderTests;
    "RDSDecoder"_test              = rdsDecoderTests;
    "digital"_test                 = digitalTests;
    "digital on every domain"_test = domainTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
