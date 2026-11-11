#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/device/DeviceRelocatable.hpp>
#include <gnuradio-4.0/math/Math.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>

namespace {
using Byte    = std::uint8_t;
using C       = std::complex<float>;
using DeviceC = gr::complex<float>;

template<typename TBlock>
constexpr bool kRunsOnDevice = gr::device::DeviceRelocatable<TBlock> && gr::AutoParallelisable<TBlock>;
static_assert(kRunsOnDevice<gr::blocks::math::Not<Byte>>);
static_assert(kRunsOnDevice<gr::blocks::math::ConstOp<Byte, std::bit_and<Byte>>>);
static_assert(kRunsOnDevice<gr::blocks::math::PairOp<DeviceC, gr::blocks::math::MultipliesConjugate<DeviceC>>>);
static_assert(kRunsOnDevice<gr::blocks::math::Power<DeviceC>>);
static_assert(kRunsOnDevice<gr::blocks::math::Log10<float>>);
static_assert(kRunsOnDevice<gr::blocks::math::Transcendental<float>>);

template<typename TBlock, typename T>
std::vector<T> foldedBy(const std::vector<std::vector<T>>& inputs, std::vector<std::string> inputPorts = {}) {
    gr::property_map settings;
    if (inputPorts.empty()) {
        settings.insert_or_assign("n_inputs", static_cast<gr::Size_t>(inputs.size()));
        inputPorts = gr::testing::indexedPorts("in", inputs.size());
    }
    return gr::testing::ProbedPorts<TBlock, T>(inputs, std::move(settings), inputPorts).run()[0];
}

const auto foldForms = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;

    "And, Or and Xor fold the bits of every input"_test = []<typename T>(const T&) {
        const std::vector<std::vector<T>> inputs{{T(0b1100), T(0b1111)}, {T(0b1010), T(0b0001)}};
        expect(foldedBy<ReduceOp<T, std::bit_and<T>>>(inputs) == std::vector<T>{T(0b1000), T(0b0001)});
        expect(foldedBy<ReduceOp<T, std::bit_or<T>>>(inputs) == std::vector<T>{T(0b1110), T(0b1111)});
        expect(foldedBy<ReduceOp<T, std::bit_xor<T>>>(inputs) == std::vector<T>{T(0b0110), T(0b1110)});
    } | std::tuple<std::uint8_t, std::int64_t>{};

    "Max and Min take the extreme over every input (GR3: Max, Min)"_test = []<typename T>(const T&) {
        const std::vector<std::vector<T>> inputs{{T(4), T(-1), T(3)}, {T(7), T(-5), T(3)}, {T(0), T(2), T(3)}};
        expect(foldedBy<ReduceOp<T, Maximum<T>>>(inputs) == std::vector<T>{T(7), T(2), T(3)});
        expect(foldedBy<ReduceOp<T, Minimum<T>>>(inputs) == std::vector<T>{T(0), T(-5), T(3)});
    } | std::tuple<std::int16_t, float>{};

    "every ConstOp form applies value to each lane of a SIMD vector"_test = [] {
        const auto laneOf = []<typename TOp, typename T>(std::type_identity<TOp>, T sample, T value) {
            ConstOp<T, TOp> block(property_map{{"value", value}});
            block.init(block.progress);
            const vir::stdx::native_simd<T> lanes = block.processOne(vir::stdx::native_simd<T>(sample));
            return std::ranges::all_of(std::views::iota(0UZ, lanes.size()), [&](std::size_t lane) { return lanes[lane] == lanes[0]; }) ? T(lanes[0]) : T{};
        };
        expect(eq(laneOf(std::type_identity<std::plus<float>>{}, 3.f, 2.f), 5.f));
        expect(eq(laneOf(std::type_identity<std::minus<float>>{}, 3.f, 2.f), 1.f));
        expect(eq(laneOf(std::type_identity<std::multiplies<float>>{}, 3.f, 2.f), 6.f));
        expect(eq(laneOf(std::type_identity<std::divides<float>>{}, 3.f, 2.f), 1.5f));
        expect(eq(laneOf(std::type_identity<std::bit_and<std::int32_t>>{}, 0b1100, 0b1010), 0b1000));
    };

    "AndConst masks each sample with value (GR3: And Const)"_test = [] {
        gr::testing::ProbedChain<ConstOp<Byte, std::bit_and<Byte>>, Byte> chain({0x3C, 0xFF, 0x00}, {{"value", Byte{0x0F}}});
        expect(chain.run() == std::vector<Byte>{0x0C, 0x0F, 0x00});
    };

    "MultiplyConjugate multiplies in0 by the conjugate of in1: (1 + 2i)(3 - 4i) = 11 + 2i"_test = [] {
        expect(PairOp<std::complex<double>, MultipliesConjugate<std::complex<double>>>().processOne({1., 2.}, {3., 4.}) == std::complex<double>(11., 2.));
        expect(foldedBy<PairOp<C, MultipliesConjugate<C>>, C>({{C(1.f, 2.f), C(0.f, 1.f)}, {C(3.f, 4.f), C(0.f, 1.f)}}, {"in0", "in1"}) == std::vector<C>{C(11.f, 2.f), C(1.f, 0.f)});
    };

    "a tag carrying value changes the factor of MultiplyConst from its sample on, unless value was set (GR3: Multiply by Tag Value)"_test = [] {
        gr::testing::ProbedChain<MultiplyConst<C>, C> chain({C(1.f, 0.f), C(1.f, 0.f), C(2.f, 0.f), C(1.f, 0.f)}, {});
        chain.source->_tags = {{2UZ, {{"value", C(0.f, 1.f)}}}};
        expect(chain.run() == std::vector<C>{C(1.f, 0.f), C(1.f, 0.f), C(0.f, 2.f), C(0.f, 1.f)});

        gr::testing::ProbedChain<MultiplyConst<C>, C> pinned({C(1.f, 0.f), C(1.f, 0.f)}, {{"value", C(2.f, 0.f)}});
        pinned.source->_tags = {{1UZ, {{"value", C(0.f, 1.f)}}}};
        expect(pinned.run() == std::vector<C>{C(2.f, 0.f), C(2.f, 0.f)});
    };
};

const auto elementWise = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;

    "Not inverts every bit (GR3: Not)"_test = [] {
        expect(eq(Not<std::int32_t>().processOne(0), -1));
        gr::testing::ProbedChain<Not<Byte>, Byte> chain({0x0F, 0xAA}, {});
        expect(chain.run() == std::vector<Byte>{0xF0, 0x55});
    };

    "Power raises (1 + i) to an integer exponent: 0 → 1, 2 → 2i, 3 → -2 + 2i, -2 → -i/2"_test = []<typename T>(const T&) {
        const auto powerOf = [](std::int32_t exponent) {
            Power<T> block(property_map{{"exponent", exponent}});
            block.init(block.progress);
            return block.processOne(T(1, 1));
        };
        expect(powerOf(0) == T(1, 0));
        expect(powerOf(1) == T(1, 1));
        expect(powerOf(2) == T(0, 2));
        expect(powerOf(3) == T(-2, 2));
        expect(powerOf(-2) == T(0, -0.5));
    } | std::tuple<std::complex<float>, std::complex<double>, gr::complex<float>>{};

    "Log10 gives scale · log10(x) + offset: 10 · log10(100) + 3 = 23"_test = []<typename T>(const T&) {
        Log10<T> block(property_map{{"scale", T(10)}, {"offset", T(3)}});
        block.init(block.progress);
        expect(approx(block.processOne(T(100)), T(23), T(1e-5)));
        expect(approx(block.processOne(T(1)), T(3), T(1e-6)));
        const T floored = T(10) * T(std::numeric_limits<T>::min_exponent - 1) * std::numbers::log10e_v<T> * std::numbers::ln2_v<T> + T(3);
        expect(approx(block.processOne(T(0)), floored, T(1e-3))) << "zero is floored at the smallest normal number, 2^(min_exponent - 1)";
        expect(approx(block.processOne(T(-1)), floored, T(1e-3))) << "a negative sample likewise";
    } | std::tuple<float, double>{};

    "Transcendental evaluates the function it is named"_test = [] {
        constexpr double                                                kPi = std::numbers::pi;
        const std::vector<std::tuple<std::string_view, double, double>> cases{{"sin", 0., 0.}, {"cos", 0., 1.}, {"tan", 0., 0.}, {"sinh", 0., 0.}, {"cosh", 0., 1.}, {"tanh", 0., 0.}, {"exp", 1., std::numbers::e}, {"log", std::numbers::e, 1.}, {"log10", 100., 2.}, {"sqrt", 4., 2.}, {"asin", 1., kPi / 2.}, {"acos", 1., 0.}, {"atan", 1., kPi / 4.}};
        for (const auto& [name, x, expected] : cases) {
            Transcendental<double> block(property_map{{"function", std::string(name)}});
            block.init(block.progress);
            expect(approx(block.processOne(x), expected, 1e-12)) << name;
        }
    };

    "the complex square root of -4 is 2i"_test = [] {
        Transcendental<std::complex<double>> block(property_map{{"function", "sqrt"}});
        block.init(block.progress);
        expect(std::abs(block.processOne({-4., 0.}) - std::complex<double>(0., 2.)) < 1e-12);
    };

    "an unknown function name is refused and the previous function kept"_test = [] {
        Transcendental<float> block(property_map{{"function", "cos"}});
        block.init(block.progress);
        std::ignore = block.settings().set({{"function", "sine"}});
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(block.function.value == TranscendentalFunction::cos);
    };
};

const auto magnitudes = [] {
    using namespace boost::ut;
    using gr::blocks::math::MagnitudeNormalise;
    "magnitude normalisation keeps the phase on the unit circle; eps shrinks it, 0 stays 0"_test = []<typename T>() {
        static_assert(gr::HasConstProcessOneFunction<MagnitudeNormalise<std::complex<T>>>);

        MagnitudeNormalise<std::complex<T>> toUnit;
        toUnit.eps = T(0);
        for (const std::complex<T> z : {std::complex<T>{3, 4}, std::complex<T>{T(-1), T(0.5)}, std::complex<T>{0, -7}, std::complex<T>{T(1e-3), T(1e-3)}}) {
            const std::complex<T> unit = toUnit.processOne(z);
            expect(approx(std::abs(unit), T(1), T(1e-6))) << "unit magnitude";
            expect(approx(std::arg(unit), std::arg(z), T(1e-6))) << "the phase it came with";
        }
        expect(eq(toUnit.processOne(std::complex<T>{3, 4}), std::complex<T>{T(0.6), T(0.8)})) << "3+4j over 5";
        expect(eq(toUnit.processOne(std::complex<T>{}), std::complex<T>{})) << "0 stays 0, not NaN";
        toUnit.eps = T(1);
        expect(approx(std::abs(toUnit.processOne(std::complex<T>{0, 1})), T(0.5), T(1e-6))) << "|j| / (1 + 1)";
    } | std::tuple<float, double>{};

    "the device-safe gr::complex normalises the same way"_test = [] {
        static_assert(gr::HasConstProcessOneFunction<MagnitudeNormalise<gr::complex<float>>>);
        MagnitudeNormalise<gr::complex<float>> toUnit;
        toUnit.eps                    = 0.f;
        const gr::complex<float> unit = toUnit.processOne(gr::complex<float>{3.f, 4.f});
        expect(approx(unit.real(), 0.6f, 1e-6f) && approx(unit.imag(), 0.8f, 1e-6f)) << "3+4j over 5";
    };
};

const auto domains = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;
    using gr::testing::operator""_domain_test;

    "the element-wise blocks give the same samples on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr std::size_t kSamples = 1024UZ;
        const std::string     placed   = std::string(domain) + "!";
        const auto            chain    = [&]<typename TBlock, typename T>(std::type_identity<TBlock>, const std::vector<T>& values, gr::property_map settings) -> gr::testing::ProbeSink<T>& {
            settings.insert_or_assign("gr:compute_domain", placed);
            auto& source = gr::testing::sourceOf(fixture, values);
            auto& block  = fixture.template emplace<TBlock>(std::move(settings));
            auto& sink   = fixture.template emplace<gr::testing::ProbeSink<T>>({{"log_samples", true}});
            expect(fixture.template connect<"out", "in">(source, block).has_value()) << fatal;
            expect(fixture.template connect<"out", "in">(block, sink).has_value()) << fatal;
            return sink;
        };

        const auto bytes    = std::views::iota(0UZ, kSamples) | std::views::transform([](std::size_t i) { return static_cast<Byte>(i * 37UZ); }) | std::ranges::to<std::vector>();
        const auto positive = std::views::iota(0UZ, kSamples) | std::views::transform([](std::size_t i) { return 0.01f + static_cast<float>(i); }) | std::ranges::to<std::vector>();
        const auto spiral   = std::views::iota(0UZ, kSamples) | std::views::transform([](std::size_t i) { return DeviceC(std::polar(1.f + 0.001f * static_cast<float>(i), 0.01f * static_cast<float>(i))); }) | std::ranges::to<std::vector>();

        auto& inverted   = chain(std::type_identity<Not<Byte>>{}, bytes, {});
        auto& masked     = chain(std::type_identity<ConstOp<Byte, std::bit_and<Byte>>>{}, bytes, {{"value", Byte{0x0F}}});
        auto& cubed      = chain(std::type_identity<Power<DeviceC>>{}, spiral, {{"exponent", std::int32_t{3}}});
        auto& decibels   = chain(std::type_identity<Log10<float>>{}, positive, {{"scale", 10.f}});
        auto& cosines    = chain(std::type_identity<Transcendental<float>>{}, positive, {{"function", "cos"}});
        auto& conjugated = fixture.template emplace<PairOp<DeviceC, MultipliesConjugate<DeviceC>>>({{"gr:compute_domain", placed}});
        auto& product    = fixture.template emplace<gr::testing::ProbeSink<DeviceC>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in0">(gr::testing::sourceOf(fixture, spiral), conjugated).has_value()) << fatal;
        expect(fixture.template connect<"out", "in1">(gr::testing::sourceOf(fixture, spiral), conjugated).has_value()) << fatal;
        expect(fixture.template connect<"out", "in">(conjugated, product).has_value()) << fatal;
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        const std::array lengths{inverted._samples.size(), masked._samples.size(), cubed._samples.size(), decibels._samples.size(), cosines._samples.size(), product._samples.size()};
        expect(fatal(std::ranges::all_of(lengths, [](std::size_t length) { return length == kSamples; }))) << domain << ": every chain passes all samples";
        const auto worst = [&](auto error) { return std::ranges::max(std::views::iota(0UZ, kSamples) | std::views::transform(error)); };
        expect(std::ranges::equal(inverted._samples, bytes | std::views::transform([](Byte b) { return static_cast<Byte>(~b); }))) << domain << ": Not is ~x";
        expect(std::ranges::equal(masked._samples, bytes | std::views::transform([](Byte b) { return static_cast<Byte>(b & 0x0F); }))) << domain << ": AndConst is x & 0x0F";
        expect(lt(worst([&](std::size_t i) { return gr::abs(cubed._samples[i] - spiral[i] * spiral[i] * spiral[i]) / gr::norm(spiral[i]); }), 1e-5f)) << domain << ": Power 3 is z · z · z";
        expect(lt(worst([&](std::size_t i) { return gr::abs(product._samples[i] - DeviceC(gr::norm(spiral[i]), 0.f)) / gr::norm(spiral[i]); }), 1e-5f)) << domain << ": z · conj(z) is |z|²";
        expect(lt(worst([&](std::size_t i) { return std::abs(decibels._samples[i] - 10.f * std::log10(positive[i])); }), 1e-4f)) << domain << ": Log10 scale 10 is 10 log10(x)";
        expect(lt(worst([&](std::size_t i) { return std::abs(cosines._samples[i] - std::cos(positive[i])); }), 1e-5f)) << domain << ": Transcendental cos is cos(x)";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
const auto dataSets = [] {
    using namespace boost::ut;
    using gr::blocks::math::DataSetBinaryOp;
    using gr::dataset::MathOp;
    const auto twoSignals = [](std::vector<float> values) {
        gr::DataSet<float> dataSet;
        dataSet.extents      = {3};
        dataSet.axis_names   = {"time"};
        dataSet.axis_units   = {"s"};
        dataSet.axis_values  = {{0.f, 0.5f, 1.f}};
        dataSet.signal_names = {"a", "b"};
        dataSet.signal_units = {"V", "V"};
        dataSet.signal_values.assign(values.begin(), values.end());
        return dataSet;
    };

    "DIVIDE by 0 is NaN, as gr::dataset::mathFunction has it"_test = [&] {
        DataSetBinaryOp<float> block;
        block.op = MathOp::DIVIDE;
        expect(std::isnan(block.processOne(twoSignals({1.f, 1.f, 1.f, 1.f, 1.f, 1.f}), twoSignals({0.f, 1.f, 1.f, 1.f, 1.f, 1.f})).signal_values[0]));
    };

    "two DataSets combine element by element, keeping in0's axes and units, each signal's range its own"_test = [&] {
        const gr::DataSet<float> x = twoSignals({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
        const gr::DataSet<float> y = twoSignals({2.f, 2.f, 2.f, 1.f, 2.f, 4.f});
        struct Case {
            MathOp             op;
            std::vector<float> expected;
        };
        for (const auto& [op, expected] : {Case{MathOp::ADD, {3.f, 4.f, 5.f, 5.f, 7.f, 10.f}}, Case{MathOp::SUBTRACT, {-1.f, 0.f, 1.f, 3.f, 3.f, 2.f}}, //
                 Case{MathOp::MULTIPLY, {2.f, 4.f, 6.f, 4.f, 10.f, 24.f}}, Case{MathOp::DIVIDE, {0.5f, 1.f, 1.5f, 4.f, 2.5f, 1.5f}}}) {
            DataSetBinaryOp<float> block;
            block.op                        = op;
            const gr::DataSet<float> result = block.processOne(x, y);
            const std::string        name{gr::meta::enumName(op).value_or("")};
            expect(std::ranges::equal(result.signal_values, expected)) << name;
            expect(result.axis_values == x.axis_values && result.signal_units == x.signal_units) << name << "axes and units from in0";
            expect(fatal(eq(result.signal_ranges.size(), 2UZ))) << name;
            expect(eq(result.signal_ranges[1].min, std::ranges::min(std::span(expected).subspan(3UZ))) && eq(result.signal_ranges[1].max, std::ranges::max(std::span(expected).subspan(3UZ)))) << name << "signal b's range";
        }
    };

    "DataSets of other extents stop the stream and pass in0 on"_test = [&] {
        gr::DataSet<float> shorter = twoSignals({1.f, 2.f, 3.f, 4.f});
        shorter.extents            = {2};
        DataSetBinaryOp<float> block;
        block.init(block.progress);
        expect(block.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;
        const gr::DataSet<float> result = block.processOne(twoSignals({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}), shorter);
        expect(eq(result.signal_values.size(), 6UZ)) << "in0 unchanged";
        expect(block.state() == gr::lifecycle::State::REQUESTED_STOP) << "the stream is asked to stop";
    };
};
} // namespace

int main() {
    using namespace boost::ut;
    "magnitude normalisation"_test              = magnitudes;
    "bitwise, extreme and conjugate forms"_test = foldForms;
    "element-wise functions"_test               = elementWise;
    "element-wise blocks on every domain"_test  = domains;
    "element-wise DataSets"_test                = dataSets;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
