#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>

#include <array>
#include <format>

using namespace boost::ut;

namespace gr::test {

constexpr std::size_t kPulseStart  = 3UZ;
constexpr std::size_t kPulseLength = 5UZ;
constexpr std::size_t kDelay       = 2UZ;
constexpr std::size_t kN           = 20UZ;

[[nodiscard]] consteval std::array<float, kN> genImpulse() noexcept {
    std::array<float, kN> impulse{};
    std::ranges::fill(impulse | std::views::drop(kPulseStart) | std::views::take(kPulseLength), 1.0f);
    return impulse;
}

struct Adder : Block<Adder> {
    PortIn<float>  in1;
    PortIn<float>  in2;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(Adder, in1, in2, out);

    [[nodiscard]] constexpr float processOne(float a, float b) const noexcept { return a + b; }
};

template<std::size_t N>
struct Delay : Block<Delay<N>> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(Delay, in, out);

    std::array<float, N> _buffer{};
    std::size_t          _pos{0};

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        float output  = _buffer[_pos];
        _buffer[_pos] = input;
        _pos          = (_pos + 1) % N;
        return output;
    }
};

// Class that behaves the same as is SplitMerge with a Delay was used.
template<std::size_t N>
struct DelayedSumReference {
    std::array<float, N> _buffer{};
    std::size_t          _pos{0};
    float                _state{0};

    [[nodiscard]] constexpr std::tuple<float, float> processOne(float input) noexcept {
        float output  = input + _state;
        float delayed = _buffer[_pos];
        _buffer[_pos] = output;
        _pos          = (_pos + 1) % N;
        _state        = delayed;
        return {output, _state};
    }
};

template<gr::refl::reflectable TBlock>
bool checkBlock(const TBlock&, const std::set<std::string>& expectedInputs, const std::set<std::string>& expectedOutputs) {
    using inputs  = gr::traits::block::stream_input_ports<TBlock>;
    using outputs = gr::traits::block::stream_output_ports<TBlock>;

    std::set<std::string> actualInputs;
    std::set<std::string> actualOutputs;

    inputs::for_each([&]<typename P>(auto, P*) {
        using NameT = typename P::NameT;
        actualInputs.insert(std::string(NameT{}));
    });
    outputs::for_each([&]<typename P>(auto, P*) {
        using NameT = typename P::NameT;
        actualOutputs.insert(std::string(NameT{}));
    });

    if (actualInputs != expectedInputs) {
        std::println("Expected inputs {}", expectedInputs);
        std::println("            got {}", actualInputs);
    }

    if (actualOutputs != expectedOutputs) {
        std::println("Expected outputs {}", expectedOutputs);
        std::println("             got {}", actualOutputs);
    }

    return actualInputs == expectedInputs && actualOutputs == expectedOutputs;
}

const boost::ut::suite<"SplitMerge"> feedbackMergeTests = [] {
    using namespace boost::ut;

    "delayed sum using SplitMerge"_test = [] {
        auto delayedSumMerged = SplitMerge<Adder, "out", Delay<kDelay>, "out", "in2">();

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s, "splitOut"s}));

        auto delayedSumReference = DelayedSumReference<kDelay>();

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};
        std::array<float, kN> splitMerged{};
        std::array<float, kN> splitReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            std::tie(outputMerged[i], splitMerged[i])       = delayedSumMerged.processOne(impulse[i]);
            std::tie(outputReference[i], splitReference[i]) = delayedSumReference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("Sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
            expect(std::abs(splitMerged[i] - splitReference[i]) < 1e-6f) << std::format("Sample {}: merged={}, reference={}", i, splitMerged[i], splitReference[i]);
        }
    };

    "SplitMerge zero initialization"_test = [] {
        auto delayedSumMerged = SplitMerge<Adder, "out", Delay<kDelay>, "out", "in2">();

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s, "splitOut"s}));

        for (std::size_t i = 0; i < kDelay + 1; ++i) {
            auto [output, split] = delayedSumMerged.processOne(5.0f);
            expect(std::abs(output - 5.0f) < 1e-6f) << std::format("Sample {}: {}", i, output);
            expect(std::abs(split - (i < kDelay ? 0.0f : 5.0f)) < 1e-6f) << std::format("Sample {}: {}", i, output);
        }

        // When first feedback sample arrives, the result will be doubled
        auto [output, split] = delayedSumMerged.processOne(5.0f);
        expect(std::abs(output - 10.0f) < 1e-6f) << std::format("First feedback output: {}", output);
    };

    "SplitMerge step response in2"_test = [] {
        auto delayedSumMerged = SplitMerge<Adder, "out", Delay<kDelay>, "out", "in2">();

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s, "splitOut"s}));

        // The delay buffer returns the value from 2 cycles ago
        std::array<float, 6> expectedOutput = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> expectedSplit  = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f};
        std::array<float, 6> output{};
        std::array<float, 6> split{};

        for (std::size_t i = 0; i < 6; ++i) {
            std::tie(output[i], split[i]) = delayedSumMerged.processOne(1.0f);
        }

        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expectedOutput[i]) < 1e-6f) << std::format("Sample {}: output={}, expectedOutput={}", i, output[i], expectedOutput[i]);
            expect(std::abs(split[i] - expectedSplit[i]) < 1e-6f) << std::format("Sample {}: output={}, expectedSplit={}", i, split[i], expectedSplit[i]);
        }
    };

    "SplitMerge step response in1"_test = [] {
        auto delayedSumMerged = SplitMerge<Adder, "out", Delay<kDelay>, "out", "in1">();

        expect(checkBlock(delayedSumMerged, {"in2"s}, {"out"s, "splitOut"s}));

        // The delay buffer returns the value from 2 cycles ago
        std::array<float, 6> expectedOutput = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> expectedSplit  = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f};
        std::array<float, 6> output{};
        std::array<float, 6> split{};

        for (std::size_t i = 0; i < 6; ++i) {
            std::tie(output[i], split[i]) = delayedSumMerged.processOne(1.0f);
        }

        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expectedOutput[i]) < 1e-6f) << std::format("Sample {}: output={}, expectedOutput={}", i, output[i], expectedOutput[i]);
            expect(std::abs(split[i] - expectedSplit[i]) < 1e-6f) << std::format("Sample {}: output={}, expectedSplit={}", i, split[i], expectedSplit[i]);
        }
    };
};

} // namespace gr::test

int main() { /* tests are self-registering */ }
