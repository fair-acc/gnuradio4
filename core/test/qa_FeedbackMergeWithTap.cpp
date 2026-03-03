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

template<typename T = float>
struct Adder : Block<Adder<T>> {
    PortIn<T>  in1;
    PortIn<T>  in2;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Adder, in1, in2, out);

    [[nodiscard]] constexpr T processOne(T a, T b) const noexcept { return a + b; }
};

template<typename T = float>
struct ValueCollector : Block<ValueCollector<T>> {
    PortIn<T>      in;
    PortOut<T>     out;
    std::vector<T> collected;

    GR_MAKE_REFLECTABLE(ValueCollector, in, out);

    [[nodiscard]] constexpr T processOne(T a) noexcept {
        collected.push_back(a);
        return a;
    }
};

template<std::size_t N, typename T = float>
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

// Class that behaves the same as is FeedbackMergeWithTap with a Delay was used.
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

const boost::ut::suite<"FeedbackMergeWithTap"> feedbackMergeTests = [] {
    using namespace boost::ut;

    "delayed sum using FeedbackMergeWithTap"_test = [] {
        auto delayedSumMerged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();

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

    "FeedbackMergeWithTap zero initialization"_test = [] {
        auto delayedSumMerged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();

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

    "FeedbackMergeWithTap step response in2"_test = [] {
        auto delayedSumMerged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();

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

    "FeedbackMergeWithTap step response in1"_test = [] {
        auto delayedSumMerged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in1">();

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

    "Composed splitter topology"_test = [] {
        //           Forward
        //           adder       ┌──────────────────────────────> monitor
        //           ┌────┐      │      ┌────┐      ┌────┐
        // ─────in1─>│ S  ├─out─-┴──in─>│ D  ├─out─>│ G  ├─────┬──> output
        //           └────┘             └────┘      └────┘     │
        //             ^                                       │
        //          in2│                                       │
        //             │               ┌──────────┐            │
        //             └───────────────┤    M     │<───────────┘
        //                             └──────────┘
        using D = Delay<2>;
        using G = ValueCollector<>;
        using M = ValueCollector<>;
        using S = Adder<>;

        using DthenG   = Merge<D, "out", G, "in">;
        using Composed = FeedbackMergeWithTap<
            /*Forward*/ S, "out",        //
            /* Feedback*/ DthenG, "out", //
            "in2",                       //
            M>;

        Composed composed;

        std::vector<float> inputSamples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outputs;
        std::vector<float> monitorOutputs;

        for (std::size_t i = 0UZ; i < inputSamples.size(); i++) {
            auto [output, monitor] = composed.processOne(inputSamples[i]);
            outputs.push_back(output);
            monitorOutputs.push_back(monitor);
        }

        auto checkEqual = [](const auto& expected, const auto& got, std::source_location location = std::source_location::current()) {
            if (expected.size() != got.size()) {
                std::println("Different sizes {} {} and {} {}, caller at {}:{}", expected, expected.size(), got, got.size(), location.file_name(), location.line());
                return false;
            }

            for (std::size_t i = 0UZ; i < expected.size(); i++) {
                if (std::abs(expected[i] - got[i]) > 1e-6f) {
                    std::println("Values differ at index {} -- {} and {}, caller at {}:{}", i, expected, got, location.file_name(), location.line());
                    return false;
                }
            }

            return true;
        };

        std::println("outputs are {}", outputs);
        expect(checkEqual(outputs, std::vector<float>{1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f}));
        std::println("monitor outputs are {}", monitorOutputs);
        expect(checkEqual(monitorOutputs, std::vector<float>{0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 5.0f}));

        // M block gets the same data that are sent to the output
        // before passing it on to the S block
        std::println("monitor collected are {}", composed.monitor.collected);
        expect(checkEqual(composed.monitor.collected, outputs));

        // G block gets the delayed data
        std::println("monitor feedback collected are {}", composed.feedback.right.collected);
        expect(checkEqual(composed.feedback.right.collected, monitorOutputs));
    };

    // Fails to compile (as expected):
    // FeedbackMergeWithTap<Adder<int>, "out", Delay<kDelay, float>, "out", "in1">{};
};

} // namespace gr::test

int main() { /* tests are self-registering */ }
