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

// Class that behaves the same as is FeedbackMerge with a Delay was used.
template<std::size_t N>
struct DelayedSumReference : Block<DelayedSumReference<N>> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(DelayedSumReference, in, out);

    std::array<float, N> _buffer{};
    std::size_t          _pos{0};
    float                _state{0};

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        float output  = input + _state;
        float delayed = _buffer[_pos];
        _buffer[_pos] = output;
        _pos          = (_pos + 1) % N;
        _state        = delayed;
        return output;
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

    return actualInputs == expectedInputs && actualOutputs == expectedOutputs;
}

const boost::ut::suite<"FeedbackMerge"> feedbackMergeTests = [] {
    using namespace boost::ut;

    "delayed sum using FeedbackMerge"_test = [] {
        Adder         adder;
        Delay<kDelay> delayed;

        auto delayedSumMerged = feedbackMerge<"out", "out", "in2">(std::move(adder), std::move(delayed));

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s}));

        auto delayedSumReference = DelayedSumReference<kDelay>();

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            outputMerged[i]    = delayedSumMerged.processOne(impulse[i]);
            outputReference[i] = delayedSumReference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("Sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
        }
    };

    "FeedbackMerge zero initialization"_test = [] {
        Adder         adder;
        Delay<kDelay> delayed;

        auto delayedSumMerged = feedbackMerge<"out", "out", "in2">(std::move(adder), std::move(delayed));

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s}));

        for (std::size_t i = 0; i < kDelay + 1; ++i) {
            float output = delayedSumMerged.processOne(5.0f);
            expect(std::abs(output - 5.0f) < 1e-6f) << std::format("Sample {}: {}", i, output);
        }

        // When first feedback sample arrives, the result will be doubled
        float output = delayedSumMerged.processOne(5.0f);
        expect(std::abs(output - 10.0f) < 1e-6f) << std::format("First feedback output: {}", output);
    };

    "FeedbackMerge step response in2"_test = [] {
        Adder         adder{};
        Delay<kDelay> delayed{};

        auto delayedSumMerged = feedbackMerge<"out", "out", "in2">(std::move(adder), std::move(delayed));

        expect(checkBlock(delayedSumMerged, {"in1"s}, {"out"s}));

        // The delay buffer returns the value from 2 cycles ago
        std::array<float, 6> expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> output{};

        for (std::size_t i = 0; i < 6; ++i) {
            output[i] = delayedSumMerged.processOne(1.0f);
        }

        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expected[i]) < 1e-6f) << std::format("Sample {}: output={}, expected={}", i, output[i], expected[i]);
        }
    };

    "FeedbackMerge step response in1"_test = [] {
        Adder         adder;
        Delay<kDelay> delayed;

        auto delayedSumMerged = feedbackMerge<"out", "out", "in1">(std::move(adder), std::move(delayed));

        expect(checkBlock(delayedSumMerged, {"in2"s}, {"out"s}));

        // The delay buffer returns the value from 2 cycles ago
        std::array<float, 6> expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> output{};

        for (std::size_t i = 0; i < 6; ++i) {
            output[i] = delayedSumMerged.processOne(1.0f);
        }

        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expected[i]) < 1e-6f) << std::format("Sample {}: output={}, expected={}", i, output[i], expected[i]);
        }
    };
};

} // namespace gr::test

int main() { /* tests are self-registering */ }
