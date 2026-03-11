#include <boost/ut.hpp>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/algorithm/ImGraph.hpp>
#include <gnuradio-4.0/math/Math.hpp>

#include <array>
#include <format>

using namespace boost::ut;

namespace gr::test {

void printTopology(std::string_view label, gr::Graph& g) { std::println("{}:\n{}", label, gr::graph::draw(g)); }
void printTopology(std::string_view label, gr::Graph&& g) { printTopology(label, g); }

constexpr std::size_t kPulseStart  = 3UZ;
constexpr std::size_t kPulseLength = 5UZ;
constexpr std::size_t kDelay       = 2UZ;
constexpr std::size_t kN           = 20UZ;
constexpr float       kAlpha       = 0.8f;

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

template<float Factor>
struct Scale : Block<Scale<Factor>> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(Scale, in, out);

    [[nodiscard]] constexpr float processOne(float input) const noexcept { return input * Factor; }
};

template<std::size_t N>
struct Delay : Block<Delay<N>> {
    PortIn<float>  in;
    PortOut<float> out;

    std::array<float, N> _buffer{};
    std::size_t          _pos{0};

    GR_MAKE_REFLECTABLE(Delay, in, out);

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        float output  = _buffer[_pos];
        _buffer[_pos] = input;
        _pos          = (_pos + 1) % N;
        return output;
    }
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
template<typename T = float>
struct Amplifier : Block<Amplifier<T>> {
    PortIn<T>  in;
    PortOut<T> out;

    Annotated<T, "gain"> gain = T{1};

    GR_MAKE_REFLECTABLE(Amplifier, in, out, gain);

    bool startCalled = false;
    bool stopCalled  = false;
    bool resetCalled = false;

    void start() { startCalled = true; }
    void stop() { stopCalled = true; }
    void reset() {
        resetCalled = false;
        startCalled = false;
        stopCalled  = false;
    }

    [[nodiscard]] constexpr T processOne(T input) const noexcept { return input * gain; }
};

template<gr::refl::reflectable TBlock>
bool checkPorts(const TBlock&, const std::set<std::string>& expectedInputs, const std::set<std::string>& expectedOutputs) {
    using inputs  = gr::traits::block::stream_input_ports<TBlock>;
    using outputs = gr::traits::block::stream_output_ports<TBlock>;

    std::set<std::string> actualInputs;
    std::set<std::string> actualOutputs;

    inputs::for_each([&]<typename P>(auto, P*) { actualInputs.insert(std::string(typename P::NameT{})); });
    outputs::for_each([&]<typename P>(auto, P*) { actualOutputs.insert(std::string(typename P::NameT{})); });

    if (actualInputs != expectedInputs) {
        std::println("expected inputs {}, got {}", expectedInputs, actualInputs);
    }
    if (actualOutputs != expectedOutputs) {
        std::println("expected outputs {}, got {}", expectedOutputs, actualOutputs);
    }

    return actualInputs == expectedInputs && actualOutputs == expectedOutputs;
}

template<std::size_t N>
struct DelayedSumReference : Block<DelayedSumReference<N>> {
    PortIn<float>  in;
    PortOut<float> out;

    std::array<float, N> _buffer{};
    std::size_t          _pos{0};
    float                _state{0};

    GR_MAKE_REFLECTABLE(DelayedSumReference, in, out);

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        float output  = input + _state;
        float delayed = _buffer[_pos];
        _buffer[_pos] = output;
        _pos          = (_pos + 1) % N;
        _state        = delayed;
        return output;
    }
};

struct IIRLowPassReference {
    float _state{};

    [[nodiscard]] constexpr float processOne(float input) noexcept {
        _state = kAlpha * input + (1.0f - kAlpha) * _state;
        return _state;
    }
};

const boost::ut::suite<"Merge"> mergeTests = [] {
    using namespace boost::ut;

    // ┌──────────┐     ┌──────────┐
    // ─in─>│ Scale*2  ├─out──in1─>│  Adder   ├─out─>
    //      └──────────┘      in2─>│          │
    //                             └──────────┘
    "Merge Scale then Adder"_test = [] {
        using ScaleThenAdd = Merge<Scale<2.0f>, "out", Adder<>, "in1">;
        ScaleThenAdd merged;
        printTopology("Scale->Adder", merged.graph());
        expect(checkPorts(merged, {"in"s, "in2"s}, {"out"s}));

        expect(eq(merged.processOne(3.0f, 10.0f), 16.0f)); // 3*2 + 10
        expect(eq(merged.processOne(0.0f, 5.0f), 5.0f));   // 0*2 + 5
    };

    // ┌──────────┐     ┌──────────┐
    // ─in─>│ Scale*2  ├─out──in─>│ Scale*3  ├─out─>
    //      └──────────┘          └──────────┘
    "Merge chained scales"_test = [] {
        using DoubleScale = Merge<Scale<2.0f>, "out", Scale<3.0f>, "in">;
        DoubleScale merged;
        printTopology("Scale*2->Scale*3", merged.graph());
        expect(checkPorts(merged, {"in"s}, {"out"s}));

        expect(eq(merged.processOne(5.0f), 30.0f)); // 5 * 2 * 3
    };

    "MergeByIndex equivalent"_test = [] {
        using M1 = Merge<Scale<2.0f>, "out", Scale<3.0f>, "in">;
        using M2 = MergeByIndex<Scale<2.0f>, 0, Scale<3.0f>, 0>;
        M1 m1;
        M2 m2;
        printTopology("Merge (by name)", m1.graph());
        printTopology("MergeByIndex (by index)", m2.graph());
        expect(eq(m1.processOne(7.0f), m2.processOne(7.0f)));
    };
};

const boost::ut::suite<"FeedbackMerge"> feedbackMergeTests = [] {
    using namespace boost::ut;

    //             ┌────────────────────────────────────────┐
    //             │              feedback                  │
    //             v                                        │
    //           ┌────┐          ┌────────┐                 │
    // ──── in1─>│ S  ├── out──>│ D(z⁻²) ├── out ──────────┘
    //           └────┘         └────────┘
    //         S = Adder      D = Delay<2>
    "delayed sum vs reference"_test = [] {
        auto merged    = FeedbackMerge<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        auto reference = DelayedSumReference<kDelay>();
        printTopology("FeedbackMerge (delayed sum)", merged.graph());

        expect(checkPorts(merged, {"in1"s}, {"out"s}));

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            outputMerged[i]    = merged.processOne(impulse[i]);
            outputReference[i] = reference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
        }
    };

    "zero initialisation"_test = [] {
        auto merged = FeedbackMerge<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        printTopology("FeedbackMerge (zero init)", merged.graph());

        // feedback buffer is zero-filled, so output == input for the first kDelay+1 samples
        for (std::size_t i = 0; i < kDelay + 1; ++i) {
            float output = merged.processOne(5.0f);
            expect(std::abs(output - 5.0f) < 1e-6f) << std::format("sample {}: {}", i, output);
        }

        // when the first feedback sample arrives, the result doubles
        float output = merged.processOne(5.0f);
        expect(std::abs(output - 10.0f) < 1e-6f) << std::format("first feedback: {}", output);
    };

    // the delay buffer returns the value from kDelay cycles ago
    "step response feedback on in2"_test = [] {
        auto merged = FeedbackMerge<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        printTopology("FeedbackMerge (step in2)", merged.graph());
        expect(checkPorts(merged, {"in1"s}, {"out"s}));

        std::array<float, 6> expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> output{};

        for (std::size_t i = 0; i < 6; ++i) {
            output[i] = merged.processOne(1.0f);
        }
        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expected[i]) < 1e-6f) << std::format("sample {}: got={}, expected={}", i, output[i], expected[i]);
        }
    };

    // same topology but feedback connects to in1 instead of in2
    "step response feedback on in1"_test = [] {
        auto merged = FeedbackMerge<Adder<>, "out", Delay<kDelay>, "out", "in1">();
        printTopology("FeedbackMerge (step in1)", merged.graph());
        expect(checkPorts(merged, {"in2"s}, {"out"s}));

        std::array<float, 6> expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> output{};

        for (std::size_t i = 0; i < 6; ++i) {
            output[i] = merged.processOne(1.0f);
        }
        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expected[i]) < 1e-6f) << std::format("sample {}: got={}, expected={}", i, output[i], expected[i]);
        }
    };

    //                                    ┌──────────────────────────────┐
    //                                    │          feedback              │
    //                                    v                               │
    //          ┌──────────┐          ┌───────┐          ┌──────────────┐  │
    // in ──>───┤ Scale(α) ├──>──in1─┤ Adder ├──>──out──┤ Scale(1 - α) ├──┘
    //          └──────────┘         └───────┘          └──────────────┘
    //   y[n] = α·x[n] + (1-α)·y[n-1]
    "IIR low-pass filter from reference proof-of-concept"_test = [] {
        using IIRMerged = Merge<Scale<kAlpha>, "out", FeedbackMerge<Adder<>, "out", Scale<1.0f - kAlpha>, "out", "in2">, "in1">;

        IIRMerged           merged;
        IIRLowPassReference reference;
        printTopology("IIR low-pass (Merge+FeedbackMerge)", merged.graph());

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            outputMerged[i]    = merged.processOne(impulse[i]);
            outputReference[i] = reference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
        }
    };

    "IIR low-pass using MultiplyConst with runtime settings"_test = [] {
        using namespace gr::blocks::math;
        using IIRMergedRT = Merge<MultiplyConst<float>, "out", FeedbackMerge<Adder<>, "out", MultiplyConst<float>, "out", "in2">, "in1">;

        IIRMergedRT merged;
        merged._leftBlock.value            = kAlpha;        // Scale(α)
        merged._rightBlock._feedback.value = 1.0f - kAlpha; // Scale(1-α) in feedback path
        printTopology("IIR low-pass (MultiplyConst runtime)", merged.graph());

        IIRLowPassReference reference;

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            outputMerged[i]    = merged.processOne(impulse[i]);
            outputReference[i] = reference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
        }
    };
};

const boost::ut::suite<"FeedbackMergeWithTap"> feedbackMergeWithTapTests = [] {
    using namespace boost::ut;

    //             ┌─────────────────────────────────────────────┐
    //             │             feedback                        │
    //             v                                             │
    //           ┌────┐         ┌────────┐                       │
    // ──── in1─>│ S  ├── out──>│ D(z⁻²) ├── out ────────────────┘
    //           └────┘    │    └────────┘
    //                     └──────────────────────────> splitOut
    //         S = Adder      D = Delay<2>
    "delayed sum with tap vs reference"_test = [] {
        auto merged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        printTopology("FeedbackMergeWithTap (delayed sum)", merged.graph());
        expect(checkPorts(merged, {"in1"s}, {"out"s, "splitOut"s}));

        DelayedSumReference<kDelay> reference;

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outMerged{};
        std::array<float, kN> outRef{};

        for (std::size_t i = 0; i < kN; ++i) {
            auto [o, _s] = merged.processOne(impulse[i]);
            outMerged[i] = o;
            outRef[i]    = reference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outMerged[i] - outRef[i]) < 1e-6f) << std::format("sample {}: merged={}, reference={}", i, outMerged[i], outRef[i]);
        }
    };

    "zero initialisation"_test = [] {
        auto merged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        printTopology("FeedbackMergeWithTap (zero init)", merged.graph());

        // feedback buffer is zero-filled; splitOut taps the feedback path before it re-enters the adder
        for (std::size_t i = 0; i < kDelay + 1; ++i) {
            auto [output, split] = merged.processOne(5.0f);
            expect(std::abs(output - 5.0f) < 1e-6f) << std::format("sample {}: {}", i, output);
            expect(std::abs(split - (i < kDelay ? 0.0f : 5.0f)) < 1e-6f) << std::format("split {}: {}", i, split);
        }

        // when the first feedback sample arrives, the result doubles
        auto [output, split] = merged.processOne(5.0f);
        expect(std::abs(output - 10.0f) < 1e-6f) << std::format("first feedback: {}", output);
    };

    // the delay buffer returns the value from kDelay cycles ago
    "step response feedback on in2"_test = [] {
        auto merged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in2">();
        printTopology("FeedbackMergeWithTap (step in2)", merged.graph());
        expect(checkPorts(merged, {"in1"s}, {"out"s, "splitOut"s}));

        std::array<float, 6> expectedOutput = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> expectedSplit  = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f};
        std::array<float, 6> output{};
        std::array<float, 6> split{};

        for (std::size_t i = 0; i < 6; ++i) {
            std::tie(output[i], split[i]) = merged.processOne(1.0f);
        }
        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expectedOutput[i]) < 1e-6f) << std::format("sample {}: got={}, expected={}", i, output[i], expectedOutput[i]);
            expect(std::abs(split[i] - expectedSplit[i]) < 1e-6f) << std::format("split {}: got={}, expected={}", i, split[i], expectedSplit[i]);
        }
    };

    // same topology but feedback connects to in1 instead of in2
    "step response feedback on in1"_test = [] {
        auto merged = FeedbackMergeWithTap<Adder<>, "out", Delay<kDelay>, "out", "in1">();
        printTopology("FeedbackMergeWithTap (step in1)", merged.graph());
        expect(checkPorts(merged, {"in2"s}, {"out"s, "splitOut"s}));

        std::array<float, 6> expectedOutput = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
        std::array<float, 6> expectedSplit  = {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f};
        std::array<float, 6> output{};
        std::array<float, 6> split{};

        for (std::size_t i = 0; i < 6; ++i) {
            std::tie(output[i], split[i]) = merged.processOne(1.0f);
        }
        for (std::size_t i = 0; i < 6; ++i) {
            expect(std::abs(output[i] - expectedOutput[i]) < 1e-6f) << std::format("sample {}: got={}, expected={}", i, output[i], expectedOutput[i]);
            expect(std::abs(split[i] - expectedSplit[i]) < 1e-6f) << std::format("split {}: got={}, expected={}", i, split[i], expectedSplit[i]);
        }
    };

    //                    forward path
    //           ┌────┐      ┌──────────────────────────────> monitor (M)
    //           │    │      │      ┌────┐      ┌────┐
    // ──── in1─>│ S  ├─out──┴──in─>│ D  ├─out─>│ G  ├──┬──> output
    //           └────┘             └────┘      └────┘  │
    //             ^                                    │
    //          in2│            feedback path            │
    //             └────────────────────────────────────┘
    //   S = Adder, D = Delay<2>, G = ValueCollector, M = ValueCollector (tap monitor)
    "composed topology with Monitor and Merge"_test = [] {
        using D = Delay<2>;
        using G = ValueCollector<>;
        using M = ValueCollector<>;
        using S = Adder<>;

        using DthenG   = Merge<D, "out", G, "in">;
        using Composed = FeedbackMergeWithTap<S, "out", DthenG, "out", "in2", M>;

        Composed composed;
        printTopology("FeedbackMergeWithTap (composed with Monitor)", composed.graph());

        std::vector<float> inputSamples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outputs;
        std::vector<float> monitorOutputs;

        for (std::size_t i = 0UZ; i < inputSamples.size(); i++) {
            auto [output, mon] = composed.processOne(inputSamples[i]);
            outputs.push_back(output);
            monitorOutputs.push_back(mon);
        }

        auto approxEqual = [](const std::vector<float>& a, const std::vector<float>& b) {
            if (a.size() != b.size()) {
                return false;
            }
            for (std::size_t i = 0UZ; i < a.size(); i++) {
                if (std::abs(a[i] - b[i]) > 1e-6f) {
                    return false;
                }
            }
            return true;
        };

        expect(approxEqual(outputs, {1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f}));
        expect(approxEqual(monitorOutputs, {0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 5.0f}));
        expect(approxEqual(composed._monitor.collected, outputs));
        expect(approxEqual(composed._feedback._rightBlock.collected, monitorOutputs));
    };
};

const boost::ut::suite<"MergeByIndex lifecycle and settings"> mergeLifecycleTests = [] {
    using namespace boost::ut;
    using namespace gr::lifecycle;

    "stateChanged propagates to inner blocks"_test = [] {
        using Merged = MergeByIndex<Amplifier<float>, 0, Amplifier<float>, 0>;
        Merged merged;
        printTopology("Amplifier->Amplifier", merged.graph());

        expect(merged.state() == State::IDLE);
        expect(merged._leftBlock.state() == State::IDLE);
        expect(merged._rightBlock.state() == State::IDLE);

        expect(merged.changeStateTo(State::INITIALISED).has_value());
        expect(merged.state() == State::INITIALISED);
        expect(merged._leftBlock.state() == State::INITIALISED);
        expect(merged._rightBlock.state() == State::INITIALISED);

        expect(merged.changeStateTo(State::RUNNING).has_value());
        expect(merged.state() == State::RUNNING);
        expect(merged._leftBlock.state() == State::RUNNING);
        expect(merged._rightBlock.state() == State::RUNNING);
        expect(merged._leftBlock.startCalled);
        expect(merged._rightBlock.startCalled);

        expect(merged.changeStateTo(State::REQUESTED_STOP).has_value());
        expect(merged._leftBlock.state() == State::REQUESTED_STOP);
        expect(merged._rightBlock.state() == State::REQUESTED_STOP);
        expect(merged._leftBlock.stopCalled);
        expect(merged._rightBlock.stopCalled);

        expect(merged.changeStateTo(State::STOPPED).has_value());
        expect(merged._leftBlock.state() == State::STOPPED);
        expect(merged._rightBlock.state() == State::STOPPED);
    };

    "stateChanged propagates through nested MergeByIndex"_test = [] {
        using Inner = MergeByIndex<Amplifier<float>, 0, Amplifier<float>, 0>;
        using Outer = MergeByIndex<Inner, 0, Amplifier<float>, 0>;
        Outer merged;
        printTopology("nested Amplifier->Amplifier->Amplifier", merged.graph());

        expect(merged.changeStateTo(State::INITIALISED).has_value());
        expect(merged.changeStateTo(State::RUNNING).has_value());

        expect(merged._leftBlock._leftBlock.state() == State::RUNNING);
        expect(merged._leftBlock._rightBlock.state() == State::RUNNING);
        expect(merged._rightBlock.state() == State::RUNNING);
        expect(merged._leftBlock._leftBlock.startCalled);
        expect(merged._leftBlock._rightBlock.startCalled);
        expect(merged._rightBlock.startCalled);
    };

    "nested settings via leftBlock/rightBlock"_test = [] {
        using Merged = MergeByIndex<Amplifier<float>, 0, Amplifier<float>, 0>;

        gr::property_map init = {{"leftBlock", gr::property_map{{"gain", 2.0f}}}, {"rightBlock", gr::property_map{{"gain", 0.5f}}}};
        Merged           merged(init);
        printTopology("Amplifier(2)->Amplifier(0.5)", merged.graph());

        expect(eq(static_cast<float>(merged._leftBlock.gain), 2.0f)) << "left gain set via nested map";
        expect(eq(static_cast<float>(merged._rightBlock.gain), 0.5f)) << "right gain set via nested map";
        expect(eq(merged.processOne(10.0f), 10.0f)) << "10 * 2.0 * 0.5 = 10";
    };

    "flat settings apply to both blocks"_test = [] {
        using Merged = MergeByIndex<Amplifier<float>, 0, Amplifier<float>, 0>;

        gr::property_map init = {{"gain", 3.0f}};
        Merged           merged(init);
        printTopology("Amplifier(3)->Amplifier(3)", merged.graph());

        expect(eq(static_cast<float>(merged._leftBlock.gain), 3.0f)) << "left gain set via flat key";
        expect(eq(static_cast<float>(merged._rightBlock.gain), 3.0f)) << "right gain set via flat key";
        expect(eq(merged.processOne(2.0f), 18.0f)) << "2 * 3 * 3 = 18";
    };

    "nested settings override flat settings"_test = [] {
        using Merged = MergeByIndex<Amplifier<float>, 0, Amplifier<float>, 0>;

        gr::property_map init = {{"gain", 3.0f}, {"leftBlock", gr::property_map{{"gain", 5.0f}}}};
        Merged           merged(init);
        printTopology("Amplifier(5)->Amplifier(3)", merged.graph());

        expect(eq(static_cast<float>(merged._leftBlock.gain), 5.0f)) << "leftBlock overrides flat gain";
        expect(eq(static_cast<float>(merged._rightBlock.gain), 3.0f)) << "right uses flat gain";
        expect(eq(merged.processOne(2.0f), 30.0f)) << "2 * 5 * 3 = 30";
    };
};

const boost::ut::suite<"SplitMergeCombine"> splitMergeCombineTests = [] {
    using namespace boost::ut;

    //               ┌──────────────┐
    //         ┌─in─>│ Scale(2)     ├─out──┐
    //         │     └──────────────┘      │
    // ─ in ──>┤                          (+)──> out
    //         │     ┌──────────────┐      │
    //         └─in─>│ Scale(3)     ├─out──┘
    //               └──────────────┘
    //   out = 2x + 3x = 5x
    "basic fan-out to two scales summed"_test = [] {
        using FanOut = SplitMergeCombine<Scale<2.0f>, Scale<3.0f>>;
        FanOut merged;
        expect(checkPorts(merged, {"in"s}, {"out"s}));

        expect(eq(merged.processOne(4.0f), 20.0f)); // 4*2 + 4*3 = 20
        expect(eq(merged.processOne(0.0f), 0.0f));
        expect(eq(merged.processOne(-1.0f), -5.0f));
    };

    "three paths"_test = [] {
        using FanOut3 = SplitMergeCombine<Scale<1.0f>, Scale<2.0f>, Scale<3.0f>>;
        FanOut3 merged;
        expect(eq(merged.processOne(1.0f), 6.0f)); // 1 + 2 + 3
    };

    "SplitMergeCombine composed with Merge"_test = [] {
        using Inner = SplitMergeCombine<Scale<2.0f>, Scale<3.0f>>;
        using Chain = Merge<Scale<10.0f>, "out", Inner, "in">;
        Chain merged;
        printTopology("Scale*10 -> SplitMergeCombine {Scale*2, Scale*3}", merged.graph());
        expect(eq(merged.processOne(1.0f), 50.0f)); // 1*10 -> {20, 30} -> 50
    };

    //   y[n] = α·x[n] + (1-α)·y[n-1]
    //   feedback path decomposes as: y + (-α)·y = (1-α)·y via SplitMergeCombine
    "IIR low-pass decomposed with SplitMergeCombine in feedback path"_test = [] {
        using FeedbackFanOut = SplitMergeCombine<Scale<1.0f>, Scale<-kAlpha>>;
        using IIRDecomposed  = Merge<Scale<kAlpha>, "out", FeedbackMerge<Adder<>, "out", FeedbackFanOut, "out", "in2">, "in1">;

        IIRDecomposed merged;
        printTopology("IIR low-pass (SplitMergeCombine decomposed)", merged.graph());
        IIRLowPassReference reference;

        constexpr auto        impulse = genImpulse();
        std::array<float, kN> outputMerged{};
        std::array<float, kN> outputReference{};

        for (std::size_t i = 0; i < kN; ++i) {
            outputMerged[i]    = merged.processOne(impulse[i]);
            outputReference[i] = reference.processOne(impulse[i]);
        }

        for (std::size_t i = 0; i < kN; ++i) {
            expect(std::abs(outputMerged[i] - outputReference[i]) < 1e-6f) << std::format("sample {}: merged={}, reference={}", i, outputMerged[i], outputReference[i]);
        }
    };

    "stateChanged propagates to all path blocks"_test = [] {
        using Merged = SplitMergeCombine<Amplifier<float>, Amplifier<float>>;
        Merged merged;
        printTopology("SplitMergeCombine {Amplifier, Amplifier}", merged.graph());

        expect(merged.changeStateTo(lifecycle::State::INITIALISED).has_value());
        expect(merged.changeStateTo(lifecycle::State::RUNNING).has_value());

        expect(merged.path<0>().startCalled);
        expect(merged.path<1>().startCalled);
        expect(merged.path<0>().state() == lifecycle::State::RUNNING);
        expect(merged.path<1>().state() == lifecycle::State::RUNNING);
    };

    "nested settings via path keys"_test = [] {
        using Merged = SplitMergeCombine<Amplifier<float>, Amplifier<float>>;

        gr::property_map init = {{"path0", gr::property_map{{"gain", 3.0f}}}, {"path1", gr::property_map{{"gain", 4.0f}}}};
        Merged           merged(init);
        printTopology("SplitMergeCombine {Amplifier(3), Amplifier(4)}", merged.graph());

        expect(eq(static_cast<float>(merged.path<0>().gain), 3.0f));
        expect(eq(static_cast<float>(merged.path<1>().gain), 4.0f));
        // input -> {path0(*3), path1(*4)} -> sum = 3x + 4x = 7x
        expect(eq(merged.processOne(1.0f), 7.0f));
    };

    "flat settings apply to all paths"_test = [] {
        using Merged = SplitMergeCombine<Amplifier<float>, Amplifier<float>>;

        gr::property_map init = {{"gain", 5.0f}};
        Merged           merged(init);
        printTopology("SplitMergeCombine {Amplifier(5), Amplifier(5)}", merged.graph());

        expect(eq(static_cast<float>(merged.path<0>().gain), 5.0f));
        expect(eq(static_cast<float>(merged.path<1>().gain), 5.0f));
        // input -> {path0(*5), path1(*5)} -> sum = 5x + 5x = 10x
        expect(eq(merged.processOne(2.0f), 20.0f));
    };

    "OutputSigns negates second path"_test = [] {
        using Merged = SplitMergeCombine<OutputSigns<+1.0f, -1.0f>, Scale<2.0f>, Scale<3.0f>>;
        Merged merged;
        printTopology("SplitMergeCombine OutputSigns(+1,-1) {Scale*2, Scale*3}", merged.graph());
        // input -> {+1 * scale(2), -1 * scale(3)} -> 2x - 3x = -x
        expect(eq(merged.processOne(10.0f), -10.0f));
    };

    "OutputSigns fewer signs than paths default to +1"_test = [] {
        using Merged = SplitMergeCombine<OutputSigns<-1.0f>, Scale<2.0f>, Scale<3.0f>>;
        Merged merged;
        printTopology("SplitMergeCombine OutputSigns(-1) {Scale*2, Scale*3}", merged.graph());
        // input -> {-1 * scale(2), +1 * scale(3)} -> -2x + 3x = x
        expect(eq(merged.processOne(10.0f), 10.0f));
    };

    "OutputSigns all positive is same as no signs"_test = [] {
        using WithSigns    = SplitMergeCombine<OutputSigns<+1.0f, +1.0f>, Scale<2.0f>, Scale<3.0f>>;
        using WithoutSigns = SplitMergeCombine<Scale<2.0f>, Scale<3.0f>>;
        WithSigns    a;
        WithoutSigns b;
        printTopology("SplitMergeCombine OutputSigns(+1,+1) {Scale*2, Scale*3}", a.graph());
        printTopology("SplitMergeCombine (no signs) {Scale*2, Scale*3}", b.graph());
        expect(eq(a.processOne(7.0f), b.processOne(7.0f)));
    };
};

const boost::ut::suite<"Merge graph() topology"> graphTests = [] {
    using namespace gr;
    using namespace gr::test;

    "MergeByIndex graph shows 2 blocks and 1 edge"_test = [] {
        auto  merged = MergeByIndex<Scale<2.0f>, 0, Scale<3.0f>, 0>();
        Graph g      = merged.graph();
        expect(eq(g.blocks().size(), 2UZ)) << "should have 2 leaf blocks";
        expect(eq(g.edges().size(), 1UZ)) << "should have 1 internal edge";
        printTopology("MergeByIndex topology", g);
    };

    "nested MergeByIndex graph is flattened"_test = [] {
        using Inner  = MergeByIndex<Scale<2.0f>, 0, Scale<3.0f>, 0>;
        auto  merged = MergeByIndex<Inner, 0, Scale<4.0f>, 0>();
        Graph g      = merged.graph();
        expect(eq(g.blocks().size(), 3UZ)) << "should have 3 leaf blocks (flattened)";
        expect(eq(g.edges().size(), 2UZ)) << "should have 2 edges";
        printTopology("nested MergeByIndex topology", g);
    };

    "SplitMergeCombine graph shows FanOut, paths, and Sum"_test = [] {
        auto  merged = SplitMergeCombine<Scale<2.0f>, Scale<3.0f>>();
        Graph g      = merged.graph();
        expect(eq(g.blocks().size(), 4UZ)) << "should have 4 blocks (FanOut + 2 paths + Sum)";
        expect(eq(g.edges().size(), 4UZ)) << "should have 4 edges (FanOut→path0, FanOut→path1, path0→Sum, path1→Sum)";
        printTopology("SplitMergeCombine topology", g);
    };

    "FeedbackMerge graph shows forward and feedback blocks"_test = [] {
        auto  merged = FeedbackMergeByIndex<Adder<float>, 0, Scale<kAlpha>, 0, 1, void>();
        Graph g      = merged.graph();
        expect(eq(g.blocks().size(), 2UZ)) << "should have 2 blocks (forward + feedback)";
        expect(eq(g.edges().size(), 2UZ)) << "should have 2 edges (forward→feedback, feedback→forward)";
        printTopology("FeedbackMerge topology", g);
    };

    "FeedbackMergeWithTap graph shows forward, monitor, and feedback"_test = [] {
        auto  merged = FeedbackMergeWithTapByIndex<Adder<float>, 0, Scale<kAlpha>, 0, 1, ValueCollector<float>>();
        Graph g      = merged.graph();
        expect(eq(g.blocks().size(), 3UZ)) << "should have 3 blocks";
        expect(eq(g.edges().size(), 3UZ)) << "should have 3 edges";
        printTopology("FeedbackMergeWithTap topology", g);
    };
};

} // namespace gr::test

int main() { /* tests are self-registering */ }
