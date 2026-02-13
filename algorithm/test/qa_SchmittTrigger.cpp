#include <boost/ut.hpp>

#include <format>
#include <vector>

#include "gnuradio-4.0/Tag.hpp"
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

using namespace boost::ut;

namespace {
using namespace gr::trigger;

template<typename T>
gr::Tensor<T> convert_signal(const gr::Tensor<double>& input) {
    using value_t = gr::meta::fundamental_base_value_type_t<T>;
    gr::Tensor<T> signal;
    signal.reserve(input.size());
    for (auto v : input) {
        if constexpr (std::is_arithmetic_v<T>) {
            signal.emplace_back(static_cast<T>(static_cast<value_t>(v)));
        } else {
            signal.emplace_back(gr::UncertainValue<value_t>(static_cast<value_t>(v), static_cast<value_t>(0.01))); // add some uncertainty
        }
    }
    return signal;
}

template<typename T, gr::fixed_string testCaseName, typename Trigger, typename EdgeType = gr::UncertainValue<float>> // always float precision similar to `sample_rate` definition
void test_schmitt_trigger_with_signal(Trigger& trigger, const gr::Tensor<T>& signal, const std::vector<std::tuple<EdgeDetection, EdgeType>>& expected_edges) {
    using enum gr::trigger::EdgeDetection;
    using value_t                  = gr::meta::fundamental_base_value_type_t<T>;
    const std::string fullTestName = std::format("{} - data: {}({})", testCaseName.c_str(), gr::meta::type_name<T>(), gr::join(signal));

    std::println("test: {}", fullTestName);
    std::vector<std::tuple<EdgeDetection, EdgeType>> detected_edges;
    std::ptrdiff_t                                   dataIndex = 0;
    for (const auto& sample : signal) {
        if (trigger.processOne(sample) != NONE) {
            EdgeType sample_index = EdgeType{static_cast<float>(dataIndex + trigger.lastEdgeIdx)} + trigger.lastEdgeOffset;
            detected_edges.emplace_back(trigger.lastEdge, sample_index);
            std::println("   {:7} edge detected at sample index: {:.4f} (Idx: {}, Offset: {:.3f})", //
                magic_enum::enum_name(trigger.lastEdge), sample_index, trigger.lastEdgeIdx, trigger.lastEdgeOffset);
        }
        dataIndex = dataIndex + 1;
    }

    expect(eq(detected_edges.size(), expected_edges.size())) << std::format("{}: n detected ({}) vs. expected ({}) edges does not match", fullTestName, detected_edges.size(), expected_edges.size());

    for (std::size_t i = 0; i < expected_edges.size(); ++i) {
        auto [expected_edge, expected_index] = expected_edges[i];
        auto [detected_edge, detected_index] = detected_edges[i];

        expect(detected_edge == expected_edge) << std::format("{}: detected {} edge type does not match expected {} at index {}\n", //
            fullTestName, magic_enum::enum_name(detected_edge), magic_enum::enum_name(expected_edge), i);
        expect(approx(gr::value(detected_index), gr::value(expected_index), std::is_floating_point_v<value_t> ? static_cast<value_t>(0.1f) : 0.1f)) //
            << std::format("{}: detected edge ({}): is {} vs. expected {}\n", fullTestName, i, gr::value(detected_index), gr::value(expected_index));
    }
}
} // namespace

const suite<"SchmittTrigger"> SchmittTriggerTests = [] {
    using namespace gr::trigger;
    using enum gr::trigger::InterpolationMethod;
    using enum gr::trigger::EdgeDetection;

    // Test suite for different types
    "SchmittTrigger w/ and w/o error propagation"_test = []<typename T>() {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;

        "no interpolation"_test = [] {
            SchmittTrigger<T> trigger(value_t(0.1f) /* threshold */, value_t(0.5f) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (no interpolation)">(trigger,                                                                //
                convert_signal<T>({0.3, 0.4, 0.45, 0.5, 0.55, 0.6 /* >= offset + threshold (RISING) */, 1.0, 1.0, 0.0 /* <= offset - threshold (FALLING) */}), //
                {{RISING, 5}, {FALLING, 8}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "slow falling edge (no interpolation)">(trigger,                                            //
                convert_signal<T>({/* RISING */ 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4 /* FALLING */, 0.35, 0.3}), //
                {{RISING, 0}, {FALLING, 11}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (no interpolation)">(trigger,                                                            //
                convert_signal<T>({0.0, 0.8 /* RISING */, 1.2, 0.9, 0.4 /* FALLING */, -0.2, -1.1, -0.5, 0.0, 1.1 /* RISING */, 1.1, 1.0, 0.0 /* FALLING */, 0.0}), //
                {{RISING, 1}, {FALLING, 4}, {RISING, 9}, {FALLING, 12}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "Dirac delta">(trigger,        //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}), //
                {{RISING, 1}, {FALLING, 2}});
        };

        "basic linear interpolation"_test = [] {
            SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> trigger(value_t(0.1f) /* threshold */, value_t(0.5f) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (basic)">(trigger,                                                             //
                convert_signal<T>({0.3, 0.4, 0.45, 0.5 /* >= offset (int. RISING) */, 0.55, 0.6, 1.0, 1.0 /* <= offset (int. FALLING) */, 0.0}), //
                {{RISING, 3}, {FALLING, 7.5f}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "slow falling edge (basic)">(trigger,                                                       //
                convert_signal<T>({/* RISING */ 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5 /* FALLING */, 0.45, 0.4, 0.35, 0.3}), //
                {{RISING, -0.5}, {FALLING, 9.0}});

            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (basic)">(trigger,                                                                       //
                convert_signal<T>({0.0 /* RISING */, 0.8, 1.2, 0.9 /* FALLING */, 0.4, -0.2, -1.1, -0.5, 0.0 /* RISING */, 1.1, 1.1, 1.0 /* FALLING */, 0.0, 0.0}), //
                {{RISING, 0.625}, {FALLING, 3.8}, {RISING, 8.45455}, {FALLING, 11.5}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "Dirac delta">(trigger,        //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}), //
                {{RISING, 0.5}, {FALLING, 1.5}});
        };

        "linear interpolation (large window)"_test = [] {
            SchmittTrigger<T, LINEAR_INTERPOLATION, 12> trigger(value_t(0.1f) /* threshold */, value_t(0.5f) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (large window)">(trigger,                    //
                convert_signal<T>({0.3, 0.4, 0.45, 0.5 /* RISING */, 0.55, 0.6, 1.0, 1.0 /* FALLING */, 0.0}), //
                {{RISING, 3}, {FALLING, 7.5f}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "slow falling edge (large window)">(trigger,                                                //
                convert_signal<T>({/* RISING */ 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5 /* FALLING */, 0.45, 0.4, 0.35, 0.3}), //
                {{RISING, -0.5}, {FALLING, 9.0}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges">(trigger,                                                                               //
                convert_signal<T>({0.0 /* RISING */, 0.8, 1.2, 0.9 /* FALLING */, 0.4, -0.2, -1.1, -0.5, 0.0 /* RISING */, 1.1, 1.1, 1.0 /* FALLING */, 0.0, 0.0}), //
                {{RISING, 0.625}, {FALLING, 3.8}, {RISING, 8.45455}, {FALLING, 11.5}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "Dirac delta">(trigger,        //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}), //
                {{RISING, 0.5}, {FALLING, 1.5}});
        };

        "linear interpolation (short window)"_test = [] {
            SchmittTrigger<T, LINEAR_INTERPOLATION, 2> trigger(value_t(0.1f) /* threshold */, value_t(0.5f) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (short window)">(trigger,                    //
                convert_signal<T>({0.3, 0.4, 0.45, 0.5 /* RISING */, 0.55, 0.6, 1.0, 1.0 /* FALLING */, 0.0}), //
                {{RISING, 3}, {FALLING, 7.5f}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "slow falling edge (short window)">(trigger,                                                //
                convert_signal<T>({/* RISING */ 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5 /* FALLING */, 0.45, 0.4, 0.35, 0.3}), //
                {{RISING, -0.5}, {FALLING, 9.0}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (short window)">(trigger,                                                                //
                convert_signal<T>({0.0 /* RISING */, 0.8, 1.2, 0.9 /* FALLING */, 0.4, -0.2, -1.1, -0.5, 0.0 /* RISING */, 1.1, 1.1, 1.0 /* FALLING */, 0.0, 0.0}), //
                {{RISING, 0.625}, {FALLING, 3.8}, {RISING, 8.45455}, {FALLING, 11.5}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "Dirac delta">(trigger,        //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}), //
                {{RISING, 0.5}, {FALLING, 1.5}});
        };

        "polynomial interpolation"_test = [] {
            SchmittTrigger<T, POLYNOMIAL_INTERPOLATION, 12> trigger(value_t(0.1f) /* threshold */, value_t(0.5f) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (polynomial)">(trigger,                      //
                convert_signal<T>({0.3, 0.4, 0.45, 0.5 /* RISING */, 0.55, 0.6, 1.0, 1.0 /* FALLING */, 0.0}), //
                {{RISING, 3}, {FALLING, 7.5f}});

            trigger.reset();
            // N.B. SG cubic fit on linear ramp has small boundary effects → crossing at ~8.79 vs. 9.0 for linear regression
            test_schmitt_trigger_with_signal<T, "slow falling edge (polynomial)">(trigger,                                                  //
                convert_signal<T>({/* RISING */ 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5 /* FALLING */, 0.45, 0.4, 0.35, 0.3}), //
                {{RISING, -0.5}, {FALLING, 8.79f}});

            trigger.reset();
            // N.B. last falling edge differs from LINEAR (9.97 vs 11.5) because SG cubic fit captures the non-linear 1.1→1.0→0 transition shape
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (polynomial)">(trigger,                                                                  //
                convert_signal<T>({0.0 /* RISING */, 0.8, 1.2, 0.9 /* FALLING */, 0.4, -0.2, -1.1, -0.5, 0.0 /* RISING */, 1.1, 1.1, 1.0 /* FALLING */, 0.0, 0.0}), //
                {{RISING, 0.625}, {FALLING, 3.8}, {RISING, 8.45455}, {FALLING, 9.97f}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "Dirac delta (polynomial)">(trigger, //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}),       //
                {{RISING, 0.5}, {FALLING, 1.5}});

            // Verify reset() clears _lastState and _historyBuffer:
            // end in high state (_lastState = true), reset, then feed a signal starting above threshold.
            // Without proper reset, the trigger would still think it's high and miss the rising edge.
            trigger.reset();
            test_schmitt_trigger_with_signal<T, "signal ending high (polynomial)">(trigger, //
                convert_signal<T>({0.0, 1.0 /* RISING */, 1.0}),                            //
                {{RISING, 0.5}});
            trigger.reset();
            test_schmitt_trigger_with_signal<T, "reset after high state (polynomial)">(trigger, //
                convert_signal<T>({0.0, 1.0 /* RISING */, 0.0 /* FALLING */}),                  //
                {{RISING, 0.5}, {FALLING, 1.5}});
        };
    } | std::tuple<float, gr::UncertainValue<float>>{};

    "SchmittTrigger (unsigned) integer values"_test = []<typename T>() {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;

        "integer-type: no interpolation"_test = [] {
            SchmittTrigger<T> trigger(value_t(1) /* threshold */, value_t(5) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (no interpolation)">(trigger,                                                                         //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0, 1, 5 /* RISING */, 7 /* >= offset + threshold */, 7, 7, 8, 0 /* <= offset - threshold */, 0})), //
                {{RISING, 3}, {FALLING, 7}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (no interpolation)">(trigger,                     //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0, 10 /* RISING */, 0 /* FALLING */, 7 /* RISING */})), //
                {{RISING, 1}, {FALLING, 2}, {RISING, 3}});
        };

        "integer-type: basic interpolation"_test = [] {
            SchmittTrigger<T, BASIC_LINEAR_INTERPOLATION> trigger(value_t(1) /* threshold */, value_t(5) /* offset */);

            test_schmitt_trigger_with_signal<T, "slow rising edge (basic interpolation)">(trigger,                                                         //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0, 1, 5 /* >= offset + threshold */, 7, 7, 7, 8 /* <= offset - threshold */, 0, 0})), //
                {{RISING, 2}, {FALLING, 6.375f}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (basic interpolation)">(trigger,                  //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0 /* RISING */, 10 /* FALLING */, 0 /* RISING */, 8})), //
                {{RISING, 0.5f}, {FALLING, 1.5f}, {RISING, 2.625f}});
        };

        "integer-type: polynomial interpolation"_test = [] {
            SchmittTrigger<T, POLYNOMIAL_INTERPOLATION> trigger(value_t(1) /* threshold */, value_t(5) /* offset */);

            // N.B. integer types lose sub-sample precision through findCrossingIndex (truncated to integer).
            // With short signals the SG path falls back to linear regression.
            test_schmitt_trigger_with_signal<T, "slow rising edge (polynomial interpolation)">(trigger,                                                    //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0, 1, 5 /* >= offset + threshold */, 7, 7, 7, 8 /* <= offset - threshold */, 0, 0})), //
                {{RISING, 2}, {FALLING, 6}});

            trigger.reset();
            test_schmitt_trigger_with_signal<T, "fast rising/falling edges (polynomial interpolation)">(trigger,             //
                convert_signal<T>(gr::Tensor<double>(gr::data_from, {0 /* RISING */, 10 /* FALLING */, 0 /* RISING */, 8})), //
                {{RISING, 0}, {FALLING, 1}, {RISING, 2}});
        };
    } | std::tuple<uint8_t, int16_t>{};
};

int main() { /* not needed for UT */ }
