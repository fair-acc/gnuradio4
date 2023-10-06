#include <unordered_set>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/graph.hpp>
#include <gnuradio-4.0/node.hpp>
#include <gnuradio-4.0/utils.hpp>

#include <gnuradio-4.0/basic/selector.hpp>

namespace fg = fair::graph;
using namespace fair::literals;

template<typename T>
struct repeated_source : public fg::node<repeated_source<T>> {
    std::uint32_t                  identifier = 0;
    std::uint32_t                  remaining_events_count;
    std::vector<T>                 values;
    std::vector<T>::const_iterator values_next;

    fg::PortOut<T>                 out;

    void
    settings_changed(const fair::graph::property_map & /*old_settings*/, const fair::graph::property_map &new_settings) noexcept {
        if (new_settings.contains("values")) {
            values_next = values.cbegin();
        }
    }

    fg::work_return_t
    work(std::size_t requested_work) {
        if (values_next == values.cend()) {
            values_next = values.cbegin();
        }

        if (remaining_events_count != 0) {
            using namespace fair::literals;
            auto &port   = fg::output_port<0>(this);
            auto &writer = port.streamWriter();
            auto  data   = writer.reserve_output_range(1_UZ);

            data[0]      = *values_next;
            data.publish(1_UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("{}: Last value sent was {}\n", static_cast<void *>(this), *values_next);
            }

            values_next++;

            return { requested_work, 1UL, fg::work_return_status_t::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            return { requested_work, 1UL, fg::work_return_status_t::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (repeated_source<T>), identifier, remaining_events_count, values, out);

template<typename T>
struct validator_sink : public fg::node<validator_sink<T>> {
    std::uint32_t                  identifier = 0;
    fg::PortIn<T>                  in;

    std::vector<T>                 expected_values;
    std::vector<T>::const_iterator expected_values_next;
    bool                           all_ok = true;

    bool
    verify() const {
        return all_ok && (expected_values_next == expected_values.cend());
    }

    void
    settings_changed(const fair::graph::property_map & /*old_settings*/, const fair::graph::property_map &new_settings) noexcept {
        if (new_settings.contains("expected_values")) {
            expected_values_next = expected_values.cbegin();
        }
    }

    void
    process_one(T value) {
        if (expected_values_next == expected_values.cend()) {
            all_ok = false;
            fmt::print("Error: {}#{}: We got more values than expected\n", static_cast<void *>(this), identifier);

        } else {
            if (value != *expected_values_next) {
                all_ok = false;
                fmt::print("Error: {}#{}: Got a value {}, but wanted {} (position {})\n", static_cast<void *>(this), identifier, value, *expected_values_next,
                           std::distance(expected_values.cbegin(), expected_values_next));
            }

            expected_values_next++;
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (validator_sink<T>), identifier, expected_values, in);

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public fg::node<adder<T>> {
    fg::PortIn<T>  addend0;
    fg::PortIn<T>  addend1;
    fg::PortOut<T> sum;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (adder<T>), addend0, addend1, sum);

struct test_definition {
    std::uint32_t                                        value_count;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> mapping;
    std::vector<std::vector<double>>                     input_values;
    std::vector<std::vector<double>>                     output_values;
    std::uint32_t                                        monitor_source;
    std::vector<double>                                  monitor_values;
    bool                                                 back_pressure;
};

void
execute_selector_test(test_definition definition) {
    using namespace boost::ut;

    const std::uint32_t                    sources_count = definition.input_values.size();
    const std::uint32_t                    sinks_count   = definition.output_values.size();

    fg::graph                              graph;
    std::vector<repeated_source<double> *> sources;
    std::vector<validator_sink<double> *>  sinks;
    gr::blocks::basic::Selector<double>   *selector;

    std::vector<std::uint32_t>             mapIn(definition.mapping.size());
    std::vector<std::uint32_t>             mapOut(definition.mapping.size());
    std::ranges::transform(definition.mapping, mapIn.begin(), [](auto &p) { return p.first; });
    std::ranges::transform(definition.mapping, mapOut.begin(), [](auto &p) { return p.second; });

    selector = std::addressof(graph.make_node<gr::blocks::basic::Selector<double>>({ { "nInputs", sources_count }, //
                                                                                     { "nOutputs", sinks_count },  //
                                                                                     { "mapIn", mapIn },           //
                                                                                     { "mapOut", mapOut },         //
                                                                                     { "backPressure", definition.back_pressure } }));

    for (std::uint32_t source_index = 0; source_index < sources_count; ++source_index) {
        sources.push_back(std::addressof(graph.make_node<repeated_source<double>>({ { "remaining_events_count", definition.value_count }, //
                                                                                    { "identifier", source_index },                       //
                                                                                    { "values", definition.input_values[source_index] } })));
        expect(sources[source_index]->settings().apply_staged_parameters().empty());
        expect(fair::graph::connection_result_t::SUCCESS == graph.dynamic_connect(*sources[source_index], 0, *selector, source_index + 1 /* there's one port before the inputs */));
    }

    for (std::uint32_t sink_index = 0; sink_index < sinks_count; ++sink_index) {
        sinks.push_back(std::addressof(graph.make_node<validator_sink<double>>({ { "identifier", sink_index }, //
                                                                                 { "expected_values", definition.output_values[sink_index] } })));
        expect(sinks[sink_index]->settings().apply_staged_parameters().empty());
        expect(fair::graph::connection_result_t::SUCCESS == graph.dynamic_connect(*selector, sink_index + 1 /* there's one port before the outputs */, *sinks[sink_index], 0));
    }

    validator_sink<double> *monitor_sink = std::addressof(graph.make_node<validator_sink<double>>({ { "identifier", static_cast<std::uint32_t>(-1) }, //
                                                                                                    { "expected_values", definition.monitor_values } }));
    expect(monitor_sink->settings().apply_staged_parameters().empty());
    expect(fair::graph::connection_result_t::SUCCESS == graph.dynamic_connect(*selector, 0, *monitor_sink, 0));

    for (std::size_t iterration = 0; iterration < definition.value_count * sources_count; ++iterration) {
        const auto max = std::numeric_limits<std::size_t>::max();
        for (auto *source : sources) {
            source->work(max);
        }
        selector->work(max);
        for (auto *sink : sinks) {
            sink->work(max);
        }
        monitor_sink->work(max);
    }

    if (!definition.back_pressure) {
        for (const auto &input : selector->inputs) {
            expect(eq(input.streamReader().available(), 0));
        }
    }

    for (auto *sink : sinks) {
        expect(sink->verify());
    }
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;
    using namespace gr::blocks::basic;

    "Selector<T> constructor"_test = [] {
        Selector<double> block_nop({ { "name", "block_nop" } });
        expect(block_nop.settings().apply_staged_parameters().empty());
        expect(eq(block_nop.nInputs, 0U));
        expect(eq(block_nop.nOutputs, 0U));
        expect(eq(block_nop.inputs.size(), 0U));
        expect(eq(block_nop.outputs.size(), 0U));
        expect(eq(block_nop._internalMapping.size(), 0U));

        Selector<double> block({ { "name", "block" }, { "nInputs", 4U }, { "nOutputs", 3U } });
        expect(block.settings().apply_staged_parameters().empty());
        expect(eq(block.nInputs, 4U));
        expect(eq(block.nOutputs, 3U));
        expect(eq(block.inputs.size(), 4U));
        expect(eq(block.outputs.size(), 3U));
        expect(eq(block._internalMapping.size(), 0U));
    };

    "basic Selector<T>"_test = [] {
        using T = double;
        const std::vector<uint32_t> outputMap{ 1U, 0U };
        Selector<T>                 block({ { "nInputs", 3U }, { "nOutputs", 2U }, { "mapIn", std::vector<uint32_t>{ 0U, 1U } }, { "mapOut", outputMap } }); // N.B. 3rd input is unconnected
        expect(block.settings().apply_staged_parameters().empty());
        expect(eq(block._internalMapping.size(), 2U));

        using internal_mapping_t = decltype(block._internalMapping);
        expect(block._internalMapping == internal_mapping_t{ { 0U, { outputMap[0] } }, { 1U, { outputMap[1] } } });
    };

    // Tests without the back pressure

    "Selector<T> 1 to 1 mapping"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 0 }, { 1, 1 }, { 2, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 1, 1, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, { 3, 3, 3, 3, 3 } }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = false });
    };

    "Selector<T> only one input used"_test = [] {
        execute_selector_test({ .value_count    = 5,                             //
                                .mapping        = { { 1, 1 } },                  //
                                .input_values   = { { 1 }, { 2 }, { 3 } },       //
                                .output_values  = { {}, { 2, 2, 2, 2, 2 }, {} }, //
                                .monitor_source = -1U,                           //
                                .monitor_values = {},                            //
                                .back_pressure  = false });
    };

    "Selector<T> all for one"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 1 }, { 1, 1 }, { 2, 1 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { {}, { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }, {} }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = false });
    };

    "Selector<T> one for all"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 1, 0 }, { 1, 1 }, { 1, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 } }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = false });
    };

    // tests with the back pressure

    "Selector<T> 1 to 1 mapping, with back pressure"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 0 }, { 1, 1 }, { 2, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 1, 1, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, { 3, 3, 3, 3, 3 } }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = true });
    };

    "Selector<T> only one input used, with back pressure"_test = [] {
        execute_selector_test({ .value_count    = 5,                             //
                                .mapping        = { { 1, 1 } },                  //
                                .input_values   = { { 1 }, { 2 }, { 3 } },       //
                                .output_values  = { {}, { 2, 2, 2, 2, 2 }, {} }, //
                                .monitor_source = -1U,                           //
                                .monitor_values = {},                            //
                                .back_pressure  = true });
    };

    "Selector<T> all for one, with back pressure"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 1 }, { 1, 1 }, { 2, 1 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { {}, { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }, {} }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = true });
    };

    "Selector<T> one for all, with back pressure"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 1, 0 }, { 1, 1 }, { 1, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 } }, //
                                .monitor_source = -1U,                                                         //
                                .monitor_values = {},                                                          //
                                .back_pressure  = true });
    };

    // Tests with a monitor

    "Selector<T> 1 to 1 mapping, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 0 }, { 1, 1 }, { 2, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 1, 1, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, { 3, 3, 3, 3, 3 } }, //
                                .monitor_source = 0U,                                                          //
                                .monitor_values = { 1, 1, 1, 1, 1 },                                           //
                                .back_pressure  = false });
    };

    "Selector<T> only one input used, with monitor, monitor source not mapped"_test = [] {
        execute_selector_test({ .value_count    = 5,                             //
                                .mapping        = { { 1, 1 } },                  //
                                .input_values   = { { 1 }, { 2 }, { 3 } },       //
                                .output_values  = { {}, { 2, 2, 2, 2, 2 }, {} }, //
                                .monitor_source = 0U,                            //
                                .monitor_values = { 1, 1, 1, 1, 1 },             //
                                .back_pressure  = false });
    };

    "Selector<T> all for one, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 0, 1 }, { 1, 1 }, { 2, 1 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { {}, { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }, {} }, //
                                .monitor_source = 1U,                                                          //
                                .monitor_values = { 2, 2, 2, 2, 2 },                                           //
                                .back_pressure  = false });
    };

    "Selector<T> one for all, with monitor, monitor source already mapped"_test = [] {
        execute_selector_test({ .value_count    = 5,                                                           //
                                .mapping        = { { 1, 0 }, { 1, 1 }, { 1, 2 } },                            //
                                .input_values   = { { 1 }, { 2 }, { 3 } },                                     //
                                .output_values  = { { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2 } }, //
                                .monitor_source = 1U,                                                          //
                                .monitor_values = { 2, 2, 2, 2, 2 },                                           //
                                .back_pressure  = false });
    };
};

int
main() { /* not needed for UT */
}
