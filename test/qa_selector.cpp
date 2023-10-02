#include <boost/ut.hpp>

#include <graph.hpp>
#include <node.hpp>
#include <scheduler.hpp>
#include <utils.hpp>

#include <unordered_set>
#include <vector>

namespace fg = fair::graph;
using namespace fair::literals;

template<typename T>
struct fixed_source : public fg::node<fixed_source<T>> {
    std::uint32_t    remaining_events_count;

    T              value = 1;
    fg::PortOut<T> out;

    fg::work_return_t
    work(std::size_t requested_work) {
        if (remaining_events_count != 0) {
            using namespace fair::literals;
            auto &port   = fg::output_port<0>(this);
            auto &writer = port.streamWriter();
            auto  data   = writer.reserve_output_range(1_UZ);
            data[0]      = value;
            data.publish(1_UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("{}: Last value sent was {}\n", static_cast<void *>(this), value);
            }

            return { requested_work, 1UL, fg::work_return_status_t::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            return { requested_work, 1UL, fg::work_return_status_t::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fixed_source<T>), remaining_events_count, value, out);

template<typename T>
struct cout_sink : public fg::node<cout_sink<T>> {
    std::uint32_t identifier = 0;
    std::uint32_t remaining  = 0;
    fg::PortIn<T> in;

    void
    process_one(T value) {
        remaining--;
        std::cerr << fmt::format("cout_sink[{}] got {}, {} still to go\n", identifier, value, remaining);
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), identifier, remaining, in);

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

namespace gr::blocks::basic {
using namespace fair::graph;

// optional shortening
template<typename T, fair::meta::fixed_string description = "", typename... Arguments>
using A           = Annotated<T, description, Arguments...>;

using SelectorDoc = Doc<R""(
@brief basic multiplexing class to route arbitrary inputs to outputs
)"">;

template<typename T>
struct Selector : node<Selector<T>, SelectorDoc> {
    // port definitions
    PortIn<std::int32_t, Async, Optional> selectOut;
    PortOut<T, Async, Optional>           monitorOut; // optional monitor output (more for demo/API purposes than actual need)
    std::vector<PortIn<T, Async>>         inputs;     // TODO: need to add exception to pmt_t that this isn't interpreted as a settings type
    std::vector<PortOut<T, Async>>        outputs;

    // settings
    A<std::uint32_t, "nInputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>    nInputs  = 0U;
    A<std::uint32_t, "nOutputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>   nOutputs = 0U;
    A<std::vector<std::uint32_t>, "mapIn", Visible, Doc<"input port index to route from">>     mapIn; // N.B. need two vectors since pmt_t doesn't support pairs (yet!?!)
    A<std::vector<std::uint32_t>, "mapOut", Visible, Doc<"output port index to route to">>     mapOut;
    A<bool, "backPressure", Visible, Doc<"true: do not consume samples from un-routed ports">> backPressure = false;
    std::vector<std::pair<std::size_t, std::size_t>>                                           _internalMapping;
    std::int32_t                                                                               _selectedSrc = -1;

    using has_process_bulk                                                                                  = std::true_type;

    constexpr Selector() noexcept : Selector({}) {}

    Selector(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter) noexcept : node<Selector<T>, SelectorDoc>(init_parameter) {
        if (empty(init_parameter)) {
            return;
        }
        std::ignore     = this->settings().apply_staged_parameters();

        selectOut.name  = "selectOut";
        monitorOut.name = "monitorOut";
    }

    void
    settings_changed(const fair::graph::property_map &old_settings, const fair::graph::property_map &new_settings) noexcept {
        if (new_settings.contains("nInputs") || new_settings.contains("nOutputs")) {
            fmt::print("{}: configuration changed: nInputs {} -> {}, nOutputs {} -> {}\n", static_cast<void *>(this), old_settings.at("nInputs"),
                       new_settings.contains("nInputs") ? new_settings.at("nInputs") : "same", old_settings.at("nOutputs"), new_settings.contains("nOutputs") ? new_settings.at("nOutputs") : "same");
            inputs.resize(nInputs);
            outputs.resize(nOutputs);
        }
        if (new_settings.contains("mapIn") || new_settings.contains("mapOut")) {
            assert(mapIn.value.size() == mapOut.value.size() && "mapIn and mapOut must have the same length");
            _internalMapping.clear();
            _internalMapping.reserve(mapIn.value.size());
            std::unordered_set<std::size_t> setOutput;
            for (std::size_t i = 0U; i < mapOut.value.size(); ++i) {
                if (mapIn.value[i] < nInputs && mapOut.value[i] < nOutputs) {
                    if (setOutput.contains(mapOut.value[i])) {
                        // communicate duplicate output and skip
                        fmt::print("{}: selector() - duplicate output index {} in map\n", static_cast<void *>(this), mapOut.value[i]);
                        continue;
                    }
                    _internalMapping.emplace_back(mapIn.value[i], mapOut.value[i]);
                    setOutput.insert(mapOut.value[i]);
                } else {
                    // report error and/or just ignore
                    fmt::print("{}: selector() - invalid requested input-output pair ({}, {}) not in range ([0, {}],[0, {}])\n", static_cast<void *>(this), mapIn.value[i], mapOut.value[i], nInputs,
                               nOutputs);
                }
            }
        }
    }

    fair::graph::work_return_status_t
    process_bulk(/* fg::ConsumableSpan */ auto  &select, //
                 /* fg::ConsumableSpan */ auto  &ins,
                 /* fg::PublishableSpan */ auto &monOut, //
                 /* fg::PublishableSpan */ auto &outs) {
        if (_internalMapping.empty()) {
            if (backPressure) {
                std::for_each(ins.begin(), ins.end(), [](auto *input) { input->consume(0_UZ); });
            } else {
                // make the implicit consume all available behaviour explicit
                std::for_each(ins.begin(), ins.end(), [](auto *input) { input->consume(input->available()); });
            }
            return fg::work_return_status_t::OK;
        }

        std::unordered_map<std::size_t, std::size_t> used_inputs;
        for (const auto &[input_index, output_index] : _internalMapping) {
            auto *input_reader    = ins[input_index];
            auto *output_writer   = outs[output_index];

            auto  input_available = [&] {
                auto it = used_inputs.find(input_index);
                if (it != used_inputs.end()) {
                    return it->second;
                }

                const auto available     = input_reader->available();
                used_inputs[input_index] = available;
                return available;
            }();

            auto input_span = input_reader->get(input_available);

            if (input_available > 0) {
                auto output_span = output_writer->reserve_output_range(input_available);

                for (std::size_t i = 0; i < input_span.size(); ++i) {
                    output_span[i] = input_span[i];
                }

                output_span.publish(input_available);
            }
        }

        if (const auto select_available = select->available(); select_available > 0) {
            auto select_span = select->get(select_available);
            _selectedSrc     = select_span.back();

            if (_selectedSrc >= 0 && _selectedSrc < ins.size()) {
                // write to optional fixed monitor output
                // auto monitor_written_count = ins[_selectedSrc]->available();
                // std::ranges::copy(monOut, ins[_selectedSrc]);
                // const std::size_t publishedSamples = std::min(monOut.size(), ins[_selectedSrc].size());
                // monOut.publish(publishedSamples);
            }

            select->consume(select_available); // consume all samples on the 'select' streaming input port
        }

        for (auto src_port = 0U; src_port < ins.size(); ++src_port) {
            if (auto it = used_inputs.find(src_port); it != used_inputs.end()) {
                // If we read from this input, consume exactly the number of bytes we read
                ins[src_port]->consume(it->second);

            } else if (backPressure) {
                ins[src_port]->consume(0_UZ);

            } else {
                // make the implicit consume all available behaviour explicit
                ins[src_port]->consume(ins[src_port]->available());
            }
        }

        // N.B. some corner case-handling to be added:
        // * one input mapped to multiple outputs -> check for consistent and produce the same min available() for all mapped outputs

        return fg::work_return_status_t::OK;
    }
};
} // namespace gr::blocks::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::blocks::basic::Selector<T>), selectOut, inputs, monitorOut, outputs, nInputs, nOutputs, mapIn, mapOut, backPressure);
static_assert(fg::HasProcessBulkFunction<gr::blocks::basic::Selector<double>>);

auto
make_graph(std::uint32_t events_count, std::uint32_t sources_count, std::uint32_t sinks_count, std::vector<std::pair<std::uint32_t, std::uint32_t>> map) {
    struct result_type {
        fg::graph                            graph;
        std::vector<fixed_source<double> *>  sources;
        std::vector<cout_sink<double> *>     sinks;
        gr::blocks::basic::Selector<double> *selector;
    };

    result_type                result;

    std::vector<std::uint32_t> mapIn(map.size());
    std::vector<std::uint32_t> mapOut(map.size());
    std::ranges::transform(map, mapIn.begin(), [](auto &p) { return p.first; });
    std::ranges::transform(map, mapOut.begin(), [](auto &p) { return p.second; });

    result.selector = std::addressof(
            result.graph.make_node<gr::blocks::basic::Selector<double>>({ { "nInputs", sources_count }, { "nOutputs", sinks_count }, { "mapIn", mapIn }, { "mapOut", mapOut } }));

    for (std::uint32_t source_index = 0; source_index < sources_count; ++source_index) {
        result.sources.push_back(std::addressof(result.graph.make_node<fixed_source<double>>({ { "remaining_events_count", events_count }, { "value", static_cast<double>(source_index) } })));
        assert(fair::graph::connection_result_t::SUCCESS == result.graph.dynamic_connect(*result.sources[source_index], 0, *result.selector, source_index));
    }

    for (std::uint32_t sink_index = 0; sink_index < sinks_count; ++sink_index) {
        result.sinks.push_back(std::addressof(result.graph.make_node<cout_sink<double>>({ { "remaining", events_count }, { "identifier", sink_index } })));
        assert(fair::graph::connection_result_t::SUCCESS == result.graph.dynamic_connect(*result.selector, sink_index, *result.sinks[sink_index], 0));
    }

    return result;
}

const boost::ut::suite SelectorTest = [] {
    using namespace boost::ut;
    using namespace gr::blocks::basic;

    "Selector<T> constructor"_test = [] {
        Selector<double> block_nop({ { "name", "block_nop" } });
        expect(eq(block_nop.nInputs, 0U));
        expect(eq(block_nop.nOutputs, 0U));
        expect(eq(block_nop.inputs.size(), 0U));
        expect(eq(block_nop.outputs.size(), 0U));
        expect(eq(block_nop._internalMapping.size(), 0U));
        Selector<double> block({ { "name", "block" }, { "nInputs", 4U }, { "nOutputs", 3U } });
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
        expect(eq(block._internalMapping.size(), 2U));
        expect(eq(block._internalMapping[0].first, 0U));
        expect(eq(block._internalMapping[1].first, 1U));
        expect(eq(block._internalMapping[0].second, outputMap[0]));
        expect(eq(block._internalMapping[1].second, outputMap[1]));
    };

    "Selector<T> vis scheduler"_test = [] {
        auto thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten

        auto [graph, sources, sinks, selector] = make_graph(10, 3, 3, { { 0, 0 }, { 1, 1 }, { 2, 2 } });

        // fg::scheduler::simple scheduler(std::move(graph), thread_pool);
        // scheduler.run_and_wait();

        for (std::size_t iterration = 0; iterration < 100; ++iterration) {
            for (auto *source : sources) {
                source->work(std::numeric_limits<std::size_t>::max());
            }
            selector->work(std::numeric_limits<std::size_t>::max());
            for (auto *sink : sinks) {
                sink->work(std::numeric_limits<std::size_t>::max());
            }
        }
    };
};

int
main() { /* not needed for UT */
}
