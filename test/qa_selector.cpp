#include <boost/ut.hpp>

#include <graph.hpp>
#include <node.hpp>
#include <scheduler.hpp>
#include <utils.hpp>

#include <unordered_set>
#include <vector>

#include "blocklib/core/selector.hpp"


namespace fg = fair::graph;
using namespace fair::literals;

template<typename T>
struct fixed_source : public fg::node<fixed_source<T>> {
    std::uint32_t  remaining_events_count;

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
