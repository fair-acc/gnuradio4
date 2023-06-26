#include <graph.hpp>

#include <list>

namespace fg = fair::graph;

#include "blocklib/core/unit-test/common_nodes.hpp"

template<typename T>
std::atomic_size_t multi_adder<T>::_unique_id_counter = 0;

template<typename T>
struct fixed_source : public fg::node<fixed_source<T>, fg::OUT<T, 0, 1024, "out">> {
    T value = 1;

    fg::work_return_t
    work() {
        using namespace fair::literals;
        auto &port = fg::output_port<0>(this);
        auto &writer = port.streamWriter();
        auto data = writer.reserve_output_range(1_UZ);
        data[0] = value;
        data.publish(1_UZ);

        value += 1;
        return fg::work_return_t::OK;
    }
};

template<typename T>
struct cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, 1024, "in">> {
    std::size_t remaining = 0;

    void
    process_one(T value) {
        remaining--;
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), remaining);

int
main() {
    constexpr const std::size_t sources_count = 10;
    constexpr const std::size_t events_count = 5;

    fg::graph flow_graph;

    // Adder has sources_count inputs in total, but let's create
    // sources_count / 2 inputs on construction, and change the number
    // via settings
    auto &adder = flow_graph.add_node(std::make_unique<multi_adder<double>>(sources_count / 2));
    auto &sink = flow_graph.make_node<cout_sink<double>>({{"remaining", events_count}});

    // Function that adds a new source node to the graph, and connects
    // it to one of adder's ports
    std::ignore = adder.settings().set({ { "input_port_count", 10 } });
    std::ignore = adder.settings().apply_staged_parameters();

    std::vector<fixed_source<double> *> sources;
    for (std::size_t i = 0; i < sources_count; ++i) {
        auto &source = flow_graph.make_node<fixed_source<double>>();
        sources.push_back(&source);
        flow_graph.dynamic_connect(source, 0, adder, sources.size() - 1);
    }

    flow_graph.dynamic_connect(adder, 0, sink, 0);

    for (std::size_t i = 0; i < events_count; ++i) {
        for (auto *source : sources) {
            source->work();
        }
        std::ignore = adder.work();
        std::ignore = sink.work();
    }
}
