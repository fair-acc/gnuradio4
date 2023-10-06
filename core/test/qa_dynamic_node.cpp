#include <list>

#include <gnuradio-4.0/graph.hpp>

#include <gnuradio-4.0/basic/common_nodes.hpp>

template<typename T>
std::atomic_size_t multi_adder<T>::_unique_id_counter = 0;

template<typename T>
struct fixed_source : public gr::node<fixed_source<T>, gr::PortOutNamed < T, "out">> {
    T value = 1;

    gr::work_return_t
    work(std::size_t requested_work) {
        using namespace gr::literals;
        auto &port   = gr::output_port<0>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve_output_range(1_UZ);
        data[0]      = value;
        data.publish(1_UZ);

        value += 1;
        return {requested_work, 1_UZ, gr::work_return_status_t::OK };
    }
};

static_assert(gr::NodeType<fixed_source<int>>);

template<typename T>
struct cout_sink : public gr::node<cout_sink<T>, gr::PortInNamed < T, "in">> {
    std::size_t remaining = 0;

    void
    process_one(T value) {
        remaining--;
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

static_assert(gr::NodeType<cout_sink<int>>);

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), remaining);

int
main() {
    constexpr const std::size_t sources_count = 10;
    constexpr const std::size_t events_count  = 5;

    gr::graph                   flow_graph;

    // Adder has sources_count inputs in total, but let's create
    // sources_count / 2 inputs on construction, and change the number
    // via settings
    auto &adder = flow_graph.add_node(std::make_unique<multi_adder<double>>(sources_count / 2));
    auto &sink  = flow_graph.make_node<cout_sink<double>>({ { "remaining", events_count } });

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
            source->work(std::numeric_limits<std::size_t>::max());
        }
        std::ignore = adder.work(std::numeric_limits<std::size_t>::max());
        std::ignore = sink.work(std::numeric_limits<std::size_t>::max());
    }
}
