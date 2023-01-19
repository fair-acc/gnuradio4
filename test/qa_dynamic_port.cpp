

#include <fmt/ranges.h>

#include "buffer.hpp"
#include "graph.hpp"
#include "refl.hpp"

#include <boost/ut.hpp>

namespace fg = fair::graph;

using namespace std::string_literals;

class dynamic_node : public fg::node<dynamic_node> {
public:
    dynamic_node(std::string name)
        : fg::node<dynamic_node>(name) {}

};


template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, Scale, R>, fg::IN<T, "original">, fg::OUT<R, "scaled">> {
public:
    scale(std::string name)
        : fg::node<scale<T, Scale, R>, fg::IN<T, "original">, fg::OUT<R, "scaled">>(std::move(name))
    {}

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>, fg::IN<T, "addend0">, fg::IN<T, "addend1">, fg::OUT<R, "sum">> {
public:
    adder(std::string name)
        : fg::node<adder<T>, fg::IN<T, "addend0">, fg::IN<T, "addend1">, fg::OUT<R, "sum">>(std::move(name))
    {}

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

template <typename T>
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, "sink">> {
public:
    cout_sink(std::string name)
        : fg::node<cout_sink<T>, fg::IN<T, "sink">>(std::move(name))
    {}

    void process_one(T value) {
        fmt::print("Sinking a value: {}\n", value);
    }

};

template <typename T, T value, std::size_t count = 10>
class repeater_source : public fg::node<repeater_source<T, value>, fg::OUT<T, "value">> {
private:
    using base = fg::node<repeater_source<T, value>, fg::OUT<T, "value">>;
    std::size_t _counter = 0;

public:
    repeater_source(std::string name)
        : fg::node<repeater_source<T, value>, fg::OUT<T, "value">>(std::move(name))
    {}

    fair::graph::work_result work() {
        if (_counter < count) {
            _counter++;
            auto& writer = base::template port<fair::graph::port_direction_t::OUTPUT, "value">().writer();
            auto [data, token] = writer.get(1);
            data[0] = value;
            writer.publish(token, 1);
        }
    }

};

const boost::ut::suite PortApiTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace fair::graph;

    "PortApi"_test = [] {
        static_assert(Port<IN<float, "in">>);
        static_assert(Port<decltype(IN<float>("in"))>);
        static_assert(Port<OUT<float, "out">>);
        static_assert(Port<IN_MSG<float, "in_msg">>);
        static_assert(Port<OUT_MSG<float, "out_msg">>);

        static_assert(IN<float, "in">::static_name() == fixed_string("in"));
        static_assert(requires { IN<float>("in").name(); });
    };

    "PortBufferApi"_test = [] {
        OUT<float, "out0"> output_port;
        BufferWriter auto& writer = output_port.writer();
        expect(ge(writer.available(), 32U));

        IN<float, "int0"> input_port;
        const BufferReader auto& reader = input_port.reader();
        expect(eq(reader.available(), 0U));
        input_port.setBuffer(output_port.buffer());

        expect(eq(output_port.buffer().n_readers(), 1U));

        int offset = 1;
        auto lambda = [&offset](auto& w) {
            std::iota(w.begin(), w.end(), offset);
            fmt::print("typed-port connected output vector: {}\n", w);
            offset += w.size();
        };

        expect(writer.try_publish(lambda, 32));
    };

    "RuntimePortApi"_test = [] {
        // declare in block
        OUT<float, "out"> out;
        IN<float, "in"> in;
        std::vector<dynamic_port> port_list;

        port_list.emplace_back(out);
        port_list.emplace_back(in);

        expect(eq(port_list.size(), 2U));
    };

    "ConnectionApi"_test = [] {
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;

        scale<int, 2> scaled("scaled"s);
        adder<int> added("added"s);
        cout_sink<int> out("out"s);

        repeater_source<int, 42> answer("answer"s);
        repeater_source<int, 6> number("number"s);

        // Nodes need to be alive for as long as the flow is
        graph flow;

        // Generators
        flow.register_node(answer);
        flow.register_node(number);

        flow.register_node(scaled);
        flow.register_node(added);
        flow.register_node(out);

        expect(eq(connection_result_t::SUCCESS, flow.connect<"value", "original">(number, scaled)));
        expect(eq(connection_result_t::SUCCESS, flow.connect<"scaled", "addend0">(scaled, added)));
        expect(eq(connection_result_t::SUCCESS, flow.connect<"value", "addend1">(answer, added)));

        expect(eq(connection_result_t::SUCCESS, flow.connect<"sum", "sink">(added, out)));

        flow.work();
    };

#ifdef ENABLE_DYNAMIC_PORTS
    "PythonToBeConnectionApi"_test = [] {
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        OUT<float, "out0"> output_port;
        BufferWriter auto& writer = output_port.writer();
        IN<float, "in0"> input_port;


        auto source = std::make_shared<dynamic_node>("source");
        source->add_port(output_port);
        source->add_port(OUT<float, "out1">());
        expect(eq(source->dynamic_output_ports().size(), 2U));
        expect(eq(source->dynamic_input_ports().size(), 0U));

        auto sink = std::make_shared<dynamic_node>("sink");
        expect(nothrow([&sink, &input_port] () { sink->add_port(input_port); } ));
        expect(nothrow([&sink] () { sink->add_port(IN<float, "in1">()); } ));
        expect(nothrow([&sink] () { sink->add_port(IN<float>("in2")); } ));
        expect(throws([&sink] () { sink->add_port(IN<float>("in1")); } ));
        expect(eq(sink->dynamic_output_ports().size(), 0U));
        expect(eq(sink->dynamic_input_ports().size(), 3U));

        fg::graph graph;

        expect(eq(graph.edges_count(), 0U));

        expect(eq(graph.add_edge(source, "out0", sink, "in0"), connection_result_t::SUCCESS));
        // N.B. should discourage indexed-based access <-> error prone, we test this anyway
        expect(eq(graph.add_edge(source, 0, sink, 1), connection_result_t::SUCCESS));

        expect(eq(graph.edges_count(), 2U));

        auto mismatched_sink = std::make_shared<dynamic_node>("mismatched_sink");
        IN<int32_t, "in0"> mismatched_typed_port;
        mismatched_sink->add_port(mismatched_typed_port);

        expect(eq(graph.add_edge(source, 0, mismatched_sink, 0), connection_result_t::FAILED));

        expect(eq(graph.edges_count(), 2U));


        // test runtime growing of src ports
        expect(ge(writer.available(), 32U));
        expect(eq(output_port.buffer().n_readers(), 0U));
        const auto old_size = writer.available();
        expect(eq(source->port<OUTPUT>("out0").value()->resize_buffer(old_size+1), connection_result_t::SUCCESS));
        expect(gt(writer.available(), old_size)) << fmt::format("runtime increase of buffer size {} -> {}", old_size, writer.available());

        // test runtime connection between ports
        expect(eq(output_port.buffer().n_readers(), 0U));
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in0").value()), connection_result_t::SUCCESS)) << "fist connection";
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in0").value()), connection_result_t::SUCCESS)) << "double connection";
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in1").value()), connection_result_t::SUCCESS)) << "second connection";

        const BufferReader auto& reader = input_port.reader();
        expect(eq(reader.available(), 0U)) << fmt::format("reader already has some bytes pending: {}", reader.available());

        expect(eq(output_port.buffer().n_readers(), 2U));

        constexpr auto lambda = [](auto& w) {
            std::iota(w.begin(), w.end(), 0U);
            fmt::print("dyn_port connected sink output: {}\n", w);
        };

        expect(writer.try_publish(lambda, 32U));
        expect(eq(input_port.reader().available(), 32U));
    };
#endif
};


int main() { /* tests are statically executed */ }
