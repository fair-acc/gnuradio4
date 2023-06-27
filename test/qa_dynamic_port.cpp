#include <fmt/ranges.h>

#include "buffer.hpp"
#include "graph.hpp"
#include "refl.hpp"
#include "scheduler.hpp"
#include "utils.hpp"

#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fg = fair::graph;

using namespace std::string_literals;
using namespace fair::literals;

#ifdef ENABLE_DYNAMIC_PORTS
class dynamic_node : public fg::node<dynamic_node> {
public:
    dynamic_node(std::string name) : fg::node<dynamic_node>(name) {}
};
#endif

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, Scale, R>> {
public:
    fg::IN<T>  original;
    fg::OUT<R> scaled;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * Scale;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T Scale, typename R), (scale<T, Scale, R>), original, scaled);

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>> {
public:
    fg::IN<T>  addend0;
    fg::IN<T>  addend1;
    fg::OUT<T> sum;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename R), (adder<T, R>), addend0, addend1, sum);

template<typename T>
class cout_sink : public fg::node<cout_sink<T>> {
public:
    fg::IN<T> sink;

    void
    process_one(T value) {
        fmt::print("Sinking a value: {}\n", value);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), sink);

template<typename T, T val, std::size_t count = 10_UZ>
class repeater_source : public fg::node<repeater_source<T, val>> {
public:
    fg::OUT<T>  value;
    std::size_t _counter = 0;

    fair::graph::work_return_t
    work() {
        if (_counter < count) {
            _counter++;
            auto &writer = output_port<"value">(this).streamWriter();
            auto  data   = writer.reserve_output_range(1);
            data[0]      = val;
            data.publish(1);

            return fair::graph::work_return_t::OK;
        } else {
            return fair::graph::work_return_t::DONE;
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T val, std::size_t count), (repeater_source<T, val, count>), value);

const boost::ut::suite PortApiTests = [] {
    using namespace boost::ut::literals;
    using boost::ut::expect, boost::ut::eq, boost::ut::ge, boost::ut::nothrow, boost::ut::throws;
    using namespace gr;
    using namespace fair::graph;

    "PortApi"_test = [] {
        static_assert(Port<IN<float, 0, 0, "in">>);
        static_assert(Port<decltype(IN<float>("in"))>);
        static_assert(Port<OUT<float, 0, 0, "out">>);
        static_assert(Port<IN_MSG<float, 0, 0, "in_msg">>);
        static_assert(Port<OUT_MSG<float, 0, 0, "out_msg">>);

        static_assert(IN<float, 0, 0, "in">::static_name() == fixed_string("in"));
        static_assert(requires { IN<float>("in").name(); });
    };

    "PortBufferApi"_test = [] {
        OUT<float, 0, std::numeric_limits<std::size_t>::max(), "out0"> output_port;
        BufferWriter auto                                             &writer    = output_port.streamWriter();
        // BufferWriter auto                                             &tagWriter = output_port.tagWriter();
        expect(ge(writer.available(), 32_UZ));

        IN<float, 0, std::numeric_limits<std::size_t>::max(), "int0"> input_port;
        const BufferReader auto                                      &reader = input_port.streamReader();
        expect(eq(reader.available(), 0_UZ));
        auto buffers = output_port.buffer();
        input_port.setBuffer(buffers.streamBuffer, buffers.tagBufferType);

        expect(eq(buffers.streamBuffer.n_readers(), 1_UZ));

        int  offset = 1;
        auto lambda = [&offset](auto &w) {
            std::iota(w.begin(), w.end(), offset);
            fmt::print("typed-port connected output vector: {}\n", w);
            offset += static_cast<int>(w.size());
        };

        expect(writer.try_publish(lambda, 32_UZ));
    };

    "RuntimePortApi"_test = [] {
        // declare in block
        OUT<float, 0, std::numeric_limits<std::size_t>::max(), "out"> out;
        IN<float, 0, std::numeric_limits<std::size_t>::max(), "in">   in;
        std::vector<dynamic_port>                                     port_list;

        port_list.emplace_back(out);
        port_list.emplace_back(in);

        expect(eq(port_list.size(), 2_UZ));
    };

    "ConnectionApi"_test = [] {
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;

        // Nodes need to be alive for as long as the flow is
        graph flow;

        // Generators
        auto &answer = flow.make_node<repeater_source<int, 42>>();
        auto &number = flow.make_node<repeater_source<int, 6>>();

        auto &scaled = flow.make_node<scale<int, 2>>();
        auto &added  = flow.make_node<adder<int>>();
        auto &out    = flow.make_node<cout_sink<int>>();

        expect(eq(connection_result_t::SUCCESS, flow.connect<"value">(number).to<"original">(scaled)));
        expect(eq(connection_result_t::SUCCESS, flow.connect<"scaled">(scaled).to<"addend0">(added)));
        expect(eq(connection_result_t::SUCCESS, flow.connect<"value">(answer).to<"addend1">(added)));

        expect(eq(connection_result_t::SUCCESS, flow.connect<"sum">(added).to<"sink">(out)));

        fair::graph::scheduler::simple sched{ std::move(flow) };
        sched.run_and_wait();
    };

#ifdef ENABLE_DYNAMIC_PORTS
    "PythonToBeConnectionApi"_test = [] {
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        OUT<float, "out0"> output_port;
        BufferWriter auto &writer = output_port.writer();
        IN<float, "in0">   input_port;

        auto               source = std::make_shared<dynamic_node>("source");
        source->add_port(output_port);
        source->add_port(OUT<float, "out1">());
        expect(eq(source->dynamic_output_ports().size(), 2U));
        expect(eq(source->dynamic_input_ports().size(), 0U));

        auto sink = std::make_shared<dynamic_node>("sink");
        expect(nothrow([&sink, &input_port]() { sink->add_port(input_port); }));
        expect(nothrow([&sink]() { sink->add_port(IN<float, "in1">()); }));
        expect(nothrow([&sink]() { sink->add_port(IN<float>("in2")); }));
        expect(throws([&sink]() { sink->add_port(IN<float>("in1")); }));
        expect(eq(sink->dynamic_output_ports().size(), 0U));
        expect(eq(sink->dynamic_input_ports().size(), 3U));

        fg::graph graph;

        expect(eq(graph.edges_count(), 0U));

        expect(eq(graph.add_edge(source, "out0", sink, "in0"), connection_result_t::SUCCESS));
        // N.B. should discourage indexed-based access <-> error prone, we test this anyway
        expect(eq(graph.add_edge(source, 0, sink, 1), connection_result_t::SUCCESS));

        expect(eq(graph.edges_count(), 2U));

        auto               mismatched_sink = std::make_shared<dynamic_node>("mismatched_sink");
        IN<int32_t, "in0"> mismatched_typed_port;
        mismatched_sink->add_port(mismatched_typed_port);

        expect(eq(graph.add_edge(source, 0, mismatched_sink, 0), connection_result_t::FAILED));

        expect(eq(graph.edges_count(), 2U));

        // test runtime growing of src ports
        expect(ge(writer.available(), 32U));
        expect(eq(output_port.buffer().n_readers(), 0U));
        const auto old_size = writer.available();
        expect(eq(source->port<OUTPUT>("out0").value()->resize_buffer(old_size + 1), connection_result_t::SUCCESS));
        expect(gt(writer.available(), old_size)) << fmt::format("runtime increase of buffer size {} -> {}", old_size, writer.available());

        // test runtime connection between ports
        expect(eq(output_port.buffer().n_readers(), 0U));
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in0").value()), connection_result_t::SUCCESS)) << "fist connection";
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in0").value()), connection_result_t::SUCCESS)) << "double connection";
        expect(eq(source->port<OUTPUT>("out0").value()->connect(*sink->port<INPUT>("in1").value()), connection_result_t::SUCCESS)) << "second connection";

        const BufferReader auto &reader = input_port.reader();
        expect(eq(reader.available(), 0U)) << fmt::format("reader already has some bytes pending: {}", reader.available());

        expect(eq(output_port.buffer().n_readers(), 2U));

        constexpr auto lambda = [](auto &w) {
            std::iota(w.begin(), w.end(), 0U);
            fmt::print("dyn_port connected sink output: {}\n", w);
        };

        expect(writer.try_publish(lambda, 32U));
        expect(eq(input_port.reader().available(), 32U));
    };
#endif
};

int
main() { /* tests are statically executed */
}
