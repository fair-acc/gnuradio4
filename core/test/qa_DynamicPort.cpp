#include <fmt/ranges.h>

#include <refl.hpp>

#include <boost/ut.hpp>

#include <gnuradio-4.0/buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace grg = gr;

using namespace std::string_literals;
using namespace gr::literals;

#ifdef ENABLE_DYNAMIC_PORTS
class dynamicBlock : public grg::node<dynamicBlock> {
public:
    dynamicBlock(std::string name) : grg::node<dynamicBlock>(name) {}
};
#endif

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public grg::Block<scale<T, Scale, R>> {
public:
    grg::PortIn<T>  original;
    grg::PortOut<R> scaled;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) const noexcept {
        return a * Scale;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T Scale, typename R), (scale<T, Scale, R>), original, scaled);

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public grg::Block<adder<T>> {
public:
    grg::PortIn<T>  addend0;
    grg::PortIn<T>  addend1;
    grg::PortOut<T> sum;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) const noexcept {
        return a + b;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename R), (adder<T, R>), addend0, addend1, sum);

template<typename T>
class cout_sink : public grg::Block<cout_sink<T>> {
public:
    grg::PortIn<T> sink;

    void
    processOne(T value) {
        fmt::print("Sinking a value: {}\n", value);
    }
};

static_assert(gr::BlockLike<cout_sink<float>>);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), sink);

template<typename T, T val, std::size_t count = 10_UZ>
class repeater_source : public grg::Block<repeater_source<T, val>> {
public:
    grg::PortOut<T> value;
    std::size_t     _counter = 0;

    gr::WorkReturn
    work(std::size_t requested_work) {
        if (_counter < count) {
            _counter++;
            auto &writer = outputPort<"value">(this).streamWriter();
            auto  data   = writer.reserve_output_range(1);
            data[0]      = val;
            data.publish(1);

            return { requested_work, 1_UZ, gr::WorkReturnStatus::OK };
        } else {
            return { requested_work, 0_UZ, gr::WorkReturnStatus::DONE };
        }
    }
};

static_assert(gr::BlockLike<repeater_source<int, 42>>);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, T val, std::size_t count), (repeater_source<T, val, count>), value);

const boost::ut::suite PortApiTests = [] {
    using namespace boost::ut::literals;
    using boost::ut::expect, boost::ut::eq, boost::ut::ge, boost::ut::nothrow, boost::ut::throws;
    using namespace gr;
    using namespace gr;

    "PortApi"_test = [] {
        static_assert(PortLike<PortIn<float>>);
        static_assert(PortLike<decltype(PortIn<float>())>);
        static_assert(PortLike<PortOut<float>>);
        static_assert(PortLike<MsgPortIn<float>>);
        static_assert(PortLike<MsgPortOut<float>>);

        static_assert(PortLike<PortInNamed<float, "in">>);
        static_assert(PortLike<decltype(PortInNamed<float, "">("in"))>);
        static_assert(PortLike<PortOutNamed<float, "out">>);
        static_assert(PortLike<MsgPortInNamed<"in_msg">>);
        static_assert(PortLike<MsgPortOutNamed<"out_msg">>);

        static_assert(PortIn<float, RequiredSamples<>>::Required::kMinSamples == 1LU);
        static_assert(PortIn<float, RequiredSamples<>>::Required::kMaxSamples == std::numeric_limits<std::size_t>::max());
        static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMinSamples == 1LU);
        static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMaxSamples == 2LU);
        expect(eq(PortIn<float, RequiredSamples<1, 2, false>>().min_samples, 1LU));
        expect(eq(PortIn<float, RequiredSamples<1, 2, false>>().max_samples, 2LU));
        expect(eq(PortIn<float, RequiredSamples<1, 2, true>>().min_samples, 1LU));
        expect(eq(PortIn<float, RequiredSamples<1, 2, true>>().max_samples, 2LU));
        static_assert(!RequiredSamples<1LU, 2LU, false>::kIsConst);
        static_assert(!std::is_const_v<decltype(PortIn<float, RequiredSamples<1LU, 2LU, false>>().min_samples)>);
        static_assert(!std::is_const_v<decltype(PortIn<float, RequiredSamples<1LU, 2LU, false>>().max_samples)>);
        static_assert(RequiredSamples<1LU, 2LU, true>::kIsConst);
        static_assert(std::is_const_v<decltype(PortIn<float, RequiredSamples<1LU, 2LU, true>>().min_samples)>);
        static_assert(std::is_const_v<decltype(PortIn<float, RequiredSamples<1LU, 2LU, true>>().max_samples)>);
        static_assert(PortIn<float>::direction() == PortDirection::INPUT);
        static_assert(PortOut<float>::direction() == PortDirection::OUTPUT);
    };

    "PortBufferApi"_test = [] {
        PortOut<float>     output_port;
        BufferWriter auto &writer = output_port.streamWriter();
        // BufferWriter auto                                             &tagWriter = output_port.tagWriter();
        expect(ge(writer.available(), 32_UZ));

        using ExplicitUnlimitedSize = RequiredSamples<1, std::numeric_limits<std::size_t>::max()>;
        PortIn<float, ExplicitUnlimitedSize> input_port;
        const BufferReader auto             &reader = input_port.streamReader();
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
        using ExplicitUnlimitedSize = RequiredSamples<1, std::numeric_limits<std::size_t>::max()>;
        PortOut<float, ExplicitUnlimitedSize> out;
        PortIn<float, ExplicitUnlimitedSize>  in;
        std::vector<DynamicPort>              port_list;

        port_list.emplace_back(out, DynamicPort::non_owned_reference_tag{});
        port_list.emplace_back(in, DynamicPort::non_owned_reference_tag{});

        expect(eq(port_list.size(), 2_UZ));
    };

    "ConnectionApi"_test = [] {
        using PortDirection::INPUT;
        using PortDirection::OUTPUT;

        // Block need to be alive for as long as the flow is
        grg::Graph flow;

        // Generators
        auto &answer = flow.emplaceBlock<repeater_source<int, 42>>();
        auto &number = flow.emplaceBlock<repeater_source<int, 6>>();

        auto &scaled = flow.emplaceBlock<scale<int, 2>>();
        auto &added  = flow.emplaceBlock<adder<int>>();
        auto &out    = flow.emplaceBlock<cout_sink<int>>();

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"value">(number).to<"original">(scaled)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"scaled">(scaled).to<"addend0">(added)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"value">(answer).to<"addend1">(added)));

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"sum">(added).to<"sink">(out)));

        gr::scheduler::Simple sched{ std::move(flow) };
        sched.runAndWait();
    };

#ifdef ENABLE_DYNAMIC_PORTS
    "PythonToBeConnectionApi"_test = [] {
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        OUT<float, "out0"> output_port;
        BufferWriter auto &writer = output_port.writer();
        IN<float, "in0">   input_port;

        auto               source = std::make_shared<dynamicBlock>("source");
        source->add_port(output_port);
        source->add_port(OUT<float, "out1">());
        expect(eq(source->dynamic_output_ports().size(), 2U));
        expect(eq(source->dynamic_input_ports().size(), 0U));

        auto sink = std::make_shared<dynamicBlock>("sink");
        expect(nothrow([&sink, &input_port]() { sink->add_port(input_port); }));
        expect(nothrow([&sink]() { sink->add_port(IN<float, "in1">()); }));
        expect(nothrow([&sink]() { sink->add_port(IN<float>("in2")); }));
        expect(throws([&sink]() { sink->add_port(IN<float>("in1")); }));
        expect(eq(sink->dynamic_output_ports().size(), 0U));
        expect(eq(sink->dynamic_input_ports().size(), 3U));

        grg::graph graph;

        expect(eq(graph.edges_count(), 0U));

        expect(eq(graph.add_edge(source, "out0", sink, "in0"), connection_result_t::SUCCESS));
        // N.B. should discourage indexed-based access <-> error prone, we test this anyway
        expect(eq(graph.add_edge(source, 0, sink, 1), connection_result_t::SUCCESS));

        expect(eq(graph.edges_count(), 2U));

        auto               mismatched_sink = std::make_shared<dynamicBlock>("mismatched_sink");
        IN<int32_t, "in0"> mismatched_typed_port;
        mismatched_sink->add_port(mismatched_typed_port);

        expect(eq(graph.add_edge(source, 0, mismatched_sink, 0), connection_result_t::FAILED));

        expect(eq(graph.edges_count(), 2U));

        // test runtime growing of src ports
        expect(ge(writer.available(), 32U));
        expect(eq(output_port.buffer().n_readers(), 0U));
        const auto old_size = writer.available();
        expect(eq(source->port<OUTPUT>("out0").value()->resizeBuffer(old_size + 1), connection_result_t::SUCCESS));
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
