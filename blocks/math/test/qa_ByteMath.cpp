#include <boost/ut.hpp>

#include <gnuradio-4.0/math/ByteMath.hpp>

#include "gnuradio-4.0/meta/utils.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters1 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    gr::Size_t size;
};

template<typename T>
struct TestParameters2 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    gr::Size_t size;
    gr::Size_t k;
};

template<typename T>
struct TestParameters3 {
    gr::Tensor<T> input;
    gr::Tensor<T> output;
    gr::Size_t size;
    gr::Size_t k;
    gr::Size_t l;
    std::string len_tag_key;
    bool align_output;
    std::endian endianness;
};

template<typename T, typename BlockUnderTest>
void test_block_type1(const TestParameters1<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>();
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", p.size}});
    expect(graph.connect(src, "out"s, block, "in"s).has_value()) << "Failed to connect output port of src to input port 'in' of block";
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.input);
};

template<typename T, typename BlockUnderTest>
void test_block_type2(const TestParameters2<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"k", p.k}});
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", p.size}});
    expect(graph.connect(src, "out"s, block, "in"s).has_value()) << "Failed to connect output port of src to input port 'in' of block";
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.input);
};

template<typename T, typename BlockUnderTest>
void test_block_type3(const TestParameters3<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    // build test graph
    Graph graph;
    auto& block = graph.emplaceBlock<BlockUnderTest>({{"k", p.k}, {"l", p.l}, {"len_tag_key", p.len_tag_key},
                                                      {"align_output", p.align_output}});
    auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", p.size}});
    expect(graph.connect(src, "out"s, block, "in"s).has_value()) << "Failed to connect output port of src to input port 'in' of block";
    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>();
    expect(graph.connect(block, "out"s, sink, "in"s).has_value()) << "Failed to connect output port 'out' of block to input port of sink";

    // execute and confirm result
    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    expect(sched.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.input);
};

const boost::ut::suite<"byte math tests"> byteMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto bitSizedTypes = std::tuple< uint8_t, uint16_t >();

    "PackKBits"_test = []<typename T>(const T&) {
        test_block_type2<T, PackKBits<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(0), T(0), T(0), T(1), T(1), T(1), T(0), T(0), //
                                                   T(0), T(0), T(1), T(1), T(1), T(1), T(1), T(1)}),
            .output = gr::Tensor<T>(gr::data_from, {T(1), T(12), T(3), T(15)}),
            .size = 16,
            .k = 4
        });
    } | bitSizedTypes;

    "PackedToUnpacked"_test = []<typename T>(const T&) {
        test_block_type1<T, PackedToUnpacked<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(19)}),
            .output = gr::Tensor<T>(gr::data_from, {T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(1)}),
            .size = 8
        });
    } | std::tuple< uint8_t >();

    "RepackBits"_test = []<typename T>(const T&) {
        test_block_type3<T, RepackBits<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(25), T(26), T(27), T(28), T(29), T(30), T(31), T(32)}),
            .output = gr::Tensor<T>(gr::data_from, {T(1), T(3), T(0), T(5), T(1), T(6), T(6), T(0), //
                                                    T(4), T(3), T(4), T(6), T(1), T(4), T(7), T(0), //
                                                    T(7), T(3), T(0), T(0), T(2), T(2)}),
            .size = 22,
            .k = 8,
            .l = 3,
            .len_tag_key = "",
            .align_output = false,
            .endianness = std::endian::little
        });
    } | std::tuple< uint8_t >();

    "UnpackKBits"_test = []<typename T>(const T&) {
        test_block_type2<T, UnpackKBits<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(6)}),
            .output = gr::Tensor<T>(gr::data_from, {T(0), T(1), T(1), T(0)}),
            .size = 4,
            .k = 4
        });
    } | bitSizedTypes;

    "UnpackedToPacked"_test = []<typename T>(const T&) {
        test_block_type1<T, UnpackedToPacked<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(1), //
                                                   T(0), T(0), T(1), T(1), T(1), T(1), T(1), T(1)}),
            .output = gr::Tensor<T>(gr::data_from, {T(19), T(63)}),
            .size = 16
        });
    } | std::tuple< uint8_t >();

    "UnpackedToPacked"_test = []<typename T>(const T&) {
        test_block_type1<T, UnpackedToPacked<T>>({
            .input = gr::Tensor<T>(gr::data_from, {T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(1), //
                                                   T(0), T(0), T(1), T(1), T(1), T(1), T(1), T(1)}),
            .output = gr::Tensor<T>(gr::data_from, {T(4927)}),
            .size = 16
        });
    } | std::tuple< uint16_t >();
};

int main() { /* not needed for UT */ }
