/*
 * Core math-block QA – covers the blocks that were present _before_
 * the 2025-06 extensions (Add/Sub/Multiply/Divide and their *Const
 * variants).  Splitting the test lowers the peak memory GCC needs.
 */

#include <boost/ut.hpp>

// No <format> – wasm/libc++ in CI doesn’t ship it
#include <gnuradio-4.0/math/Math.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters {
    std::vector<T>              input{};
    std::vector<std::vector<T>> inputs{};
    std::vector<T>              output{};
};

template<typename T, typename BlockUnderTest>
void test_block_with_graph(const TestParameters<T>& p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    Graph g;
    auto& sink = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();

    if (!p.input.empty()) {
        /* unary path --------------------------------------------------- */
        auto& blk = g.emplaceBlock<BlockUnderTest>();
        auto& src = g.emplaceBlock<TagSource<T>>(property_map{{"values", p.input}, {"n_samples_max", static_cast<Size_t>(p.input.size())}});
        expect(eq(g.connect(src, "out"s, blk, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(blk).template to<"in">(sink), ConnectionResult::SUCCESS));

    } else {
        /* n-input path -------------------------------------------------- */
        const Size_t n_in = static_cast<Size_t>(p.inputs.size());
        auto&        blk  = g.emplaceBlock<BlockUnderTest>(property_map{{"n_inputs", n_in}});
        for (Size_t i = 0; i < n_in; ++i) {
            auto& src = g.emplaceBlock<TagSource<T>>(property_map{{"values", p.inputs[i]}, {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
            expect(eq(g.connect(src, "out"s, blk, "in#"s + std::to_string(i)), ConnectionResult::SUCCESS));
        }
        expect(eq(g.connect<"out">(blk).template to<"in">(sink), ConnectionResult::SUCCESS));
    }

    scheduler::Simple sch{std::move(g)};
    expect(sch.runAndWait().has_value());
    expect(std::ranges::equal(sink._samples, p.output));
}

template<typename T, typename BlockUnderTest>
void test_block_process_bulk(const TestParameters<T>& p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    size_t n_inputs = p.inputs.size();
    auto   blk      = BlockUnderTest(gr::property_map{{"n_inputs", n_inputs}});

    size_t         num_samples = p.inputs[0].size();
    std::vector<T> out(num_samples);

    std::vector<std::span<const T>> vec_spans;
    vec_spans.reserve(p.inputs.size());
    for (const auto& v : p.inputs) {
        vec_spans.emplace_back(v);
    }

    std::span<const std::span<const T>> input_spans(vec_spans);

    blk.processBulk(input_spans, out);

    expect(std::ranges::equal(out, p.output));
}

template<typename T, typename BlockUnderTest>
void test_block_process_one(const TestParameters<T>& p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    auto blk = BlockUnderTest();

    std::vector<T> out;
    out.reserve(out.size());
    for (auto& v : p.input) {
        out.push_back(blk.processOne(v));
    }

    expect(std::ranges::equal(out, p.output));
}

const boost::ut::suite<"Math blocks"> suite_core = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;

    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>();
    constexpr auto kComplexTypes    = std::tuple<std::complex<float>>();
    constexpr auto kLogicTypes      = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>();

    // Only test with a full graph for a limited number of types
    constexpr auto kLimitedTypes = std::tuple<float>();

    /* ---------------------------------------------------------------- */
    "Add"_test = []<typename T>(const T&) {
        test_block_with_graph<T, Add<T>>({.inputs = {{1, 2, 3}}, .output = {1, 2, 3}});
        test_block_with_graph<T, Add<T>>({.inputs = {{1, 2}, {3, 4}}, .output = {4, 6}});
    } | kLimitedTypes;

    "Subtract"_test = []<typename T>(const T&) {
        test_block_with_graph<T, Subtract<T>>({.inputs = {{5, 4}}, .output = {5, 4}});
        test_block_with_graph<T, Subtract<T>>({.inputs = {{5, 4}, {3, 1}}, .output = {2, 3}});
    } | kLimitedTypes;

    "Multiply"_test = []<typename T>(const T&) {
        test_block_with_graph<T, Multiply<T>>({.inputs = {{2, 3}}, .output = {2, 3}});
        test_block_with_graph<T, Multiply<T>>({.inputs = {{2, 3}, {4, 5}}, .output = {8, 15}});
    } | kLimitedTypes;

    "Divide"_test = []<typename T>(const T&) {
        test_block_with_graph<T, Divide<T>>({.inputs = {{8, 6}}, .output = {8, 6}});
        test_block_with_graph<T, Divide<T>>({.inputs = {{8, 6}, {2, 3}}, .output = {4, 2}});
    } | kLimitedTypes;

    "Add"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Add<T>>({.inputs = {{1, 2, 3}}, .output = {1, 2, 3}});
        test_block_process_bulk<T, Add<T>>({.inputs = {{1, 2}, {3, 4}}, .output = {4, 6}});
    } | kArithmeticTypes;

    "Subtract"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Subtract<T>>({.inputs = {{5, 4}}, .output = {5, 4}});
        test_block_process_bulk<T, Subtract<T>>({.inputs = {{5, 4}, {3, 1}}, .output = {2, 3}});
    } | kArithmeticTypes;

    "Multiply"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Multiply<T>>({.inputs = {{2, 3}}, .output = {2, 3}});
        test_block_process_bulk<T, Multiply<T>>({.inputs = {{2, 3}, {4, 5}}, .output = {8, 15}});
    } | kArithmeticTypes;

    "Divide"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Divide<T>>({.inputs = {{8, 6}}, .output = {8, 6}});
        test_block_process_bulk<T, Divide<T>>({.inputs = {{8, 6}, {2, 3}}, .output = {4, 2}});
    } | kArithmeticTypes;

    "ComplexAdd"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Add<T>>({.inputs = {{{1, 2}, {3, 4}, {5, 6}}}, .output = {{1, 2}, {3, 4}, {5, 6}}});
        test_block_process_bulk<T, Add<T>>({.inputs = {{{1, 2}, {3, 4}}, {{-5, 6}, {7, -8}}}, .output = {{-4, 8}, {10, -4}}});
    } | kComplexTypes;

    "ComplexSubtract"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Subtract<T>>({.inputs = {{{1, 2}, {3, 4}, {5, 6}}}, .output = {{1, 2}, {3, 4}, {5, 6}}});
        test_block_process_bulk<T, Subtract<T>>({.inputs = {{{1, 2}, {3, 4}}, {{-5, 6}, {7, -8}}}, .output = {{6, -4}, {-4, 12}}});
    } | kComplexTypes;

    "ComplexMultiply"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Multiply<T>>({.inputs = {{{1, 2}, {3, 4}, {5, 6}}}, .output = {{1, 2}, {3, 4}, {5, 6}}});
        test_block_process_bulk<T, Multiply<T>>({.inputs = {{{1, 2}, {3, 4}}, {{-5, 6}, {7, -8}}}, .output = {{-17, -4}, {53, 4}}});
    } | kComplexTypes;

    "ComplexDivide"_test = []<typename T>(const T&) {
        test_block_process_bulk<T, Divide<T>>({.inputs = {{{1, 2}, {3, 4}, {5, 6}}}, .output = {{1, 2}, {3, 4}, {5, 6}}});
        test_block_process_bulk<T, Divide<T>>({.inputs = {{{-5, 10}, {10, -5}}, {{3, 4}, {3, -4}}}, .output = {{1, 2}, {2, 1}}});
    } | kComplexTypes;

    /* ----- *Const variants ------------------------------------------ */
    "AddConst"_test = []<typename T>(const T&) {
        expect(eq(AddConst<T>().processOne(T(4)), T(5)));
        auto blk = AddConst<T>(gr::property_map{{"value", T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(7)));
    } | kArithmeticTypes;

    "SubtractConst"_test = []<typename T>(const T&) {
        expect(eq(SubtractConst<T>().processOne(T(4)), T(3)));
        auto blk = SubtractConst<T>(gr::property_map{{"value", T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(1)));
    } | kArithmeticTypes;

    "MultiplyConst"_test = []<typename T>(const T&) {
        expect(eq(MultiplyConst<T>().processOne(T(4)), T(4)));
        auto blk = MultiplyConst<T>(gr::property_map{{"value", T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(12)));
    } | kArithmeticTypes;

    "DivideConst"_test = []<typename T>(const T&) {
        expect(eq(DivideConst<T>().processOne(T(4)), T(4)));
        auto blk = DivideConst<T>(gr::property_map{{"value", T(2)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(2)));
    } | kArithmeticTypes;

    /* ---------- Max / Min ----------------------------------------- */
    "Max"_test = []<typename T>(const T&) {
        using namespace gr; // for property_map
        test_block_process_bulk<T, Max<T>>({.inputs = {{1, 5, 2}, {3, 4, 7}}, .output = {3, 5, 7}});
    } | kArithmeticTypes;

    "Min"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_bulk<T, Min<T>>({.inputs = {{1, 5, 2}, {3, 4, 7}}, .output = {1, 4, 2}});
    } | kArithmeticTypes;

    /* ---------- bit-wise ops -------------------------------------- */
    "And"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_bulk<T, And<T>>({.inputs = {{0b1010}, {0b1100}}, .output = {0b1000}});
    } | kLogicTypes;

    "Or"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_bulk<T, Or<T>>({.inputs = {{0b1010}, {0b1100}}, .output = {0b1110}});
    } | kLogicTypes;

    "Xor"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_bulk<T, Xor<T>>({.inputs = {{0b1010}, {0b1100}}, .output = {0b0110}});
    } | kLogicTypes;

    /* ---------- unary ops ----------------------------------------- */
    "Negate"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_one<T, Negate<T>>({.input = {T(5), T(-3)}, .output = {T(-5), T(3)}});
    } | kArithmeticTypes;

    "Not"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block_process_one<T, Not<T>>({.input = {T(0b0101)}, .output = {T(~0b0101)}});
    } | kLogicTypes;

    "Abs"_test = []<typename T>(const T&) {
        using namespace gr;
        TestParameters<T> p;
        if constexpr (std::is_signed_v<T>) {
            p.input  = {T(-4), T(3)};
            p.output = {T(4), T(3)};
        } else {
            p.input = p.output = {T(4), T(3)};
        }
        test_block_process_one<T, Abs<T>>(p);
    } | kArithmeticTypes;

    /* ---------- Integrate ----------------------------------------- */
    "Integrate"_test = []<typename T>(const T&) {
        using namespace gr;
        using namespace gr::testing;
        using gr::blocks::math::Integrate;

        auto run_case = [](std::vector<T> in, Size_t decim, std::vector<T> expected) {
            Graph g;
            auto& integ = g.emplaceBlock<Integrate<T>>(property_map{{"decim", decim}});
            auto& src   = g.emplaceBlock<TagSource<T>>(property_map{{"values", in}, {"n_samples_max", static_cast<Size_t>(in.size())}});
            auto& sink  = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();

            expect(eq(g.connect(src, "out"s, integ, "in"s), ConnectionResult::SUCCESS));
            expect(eq(g.connect<"out">(integ).template to<"in">(sink), ConnectionResult::SUCCESS));

            scheduler::Simple sch{std::move(g)};
            expect(sch.runAndWait().has_value());
            expect(std::ranges::equal(sink._samples, expected));
        };

        run_case({T(1), T(2), T(3), T(4)}, 4, {T(10)});
        run_case({T(1), T(2), T(3), T(4), T(5)}, 2, {T(3), T(7)});
    } | kLimitedTypes;

    /* ---------- Argmax -------------------------------------------- */
    "Argmax"_test = []<typename T>(const T&) {
        using namespace gr;
        using namespace gr::testing;
        using gr::blocks::math::Argmax;

        Graph g;
        auto& arg  = g.emplaceBlock<Argmax<T>>(property_map{{"vlen", static_cast<Size_t>(3)}});
        auto& src  = g.emplaceBlock<TagSource<T>>(property_map{{"values", std::vector<T>{T(1), T(9), T(3), T(4), T(5), T(6)}}, {"n_samples_max", static_cast<Size_t>(6)}});
        auto& sink = g.emplaceBlock<TagSink<Size_t, ProcessFunction::USE_PROCESS_ONE>>();

        expect(eq(g.connect(src, "out"s, arg, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(arg).template to<"in">(sink), ConnectionResult::SUCCESS));

        scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(std::ranges::equal(sink._samples, std::vector<Size_t>{1U, 2U}));
    } | kLimitedTypes;

    /* ---------- Log10 --------------------------------------------- */
    "Log10"_test = []<typename FP>(const FP&) {
        using namespace gr;
        using gr::blocks::math::Log10;

        test_block_process_one<FP, Log10<FP>>({.input = {FP(1.0), FP(10.0)}, .output = {FP(0.0), FP(10.0)}});

        auto blk = Log10<FP>(property_map{{"n", FP(20)}, {"k", FP(-10)}});
        expect(eq(blk.processOne(FP(10)), FP(10)));
    } | std::tuple<float, double>();
};

int main() {} // not used by Boost.UT
