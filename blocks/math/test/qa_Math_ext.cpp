/*
 * qa_Math_ext.cpp  ── “extended” QA for math blocks.   Split out of the
 * original qa_Math.cpp so that large CI jobs can be parallelised.
 */

#include <boost/ut.hpp>

// We avoid <format> because wasm/libc++ on CI doesn’t ship it.
#include <gnuradio-4.0/math/Math.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <limits>
#include <cmath>

/* ------------------------------------------------------------------ */
/*                         shared test helpers                         */
/* ------------------------------------------------------------------ */

template<typename T>
struct TestParameters {
    std::vector<T>              input{};
    std::vector<std::vector<T>> inputs{};
    std::vector<T>              output{};
};

template<typename T, typename BlockUnderTest>
void test_block(const TestParameters<T>& p)
{
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;          // TagSource / TagSink definitions
    using namespace gr::blocks::math;

    const Size_t n_inputs = static_cast<Size_t>(p.inputs.size());

    Graph g;
    auto& sink = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();

    if (!p.input.empty()) {
        /* single input */
        auto& blk = g.emplaceBlock<BlockUnderTest>();
        auto& src = g.emplaceBlock<TagSource<T>>(gr::property_map{
                       {"values",        p.input},
                       {"n_samples_max", static_cast<Size_t>(p.input.size())}});

        expect(eq(g.connect(src, "out"s, blk, "in"s),
                  ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(blk).template to<"in">(sink),
                  ConnectionResult::SUCCESS));
    } else {
        /* multiple inputs */
        auto& blk = g.emplaceBlock<BlockUnderTest>(
                        gr::property_map{{"n_inputs", n_inputs}});

        for (Size_t i = 0; i < n_inputs; ++i) {
            auto& src = g.emplaceBlock<TagSource<T>>(gr::property_map{
                           {"values",        p.inputs[i]},
                           {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
            expect(eq(g.connect(src, "out"s, blk, "in#"s + std::to_string(i)),
                      ConnectionResult::SUCCESS));
        }

        expect(eq(g.connect<"out">(blk).template to<"in">(sink),
                  ConnectionResult::SUCCESS));
    }

    scheduler::Simple sch{std::move(g)};
    expect(sch.runAndWait().has_value());
    expect(std::ranges::equal(sink._samples, p.output));
}

/* ------------------------------------------------------------------ */
/*                                tests                                */
/* ------------------------------------------------------------------ */

const boost::ut::suite<"math-ext"> math_ext = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;

    constexpr auto kArith = std::tuple<
        uint8_t,uint16_t,uint32_t,uint64_t,
        int8_t,int16_t,int32_t,int64_t,
        float,double>();

    constexpr auto kLogic = std::tuple<
        uint8_t,uint16_t,uint32_t,uint64_t,
        int8_t,int16_t,int32_t,int64_t>();

    /* ---------- Max / Min ----------------------------------------- */
    "Max"_test = []<typename T>(const T&) {
        using namespace gr;               // for property_map
        test_block<T, Max<T>>({
            .inputs = {{1,5,2},
                       {3,4,7}},
            .output = {3,5,7}});
    } | kArith;

    "Min"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, Min<T>>({
            .inputs = {{1,5,2},
                       {3,4,7}},
            .output = {1,4,2}});
    } | kArith;

    /* ---------- bit-wise ops -------------------------------------- */
    "And"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, And<T>>({
            .inputs = {{0b1010}, {0b1100}},
            .output = {0b1000}});
    } | kLogic;

    "Or"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, Or<T>>({
            .inputs = {{0b1010}, {0b1100}},
            .output = {0b1110}});
    } | kLogic;

    "Xor"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, Xor<T>>({
            .inputs = {{0b1010}, {0b1100}},
            .output = {0b0110}});
    } | kLogic;

    /* ---------- unary ops ----------------------------------------- */
    "Negate"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, Negate<T>>({
            .input  = { T( 5), T(-3) },
            .output = { T(-5), T( 3) }});
    } | kArith;

    "Not"_test = []<typename T>(const T&) {
        using namespace gr;
        test_block<T, Not<T>>({
            .input  = { T(0b0101) },
            .output = { T(~0b0101) }});
    } | kLogic;

    "Abs"_test = []<typename T>(const T&) {
        using namespace gr;
        TestParameters<T> p;
        if constexpr (std::is_signed_v<T>) {
            p.input  = { T(-4), T(3) };
            p.output = { T( 4), T(3) };
        } else {
            p.input = p.output = { T(4), T(3) };
        }
        test_block<T, Abs<T>>(p);
    } | kArith;

    /* ---------- Integrate ----------------------------------------- */
    "Integrate"_test = []<typename T>(const T&) {
        using namespace gr;
        using namespace gr::testing;
        using gr::blocks::math::Integrate;

        auto run_case = [] (std::vector<T> in,
                            Size_t        decim,
                            std::vector<T> expected)
        {
            Graph g;
            auto& integ = g.emplaceBlock<Integrate<T>>(property_map{
                               {"decim", decim}});
            auto& src = g.emplaceBlock<TagSource<T>>(property_map{
                               {"values", in},
                               {"n_samples_max", static_cast<Size_t>(in.size())}});
            auto& sink = g.emplaceBlock<
                             TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();

            expect(eq(g.connect(src,"out"s,integ,"in"s),
                      ConnectionResult::SUCCESS));
            expect(eq(g.connect<"out">(integ).template to<"in">(sink),
                      ConnectionResult::SUCCESS));

            scheduler::Simple sch{std::move(g)};
            expect(sch.runAndWait().has_value());
            expect(std::ranges::equal(sink._samples, expected));
        };

        run_case({T(1),T(2),T(3),T(4)}, 4, {T(10)});
        run_case({T(1),T(2),T(3),T(4),T(5)}, 2, {T(3),T(7)});
    } | kArith;

    /* ---------- Argmax -------------------------------------------- */
    "Argmax"_test = []<typename T>(const T&) {
        using namespace gr;
        using namespace gr::testing;
        using gr::blocks::math::Argmax;

        Graph g;
        auto& arg  = g.emplaceBlock<Argmax<T>>(property_map{
                           {"vlen", static_cast<Size_t>(3)}});
        auto& src = g.emplaceBlock<TagSource<T>>(property_map{
        {"values", std::vector<T>{T(1), T(9), T(3), T(4), T(5), T(6)}},
        {"n_samples_max", static_cast<Size_t>(6)}});
        auto& sink = g.emplaceBlock<
                         TagSink<Size_t, ProcessFunction::USE_PROCESS_ONE>>();

        expect(eq(g.connect(src,"out"s,arg,"in"s),
                  ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(arg).template to<"in">(sink),
                  ConnectionResult::SUCCESS));

        scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(std::ranges::equal(sink._samples,
               std::vector<Size_t>{1U, 2U}));
    } | kArith;

    /* ---------- Log10 --------------------------------------------- */
    "Log10"_test = []<typename FP>(const FP&) {
        using namespace gr;
        using gr::blocks::math::Log10;

        test_block<FP, Log10<FP>>({
            .input  = { FP(1.0), FP(10.0) },
            .output = { FP(0.0), FP(10.0) }});

        auto blk = Log10<FP>(property_map{{"n",FP(20)}, {"k",FP(-10)}});
        expect(eq(blk.processOne(FP(10)), FP(10)));
    } | std::tuple<float,double>();
};

int main() {}
