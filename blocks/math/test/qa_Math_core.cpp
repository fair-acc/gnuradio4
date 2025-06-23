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
void test_block(const TestParameters<T>& p)
{
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;

    Graph g;
    auto& sink = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();

    if (!p.input.empty()) {
        /* unary path --------------------------------------------------- */
        auto& blk = g.emplaceBlock<BlockUnderTest>();
        auto& src = g.emplaceBlock<TagSource<T>>(
            property_map{{"values",        p.input},
                         {"n_samples_max", static_cast<Size_t>(p.input.size())}});
        expect(eq(g.connect(src,"out"s,blk,"in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(blk).template to<"in">(sink),
           ConnectionResult::SUCCESS));

    } else {
        /* n-input path -------------------------------------------------- */
        const Size_t n_in = static_cast<Size_t>(p.inputs.size());
        auto& blk = g.emplaceBlock<BlockUnderTest>(property_map{{"n_inputs",n_in}});
        for (Size_t i=0;i<n_in;++i) {
            auto& src = g.emplaceBlock<TagSource<T>>(
                property_map{{"values",        p.inputs[i]},
                             {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
            expect(eq(g.connect(src,"out"s,blk,"in#"s+std::to_string(i)),
                      ConnectionResult::SUCCESS));
        }
        expect(eq(g.connect<"out">(blk).template to<"in">(sink),
           ConnectionResult::SUCCESS));
    }

    scheduler::Simple sch{std::move(g)};
    expect( sch.runAndWait().has_value() );
    expect( std::ranges::equal(sink._samples, p.output) );
}

const boost::ut::suite<"core math blocks"> suite_core = []{
    using namespace boost::ut;
    using namespace gr::blocks::math;

    constexpr auto kArithmeticTypes =
        std::tuple<uint8_t,uint16_t,uint32_t,uint64_t,
                   int8_t,int16_t,int32_t,int64_t,
                   float,double>();

    /* ---------------------------------------------------------------- */
    "Add"_test = []<typename T>(const T&){
        test_block<T,Add<T>>({ .inputs={{1,2,3}}, .output={1,2,3} });
        test_block<T,Add<T>>({ .inputs={{1,2},{3,4}}, .output={4,6} });
    } | kArithmeticTypes;

    "Subtract"_test = []<typename T>(const T&){
        test_block<T,Subtract<T>>({ .inputs={{5,4}}, .output={5,4} });
        test_block<T,Subtract<T>>({ .inputs={{5,4},{3,1}}, .output={2,3} });
    } | kArithmeticTypes;

    "Multiply"_test = []<typename T>(const T&){
        test_block<T,Multiply<T>>({ .inputs={{2,3}}, .output={2,3} });
        test_block<T,Multiply<T>>({ .inputs={{2,3},{4,5}}, .output={8,15} });
    } | kArithmeticTypes;

    "Divide"_test = []<typename T>(const T&){
        test_block<T,Divide<T>>({ .inputs={{8,6}}, .output={8,6} });
        test_block<T,Divide<T>>({ .inputs={{8,6},{2,3}}, .output={4,2} });
    } | kArithmeticTypes;

    /* ----- *Const variants ------------------------------------------ */
    "AddConst"_test = []<typename T>(const T&){
        expect(eq(AddConst<T>().processOne(T(4)), T(5)));
        auto blk = AddConst<T>(gr::property_map{{"value",T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(7)));
    } | kArithmeticTypes;

    "SubtractConst"_test = []<typename T>(const T&){
        expect(eq(SubtractConst<T>().processOne(T(4)), T(3)));
        auto blk = SubtractConst<T>(gr::property_map{{"value",T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(1)));
    } | kArithmeticTypes;

    "MultiplyConst"_test = []<typename T>(const T&){
        expect(eq(MultiplyConst<T>().processOne(T(4)), T(4)));
        auto blk = MultiplyConst<T>(gr::property_map{{"value",T(3)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(12)));
    } | kArithmeticTypes;

    "DivideConst"_test = []<typename T>(const T&){
        expect(eq(DivideConst<T>().processOne(T(4)), T(4)));
        auto blk = DivideConst<T>(gr::property_map{{"value",T(2)}});
        blk.init(blk.progress);
        expect(eq(blk.processOne(T(4)), T(2)));
    } | kArithmeticTypes;
};

int main() {}   // not used by Boost.UT
