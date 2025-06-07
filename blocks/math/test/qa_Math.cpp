#include <boost/ut.hpp>

#include <gnuradio-4.0/math/Math.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

template<typename T>
struct TestParameters {
    std::vector<T> input{};  // default value
    std::vector<std::vector<T>> inputs{};
    std::vector<T> output{};
};

template<typename T, typename BlockUnderTest>
void test_block(const TestParameters<T> p) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace gr::blocks::math;
    const Size_t n_inputs = static_cast<Size_t>(p.inputs.size());

    // build test graph
    Graph graph;

    auto& sink = graph.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();
    
    if (p.input.size() > 0) {

        // single input
        auto& block = graph.emplaceBlock<BlockUnderTest>();
        auto& src   = graph.emplaceBlock<TagSource<T>>({{"values", p.input}, {"n_samples_max", static_cast<Size_t>(p.input.size())}});
        expect(eq(graph.connect(src, "out"s, block, "in"s), ConnectionResult::SUCCESS)) << std::format("Failed to connect output port of src to input port of block");
        expect(eq(graph.connect<"out">(block).template to<"in">(sink), ConnectionResult::SUCCESS)) << "Failed to connect output port 'out' of block to input port of sink";

    } else {

        // multiple inputs (1 or more)
        auto& block = graph.emplaceBlock<BlockUnderTest>({{"n_inputs", n_inputs}});
        for (Size_t i = 0; i < n_inputs; ++i) {
            auto& src = graph.emplaceBlock<TagSource<T>>({{"values", p.inputs[i]}, {"n_samples_max", static_cast<Size_t>(p.inputs[i].size())}});
            expect(eq(graph.connect(src, "out"s, block, "in#"s + std::to_string(i)), ConnectionResult::SUCCESS)) << std::format("Failed to connect output port of src {} to input port 'in#{}' of block", i, i);
        }
        expect(eq(graph.connect<"out">(block).template to<"in">(sink), ConnectionResult::SUCCESS)) << "Failed to connect output port 'out' of block to input port of sink";
    }

    // execute and confirm result
    gr::scheduler::Simple scheduler{std::move(graph)};
    expect(scheduler.runAndWait().has_value()) << "Failed to run graph: No value";
    expect(std::ranges::equal(sink._samples, p.output)) << std::format("Failed to validate block output: Expected {} but got {} for input {}", p.output, sink._samples, p.inputs);
};

const boost::ut::suite<"basic math tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    constexpr auto kLogicalTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>();
    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float,
                                                 double /*, gr::UncertainValue<float>, gr::UncertainValue<double>,
std::complex<float>, std::complex<double>*/>();

    // clang-format off

    "Add"_test = []<typename T>(const T&) {
        test_block<T, Add<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Add<T>>({
            .inputs = {{1, 2,  3, T( 4.2)},
                       {5, 6,  7, T( 8.3)}},
            .output = { 6, 8, 10, T(12.5)}
        });
        test_block<T, Add<T>>({
            .inputs = {{12, 35, 18, 17},
                       {31, 15, 27, 36},
                       {83, 46, 37, 41}},
            .output = {126, 96, 82, 94}
        });
    } | kArithmeticTypes;

    "Subtract"_test = []<typename T>(const T&) {
        test_block<T, Subtract<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Subtract<T>>({
            .inputs = {{9, 7, 5, T(3.5)},
                       {3, 2, 0, T(1.2)}},
            .output = { 6, 5, 5, T(2.3)}});
        test_block<T, Subtract<T>>({
            .inputs = {{15, 38, 88, 29},
                       { 3, 12, 26, 18},
                       { 0, 10, 50,  7}},
            .output = { 12, 16, 12,  4}});
    } | kArithmeticTypes;

    "Multiply"_test = []<typename T>(const T&) {
        test_block<T, Multiply<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Multiply<T>>({
            .inputs = {{1,  2,  3, T( 4.0)},
                       {4,  5,  6, T( 7.1)}},
            .output = { 4, 10, 18, T(28.4)}});
        test_block<T, Multiply<T>>({
            .inputs = {{0,  1,   2,  3},
                       {4,  5,   6,  2},
                       {8,  9,  10, 11}},
            .output = { 0, 45, 120, 66}});
    } | kArithmeticTypes;

    "Divide"_test = []<typename T>(const T&) {
        test_block<T, Divide<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Divide<T>>({
            .inputs = {{9, 4, 5, T(7.0)},
                       {3, 4, 1, T(2.0)}},
            .output = { 3, 1, 5, T(3.5)}});
        test_block<T, Divide<T>>({
            .inputs = {{0, 10, 40, 80},
                       {1,  2,  4, 20},
                       {1,  5,  5,  2}},
            .output = { 0,  1,  2,  2}});
    } | kArithmeticTypes;

    "Max"_test = []<typename T>(const T&) {
        test_block<T, Max<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Max<T>>({
            .inputs = {{9, 4, 5, T(7.0)},
                       {3, 4, 1, T(2.0)}},
            .output = { 9, 4, 5, T(7.0)}});
        test_block<T, Max<T>>({
            .inputs = {{0, 10, 40, 80},
                       {1,  2,  4, 20},
                       {1,  5,  5,  2}},
            .output = { 1, 10, 40, 80}});
    } | kArithmeticTypes;

    "Min"_test = []<typename T>(const T&) {
        test_block<T, Min<T>>({
            .inputs = {{1, 2, 8, 17}},
            .output = { 1, 2, 8, 17}
        });
        test_block<T, Min<T>>({
            .inputs = {{9, 4, 5, T(7.0)},
                       {3, 4, 1, T(2.0)}},
            .output = { 3, 4, 1, T(2.0)}});
        test_block<T, Min<T>>({
            .inputs = {{0, 10, 40, 80},
                       {1,  2,  4, 20},
                       {1,  5,  5,  2}},
            .output = { 0,  2,  4,  2}});
    } | kArithmeticTypes;

    "And"_test = []<typename T>(const T&) {
        test_block<T, And<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110}},
            .output = { 0b0000, 0b0101, 0b1011, 0b1110}
        });
        test_block<T, And<T>>({        
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110}},
            .output = { 0b0000, 0b0100, 0b0011, 0b0110}});
        test_block<T, And<T>>({        
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110},
                       {0b1010, 0b1011, 0b1111, 0b1100}},
            .output = { 0b0000, 0b0000, 0b0011, 0b0100}});
    } | kLogicalTypes;

    "Or"_test = []<typename T>(const T&) {
        test_block<T, Or<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110}},
            .output = { 0b0000, 0b0101, 0b1011, 0b1110}
        });
        test_block<T, Or<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110}},
            .output = { 0b0010, 0b1101, 0b1011, 0b1110}});
        test_block<T, Or<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110},
                       {0b1010, 0b1011, 0b1111, 0b1100}},
            .output = { 0b1010, 0b1111, 0b1111, 0b1110}});
    } | kLogicalTypes;

    "Xor"_test = []<typename T>(const T&) {
        test_block<T, Xor<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110}},
            .output = { 0b0000, 0b0101, 0b1011, 0b1110}
        });
        test_block<T, Xor<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110}},
            .output = { 0b0010, 0b1001, 0b1000, 0b1000}});
        test_block<T, Xor<T>>({
            .inputs = {{0b0000, 0b0101, 0b1011, 0b1110},
                       {0b0010, 0b1100, 0b0011, 0b0110},
                       {0b1010, 0b1011, 0b1111, 0b1100}},
            .output = { 0b1000, 0b0010, 0b0111, 0b0100}});
    } | kLogicalTypes;

    "Negate"_test = []<typename T>(const T&) {
        test_block<T, Negate<T>>({
            .input  = {   1,     2,     8,     17 },
            .output = {T(-1), T(-2), T(-8), T(-17)}
        });
    } | kArithmeticTypes;

    "Not"_test = []<typename T>(const T&) {
        test_block<T, Not<T>>({
            .input  = {   0b0000,     0b0101,     0b1011,     0b1110},
            .output = {T(~0b0000), T(~0b0101), T(~0b1011), T(~0b1110)}
        });
    } | kLogicalTypes;

    "Abs"_test = []<typename T>(const T&) {
        TestParameters<T> p{};
        if constexpr (std::is_signed_v<T>) {
            p.input  = { T(-5),  T(4),  T(0),  T(-1) };
            p.output = { T( 5),  T(4),  T(0),  T( 1) };
        } else {
            // unsigned – abs(x) == x
            p.input  = { T(5), T(4), T(0), T(1) };
            p.output = p.input;
        }
        test_block<T, Abs<T>>(p);
    } | kArithmeticTypes;

    // clang-format on

    "AddConst"_test = []<typename T>(const T&) {
        expect(eq(AddConst<T>().processOne(T(4)), T(4) + T(1))) << std::format("AddConst test for type {}\n", meta::type_name<T>());
        auto block = AddConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) + T(2))) << std::format("AddConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "SubtractConst"_test = []<typename T>(const T&) {
        expect(eq(SubtractConst<T>().processOne(T(4)), T(4) - T(1))) << std::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = SubtractConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) - T(2))) << std::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "MultiplyConst"_test = []<typename T>(const T&) {
        expect(eq(MultiplyConst<T>().processOne(T(4)), T(4) * T(1))) << std::format("MultiplyConst test for type {}\n", meta::type_name<T>());
        auto block = MultiplyConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) * T(2))) << std::format("MultiplyConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

    "DivideConst"_test = []<typename T>(const T&) {
        expect(eq(DivideConst<T>().processOne(T(4)), T(4) / T(1))) << std::format("SubtractConst test for type {}\n", meta::type_name<T>());
        auto block = DivideConst<T>(property_map{{"value", T(2)}});
        block.init(block.progress, block.ioThreadPool);
        expect(eq(block.processOne(T(4)), T(4) / T(2))) << std::format("SubtractConst(2) test for type {}\n", meta::type_name<T>());
    } | kArithmeticTypes;

        /* ------------------------------------------------------------------ */
    "Integrate"_test = []<typename T>(const T&) {
            using namespace boost::ut;
            using namespace gr;
            using gr::blocks::math::Integrate;   // template alias from Math.hpp
    
            /* ---------- helper to run one integrate scenario ------------- */
            auto run_case = [] (std::vector<T> in,
                                Size_t        decim,
                                std::vector<T> expected)
            {
                using namespace gr::testing;
                Graph g;
    
                // block under test with property "decim"
                auto& integ = g.emplaceBlock<Integrate<T>>({ { "decim", decim } });
    
                // src / sink
                auto& src = g.emplaceBlock<TagSource<T>>(
                                        { { "values",        in },
                                          { "n_samples_max", static_cast<gr::Size_t>(in.size()) } } );
    
                auto& sink = g.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_ONE>>();
    
                expect( eq(g.connect(src , "out"s,  integ, "in"s ), ConnectionResult::SUCCESS) );
                expect( eq(g.connect<"out">(integ).template to<"in">(sink), ConnectionResult::SUCCESS) );
    
                // run
                scheduler::Simple sch{ std::move(g) };
                expect( sch.runAndWait().has_value() ) << "scheduler failed";
    
                expect( std::ranges::equal(sink._samples, expected) )
                    << std::format("Integrate (decim {}) expected {}, got {}",
                                   decim, expected, sink._samples);
            };
    
            /* ---------- Test-case #1  (decim = 4) ------------------------ */
            {
                std::vector<T> in{ T(1),T(2),T(3),T(4),  T(5),T(6),T(7),T(8) };
                std::vector<T> out{ T(10), T(26) };      // 1+2+3+4 , 5+6+7+8
                run_case(std::move(in), 4, std::move(out));
            }
    
            /* ---------- Test-case #2  (decim = 3, uneven tail) ----------- */
            {
                std::vector<T> in{ T(1),T(2),T(3),  T(4),T(5),T(6),  T(7) };
                std::vector<T> out{ T(6), T(15) };       // last sample (7) dropped
                run_case(std::move(in), 3, std::move(out));
            }
    
            /* ---------- Test-case #3  (decim = 1 => pass-through) -------- */
            {
                std::vector<T> in{ T(9),T(8),T(7) };
                run_case(in, 1, in);                     // expect identical stream
            }
        } | kArithmeticTypes;

            /* ------------------------------------------------------------------ */
    "Argmax"_test = []<typename T>(const T&) {
        using namespace gr;
        using namespace gr::testing;
        using gr::blocks::math::Argmax;

        /* helper ------------------------------------------------------- */
        auto run_case = [] (std::vector<T>  in,
                            gr::Size_t      vlen,
                            std::vector<gr::Size_t> expected)
        {
            Graph g;

            /* block under test */
            auto& arg = g.emplaceBlock<Argmax<T>>( property_map{ { "vlen", vlen } } );

            /* source & sink                                              */
            auto& src = g.emplaceBlock<TagSource<T>>( property_map{
                               { "values",        in },
                               { "n_samples_max", static_cast<gr::Size_t>(in.size()) } } );

            auto& sink = g.emplaceBlock<
                             TagSink<gr::Size_t, ProcessFunction::USE_PROCESS_ONE>>();

            /* wire it up                                                 */
            expect( eq(g.connect(src, "out"s, arg, "in"s),
                       ConnectionResult::SUCCESS) );

            expect( eq(g.connect<"out">(arg).template to<"in">(sink),
                       ConnectionResult::SUCCESS) );

            /* run & check                                                */
            scheduler::Simple sch{ std::move(g) };
            expect( sch.runAndWait().has_value() ) << "scheduler failed";

            expect( std::ranges::equal(sink._samples, expected) )
                << std::format("Argmax(vlen {}) expected {}, got {}",
                               vlen, expected, sink._samples);
        };

        /* ---------- Test-case #1  vlen = 4 --------------------------- *
         *   vectors: [1 2 8 3]  →  idx 2                               *
         *             [0 1 0 2]  →  idx 3                              */
        run_case({ T(1),T(2),T(8),T(3),   T(0),T(1),T(0),T(2) },
                 4, { 2U, 3U });

        /* ---------- Test-case #2  equal maxima ----------------------- */
        run_case({ T(5),T(5),T(1) }, 3, { 0U });   // first max wins

        /* ---------- Test-case #3  vlen = 2, several vectors ---------- */
        run_case({ T(7),T(8),  T(9),T(6),  T(10),T(2) },
                 2, { 1U, 0U, 0U });                // three results
    } | kArithmeticTypes;    
    
};

int main() { /* not needed for UT */ }
