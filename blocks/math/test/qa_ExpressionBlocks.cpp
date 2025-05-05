#include <boost/ut.hpp>

#include <gnuradio-4.0/math/ExpressionBlocks.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::blocks::math {
static_assert(std::is_constructible_v<ExpressionSISO<float>, property_map>, "Block type ExpressionSISO must be constructible from property_map");
static_assert(std::is_constructible_v<ExpressionDISO<float>, property_map>, "Block type ExpressionDISO must be constructible from property_map");
static_assert(std::is_constructible_v<ExpressionBulk<float>, property_map>, "Block type ExpressionBulk must be constructible from property_map");
} // namespace gr::blocks::math

const boost::ut::suite<"basic expression block tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::math;
    using testing::ProcessFunction::USE_PROCESS_ONE;

    "ExpressionSISO"_test = []<typename T>(const T&) {
        Graph graph;

        auto& source    = graph.emplaceBlock<testing::ConstantSource<T>>({{"n_samples_max", 10U}, {"default_value", T(21)}});
        auto& exprBlock = graph.emplaceBlock<ExpressionSISO<T>>({{"expr_string", "a*x"}, {"param_a", T(2)}});
        auto& tagSink   = graph.emplaceBlock<testing::TagSink<T, USE_PROCESS_ONE>>({{"log_tags", true}, {"log_samples", true}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source).template to<"in">(exprBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(exprBlock).template to<"in">(tagSink)));

        auto sched = gr::scheduler::Simple<>(std::move(graph));
        expect(sched.runAndWait().has_value());

        expect(approx(source.default_value, T(21), T(1e-3f)));
        expect(approx(exprBlock.param_a.value, T(2), T(1e-3f)));
        expect(eq(tagSink._samples.size(), source.n_samples_max));
        expect(approx(tagSink._samples[0], T(42), T(1e-3f)));
    } | std::tuple<float, double>{};

    "ExpressionDISO"_test = []<typename T>(const T&) {
        Graph graph;

        auto& source1   = graph.emplaceBlock<testing::ConstantSource<T>>({{"n_samples_max", 10U}, {"default_value", T(7)}});
        auto& source2   = graph.emplaceBlock<testing::ConstantSource<T>>({{"n_samples_max", 10U}, {"default_value", T(5)}});
        auto& exprBlock = graph.emplaceBlock<ExpressionDISO<T>>({{"expr_string", "z := a * (x + y + 2)"}, {"param_a", T(3)}});
        auto& tagSink   = graph.emplaceBlock<testing::TagSink<T, USE_PROCESS_ONE>>({{"log_tags", true}, {"log_samples", true}});
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source1).template to<"in0">(exprBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source2).template to<"in1">(exprBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(exprBlock).template to<"in">(tagSink)));

        auto sched = gr::scheduler::Simple<>(std::move(graph));
        expect(sched.runAndWait().has_value());

        expect(approx(source1.default_value, T(7), T(1e-3f)));
        expect(approx(source2.default_value, T(5), T(1e-3f)));
        expect(approx(exprBlock.param_a.value, T(3), T(1e-3f)));
        expect(eq(tagSink._samples.size(), source1.n_samples_max));
        expect(eq(tagSink._samples.size(), source2.n_samples_max));
        expect(approx(tagSink._samples[0], T(42), T(1e-3f)));
    } | std::tuple<float, double>{};

    "ExpressionBulk"_test = []<typename T>(const T&) {
        Graph graph;

        auto& source    = graph.emplaceBlock<testing::CountingSource<T>>({{"n_samples_max", 10U}});
        auto& exprBlock = graph.emplaceBlock<ExpressionBulk<T>>({{"expr_string", "vecOut := a * vecIn"}, {"param_a", T(2)}});
        auto& tagSink   = graph.emplaceBlock<testing::TagSink<T, USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source).template to<"in">(exprBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(exprBlock).template to<"in">(tagSink)));

        auto sched = gr::scheduler::Simple<>(std::move(graph));
        expect(sched.runAndWait().has_value());

        expect(eq(tagSink._samples.size(), source.n_samples_max));
        expect(approx(exprBlock.param_a.value, T(2), T(1e-3f)));
        for (std::size_t i = 0; i < tagSink._samples.size(); ++i) {
            T expected = T(1 + i) * T(2);
            expect(approx(tagSink._samples[i], expected, T(1e-6))) << std::format("should be: output[{}] ({}) == {} * input[{}] ({}) ", i, tagSink._samples[i], exprBlock.param_a, i, expected);
        }
    } | std::tuple<float, double>{};

    "ExpressionBulk - exceptions"_test = [](const bool enableERuntimeChecks) {
        Graph graph;

        auto&             source    = graph.emplaceBlock<testing::CountingSource<float>>({{"n_samples_max", 10U}});
        const std::string exprStr   = R""(for (var i := 0; i < 100; i += 1) {
   vecOut[i] := vecIn[i+1];
}
)"";
        auto&             exprBlock = graph.emplaceBlock<ExpressionBulk<float>>({{"expr_string", exprStr}, {"runtime_checks", enableERuntimeChecks}});
        auto&             tagSink   = graph.emplaceBlock<testing::TagSink<float, USE_PROCESS_ONE>>({{"log_samples", true}});

        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source).template to<"in">(exprBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(exprBlock).template to<"in">(tagSink)));

        auto sched = gr::scheduler::Simple<>(std::move(graph));

        try {
            expect(sched.runAndWait().has_value());
            expect(false) << std::format("should have failed");
        } catch (const gr::exception& ex) {
            expect(true);
            std::println("failed correctly with:\n{}\n", ex);
        } catch (...) {
            expect(false) << std::format("caught unknown/unexpected exception");
        }
    } | std::vector{true /*, false -- disabled on purpose as this would trigger correctly trigger the ASAN checks*/};
};

int main() { /* not needed for UT */ }
