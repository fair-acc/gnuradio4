#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/testing/NullSources.hpp>

const boost::ut::suite<"Null[..] and Testing Blocks"> nullSourcesTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    constexpr auto kTestTypes = std::tuple<uint8_t, int16_t, int32_t, float, double>(); // only a limited set of test cases, could be improved to use pre-built modules

    "NullSource->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 12;

        Graph g;
        auto& src  = g.emplaceBlock<NullSource<T>>();
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "CountingSource->NullSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N_total     = 7;
        constexpr T          start_value = T(3);

        Graph g;
        auto& src  = g.emplaceBlock<CountingSource<T>>(property_map{{"default_value", start_value}, {"n_samples_max", N_total}});
        auto& sink = g.emplaceBlock<NullSink<T>>();

        expect(eq(g.connect(src, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
    } | kTestTypes;

    "ConstantSource->Copy->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N     = 10;
        constexpr T          value = T(7);

        Graph g;
        auto& src  = g.emplaceBlock<ConstantSource<T>>(property_map{{"default_value", value}, {"n_samples_max", N}});
        auto& copy = g.emplaceBlock<Copy<T>>();
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, copy, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect(copy, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "CountingSource->HeadBlock->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N_total     = 10;
        constexpr gr::Size_t N_head      = 4;
        constexpr T          start_value = T(5);

        Graph g;
        auto& src  = g.emplaceBlock<CountingSource<T>>(property_map{{"default_value", start_value}, {"n_samples_max", N_total}});
        auto& head = g.emplaceBlock<HeadBlock<T>>(property_map{{"n_samples_max", N_head}});
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N_head}});

        expect(eq(g.connect(src, "out"s, head, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect(head, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N_head));
    } | kTestTypes;

    "ConstantSource->NullSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 5;

        Graph g;
        auto& src  = g.emplaceBlock<ConstantSource<T>>(property_map{{"default_value", T(99)}, {"n_samples_max", N}});
        auto& sink = g.emplaceBlock<NullSink<T>>();

        expect(eq(g.connect(src, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
    } | kTestTypes;

    "SlowSource->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 3;

        Graph g;
        auto& src  = g.emplaceBlock<SlowSource<T>>(property_map{{"default_value", T(77)}, {"delay", 10U}});
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "SimCompute(zero)->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 8;

        Graph g;
        auto& src  = g.emplaceBlock<ConstantSource<T>>(property_map{{"default_value", T(5)}, {"n_samples_max", N}});
        auto& sim  = g.emplaceBlock<SimCompute<T>>(property_map{{"complexity_order", 0.0f}, {"busy_wait", true}});
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, sim, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect(sim, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "SimCompute(linear)->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 8;

        Graph g;
        auto& src  = g.emplaceBlock<ConstantSource<T>>(property_map{{"default_value", T(5)}, {"n_samples_max", N}});
        auto& sim  = g.emplaceBlock<SimCompute<T>>(property_map{{"complexity_order", 1.0f}, {"busy_wait", true}});
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, sim, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect(sim, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "SimCompute(quadratic)->CountingSink"_test = []<typename T>(const T&) {
        constexpr gr::Size_t N = 6;

        Graph g;
        auto& src  = g.emplaceBlock<CountingSource<T>>(property_map{{"default_value", T(0)}, {"n_samples_max", N}});
        auto& sim  = g.emplaceBlock<SimCompute<T>>(property_map{{"complexity_order", 2.0f}, {"busy_wait", false}});
        auto& sink = g.emplaceBlock<CountingSink<T>>(property_map{{"n_samples_max", N}});

        expect(eq(g.connect(src, "out"s, sim, "in"s), ConnectionResult::SUCCESS));
        expect(eq(g.connect(sim, "out"s, sink, "in"s), ConnectionResult::SUCCESS));

        gr::scheduler::Simple sch{std::move(g)};
        expect(sch.runAndWait().has_value());
        expect(eq(sink.count, N));
    } | kTestTypes;

    "SimCompute.compute_delay_seconds"_test = [] {
        using namespace gr::testing;

        SimCompute<float> sim;
        sim.target_throughput   = 1e6f;  // 1 MS/s
        sim.reference_work_size = 1000U; // 1000 samples
        sim.complexity_order    = 2.0f;  // quadratic

        constexpr std::size_t N = 2000;

        const auto   delay        = sim.compute_delay_seconds(N);
        const double expected_sec = std::pow(double(N) / 1000.0, 2.0) * (1000.0 / 1e6);

        expect(approx(delay.count(), expected_sec, 1e-3));
    };
};

int main() { /* not needed for UT */ }
