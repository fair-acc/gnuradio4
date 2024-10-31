#include <benchmark.hpp>

#include <algorithm>
#include <functional>

#include <vir/simd.h>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/scheduler.hpp>

#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/testing/bm_test_helper.hpp>

inline constexpr std::size_t N_ITER = 10;
// inline constexpr std::size_t N_SAMPLES = gr::util::round_up(1'000'000, 1024);
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000, 1024);

void loop_over_work(auto& node) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    while (test::n_samples_consumed < N_SAMPLES) {
        std::ignore = node.work(std::numeric_limits<std::size_t>::max());
    }
    expect(eq(test::n_samples_produced, N_SAMPLES)) << "produced too many/few samples";
    expect(eq(test::n_samples_consumed, N_SAMPLES)) << "consumed too many/few samples";
}

void invoke_work(auto& sched) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    sched.run_and_wait();
    expect(eq(test::n_samples_produced, N_SAMPLES)) << "did not produce enough output samples";
    expect(eq(test::n_samples_consumed, N_SAMPLES)) << "did not consume enough input samples";
}

inline const boost::ut::suite _constexpr_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    using gr::merge_by_index;
    using gr::merge;
    using namespace gr::blocks::filter;

    std::vector<float> fir_coeffs(10.f, 0.1f); // box car filter
    std::vector<float> iir_coeffs_b{0.55f, 0.f};
    std::vector<float> iir_coeffs_a{1.f, -0.45f};

    {
        auto mergedBlock = merge<"out", "in">(::test::source<float>(N_SAMPLES), ::test::sink<float>());
        //
        "merged src->sink work"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&mergedBlock]() { loop_over_work(mergedBlock); };
    }

    {
        fir_filter<float> filter;
        filter.b         = fir_coeffs;
        auto mergedBlock = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->fir_filter->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&mergedBlock]() { loop_over_work(mergedBlock); };
    }

    {
        iir_filter<float, IIRForm::DF_I> filter;
        filter.b         = iir_coeffs_b;
        filter.a         = iir_coeffs_a;
        auto mergedBlock = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->iir_filter->sink - direct form I"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&mergedBlock]() { loop_over_work(mergedBlock); };
    }

    {
        iir_filter<float, IIRForm::DF_II> filter;
        filter.b         = iir_coeffs_b;
        filter.a         = iir_coeffs_a;
        auto mergedBlock = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->iir_filter->sink - direct form II"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&mergedBlock]() { loop_over_work(mergedBlock); };
    }

    {
        gr::graph testGraph;
        auto&     src  = testGraph.emplaceBlock<::test::source<float>>(N_SAMPLES);
        auto&     sink = testGraph.emplaceBlock<::test::sink<float>>();

        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        gr::scheduler::simple sched{std::move(testGraph)};

        "runtime   src->sink overhead"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        gr::graph testGraph;
        auto&     src    = testGraph.emplaceBlock<::test::source<float>>(N_SAMPLES);
        auto&     sink   = testGraph.emplaceBlock<::test::sink<float>>();
        auto&     filter = testGraph.emplaceBlock<fir_filter<float>>({{"b", fir_coeffs}});

        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        gr::scheduler::simple sched{std::move(testGraph)};

        "runtime   src->fir_filter->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        gr::graph testGraph;
        auto&     src    = testGraph.emplaceBlock<::test::source<float>>(N_SAMPLES);
        auto&     sink   = testGraph.emplaceBlock<::test::sink<float>>();
        auto&     filter = testGraph.emplaceBlock<iir_filter<float, IIRForm::DF_I>>({{"b", iir_coeffs_b}, {"a", iir_coeffs_a}});

        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        gr::scheduler::simple sched{std::move(testGraph)};

        "runtime   src->iir_filter->sink - direct-form I"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        gr::graph testGraph;
        auto&     src    = testGraph.emplaceBlock<::test::source<float>>(N_SAMPLES);
        auto&     sink   = testGraph.emplaceBlock<::test::sink<float>>();
        auto&     filter = testGraph.emplaceBlock<iir_filter<float, IIRForm::DF_II>>({{"b", iir_coeffs_b}, {"a", iir_coeffs_a}});

        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        gr::scheduler::simple sched{std::move(testGraph)};

        "runtime   src->iir_filter->sink - direct-form II"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }
};

int main() { /* not needed by the UT framework */ }
