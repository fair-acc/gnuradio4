#include "benchmark.hpp"

#include <algorithm>
#include <boost/ut.hpp>
#include <functional>

#include "../test/blocklib/core/filter/time_domain_filter.hpp"
#include "bm_test_helper.hpp"
#include "scheduler.hpp"

#include <graph.hpp>
#include <node_traits.hpp>

#include <vir/simd.h>

inline constexpr std::size_t N_ITER = 10;
// inline constexpr std::size_t N_SAMPLES = gr::util::round_up(1'000'000, 1024);
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000, 1024);

void
loop_over_work(auto &node) {
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

void
invoke_work(auto &sched) {
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
    using fair::graph::merge_by_index;
    using fair::graph::merge;
    using namespace gr::blocks::filter;
    namespace fg = fair::graph;

    std::vector<float> fir_coeffs(10.f, 0.1f); // box car filter
    std::vector<float> iir_coeffs_b{ 0.55f, 0.f };
    std::vector<float> iir_coeffs_a{ 1.f, -0.45f };

    {
        auto merged_node = merge<"out", "in">(::test::source<float>(N_SAMPLES), ::test::sink<float>());
        //
        "merged src->sink work"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        fir_filter<float> filter;
        filter.b         = fir_coeffs;
        auto merged_node = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->fir_filter->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        iir_filter<float, IIRForm::DF_I> filter;
        filter.b         = iir_coeffs_b;
        filter.a         = iir_coeffs_a;
        auto merged_node = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->iir_filter->sink - direct form I"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        iir_filter<float, IIRForm::DF_II> filter;
        filter.b         = iir_coeffs_b;
        filter.a         = iir_coeffs_a;
        auto merged_node = merge<"out", "in">(merge<"out", "in">(::test::source<float>(N_SAMPLES), std::move(filter)), ::test::sink<float>());
        //
        "merged src->iir_filter->sink - direct form II"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&merged_node]() { loop_over_work(merged_node); };
    }

    {
        fg::graph flow_graph;
        auto     &src  = flow_graph.make_node<::test::source<float>>(N_SAMPLES);
        auto     &sink = flow_graph.make_node<::test::sink<float>>();

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->sink overhead"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src    = flow_graph.make_node<::test::source<float>>(N_SAMPLES);
        auto     &sink   = flow_graph.make_node<::test::sink<float>>();
        auto     &filter = flow_graph.make_node<fir_filter<float>>({ { "b", fir_coeffs } });

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->fir_filter->sink"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src    = flow_graph.make_node<::test::source<float>>(N_SAMPLES);
        auto     &sink   = flow_graph.make_node<::test::sink<float>>();
        auto     &filter = flow_graph.make_node<iir_filter<float, IIRForm::DF_I>>({ { "b", iir_coeffs_b }, { "a", iir_coeffs_a } });

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->iir_filter->sink - direct-form I"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }

    {
        fg::graph flow_graph;
        auto     &src    = flow_graph.make_node<::test::source<float>>(N_SAMPLES);
        auto     &sink   = flow_graph.make_node<::test::sink<float>>();
        auto     &filter = flow_graph.make_node<iir_filter<float, IIRForm::DF_II>>({ { "b", iir_coeffs_b }, { "a", iir_coeffs_a } });

        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect(src, &::test::source<float>::out).to<"in">(filter)));
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(filter).to(sink, &::test::sink<float>::in)));

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };

        "runtime   src->iir_filter->sink - direct-form II"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched]() { invoke_work(sched); };
    }
};

int
main() { /* not needed by the UT framework */
}
