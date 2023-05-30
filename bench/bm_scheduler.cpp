#include "benchmark.hpp"

#include <boost/ut.hpp>
#include <graph.hpp>
#include <scheduler.hpp>

#include "bm_test_helper.hpp"

namespace fg                           = fair::graph;

inline constexpr std::size_t N_ITER    = 10;
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000'000, 1024);

template<typename T, char op>
class math_op : public fg::node<math_op<T, op>, fg::IN<T, 0, N_MAX, "in">, fg::OUT<T, 0, N_MAX, "out">> {
    T _factor = static_cast<T>(1.0f);

public:
    math_op() = delete;

    explicit math_op(T factor, std::string name = fair::graph::this_source_location()) : _factor(factor) { this->set_name(name); }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(const V &a) const noexcept {
        if constexpr (op == '*') {
            return a * _factor;
        } else if constexpr (op == '/') {
            return a / _factor;
        } else if constexpr (op == '+') {
            return a + _factor;
        } else if constexpr (op == '-') {
            return a - _factor;
        } else {
            static_assert(fair::meta::always_false<T>, "unknown op");
        }
    }
};

template<typename T>
using multiply = math_op<T, '*'>;
template<typename T>
using divide = math_op<T, '/'>;

template<typename T, typename Sink, typename Source>
void create_cascade(fg::graph& flow_graph, Sink& src, Source& sink, std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;

    std::vector<multiply<T> *> mult1;
    std::vector<divide<T> *> mult2;
    for (std::size_t i = 0; i < depth; i++) {
        mult1.emplace_back(std::addressof(flow_graph.make_node<multiply<T>>(T(2), fmt::format("mult.{}", i))));
        mult2.emplace_back(std::addressof(flow_graph.make_node<divide<T>>(T(2), fmt::format("div.{}", i))));
    }

    for (std::size_t i = 0; i < mult1.size(); i++) {
        if (i == 0) {
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(src).template to<"in">(*mult1[i])));
        } else {
            expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult2[i - 1]).template to<"in">(*mult1[i])));
        }
        expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult1[i]).template to<"in">(*mult2[i])));
    }
    expect(eq(fg::connection_result_t::SUCCESS, flow_graph.connect<"out">(*mult2[mult2.size() - 1]).template to<"in">(sink)));
}

template<typename T>
fg::graph test_graph_linear(std::size_t depth = 1) {
    fg::graph flow_graph;

    auto &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
    auto &sink = flow_graph.make_node<test::sink<T>>();

    create_cascade<T>(flow_graph, src, sink, depth);

    return flow_graph;
}

template<typename T>
fg::graph test_graph_bifurcated(std::size_t depth = 1) {
    using namespace boost::ut;
    using namespace benchmark;
    fg::graph flow_graph;

    auto &src  = flow_graph.make_node<test::source<T>>(N_SAMPLES);
    auto &sink1 = flow_graph.make_node<test::sink<T>>();
    auto &sink2 = flow_graph.make_node<test::sink<T>>();

    create_cascade<T>(flow_graph, src, sink1, depth);
    create_cascade<T>(flow_graph, src, sink2, depth);

    return flow_graph;
}

void exec_bm(auto& scheduler, const std::string& test_case) {
    using namespace boost::ut;
    using namespace benchmark;
    test::n_samples_produced = 0LU;
    test::n_samples_consumed = 0LU;
    scheduler.work();
    expect(eq(test::n_samples_produced, N_SAMPLES)) << fmt::format("did not produce enough output samples for {}", test_case);
    expect(ge(test::n_samples_consumed, N_SAMPLES)) << fmt::format("did not consume enough input samples for {}", test_case);
}

[[maybe_unused]] inline const boost::ut::suite _scheduler = [] {
    using namespace boost::ut;
    using namespace benchmark;

        fg::scheduler::simple sched1(test_graph_linear<float>(10));
        "linear graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched1]() {
            exec_bm(sched1, "linear-graph simple-sched");
        };

        fg::scheduler::breadth_first sched2(test_graph_linear<float>(10));
        "linear graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched2]() {
            exec_bm(sched2, "linear-graph BFS-sched");
        };

        fg::scheduler::simple sched3(test_graph_bifurcated<float>(5));
        "bifurcated graph - simple scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched3]() {
             exec_bm(sched3, "bifurcated-graph simple-sched");
        };

        fg::scheduler::breadth_first sched4(test_graph_bifurcated<float>(5));
        "bifurcated graph - BFS scheduler"_benchmark.repeat<N_ITER>(N_SAMPLES) = [&sched4]() {
            exec_bm(sched4, "bifurcated-graph BFS-sched");
        };
};

int
main() { /* not needed by the UT framework */
}
