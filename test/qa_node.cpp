#include <boost/ut.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

#include <fmt/format.h>
#include <graph.hpp>
#include <node.hpp>
#include <numbers>
#include <scheduler.hpp>

namespace fg = fair::graph;

struct ProcessStatus {
    std::size_t n_inputs{ 0 };
    std::size_t n_outputs{ 0 };
    std::size_t process_counter{ 0 };
};

struct TestData {
    std::size_t n_samples{};
    std::size_t numerator{};
    std::size_t denominator{};
    int         out_port_min_samples{ -1 }; // -1 for not used
    int         out_port_max_samples{ -1 }; // -1 for not used
    std::size_t expected_in{};
    std::size_t expected_out{};
    std::size_t expected_counter{};
};

template<typename T>
struct CountSource : public fg::node<CountSource<T>> {
    fg::OUT<T> out{};
    int        count{ 0 };
    int        n_samples{ 1024 };

    constexpr std::make_signed_t<std::size_t>
    available_samples(const CountSource & /*d*/) noexcept {
        const auto ret = n_samples - count;
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    constexpr T
    process_one() {
        return static_cast<T>(count++);
    }
};

template<typename T>
struct IntDecBlock : public fg::node<IntDecBlock<T>, fg::PerformDecimationInterpolation> {
    fg::IN<T>     in{};
    fg::OUT<T>    out{};

    ProcessStatus status;

    fg::work_return_status_t
    process_bulk(std::span<const T> input, std::span<T> output) noexcept {
        status.n_inputs  = input.size();
        status.n_outputs = output.size();
        status.process_counter++;

        return fg::work_return_status_t::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (CountSource<T>), out, count, n_samples);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (IntDecBlock<T>), in, out);

void
interpolation_decimation_test(const TestData &data, std::shared_ptr<fair::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using scheduler = fair::graph::scheduler::simple<>;

    fg::graph flow;
    auto     &source          = flow.make_node<CountSource<int>>();
    source.n_samples          = static_cast<int>(data.n_samples);

    auto &int_dec_block       = flow.make_node<IntDecBlock<int>>();
    int_dec_block.numerator   = data.numerator;
    int_dec_block.denominator = data.denominator;
    if (data.out_port_max_samples >= 0) int_dec_block.out.max_samples = static_cast<int>(data.out_port_max_samples);
    if (data.out_port_min_samples >= 0) int_dec_block.out.min_samples = static_cast<int>(data.out_port_min_samples);

    std::ignore = flow.connect<"out">(source).to<"in">(int_dec_block);
    auto sched  = scheduler(std::move(flow), thread_pool);
    sched.run_and_wait();

    expect(eq(int_dec_block.status.process_counter, data.expected_counter)) << "process_bulk invokes counter";
    expect(eq(int_dec_block.status.n_inputs, data.expected_in)) << "number of input samples";
    expect(eq(int_dec_block.status.n_outputs, data.expected_out)) << "number of output samples";
}

const boost::ut::suite _fft_tests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    auto thread_pool                      = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2);

    "Interpolation/Decimation tests"_test = [&thread_pool] {
        interpolation_decimation_test({ .n_samples{ 1024 }, .numerator{ 1 }, .denominator{ 1 }, .expected_in{ 1024 }, .expected_out{ 1024 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 1024 }, .numerator{ 1 }, .denominator{ 2 }, .expected_in{ 1024 }, .expected_out{ 512 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 1024 }, .numerator{ 2 }, .denominator{ 1 }, .expected_in{ 1024 }, .expected_out{ 2048 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 1000 }, .numerator{ 5 }, .denominator{ 6 }, .expected_in{ 996 }, .expected_out{ 830 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 549 }, .numerator{ 1 }, .denominator{ 50 }, .expected_in{ 500 }, .expected_out{ 10 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 100 }, .numerator{ 3 }, .denominator{ 7 }, .expected_in{ 98 }, .expected_out{ 42 }, .expected_counter{ 1 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 100 }, .numerator{ 100 }, .denominator{ 100 }, .expected_in{ 100 }, .expected_out{ 100 }, .expected_counter{ 1 } }, thread_pool);

        interpolation_decimation_test({ .n_samples{ 1000 }, .numerator{ 10 }, .denominator{ 1100 }, .expected_in{ 0 }, .expected_out{ 0 }, .expected_counter{ 0 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 1000 }, .numerator{ 1 }, .denominator{ 1001 }, .expected_in{ 0 }, .expected_out{ 0 }, .expected_counter{ 0 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 100 }, .numerator{ 100000 }, .denominator{ 1 }, .expected_in{ 0 }, .expected_out{ 0 }, .expected_counter{ 0 } }, thread_pool);
        interpolation_decimation_test({ .n_samples{ 100 }, .numerator{ 101 }, .denominator{ 101 }, .expected_in{ 0 }, .expected_out{ 0 }, .expected_counter{ 0 } }, thread_pool);

        interpolation_decimation_test(
                { .n_samples{ 100 }, .numerator{ 5 }, .denominator{ 11 }, .out_port_min_samples{ 10 }, .out_port_max_samples{ 41 }, .expected_in{ 88 }, .expected_out{ 40 }, .expected_counter{ 1 } },
                thread_pool);
        interpolation_decimation_test(
                { .n_samples{ 100 }, .numerator{ 7 }, .denominator{ 3 }, .out_port_min_samples{ 10 }, .out_port_max_samples{ 10 }, .expected_in{ 0 }, .expected_out{ 0 }, .expected_counter{ 0 } },
                thread_pool);
        interpolation_decimation_test(
                { .n_samples{ 80 }, .numerator{ 2 }, .denominator{ 4 }, .out_port_min_samples{ 20 }, .out_port_max_samples{ 20 }, .expected_in{ 40 }, .expected_out{ 20 }, .expected_counter{ 2 } },
                thread_pool);
        interpolation_decimation_test(
                { .n_samples{ 100 }, .numerator{ 7 }, .denominator{ 3 }, .out_port_min_samples{ 10 }, .out_port_max_samples{ 20 }, .expected_in{ 6 }, .expected_out{ 14 }, .expected_counter{ 16 } },
                thread_pool);
    };
};

int
main() { /* not needed for UT */
}
