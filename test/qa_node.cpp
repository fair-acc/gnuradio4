#include <boost/ut.hpp>
#include <vector>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

#include <fmt/format.h>
#include <graph.hpp>
#include <node.hpp>
#include <scheduler.hpp>

namespace fg = fair::graph;

struct ProcessStatus {
    std::size_t      n_inputs{ 0 };
    std::size_t      n_outputs{ 0 };
    std::size_t      process_counter{ 0 };
    std::size_t      total_in{ 0 };
    std::size_t      total_out{ 0 };
    std::vector<int> in_vector{};
};

struct IntDecTestData {
    std::size_t n_samples{};
    std::size_t numerator{};
    std::size_t denominator{};
    int         out_port_min{ -1 }; // -1 for not used
    int         out_port_max{ -1 }; // -1 for not used
    std::size_t exp_in{};
    std::size_t exp_out{};
    std::size_t exp_counter{};

    std::string
    to_string() const {
        return fmt::format("n_samples: {}, numerator: {}, denominator: {}, out_port_min: {}, out_port_max: {}, exp_in: {}, exp_out: {}, exp_counter: {}", n_samples, numerator, denominator,
                           out_port_min, out_port_max, exp_in, exp_out, exp_counter);
    }
};

struct StrideTestData {
    std::size_t      n_samples{};
    std::size_t      numerator{ 1 };
    std::size_t      denominator{ 1 };
    std::size_t      stride{};
    int              in_port_min{ -1 }; // -1 for not used
    int              in_port_max{ -1 }; // -1 for not used
    std::size_t      exp_in{};
    std::size_t      exp_out{};
    std::size_t      exp_counter{};
    std::size_t      exp_total_in{ 0 };
    std::size_t      exp_total_out{ 0 };
    std::vector<int> exp_in_vector{};

    std::string
    to_string() const {
        return fmt::format("n_samples: {}, numerator: {}, denominator: {}, stride: {}, in_port_min: {}, in_port_max: {}, exp_in: {}, exp_out: {}, exp_counter: {}, exp_total_in: {}, exp_total_out: {}",
                           n_samples, numerator, denominator, stride, in_port_min, in_port_max, exp_in, exp_out, exp_counter, exp_total_in, exp_total_out);
    }
};

template<typename T>
struct CountSource : public fg::node<CountSource<T>> {
    fg::PortOut<T> out{};
    int            count{ 0 };
    int            n_samples{ 1024 };

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
struct IntDecBlock : public fg::node<IntDecBlock<T>, fg::PerformDecimationInterpolation, fg::PerformStride> {
    fg::PortIn<T>  in{};
    fg::PortOut<T> out{};

    ProcessStatus  status{};
    bool           write_to_vector{ false };

    fg::work_return_status_t
    process_bulk(std::span<const T> input, std::span<T> output) noexcept {
        status.n_inputs  = input.size();
        status.n_outputs = output.size();
        status.process_counter++;
        status.total_in += input.size();
        status.total_out += output.size();
        if (write_to_vector) status.in_vector.insert(status.in_vector.end(), input.begin(), input.end());

        return fg::work_return_status_t::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (CountSource<T>), out, count, n_samples);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (IntDecBlock<T>), in, out);

void
interpolation_decimation_test(const IntDecTestData &data, std::shared_ptr<fair::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using scheduler = fair::graph::scheduler::simple<>;

    fg::graph flow;
    auto     &source          = flow.make_node<CountSource<int>>();
    source.n_samples          = static_cast<int>(data.n_samples);

    auto &int_dec_block       = flow.make_node<IntDecBlock<int>>();
    int_dec_block.numerator   = data.numerator;
    int_dec_block.denominator = data.denominator;
    if (data.out_port_max >= 0) int_dec_block.out.max_samples = static_cast<size_t>(data.out_port_max);
    if (data.out_port_min >= 0) int_dec_block.out.min_samples = static_cast<size_t>(data.out_port_min);

    std::ignore = flow.connect<"out">(source).to<"in">(int_dec_block);
    auto sched  = scheduler(std::move(flow), thread_pool);
    sched.run_and_wait();

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "process_bulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
}

void
stride_test(const StrideTestData &data, std::shared_ptr<fair::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using scheduler = fair::graph::scheduler::simple<>;

    const bool write_to_vector{ data.exp_in_vector.size() != 0 };

    fg::graph  flow;
    auto      &source             = flow.make_node<CountSource<int>>();
    source.n_samples              = static_cast<int>(data.n_samples);

    auto &int_dec_block           = flow.make_node<IntDecBlock<int>>();
    int_dec_block.write_to_vector = write_to_vector;
    int_dec_block.numerator       = data.numerator;
    int_dec_block.denominator     = data.denominator;
    int_dec_block.stride          = data.stride;
    if (data.in_port_max >= 0) int_dec_block.in.max_samples = static_cast<size_t>(data.in_port_max);
    if (data.in_port_min >= 0) int_dec_block.in.min_samples = static_cast<size_t>(data.in_port_min);

    std::ignore = flow.connect<"out">(source).to<"in">(int_dec_block);
    auto sched  = scheduler(std::move(flow), thread_pool);
    sched.run_and_wait();

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "process_bulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_in, data.exp_total_in)) << "total number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_out, data.exp_total_out)) << "total number of output samples, parameters = " << data.to_string();
    if (write_to_vector) {
        expect(eq(int_dec_block.status.in_vector, data.exp_in_vector)) << "in vector of samples, parameters = " << data.to_string();
    }
}

const boost::ut::suite _fft_tests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    auto thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2);

    // clang-format off
    "Interpolation/Decimation tests"_test = [&thread_pool] {
        interpolation_decimation_test({ .n_samples = 1024, .numerator =   1, .denominator =   1, .exp_in = 1024, .exp_out = 1024, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples = 1024, .numerator =   1, .denominator =   2, .exp_in = 1024, .exp_out =  512, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples = 1024, .numerator =   2, .denominator =   1, .exp_in = 1024, .exp_out = 2048, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples = 1000, .numerator =   5, .denominator =   6, .exp_in =  996, .exp_out =  830, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  549, .numerator =   1, .denominator =  50, .exp_in =  500, .exp_out =   10, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator =   3, .denominator =   7, .exp_in =   98, .exp_out =   42, .exp_counter = 1 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator = 100, .denominator = 100, .exp_in =  100, .exp_out =  100, .exp_counter = 1 }, thread_pool);

        interpolation_decimation_test({ .n_samples = 1000, .numerator =     10, .denominator = 1100, .exp_in = 0 , .exp_out = 0, .exp_counter = 0 }, thread_pool);
        interpolation_decimation_test({ .n_samples = 1000, .numerator =      1, .denominator = 1001, .exp_in = 0 , .exp_out = 0, .exp_counter = 0 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator = 100000, .denominator =    1, .exp_in = 0 , .exp_out = 0, .exp_counter = 0 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator =    101, .denominator =  101, .exp_in = 0 , .exp_out = 0, .exp_counter = 0 }, thread_pool);

        interpolation_decimation_test({ .n_samples =  100, .numerator = 5, .denominator = 11, .out_port_min = 10 , .out_port_max = 41, .exp_in = 88, .exp_out = 40, .exp_counter =  1 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator = 7, .denominator =  3, .out_port_min = 10 , .out_port_max = 10, .exp_in =  0, .exp_out =  0, .exp_counter =  0 }, thread_pool);
        interpolation_decimation_test({ .n_samples =   80, .numerator = 2, .denominator =  4, .out_port_min = 20 , .out_port_max = 20, .exp_in = 40, .exp_out = 20, .exp_counter =  2 }, thread_pool);
        interpolation_decimation_test({ .n_samples =  100, .numerator = 7, .denominator =  3, .out_port_min = 10 , .out_port_max = 20, .exp_in =  6, .exp_out = 14, .exp_counter = 16 }, thread_pool);
    };

    "Stride tests"_test = [&thread_pool] {
        stride_test( {.n_samples = 1024 , .stride =   0 , .in_port_max = 1024 , .exp_in = 1024 , .exp_out = 1024 , .exp_counter =  1 , .exp_total_in = 1024 , .exp_total_out = 1024 }, thread_pool);
        stride_test( {.n_samples = 1000 , .stride = 100 , .in_port_max =   50 , .exp_in =   50 , .exp_out =   50 , .exp_counter = 10 , .exp_total_in =  500 , .exp_total_out =  500 }, thread_pool);
        stride_test( {.n_samples = 1000 , .stride = 133 , .in_port_max =   50 , .exp_in =   50 , .exp_out =   50 , .exp_counter =  8 , .exp_total_in =  400 , .exp_total_out =  400 }, thread_pool);
        stride_test( {.n_samples = 1000 , .stride =  50 , .in_port_max =  100 , .exp_in =   50 , .exp_out =   50 , .exp_counter = 20 , .exp_total_in = 1950 , .exp_total_out = 1950 }, thread_pool);
        stride_test( {.n_samples = 1000 , .stride =  33 , .in_port_max =  100 , .exp_in =   10 , .exp_out =   10 , .exp_counter = 31 , .exp_total_in = 2929 , .exp_total_out = 2929 }, thread_pool);
        stride_test( {.n_samples = 1000 , .numerator = 2 , .denominator = 4 , .stride = 50 , .in_port_max = 100 , .exp_in = 48 , .exp_out = 24 , .exp_counter = 20 , .exp_total_in = 1948 , .exp_total_out = 974 }, thread_pool);
        stride_test( {.n_samples = 1000 , .numerator = 2 , .denominator = 4 , .stride = 50 , .in_port_max =  50 , .exp_in = 48 , .exp_out = 24 , .exp_counter = 20 , .exp_total_in =  960 , .exp_total_out = 480 }, thread_pool);

        std::vector<int> exp_v1 = {0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 12, 13, 14};
        stride_test( {.n_samples = 15, .stride = 3, .in_port_max = 5, .exp_in = 3, .exp_out = 3, .exp_counter = 5, .exp_total_in = 23, .exp_total_out = 23, .exp_in_vector = exp_v1 }, thread_pool);

        std::vector<int> exp_v2 = {0, 1, 2, 5, 6, 7, 10, 11, 12};
        stride_test( {.n_samples = 15, .stride = 5, .in_port_max = 3, .exp_in = 3, .exp_out = 3, .exp_counter = 3, .exp_total_in = 9, .exp_total_out = 9, .exp_in_vector = exp_v2 }, thread_pool);

        // assuming buffer size is approx 65k
        stride_test( {.n_samples = 1000000, .stride = 250000, .in_port_max = 100, .exp_in = 100, .exp_out = 100, .exp_counter = 4, .exp_total_in = 400, .exp_total_out = 400 }, thread_pool);
        stride_test( {.n_samples = 1000000, .stride = 249900, .in_port_max = 100, .exp_in = 100, .exp_out = 100, .exp_counter = 5, .exp_total_in = 500, .exp_total_out = 500 }, thread_pool);
    };
    // clang-format on
};

int
main() { /* not needed for UT */
}
