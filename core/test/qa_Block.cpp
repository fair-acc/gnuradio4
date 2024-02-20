#include <vector>

#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

template<typename T>
struct BlockSignaturesNone : public gr::Block<BlockSignaturesNone<T>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesNone, in, out);
static_assert(!gr::HasRequiredProcessFunction<BlockSignaturesNone<float>>);

template<typename T>
struct BlockSignaturesVoid : public gr::Block<BlockSignaturesVoid<T>> {
    T value;

    void
    processOne() {}
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesVoid, value);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid<float>>);

template<typename T>
struct BlockSignaturesVoid2 : public gr::Block<BlockSignaturesVoid2<T>> {
    T value;

    gr::work::Status
    processBulk() {
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesVoid2, value);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid2<float>>);

template<typename T>
struct BlockSignaturesProcessOne : public gr::Block<BlockSignaturesProcessOne<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    T
    processOne(T) {
        return T();
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesProcessOne, in, out);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessOne<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesProcessOne<float>>);
static_assert(!gr::HasConstProcessOneFunction<BlockSignaturesProcessOne<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessOne<float>>);

template<typename T>
struct BlockSignaturesProcessOneConst : public gr::Block<BlockSignaturesProcessOneConst<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    T
    processOne(T) const {
        return T();
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesProcessOneConst, in, out);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasConstProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessOneConst<float>>);

enum class ProcessBulkVariant { SPAN, PUBLISHABLE_SPAN, PUBLISHABLE_SPAN2, CONSUMABLE_SPAN, CONSUMABLE_SPAN2 };

template<typename T, ProcessBulkVariant processVariant>
struct BlockSignaturesProcessBulkSpan : public gr::Block<BlockSignaturesProcessBulkSpan<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    gr::work::Status
    processBulk(std::span<const T>, std::span<T>)
        requires(processVariant == ProcessBulkVariant::SPAN)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(std::span<const T>, gr::PublishableSpan auto &)
        requires(processVariant == ProcessBulkVariant::PUBLISHABLE_SPAN)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(std::span<const T>, gr::PublishableSpan auto)
        requires(processVariant == ProcessBulkVariant::PUBLISHABLE_SPAN2)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(gr::ConsumableSpan auto, std::span<T>)
        requires(processVariant == ProcessBulkVariant::CONSUMABLE_SPAN)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(gr::ConsumableSpan auto &, std::span<T>)
        requires(processVariant == ProcessBulkVariant::CONSUMABLE_SPAN2)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, ProcessBulkVariant processVariant), (BlockSignaturesProcessBulkSpan<T, processVariant>), in, out);

static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN>>);
static_assert(!gr::HasProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN>>);
static_assert(!gr::HasConstProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::PUBLISHABLE_SPAN>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::PUBLISHABLE_SPAN2>>); // TODO: fix the signature is required to work
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_SPAN2>>);

static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::PUBLISHABLE_SPAN>, 0>);

enum class ProcessBulkTwoOutsVariant { SPAN_SPAN, PUBLISHABLE_SPAN, PUBLISHABLE_PUBLISHABLE, SPAN_PUBLISHABLE };

template<typename T, ProcessBulkTwoOutsVariant processVariant>
struct BlockSignaturesProcessBulkTwoOuts : public gr::Block<BlockSignaturesProcessBulkTwoOuts<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out1{};
    gr::PortOut<T> out2{};

    gr::work::Status
    processBulk(std::span<const T>, std::span<T>, std::span<T>)
        requires(processVariant == ProcessBulkTwoOutsVariant::SPAN_SPAN)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(std::span<const T>, gr::PublishableSpan auto &, std::span<T>)
        requires(processVariant == ProcessBulkTwoOutsVariant::PUBLISHABLE_SPAN)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(std::span<const T>, gr::PublishableSpan auto &, gr::PublishableSpan auto &)
        requires(processVariant == ProcessBulkTwoOutsVariant::PUBLISHABLE_PUBLISHABLE)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status
    processBulk(std::span<const T>, std::span<T>, gr::PublishableSpan auto &)
        requires(processVariant == ProcessBulkTwoOutsVariant::SPAN_PUBLISHABLE)
    {
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, ProcessBulkTwoOutsVariant processVariant), (BlockSignaturesProcessBulkTwoOuts<T, processVariant>), in, out1, out2);

static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::SPAN_SPAN>, 0>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::SPAN_SPAN>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::PUBLISHABLE_SPAN>, 0>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::PUBLISHABLE_SPAN>, 1>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::SPAN_PUBLISHABLE>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::SPAN_PUBLISHABLE>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::PUBLISHABLE_PUBLISHABLE>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::PUBLISHABLE_PUBLISHABLE>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::SPAN_SPAN>, 2>); // out-of-range check

struct InvalidSettingBlock : gr::Block<InvalidSettingBlock> {
    std::tuple<int> tuple; // this type is not supported and should cause the checkBlockContracts<T>() to throw
};

ENABLE_REFLECTION(InvalidSettingBlock, tuple);

struct MissingProcessSignature1 : gr::Block<MissingProcessSignature1> {
    gr::PortIn<int>    in;
    gr::PortOut<int>   out0;
    gr::PortOut<float> out1;
};

ENABLE_REFLECTION(MissingProcessSignature1, in, out0, out1);

struct MissingProcessSignature2 : gr::Block<MissingProcessSignature2> {
    gr::PortIn<int>    in0;
    gr::PortIn<float>  in1;
    gr::PortOut<int>   out0;
    gr::PortOut<float> out1;
};

ENABLE_REFLECTION(MissingProcessSignature2, in0, in1, out0, out1);

struct MissingProcessSignature3 : gr::Block<MissingProcessSignature3> {
    std::vector<gr::PortOut<float>>   outA;
    std::array<gr::PortOut<float>, 2> outB;

    template<typename PublishableSpan2>
    gr::work::Status
    processBulk(std::span<std::vector<float>> &, PublishableSpan2 &) { // TODO: needs proper explicit signature
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION(MissingProcessSignature3, outA, outB);

struct ProcessStatus {
    std::size_t      n_inputs{ 0 };
    std::size_t      n_outputs{ 0 };
    std::size_t      process_counter{ 0 };
    std::size_t      total_in{ 0 };
    std::size_t      total_out{ 0 };
    std::vector<int> in_vector{};
};

struct IntDecTestData {
    gr::Size_t  n_samples{};
    gr::Size_t  numerator{};
    gr::Size_t  denominator{};
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
    gr::Size_t       n_samples{};
    gr::Size_t       numerator{ 1U };
    gr::Size_t       denominator{ 1U };
    gr::Size_t       stride{};
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
struct IntDecBlock : public gr::Block<IntDecBlock<T>, gr::ResamplingRatio<>, gr::Stride<>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    ProcessStatus status{};
    bool          write_to_vector{ false };

    gr::work::Status
    processBulk(std::span<const T> input, std::span<T> output) noexcept {
        status.n_inputs  = input.size();
        status.n_outputs = output.size();
        status.process_counter++;
        status.total_in += input.size();
        status.total_out += output.size();
        if (write_to_vector) status.in_vector.insert(status.in_vector.end(), input.begin(), input.end());

        return gr::work::Status::OK;
    }
};

template<typename T>
struct AsyncBlock : gr::Block<AsyncBlock<T>> {
    gr::PortIn<T, gr::Async>  in;
    gr::PortOut<T, gr::Async> out;

    gr::work::Status
    processBulk(gr::PortIn<T, gr::Async>::ReaderType *inReader, gr::PortOut<T, gr::Async>::WriterType *outputWriter) {
        auto available = std::min(inReader->available(), outputWriter->available());
        if (available == 0) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        auto inSpan  = inReader->get(available);
        auto outSpan = outputWriter->reserve(available);
        std::copy(inSpan.begin(), std::next(inSpan.begin(), static_cast<std::ptrdiff_t>(available)), outSpan.begin());
        boost::ut::expect(inReader->consume(available)) << "Samples were not consumed";
        outSpan.publish(available);
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (IntDecBlock<T>), in, out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (AsyncBlock<T>), in, out);

const boost::ut::suite _block_signature = [] {
    using namespace boost::ut;

    "failure"_test = [] {
        expect(throws([] { throw 0; })) << "throws any exception";

        try {
            std::ignore = InvalidSettingBlock();
            expect(false) << "unsupported std::tuple setting not caught";
        } catch (const std::exception &e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = BlockSignaturesNone<float>();
            expect(false) << "missing process function not caught";
        } catch (const std::exception &e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature1();
            expect(false) << "missing process function not caught";
        } catch (const std::exception &e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature2();
            expect(false) << "missing process function not caught";
        } catch (const std::exception &e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature3();
            expect(false) << "missing process function not caught";
        } catch (const std::exception &e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }
    };
};

void
interpolation_decimation_test(const IntDecTestData &data, std::shared_ptr<gr::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using scheduler = gr::scheduler::Simple<>;

    gr::Graph flow;
    auto     &source = flow.emplaceBlock<gr::testing::TagSource<int>>({ { "n_samples_max", data.n_samples }, { "mark_tag", false } });

    auto &int_dec_block = flow.emplaceBlock<IntDecBlock<int>>({ { "numerator", data.numerator }, { "denominator", data.denominator } });
    if (data.out_port_max >= 0) int_dec_block.out.max_samples = static_cast<size_t>(data.out_port_max);
    if (data.out_port_min >= 0) int_dec_block.out.min_samples = static_cast<size_t>(data.out_port_min);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(int_dec_block)));
    auto sched = scheduler(std::move(flow), thread_pool);
    sched.runAndWait();

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "processBulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
}

void
stride_test(const StrideTestData &data, std::shared_ptr<gr::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using scheduler = gr::scheduler::Simple<>;

    const bool write_to_vector{ data.exp_in_vector.size() != 0 };

    gr::Graph flow;
    auto     &source = flow.emplaceBlock<gr::testing::TagSource<int>>({ { "n_samples_max", data.n_samples }, { "mark_tag", false } });

    auto &int_dec_block           = flow.emplaceBlock<IntDecBlock<int>>({ { "numerator", data.numerator }, { "denominator", data.denominator }, { "stride", data.stride } });
    int_dec_block.write_to_vector = write_to_vector;
    if (data.in_port_max >= 0) int_dec_block.in.max_samples = static_cast<size_t>(data.in_port_max);
    if (data.in_port_min >= 0) int_dec_block.in.min_samples = static_cast<size_t>(data.in_port_min);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(int_dec_block)));
    auto sched = scheduler(std::move(flow), thread_pool);
    sched.runAndWait();

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "processBulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_in, data.exp_total_in)) << "total number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_out, data.exp_total_out)) << "total number of output samples, parameters = " << data.to_string();
    if (write_to_vector) {
        expect(eq(int_dec_block.status.in_vector, data.exp_in_vector)) << "in vector of samples, parameters = " << data.to_string();
    }
}

const boost::ut::suite _stride_tests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2);

    "ResamplingRatio"_test = [] {
        static_assert(gr::ResamplingRatio<>::kNumerator == 1LU);
        static_assert(gr::ResamplingRatio<>::kDenominator == 1LU);
        static_assert(gr::ResamplingRatio<>::kIsConst == false);
        static_assert(gr::ResamplingRatio<>::kEnabled == true);

        static_assert(gr::ResamplingRatio<2LU>::kNumerator == 2LU);
        static_assert(gr::ResamplingRatio<2LU>::kDenominator == 1LU);
        static_assert(gr::ResamplingRatio<2LU>::kIsConst == false);
        static_assert(gr::ResamplingRatio<2LU>::kEnabled == true);

        static_assert(gr::ResamplingRatio<1LU, 1LU, true>::kNumerator == 1LU);
        static_assert(gr::ResamplingRatio<1LU, 1LU, true>::kDenominator == 1LU);
        static_assert(gr::ResamplingRatio<1LU, 1LU, true>::kIsConst == true);
        static_assert(gr::ResamplingRatio<1LU, 1LU, true>::kEnabled == false);

        static_assert(gr::ResamplingRatio<2LU, 1LU, true>::kNumerator == 2LU);
        static_assert(gr::ResamplingRatio<2LU, 1LU, true>::kDenominator == 1LU);
        static_assert(gr::ResamplingRatio<2LU, 1LU, true>::kIsConst == true);
        static_assert(gr::ResamplingRatio<2LU, 1LU, true>::kEnabled == true);

        struct TestBlock0 : gr::Block<TestBlock0> {
        } testBlock0;
        static_assert(std::is_const_v<decltype(testBlock0.numerator.value)>);
        static_assert(std::is_const_v<decltype(testBlock0.denominator.value)>);

        struct TestBlock1 : gr::Block<TestBlock1, gr::ResamplingRatio<>> {
        } testBlock1;
        static_assert(!std::is_const_v<decltype(testBlock1.numerator.value)>);
        static_assert(!std::is_const_v<decltype(testBlock1.denominator.value)>);

        struct TestBlock2 : gr::Block<TestBlock2, gr::ResamplingRatio<2LU, 1LU, true>> {
        } testBlock2;
        static_assert(std::is_const_v<decltype(testBlock2.numerator.value)>);
        static_assert(std::is_const_v<decltype(testBlock2.denominator.value)>);
        expect(eq(testBlock2.numerator, 2LU));
        expect(eq(testBlock2.denominator, 1LU));
    };

    "Stride"_test = [] {
        static_assert(gr::Stride<>::kStride == 0LU);
        static_assert(gr::Stride<>::kIsConst == false);
        static_assert(gr::Stride<>::kEnabled == true);

        static_assert(gr::Stride<2LU>::kStride == 2LU);
        static_assert(gr::Stride<2LU>::kIsConst == false);
        static_assert(gr::Stride<2LU>::kEnabled == true);

        static_assert(gr::Stride<0LU, true>::kStride == 0LU);
        static_assert(gr::Stride<0LU, true>::kIsConst == true);
        static_assert(gr::Stride<0LU, true>::kEnabled == false);

        static_assert(gr::Stride<1LU, true>::kStride == 1LU);
        static_assert(gr::Stride<1LU, true>::kIsConst == true);
        static_assert(gr::Stride<1LU, true>::kEnabled == true);

        struct TestBlock0 : gr::Block<TestBlock0> {
        } testBlock0;
        static_assert(std::is_const_v<decltype(testBlock0.stride.value)>);

        struct TestBlock1 : gr::Block<TestBlock1, gr::Stride<>> {
        } testBlock1;
        static_assert(!std::is_const_v<decltype(testBlock1.stride.value)>);

        struct TestBlock2 : gr::Block<TestBlock2, gr::Stride<2LU, true>> {
        } testBlock2;
        static_assert(std::is_const_v<decltype(testBlock2.stride.value)>);
        expect(eq(testBlock2.stride, 2LU));
    };

    "User ResamplingRatio & Stride"_test = [] {
        using namespace gr;

        struct TestBlock : gr::Block<TestBlock, gr::ResamplingRatio<2LU, 1LU, true>, gr::Stride<2LU, false>> {
        } testBlock;
        static_assert(std::is_const_v<decltype(testBlock.numerator.value)>);
        static_assert(std::is_const_v<decltype(testBlock.denominator.value)>);
        static_assert(!std::is_const_v<decltype(testBlock.stride.value)>);
        expect(eq(testBlock.numerator, 2LU));
        expect(eq(testBlock.denominator, 1LU));
        expect(eq(testBlock.stride, 2LU));
    };

    // clang-format off
    "Interpolation/Decimation"_test = [&thread_pool] {
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
// TODO: check with Semen if this test is possibly ill-defined and can never finish
//        interpolation_decimation_test({ .n_samples =  100, .numerator = 7, .denominator =  3, .out_port_min = 10 , .out_port_max = 10, .exp_in =  0, .exp_out =  0, .exp_counter =  0 }, thread_pool);
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

    "Async ports tests"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        constexpr gr::Size_t n_samples   = 1000;
        constexpr float      sample_rate = 1000.f;
        Graph                testGraph;
        auto                &tagSrc     = testGraph.emplaceBlock<TagSource<float>>({ { "sample_rate", sample_rate }, { "n_samples_max", n_samples }, { "name", "TagSource" } });
        auto                &asyncBlock = testGraph.emplaceBlock<AsyncBlock<float>>({ { "name", "AsyncBlock" } });
        auto                &sink       = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink" }, { "verbose_console", true } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(tagSrc).to<"in">(asyncBlock)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(asyncBlock).to<"in">(sink)));

        scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        expect(eq(n_samples, static_cast<gr::Size_t>(sink.n_samples_produced))) << "Number of samples does not match";
    };
};

int
main() { /* not needed for UT */
}
