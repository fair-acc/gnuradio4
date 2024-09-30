#include <utility>
#include <vector>

#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if !DISABLE_SIMD
namespace gr::test {
struct copy : public Block<copy> {
    PortIn<float>  in;
    PortOut<float> out;

public:
    template<meta::t_or_simd<float> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a;
    }
};
} // namespace gr::test

ENABLE_REFLECTION(gr::test::copy, in, out);

namespace gr::test {
static_assert(traits::block::stream_input_port_types<copy>::size() == 1);
static_assert(std::same_as<traits::block::stream_return_type<copy>, float>);
static_assert(traits::block::can_processOne_scalar<copy>);
static_assert(traits::block::can_processOne_simd<copy>);
static_assert(traits::block::can_processOne_scalar<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(traits::block::can_processOne_simd<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(SourceBlockLike<copy>);
static_assert(SinkBlockLike<copy>);
static_assert(SourceBlockLike<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
static_assert(SinkBlockLike<decltype(mergeByIndex<0, 0>(copy(), copy()))>);
} // namespace gr::test
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

    void processOne() {}
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesVoid, value);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid<float>>);

template<typename T>
struct BlockSignaturesVoid2 : public gr::Block<BlockSignaturesVoid2<T>> {
    T value;

    gr::work::Status processBulk() { return gr::work::Status::OK; }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesVoid2, value);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid2<float>>);

template<typename T>
struct BlockSignaturesProcessOne : public gr::Block<BlockSignaturesProcessOne<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    T processOne(T) { return T(); }
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

    T processOne(T) const { return T(); }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesProcessOneConst, in, out);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasConstProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessOneConst<float>>);

template<typename T>
struct BlockSignaturesTemplatedProcessOneConst : public gr::Block<BlockSignaturesTemplatedProcessOneConst<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& /*input*/) const noexcept {
        return V();
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(BlockSignaturesTemplatedProcessOneConst, in, out);
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(gr::HasConstProcessOneFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesTemplatedProcessOneConst<float>>);

enum class ProcessBulkVariant { SPAN_SPAN, SPAN_PUBLISHABLE, SPAN_PUBLISHABLE2, CONSUMABLE_SPAN, CONSUMABLE_SPAN2, CONSUMABLE_PUBLISHABLE, CONSUMABLE_PUBLISHABLE2 };

template<typename T, ProcessBulkVariant processVariant>
struct BlockSignaturesProcessBulkSpan : public gr::Block<BlockSignaturesProcessBulkSpan<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    gr::work::Status processBulk(std::span<const T>, std::span<T>)
    requires(processVariant == ProcessBulkVariant::SPAN_SPAN)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::PublishableSpan auto&)
    requires(processVariant == ProcessBulkVariant::SPAN_PUBLISHABLE)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::PublishableSpan auto)
    requires(processVariant == ProcessBulkVariant::SPAN_PUBLISHABLE2)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::ConsumableSpan auto, std::span<T>)
    requires(processVariant == ProcessBulkVariant::CONSUMABLE_SPAN)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::ConsumableSpan auto&, std::span<T>)
    requires(processVariant == ProcessBulkVariant::CONSUMABLE_SPAN2)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::ConsumableSpan auto&, gr::PublishableSpan auto&)
    requires(processVariant == ProcessBulkVariant::CONSUMABLE_PUBLISHABLE)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::ConsumableSpan auto, gr::PublishableSpan auto)
    requires(processVariant == ProcessBulkVariant::CONSUMABLE_PUBLISHABLE2)
    {
        // do some bulk-type processing
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, ProcessBulkVariant processVariant), (BlockSignaturesProcessBulkSpan<T, processVariant>), in, out);

static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_SPAN>>);
static_assert(!gr::HasProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_SPAN>>);
static_assert(!gr::HasConstProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_PUBLISHABLE>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_PUBLISHABLE2>>); // TODO: fix the signature is required to work
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_SPAN2>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_PUBLISHABLE>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::CONSUMABLE_PUBLISHABLE2>>); // TODO: fix the signature is required to work

static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_SPAN>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::SPAN_PUBLISHABLE>, 0>);

enum class ProcessBulkTwoOutsVariant { SPAN_SPAN, PUBLISHABLE_SPAN, PUBLISHABLE_PUBLISHABLE, SPAN_PUBLISHABLE };

template<typename T, ProcessBulkTwoOutsVariant processVariant>
struct BlockSignaturesProcessBulkTwoOuts : public gr::Block<BlockSignaturesProcessBulkTwoOuts<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out1{};
    gr::PortOut<T> out2{};

    gr::work::Status processBulk(std::span<const T>, std::span<T>, std::span<T>)
    requires(processVariant == ProcessBulkTwoOutsVariant::SPAN_SPAN)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::PublishableSpan auto&, std::span<T>)
    requires(processVariant == ProcessBulkTwoOutsVariant::PUBLISHABLE_SPAN)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::PublishableSpan auto&, gr::PublishableSpan auto&)
    requires(processVariant == ProcessBulkTwoOutsVariant::PUBLISHABLE_PUBLISHABLE)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, std::span<T>, gr::PublishableSpan auto&)
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

enum class ProcessBulkVectorVariant { SPAN_SPAN, SPAN_SPAN2, CONSUMABLE_SPAN, CONSUMABLE_PUBLISHABLE, CONSUMABLE_PUBLISHABLE2, SPAN_PUBLISHABLE };

template<typename T, ProcessBulkVectorVariant processVariant>
struct BlockSignaturesProcessBulkVector : public gr::Block<BlockSignaturesProcessBulkVector<T, processVariant>> {
    std::array<gr::PortIn<T>, 3>  inputs{};
    std::array<gr::PortOut<T>, 6> outputs{};

    gr::work::Status processBulk(const std::span<std::span<const T>>&, std::span<std::span<T>>&)
    requires(processVariant == ProcessBulkVectorVariant::SPAN_SPAN)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<std::span<const T>>, std::span<std::span<T>>)
    requires(processVariant == ProcessBulkVectorVariant::SPAN_SPAN2)
    {
        return gr::work::Status::OK;
    }

    template<gr::ConsumableSpan TInput>
    gr::work::Status processBulk(const std::span<TInput>&, std::span<std::span<T>>&)
    requires(processVariant == ProcessBulkVectorVariant::CONSUMABLE_SPAN)
    {
        return gr::work::Status::OK;
    }

    template<gr::ConsumableSpan TInput, gr::PublishableSpan TOutput>
    gr::work::Status processBulk(const std::span<TInput>&, std::span<TOutput>&)
    requires(processVariant == ProcessBulkVectorVariant::CONSUMABLE_PUBLISHABLE)
    {
        return gr::work::Status::OK;
    }

    template<gr::ConsumableSpan TInput, gr::PublishableSpan TOutput>
    gr::work::Status processBulk(std::span<TInput>, std::span<TOutput>)
    requires(processVariant == ProcessBulkVectorVariant::CONSUMABLE_PUBLISHABLE2)
    {
        return gr::work::Status::OK;
    }

    template<gr::PublishableSpan TOutput>
    gr::work::Status processBulk(const std::span<std::span<const T>>&, std::span<TOutput>&)
    requires(processVariant == ProcessBulkVectorVariant::SPAN_PUBLISHABLE)
    {
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, ProcessBulkVectorVariant processVariant), (BlockSignaturesProcessBulkVector<T, processVariant>), inputs, outputs);

static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::SPAN_SPAN>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::SPAN_SPAN2>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::CONSUMABLE_SPAN>>); // combinations are not supported yet
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::CONSUMABLE_PUBLISHABLE>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::CONSUMABLE_PUBLISHABLE2>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::SPAN_PUBLISHABLE>>); // combinations are not supported yet

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
    gr::work::Status processBulk(std::span<std::vector<float>>&, PublishableSpan2&) { // TODO: needs proper explicit signature
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION(MissingProcessSignature3, outA, outB);

struct ProcessStatus {
    std::size_t      n_inputs{0};
    std::size_t      n_outputs{0};
    std::size_t      process_counter{0};
    std::size_t      total_in{0};
    std::size_t      total_out{0};
    std::vector<int> in_vector{};
};

struct IntDecTestData {
    gr::Size_t  n_samples{};
    gr::Size_t  output_chunk_size{};
    gr::Size_t  input_chunk_size{};
    int         out_port_min{-1}; // -1 for not used
    int         out_port_max{-1}; // -1 for not used
    std::size_t exp_in{};
    std::size_t exp_out{};
    std::size_t exp_counter{};

    std::string to_string() const { return fmt::format("n_samples: {}, output_chunk_size: {}, input_chunk_size: {}, out_port_min: {}, out_port_max: {}, exp_in: {}, exp_out: {}, exp_counter: {}", n_samples, output_chunk_size, input_chunk_size, out_port_min, out_port_max, exp_in, exp_out, exp_counter); }
};

struct StrideTestData {
    gr::Size_t       n_samples{};
    gr::Size_t       output_chunk_size{1U};
    gr::Size_t       input_chunk_size{1U};
    gr::Size_t       stride{};
    int              in_port_min{-1}; // -1 for not used
    int              in_port_max{-1}; // -1 for not used
    std::size_t      exp_in{};
    std::size_t      exp_out{};
    std::size_t      exp_counter{};
    std::size_t      exp_total_in{0};
    std::size_t      exp_total_out{0};
    std::vector<int> exp_in_vector{};

    std::string to_string() const { return fmt::format("n_samples: {}, output_chunk_size: {}, input_chunk_size: {}, stride: {}, in_port_min: {}, in_port_max: {}, exp_in: {}, exp_out: {}, exp_counter: {}, exp_total_in: {}, exp_total_out: {}", n_samples, output_chunk_size, input_chunk_size, stride, in_port_min, in_port_max, exp_in, exp_out, exp_counter, exp_total_in, exp_total_out); }
};

template<typename T>
struct IntDecBlock : public gr::Block<IntDecBlock<T>, gr::Resampling<>, gr::Stride<>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    ProcessStatus status{};
    bool          write_to_vector{false};

    gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        status.n_inputs  = input.size();
        status.n_outputs = output.size();
        status.process_counter++;
        status.total_in += input.size();
        status.total_out += output.size();
        if (write_to_vector) {
            status.in_vector.insert(status.in_vector.end(), input.begin(), input.end());
        }

        return gr::work::Status::OK;
    }
};

// This block is used to test different combination of Sync/Async input/output ports
template<typename T, bool isInputAsync, bool isOutputAsync>
struct SyncOrAsyncBlock : gr::Block<SyncOrAsyncBlock<T, isInputAsync, isOutputAsync>> {

    using InputPortType  = std::conditional_t<isInputAsync, gr::PortIn<T, gr::Async>, gr::PortIn<T>>;
    using OutputPortType = std::conditional_t<isOutputAsync, gr::PortOut<T, gr::Async>, gr::PortOut<T>>;
    InputPortType  in{};
    OutputPortType out{};

    gr::work::Status processBulk(const gr::ConsumableSpan auto& inSpan, gr::PublishableSpan auto& outSpan) {
        const auto available = std::min(inSpan.size(), outSpan.size());
        if (available != 0) {
            std::copy(inSpan.begin(), std::next(inSpan.begin(), static_cast<std::ptrdiff_t>(available)), outSpan.begin());
        }
        outSpan.publish(available);
        boost::ut::expect(inSpan.tryConsume(available)) << "Samples were not consumed";
        return gr::work::Status::OK;
    }
};
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool isInputAsync, bool isOututAsync), (SyncOrAsyncBlock<T, isInputAsync, isOututAsync>), in, out);
static_assert(gr::HasProcessBulkFunction<SyncOrAsyncBlock<float, true, true>>);

ENABLE_REFLECTION_FOR_TEMPLATE(IntDecBlock, in, out);

template<typename T>
struct ArrayPortsNode : gr::Block<ArrayPortsNode<T>> {
    static constexpr std::size_t nPorts = 4;

    std::array<gr::PortIn<T, gr::Async>, nPorts>  inputs;
    std::array<gr::PortOut<T, gr::Async>, nPorts> outputs;

    template<typename TInSpan, typename TOutSpan>
    gr::work::Status processBulk(const std::span<TInSpan>& ins, const std::span<TOutSpan>& outs) {
        for (std::size_t channelIndex = 0; channelIndex < ins.size(); ++channelIndex) {
            gr::ConsumableSpan auto  inputSpan  = ins[channelIndex];
            gr::PublishableSpan auto outputSpan = outs[channelIndex];
            auto                     available  = std::min(inputSpan.size(), outputSpan.size());

            for (std::size_t valueIndex = 0; valueIndex < available; ++valueIndex) {
                outputSpan[valueIndex] = inputSpan[valueIndex];
            }

            std::ignore = inputSpan.consume(available);
            outputSpan.publish(available);
        }
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(ArrayPortsNode, inputs, outputs);
static_assert(gr::HasProcessBulkFunction<ArrayPortsNode<int>>);
const boost::ut::suite<"Block signatures"> _block_signature = [] {
    using namespace boost::ut;

    "failure"_test = [] {
        expect(throws([] { throw 0; })) << "throws any exception";

        try {
            std::ignore = InvalidSettingBlock();
            expect(false) << "unsupported std::tuple setting not caught";
        } catch (const std::exception& e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = BlockSignaturesNone<float>();
            expect(false) << "missing process function not caught";
        } catch (const std::exception& e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature1();
            expect(false) << "missing process function not caught";
        } catch (const std::exception& e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature2();
            expect(false) << "missing process function not caught";
        } catch (const std::exception& e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }

        try {
            std::ignore = MissingProcessSignature3();
            expect(false) << "missing process function not caught";
        } catch (const std::exception& e) {
            fmt::println("correctly thrown exception:\n{}", e.what());
            expect(true);
        } catch (...) {
            expect(false);
        }
    };
};

void interpolation_decimation_test(const IntDecTestData& data, std::shared_ptr<gr::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using namespace gr::testing;
    using scheduler = gr::scheduler::Simple<>;

    gr::Graph flow;
    auto&     source        = flow.emplaceBlock<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", data.n_samples}, {"mark_tag", false}});
    auto&     int_dec_block = flow.emplaceBlock<IntDecBlock<int>>({{"output_chunk_size", data.output_chunk_size}, {"input_chunk_size", data.input_chunk_size}});
    auto&     sink          = flow.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>();
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(int_dec_block)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(int_dec_block).to<"in">(sink)));
    if (data.out_port_max >= 0) {
        int_dec_block.out.max_samples = static_cast<size_t>(data.out_port_max);
    }
    if (data.out_port_min >= 0) {
        int_dec_block.out.min_samples = static_cast<size_t>(data.out_port_min);
    }

    auto sched = scheduler(std::move(flow), std::move(thread_pool));
    expect(sched.runAndWait().has_value());

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "processBulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
}

void stride_test(const StrideTestData& data, std::shared_ptr<gr::thread_pool::BasicThreadPool> thread_pool) {
    using namespace boost::ut;
    using namespace gr::testing;
    using scheduler = gr::scheduler::Simple<>;

    const bool write_to_vector{data.exp_in_vector.size() != 0};

    gr::Graph flow;
    auto&     source        = flow.emplaceBlock<TagSource<int>>({{"n_samples_max", data.n_samples}, {"mark_tag", false}});
    auto&     int_dec_block = flow.emplaceBlock<IntDecBlock<int>>({{"output_chunk_size", data.output_chunk_size}, {"input_chunk_size", data.input_chunk_size}, {"stride", data.stride}});
    auto&     sink          = flow.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>();
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(int_dec_block)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(int_dec_block).to<"in">(sink)));
    int_dec_block.write_to_vector = write_to_vector;
    if (data.in_port_max >= 0) {
        int_dec_block.in.max_samples = static_cast<size_t>(data.in_port_max);
    }
    if (data.in_port_min >= 0) {
        int_dec_block.in.min_samples = static_cast<size_t>(data.in_port_min);
    }

    auto sched = scheduler(std::move(flow), std::move(thread_pool));
    expect(sched.runAndWait().has_value());

    expect(eq(int_dec_block.status.process_counter, data.exp_counter)) << "processBulk invokes counter, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_inputs, data.exp_in)) << "last number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.n_outputs, data.exp_out)) << "last number of output samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_in, data.exp_total_in)) << "total number of input samples, parameters = " << data.to_string();
    expect(eq(int_dec_block.status.total_out, data.exp_total_out)) << "total number of output samples, parameters = " << data.to_string();
    if (write_to_vector) {
        expect(eq(int_dec_block.status.in_vector, data.exp_in_vector)) << "in vector of samples, parameters = " << data.to_string();
    }
}

template<bool isInputAsync, bool isOutputAsync>
void syncOrAsyncTest() {
    using namespace gr;
    using namespace gr::testing;
    using namespace boost::ut;
    constexpr gr::Size_t n_samples = 100;

    using BlockType = SyncOrAsyncBlock<float, isInputAsync, isOutputAsync>;

    Graph             testGraph;
    auto&             tagSrc     = testGraph.emplaceBlock<TagSource<float>>({{"n_samples_max", n_samples}});
    auto&             asyncBlock = testGraph.emplaceBlock<BlockType>();
    auto&             sink       = testGraph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
    const std::string testInfo   = fmt::format("syncOrAsyncTest<{}, {}>", isInputAsync, isOutputAsync);
    expect(asyncBlock.in.kIsSynch == !isInputAsync) << testInfo;
    expect(asyncBlock.out.kIsSynch == !isOutputAsync) << testInfo;

    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(tagSrc).to<"in">(asyncBlock))) << testInfo;
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(asyncBlock).template to<"in">(sink))) << testInfo;

    scheduler::Simple sched{std::move(testGraph)};
    expect(sched.runAndWait().has_value()) << testInfo;
    expect(eq(n_samples, sink._nSamplesProduced)) << testInfo;
}

const boost::ut::suite<"Stride Tests"> _stride_tests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using namespace gr;

    auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2);

    "Resampling"_test = [] {
        static_assert(Resampling<>::kInputChunkSize == 1LU);
        static_assert(Resampling<>::kOutputChunkSize == 1LU);
        static_assert(Resampling<>::kIsConst == false);
        static_assert(Resampling<>::kEnabled == true);

        static_assert(Resampling<2LU>::kInputChunkSize == 2LU);
        static_assert(Resampling<2LU>::kOutputChunkSize == 1LU);
        static_assert(Resampling<2LU>::kIsConst == false);
        static_assert(Resampling<2LU>::kEnabled == true);

        static_assert(Resampling<1LU, 1LU, true>::kInputChunkSize == 1LU);
        static_assert(Resampling<1LU, 1LU, true>::kOutputChunkSize == 1LU);
        static_assert(Resampling<1LU, 1LU, true>::kIsConst == true);
        static_assert(Resampling<1LU, 1LU, true>::kEnabled == false);

        static_assert(Resampling<2LU, 1LU, true>::kInputChunkSize == 2LU);
        static_assert(Resampling<2LU, 1LU, true>::kOutputChunkSize == 1LU);
        static_assert(Resampling<2LU, 1LU, true>::kIsConst == true);
        static_assert(Resampling<2LU, 1LU, true>::kEnabled == true);

        struct TestBlock0 : Block<TestBlock0> {
        } testBlock0;
        static_assert(std::is_const_v<decltype(testBlock0.input_chunk_size.value)>);
        static_assert(std::is_const_v<decltype(testBlock0.output_chunk_size.value)>);

        struct TestBlock1 : Block<TestBlock1, Resampling<>> {
        } testBlock1;
        static_assert(!std::is_const_v<decltype(testBlock1.input_chunk_size.value)>);
        static_assert(!std::is_const_v<decltype(testBlock1.output_chunk_size.value)>);

        struct TestBlock2 : Block<TestBlock2, Resampling<2LU, 1LU, true>> {
        } testBlock2;
        static_assert(std::is_const_v<decltype(testBlock2.input_chunk_size.value)>);
        static_assert(std::is_const_v<decltype(testBlock2.output_chunk_size.value)>);
        expect(eq(testBlock2.input_chunk_size, 2LU));
        expect(eq(testBlock2.output_chunk_size, 1LU));
    };

    "Stride"_test = [] {
        static_assert(Stride<>::kStride == 0LU);
        static_assert(Stride<>::kIsConst == false);
        static_assert(Stride<>::kEnabled == true);

        static_assert(Stride<2LU>::kStride == 2LU);
        static_assert(Stride<2LU>::kIsConst == false);
        static_assert(Stride<2LU>::kEnabled == true);

        static_assert(Stride<0LU, true>::kStride == 0LU);
        static_assert(Stride<0LU, true>::kIsConst == true);
        static_assert(Stride<0LU, true>::kEnabled == false);

        static_assert(Stride<1LU, true>::kStride == 1LU);
        static_assert(Stride<1LU, true>::kIsConst == true);
        static_assert(Stride<1LU, true>::kEnabled == true);

        struct TestBlock0 : Block<TestBlock0> {
        } testBlock0;
        static_assert(std::is_const_v<decltype(testBlock0.stride.value)>);

        struct TestBlock1 : Block<TestBlock1, gr::Stride<>> {
        } testBlock1;
        static_assert(!std::is_const_v<decltype(testBlock1.stride.value)>);

        struct TestBlock2 : Block<TestBlock2, gr::Stride<2LU, true>> {
        } testBlock2;
        static_assert(std::is_const_v<decltype(testBlock2.stride.value)>);
        expect(eq(testBlock2.stride, 2LU));
    };

    "User ResamplingRatio & Stride"_test = [] {
        using namespace gr;

        struct TestBlock : gr::Block<TestBlock, gr::Resampling<2LU, 1LU, true>, gr::Stride<2LU, false>> {
        } testBlock;
        static_assert(std::is_const_v<decltype(testBlock.input_chunk_size.value)>);
        static_assert(std::is_const_v<decltype(testBlock.output_chunk_size.value)>);
        static_assert(!std::is_const_v<decltype(testBlock.stride.value)>);
        expect(eq(testBlock.input_chunk_size, 2LU));
        expect(eq(testBlock.output_chunk_size, 1LU));
        expect(eq(testBlock.stride, 2LU));
    };

    "Interpolation/Decimation"_test = [&thread_pool] {
        interpolation_decimation_test({.n_samples = 1024, .output_chunk_size = 1, .input_chunk_size = 1, .exp_in = 1024, .exp_out = 1024, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 1024, .output_chunk_size = 1, .input_chunk_size = 2, .exp_in = 1024, .exp_out = 512, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 1024, .output_chunk_size = 2, .input_chunk_size = 1, .exp_in = 1024, .exp_out = 2048, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 1000, .output_chunk_size = 5, .input_chunk_size = 6, .exp_in = 996, .exp_out = 830, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 549, .output_chunk_size = 1, .input_chunk_size = 50, .exp_in = 500, .exp_out = 10, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 100, .output_chunk_size = 3, .input_chunk_size = 7, .exp_in = 98, .exp_out = 42, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 100, .output_chunk_size = 100, .input_chunk_size = 100, .exp_in = 100, .exp_out = 100, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 1000, .output_chunk_size = 10, .input_chunk_size = 1100, .exp_in = 0, .exp_out = 0, .exp_counter = 0}, thread_pool);
        interpolation_decimation_test({.n_samples = 1000, .output_chunk_size = 1, .input_chunk_size = 1001, .exp_in = 0, .exp_out = 0, .exp_counter = 0}, thread_pool);
        interpolation_decimation_test({.n_samples = 100, .output_chunk_size = 101, .input_chunk_size = 101, .exp_in = 0, .exp_out = 0, .exp_counter = 0}, thread_pool);
        interpolation_decimation_test({.n_samples = 100, .output_chunk_size = 5, .input_chunk_size = 11, .out_port_min = 10, .out_port_max = 41, .exp_in = 88, .exp_out = 40, .exp_counter = 1}, thread_pool);
        interpolation_decimation_test({.n_samples = 80, .output_chunk_size = 2, .input_chunk_size = 4, .out_port_min = 20, .out_port_max = 20, .exp_in = 40, .exp_out = 20, .exp_counter = 2}, thread_pool);
        interpolation_decimation_test({.n_samples = 100, .output_chunk_size = 7, .input_chunk_size = 3, .out_port_min = 10, .out_port_max = 20, .exp_in = 6, .exp_out = 14, .exp_counter = 16}, thread_pool);
    };

    "Stride tests"_test = [&thread_pool] {
        stride_test({.n_samples = 1024, .stride = 0, .in_port_max = 1024, .exp_in = 1024, .exp_out = 1024, .exp_counter = 1, .exp_total_in = 1024, .exp_total_out = 1024}, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 50, .input_chunk_size = 50, .stride = 100, .exp_in = 50, .exp_out = 50, .exp_counter = 10, .exp_total_in = 500, .exp_total_out = 500}, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 50, .input_chunk_size = 50, .stride = 133, .exp_in = 50, .exp_out = 50, .exp_counter = 8, .exp_total_in = 400, .exp_total_out = 400}, thread_pool);
        // the original test assumes that the incomplete chunk is also processed, currently we drop that. todo: switch to last sample update type incomplete
        // stride_test( {.n_samples = 1000, .stride =  50 , .in_port_max =  100 , .exp_in = 50 , .exp_out =   50 , .exp_counter = 20 , .exp_total_in = 1950 , .exp_total_out = 1950 }, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 100, .input_chunk_size = 100, .stride = 50, .exp_in = 100, .exp_out = 100, .exp_counter = 19, .exp_total_in = 1900, .exp_total_out = 1900}, thread_pool);
        // this one is tricky, it assumes that there are multiple incomplete last chunks :/ not sure what to do here...
        // stride_test( {.n_samples = 1000, .stride =  33 , .in_port_max = 100 , .exp_in =   10 , .exp_out =   10 , .exp_counter = 31 , .exp_total_in = 2929 , .exp_total_out = 2929 }, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 100, .input_chunk_size = 100, .stride = 33, .exp_in = 100, .exp_out = 100, .exp_counter = 28, .exp_total_in = 2800, .exp_total_out = 2800}, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 50, .input_chunk_size = 100, .stride = 50, .exp_in = 100, .exp_out = 50, .exp_counter = 19, .exp_total_in = 1900, .exp_total_out = 950}, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 25, .input_chunk_size = 50, .stride = 50, .exp_in = 1000, .exp_out = 500, .exp_counter = 1, .exp_total_in = 1000, .exp_total_out = 500}, thread_pool);
        stride_test({.n_samples = 1000, .output_chunk_size = 24, .input_chunk_size = 48, .stride = 50, .exp_in = 48, .exp_out = 24, .exp_counter = 20, .exp_total_in = 960, .exp_total_out = 480}, thread_pool);
        // std::vector<int> exp_v1 = {0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 12, 13, 14};
        // stride_test( {.n_samples = 15, .stride = 3, .in_port_max = 5, .exp_in = 3, .exp_out = 3, .exp_counter = 5, .exp_total_in = 23, .exp_total_out = 23, .exp_in_vector = exp_v1 }, thread_pool);
        std::vector<int> exp_v1 = {0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13};
        stride_test({.n_samples = 15, .output_chunk_size = 5, .input_chunk_size = 5, .stride = 3, .exp_in = 5, .exp_out = 5, .exp_counter = 4, .exp_total_in = 20, .exp_total_out = 20, .exp_in_vector = exp_v1}, thread_pool);
        std::vector<int> exp_v2 = {0, 1, 2, 5, 6, 7, 10, 11, 12};
        stride_test({.n_samples = 15, .output_chunk_size = 3, .input_chunk_size = 3, .stride = 5, .exp_in = 3, .exp_out = 3, .exp_counter = 3, .exp_total_in = 9, .exp_total_out = 9, .exp_in_vector = exp_v2}, thread_pool);
        // assuming buffer size is approx 65k
        stride_test({.n_samples = 1000000, .output_chunk_size = 100, .input_chunk_size = 100, .stride = 250000, .exp_in = 100, .exp_out = 100, .exp_counter = 4, .exp_total_in = 400, .exp_total_out = 400}, thread_pool);
        stride_test({.n_samples = 1000000, .output_chunk_size = 100, .input_chunk_size = 100, .stride = 249900, .exp_in = 100, .exp_out = 100, .exp_counter = 5, .exp_total_in = 500, .exp_total_out = 500}, thread_pool);
    };

    "SyncOrAsync ports tests"_test = [] {
        syncOrAsyncTest<true, true>();
        syncOrAsyncTest<false, true>();
        syncOrAsyncTest<true, false>();
        syncOrAsyncTest<false, false>();
    };

    "basic ports in arrays"_test = [] {
        using namespace gr::testing;
        using namespace std::string_literals;

        using TestNode = ArrayPortsNode<double>;

        const gr::Size_t nSamples = 5;

        gr::Graph                                                         graph;
        std::array<TagSource<double>*, 4>                                 sources;
        std::array<TagSink<double, ProcessFunction::USE_PROCESS_ONE>*, 4> sinks;

        auto* testNode = std::addressof(graph.emplaceBlock<TestNode>());

        sources[0] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{0.}}}));
        sources[1] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{1.}}}));
        sources[2] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{2.}}}));
        sources[3] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{3.}}}));

        sinks[0] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[1] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[2] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[3] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());

        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[0]).to<"inputs", 0UZ>(*testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[1]).to<"inputs", 1UZ>(*testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[2]).to<"inputs", 2UZ>(*testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[3]).to<"inputs", 3UZ>(*testNode)));

        // test also different connect API
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, "outputs#0"s, *sinks[0], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, "outputs#1"s, *sinks[1], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, "outputs#2"s, *sinks[2], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(*testNode, "outputs#3"s, *sinks[3], "in"s)));

        gr::scheduler::Simple sched{std::move(graph)};
        expect(sched.runAndWait().has_value());

        std::vector<std::vector<double>> expected_values{{0., 0., 0., 0., 0.}, {1., 1., 1., 1., 1.}, {2., 2., 2., 2., 2.}, {3., 3., 3., 3., 3.}};
        for (std::size_t i = 0UZ; i < sinks.size(); i++) {
            expect(sinks[i]->_nSamplesProduced == nSamples) << fmt::format("sinks[{}] mismatch in number of produced samples", i);
            expect(std::ranges::equal(sinks[i]->_samples, expected_values[i])) << fmt::format("sinks[{}]->_samples does not match to expected values", i);
        }
    };
};

const boost::ut::suite<"Drawable Annotations"> _drawableAnnotations = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    "drawable"_test = [] {
        struct TestBlock0 : gr::Block<TestBlock0> {
        } testBlock0;
        expect(!testBlock0.meta_information.value.contains("Drawable")) << "not drawable";

        struct TestBlock1 : gr::Block<TestBlock1, gr::Drawable<gr::UICategory::Toolbar, "console">> {
            gr::work::Status draw() { return gr::work::Status::OK; }
        } testBlock1;
        expect(testBlock1.meta_information.value.contains("Drawable")) << "drawable";
        const auto& drawableConfigMap = std::get<gr::property_map>(testBlock1.meta_information.value.at("Drawable"s));
        expect(drawableConfigMap.contains("Category"));
        expect(eq(std::get<std::string>(drawableConfigMap.at("Category")), "Toolbar"s));
        expect(drawableConfigMap.contains("Toolkit"));
        expect(eq(std::get<std::string>(drawableConfigMap.at("Toolkit")), "console"s));
    };
};

const boost::ut::suite<"Port MetaInfo Tests"> _portMetaInfoTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    using namespace gr;

    "constructor test"_test = [] {
        // Test the initializer list constructor
        PortMetaInfo portMetaInfo({{"sample_rate", 48000.f}, //
            {"signal_name", "TestSignal"}, {"signal_quantity", "voltage"}, {"signal_unit", "V"}, {"signal_min", -1.f}, {"signal_max", 1.f}});

        expect(eq(48000.f, portMetaInfo.sample_rate.value));
        expect(eq("TestSignal"s, portMetaInfo.signal_name.value));
        expect(eq("voltage"s, portMetaInfo.signal_quantity.value));
        expect(eq("V"s, portMetaInfo.signal_unit.value));
        expect(eq(-1.f, portMetaInfo.signal_min.value));
        expect(eq(+1.f, portMetaInfo.signal_max.value));
    };

    "reset test"_test = [] {
        PortMetaInfo portMetaInfo;
        portMetaInfo.auto_update.clear();
        expect(portMetaInfo.auto_update.empty());

        portMetaInfo.reset();

        expect(portMetaInfo.auto_update.contains("sample_rate"));
        expect(portMetaInfo.auto_update.contains("signal_name"));
        expect(portMetaInfo.auto_update.contains("signal_quantity"));
        expect(portMetaInfo.auto_update.contains("signal_unit"));
        expect(portMetaInfo.auto_update.contains("signal_min"));
        expect(portMetaInfo.auto_update.contains("signal_max"));
        expect(eq(portMetaInfo.auto_update.size(), 6UZ));
    };

    "update test"_test = [] {
        PortMetaInfo portMetaInfo;
        property_map updateProps{{"sample_rate", 96000.f}, {"signal_name", "UpdatedSignal"}};
        portMetaInfo.update(updateProps);

        expect(eq(96000.f, portMetaInfo.sample_rate));
        expect(eq("UpdatedSignal"s, portMetaInfo.signal_name));
    };

    "get test"_test = [] {
        PortMetaInfo portMetaInfo({{"sample_rate", 48000.f}, {"signal_name", "TestSignal"}});
        const auto   props = portMetaInfo.get();

        expect(eq(48000.f, std::get<float>(props.at("sample_rate"))));
        expect(eq("TestSignal"s, std::get<std::string>(props.at("signal_name"))));
    };
};

const boost::ut::suite<"Requested Work Tests"> _requestedWorkTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "work() test"_test = [] {
        gr::Size_t nSamples = 1000000;

        gr::Graph graph;
        auto&     src   = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"disconnect_on_done", false}});
        auto&     block = graph.emplaceBlock<IntDecBlock<float>>({{"disconnect_on_done", false}});
        auto&     sink  = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"disconnect_on_done", false}});

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src).to<"in">(block)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(block).template to<"in">(sink)));

        graph.reconnectAllEdges();
        auto blockInit = [](auto& block) {
            if (block.state() == lifecycle::State::IDLE) {
                std::ignore = block.changeStateTo(lifecycle::State::INITIALISED);
            }
            std::ignore = block.changeStateTo(lifecycle::State::RUNNING);
            expect(block.state() == lifecycle::State::RUNNING);
        };
        blockInit(src);
        blockInit(block);
        blockInit(sink);

        auto resultSrc = src.work(100);
        expect(eq(resultSrc.performed_work, 100UZ));

        auto resultBlock = block.work(10); // requestedWork is applied, process 10 samples
        expect(eq(resultBlock.requested_work, 10UZ));
        expect(eq(resultBlock.performed_work, 10UZ));

        expect(block.settings().set({{"output_chunk_size", gr::Size_t(7)}, {"input_chunk_size", gr::Size_t(7)}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        resultBlock = block.work(8); // requestedWork is applied, process only one `input_chunk_size` which fits to requestedWork
        expect(eq(resultBlock.requested_work, 8UZ));
        expect(eq(resultBlock.performed_work, 7UZ));
        resultBlock = block.work(28); // requestedWork is applied, process 4 `input_chunk_size` which fits to requestedWork
        expect(eq(resultBlock.requested_work, 28UZ));
        expect(eq(resultBlock.performed_work, 28UZ));
        resultBlock = block.work(5); // requestedWork is clamped to `input_chunk_size`
        expect(eq(resultBlock.requested_work, 5UZ));
        expect(eq(resultBlock.performed_work, 7UZ)); // 7 samples are processed
        expect(block.settings().set({{"output_chunk_size", gr::Size_t(1)}, {"input_chunk_size", gr::Size_t(1)}}).empty());
        expect(block.settings().activateContext() != std::nullopt);
        resultBlock = block.work(); // process last 48 samples
        expect(eq(resultBlock.requested_work, std::numeric_limits<std::size_t>::max()));
        expect(eq(resultBlock.performed_work, 48UZ));
        auto resultSink = sink.work(100);
        expect(eq(resultSink.performed_work, 100UZ));
    };
};

const boost::ut::suite<"reflFirstTypeName Tests"> _reflFirstTypeNameTests = [] {
    using namespace boost::ut;
    using namespace gr::detail;
    using namespace std::literals::string_literals;

    "std::complex"_test = []() {
        expect(eq(reflFirstTypeName<std::complex<double>>(),
                  "std::complex<double>"s));
        expect(eq(reflFirstTypeName<std::complex<float>>(),
                  "std::complex<float>"s));
    };
};

int main() { /* not needed for UT */ }
