#include <utility>
#include <vector>

#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#if !DISABLE_SIMD
namespace gr::test {
struct copy : public Block<copy> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(copy, in, out);

public:
    template<meta::t_or_simd<float> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        return a;
    }
};
} // namespace gr::test

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

    GR_MAKE_REFLECTABLE(BlockSignaturesNone, in, out);
};
static_assert(!gr::HasRequiredProcessFunction<BlockSignaturesNone<float>>);

template<typename T>
struct BlockSignaturesVoid : public gr::Block<BlockSignaturesVoid<T>> {
    T value;

    GR_MAKE_REFLECTABLE(BlockSignaturesVoid, value);

    void processOne() {}
};
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid<float>>);

template<typename T>
struct BlockSignaturesVoid2 : public gr::Block<BlockSignaturesVoid2<T>> {
    T value;

    GR_MAKE_REFLECTABLE(BlockSignaturesVoid2, value);

    gr::work::Status processBulk() { return gr::work::Status::OK; }
};
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesVoid2<float>>);

template<typename T>
struct BlockSignaturesProcessOne : public gr::Block<BlockSignaturesProcessOne<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BlockSignaturesProcessOne, in, out);

    T processOne(T) { return T(); }
};
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessOne<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesProcessOne<float>>);
static_assert(!gr::HasConstProcessOneFunction<BlockSignaturesProcessOne<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessOne<float>>);

template<typename T>
struct BlockSignaturesProcessOneConst : public gr::Block<BlockSignaturesProcessOneConst<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BlockSignaturesProcessOneConst, in, out);

    T processOne(T) const { return T(); }
};
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(gr::HasConstProcessOneFunction<BlockSignaturesProcessOneConst<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesProcessOneConst<float>>);

template<typename T>
struct BlockSignaturesTemplatedProcessOneConst : public gr::Block<BlockSignaturesTemplatedProcessOneConst<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BlockSignaturesTemplatedProcessOneConst, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(const V& /*input*/) const noexcept {
        return V();
    }
};
static_assert(gr::HasRequiredProcessFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(gr::HasProcessOneFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(gr::HasConstProcessOneFunction<BlockSignaturesTemplatedProcessOneConst<float>>);
static_assert(!gr::HasProcessBulkFunction<BlockSignaturesTemplatedProcessOneConst<float>>);

enum class ProcessBulkVariant { STD_STD, STD_STD_REF, STD_OUTPUT, STD_OUTPUT_REF, INPUT_STD, INPUT_STD_REF, INPUT_OUTPUT, INPUT_OUTPUT_REF };

template<typename T, ProcessBulkVariant processVariant>
struct BlockSignaturesProcessBulkSpan : public gr::Block<BlockSignaturesProcessBulkSpan<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    GR_MAKE_REFLECTABLE(BlockSignaturesProcessBulkSpan, in, out);

    gr::work::Status processBulk(std::span<const T>, std::span<T>)
    requires(processVariant == ProcessBulkVariant::STD_STD)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>&, std::span<T>&)
    requires(processVariant == ProcessBulkVariant::STD_STD_REF)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::OutputSpanLike auto)
    requires(processVariant == ProcessBulkVariant::STD_OUTPUT)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>&, gr::OutputSpanLike auto&)
    requires(processVariant == ProcessBulkVariant::STD_OUTPUT_REF)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto, std::span<T>)
    requires(processVariant == ProcessBulkVariant::INPUT_STD)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto&, std::span<T>&)
    requires(processVariant == ProcessBulkVariant::INPUT_STD_REF)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto, gr::OutputSpanLike auto)
    requires(processVariant == ProcessBulkVariant::INPUT_OUTPUT)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto&, gr::OutputSpanLike auto&)
    requires(processVariant == ProcessBulkVariant::INPUT_OUTPUT_REF)
    {
        return gr::work::Status::OK;
    }
};

static_assert(gr::HasRequiredProcessFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD>>);
static_assert(!gr::HasProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD>>);
static_assert(!gr::HasConstProcessOneFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD>>);

static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD_REF>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_OUTPUT>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_OUTPUT_REF>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::INPUT_STD>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::INPUT_STD_REF>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::INPUT_OUTPUT>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::INPUT_OUTPUT_REF>>);

// static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_STD>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkSpan<float, ProcessBulkVariant::STD_OUTPUT>, 0>);

enum class ProcessBulkTwoOutsVariant { STD_STD, OUTPUT_STD, OUTPUT_OUTPUT, STD_OUTPUT };

template<typename T, ProcessBulkTwoOutsVariant processVariant>
struct BlockSignaturesProcessBulkTwoOuts : public gr::Block<BlockSignaturesProcessBulkTwoOuts<T, processVariant>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out1{};
    gr::PortOut<T> out2{};

    GR_MAKE_REFLECTABLE(BlockSignaturesProcessBulkTwoOuts, in, out1, out2);

    gr::work::Status processBulk(std::span<const T>, std::span<T>, std::span<T>)
    requires(processVariant == ProcessBulkTwoOutsVariant::STD_STD)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::OutputSpanLike auto, std::span<T>)
    requires(processVariant == ProcessBulkTwoOutsVariant::OUTPUT_STD)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, gr::OutputSpanLike auto, gr::OutputSpanLike auto)
    requires(processVariant == ProcessBulkTwoOutsVariant::OUTPUT_OUTPUT)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<const T>, std::span<T>, gr::OutputSpanLike auto)
    requires(processVariant == ProcessBulkTwoOutsVariant::STD_OUTPUT)
    {
        return gr::work::Status::OK;
    }
};

static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::STD_STD>, 0>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::STD_STD>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::OUTPUT_STD>, 0>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::OUTPUT_STD>, 1>);
static_assert(gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::STD_OUTPUT>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::STD_OUTPUT>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::OUTPUT_OUTPUT>, 0>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::OUTPUT_OUTPUT>, 1>);
static_assert(!gr::traits::block::processBulk_requires_ith_output_as_span<BlockSignaturesProcessBulkTwoOuts<float, ProcessBulkTwoOutsVariant::STD_STD>, 2>); // out-of-range check

enum class ProcessBulkVectorVariant { STD_STD, STD_STD_REF, INPUT_STD, STD_OUTPUT, INPUT_OUTPUT, INPUT_OUTPUT_REF };

template<typename T, ProcessBulkVectorVariant processVariant>
struct BlockSignaturesProcessBulkVector : public gr::Block<BlockSignaturesProcessBulkVector<T, processVariant>> {
    std::vector<gr::PortIn<T>>  inputs{};
    std::vector<gr::PortOut<T>> outputs{};

    GR_MAKE_REFLECTABLE(BlockSignaturesProcessBulkVector, inputs, outputs);

    gr::work::Status processBulk(std::span<std::span<const T>>, std::span<std::span<T>>)
    requires(processVariant == ProcessBulkVectorVariant::STD_STD)
    {
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(std::span<std::span<const T>>&, std::span<std::span<T>>&)
    requires(processVariant == ProcessBulkVectorVariant::STD_STD_REF)
    {
        return gr::work::Status::OK;
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(std::span<TInput>&, std::span<std::span<T>>&)
    requires(processVariant == ProcessBulkVectorVariant::INPUT_STD)
    {
        return gr::work::Status::OK;
    }

    template<gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(std::span<std::span<const T>>&, std::span<TOutput>&)
    requires(processVariant == ProcessBulkVectorVariant::STD_OUTPUT)
    {
        return gr::work::Status::OK;
    }

    template<gr::InputSpanLike TInput, gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(std::span<TInput>, std::span<TOutput>)
    requires(processVariant == ProcessBulkVectorVariant::INPUT_OUTPUT)
    {
        return gr::work::Status::OK;
    }

    template<gr::InputSpanLike TInput, gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(std::span<TInput>&, std::span<TOutput>&)
    requires(processVariant == ProcessBulkVectorVariant::INPUT_OUTPUT_REF)
    {
        return gr::work::Status::OK;
    }
};

static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::STD_STD>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::STD_STD_REF>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::INPUT_STD>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::STD_OUTPUT>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::INPUT_OUTPUT>>);
static_assert(gr::HasProcessBulkFunction<BlockSignaturesProcessBulkVector<float, ProcessBulkVectorVariant::INPUT_OUTPUT_REF>>);

struct InvalidSettingBlock : gr::Block<InvalidSettingBlock> {
    std::tuple<int> tuple; // this type is not supported and should cause the checkBlockContracts<T>() to throw
                           //
    GR_MAKE_REFLECTABLE(InvalidSettingBlock, tuple);
};

struct MissingProcessSignature1 : gr::Block<MissingProcessSignature1> {
    gr::PortIn<int>    in;
    gr::PortOut<int>   out0;
    gr::PortOut<float> out1;

    GR_MAKE_REFLECTABLE(MissingProcessSignature1, in, out0, out1);
};

struct MissingProcessSignature2 : gr::Block<MissingProcessSignature2> {
    gr::PortIn<int>    in0;
    gr::PortIn<float>  in1;
    gr::PortOut<int>   out0;
    gr::PortOut<float> out1;

    GR_MAKE_REFLECTABLE(MissingProcessSignature2, in0, in1, out0, out1);
};

struct MissingProcessSignature3 : gr::Block<MissingProcessSignature3> {
    std::vector<gr::PortOut<float>>   outA;
    std::array<gr::PortOut<float>, 2> outB;

    template<gr::OutputSpanLike TOutputSpan2>
    gr::work::Status processBulk(std::span<std::vector<float>>&, std::span<TOutputSpan2>&) {
        return gr::work::Status::OK;
    }

    GR_MAKE_REFLECTABLE(MissingProcessSignature3, outA, outB);
};

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

    GR_MAKE_REFLECTABLE(IntDecBlock, in, out);

    ProcessStatus status{};
    bool          write_to_vector{false};

    gr::work::Status processBulk(std::span<const T>& input, std::span<T>& output) noexcept {
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

    GR_MAKE_REFLECTABLE(SyncOrAsyncBlock, in, out);

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan) {
        const auto available = std::min(inSpan.size(), outSpan.size());
        if (available != 0) {
            std::copy(inSpan.begin(), std::next(inSpan.begin(), static_cast<std::ptrdiff_t>(available)), outSpan.begin());
        }
        outSpan.publish(available);
        boost::ut::expect(inSpan.tryConsume(available)) << "Samples were not consumed";
        return gr::work::Status::OK;
    }
};
static_assert(gr::HasProcessBulkFunction<SyncOrAsyncBlock<float, true, true>>);

template<typename T>
struct VectorPortsBlock : gr::Block<VectorPortsBlock<T>> {
    static constexpr std::size_t nPorts = 4;

    std::vector<gr::PortIn<T, gr::Async>>  input{nPorts};
    std::vector<gr::PortOut<T, gr::Async>> output{nPorts};

    GR_MAKE_REFLECTABLE(VectorPortsBlock, input, output);

    template<gr::InputSpanLike TInSpan, gr::OutputSpanLike TOutSpan>
    gr::work::Status processBulk(std::span<TInSpan>& ins, std::span<TOutSpan>& outs) {
        auto available = std::min(ins[0].size(), outs[0].size());
        std::copy_n(ins[0].begin(), available, outs[0].begin());
        std::ignore = ins[0].consume(available);
        outs[0].publish(available);

        available = std::min(ins[1].size(), outs[1].size());
        std::copy_n(ins[1].begin(), available, outs[1].begin());
        std::ignore = ins[1].consume(available);
        outs[1].publish(available);

        available = std::min(ins[2].size(), outs[2].size());
        std::copy_n(ins[2].begin(), available, outs[2].begin());
        std::ignore = ins[2].consume(available);
        outs[2].publish(available);

        available = std::min(ins[3].size(), outs[3].size());
        std::copy_n(ins[3].begin(), available, outs[3].begin());
        std::ignore = ins[3].consume(available);
        outs[3].publish(available);

        return gr::work::Status::OK;
    }
};
static_assert(gr::HasProcessBulkFunction<VectorPortsBlock<int>>);

template<typename T>
struct ArrayPortsBlock : gr::Block<ArrayPortsBlock<T>> {
    static constexpr std::size_t nPorts = 4;

    std::array<gr::PortIn<T, gr::Async>, nPorts>  input;
    std::array<gr::PortOut<T, gr::Async>, nPorts> output;

    GR_MAKE_REFLECTABLE(ArrayPortsBlock, input, output);

    template<gr::InputSpanLike TInSpan, gr::OutputSpanLike TOutSpan>
    gr::work::Status processBulk(std::span<TInSpan>& ins, std::span<TOutSpan>& outs) {
        auto available = std::min(ins[0].size(), outs[0].size());
        std::copy_n(ins[0].begin(), available, outs[0].begin());
        std::ignore = ins[0].consume(available);
        outs[0].publish(available);

        available = std::min(ins[1].size(), outs[1].size());
        std::copy_n(ins[1].begin(), available, outs[1].begin());
        std::ignore = ins[1].consume(available);
        outs[1].publish(available);

        available = std::min(ins[2].size(), outs[2].size());
        std::copy_n(ins[2].begin(), available, outs[2].begin());
        std::ignore = ins[2].consume(available);
        outs[2].publish(available);

        available = std::min(ins[3].size(), outs[3].size());
        std::copy_n(ins[3].begin(), available, outs[3].begin());
        std::ignore = ins[3].consume(available);
        outs[3].publish(available);

        return gr::work::Status::OK;
    }
};
static_assert(gr::HasProcessBulkFunction<ArrayPortsBlock<int>>);

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

    "Interpolation/Decimation with many tags, tags forward policy"_test = [] {
        using namespace boost::ut;
        using namespace gr::testing;

        gr::Graph testGraph;
        auto&     source = testGraph.emplaceBlock<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(20)}});
        source._tags     = {
            {0, {{"key0", "value@0"}}},    //
            {2, {{"key2", "value@2"}}},    //
            {4, {{"key4", "value@4"}}},    //
            {6, {{"key6", "value@6"}}},    //
            {8, {{"ke8", "value@8"}}},     //
            {10, {{"key10", "value@10"}}}, //
            {12, {{"key12", "value@12"}}}, //
            {14, {{"key14", "value@14"}}}  //
        };

        auto& intDecBlock = testGraph.emplaceBlock<IntDecBlock<int>>({{"output_chunk_size", gr::Size_t(10)}, {"input_chunk_size", gr::Size_t(10)}});
        auto& sink        = testGraph.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>();
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(source).to<"in">(intDecBlock)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(intDecBlock).to<"in">(sink)));

        gr::scheduler::Simple sched{std::move(testGraph)};
        expect(sched.runAndWait().has_value());

        expect(eq(intDecBlock.status.process_counter, 2UZ));
        expect(eq(intDecBlock.status.total_in, 20UZ));
        expect(eq(intDecBlock.status.total_out, 20UZ));
    };

    "SyncOrAsync ports tests"_test = [] {
        syncOrAsyncTest<true, true>();
        syncOrAsyncTest<false, true>();
        syncOrAsyncTest<true, false>();
        syncOrAsyncTest<false, false>();
    };

    "basic ports in vectors"_test = [] {
        using namespace gr::testing;
        using namespace std::string_literals;

        using TestNode = VectorPortsBlock<double>;

        const gr::Size_t nSamples = 5;

        gr::Graph                                                         graph;
        std::array<TagSource<double>*, 4>                                 sources;
        std::array<TagSink<double, ProcessFunction::USE_PROCESS_ONE>*, 4> sinks;

        auto& testNode = graph.emplaceBlock<TestNode>();

        sources[0] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{0.}}}));
        sources[1] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{1.}}}));
        sources[2] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{2.}}}));
        sources[3] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{3.}}}));

        sinks[0] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[1] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[2] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[3] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());

        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[0]).to<"input", 0>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[1]).to<"input", 1>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[2]).to<"input", 2>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[3]).to<"input", 3>(testNode)));

        // test also different connect API
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#0"s, *sinks[0], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#1"s, *sinks[1], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#2"s, *sinks[2], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#3"s, *sinks[3], "in"s)));

        gr::scheduler::Simple sched{std::move(graph)};
        expect(sched.runAndWait().has_value());

        std::vector<std::vector<double>> expected_values{{0., 0., 0., 0., 0.}, {1., 1., 1., 1., 1.}, {2., 2., 2., 2., 2.}, {3., 3., 3., 3., 3.}};
        for (std::size_t i = 0UZ; i < sinks.size(); i++) {
            expect(sinks[i]->_nSamplesProduced == nSamples) << fmt::format("sinks[{}] mismatch in number of produced samples", i);
            expect(std::ranges::equal(sinks[i]->_samples, expected_values[i])) << fmt::format("sinks[{}]->_samples does not match to expected values", i);
        }
    };

    "basic ports in arrays"_test = [] {
        using namespace gr::testing;
        using namespace std::string_literals;

        using TestNode = ArrayPortsBlock<double>;

        const gr::Size_t nSamples = 5;

        gr::Graph                                                         graph;
        std::array<TagSource<double>*, 4>                                 sources;
        std::array<TagSink<double, ProcessFunction::USE_PROCESS_ONE>*, 4> sinks;

        auto& testNode = graph.emplaceBlock<TestNode>();

        sources[0] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{0.}}}));
        sources[1] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{1.}}}));
        sources[2] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{2.}}}));
        sources[3] = std::addressof(graph.emplaceBlock<TagSource<double>>({{"n_samples_max", nSamples}, {"values", std::vector{3.}}}));

        sinks[0] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[1] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[2] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());
        sinks[3] = std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_ONE>>());

        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[0]).to<"input", 0>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[1]).to<"input", 1>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[2]).to<"input", 2>(testNode)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(*sources[3]).to<"input", 3>(testNode)));

        // test also different connect API
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#0"s, *sinks[0], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#1"s, *sinks[1], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#2"s, *sinks[2], "in"s)));
        expect(eq(gr::ConnectionResult::SUCCESS, graph.connect(testNode, "output#3"s, *sinks[3], "in"s)));

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
        PortMetaInfo portMetaInfo({{gr::tag::SAMPLE_RATE.shortKey(), 48000.f}, //
            {gr::tag::SIGNAL_NAME.shortKey(), "TestSignal"}, {gr::tag::SIGNAL_QUANTITY.shortKey(), "voltage"}, {gr::tag::SIGNAL_UNIT.shortKey(), "V"}, {gr::tag::SIGNAL_MIN.shortKey(), -1.f}, {gr::tag::SIGNAL_MAX.shortKey(), 1.f}});

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

        expect(portMetaInfo.auto_update.contains(gr::tag::SAMPLE_RATE.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_NAME.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_QUANTITY.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_UNIT.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_MIN.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_MAX.shortKey()));
        expect(eq(portMetaInfo.auto_update.size(), 16UZ));
    };

    "update test"_test = [] {
        PortMetaInfo portMetaInfo;
        property_map updateProps{{gr::tag::SAMPLE_RATE.shortKey(), 96000.f}, {gr::tag::SIGNAL_NAME.shortKey(), "UpdatedSignal"}};
        portMetaInfo.update(updateProps);

        expect(eq(96000.f, portMetaInfo.sample_rate));
        expect(eq("UpdatedSignal"s, portMetaInfo.signal_name));
    };

    "get test"_test = [] {
        PortMetaInfo portMetaInfo({{gr::tag::SAMPLE_RATE.shortKey(), 48000.f}, {gr::tag::SIGNAL_NAME.shortKey(), "TestSignal"}});
        const auto   props = portMetaInfo.get();

        expect(eq(48000.f, std::get<float>(props.at(gr::tag::SAMPLE_RATE.shortKey()))));
        expect(eq("TestSignal"s, std::get<std::string>(props.at(gr::tag::SIGNAL_NAME.shortKey()))));
    };
};

const boost::ut::suite<"Requested Work Tests"> _requestedWorkTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    "work() test"_test = [] {
        gr::Size_t nSamples = 1000000;

        gr::Graph graph;
        auto&     src       = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"disconnect_on_done", false}});
        auto&     testBlock = graph.emplaceBlock<IntDecBlock<float>>({{"disconnect_on_done", false}});
        auto&     sink      = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"disconnect_on_done", false}});

        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(src).to<"in">(testBlock)));
        expect(eq(ConnectionResult::SUCCESS, graph.connect<"out">(testBlock).template to<"in">(sink)));

        graph.reconnectAllEdges();
        auto blockInit = [](auto& blockToInit) {
            if (blockToInit.state() == lifecycle::State::IDLE) {
                std::ignore = blockToInit.changeStateTo(lifecycle::State::INITIALISED);
            }
            std::ignore = blockToInit.changeStateTo(lifecycle::State::RUNNING);
            expect(blockToInit.state() == lifecycle::State::RUNNING);
        };
        blockInit(src);
        blockInit(testBlock);
        blockInit(sink);

        auto resultSrc = src.work(100);
        expect(eq(resultSrc.performed_work, 100UZ));

        auto resultBlock = testBlock.work(10); // requestedWork is applied, process 10 samples
        expect(eq(resultBlock.requested_work, 10UZ));
        expect(eq(resultBlock.performed_work, 10UZ));

        expect(testBlock.settings().set({{"output_chunk_size", gr::Size_t(7)}, {"input_chunk_size", gr::Size_t(7)}}).empty());
        expect(testBlock.settings().activateContext() != std::nullopt);
        resultBlock = testBlock.work(8); // requestedWork is applied, process only one `input_chunk_size` which fits to requestedWork
        expect(eq(resultBlock.requested_work, 8UZ));
        expect(eq(resultBlock.performed_work, 7UZ));
        resultBlock = testBlock.work(28); // requestedWork is applied, process 4 `input_chunk_size` which fits to requestedWork
        expect(eq(resultBlock.requested_work, 28UZ));
        expect(eq(resultBlock.performed_work, 28UZ));
        resultBlock = testBlock.work(5); // requestedWork is clamped to `input_chunk_size`
        expect(eq(resultBlock.requested_work, 5UZ));
        expect(eq(resultBlock.performed_work, 7UZ)); // 7 samples are processed
        expect(testBlock.settings().set({{"output_chunk_size", gr::Size_t(1)}, {"input_chunk_size", gr::Size_t(1)}}).empty());
        expect(testBlock.settings().activateContext() != std::nullopt);
        resultBlock = testBlock.work(); // process last 48 samples
        expect(eq(resultBlock.requested_work, std::numeric_limits<std::size_t>::max()));
        expect(eq(resultBlock.performed_work, 48UZ));
        auto resultSink = sink.work(100);
        expect(eq(resultSink.performed_work, 100UZ));
    };
};

const boost::ut::suite<"BlockingIO Tests"> _blockingIOTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace std::chrono_literals;
    using namespace gr::basic;

    "Test BlockingIO"_test = [] {
        // This test demonstrates how to properly verify that a BlockingIO block has finished execution.
        // The main issue is that BlockingIO blocks run in a separate thread and may continue executing for some time after join().
        // Standard detection mechanisms might not always accurately determine the block's completion status.
        // Therefore, we need to implement additional checks to ensure that the BlockingIO block has fully stopped.

        gr::Graph flow;
        // ClockSource has a BlockingIO attribute
        auto& source  = flow.emplaceBlock<ClockSource<float>>({{gr::tag::SAMPLE_RATE.shortKey(), 10.f}, {"n_samples_max", gr::Size_t(0)}});
        auto& monitor = flow.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
        auto& sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(monitor)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(monitor).to<"in">(sink)));

        auto scheduler = scheduler::Simple(std::move(flow));

        auto client = std::thread([&scheduler] {
            const auto startTime = std::chrono::steady_clock::now();
            auto       isExpired = [&startTime] { return std::chrono::steady_clock::now() - startTime > 3s; };
            bool       expired   = false;
            while (!expired) {
                expired = isExpired();
                std::this_thread::sleep_for(100ms);
            }
            scheduler.requestStop();
        });

        auto schedulerThread = std::thread([&scheduler] { scheduler.runAndWait(); });
        client.join();

        // Additional check to be sure that ClockSource is in STOPPED state.
        while (source.state() != lifecycle::State::STOPPED) {
            std::this_thread::sleep_for(10ms);
        }
        schedulerThread.join();
    };
};

const boost::ut::suite<"reflFirstTypeName Tests"> _reflFirstTypeNameTests = [] {
    using namespace boost::ut;
    using namespace gr::detail;
    using namespace std::literals::string_view_literals;

    "std::complex"_test = []() {
        expect(eq(gr::refl::type_name<std::complex<double>>.view(), "std::complex<double>"sv));
        expect(eq(gr::refl::type_name<std::complex<float>>.view(), "std::complex<float>"sv));
    };
};

int main() { /* not needed for UT */ }
