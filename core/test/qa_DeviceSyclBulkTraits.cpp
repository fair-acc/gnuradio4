#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>

namespace gr::test {

struct TemplatedSyclBulk : Block<TemplatedSyclBulk> {
    PortIn<float>  in;
    PortOut<float> out;

    GR_MAKE_REFLECTABLE(TemplatedSyclBulk, in, out);

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        std::ignore = input.consume(0);
        output.publish(0);
        return work::Status::OK;
    }

    [[nodiscard]] work::Status processBulk_sycl(gr::device::SyclQueue&, InputSpanLike auto& input, OutputSpanLike auto& output) {
        std::ignore = input.consume(0);
        output.publish(0);
        return work::Status::OK;
    }
};

struct SyclBulkOnly {
    [[nodiscard]] work::Status processBulk_sycl(gr::device::SyclQueue&, InputSpanLike auto& input, OutputSpanLike auto& output) {
        std::ignore = input.consume(0);
        output.publish(0);
        return work::Status::OK;
    }
};

// item #7: the sycl bulk hatch probes an N-ary signature but used to dispatch via std::get<0> only; a two-input
// block pins the fix (see gr::device::detail::invokeProcessBulkSycl below)
struct TwoInputSyclBulk {
    [[nodiscard]] work::Status processBulk_sycl(gr::device::SyclQueue&, InputSpanLike auto& inputA, InputSpanLike auto& inputB, OutputSpanLike auto& output) {
        std::ignore = inputA.consume(0);
        std::ignore = inputB.consume(0);
        output.publish(0);
        return work::Status::OK;
    }
};

/// the CPU fallback's N-ary arms are plain host code, so they are checked here rather than behind a GPU: both
/// results asymmetric in both arguments, so a swapped input or a swapped output shows up in the numbers
struct CrossMixHost {
    [[nodiscard]] constexpr std::tuple<float, float> processOne(float a, float b) const noexcept { return {a - 2.f * b, 3.f * a + b}; }
};

struct TwoInBulkHost {
    [[nodiscard]] work::Status processBulk(std::span<const float> a, std::span<const float> b, std::span<float> out) const noexcept {
        for (std::size_t i = 0UZ; i < out.size(); ++i) {
            out[i] = a[i] - 2.f * b[i];
        }
        return work::Status::OK;
    }
};

/// a source and a sink with more than one port: the fallback used to serve these shapes only at one port a side
struct TwoOutSourceHost {
    [[nodiscard]] constexpr std::tuple<float, float> processOne() const noexcept { return {2.f, 5.f}; }
};

struct TwoInSinkHost {
    mutable float total = 0.f;

    constexpr void processOne(float a, float b) const noexcept { total += a - 2.f * b; }
};

} // namespace gr::test

using InputSpans    = std::tuple<gr::traits::block::detail::DummyInputSpan<const float>>;
using OutputSpans   = std::tuple<gr::traits::block::detail::DummyOutputSpan<float>>;
using TwoInputSpans = std::tuple<gr::traits::block::detail::DummyInputSpan<const float>, gr::traits::block::detail::DummyInputSpan<const float>>;

static_assert(gr::device::HasSyclBulkForSpans<gr::test::TemplatedSyclBulk, InputSpans, OutputSpans>);
static_assert(gr::device::ExecutionStrategy<gr::test::TemplatedSyclBulk>::canDispatch<InputSpans, OutputSpans>());
static_assert(gr::device::HasSyclBulkForSpans<gr::test::SyclBulkOnly, InputSpans, OutputSpans>);
static_assert(gr::device::HasSyclBulkForSpans<gr::test::TwoInputSyclBulk, TwoInputSpans, OutputSpans>);
static_assert(gr::device::ExecutionStrategy<gr::test::TwoInputSyclBulk>::canDispatch<TwoInputSpans, OutputSpans>());

namespace {

void detectDispatchError(const gr::log::LogRecord& record, void* user) noexcept {
    auto&                  saw = *static_cast<bool*>(user);
    const std::string_view text{record.text, record.textLength};
    saw = saw || (record.level == gr::log::Level::error && text.contains("no backend is wired"));
}

} // namespace

int main() {
    // the N-ary CPU fallback, exercised without a device: these arms run on every build, unlike the GPU tests
    {
        const std::vector<float> a{1.f, 2.f, 3.f};
        const std::vector<float> b{10.f, 20.f, 30.f};
        std::vector<float>       out0(3UZ, 0.f);
        std::vector<float>       out1(3UZ, 0.f);
        auto                     ins  = std::tuple{std::span<const float>{a}, std::span<const float>{b}};
        auto                     outs = std::tuple{std::span<float>{out0}, std::span<float>{out1}};

        gr::test::CrossMixHost crossMix;
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            gr::device::detail::invokeProcessOneOverSpans(crossMix, ins, outs, i, std::make_index_sequence<2UZ>(), std::make_index_sequence<2UZ>());
        }
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            if (out0[i] != a[i] - 2.f * b[i] || out1[i] != 3.f * a[i] + b[i]) {
                return 4; // an input reached the wrong parameter, or a result the wrong port
            }
        }

        // a source has an empty input pack, a sink an empty output pack -- both were unreachable for N > 1 ports
        std::vector<float> src0(2UZ, 0.f);
        std::vector<float> src1(2UZ, 0.f);
        auto               noInputs  = std::tuple{};
        auto               srcOuts   = std::tuple{std::span<float>{src0}, std::span<float>{src1}};
        gr::test::TwoOutSourceHost source;
        for (std::size_t i = 0UZ; i < src0.size(); ++i) {
            gr::device::detail::invokeProcessOneOverSpans(source, noInputs, srcOuts, i, std::make_index_sequence<0UZ>(), std::make_index_sequence<2UZ>());
        }
        if (src0[0] != 2.f || src0[1] != 2.f || src1[0] != 5.f || src1[1] != 5.f) {
            return 7; // a source's results did not reach the ports that name them
        }

        auto                  noOutputs = std::tuple{};
        gr::test::TwoInSinkHost sink;
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            gr::device::detail::invokeProcessOneOverSpans(sink, ins, noOutputs, i, std::make_index_sequence<2UZ>(), std::make_index_sequence<0UZ>());
        }
        if (sink.total != (1.f - 20.f) + (2.f - 40.f) + (3.f - 60.f)) {
            return 8; // a sink did not receive every declared input port
        }

        std::vector<float> bulkOut(3UZ, 0.f);
        auto               bulkOuts = std::tuple{std::span<float>{bulkOut}};
        gr::test::TwoInBulkHost twoInBulk;
        if (gr::device::detail::invokeBulkOverSpans(twoInBulk, ins, bulkOuts, std::make_index_sequence<2UZ>(), std::make_index_sequence<1UZ>()) != gr::work::Status::OK) {
            return 5;
        }
        for (std::size_t i = 0UZ; i < a.size(); ++i) {
            if (bulkOut[i] != a[i] - 2.f * b[i]) {
                return 6; // the bulk fallback did not hand every declared port to the body
            }
        }
    }

    // proves the dispatch-time helper (not just the probe) actually expands both input spans, exercising
    // exactly what gr::device::ExecutionStrategy<T>::dispatchSyclBulk now calls
    gr::test::TwoInputSyclBulk twoInputBlock;
    TwoInputSpans              twoInputs;
    OutputSpans                twoOutputs;
    gr::device::SyclQueue      queue;
    const gr::work::Status     twoInputStatus = gr::device::detail::invokeProcessBulkSycl(queue, twoInputBlock, twoInputs, twoOutputs, std::make_index_sequence<std::tuple_size_v<TwoInputSpans>>(), std::make_index_sequence<std::tuple_size_v<OutputSpans>>());
    if (twoInputStatus != gr::work::Status::OK) {
        return 3;
    }

    gr::log::HistoryLoggerBackend capture;
    auto* const                   previousBackend = gr::log::setBackend(&capture);

    gr::test::SyclBulkOnly          block;
    InputSpans                      inputs;
    OutputSpans                     outputs;
    gr::device::DeviceContext*      contextCache   = nullptr;
    const auto                      outcome        = gr::device::ExecutionStrategy<gr::test::SyclBulkOnly>::dispatch(block, inputs, outputs, 0UZ, 0UZ, "gpu:not-registered", contextCache);

    bool sawDispatchError = false;
    std::ignore           = capture.drain(detectDispatchError, &sawDispatchError);
    std::ignore           = gr::log::setBackend(previousBackend);

    if (outcome.has_value()) { // an unserviceable domain with no CPU fallback must surface the cause, not a status
        return 1;
    }
    return sawDispatchError ? 0 : 2;
}
