#include <string_view>
#include <tuple>

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

} // namespace gr::test

using InputSpans  = std::tuple<gr::traits::block::detail::DummyInputSpan<const float>>;
using OutputSpans = std::tuple<gr::traits::block::detail::DummyOutputSpan<float>>;

static_assert(gr::device::HasSyclBulkForSpans<gr::test::TemplatedSyclBulk, InputSpans, OutputSpans>);
static_assert(gr::device::ExecutionStrategy<gr::test::TemplatedSyclBulk>::canDispatch<InputSpans, OutputSpans>());
static_assert(gr::device::HasSyclBulkForSpans<gr::test::SyclBulkOnly, InputSpans, OutputSpans>);

namespace {

void detectDispatchError(const gr::log::LogRecord& record, void* user) noexcept {
    auto&                  saw = *static_cast<bool*>(user);
    const std::string_view text{record.text, record.textLength};
    saw = saw || (record.level == gr::log::Level::error && text.contains("no backend is wired"));
}

} // namespace

int main() {
    gr::log::HistoryLoggerBackend capture;
    auto* const                   previousBackend = gr::log::setBackend(&capture);

    gr::test::SyclBulkOnly block;
    InputSpans             inputs;
    OutputSpans            outputs;
    const auto             outcome = gr::device::ExecutionStrategy<gr::test::SyclBulkOnly>::dispatch(block, inputs, outputs, 0UZ, "gpu:not-registered");

    bool sawDispatchError = false;
    std::ignore           = capture.drain(detectDispatchError, &sawDispatchError);
    std::ignore           = gr::log::setBackend(previousBackend);

    if (outcome.has_value()) { // an unserviceable domain with no CPU fallback must surface the cause, not a status
        return 1;
    }
    return sawDispatchError ? 0 : 2;
}
