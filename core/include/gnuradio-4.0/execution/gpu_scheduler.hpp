#ifndef GNURADIO_DEVICE_SCHEDULER_HPP
#define GNURADIO_DEVICE_SCHEDULER_HPP

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/execution/execution.hpp>

namespace gr::execution {

/**
 * @brief P2300-compatible scheduler that dispatches work to a device (GPU, FPGA, accelerator).
 *
 * Models the `gr::execution::scheduler` concept. `schedule()` returns a sender that completes
 * on the device context. Combined with `bulk`, `continues_on`, and other P2300 algorithms to
 * build heterogeneous CPU↔device pipelines.
 *
 * Usage:
 * @code
 * gr::device::DeviceContext ctx;
 * gr::execution::DeviceScheduler sched(ctx);
 * auto result = gr::execution::sync_wait(
 *     sched.schedule()
 *     | gr::execution::then([] { return 42; })
 *     | gr::execution::bulk(N, [](std::size_t i, int& v) { v += i; }));
 * @endcode
 */
class DeviceScheduler {
    device::DeviceContext* _ctx;

public:
    explicit DeviceScheduler(device::DeviceContext& ctx) : _ctx(&ctx) {}

    struct ScheduleSender {
        using completion_signatures_t = completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;
        using value_tuple_t           = std::tuple<>;

        device::DeviceContext* ctx;

        template<receiver R>
        struct Op {
            device::DeviceContext* ctx;
            std::remove_cvref_t<R> recv;

            void start() noexcept {
                (void)ctx;
                gr::execution::set_value(std::move(recv));
            }
        };

        template<receiver R>
        [[nodiscard]] auto connect(R&& r) const& {
            return Op<R>{ctx, std::forward<R>(r)};
        }

        template<receiver R>
        [[nodiscard]] auto connect(R&& r) && {
            return Op<R>{ctx, std::forward<R>(r)};
        }
    };

    [[nodiscard]] ScheduleSender schedule() const { return {_ctx}; }

    [[nodiscard]] device::DeviceContext&       context() noexcept { return *_ctx; }
    [[nodiscard]] const device::DeviceContext& context() const noexcept { return *_ctx; }
    [[nodiscard]] device::DeviceBackend        backend() const noexcept { return _ctx->backend(); }

    bool operator==(const DeviceScheduler&) const = default;
};

} // namespace gr::execution

#endif // GNURADIO_DEVICE_SCHEDULER_HPP
