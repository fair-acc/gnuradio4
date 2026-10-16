#ifndef GNURADIO_FILTER_DELAY_HPP
#define GNURADIO_FILTER_DELAY_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::Delay, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>)
struct Delay : Block<Delay<T>, NoTagPropagation> {
    using Description = Doc<R""(@brief delays its stream by `delay` samples (GR3: Delay); the first `delay` samples out are zero.

Raising `delay` by k inserts k zeros where it is set; lowering it drops the k samples due next.
Tags move with their samples, and those on dropped samples are dropped; the last `delay` samples of a stream, and their tags, are not emitted.)"">;

    static constexpr gr::Size_t kMaxDelay = 1U << 24U;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "delay", Doc<"samples">, Limits<0U, kMaxDelay>> delay = 0U;

    std::pmr::vector<T> _history;
    std::pmr::vector<T> _historyNext;

    GR_MAKE_REFLECTABLE(Delay, in, out, delay, _history, _historyNext);

    DelayedTags _delayedTags;

    void start() {
        _history.assign(delay.value, T{});
        _delayedTags.clear();
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return delayInto(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return delayInto(ctx, input, output); }

    [[nodiscard]] gr::work::Status delayInto(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t n = std::min(input.size(), output.size());
        const std::size_t d = delay.value;
        _historyNext.resize(_history.size());
        const T* samples = input.data();
        const T* history = _history.data();
        gr::device::parallelFor(ctx, n, [samples, history, delayed = output.data(), d] GR_DEVICE_LAMBDA(std::size_t i) { delayed[i] = i < d ? history[i] : samples[i - d]; });
        gr::device::parallelFor(ctx, d, [samples, history, next = _historyNext.data(), n, d] GR_DEVICE_LAMBDA(std::size_t j) { next[j] = j + n < d ? history[j + n] : samples[j + n - d]; });
        std::swap(_history, _historyNext);
        this->forwardDelayedTags(_delayedTags, input, std::tie(output), n, delay.value);
        std::ignore = input.consume(n);
        output.publish(n);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("delay")) {
            const std::size_t   was = oldSettings.value_or<gr::Size_t>("delay", 0U);
            const std::size_t   now = delay.value;
            std::pmr::vector<T> resized(now, T{}, _history.get_allocator());
            if (!_history.empty()) {
                const std::size_t kept = std::min(was, now);
                std::ranges::copy(std::span(_history).last(kept), resized.begin() + static_cast<std::ptrdiff_t>(now - kept));
            }
            _history = std::move(resized);
            for (auto& [index, map] : _delayedTags) {
                index = index + now - was;
            }
        }
    }
};

} // namespace gr::filter

#endif // GNURADIO_FILTER_DELAY_HPP
