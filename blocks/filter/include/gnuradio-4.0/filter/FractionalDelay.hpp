#ifndef GNURADIO_FILTER_FRACTIONAL_DELAY_HPP
#define GNURADIO_FILTER_FRACTIONAL_DELAY_HPP

#include <array>
#include <cmath>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/filter/Delay.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

// causal third-order Lagrange interpolation: T. I. Laakso, V. Välimäki, M. Karjalainen and U. K. Laine,
// "Splitting the unit delay," IEEE Signal Processing Magazine, vol. 13, no. 1, pp. 30-60, 1996
template<typename T>
[[nodiscard]] GR_DEVICE_FN constexpr T lagrangeDelay(const std::array<T, 4UZ>& newestFirst, double fraction) noexcept {
    using R    = gr::meta::fundamental_base_value_type_t<T>;
    const R d  = static_cast<R>(fraction);
    const R c0 = -(d - R{1}) * (d - R{2}) * (d - R{3}) / R{6};
    const R c1 = d * (d - R{2}) * (d - R{3}) / R{2};
    const R c2 = -d * (d - R{1}) * (d - R{3}) / R{2};
    const R c3 = d * (d - R{1}) * (d - R{2}) / R{6};
    return newestFirst[0] * c0 + newestFirst[1] * c1 + newestFirst[2] * c2 + newestFirst[3] * c3;
}

GR_REGISTER_BLOCK(gr::filter::FractionalDelay, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct FractionalDelay : Block<FractionalDelay<T>, NoTagPropagation> {
    using Description = Doc<"Delays samples by a nonnegative fractional sample count (third-order Lagrange); zero initial history, no tail; tags move by the rounded delay.">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<double, "delay", Doc<"samples; exact for cubic signals, error grows towards Nyquist (causal stencil, no added latency)">, Limits<0.0, static_cast<double>(Delay<T>::kMaxDelay)>> delay = 0.0;

    std::pmr::vector<T> _history;
    std::size_t         _writeIndex = 0UZ;

    GR_MAKE_REFLECTABLE(FractionalDelay, in, out, delay, _history, _writeIndex);

    DelayedTags _delayedTags;

    void start() {
        _history.assign(historySize(delay), T{});
        _writeIndex = 0UZ;
        _delayedTags.clear();
    }

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return processBulk(gr::device::hostBackend(), input, output); }
    work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count        = std::min(input.size(), output.size());
        const std::size_t integerDelay = static_cast<std::size_t>(std::floor(delay));
        const double      fraction     = delay.value - static_cast<double>(integerDelay);
        const std::size_t size         = _history.size();
        const std::size_t start        = _writeIndex;
        T*                history      = _history.data();
        const T*          samples      = input.data();
        T*                delayed      = output.data();
        gr::device::parallelFor(ctx, 1UZ, [=] GR_DEVICE_LAMBDA(std::size_t) {
            for (std::size_t i = 0UZ; i < count; ++i) {
                const std::size_t position = (start + i) % size;
                history[position]          = samples[i];
                std::array<T, 4UZ> values{};
                for (std::size_t tap = 0UZ; tap < values.size(); ++tap) {
                    values[tap] = history[(position + size - integerDelay - tap) % size];
                }
                delayed[i] = lagrangeDelay(values, fraction);
            }
        });
        _writeIndex = (start + count) % size;
        this->forwardDelayedTags(_delayedTags, input, std::tie(output), count, static_cast<std::size_t>(std::llround(delay)));
        std::ignore = input.consume(count);
        output.publish(count);
        return work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (!newSettings.contains("delay") || _history.empty()) {
            return;
        }
        const double        was = oldSettings.value_or<double>("delay", 0.0);
        std::pmr::vector<T> resized(historySize(delay), T{}, _history.get_allocator());
        const std::size_t   kept = std::min(_history.size(), resized.size());
        for (std::size_t back = 1UZ; back <= kept; ++back) {
            resized[resized.size() - back] = _history[(_writeIndex + _history.size() - back) % _history.size()];
        }
        _history         = std::move(resized);
        _writeIndex      = 0UZ;
        const auto shift = static_cast<std::ptrdiff_t>(std::llround(delay.value)) - static_cast<std::ptrdiff_t>(std::llround(was));
        for (auto& [index, map] : _delayedTags) {
            index = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(index) + shift);
        }
    }

private:
    [[nodiscard]] static std::size_t historySize(double delaySamples) noexcept { return static_cast<std::size_t>(std::floor(delaySamples)) + 4UZ; }
};

} // namespace gr::filter
#endif // GNURADIO_FILTER_FRACTIONAL_DELAY_HPP
