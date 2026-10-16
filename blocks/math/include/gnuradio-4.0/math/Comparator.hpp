#ifndef GNURADIO_MATH_COMPARATOR_HPP
#define GNURADIO_MATH_COMPARATOR_HPP

#include <algorithm>
#include <complex>
#include <cstdint>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK(gr::blocks::math::Comparator, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK(gr::blocks::math::Comparator, ([T], uint8_t), [ int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, typename TOut = meta::fundamental_base_value_type_t<T>>
requires(std::is_arithmetic_v<T> || meta::complex_like<T>) && (std::same_as<TOut, meta::fundamental_base_value_type_t<T>> || std::same_as<TOut, std::uint8_t>)
struct Comparator : Block<Comparator<T, TOut>> {
    using Description = Doc<R""(@brief 1 above `high`, 0 below `low`, and inside the band the previous output (GR3: Threshold); `low == high` is a plain comparator.

A complex sample is compared by its magnitude; NaN keeps the previous output; a reversed band is swapped.
The output is the real sample type, or `uint8_t` to drive a `clk_in`.)"">;
    using R           = meta::fundamental_base_value_type_t<T>;

    PortIn<T>     in;
    PortOut<TOut> out;

    Annotated<R, "low", Doc<"falls to 0 below; of the magnitude for complex samples">>  low     = R{0};
    Annotated<R, "high", Doc<"rises to 1 above; of the magnitude for complex samples">> high    = R{0};
    Annotated<bool, "initial", Doc<"output until the first sample leaves the band">>    initial = false;

    GR_MAKE_REFLECTABLE(Comparator, in, out, low, high, initial);

    bool _above = false;

    void start() { reset(); }
    void reset() { _above = initial; }

    [[nodiscard]] constexpr TOut processOne(T sample) noexcept {
        const R level = [&] {
            if constexpr (meta::complex_like<T>) {
                return std::abs(sample);
            } else {
                return sample;
            }
        }();
        if (level > std::max(low.value, high.value)) {
            _above = true;
        } else if (level < std::min(low.value, high.value)) {
            _above = false;
        }
        return static_cast<TOut>(_above);
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_COMPARATOR_HPP
