#ifndef GNURADIO_ALGORITHM_REDUCE_HPP
#define GNURADIO_ALGORITHM_REDUCE_HPP

#include <cstddef>
#include <functional>
#include <limits>
#include <span>

#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::algorithm {

/**
 * Reductions (argmax, sum) over one span, host and device (SYCL) implementations behind one call. A single
 * large span needs a work-group launch with barriers, which a per-window block body cannot express, so it
 * lives here and a block reaches it through `processBulk(ctx, ...)`; without SYCL only the host path compiles.
 */
struct Reduce {
    static constexpr std::size_t kWorkGroupSize = 256UZ;

    /// index of the largest element, and its value; an empty span yields {0, lowest}
    template<typename T>
    struct ArgMax {
        std::size_t index = 0UZ;
        T           value = std::numeric_limits<T>::lowest();
    };

    template<typename T>
    [[nodiscard]] static ArgMax<T> argMaxHost(std::span<const T> values) noexcept {
        ArgMax<T> best{};
        for (std::size_t i = 0UZ; i < values.size(); ++i) {
            if (values[i] > best.value) {
                best = {.index = i, .value = values[i]};
            }
        }
        return best;
    }

    template<typename T>
    [[nodiscard]] static T sumHost(std::span<const T> values) noexcept {
        T total{};
        for (const T value : values) {
            total += value;
        }
        return total;
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_REDUCE_HPP
