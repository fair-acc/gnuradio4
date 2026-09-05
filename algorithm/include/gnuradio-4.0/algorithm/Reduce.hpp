#ifndef GNURADIO_ALGORITHM_REDUCE_HPP
#define GNURADIO_ALGORITHM_REDUCE_HPP

#include <cstddef>
#include <functional>
#include <limits>
#include <span>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::algorithm {

/**
 * @brief Reductions over one span, with a host and a device implementation behind one call.
 *
 * A reduction over a single large span cannot be expressed as a block body: the framework's per-window tier
 * gives one work item per output, which is the right shape when there are many frames in flight but leaves a
 * lone frame serial. Cooperating over one frame needs a work-group launch with barriers, and only whoever
 * submits the kernel can ask for one — so it lives here, beside `SyclFFT`, and a block reaches it through
 * `processBulk_sycl`.
 *
 * Without SYCL the device path is not compiled at all and the host implementation is the whole of it.
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

#if GR_DEVICE_HAS_SYCL_IMPL
    /// one pass per work group into a partials buffer, then a final pass on the host over the partials
    template<typename T>
    [[nodiscard]] static ArgMax<T> argMax(sycl::queue& queue, const T* values, std::size_t count) {
        if (count == 0UZ) {
            return {};
        }
        const std::size_t nGroups = (count + kWorkGroupSize - 1UZ) / kWorkGroupSize;
        T*                bestVal = sycl::malloc_shared<T>(nGroups, queue);
        std::size_t*      bestIdx = sycl::malloc_shared<std::size_t>(nGroups, queue);
        if (bestVal == nullptr || bestIdx == nullptr) {
            sycl::free(bestVal, queue);
            sycl::free(bestIdx, queue);
            return argMaxHost(std::span<const T>{values, count});
        }

        queue
            .submit([&](sycl::handler& h) {
                sycl::local_accessor<T, 1>           localVal(sycl::range<1>{kWorkGroupSize}, h);
                sycl::local_accessor<std::size_t, 1> localIdx(sycl::range<1>{kWorkGroupSize}, h);
                h.parallel_for(sycl::nd_range<1>{nGroups * kWorkGroupSize, kWorkGroupSize}, [=](sycl::nd_item<1> item) {
                    const std::size_t global = item.get_global_id(0);
                    const std::size_t local  = item.get_local_id(0);
                    localVal[local]          = global < count ? values[global] : std::numeric_limits<T>::lowest();
                    localIdx[local]          = global;
                    sycl::group_barrier(item.get_group());
                    for (std::size_t stride = kWorkGroupSize / 2UZ; stride > 0UZ; stride /= 2UZ) {
                        if (local < stride && localVal[local + stride] > localVal[local]) {
                            localVal[local] = localVal[local + stride];
                            localIdx[local] = localIdx[local + stride];
                        }
                        sycl::group_barrier(item.get_group());
                    }
                    if (local == 0UZ) {
                        bestVal[item.get_group(0)] = localVal[0];
                        bestIdx[item.get_group(0)] = localIdx[0];
                    }
                });
            })
            .wait();

        ArgMax<T> best{};
        for (std::size_t group = 0UZ; group < nGroups; ++group) {
            if (bestVal[group] > best.value) {
                best = {.index = bestIdx[group], .value = bestVal[group]};
            }
        }
        sycl::free(bestVal, queue);
        sycl::free(bestIdx, queue);
        return best;
    }
#endif
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_REDUCE_HPP
