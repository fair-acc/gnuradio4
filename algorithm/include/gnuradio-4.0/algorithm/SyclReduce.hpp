#ifndef GNURADIO_ALGORITHM_SYCL_REDUCE_HPP
#define GNURADIO_ALGORITHM_SYCL_REDUCE_HPP

#include <cstddef>
#include <span>

#include <gnuradio-4.0/algorithm/Reduce.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>

#if GR_DEVICE_HAS_SYCL

namespace gr::algorithm {

/// the reductions that have to be written as a kernel, kept apart from `Reduce` so the algorithm library names no
/// vendor type: a SYCL header is the one place `sycl::` belongs
struct SyclReduce {
    static constexpr std::size_t kWorkGroupSize = Reduce::kWorkGroupSize;

    template<typename T>
    using ArgMax = Reduce::ArgMax<T>;

    /// one pass per work group into a partials buffer, then a final pass on the host over the partials.
    ///
    /// `values` is dereferenced inside the kernel, so it must be memory `queue`'s device can read -- USM, or a
    /// block's field already seated on the device resource. A host pointer faults rather than being copied.
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
            // an algorithm has no block whose compute_domain it could consult, so it says so and answers on the host;
            // a block calling this must put the result to its own `deviceFallbackIsAllowed()` before using it
            gr::log::warning("Reduce::argMax: no device memory for {} work-group partials — reduced on the host instead", nGroups);
            return Reduce::argMaxHost(std::span<const T>{values, count});
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
                        // ties go to the lower ORIGINAL index, not the lower lane: after a butterfly step the two
                        // candidates are no longer in index order, so `>` alone makes the winner depend on the
                        // work-group size, and the host scan -- which keeps the first maximum -- would disagree
                        const bool takesOther = local < stride && (localVal[local + stride] > localVal[local] || (localVal[local + stride] == localVal[local] && localIdx[local + stride] < localIdx[local]));
                        if (takesOther) {
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
};

} // namespace gr::algorithm

#endif // GR_DEVICE_HAS_SYCL
#endif // GNURADIO_ALGORITHM_SYCL_REDUCE_HPP
