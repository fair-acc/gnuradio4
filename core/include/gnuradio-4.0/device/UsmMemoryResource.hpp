#ifndef GNURADIO_USM_MEMORY_RESOURCE_HPP
#define GNURADIO_USM_MEMORY_RESOURCE_HPP

#include <cstddef>
#include <memory_resource>
#include <new>

#include <gnuradio-4.0/ComputeDomain.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

/**
 * @brief PMR memory resource backed by SYCL Unified Shared Memory.
 *
 * Wraps `sycl::aligned_alloc_shared` / `sycl::free` as a `std::pmr::memory_resource`.
 * When SYCL is unavailable, falls back to aligned `operator new` / `delete`. Register
 * with `ComputeRegistry` via `registerUsmProvider()` to make device edges allocate through USM.
 *
 * Usage:
 * @code
 * gr::device::UsmMemoryResource mr;
 * std::pmr::vector<float> v(1024, 0.f, &mr);
 * @endcode
 */
class UsmMemoryResource : public std::pmr::memory_resource {
#if GR_DEVICE_HAS_SYCL_IMPL
    sycl::queue* _queue = nullptr;
#endif

public:
    UsmMemoryResource() = default;

#if GR_DEVICE_HAS_SYCL_IMPL
    explicit UsmMemoryResource(sycl::queue& q) : _queue(&q) {}
#endif

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0) {
            bytes = 1;
        }
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL_IMPL
            if (_queue) {
                return sycl::aligned_alloc_shared(alignment, bytes, *_queue);
            }
#endif
        }
        return ::operator new(bytes, std::align_val_t{alignment});
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL_IMPL
            if (_queue) {
                sycl::free(p, *_queue);
                return;
            }
#endif
        }
        ::operator delete(p, std::align_val_t{alignment});
    }

    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        const auto* o = dynamic_cast<const UsmMemoryResource*>(&other);
        if (!o) {
            return false;
        }
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL_IMPL
            return _queue == o->_queue;
#endif
        }
        return true;
    }
};

namespace detail {

inline UsmMemoryResource& defaultUsmResource() {
    static UsmMemoryResource instance;
    return instance;
}

inline std::pmr::memory_resource* usmProvider(const ComputeDomain& /*dom*/, void* ctx) {
    if (ctx) {
        return static_cast<std::pmr::memory_resource*>(ctx);
    }
    return &defaultUsmResource();
}

} // namespace detail

inline void registerUsmProvider() { ComputeRegistry::instance().register_provider("sycl", &detail::usmProvider); }

} // namespace gr::device

#endif // GNURADIO_USM_MEMORY_RESOURCE_HPP
