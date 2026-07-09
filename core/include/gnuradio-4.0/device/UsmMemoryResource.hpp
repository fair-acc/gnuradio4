#ifndef GNURADIO_USM_MEMORY_RESOURCE_HPP
#define GNURADIO_USM_MEMORY_RESOURCE_HPP

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <optional>

#include <gnuradio-4.0/ComputeDomain.hpp>
#include <gnuradio-4.0/Export.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>

namespace gr::device {

/**
 * PMR memory resource backed by SYCL Unified Shared Memory; aligned `operator new` without SYCL.
 *
 * `shared` for a buffer a kernel writes, `hostPinned` for one that crosses back to the host.
 * `registerUsmProvider()` makes device edges allocate through it.
 */
enum class UsmKind : std::uint8_t { shared, hostPinned, deviceOnly };

class UsmMemoryResource : public gr::MemoryResource {
#if GR_DEVICE_HAS_SYCL
    // owned, not borrowed: a SYCL queue is a reference-counted handle, so a copy costs nothing and keeps the
    // context alive for as long as any USM allocated from it can still be freed
    std::optional<sycl::queue> _queue;
    UsmKind                    _kind = UsmKind::shared; // only ever read by the allocating paths, which are themselves SYCL-only
#endif

public:
    UsmMemoryResource() = default;

#if GR_DEVICE_HAS_SYCL
    explicit UsmMemoryResource(sycl::queue& q, UsmKind kind = UsmKind::shared) : _queue(q), _kind(kind) {}

    // USM pointers are bound to their queue's context. A queue is a handle: every operation on it is non-const,
    // so the accessor hands out a mutable one even from a const resource.
    [[nodiscard]] sycl::queue* queue() const noexcept { return _queue.has_value() ? const_cast<sycl::queue*>(&*_queue) : nullptr; }
#endif

    [[nodiscard]] MemoryResourceCapabilities capabilities() const noexcept override {
#if GR_DEVICE_HAS_SYCL
        if (_kind != UsmKind::deviceOnly) {
            return {}; // shared and host-pinned USM are addressable from the host, so they wrap as ordinary memory
        }
        // no wait: the queue is in-order and only device work reads this memory, so the copy is ordered before the
        // next kernel; the host never touches it, and a teardown free drains the queue first
        return {.deviceOnly = true, //
            .copyWithin     = [](void* destination, const void* source, std::size_t bytes, void* context) { std::ignore = static_cast<sycl::queue*>(context)->memcpy(destination, source, bytes); },
            .copyContext    = queue()};
#else
        return {};
#endif
    }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0) {
            bytes = 1;
        }
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            if (_queue) {
                void* p = _kind == UsmKind::hostPinned ? sycl::aligned_alloc_host(alignment, bytes, *_queue) : _kind == UsmKind::deviceOnly ? sycl::aligned_alloc_device(alignment, bytes, *_queue) : sycl::aligned_alloc_shared(alignment, bytes, *_queue);
                if (p) {
                    return p;
                }
                throw std::bad_alloc();
            }
#endif
        }
        return ::operator new(bytes, std::align_val_t{alignment});
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t alignment) override {
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            if (_queue) {
                if (_kind == UsmKind::deviceOnly) {
                    _queue->wait(); // a kernel still reading this allocation would fault: nothing else orders the free
                }
                sycl::free(p, *_queue);
                return;
            }
#endif
        }
        ::operator delete(p, std::align_val_t{alignment});
    }

    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        if (this == &other) {
            return true; // identity holds whether or not the build can identify the type
        }
#if defined(__cpp_rtti) && __cpp_rtti
        const auto* o = dynamic_cast<const UsmMemoryResource*>(&other);
        if (!o) {
            return false;
        }
#else
        return false; // -fno-rtti: two distinct resources cannot be proven interchangeable, so they are not
#endif
        if constexpr (kHasSycl) {
#if GR_DEVICE_HAS_SYCL
            if (_kind != o->_kind) {
                return false; // device-only, host-pinned and shared storage are not interchangeable, whatever queue made them
            }
            return _queue == o->_queue;
#endif
        }
        return true; // without a SYCL backend there is one kind of storage, so every instance is interchangeable
    }
};

namespace detail {

GNURADIO_EXPORT inline UsmMemoryResource& defaultUsmResource() {
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

inline void registerUsmProvider() { ComputeRegistry::instance().registerProvider("sycl", &detail::usmProvider); }

} // namespace gr::device

#endif // GNURADIO_USM_MEMORY_RESOURCE_HPP
