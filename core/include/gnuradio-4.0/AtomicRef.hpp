#ifndef GNURADIO_ATOMICREF_HPP
#define GNURADIO_ATOMICREF_HPP

#include <atomic>
#include <cstddef>

#if __has_include(<sycl/sycl.hpp>) && defined(__ACPP__)
#include <sycl/sycl.hpp>
#define GR_HAS_SYCL 1
#endif

#ifndef forceinline
#define forceinline inline __attribute__((always_inline))
#endif

namespace gr {

// sycl::atomic_ref supports integral (not bool/enum), floating-point, and pointer types
template<typename U>
inline constexpr bool kSyclAtomicCompatible = (std::is_integral_v<U> && !std::is_same_v<U, bool>) || std::is_floating_point_v<U> || std::is_pointer_v<U>;

template<typename T>
struct AtomicRef {
    using value_type = std::remove_cvref_t<T>;
    value_type& _x;
    explicit AtomicRef(T& x) noexcept : _x(x) {}

    // Uses sycl::atomic_ref directly (no __acpp_if_target_device) to avoid SSCP kernel metadata
    // collisions. sycl::atomic_ref with memory_scope::system works on both host and device
    // with USM shared memory. For non-SYCL-compatible types (bool, enum), falls back to std::atomic_ref.
    // AdaptiveCpp only supports relaxed and acq_rel memory orders — acquire/release mapped to acq_rel.

    forceinline value_type load_acquire() const noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<value_type, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).load();
        } else {
            return std::atomic_ref<value_type>(_x).load(std::memory_order_acquire);
        }
#else
        return std::atomic_ref<value_type>(_x).load(std::memory_order_acquire);
#endif
    }

    forceinline value_type load_relaxed() const noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).load();
        } else {
            return std::atomic_ref<value_type>(_x).load(std::memory_order_relaxed);
        }
#else
        return std::atomic_ref<value_type>(_x).load(std::memory_order_relaxed);
#endif
    }

    forceinline constexpr void store_release(T v) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).store(v);
        } else {
            std::atomic_ref<T>(_x).store(v, std::memory_order_release);
        }
#else
        std::atomic_ref<T>(_x).store(v, std::memory_order_release);
#endif
    }

    forceinline constexpr void store_relaxed(T v) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).store(v);
        } else {
            std::atomic_ref<T>(_x).store(v, std::memory_order_relaxed);
        }
#else
        std::atomic_ref<T>(_x).store(v, std::memory_order_relaxed);
#endif
    }

    forceinline constexpr bool compare_exchange(T& expected, T desired) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).compare_exchange_strong(expected, desired);
        } else {
            return std::atomic_ref<T>(_x).compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
        }
#else
        return std::atomic_ref<T>(_x).compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
#endif
    }

    forceinline constexpr T exchange(T desired) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).exchange(desired);
        } else {
            return std::atomic_ref<T>(_x).exchange(desired, std::memory_order_acq_rel);
        }
#else
        return std::atomic_ref<T>(_x).exchange(desired, std::memory_order_acq_rel);
#endif
    }

    forceinline constexpr T fetch_add(T inc) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).fetch_add(inc);
        } else {
            return std::atomic_ref<T>(_x).fetch_add(inc, std::memory_order_acq_rel);
        }
#else
        return std::atomic_ref<T>(_x).fetch_add(inc, std::memory_order_acq_rel);
#endif
    }

    forceinline constexpr T fetch_sub(T dec) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        if constexpr (kSyclAtomicCompatible<value_type>) {
            return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).fetch_sub(dec);
        } else {
            return std::atomic_ref<T>(_x).fetch_sub(dec, std::memory_order_acq_rel);
        }
#else
        return std::atomic_ref<T>(_x).fetch_sub(dec, std::memory_order_acq_rel);
#endif
    }

    // wait/notify are host-only (no SYCL device equivalent)
    forceinline constexpr void wait(T oldValue) const noexcept { std::atomic_ref<value_type>(_x).wait(oldValue); }

    forceinline constexpr void notify_all() noexcept { std::atomic_ref<value_type>(_x).notify_all(); }

    forceinline constexpr void notify_one() noexcept { std::atomic_ref<value_type>(_x).notify_one(); }
};

template<typename T>
[[nodiscard]] forceinline constexpr gr::AtomicRef<T> atomic_ref(T& x) noexcept {
    static_assert(!std::is_const_v<T>, "atomic_ref requires non-const T");
    return AtomicRef<T>(x);
}

/**
 * @brief Portable release fence for ordering non-atomic writes before atomic publishes.
 *
 * x86/x86-64: explicit no-op (TSO provides release semantics for all stores).
 * TSAN: atomic store on a dummy variable (TSAN doesn't instrument std::atomic_thread_fence).
 * Weakly-ordered architectures (ARM64, RISC-V, POWER, etc.): std::atomic_thread_fence(release).
 */
inline void atomicThreadFence() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86 TSO: all stores are release stores — fence is a no-op
#elif defined(__SANITIZE_THREAD__) || (defined(__has_feature) && __has_feature(thread_sanitizer))
    // TSAN doesn't instrument std::atomic_thread_fence; use an atomic store to make the ordering visible
    alignas(sizeof(std::size_t)) static std::size_t tsanDummy = 0;
    gr::atomic_ref(tsanDummy).store_release(1);
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
}

} // namespace gr

#ifdef forceinline
#undef forceinline
#endif

#endif // GNURADIO_ATOMICREF_HPP
