#ifndef GNURADIO_ATOMICREF_HPP
#define GNURADIO_ATOMICREF_HPP

#include <atomic>
#include <cstddef>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#define GR_HAS_SYCL 1
#endif

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

namespace gr {

template<typename T>
struct AtomicRef {
    using value_type = std::remove_cvref_t<T>;
    value_type& _x;
    explicit AtomicRef(T& x) noexcept : _x(x) {}

    forceinline value_type load_acquire() const noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<value_type, sycl::memory_order::acquire, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).load(););
        __acpp_if_target_host(return std::atomic_ref<value_type>(_x).load(std::memory_order_acquire););
#else
        return std::atomic_ref<value_type>(_x).load(std::memory_order_acquire);
#endif
    }

    forceinline value_type load_relaxed() const noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<value_type, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).load(););
        __acpp_if_target_host(return std::atomic_ref<value_type>(_x).load(std::memory_order_relaxed););
#else
        return std::atomic_ref<value_type>(_x).load(std::memory_order_relaxed);
#endif
    }

    forceinline constexpr void store_release(T v) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(sycl::atomic_ref<T, sycl::memory_order::release, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).store(v););
        __acpp_if_target_host(std::atomic_ref<T>(_x).store(v, std::memory_order_release););
#else
        std::atomic_ref<T>(_x).store(v, std::memory_order_release);
#endif
    }

    forceinline constexpr void store_relaxed(T v) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).store(v););
        __acpp_if_target_host(std::atomic_ref<T>(_x).store(v, std::memory_order_relaxed););
#else
        std::atomic_ref<T>(_x).store(v, std::memory_order_relaxed);
#endif
    }

    forceinline constexpr bool compare_exchange(T& expected, T desired) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).compare_exchange_strong(expected, desired););
        __acpp_if_target_host(return std::atomic_ref<T>(_x).compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire););
#else
        return std::atomic_ref<T>(_x).compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
#endif
    }

    forceinline constexpr T exchange(T desired) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).exchange(desired););
        __acpp_if_target_host(return std::atomic_ref<T>(_x).exchange(desired, std::memory_order_acq_rel););
#else
        return std::atomic_ref<T>(_x).exchange(desired, std::memory_order_acq_rel);
#endif
    }

    forceinline constexpr T fetch_add(T inc) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).fetch_add(inc););
        __acpp_if_target_host(return std::atomic_ref<T>(_x).fetch_add(inc, std::memory_order_acq_rel););
#else
        return std::atomic_ref<T>(_x).fetch_add(inc, std::memory_order_acq_rel);
#endif
    }

    forceinline constexpr T fetch_sub(T dec) noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_device(return sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::system, sycl::access::address_space::global_space>(_x).fetch_sub(dec););
        __acpp_if_target_host(return std::atomic_ref<T>(_x).fetch_sub(dec, std::memory_order_acq_rel););
#else
        return std::atomic_ref<T>(_x).fetch_sub(dec, std::memory_order_acq_rel);
#endif
    }

    forceinline constexpr void wait(T oldValue) const noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_host(std::atomic_ref<value_type>(_x).wait(oldValue););
#else
        std::atomic_ref<T>(_x).wait(oldValue);
#endif
    }

    forceinline constexpr void notify_all() noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_host(std::atomic_ref<value_type>(_x).notify_all(););
#else
        std::atomic_ref<T>(_x).notify_all();
#endif
    }

    forceinline constexpr void notify_one() noexcept {
#if defined(GR_HAS_SYCL) && defined(__ACPP__)
        __acpp_if_target_host(std::atomic_ref<value_type>(_x).notify_one(););
#else
        std::atomic_ref<T>(_x).notify_one();
#endif
    }
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
