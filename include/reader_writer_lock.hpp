#ifndef READER_WRITER_LOCK_HPP
#define READER_WRITER_LOCK_HPP

#include "utils.hpp"
#include <atomic>
#include <cstdint>

namespace fair::graph {

enum class ReaderWriterLockType { READ, WRITE };

/**
 * @brief ReaderWriterLock is multi-reader-multi-writer atomic lock meant to protect a resource
 * in situations where the thread is not allowed to block.
 *
 * The lock is implemented using atomic CAS-loops on a counter, which is
 * incremented (/decremented) when a thread acquires a read (/write) lock, and
 * decremented (/incremented) when the thread releases the read (/write) lock.
 *
 * N.B. The lock is unlocked when the counter reaches 0.
 */
class ReaderWriterLock {
    alignas(fair::meta::kCacheLine) mutable std::atomic<std::int64_t> _activeReaderCount{ 0 };

public:
    ReaderWriterLock() = default;

    [[nodiscard]] std::int64_t
    value() const noexcept {
        return std::atomic_load_explicit(&_activeReaderCount, std::memory_order_acquire);
    }

    template<ReaderWriterLockType lockType>
    std::int64_t
    tryLock() const noexcept {
        std::int64_t expected = _activeReaderCount.load(std::memory_order_relaxed);
        if constexpr (lockType == ReaderWriterLockType::READ) {
            if (expected < 0L) {
                expected = 0L;
            }
            return std::atomic_compare_exchange_strong(&_activeReaderCount, &expected, expected + 1L);
        } else {
            if (expected > 0L) {
                expected = 0L;
            }
            return std::atomic_compare_exchange_strong(&_activeReaderCount, &expected, expected - 1L);
        }
    }

    template<ReaderWriterLockType lockType>
    std::int64_t
    lock() const noexcept {
        if constexpr (lockType == ReaderWriterLockType::READ) {
            std::int64_t expected = _activeReaderCount.load(std::memory_order_relaxed);
            do {
                if (expected < 0L) {
                    expected = 0L;
                }
            } while (!std::atomic_compare_exchange_strong(&_activeReaderCount, &expected, expected + 1L));
            return expected + 1L;
        } else {
            std::int64_t expected = _activeReaderCount.load(std::memory_order_relaxed);
            do {
                if (expected > 0L) {
                    expected = 0L;
                }
            } while (!std::atomic_compare_exchange_strong(&_activeReaderCount, &expected, expected - 1L));
            return expected - 1L;
        }
    }

    template<ReaderWriterLockType lockType>
    std::int64_t
    unlock() const noexcept {
        if constexpr (lockType == ReaderWriterLockType::READ) {
            return std::atomic_fetch_sub(&_activeReaderCount, 1L) - 1L;
        } else {
            return std::atomic_fetch_add(&_activeReaderCount, 1L) + 1L;
        }
    }

    template<ReaderWriterLockType lockType>
    auto
    scopedGuard() {
        return ScopedLock<lockType>(*this);
    }

    template<ReaderWriterLockType lockType>
    class ScopedLock { // NOSONAR - class destructor is needed for guard functionality
        ReaderWriterLock *_readWriteLock;

    public:
        ScopedLock()                   = delete;
        ScopedLock(const ScopedLock &) = delete;
        ScopedLock(ScopedLock &&)      = delete;
        ScopedLock &
        operator=(const ScopedLock &)
                = delete;
        ScopedLock &
        operator=(ScopedLock &&)
                = delete;

        explicit constexpr ScopedLock(ReaderWriterLock &parent) noexcept : _readWriteLock(&parent) { _readWriteLock->lock<lockType>(); }

        ~ScopedLock() { _readWriteLock->unlock<lockType>(); }
    };
};

} // namespace fair::graph

#endif // READER_WRITER_LOCK_HPP
