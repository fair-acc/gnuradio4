#ifndef GNURADIO_CLAIMSTRATEGY_HPP
#define GNURADIO_CLAIMSTRATEGY_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/meta/CacheLineSize.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "Sequence.hpp"
#include "WaitStrategy.hpp"

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

namespace gr {

template<typename T>
concept ClaimStrategyLike = requires(T /*const*/ t, const std::size_t sequence, const std::size_t offset, const std::size_t nSlotsToClaim) {
    { t.next(nSlotsToClaim) } -> std::same_as<std::size_t>;
    { t.tryNext(nSlotsToClaim) } -> std::same_as<std::optional<std::size_t>>;
    { t.getRemainingCapacity() } -> std::same_as<std::size_t>;
    { t.publish(offset, nSlotsToClaim) } -> std::same_as<void>;
};

template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(kCacheLine) SingleProducerStrategy {
    const std::size_t   _size = SIZE;
    mutable std::size_t _cachedMinReaderCursor{kInitialCursorValue};
    mutable std::size_t _cachedReaderCount{0UZ};
    mutable Sequence*   _cachedSingleReader{nullptr}; // fast path: direct pointer when ≤1 reader

public:
    Sequence                                                _publishCursor;                      // slots are published and ready to be read until _publishCursor
    std::size_t                                             _reserveCursor{kInitialCursorValue}; // slots can be reserved starting from _reserveCursor, no need for atomics since this is called by a single publisher
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    explicit SingleProducerStrategy(const std::size_t bufferSize = SIZE) : _size(bufferSize) {};
    SingleProducerStrategy(const SingleProducerStrategy&)  = delete;
    SingleProducerStrategy(const SingleProducerStrategy&&) = delete;
    void operator=(const SingleProducerStrategy&)          = delete;

    void updateCachedReaderInfo() const noexcept {
        _cachedReaderCount  = _readSequences->size();
        _cachedSingleReader = (_cachedReaderCount == 1UZ) ? _readSequences->front().get() : nullptr;
    }

    std::size_t next(const std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        SpinWait spinWait;
        while (getRemainingCapacity(nSlotsToClaim) < nSlotsToClaim) {
            if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
                _waitStrategy.signalAllWhenBlocking();
            }
            spinWait.spinOnce();
        }
        _reserveCursor += nSlotsToClaim;
        return _reserveCursor;
    }

    [[nodiscard]] std::optional<std::size_t> tryNext(const std::size_t nSlotsToClaim) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        if (getRemainingCapacity(nSlotsToClaim) < nSlotsToClaim) {
            return std::nullopt;
        }
        _reserveCursor += nSlotsToClaim;
        return _reserveCursor;
    }

    [[nodiscard]] forceinline std::size_t getRemainingCapacity() const noexcept { return getRemainingCapacity(1); }

    void publish(std::size_t offset, std::size_t nSlotsToClaim) {
        const auto sequence = offset + nSlotsToClaim;
        _publishCursor.setValue(sequence);
        _reserveCursor = sequence;
        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline std::size_t getRemainingCapacity(std::size_t required) const noexcept {
        if (_cachedReaderCount <= 2UZ) { // ≤2 readers: cache avoids scanning all cursors on every call
            const std::size_t remaining = _size - (_reserveCursor - _cachedMinReaderCursor);
            if (remaining >= required && remaining <= _size) [[likely]] {
                return remaining;
            }
        }
        // >2 readers or cache miss: rescan paces the producer, preventing coherence saturation on _publishCursor
        _cachedMinReaderCursor = getMinReaderCursor();
        return _size - (_reserveCursor - _cachedMinReaderCursor);
    }

    [[nodiscard]] forceinline std::size_t getMinReaderCursor() const noexcept {
        if (_cachedSingleReader) {
            return _cachedSingleReader->value();
        }
        if (_cachedReaderCount == 0UZ) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }
};

static_assert(ClaimStrategyLike<SingleProducerStrategy<1024, NoWaitStrategy>>);

/**
 * @brief Multi-producer claim strategy using per-slot sequence counters (LMAX Disruptor pattern).
 *
 * Each ring-buffer slot stores the sequence number of its last completed write.
 * publish() marks slots with a single atomic store (no CAS), then advances the
 * publish cursor past all contiguously completed slots.  The stored sequence
 * naturally distinguishes wrap-around rounds, so no separate clear pass is needed.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(kCacheLine) MultiProducerStrategy {
    static constexpr bool kIsSizeDynamic = (SIZE == std::dynamic_extent);
    static constexpr bool kIsSizePow2    = !kIsSizeDynamic && std::has_single_bit(SIZE);

    using AvailableBufferType = std::conditional_t<kIsSizeDynamic, std::vector<std::size_t>, std::array<std::size_t, kIsSizeDynamic ? 1UZ : SIZE>>;

    AvailableBufferType _availableBuffer;
    const std::size_t   _size   = SIZE;
    const bool          _isPow2 = kIsSizePow2;
    const std::size_t   _mask   = SIZE - 1;
    mutable std::size_t _cachedMinReaderCursor{kInitialCursorValue};
    mutable std::size_t _cachedReaderCount{0UZ};
    mutable Sequence*   _cachedSingleReader{nullptr}; // fast path: direct pointer when ≤1 reader

    forceinline constexpr std::size_t calculateIndex(std::size_t seq) const noexcept {
        if constexpr (!kIsSizeDynamic) {
            if constexpr (kIsSizePow2) {
                return seq & (SIZE - 1);
            } else {
                return seq % SIZE;
            }
        } else {
            if (_isPow2) [[likely]] {
                return seq & _mask;
            }
            return seq % _size;
        }
    }

    void initAvailableBuffer() noexcept {
        for (auto& slot : _availableBuffer) {
            gr::atomic_ref(slot).store_relaxed(std::numeric_limits<std::size_t>::max());
        }
    }

public:
    Sequence                                                _reserveCursor;
    Sequence                                                _publishCursor;
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()};

    MultiProducerStrategy() = delete;

    explicit MultiProducerStrategy(std::size_t bufferSize)
    requires(kIsSizeDynamic)
        : _availableBuffer(bufferSize), _size(bufferSize), _isPow2(std::has_single_bit(bufferSize)), _mask(bufferSize - 1) {
        initAvailableBuffer();
    }

    explicit MultiProducerStrategy(std::size_t /*bufferSize*/ = SIZE)
    requires(!kIsSizeDynamic)
    {
        initAvailableBuffer();
    }

    MultiProducerStrategy(const MultiProducerStrategy&)  = delete;
    MultiProducerStrategy(const MultiProducerStrategy&&) = delete;
    void operator=(const MultiProducerStrategy&)         = delete;

    void updateCachedReaderInfo() const noexcept {
        _cachedReaderCount  = _readSequences->size();
        _cachedSingleReader = (_cachedReaderCount == 1UZ) ? _readSequences->front().get() : nullptr;
    }

    [[nodiscard]] std::size_t next(std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        std::size_t currentReserveCursor;
        std::size_t nextReserveCursor;
        SpinWait    spinWait;
        do {
            currentReserveCursor        = _reserveCursor.value();
            nextReserveCursor           = currentReserveCursor + nSlotsToClaim;
            const std::size_t cachedMin = gr::atomic_ref(_cachedMinReaderCursor).load_relaxed();
            if (nextReserveCursor - cachedMin > _size) {
                const std::size_t freshMin = getMinReaderCursor();
                gr::atomic_ref(_cachedMinReaderCursor).store_relaxed(freshMin);
                if (nextReserveCursor - freshMin > _size) {
                    if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
                        _waitStrategy.signalAllWhenBlocking();
                    }
                    spinWait.spinOnce();
                    continue;
                }
            }
            if (_reserveCursor.compareAndSet(currentReserveCursor, nextReserveCursor)) {
                break;
            }
        } while (true);

        return nextReserveCursor;
    }

    [[nodiscard]] std::optional<std::size_t> tryNext(std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        std::size_t currentReserveCursor;
        std::size_t nextReserveCursor;

        do {
            currentReserveCursor        = _reserveCursor.value();
            nextReserveCursor           = currentReserveCursor + nSlotsToClaim;
            const std::size_t cachedMin = gr::atomic_ref(_cachedMinReaderCursor).load_relaxed();
            if (nextReserveCursor - cachedMin > _size) {
                const std::size_t freshMin = getMinReaderCursor();
                gr::atomic_ref(_cachedMinReaderCursor).store_relaxed(freshMin);
                if (nextReserveCursor - freshMin > _size) {
                    return std::nullopt;
                }
            }
        } while (!_reserveCursor.compareAndSet(currentReserveCursor, nextReserveCursor));
        return nextReserveCursor;
    }

    [[nodiscard]] forceinline std::size_t getRemainingCapacity() const noexcept {
        const std::size_t cachedMin = gr::atomic_ref(_cachedMinReaderCursor).load_relaxed();
        const std::size_t used      = _reserveCursor.value() - cachedMin;
        if (used < _size) [[likely]] {
            return _size - used;
        }
        const std::size_t freshMin = getMinReaderCursor();
        gr::atomic_ref(_cachedMinReaderCursor).store_relaxed(freshMin);
        return _size - (_reserveCursor.value() - freshMin);
    }

    void publish(std::size_t offset, std::size_t nSlotsToClaim) {
        if (nSlotsToClaim == 0) {
            return;
        }

        for (std::size_t seq = offset; seq < offset + nSlotsToClaim; ++seq) {
            gr::atomic_ref(_availableBuffer[calculateIndex(seq)]).store_release(seq);
        }

        std::size_t currentPublishCursor;
        std::size_t nextPublishCursor;
        do {
            currentPublishCursor = _publishCursor.value();
            nextPublishCursor    = currentPublishCursor;

            while (nextPublishCursor - currentPublishCursor < _size && gr::atomic_ref(_availableBuffer[calculateIndex(nextPublishCursor)]).load_acquire() == nextPublishCursor) {
                ++nextPublishCursor;
            }
            if (nextPublishCursor == currentPublishCursor) {
                return;
            }
        } while (!_publishCursor.compareAndSet(currentPublishCursor, nextPublishCursor));

        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline std::size_t getMinReaderCursor() const noexcept {
        if (_cachedSingleReader) {
            return _cachedSingleReader->value();
        }
        if (_cachedReaderCount == 0UZ) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }
};

static_assert(ClaimStrategyLike<MultiProducerStrategy<1024, NoWaitStrategy>>);

enum class ProducerType {
    /**
     * creates a buffer assuming a single producer-thread and multiple consumer
     */
    Single,

    /**
     * creates a buffer assuming multiple producer-threads and multiple consumer
     */
    Multi
};

namespace detail {
template<std::size_t size, ProducerType producerType, WaitStrategyLike TWaitStrategy>
struct producer_type;

template<std::size_t size, WaitStrategyLike TWaitStrategy>
struct producer_type<size, ProducerType::Single, TWaitStrategy> {
    using value_type = SingleProducerStrategy<size, TWaitStrategy>;
};

template<std::size_t size, WaitStrategyLike TWaitStrategy>
struct producer_type<size, ProducerType::Multi, TWaitStrategy> {
    using value_type = MultiProducerStrategy<size, TWaitStrategy>;
};

template<std::size_t size, ProducerType producerType, WaitStrategyLike TWaitStrategy>
using producer_type_v = typename producer_type<size, producerType, TWaitStrategy>::value_type;

} // namespace detail

} // namespace gr

#ifdef forceinline
#undef forceinline
#endif

#endif // GNURADIO_CLAIMSTRATEGY_HPP
