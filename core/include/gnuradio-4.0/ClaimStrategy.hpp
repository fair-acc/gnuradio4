#ifndef GNURADIO_CLAIMSTRATEGY_HPP
#define GNURADIO_CLAIMSTRATEGY_HPP

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/meta/utils.hpp>

#include "Sequence.hpp"
#include "WaitStrategy.hpp"

namespace gr {

namespace detail {

/*
 * `AtomicBitset` is a lock-free, thread-safe bitset.
 * It allows for efficient and thread-safe manipulation of individual bits.
 */
template<std::size_t Size = std::dynamic_extent>
class AtomicBitset {
    static_assert(Size > 0, "Size must be greater than 0");
    static constexpr bool isSizeDynamic = Size == std::dynamic_extent;

    static constexpr std::size_t _bitsPerWord  = sizeof(size_t) * 8UZ;
    static constexpr std::size_t _nStaticWords = isSizeDynamic ? 1UZ : (Size + _bitsPerWord - 1UZ) / _bitsPerWord;

    // using DynamicArrayType = std::unique_ptr<std::atomic<std::size_t>[]>;
    using DynamicArrayType = std::vector<std::atomic<std::size_t>>;
    using StaticArrayType  = std::array<std::atomic<std::size_t>, _nStaticWords>;
    using ArrayType        = std::conditional_t<isSizeDynamic, DynamicArrayType, StaticArrayType>;

    std::size_t _size = Size;
    ArrayType   _bits;

public:
    AtomicBitset()
    requires(!isSizeDynamic)
    {
        for (auto& word : _bits) {
            word.store(0, std::memory_order_relaxed);
        }
    }

    explicit AtomicBitset(std::size_t size = 0UZ)
    requires(isSizeDynamic)
        : _size(size), _bits(std::vector<std::atomic<std::size_t>>(size)) {
        // assert(size > 0UZ);
        for (std::size_t i = 0; i < _size; i++) {
            _bits[i].store(0, std::memory_order_relaxed);
        }
    }

    void set(std::size_t bitPosition) {
        assert(bitPosition < _size);
        const std::size_t wordIndex = bitPosition / _bitsPerWord;
        const std::size_t bitIndex  = bitPosition % _bitsPerWord;
        const std::size_t mask      = 1UL << bitIndex;

        std::size_t oldBits;
        std::size_t newBits;
        do {
            oldBits = _bits[wordIndex].load(std::memory_order_relaxed);
            newBits = oldBits | mask;
        } while (!_bits[wordIndex].compare_exchange_weak(oldBits, newBits, std::memory_order_release, std::memory_order_relaxed));
    }

    void reset(std::size_t bitPosition) {
        assert(bitPosition < _size);
        const std::size_t wordIndex = bitPosition / _bitsPerWord;
        const std::size_t bitIndex  = bitPosition % _bitsPerWord;
        const std::size_t mask      = ~(1UL << bitIndex);

        std::size_t oldBits;
        std::size_t newBits;
        do {
            oldBits = _bits[wordIndex].load(std::memory_order_relaxed);
            newBits = oldBits & mask;
        } while (!_bits[wordIndex].compare_exchange_weak(oldBits, newBits, std::memory_order_release, std::memory_order_relaxed));
    }

    bool test(std::size_t bitPosition) const {
        assert(bitPosition < _size);
        const std::size_t wordIndex = bitPosition / _bitsPerWord;
        const std::size_t bitIndex  = bitPosition % _bitsPerWord;
        const std::size_t mask      = 1UL << bitIndex;

        return (_bits[wordIndex].load(std::memory_order_acquire) & mask) != 0;
    }

    [[nodiscard]] constexpr std::size_t size() const { return _size; }
};

} // namespace detail

template<typename T>
concept ClaimStrategyLike = requires(T /*const*/ t, const Sequence::signed_index_type sequence, const Sequence::signed_index_type offset, const std::size_t nSlotsToClaim) {
    { t.next(nSlotsToClaim) } -> std::same_as<Sequence::signed_index_type>;
    { t.tryNext(nSlotsToClaim) } -> std::same_as<std::optional<Sequence::signed_index_type>>;
    { t.getRemainingCapacity() } -> std::same_as<Sequence::signed_index_type>;
    { t.publish(offset, nSlotsToClaim) } -> std::same_as<void>;
};

template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(hardware_constructive_interference_size) SingleThreadedStrategy {
    using signed_index_type = Sequence::signed_index_type;

    const std::size_t _size = SIZE;

public:
    Sequence                                                _publishCursor;                      // slots are published and ready to be read until _publishCursor
    signed_index_type                                       _reserveCursor{kInitialCursorValue}; // slots can be reserved starting from _reserveCursor, no need for atomics since this is called by a single publisher
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    explicit SingleThreadedStrategy(const std::size_t bufferSize = SIZE) : _size(bufferSize){};
    SingleThreadedStrategy(const SingleThreadedStrategy&)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy&&) = delete;
    void operator=(const SingleThreadedStrategy&)          = delete;

    signed_index_type next(const std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");

        SpinWait spinWait;
        while (getRemainingCapacity() < nSlotsToClaim) { // while not enough slots in buffer
            if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
                _waitStrategy.signalAllWhenBlocking();
            }
            spinWait.spinOnce();
        }
        _reserveCursor += nSlotsToClaim;
        return _reserveCursor;
    }

    [[nodiscard]] std::optional<signed_index_type> tryNext(const std::size_t nSlotsToClaim) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");
        static_cast<signed_index_type>(_size) < nSlotsToClaim + _reserveCursor - getMinReaderCursor();

        if (getRemainingCapacity() < nSlotsToClaim) { // not enough slots in buffer
            return std::nullopt;
        }
        _reserveCursor += nSlotsToClaim;
        return _reserveCursor;
    }

    [[nodiscard]] forceinline signed_index_type getRemainingCapacity() const noexcept { return static_cast<signed_index_type>(_size) - (_reserveCursor - getMinReaderCursor()); }

    void publish(signed_index_type offset, std::size_t nSlotsToClaim) {
        const auto sequence = offset + static_cast<signed_index_type>(nSlotsToClaim);
        _publishCursor.setValue(sequence);
        _reserveCursor = sequence;
        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline signed_index_type getMinReaderCursor() const noexcept {
        if (_readSequences->empty()) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }
};

static_assert(ClaimStrategyLike<SingleThreadedStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 *
 * The size argument (compile-time and run-time) must be a power-of-2 value.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
requires(SIZE == std::dynamic_extent || std::has_single_bit(SIZE))
class alignas(hardware_constructive_interference_size) MultiThreadedStrategy {
    using signed_index_type = Sequence::signed_index_type;

    detail::AtomicBitset<SIZE> _slotStates; // tracks the state of each ringbuffer slot, true -> completed and ready to be read
    const std::size_t          _size = SIZE;
    const std::size_t          _mask = SIZE - 1;

public:
    Sequence                                                _reserveCursor; // slots can be reserved starting from _reserveCursor
    Sequence                                                _publishCursor; // slots are published and ready to be read until _publishCursor
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    MultiThreadedStrategy() = delete;

    explicit MultiThreadedStrategy()
    requires(SIZE != std::dynamic_extent)
    {}

    explicit MultiThreadedStrategy(std::size_t bufferSize)
    requires(SIZE == std::dynamic_extent)
        : _slotStates(detail::AtomicBitset<>(bufferSize)), _size(bufferSize), _mask(bufferSize - 1) {}

    MultiThreadedStrategy(const MultiThreadedStrategy&)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy&&) = delete;
    void operator=(const MultiThreadedStrategy&)         = delete;

    [[nodiscard]] signed_index_type next(std::size_t nSlotsToClaim = 1) {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");

        signed_index_type currentReserveCursor;
        signed_index_type nextReserveCursor;
        SpinWait          spinWait;
        do {
            currentReserveCursor = _reserveCursor.value();
            nextReserveCursor    = currentReserveCursor + static_cast<signed_index_type>(nSlotsToClaim);
            if (nextReserveCursor - getMinReaderCursor() > static_cast<signed_index_type>(_size)) { // not enough slots in buffer
                if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
                    _waitStrategy.signalAllWhenBlocking();
                }
                spinWait.spinOnce();
                continue;
            } else if (_reserveCursor.compareAndSet(currentReserveCursor, nextReserveCursor)) {
                break;
            }
        } while (true);

        return nextReserveCursor;
    }

    [[nodiscard]] std::optional<signed_index_type> tryNext(std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");

        signed_index_type currentReserveCursor;
        signed_index_type nextReserveCursor;

        do {
            currentReserveCursor = _reserveCursor.value();
            nextReserveCursor    = currentReserveCursor + static_cast<signed_index_type>(nSlotsToClaim);
            if (nextReserveCursor - getMinReaderCursor() > static_cast<signed_index_type>(_size)) { // not enough slots in buffer
                return std::nullopt;
            }
        } while (!_reserveCursor.compareAndSet(currentReserveCursor, nextReserveCursor));
        return nextReserveCursor;
    }

    [[nodiscard]] forceinline signed_index_type getRemainingCapacity() const noexcept { return static_cast<signed_index_type>(_size) - (_reserveCursor.value() - getMinReaderCursor()); }

    void publish(signed_index_type offset, std::size_t nSlotsToClaim) {
        for (std::size_t i = 0; i < nSlotsToClaim; i++) {
            _slotStates.set((offset + i) & _mask); // mark slots as published
        }

        // ensure publish cursor is only advanced after all prior slots are published
        signed_index_type currentPublishCursor;
        signed_index_type nextPublishCursor;
        do {
            currentPublishCursor = _publishCursor.value();
            nextPublishCursor    = currentPublishCursor;

            while (_slotStates.test(nextPublishCursor & _mask) && nextPublishCursor - currentPublishCursor < _slotStates.size()) {
                nextPublishCursor++;
            }
        } while (!_publishCursor.compareAndSet(currentPublishCursor, nextPublishCursor));

        //  clear completed slots up to the new published cursor
        for (std::size_t seq = static_cast<std::size_t>(currentPublishCursor); seq < nextPublishCursor; seq++) {
            _slotStates.reset(seq & _mask);
        }

        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline signed_index_type getMinReaderCursor() const noexcept {
        if (_readSequences->empty()) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }
};

static_assert(ClaimStrategyLike<MultiThreadedStrategy<1024, NoWaitStrategy>>);

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
    using value_type = SingleThreadedStrategy<size, TWaitStrategy>;
};

template<std::size_t size, WaitStrategyLike TWaitStrategy>
struct producer_type<size, ProducerType::Multi, TWaitStrategy> {
    using value_type = MultiThreadedStrategy<size, TWaitStrategy>;
};

template<std::size_t size, ProducerType producerType, WaitStrategyLike TWaitStrategy>
using producer_type_v = typename producer_type<size, producerType, TWaitStrategy>::value_type;

} // namespace detail

} // namespace gr

#endif // GNURADIO_CLAIMSTRATEGY_HPP
