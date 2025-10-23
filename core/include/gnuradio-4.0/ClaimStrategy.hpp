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

#include "AtomicBitset.hpp"
#include "Sequence.hpp"
#include "WaitStrategy.hpp"

namespace gr {

template<typename T>
concept ClaimStrategyLike = requires(T /*const*/ t, const std::size_t sequence, const std::size_t offset, const std::size_t nSlotsToClaim) {
    { t.next(nSlotsToClaim) } -> std::same_as<std::size_t>;
    { t.tryNext(nSlotsToClaim) } -> std::same_as<std::optional<std::size_t>>;
    { t.getRemainingCapacity() } -> std::same_as<std::size_t>;
    { t.publish(offset, nSlotsToClaim) } -> std::same_as<void>;
};

template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(hardware_constructive_interference_size) SingleProducerStrategy {
    const std::size_t _size = SIZE;

public:
    Sequence                                                _publishCursor;                      // slots are published and ready to be read until _publishCursor
    std::size_t                                             _reserveCursor{kInitialCursorValue}; // slots can be reserved starting from _reserveCursor, no need for atomics since this is called by a single publisher
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    explicit SingleProducerStrategy(const std::size_t bufferSize = SIZE) : _size(bufferSize) {};
    SingleProducerStrategy(const SingleProducerStrategy&)  = delete;
    SingleProducerStrategy(const SingleProducerStrategy&&) = delete;
    void operator=(const SingleProducerStrategy&)          = delete;

    std::size_t next(const std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

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

    [[nodiscard]] std::optional<std::size_t> tryNext(const std::size_t nSlotsToClaim) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        if (getRemainingCapacity() < nSlotsToClaim) { // not enough slots in buffer
            return std::nullopt;
        }
        _reserveCursor += nSlotsToClaim;
        return _reserveCursor;
    }

    [[nodiscard]] forceinline std::size_t getRemainingCapacity() const noexcept { return _size - (_reserveCursor - getMinReaderCursor()); }

    void publish(std::size_t offset, std::size_t nSlotsToClaim) {
        const auto sequence = offset + nSlotsToClaim;
        _publishCursor.setValue(sequence);
        _reserveCursor = sequence;
        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline std::size_t getMinReaderCursor() const noexcept {
        if (_readSequences->empty()) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }
};

static_assert(ClaimStrategyLike<SingleProducerStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(hardware_constructive_interference_size) MultiProducerStrategy {
    AtomicBitset<SIZE> _slotStates; // tracks the state of each ringbuffer slot, true -> completed and ready to be read
    const std::size_t  _size   = SIZE;
    const bool         _isPow2 = std::has_single_bit(SIZE);
    const std::size_t  _mask   = SIZE - 1; // valid only when _isPow2

    forceinline constexpr std::size_t calculateIndex(std::size_t seq) const noexcept {
        if constexpr (SIZE == std::dynamic_extent) {
            return _isPow2 ? (seq & _mask) : (seq % _size);
        } else {
            if constexpr (std::has_single_bit(SIZE)) {
                return seq & (SIZE - 1);
            } else {
                return seq % SIZE;
            }
        }
    }

public:
    Sequence                                                _reserveCursor; // slots can be reserved starting from _reserveCursor
    Sequence                                                _publishCursor; // slots are published and ready to be read until _publishCursor
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    MultiProducerStrategy() = delete;

    explicit MultiProducerStrategy(std::size_t bufferSize)
    requires(SIZE == std::dynamic_extent)
        : _slotStates(AtomicBitset<>(bufferSize)), _size(bufferSize), _mask(bufferSize - 1) {}

    MultiProducerStrategy(const MultiProducerStrategy&)  = delete;
    MultiProducerStrategy(const MultiProducerStrategy&&) = delete;
    void operator=(const MultiProducerStrategy&)         = delete;

    [[nodiscard]] std::size_t next(std::size_t nSlotsToClaim = 1) {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        std::size_t currentReserveCursor;
        std::size_t nextReserveCursor;
        SpinWait    spinWait;
        do {
            currentReserveCursor = _reserveCursor.value();
            nextReserveCursor    = currentReserveCursor + nSlotsToClaim;
            if (nextReserveCursor - getMinReaderCursor() > _size) { // not enough slots in buffer
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

    [[nodiscard]] std::optional<std::size_t> tryNext(std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= _size) && "nSlotsToClaim must be > 0 and <= bufferSize");

        std::size_t currentReserveCursor;
        std::size_t nextReserveCursor;

        do {
            currentReserveCursor = _reserveCursor.value();
            nextReserveCursor    = currentReserveCursor + nSlotsToClaim;
            if (nextReserveCursor - getMinReaderCursor() > _size) { // not enough slots in buffer
                return std::nullopt;
            }
        } while (!_reserveCursor.compareAndSet(currentReserveCursor, nextReserveCursor));
        return nextReserveCursor;
    }

    [[nodiscard]] forceinline std::size_t getRemainingCapacity() const noexcept { return _size - (_reserveCursor.value() - getMinReaderCursor()); }

    void publish(std::size_t offset, std::size_t nSlotsToClaim) {
        if (nSlotsToClaim == 0) {
            return;
        }
        setSlotsStates(offset, offset + nSlotsToClaim, true);

        // ensure publish cursor is only advanced after all prior slots are published
        std::size_t currentPublishCursor;
        std::size_t nextPublishCursor;
        do {
            currentPublishCursor = _publishCursor.value();
            nextPublishCursor    = currentPublishCursor;

            while (_slotStates.test(calculateIndex(nextPublishCursor)) && nextPublishCursor - currentPublishCursor < _slotStates.size()) {
                nextPublishCursor++;
            }
        } while (!_publishCursor.compareAndSet(currentPublishCursor, nextPublishCursor));

        //  clear completed slots up to the new published cursor
        setSlotsStates(currentPublishCursor, nextPublishCursor, false);

        if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

private:
    [[nodiscard]] forceinline std::size_t getMinReaderCursor() const noexcept {
        if (_readSequences->empty()) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }

    void setSlotsStates(std::size_t seqBegin, std::size_t seqEnd, bool value) {
        assert(seqBegin <= seqEnd);
        assert(seqEnd - seqBegin <= _size && "Begin cannot overturn end");
        const std::size_t beginSet  = calculateIndex(seqBegin);
        const std::size_t endSet    = calculateIndex(seqEnd);
        const auto        diffReset = seqEnd - seqBegin;

        if (beginSet <= endSet && diffReset < _size) {
            _slotStates.set(beginSet, endSet, value);
        } else {
            _slotStates.set(beginSet, _size, value);
            if (endSet > 0UZ) {
                _slotStates.set(0UZ, endSet, value);
            }
        }
        // Non-bulk AtomicBitset API
        //        for (std::size_t seq = static_cast<std::size_t>(seqBegin); seq < static_cast<std::size_t>(seqEnd); seq++) {
        //            _slotStates.set(wrap(seq), value);
        //        }
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

#endif // GNURADIO_CLAIMSTRATEGY_HPP
