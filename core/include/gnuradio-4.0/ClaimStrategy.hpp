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
concept ClaimStrategyLike = requires(T /*const*/ t, const Sequence::signed_index_type sequence, const Sequence::signed_index_type offset, const std::size_t nSlotsToClaim) {
    { t.next(nSlotsToClaim) } -> std::same_as<Sequence::signed_index_type>;
    { t.tryNext(nSlotsToClaim) } -> std::same_as<std::optional<Sequence::signed_index_type>>;
    { t.getRemainingCapacity() } -> std::same_as<Sequence::signed_index_type>;
    { t.publish(offset, nSlotsToClaim) } -> std::same_as<void>;
};

template<std::size_t SIZE = std::dynamic_extent, WaitStrategyLike TWaitStrategy = BusySpinWaitStrategy>
class alignas(hardware_constructive_interference_size) SingleProducerStrategy {
    using signed_index_type = Sequence::signed_index_type;

    const std::size_t _size = SIZE;

public:
    Sequence                                                _publishCursor;                      // slots are published and ready to be read until _publishCursor
    signed_index_type                                       _reserveCursor{kInitialCursorValue}; // slots can be reserved starting from _reserveCursor, no need for atomics since this is called by a single publisher
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    explicit SingleProducerStrategy(const std::size_t bufferSize = SIZE) : _size(bufferSize) {};
    SingleProducerStrategy(const SingleProducerStrategy&)  = delete;
    SingleProducerStrategy(const SingleProducerStrategy&&) = delete;
    void operator=(const SingleProducerStrategy&)          = delete;

    signed_index_type next(const std::size_t nSlotsToClaim = 1) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");

        SpinWait spinWait;
        while (getRemainingCapacity() < static_cast<signed_index_type>(nSlotsToClaim)) { // while not enough slots in buffer
            if constexpr (hasSignalAllWhenBlocking<TWaitStrategy>) {
                _waitStrategy.signalAllWhenBlocking();
            }
            spinWait.spinOnce();
        }
        _reserveCursor += static_cast<signed_index_type>(nSlotsToClaim);
        return _reserveCursor;
    }

    [[nodiscard]] std::optional<signed_index_type> tryNext(const std::size_t nSlotsToClaim) noexcept {
        assert((nSlotsToClaim > 0 && nSlotsToClaim <= static_cast<std::size_t>(_size)) && "nSlotsToClaim must be > 0 and <= bufferSize");

        if (getRemainingCapacity() < static_cast<signed_index_type>(nSlotsToClaim)) { // not enough slots in buffer
            return std::nullopt;
        }
        _reserveCursor += static_cast<signed_index_type>(nSlotsToClaim);
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

static_assert(ClaimStrategyLike<SingleProducerStrategy<1024, NoWaitStrategy>>);

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
class alignas(hardware_constructive_interference_size) MultiProducerStrategy {
    using signed_index_type = Sequence::signed_index_type;

    AtomicBitset<SIZE> _slotStates; // tracks the state of each ringbuffer slot, true -> completed and ready to be read
    const std::size_t  _size = SIZE;
    const std::size_t  _mask = SIZE - 1;

public:
    Sequence                                                _reserveCursor; // slots can be reserved starting from _reserveCursor
    Sequence                                                _publishCursor; // slots are published and ready to be read until _publishCursor
    TWaitStrategy                                           _waitStrategy;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> _readSequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()}; // list of dependent reader sequences

    MultiProducerStrategy() = delete;

    explicit MultiProducerStrategy()
    requires(SIZE != std::dynamic_extent)
    {}

    explicit MultiProducerStrategy(std::size_t bufferSize)
    requires(SIZE == std::dynamic_extent)
        : _slotStates(AtomicBitset<>(bufferSize)), _size(bufferSize), _mask(bufferSize - 1) {}

    MultiProducerStrategy(const MultiProducerStrategy&)  = delete;
    MultiProducerStrategy(const MultiProducerStrategy&&) = delete;
    void operator=(const MultiProducerStrategy&)         = delete;

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
        if (nSlotsToClaim == 0) {
            return;
        }
        setSlotsStates(offset, offset + static_cast<signed_index_type>(nSlotsToClaim), true);

        // ensure publish cursor is only advanced after all prior slots are published
        signed_index_type currentPublishCursor;
        signed_index_type nextPublishCursor;
        do {
            currentPublishCursor = _publishCursor.value();
            nextPublishCursor    = currentPublishCursor;

            while (_slotStates.test(static_cast<std::size_t>(nextPublishCursor) & _mask) && static_cast<std::size_t>(nextPublishCursor - currentPublishCursor) < _slotStates.size()) {
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
    [[nodiscard]] forceinline signed_index_type getMinReaderCursor() const noexcept {
        if (_readSequences->empty()) {
            return kInitialCursorValue;
        }
        return std::ranges::min(*_readSequences | std::views::transform([](const auto& cursor) { return cursor->value(); }));
    }

    void setSlotsStates(signed_index_type seqBegin, signed_index_type seqEnd, bool value) {
        assert(static_cast<std::size_t>(seqEnd - seqBegin) <= _size && "Begin cannot overturn end");
        const std::size_t beginSet  = static_cast<std::size_t>(seqBegin) & _mask;
        const std::size_t endSet    = static_cast<std::size_t>(seqEnd) & _mask;
        const auto        diffReset = static_cast<std::size_t>(seqEnd - seqBegin);

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
        //            _slotStates.set(seq & _mask, value);
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
