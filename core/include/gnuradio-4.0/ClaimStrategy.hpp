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

struct NoCapacityException : public std::runtime_error {
    NoCapacityException() : std::runtime_error("NoCapacityException"){};
};

// clang-format off

template<typename T>
concept ClaimStrategy = requires(T /*const*/ t, const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t requiredCapacity,
        const std::make_signed_t<std::size_t> cursorValue, const std::make_signed_t<std::size_t> sequence, const std::make_signed_t<std::size_t> availableSequence, const std::size_t n_slots_to_claim) {
    { t.hasAvailableCapacity(dependents, requiredCapacity, cursorValue) } -> std::same_as<bool>;
    { t.next(dependents, n_slots_to_claim) } -> std::same_as<std::make_signed_t<std::size_t>>;
    { t.tryNext(dependents, n_slots_to_claim) } -> std::same_as<std::make_signed_t<std::size_t>>;
    { t.getRemainingCapacity(dependents) } -> std::same_as<std::make_signed_t<std::size_t>>;
    { t.publish(sequence) } -> std::same_as<void>;
    { t.isAvailable(sequence) } -> std::same_as<bool>;
    { t.getHighestPublishedSequence(sequence, availableSequence) } -> std::same_as<std::make_signed_t<std::size_t>>;
};

namespace claim_strategy::util {
constexpr unsigned    floorlog2(std::size_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }
constexpr unsigned    ceillog2(std::size_t x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }
}

template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class alignas(hardware_constructive_interference_size) SingleThreadedStrategy {
    using signed_index_type = Sequence::signed_index_type;
    const std::size_t _size;
    Sequence &_cursor;
    WAIT_STRATEGY &_waitStrategy;
    signed_index_type _nextValue{ kInitialCursorValue }; // N.B. no need for atomics since this is called by a single publisher
    mutable signed_index_type _cachedValue{ kInitialCursorValue };

public:
    SingleThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy){};
    SingleThreadedStrategy(const SingleThreadedStrategy &)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy &&) = delete;
    void operator=(const SingleThreadedStrategy &) = delete;

    bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t requiredCapacity, const signed_index_type/*cursorValue*/) const noexcept {
        if (const signed_index_type wrapPoint = (_nextValue + static_cast<signed_index_type>(requiredCapacity)) - static_cast<signed_index_type>(_size); wrapPoint > _cachedValue || _cachedValue > _nextValue) {
            auto minSequence = detail::getMinimumSequence(dependents, _nextValue);
            _cachedValue     = minSequence;
            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    signed_index_type next(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim = 1) noexcept {
        assert((n_slots_to_claim > 0 && n_slots_to_claim <= _size) && "n_slots_to_claim must be > 0 and <= bufferSize");

        auto nextSequence = _nextValue + static_cast<signed_index_type>(n_slots_to_claim);
        auto wrapPoint    = nextSequence - static_cast<signed_index_type>(_size);

        if (const auto cachedGatingSequence = _cachedValue; wrapPoint > cachedGatingSequence || cachedGatingSequence > _nextValue) {
            SpinWait     spinWait;
            signed_index_type minSequence;
            while (wrapPoint > (minSequence = detail::getMinimumSequence(dependents, _nextValue))) {
                if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                    _waitStrategy.signalAllWhenBlocking();
                }
                spinWait.spinOnce();
            }
            _cachedValue = minSequence;
        }
        _nextValue = nextSequence;

        return nextSequence;
    }

    signed_index_type tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        if (!hasAvailableCapacity(dependents, n_slots_to_claim, 0 /* unused cursor value */)) {
            throw NoCapacityException();
        }

        const auto nextSequence = _nextValue + static_cast<signed_index_type>(n_slots_to_claim);
        _nextValue              = nextSequence;

        return nextSequence;
    }

    signed_index_type getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto consumed = detail::getMinimumSequence(dependents, _nextValue);
        const auto produced = _nextValue;

        return static_cast<signed_index_type>(_size) - (produced - consumed);
    }

    void publish(signed_index_type sequence) {
        _cursor.setValue(sequence);
        _nextValue = sequence;
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(signed_index_type sequence) const noexcept { return sequence <= _cursor.value(); }
    [[nodiscard]] signed_index_type getHighestPublishedSequence(signed_index_type /*nextSequence*/, signed_index_type availableSequence) const noexcept { return availableSequence; }
};

static_assert(ClaimStrategy<SingleThreadedStrategy<1024, NoWaitStrategy>>);

template <std::size_t Size>
struct MultiThreadedStrategySizeMembers
{
    static constexpr std::int32_t _size = Size;
    static constexpr std::int32_t _indexShift = std::bit_width(Size);
};

template <>
struct MultiThreadedStrategySizeMembers<std::dynamic_extent> {
    const std::int32_t _size;
    const std::int32_t _indexShift;

    #ifdef __clang__
    explicit MultiThreadedStrategySizeMembers(std::size_t size) : _size(static_cast<std::int32_t>(size)), _indexShift(static_cast<std::int32_t>(std::bit_width(size))) {} //NOSONAR
    #else
    #pragma GCC diagnostic push // std::bit_width seems to be compiler and platform specific
    #pragma GCC diagnostic ignored "-Wuseless-cast"
    explicit MultiThreadedStrategySizeMembers(std::size_t size) : _size(static_cast<std::int32_t>(size)), _indexShift(static_cast<std::int32_t>(std::bit_width(size))) {} //NOSONAR
    #pragma GCC diagnostic pop
    #endif
};

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 *
 * The size argument (compile-time and run-time) must be a power-of-2 value.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
requires (SIZE == std::dynamic_extent or std::has_single_bit(SIZE))
class alignas(hardware_constructive_interference_size) MultiThreadedStrategy
: private MultiThreadedStrategySizeMembers<SIZE> {
    Sequence &_cursor;
    WAIT_STRATEGY &_waitStrategy;
    std::vector<std::int32_t> _availableBuffer; // tracks the state of each ringbuffer slot
    std::shared_ptr<Sequence> _gatingSequenceCache = std::make_shared<Sequence>();
    using MultiThreadedStrategySizeMembers<SIZE>::_size;
    using MultiThreadedStrategySizeMembers<SIZE>::_indexShift;
    using signed_index_type = Sequence::signed_index_type;

public:
    MultiThreadedStrategy() = delete;

    explicit
    MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy) requires (SIZE != std::dynamic_extent)
    : _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(SIZE, -1) {
    }

    explicit
    MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, std::size_t buffer_size)
    requires (SIZE == std::dynamic_extent)
    : MultiThreadedStrategySizeMembers<SIZE>(buffer_size),
      _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(buffer_size, -1) {
    }

    MultiThreadedStrategy(const MultiThreadedStrategy &)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy &&) = delete;
    void               operator=(const MultiThreadedStrategy &) = delete;

    [[nodiscard]] bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t requiredCapacity, const signed_index_type cursorValue) const noexcept {
        const auto wrapPoint = (cursorValue + static_cast<signed_index_type>(requiredCapacity)) - static_cast<signed_index_type>(_size);

        if (const auto cachedGatingSequence = _gatingSequenceCache->value(); wrapPoint > cachedGatingSequence || cachedGatingSequence > cursorValue) {
            const auto minSequence = detail::getMinimumSequence(dependents, cursorValue);
            _gatingSequenceCache->setValue(minSequence);

            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] signed_index_type next(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        signed_index_type current;
        signed_index_type next;

        SpinWait     spinWait;
        do {
            current = _cursor.value();
            next = current + static_cast<signed_index_type>(n_slots_to_claim);

            signed_index_type wrapPoint            = next - static_cast<signed_index_type>(_size);
            signed_index_type cachedGatingSequence = _gatingSequenceCache->value();

            if (wrapPoint > cachedGatingSequence || cachedGatingSequence > current) {
                signed_index_type gatingSequence = detail::getMinimumSequence(dependents, current);

                if (wrapPoint > gatingSequence) {
                    if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                        _waitStrategy.signalAllWhenBlocking();
                    }
                    spinWait.spinOnce();
                    continue;
                }

                _gatingSequenceCache->setValue(gatingSequence);
            } else if (_cursor.compareAndSet(current, next)) {
                break;
            }
        } while (true);

        return next;
    }

    [[nodiscard]] signed_index_type tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        signed_index_type current;
        signed_index_type next;

        do {
            current = _cursor.value();
            next    = current + static_cast<signed_index_type>(n_slots_to_claim);

            if (!hasAvailableCapacity(dependents, n_slots_to_claim, current)) {
                throw NoCapacityException();
            }
        } while (!_cursor.compareAndSet(current, next));

        return next;
    }

    [[nodiscard]] signed_index_type getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto produced = _cursor.value();
        const auto consumed = detail::getMinimumSequence(dependents, produced);

        return static_cast<signed_index_type>(_size) - (produced - consumed);
    }

    void publish(signed_index_type sequence) {
        setAvailable(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(signed_index_type sequence) const noexcept {
        const auto index = calculateIndex(sequence);
        const auto flag  = calculateAvailabilityFlag(sequence);

        return _availableBuffer[static_cast<std::size_t>(index)] == flag;
    }

    [[nodiscard]] forceinline signed_index_type getHighestPublishedSequence(const signed_index_type lowerBound, const signed_index_type availableSequence) const noexcept {
        for (signed_index_type sequence = lowerBound; sequence <= availableSequence; sequence++) {
            if (!isAvailable(sequence)) {
                return sequence - 1;
            }
        }

        return availableSequence;
    }

private:
    void                      setAvailable(signed_index_type sequence) noexcept { setAvailableBufferValue(calculateIndex(sequence), calculateAvailabilityFlag(sequence)); }
    forceinline void          setAvailableBufferValue(std::size_t index, std::int32_t flag) noexcept { _availableBuffer[index] = flag; }
    [[nodiscard]] forceinline std::int32_t calculateAvailabilityFlag(const signed_index_type sequence) const noexcept { return static_cast<std::int32_t>(static_cast<signed_index_type>(sequence) >> _indexShift); }
    [[nodiscard]] forceinline std::size_t calculateIndex(const signed_index_type sequence) const noexcept { return static_cast<std::size_t>(static_cast<std::int32_t>(sequence) & (_size - 1)); }
};

static_assert(ClaimStrategy<MultiThreadedStrategy<1024, NoWaitStrategy>>);
// clang-format on

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
template<std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
struct producer_type;

template<std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Single, WAIT_STRATEGY> {
    using value_type = SingleThreadedStrategy<size, WAIT_STRATEGY>;
};

template<std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Multi, WAIT_STRATEGY> {
    using value_type = MultiThreadedStrategy<size, WAIT_STRATEGY>;
};

template<std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
using producer_type_v = typename producer_type<size, producerType, WAIT_STRATEGY>::value_type;

} // namespace detail

} // namespace gr

#endif // GNURADIO_CLAIMSTRATEGY_HPP
