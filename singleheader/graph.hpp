#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#define GNURADIO_CLAIM_STRATEGY_HPP

#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef GNURADIO_SEQUENCE_HPP
#include <sequence.hpp>
#endif
#ifndef GNURADIO_WAIT_STRATEGY_HPP
#include <wait_strategy.hpp>
#endif

namespace gr {

struct NoCapacityException : public std::runtime_error {
    NoCapacityException() : std::runtime_error("NoCapacityException"){};
};

// clang-format off

template<typename T>
concept ClaimStrategy = requires(T /*const*/ t, const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity,
        const std::int64_t cursorValue, const std::int64_t sequence, const std::int64_t availableSequence, const std::int32_t n_slots_to_claim) {
    { t.hasAvailableCapacity(dependents, requiredCapacity, cursorValue) } -> std::same_as<bool>;
    { t.next(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.tryNext(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.getRemainingCapacity(dependents) } -> std::same_as<std::int64_t>;
    { t.publish(sequence) } -> std::same_as<void>;
    { t.isAvailable(sequence) } -> std::same_as<bool>;
    { t.getHighestPublishedSequence(sequence, availableSequence) } -> std::same_as<std::int64_t>;
};

namespace claim_strategy::util {
constexpr unsigned    floorlog2(std::size_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }
constexpr unsigned    ceillog2(std::size_t x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }
}

template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class alignas(kCacheLine) SingleThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::int64_t _nextValue{ kInitialCursorValue }; // N.B. no need for atomics since this is called by a single publisher
    alignas(kCacheLine) mutable std::int64_t _cachedValue{ kInitialCursorValue };

public:
    SingleThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy){};
    SingleThreadedStrategy(const SingleThreadedStrategy &)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy &&) = delete;
    void operator=(const SingleThreadedStrategy &) = delete;

    bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity, const std::int64_t /*cursorValue*/) const noexcept {
        if (const std::int64_t wrapPoint = (_nextValue + requiredCapacity) - static_cast<std::int64_t>(_size); wrapPoint > _cachedValue || _cachedValue > _nextValue) {
            auto minSequence = detail::getMinimumSequence(dependents, _nextValue);
            _cachedValue     = minSequence;
            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int32_t n_slots_to_claim = 1) noexcept {
        assert((n_slots_to_claim > 0 && n_slots_to_claim <= static_cast<std::int32_t>(_size)) && "n_slots_to_claim must be > 0 and <= bufferSize");

        auto nextSequence = _nextValue + n_slots_to_claim;
        auto wrapPoint    = nextSequence - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _cachedValue; wrapPoint > cachedGatingSequence || cachedGatingSequence > _nextValue) {
            _cursor.setValue(_nextValue);

            SpinWait     spinWait;
            std::int64_t minSequence;
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

    std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        if (!hasAvailableCapacity(dependents, n_slots_to_claim, 0 /* unused cursor value */)) {
            throw NoCapacityException();
        }

        const auto nextSequence = _nextValue + n_slots_to_claim;
        _nextValue              = nextSequence;

        return nextSequence;
    }

    std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto consumed = detail::getMinimumSequence(dependents, _nextValue);
        const auto produced = _nextValue;

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        _cursor.setValue(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept { return sequence <= _cursor.value(); }
    [[nodiscard]] std::int64_t     getHighestPublishedSequence(std::int64_t /*nextSequence*/, std::int64_t availableSequence) const noexcept { return availableSequence; }
};
static_assert(ClaimStrategy<SingleThreadedStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class MultiThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::vector<std::int32_t> _availableBuffer; // tracks the state of each ringbuffer slot
    alignas(kCacheLine) std::shared_ptr<Sequence> _gatingSequenceCache = std::make_shared<Sequence>();
    const std::int32_t _indexMask;
    const std::int32_t _indexShift;

public:
    MultiThreadedStrategy() = delete;
    explicit MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(_size),
        _indexMask(_size - 1), _indexShift(claim_strategy::util::ceillog2(_size)) {
        for (std::size_t i = _size - 1; i != 0; i--) {
            setAvailableBufferValue(i, -1);
        }
        setAvailableBufferValue(0, -1);
    }
    MultiThreadedStrategy(const MultiThreadedStrategy &)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy &&) = delete;
    void               operator=(const MultiThreadedStrategy &) = delete;

    [[nodiscard]] bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int64_t requiredCapacity, const std::int64_t cursorValue) const noexcept {
        const auto wrapPoint = (cursorValue + requiredCapacity) - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _gatingSequenceCache->value(); wrapPoint > cachedGatingSequence || cachedGatingSequence > cursorValue) {
            const auto minSequence = detail::getMinimumSequence(dependents, cursorValue);
            _gatingSequenceCache->setValue(minSequence);

            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        SpinWait     spinWait;
        do {
            current                           = _cursor.value();
            next                              = current + n_slots_to_claim;

            std::int64_t wrapPoint            = next - static_cast<std::int64_t>(_size);
            std::int64_t cachedGatingSequence = _gatingSequenceCache->value();

            if (wrapPoint > cachedGatingSequence || cachedGatingSequence > current) {
                std::int64_t gatingSequence = detail::getMinimumSequence(dependents, current);

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

    [[nodiscard]] std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        do {
            current = _cursor.value();
            next    = current + n_slots_to_claim;

            if (!hasAvailableCapacity(dependents, n_slots_to_claim, current)) {
                throw NoCapacityException();
            }
        } while (!_cursor.compareAndSet(current, next));

        return next;
    }

    [[nodiscard]] std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto produced = _cursor.value();
        const auto consumed = detail::getMinimumSequence(dependents, produced);

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        setAvailable(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept {
        const auto index = calculateIndex(sequence);
        const auto flag  = calculateAvailabilityFlag(sequence);

        return _availableBuffer[static_cast<std::size_t>(index)] == flag;
    }

    [[nodiscard]] forceinline std::int64_t getHighestPublishedSequence(const std::int64_t lowerBound, const std::int64_t availableSequence) const noexcept {
        for (std::int64_t sequence = lowerBound; sequence <= availableSequence; sequence++) {
            if (!isAvailable(sequence)) {
                return sequence - 1;
            }
        }

        return availableSequence;
    }

private:
    void                      setAvailable(std::int64_t sequence) noexcept { setAvailableBufferValue(calculateIndex(sequence), calculateAvailabilityFlag(sequence)); }
    forceinline void          setAvailableBufferValue(std::size_t index, std::int32_t flag) noexcept { _availableBuffer[index] = flag; }
    [[nodiscard]] forceinline std::int32_t calculateAvailabilityFlag(const std::int64_t sequence) const noexcept { return static_cast<std::int32_t>(static_cast<std::uint64_t>(sequence) >> _indexShift); }
    [[nodiscard]] forceinline std::size_t calculateIndex(const std::int64_t sequence) const noexcept { return static_cast<std::size_t>(static_cast<std::int32_t>(sequence) & _indexMask); }
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
template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
struct producer_type;

template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Single, WAIT_STRATEGY> {
    using value_type = SingleThreadedStrategy<size, WAIT_STRATEGY>;
};
template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Multi, WAIT_STRATEGY> {
    using value_type = MultiThreadedStrategy<size, WAIT_STRATEGY>;
};

template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
using producer_type_v = typename producer_type<size, producerType, WAIT_STRATEGY>::value_type;

} // namespace detail

} // namespace gr


#endif // GNURADIO_CLAIM_STRATEGY_HPP
#ifndef GNURADIO_CIRCULAR_BUFFER_HPP
#define GNURADIO_CIRCULAR_BUFFER_HPP

#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 16000
#include <experimental/memory_resource>

namespace std::pmr {
using memory_resource = std::experimental::pmr::memory_resource;
template<typename T>
using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
} // namespace std::pmr
#else
#include <memory_resource>
#endif
#include <algorithm>
#include <bit>
#include <cassert> // to assert if compiled for debugging
#include <functional>
#include <numeric>
#include <ranges>
#include <span>

#include <fmt/format.h>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gr {
static constexpr bool has_posix_mmap_interface = true;
}

#define HAS_POSIX_MAP_INTERFACE
#else
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif
#else // #if defined __has_include -- required for portability
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif

#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#endif
#ifndef GNURADIO_SEQUENCE_HPP
#endif
#ifdef GNURADIO_WAIT_STRATEGY_HPP
#endif

namespace gr {

namespace util {
constexpr std::size_t
round_up(std::size_t num_to_round, std::size_t multiple) noexcept {
    if (multiple == 0) {
        return num_to_round;
    }
    const auto remainder = num_to_round % multiple;
    if (remainder == 0) {
        return num_to_round;
    }
    return num_to_round + multiple - remainder;
}
} // namespace util

// clang-format off
class double_mapped_memory_resource : public std::pmr::memory_resource {
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {

        const std::size_t size = 2 * required_size;
        if (size % 2LU != 0LU || size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::runtime_error(fmt::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size/2;

        static std::size_t _counter;
        const auto buffer_name = fmt::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto memfd_create = [name = buffer_name.c_str()](unsigned int flags) -> long {
            return syscall(__NR_memfd_create, name, flags);
        };
        int shm_fd = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::runtime_error(fmt::format("{} - memfd_create error {}: {}",  buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - ftruncate {}: {}",  buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // map the first half into the now available hole where the
        if (const void* second_copy = mmap(static_cast<char*> (first_copy) + size_half, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); second_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
}
#else
    [[nodiscard]] void* do_allocate(const std::size_t, std::size_t) override {
        throw std::runtime_error("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif


#ifdef HAS_POSIX_MAP_INTERFACE
    void  do_deallocate(void* p, std::size_t size, size_t alignment) override {

        if (munmap(p, size) == -1) {
            throw std::runtime_error(fmt::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
        }
    }
#else
    void  do_deallocate(void*, std::size_t, size_t) override { }
#endif

    bool  do_is_equal(const memory_resource& other) const noexcept override { return this == &other; }

public:
    static inline double_mapped_memory_resource* defaultAllocator() {
        static auto instance = double_mapped_memory_resource();
        return &instance;
    }

    template<typename T>
    requires (std::has_single_bit(sizeof(T)))
    static inline std::pmr::polymorphic_allocator<T> allocator()
    {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};



/**
 * @brief circular buffer implementation using double-mapped memory allocations
 * where the first SIZE-ed buffer is mirrored directly its end to mimic wrap-around
 * free bulk memory access. The buffer keeps a list of indices (using 'Sequence')
 * to keep track of which parts can be tread-safely read and/or written
 *
 *                          wrap-around point
 *                                 |
 *                                 v
 *  | buffer segment #1 (original) | buffer segment #2 (copy of #1) |
 *  0                            SIZE                            2*SIZE
 *                      writeIndex
 *                          v
 *  wrap-free write access  |<-  N_1 < SIZE   ->|
 *
 *  readIndex < writeIndex-N_2
 *      v
 *      |<- N_2 < SIZE ->|
 *
 * N.B N_AVAILABLE := (SIZE + writeIndex - readIndex ) % SIZE
 *
 * citation: <find appropriate first and educational reference>
 *
 * This implementation provides single- as well as multi-producer/consumer buffer
 * combinations for thread-safe CPU-to-CPU data transfer (optionally) using either
 * a) the POSIX mmaped(..)/munmapped(..) MMU interface, if available, and/or
 * b) the guaranteed portable standard C/C++ (de-)allocators as a fall-back.
 *
 * for more details see
 */
template <typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producer_type = ProducerType::Single, WaitStrategy WAIT_STRATEGY = SleepingWaitStrategy>
requires (std::has_single_bit(sizeof(T)))
class circular_buffer
{
    using Allocator         = std::pmr::polymorphic_allocator<T>;
    using BufferType        = circular_buffer<T, SIZE, producer_type, WAIT_STRATEGY>;
    using ClaimType         = detail::producer_type_v<SIZE, producer_type, WAIT_STRATEGY>;
    using DependendsType    = std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>;

    struct buffer_impl {
        alignas(kCacheLine) Allocator                   _allocator{};
        alignas(kCacheLine) const bool                  _is_mmap_allocated;
        alignas(kCacheLine) const std::size_t           _size;
        alignas(kCacheLine) std::vector<T, Allocator>   _data;
        alignas(kCacheLine) Sequence                    _cursor;
        alignas(kCacheLine) WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        alignas(kCacheLine) ClaimType                   _claim_strategy;
        // list of dependent reader indices
        alignas(kCacheLine) DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _is_mmap_allocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(min_size, _is_mmap_allocated)), _data(buffer_size(_size, _is_mmap_allocated), _allocator), _claim_strategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _is_mmap_allocated) {
            return _is_mmap_allocated ? util::round_up(min_size * sizeof(T), static_cast<std::size_t>(getpagesize())) / sizeof(T) : min_size;
        }
#else
        static std::size_t align_with_page_size(const std::size_t min_size, bool) {
            return min_size; // mmap() & getpagesize() not supported for non-POSIX OS
        }
#endif

        static std::size_t buffer_size(const std::size_t size, bool _is_mmap_allocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return _is_mmap_allocated ? size : 2 * size;
        }
    };

    template <typename U = T>
    class buffer_writer {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) bool                        _is_mmap_allocated;
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;
        alignas(kCacheLine) ClaimType*                  _claim_strategy;

    public:
        buffer_writer() = delete;
        buffer_writer(std::shared_ptr<buffer_impl> buffer) :
            _buffer(std::move(buffer)), _is_mmap_allocated(_buffer->_is_mmap_allocated),
            _size(_buffer->_size), _data(std::addressof(_buffer->_data)), _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer(buffer_writer&& other)
            : _buffer(std::move(other._buffer))
            , _is_mmap_allocated(_buffer->_is_mmap_allocated)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data))
            , _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer& operator=(buffer_writer tmp) {
            std::swap(_buffer, tmp._buffer);
            _is_mmap_allocated = _buffer->_is_mmap_allocated;
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            _claim_strategy = std::addressof(_buffer->_claim_strategy);

            return *this;
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        [[nodiscard]] constexpr auto get(std::size_t n_slots_to_claim) noexcept -> std::pair<std::span<U>, std::pair<std::size_t, std::int64_t>> {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (sequence + _size - n_slots_to_claim) % _size;
                return {{ &(*_data)[index], n_slots_to_claim }, {index, sequence - n_slots_to_claim } };
            } catch (const NoCapacityException &) {
                return { { /* empty span */ }, { 0, 0 }};
            }
        }

        constexpr void publish(std::pair<std::size_t, std::int64_t> token, std::size_t n_produced) {
            if (!_is_mmap_allocated) {
                // mirror samples below/above the buffer's wrap-around point
                const std::size_t index = token.first;
                const size_t nFirstHalf = std::min(_size - index, n_produced);
                const size_t nSecondHalf = n_produced  - nFirstHalf;

                auto& data = *_data;
                std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
            }
            _claim_strategy->publish(token.second + n_produced); // points at first non-writable index
        }

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return;
            }
            const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim);
            translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
        } // blocks until elements are available

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr bool try_publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return true;
            }
            try {
                const auto sequence = _claim_strategy->tryNext(*_buffer->_read_indices, n_slots_to_claim);
                translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
                return true;
            } catch (const NoCapacityException &) {
                return false;
            }
        }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _claim_strategy->getRemainingCapacity(*_buffer->_read_indices);
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const std::int64_t publishSequence, const Args&... args) {
            try {
                auto& data = *_data;
                const std::size_t index = (publishSequence + _size - n_slots_to_claim) % _size;
                std::span<U> writable_data(&data[index], n_slots_to_claim);
                if constexpr (std::is_invocable<Translator, std::span<T>&, std::int64_t, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), publishSequence - n_slots_to_claim, args...);
                } else {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), args...);
                }

                if (!_is_mmap_allocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claim_strategy->publish(publishSequence); // points at first non-writable index
            } catch (const std::exception& e) {
                throw e;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    };

    template<typename U = T>
    class buffer_reader
    {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) std::shared_ptr<Sequence>   _read_index = std::make_shared<Sequence>();
        alignas(kCacheLine) std::int64_t                _read_index_cached;
        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;

    public:
        buffer_reader() = delete;
        buffer_reader(std::shared_ptr<buffer_impl> buffer) :
            _buffer(buffer), _size(buffer->_size), _data(std::addressof(buffer->_data)) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_read_index});
            _read_index_cached = _read_index->value();
        }
        buffer_reader(buffer_reader&& other)
            : _read_index(std::move(other._read_index))
            , _read_index_cached(std::exchange(other._read_index_cached, _read_index->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data)) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_read_index, tmp._read_index);
            std::swap(_read_index_cached, tmp._read_index_cached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _read_index); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr std::span<const U> get(const std::size_t n_requested = 0) const noexcept {
            const auto& data = *_data;
            if constexpr (strict_check) {
                const std::size_t n = n_requested > 0 ? std::min(n_requested, available()) : available();
                return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
            }
            const std::size_t n = n_requested > 0 ? n_requested : available();
            return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
        }

        template <bool strict_check = true>
        [[nodiscard]] constexpr bool consume(const std::size_t n_elements = 1) noexcept {
            if constexpr (strict_check) {
                if (n_elements <= 0) {
                    return true;
                }
                if (n_elements > available()) {
                    return false;
                }
            }
            _read_index_cached = _read_index->addAndGet(static_cast<int64_t>(n_elements));
            return true;
        }

        [[nodiscard]] constexpr std::int64_t position() const noexcept { return _read_index_cached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _buffer->_cursor.value() - _read_index_cached;
        }
    };

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<buffer_impl> _shared_buffer_ptr;
    circular_buffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    circular_buffer() = delete;
    circular_buffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~circular_buffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] auto n_readers()       { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] auto claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] auto wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] auto cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<circular_buffer<int32_t>>);
// clang-format on

} // namespace gr

#endif // GNURADIO_CIRCULAR_BUFFER_HPP
#ifndef GNURADIO_BUFFER2_H
#define GNURADIO_BUFFER2_H

#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <span>

namespace gr {
namespace util {
template <typename T, typename...>
struct first_template_arg_helper;

template <template <typename...> class TemplateType,
          typename ValueType,
          typename... OtherTypes>
struct first_template_arg_helper<TemplateType<ValueType, OtherTypes...>> {
    using value_type = ValueType;
};

template <typename T>
constexpr auto* value_type_helper()
{
    if constexpr (requires { typename T::value_type; }) {
        return static_cast<typename T::value_type*>(nullptr);
    }
    else {
        return static_cast<typename first_template_arg_helper<T>::value_type*>(nullptr);
    }
}

template <typename T>
using value_type_t = std::remove_pointer_t<decltype(value_type_helper<T>())>;

template <typename... A>
struct test_fallback {
};

template <typename, typename ValueType>
struct test_value_type {
    using value_type = ValueType;
};

static_assert(std::is_same_v<int, value_type_t<test_fallback<int, float, double>>>);
static_assert(std::is_same_v<float, value_type_t<test_value_type<int, float>>>);
static_assert(std::is_same_v<int, value_type_t<std::span<int>>>);
static_assert(std::is_same_v<double, value_type_t<std::array<double, 42>>>);

} // namespace util

// clang-format off
// disable formatting until clang-format (v16) supporting concepts
template<class T>
concept BufferReader = requires(T /*const*/ t, const std::size_t n_items) {
    { t.get(n_items) }     -> std::same_as<std::span<const util::value_type_t<T>>>;
    { t.consume(n_items) } -> std::same_as<bool>;
    { t.position() }       -> std::same_as<std::int64_t>;
    { t.available() }      -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class Fn, typename T, typename ...Args>
concept WriterCallback = std::is_invocable_v<Fn, std::span<T>&, std::int64_t, Args...> || std::is_invocable_v<Fn, std::span<T>&, Args...>;

template<class T, typename ...Args>
concept BufferWriter = requires(T t, const std::size_t n_items, std::pair<std::size_t, std::int64_t> token, Args ...args) {
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                                 -> std::same_as<void>;
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }   -> std::same_as<void>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                             -> std::same_as<bool>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }-> std::same_as<bool>;
    { t.get(n_items) } -> std::same_as<std::pair<std::span<util::value_type_t<T>>, std::pair<std::size_t, std::int64_t>>>;
    { t.publish(token, n_items) } -> std::same_as<void>;
    { t.available() }         -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class T, typename ...Args>
concept Buffer = requires(T t, const std::size_t min_size, Args ...args) {
    { T(min_size, args...) };
    { t.size() }       -> std::same_as<std::size_t>;
    { t.new_reader() } -> BufferReader;
    { t.new_writer() } -> BufferWriter;
};

// compile-time unit-tests
namespace test {
template <typename T>
struct non_compliant_class {
};
template <typename T, typename... Args>
using WithSequenceParameter = decltype([](std::span<T>&, std::int64_t, Args...) { /* */ });
template <typename T, typename... Args>
using NoSequenceParameter = decltype([](std::span<T>&, Args...) { /* */ });
} // namespace test

static_assert(!Buffer<test::non_compliant_class<int>>);
static_assert(!BufferReader<test::non_compliant_class<int>>);
static_assert(!BufferWriter<test::non_compliant_class<int>>);

static_assert(WriterCallback<test::WithSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::WithSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::WithSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::NoSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
// clang-format on
} // namespace gr

#endif // GNURADIO_BUFFER2_H
#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP


#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <ranges>
#include <tuple>
#include <variant>

#if !__has_include(<source_location>)
#define HAVE_SOURCE_LOCATION 0
#else

#include <source_location>

#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace fair::graph {

namespace stdx = vir::stdx;

using fair::meta::fixed_string;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t { STREAM, MESSAGE }; // TODO: Think of a better name
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
                   typename T::value_type;
                   { t.pmt_type() } -> std::same_as<supported_type>;
                   { t.type() } -> std::same_as<port_type_t>;
                   { t.direction() } -> std::same_as<port_direction_t>;
                   { t.name() } -> std::same_as<std::string_view>;
                   { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
                   { t.disconnect() } -> std::same_as<connection_result_t>;
               };

class dynamic_port;

template<typename T>
concept has_fixed_port_info_v = requires {
                                    typename T::value_type;
                                    { T::static_name() };
                                    { T::direction() } -> std::same_as<port_direction_t>;
                                    { T::type() } -> std::same_as<port_type_t>;
                                };

template<typename T>
using has_fixed_port_info = std::integral_constant<bool, has_fixed_port_info_v<T>>;

template<typename T>
struct has_fixed_port_info_or_is_typelist : std::false_type {};

template<typename T>
    requires has_fixed_port_info_v<T>
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_port_info>)
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<std::size_t _min, std::size_t _max>
struct limits {
    using limits_tag                 = std::true_type;
    static constexpr std::size_t min = _min;
    static constexpr std::size_t max = _max;
};

template<typename T>
concept is_limits_v = requires { typename T::limits_tag; };

static_assert(!is_limits_v<int>);
static_assert(!is_limits_v<std::size_t>);
static_assert(is_limits_v<limits<0, 1024>>);

template<typename T>
using is_limits = std::integral_constant<bool, is_limits_v<T>>;

template<typename Port>
using port_type = typename Port::value_type;

template<typename Port>
using is_in_port = std::integral_constant<bool, Port::direction() == port_direction_t::INPUT>;

template<typename Port>
concept InPort = is_in_port<Port>::value;

template<typename Port>
using is_out_port = std::integral_constant<bool, Port::direction() == port_direction_t::OUTPUT>;

template<typename Port>
concept OutPort = is_out_port<Port>::value;

enum class work_result { success, has_unprocessed_data, inputs_empty, writers_not_available, error };

namespace work_policies {
struct one_by_one { // TODO: remove -- benchmark indicate this being inefficient

    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        auto inputs_available = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return ((input_port<Idx>(&self).reader().available() > 0) && ...); }
        (std::make_index_sequence<Self::input_ports::size>());

        if (!inputs_available) {
            return work_result::inputs_empty;
        }

        auto results = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            auto result  = meta::invoke_void_wrapped([&self]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); }, input_port<Idx>(&self).reader().get(1)[0]...);

            bool success = true;
            ((success = success && input_port<Idx>(&self).reader().consume(1)), ...);
            if (!success) throw fmt::format("Buffers reported more available data than was available");

            return result;
        }
        (std::make_index_sequence<Self::input_ports::size>());

        if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
            // process_one returned void

        } else if constexpr (requires { std::get<0>(results); }) {
            static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
            [&self, &results]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((output_port<Idx>(&self).writer().publish([&results](auto &w) { w[0] = std::get<Idx>(std::move(results)); }, 1)), ...);
            }
            (std::make_index_sequence<Self::output_ports::size>());

        } else {
            static_assert(Self::output_ports::size == 1);
            output_port<0>(&self).writer().publish([&results](auto &w) { w[0] = std::move(results); }, 1);
        }

        return work_result::success;
    }
};

struct read_many_and_publish_one_by_one { // TODO: remove -- benchmark indicate this being inefficient

    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        auto available_values_count = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return std::min({ std::numeric_limits<std::size_t>::max(), std::clamp((input_port<Idx>(&self).reader().available()), std::size_t{ 0 }, std::size_t{ 1024 })... }); // TODO min and max
        }
        (std::make_index_sequence<Self::input_ports::size>());

        if (available_values_count == 0) {
            return work_result::inputs_empty;
        }

        auto input_spans = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().get(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());

        for (std::size_t i = 0; i < available_values_count; ++i) {
            auto results = [&self, &input_spans, i]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                auto result = meta::invoke_void_wrapped([&self, &input_spans, i]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); },
                                                        std::get<Idx>(input_spans)[i]...);

                return result;
            }
            (std::make_index_sequence<Self::input_ports::size>());

            if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
                // process_one returned void

            } else if constexpr (requires { std::get<0>(results); }) {
                static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
                [&self, &results]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                    ((output_port<Idx>(&self).writer().publish([&results](auto &w) { w[0] = std::get<Idx>(std::move(results)); }, 1)), ...);
                }
                (std::make_index_sequence<Self::output_ports::size>());

            } else {
                static_assert(Self::output_ports::size == 1);
                output_port<0>(&self).writer().publish([&results](auto &w) { w[0] = std::move(results); }, 1);
            }
        }

        [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().consume(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());
    }
};

struct read_many_and_publish_many {
    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        bool              at_least_one_input_has_data = false;
        const std::size_t available_values_count      = [&self, &at_least_one_input_has_data]() {
            if constexpr (Self::input_ports::size > 0) {
                const auto availableForPort = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept {
                    const std::size_t available = port.reader().available();
                    if (available > 0LU) at_least_one_input_has_data = true;
                    if (available < port.min_buffer_size()) {
                        return 0LU;
                    } else {
                        return std::min(available, port.max_buffer_size());
                    }
                };

                const auto availableInAll = [&self, &availableForPort]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                    const auto betterMin = []<typename Arg, typename... Args>(Arg arg, Args &&...args) noexcept -> std::size_t {
                        if constexpr (sizeof...(Args) == 0) {
                            return arg;
                        } else {
                            return std::min(arg, args...);
                        }
                    };
                    return betterMin(availableForPort(input_port<Idx>(&self))...);
                }
                (std::make_index_sequence<Self::input_ports::size>());

                if (availableInAll < self.min_samples()) {
                    return 0LU;
                } else {
                    return std::min(availableInAll, self.max_samples());
                }
            } else {
                (void) self;
                return std::size_t{ 1 };
            }
        }();

        if (available_values_count == 0) {
            return at_least_one_input_has_data ? work_result::has_unprocessed_data : work_result::inputs_empty;
        }

        bool all_writers_available = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return ((output_port<Idx>(&self).writer().available() >= available_values_count) && ... && true);
        }
        (std::make_index_sequence<Self::output_ports::size>());

        if (!all_writers_available) {
            return work_result::writers_not_available;
        }

        auto input_spans = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().get(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());

        auto writers_tuple = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return std::make_tuple(output_port<Idx>(&self).writer().get(available_values_count)...);
        }
        (std::make_index_sequence<Self::output_ports::size>());

        // TODO: check here whether a process_one(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> process_one(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> process_bulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
        for (std::size_t i = 0; i < available_values_count; ++i) {
            auto results = [&self, &input_spans, i]<std::size_t... Idx>(std::index_sequence<Idx...>) noexcept {
                return meta::invoke_void_wrapped([&self]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); }, std::get<Idx>(input_spans)[i]...);
            }
            (std::make_index_sequence<Self::input_ports::size>());

            if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
                // process_one returned void

            } else if constexpr (requires { std::get<0>(results); }) {
                static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
                [&self, &results, &writers_tuple, i]<std::size_t... Idx>(std::index_sequence<Idx...>) { ((std::get<Idx>(writers_tuple).first /*data*/[i] = std::get<Idx>(std::move(results))), ...); }
                (std::make_index_sequence<Self::output_ports::size>());

            } else {
                static_assert(Self::output_ports::size == 1);
                std::get<0>(writers_tuple).first /*data*/[i] = std::move(results);
            }
        }

        if constexpr (Self::output_ports::size > 0) {
            [&self, &writers_tuple, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((output_port<Idx>(&self).writer().publish(std::get<Idx>(writers_tuple).second, available_values_count)), ...);
            }
            (std::make_index_sequence<Self::output_ports::size>());
        }

        bool success = true;
        if constexpr (Self::input_ports::size > 0) {
            [&self, available_values_count, &success]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((success = success && input_port<Idx>(&self).reader().consume(available_values_count)), ...);
            }
            (std::make_index_sequence<Self::input_ports::size>());
        }

        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self.name(), available_values_count);
        }

        return success ? work_result::success : work_result::error;
    }
};

using default_policy = read_many_and_publish_many;
} // namespace work_policies

template<typename T, fixed_string PortName, port_type_t Porttype, port_direction_t PortDirection, // TODO: sort default arguments
         std::size_t N_HISTORY = std::dynamic_extent, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, bool OPTIONAL = false,
         gr::Buffer BufferType = gr::circular_buffer<T>>
class port {
public:
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                = T;

    static constexpr bool IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;

    using port_tag                  = std::true_type;

private:
    using ReaderType          = decltype(std::declval<BufferType>().new_reader());
    using WriterType          = decltype(std::declval<BufferType>().new_writer());
    using IoType              = std::conditional_t<IS_INPUT, ReaderType, WriterType>;

    std::string  _name        = static_cast<std::string>(PortName);
    std::int16_t _priority    = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t  _n_history   = N_HISTORY;
    std::size_t  _min_samples = (MIN_SAMPLES == std::dynamic_extent ? 1 : MIN_SAMPLES);
    std::size_t  _max_samples = MAX_SAMPLES;
    bool         _connected   = false;

    IoType       _ioHandler   = new_io_handler();

public:
    [[nodiscard]] constexpr auto
    new_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(65536).new_reader();
        } else {
            return BufferType(65536).new_writer();
        }
    }

    [[nodiscard]] void *
    writer_handler_internal() noexcept {
        static_assert(IS_OUTPUT, "only to be used with output ports");
        return static_cast<void *>(std::addressof(_ioHandler));
    }

    [[nodiscard]] bool
    update_reader_internal(void *buffer_writer_handler_other) noexcept {
        static_assert(IS_INPUT, "only to be used with input ports");

        if (buffer_writer_handler_other == nullptr) {
            return false;
        }

        // TODO: If we want to allow ports with different buffer types to be mixed
        //       this will fail. We need to add a check that two ports that
        //       connect to each other use the same buffer type
        //       (std::any could be a viable approach)
        auto typed_buffer_writer = static_cast<WriterType *>(buffer_writer_handler_other);
        setBuffer(typed_buffer_writer->buffer());
        return true;
    }

public:
    port()             = default;

    port(const port &) = delete;

    auto
    operator=(const port &)
            = delete;

    port(std::string port_name, std::int16_t priority = 0, std::size_t n_history = 0, std::size_t min_samples = 0U, std::size_t max_samples = SIZE_MAX) noexcept
        : _name(std::move(port_name))
        , _priority{ priority }
        , _n_history(n_history)
        , _min_samples(min_samples)
        , _max_samples(max_samples) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr port(port &&other) noexcept
        : _name(std::move(other._name))
        , _priority{ other._priority }
        , _n_history(other._n_history)
        , _min_samples(other._min_samples)
        , _max_samples(other._max_samples) {}

    constexpr port &
    operator=(port &&other) {
        port tmp(std::move(other));
        std::swap(_name, tmp._name);
        std::swap(_priority, tmp._priority);
        std::swap(_n_history, tmp._n_history);
        std::swap(_min_samples, tmp._min_samples);
        std::swap(_max_samples, tmp._max_samples);
        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        return *this;
    }

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return Porttype;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    [[nodiscard]] constexpr supported_type
    pmt_type() const noexcept {
        return T();
    }

    [[nodiscard]] constexpr std::string_view
    name() const noexcept {
        if constexpr (!PortName.empty()) {
            return static_cast<std::string_view>(PortName);
        } else {
            return _name;
        }
    }

    [[nodiscard]] constexpr static bool
    is_optional() noexcept {
        return OPTIONAL;
    }

    [[nodiscard]] constexpr std::int16_t
    priority() const noexcept {
        return _priority;
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    n_history() const noexcept {
        if constexpr (N_HISTORY == std::dynamic_extent) {
            return _n_history;
        } else {
            return N_HISTORY;
        }
    }

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (MIN_SAMPLES == std::dynamic_extent) {
            return _min_samples;
        } else {
            return MIN_SAMPLES;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (MAX_SAMPLES == std::dynamic_extent) {
            return _max_samples;
        } else {
            return MAX_SAMPLES;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        if constexpr (IS_INPUT) {
            return connection_result_t::SUCCESS;
        } else {
            try {
                _ioHandler = BufferType(min_size).new_writer();
            } catch (...) {
                return connection_result_t::FAILED;
            }
        }
        return connection_result_t::SUCCESS;
    }

    [[nodiscard]] BufferType
    buffer() {
        return _ioHandler.buffer();
    }

    void
    setBuffer(gr::Buffer auto buffer) noexcept {
        if constexpr (IS_INPUT) {
            _ioHandler = std::move(buffer.new_reader());
            _connected = true;
        } else {
            _ioHandler = std::move(buffer.new_writer());
        }
    }

    [[nodiscard]] constexpr ReaderType &
    reader() const noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType &
    reader() noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() const noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        if (_connected == false) {
            return connection_result_t::FAILED;
        }
        _ioHandler = new_io_handler();
        _connected = false;
        return connection_result_t::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] connection_result_t
    connect(Other &&other) noexcept {
        static_assert(IS_OUTPUT && std::remove_cvref_t<Other>::IS_INPUT);
        auto src_buffer = writer_handler_internal();
        return std::forward<Other>(other).update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
    }

    friend class dynamic_port;
};

namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, std::size_t... Is>
consteval fair::meta::typelist<just_t<T, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, typename T, fixed_string BaseName, port_type_t Porttype, port_direction_t PortDirection, std::size_t MIN_SAMPLES = std::dynamic_extent,
         std::size_t MAX_SAMPLES = std::dynamic_extent>
using repeated_ports = decltype(detail::repeated_ports_impl<port<T, BaseName, Porttype, PortDirection, MIN_SAMPLES, MAX_SAMPLES>>(std::make_index_sequence<Count>()));

template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN = port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT = port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float, "in">>);
static_assert(Port<decltype(IN<float>("in"))>);
static_assert(Port<OUT<float, "out">>);
static_assert(Port<IN_MSG<float, "in_msg">>);
static_assert(Port<OUT_MSG<float, "out_msg">>);

static_assert(IN<float, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name(); });

/**
 *  Runtime capable wrapper to be used within a block. It's primary purpose is to allow the runtime
 *  initialisation/connections between blocks that are not in the same compilation unit.
 *  Ownership is defined by if the strongly-typed port P is either passed
 *  a) as an lvalue (i.e. P& -> keep reference), or
 *  b) as an rvalue (P&& -> being moved into dyn_port).
 *
 *  N.B. the intended use is within the node/block interface where there is -- once initialised --
 *  always a strong-reference between the strongly-typed port and it's dyn_port wrapper. I.e no ports
 *  are added or removed after the initialisation and the port life-time is coupled to that of it's
 *  parent block/node.
 */
class dynamic_port {
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type
        pmt_type() const noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
                = 0;

        [[nodiscard]] virtual std::string_view
        name() const noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port) noexcept
                = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(void *buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] void *
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(void *buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(!"This works only on input ports");
                return false;
            }
        }

    public:
        wrapper()                = delete;

        wrapper(const wrapper &) = delete;

        auto &
        operator=(const wrapper &)
                = delete;

        auto &
        operator=(wrapper &&)
                = delete;

        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        ~wrapper() override = default;

        [[nodiscard]] constexpr supported_type
        pmt_type() const noexcept override {
            return _value.pmt_type();
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept override {
            return _value.name();
        }

        [[nodiscard]] connection_result_t
        resize_buffer(std::size_t min_size) noexcept override {
            return _value.resize_buffer(min_size);
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept override {
            return _value.disconnect();
        }

        [[nodiscard]] connection_result_t
        connect(dynamic_port &dst_port) noexcept override {
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
            } else {
                assert(!"This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(void *buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type         = void; // a sterile port

    constexpr dynamic_port() = delete;

    template<Port T>
    constexpr dynamic_port(const T &arg) = delete;

    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    pmt_type() const noexcept {
        return _accessor->pmt_type();
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _accessor->name();
    }

    [[nodiscard]] connection_result_t
    resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        return _accessor->disconnect();
    }

    [[nodiscard]] connection_result_t
    connect(dynamic_port &dst_port) noexcept {
        return _accessor->connect(dst_port);
    }
};

static_assert(Port<dynamic_port>);

template<typename...>
struct fixed_node_ports_data_helper;

template<meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template
all_of<has_fixed_port_info> &&OutputPorts::template all_of<has_fixed_port_info> struct fixed_node_ports_data_helper<InputPorts, OutputPorts> {
    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

template<has_fixed_port_info_v... Ports>
struct fixed_node_ports_data_helper<Ports...> {
    using all_ports         = meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>;

    using input_ports       = typename all_ports ::template filter<is_in_port>;
    using output_ports      = typename all_ports ::template filter<is_out_port>;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;
};

template<typename... Arguments>
using fixed_node_ports_data = typename meta::typelist<Arguments...>::template filter<has_fixed_port_info_or_is_typelist>::template apply<fixed_node_ports_data_helper>;

// Ports can either be a list of ports instances,
// or two typelists containing port instances -- one for input
// ports and one for output ports
template<typename Derived, typename... Arguments>
// class node: fixed_node_ports_data<Arguments...>::all_ports::tuple_type {
class node : protected std::tuple<Arguments...> {
public:
    using fixed_ports_data  = fixed_node_ports_data<Arguments...>;

    using all_ports         = typename fixed_ports_data::all_ports;
    using input_ports       = typename fixed_ports_data::input_ports;
    using output_ports      = typename fixed_ports_data::output_ports;
    using input_port_types  = typename fixed_ports_data::input_port_types;
    using output_port_types = typename fixed_ports_data::output_port_types;

    using return_type       = typename output_port_types::tuple_or_type;
    using work_policy       = work_policies::default_policy;
    friend work_policy;

    using min_max_limits = typename meta::typelist<Arguments...>::template filter<is_limits>;
    static_assert(min_max_limits::size <= 1);

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{ std::string(fair::meta::type_name<Derived>()) };

    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv

    friend class graph;

public:
    auto &
    self() {
        return *static_cast<Derived *>(this);
    }

    const auto &
    self() const {
        return *static_cast<const Derived *>(this);
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _name;
    }

    void
    set_name(std::string name) noexcept {
        _name = std::move(name);
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(std::size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0                    = meta::first_type<input_port_types>;
            using V                      = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs                     = meta::transform_types<meta::rebind_simd_helper<V>::template rebind, input_port_types>;
            const std::tuple input_simds = Vs::template construct<meta::simd_load_ctor>(std::tuple{ in_ptr... });
            const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

#ifdef NOT_YET_PORTED_AS_IT_IS_UNUSED
    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...) && input_port_types::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto       &in0 = std::get<0>(std::tie(inputs...));
        const std::size_t n   = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ... && std::ranges::contiguous_range<Ins>) &&detail::vectorizable<return_type> && detail::node_can_process_simd<Derived>
                      && input_port_types ::template transform<stdx::native_simd>::template all_of<std::is_constructible>) {
            using V       = detail::reduce_to_widest_simd<input_port_types>;
            using Vs      = detail::transform_by_rebind_simd<V, input_port_types>;
            std::size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
                result.copy_to(std::ranges::data(out_range) + i, stdx::element_aligned);
            }

            return process_batch_simd_epilogue<std::bit_ceil(V::size()) / 2>(n - i, std::ranges::data(out_range) + i, (std::ranges::data(inputs) + i)...);
        } else { // no explicit SIMD
            auto             out_it    = out_range.begin();
            std::tuple       it_tuple  = { std::ranges::begin(inputs)... };
            const std::tuple end_tuple = { std::ranges::end(inputs)... };
            while (std::get<0>(it_tuple) != std::get<0>(end_tuple)) {
                *out_it = std::apply([this](auto &...its) { return self().process_one((*its++)...); }, it_tuple);
                ++out_it;
            }
            return true;
        }
    }
#endif

    [[nodiscard]] setting_map &
    exec_metrics() noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] setting_map const &
    exec_metrics() const noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] constexpr std::size_t
    min_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::min;
        } else {
            return 0;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::max;
        } else {
            return std::numeric_limits<std::size_t>::max();
        }
    }

    work_result
    work() noexcept {
        return work_policy::work(self());
    }
};

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::input_ports>();
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::output_ports>();
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

#define ENABLE_PYTHON_INTEGRATION
#ifdef ENABLE_PYTHON_INTEGRATION

// TODO: Not yet implemented
class dynamic_node {
private:
    // TODO: replace the following with array<2, vector<dynamic_port>>
    using dynamic_ports = std::vector<dynamic_port>;
    dynamic_ports                                         _dynamic_input_ports;
    dynamic_ports                                         _dynamic_output_ports;

    std::function<void(dynamic_ports &, dynamic_ports &)> _process;

public:
    void
    work() {
        _process(_dynamic_input_ports, _dynamic_output_ports);
    }

    template<typename T>
    void
    add_port(T &&port) {
        switch (port.direction()) {
        case port_direction_t::INPUT:
            if (auto portID = port_index<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_input_ports.emplace_back(std::forward<T>(port));
            break;

        case port_direction_t::OUTPUT:
            if (auto portID = port_index<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_output_ports.emplace_back(std::forward<T>(port));
            break;

        default: assert(false && "cannot add port with ANY designation");
        }
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::size_t index) {
        return index < _dynamic_input_ports.size() ? std::optional{ &_dynamic_input_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_input_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_input_ports.cbegin(), _dynamic_input_ports.cend(), portNameMatches);
        return it != _dynamic_input_ports.cend() ? std::optional{ std::distance(_dynamic_input_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::string_view name) {
        if (const auto index = dynamic_input_port_index(name); index.has_value()) {
            return &_dynamic_input_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::size_t index) {
        return index < _dynamic_output_ports.size() ? std::optional{ &_dynamic_output_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_output_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_output_ports.cbegin(), _dynamic_output_ports.cend(), portNameMatches);
        return it != _dynamic_output_ports.cend() ? std::optional{ std::distance(_dynamic_output_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::string_view name) {
        if (const auto index = dynamic_output_port_index(name); index.has_value()) {
            return &_dynamic_output_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_input_ports() const noexcept {
        return _dynamic_input_ports;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_output_ports() const noexcept {
        return _dynamic_output_ports;
    }
};

#endif

template<meta::source_node Left, meta::sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename Left::input_ports, meta::remove_at<InId, typename Right::input_ports>>,
                                meta::concat<meta::remove_at<OutId, typename Left::output_ports>, typename Right::output_ports>> {
private:
    // copy-paste from above, keep in sync
    using base = node<merged_node<Left, Right, OutId, InId>, meta::concat<typename Left::input_ports, meta::remove_at<InId, typename Right::input_ports>>,
                      meta::concat<meta::remove_at<OutId, typename Left::output_ports>, typename Right::output_ports>>;

    Left  left;
    Right right;

    template<std::size_t I>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return left.process_one(std::get<Is>(input_tuple)...); }
        (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = Left::input_port_types::size;
            constexpr std::size_t second_offset = Left::input_port_types::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp), std::get<second_offset + Js>(input_tuple)...);
        }
        (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename base::input_port_types;
    using output_port_types = typename base::output_port_types;
    using return_type       = typename base::return_type;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    template<meta::any_simd... Ts>
        requires meta::vectorizable<return_type> && input_port_types::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> &&meta::node_can_process_simd<Left>
            &&meta::node_can_process_simd<Right> constexpr stdx::rebind_simd_t<return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
              process_one(Ts... inputs) {
        return apply_right<InId, Right::input_port_types::size() - InId - 1>(std::tie(inputs...), apply_left<Left::input_port_types::size()>(std::tie(inputs...)));
    }

    template<typename... Ts>
    // In order to have nicer error messages, this is checked in the function body
    // requires input_port_types::template are_equal<std::remove_cvref_t<Ts>...>
    constexpr return_type
    process_one(Ts &&...inputs) {
        if constexpr (!input_port_types::template are_equal<std::remove_cvref_t<Ts>...>) {
            meta::print_types<decltype(this), input_port_types, std::remove_cvref_t<Ts>...> error{};
        }

        if constexpr (Left::output_port_types::size == 1) { // only the result from the right node needs to be returned
            return apply_right<InId, Right::input_port_types::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                 apply_left<Left::input_port_types::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<Left::input_port_types::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, Right::input_port_types::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (Left::output_port_types::size == 2 && Right::output_port_types::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (Left::output_port_types::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (Right::output_port_types::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<Left::output_port_types::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<Left::output_port_types::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    }
};

template<std::size_t OutId, std::size_t InId, meta::source_node A, meta::sink_node B>
[[gnu::always_inline]] constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>, typename std::remove_cvref_t<B>::input_port_types::template at<InId>>) {
        fair::meta::print_types<fair::meta::message_type<"OUTPUT_PORTS_ARE:">, typename std::remove_cvref_t<A>::output_port_types, std::integral_constant<int, OutId>,
                                typename std::remove_cvref_t<A>::output_port_types::template at<OutId>,

                                fair::meta::message_type<"INPUT_PORTS_ARE:">, typename std::remove_cvref_t<A>::input_port_types, std::integral_constant<int, InId>,
                                typename std::remove_cvref_t<A>::input_port_types::template at<InId>>{};
    }
    return { std::forward<A>(a), std::forward<B>(b) };
}

template<fixed_string OutName, fixed_string InName, meta::source_node A, meta::sink_node B>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) {
    constexpr std::size_t OutId = meta::indexForName<OutName, typename A::output_ports>();
    constexpr std::size_t InId  = meta::indexForName<InName, typename B::input_ports>();
    static_assert(OutId != -1);
    static_assert(InId != -1);
    static_assert(std::same_as<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>, typename std::remove_cvref_t<B>::input_port_types::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

class graph {
private:
    class node_model {
    public:
        virtual ~node_model() = default;

        virtual std::string_view
        name() const
                = 0;

        virtual work_result
        work() = 0;

        virtual void *
        raw() const
                = 0;
    };

    template<typename T>
    class reference_node_wrapper final : public node_model {
    private:
        T *_node;

        auto &
        data() {
            return *_node;
        }

        const auto &
        data() const {
            return *_node;
        }

    public:
        reference_node_wrapper(const reference_node_wrapper &other) = delete;

        reference_node_wrapper &
        operator=(const reference_node_wrapper &other)
                = delete;

        reference_node_wrapper(reference_node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        reference_node_wrapper &
        operator=(reference_node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~reference_node_wrapper() override = default;

        template<typename In>
        reference_node_wrapper(In &&node) : _node(std::forward<In>(node)) {}

        work_result
        work() override {
            return data().work();
        }

        std::string_view
        name() const override {
            return data().name();
        }

        void *
        raw() const override {
            return _node;
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        std::unique_ptr<node_model> _src_node;
        std::unique_ptr<node_model> _dst_node;
        std::size_t                 _src_port_index;
        std::size_t                 _dst_port_index;
        int32_t                     _weight;
        std::string                 _name; // custom edge name
        bool                        _connected;

    public:
        edge()             = delete;

        edge(const edge &) = delete;

        edge &
        operator=(const edge &)
                = delete;

        edge(edge &&) noexcept = default;

        edge &
        operator=(edge &&) noexcept
                = default;

        edge(std::unique_ptr<node_model> src_node, std::size_t src_port_index, std::unique_ptr<node_model> dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(std::move(src_node))
            , _dst_node(std::move(dst_node))
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
            // if (!_src_node->port<OUTPUT>(_src_port_index)) {
            //     throw fmt::format("source node '{}' has not output port id {}", std::string() /* _src_node->name() */, _src_port_index);
            // }
            // if (!_dst_node->port<INPUT>(_dst_port_index)) {
            //     throw fmt::format("destination node '{}' has not output port id {}", std::string() /*_dst_node->name()*/, _dst_port_index);
            // }
            // const dynamic_port& src_port = *_src_node->port<OUTPUT>(_src_port_index).value();
            // const dynamic_port& dst_port = *_dst_node->port<INPUT>(_dst_port_index).value();
            // if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            //     throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
            //         std::string() /*_src_node->name()*/, std::string() /*src_port.name()*/, src_port.pmt_type().index(),
            //         std::string() /*_dst_node->name()*/, std::string() /*dst_port.name()*/, dst_port.pmt_type().index(),
            //         _weight, _name, dst_port.pmt_type().index());
            // }
        }

        // edge(std::shared_ptr<node_model> src_node, std::string_view src_port_name, std::shared_ptr<node_model> dst_node, std::string_view dst_port_name, int32_t weight, std::string_view name) :
        //         _src_node(src_node), _dst_node(dst_node), _weight(weight), _name(name) {
        //     const auto src_id = _src_node->port_index<OUTPUT>(src_port_name);
        //     const auto dst_id = _dst_node->port_index<INPUT>(dst_port_name);
        //     if (!src_id) {
        //         throw std::invalid_argument(fmt::format("source node '{}' has not output port '{}'", std::string() /*_src_node->name()*/, src_port_name));
        //     }
        //     if (!dst_id) {
        //         throw fmt::format("destination node '{}' has not output port '{}'", std::string() /*_dst_node->name()*/, dst_port_name);
        //     }
        //     _src_port_index = src_id.value();
        //     _dst_port_index = dst_id.value();
        //     const dynamic_port& src_port = *src_node->port<OUTPUT>(_src_port_index).value();
        //     const dynamic_port& dst_port = *dst_node->port<INPUT>(_dst_port_index).value();
        //     if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
        //         throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
        //                           std::string() /*_src_node->name()*/, src_port.name(), src_port.pmt_type().index(),
        //                           std::string() /*_dst_node->name()*/, dst_port.name(), dst_port.pmt_type().index(),
        //                           _weight, _name, dst_port.pmt_type().index());
        //     }
        // }

        [[nodiscard]] constexpr int32_t
        weight() const noexcept {
            return _weight;
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept {
            return _name;
        }

        [[nodiscard]] constexpr bool
        connected() const noexcept {
            return _connected;
        }

        [[nodiscard]] connection_result_t
        connect() noexcept {
            return connection_result_t::FAILED;
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept { /* return _dst_node->port<INPUT>(_dst_port_index).value()->disconnect(); */
            return connection_result_t::FAILED;
        }
    };

    std::vector<edge>                        _edges;
    std::vector<std::unique_ptr<node_model>> _nodes;

public:
    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect(Source_ &src_node_raw, Destination_ &dst_node_raw, int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
        using Source      = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        static_assert(std::is_same_v<typename Source::output_port_types::template at<src_port_index>, typename Destination::input_port_types::template at<dst_port_index>>,
                      "The source port type needs to match the sink port type");

        OutPort auto &source_port      = output_port<src_port_index>(&src_node_raw);
        InPort auto  &destination_port = input_port<dst_port_index>(&dst_node_raw);

        if (!std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) { return registered_node->raw() == std::addressof(src_node_raw); })
            || !std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) { return registered_node->raw() == std::addressof(dst_node_raw); })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_node_raw.name(), src_port_index, dst_node_raw.name(), dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            std::unique_ptr<node_model> src_node = std::make_unique<reference_node_wrapper<Source>>(std::addressof(src_node_raw));
            std::unique_ptr<node_model> dst_node = std::make_unique<reference_node_wrapper<Destination>>(std::addressof(dst_node_raw));
            _edges.emplace_back(std::move(src_node), src_port_index, std::move(dst_node), src_port_index, weight, name);
        }

        return result;
    }

    template<fixed_string src_port_name, fixed_string dst_port_name, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect(Source_ &&src_node_raw, Destination_ &&dst_node_raw, int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
        using Source      = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        return connect<meta::indexForName<src_port_name, typename Source::output_ports>(), meta::indexForName<dst_port_name, typename Destination::input_ports>()>(std::forward<Source_>(src_node_raw),
                                                                                                                                                                   std::forward<Destination_>(
                                                                                                                                                                           dst_node_raw),
                                                                                                                                                                   weight, name);
    }

    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node>
    void
    register_node(Node &node) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        _nodes.push_back(std::make_unique<reference_node_wrapper<Node>>(std::addressof(node)));
    }

    work_result
    work() {
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == work_result::error) {
                    return work_result::error;
                } else if (result == work_result::has_unprocessed_data) {
                    // nothing
                } else if (result == work_result::inputs_empty) {
                    // nothing
                } else if (result == work_result::success) {
                    something_happened = true;
                } else if (result == work_result::writers_not_available) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return work_result::inputs_empty;
    }
};

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const connection_result_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_type_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_direction_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_domain_t &value) {
    return os << static_cast<int>(value);
}

#ifndef __EMSCRIPTEN__
auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
auto
this_source_location() {
    return "not yet implemented";
}
#endif // __EMSCRIPTEN__

} // namespace fair::graph

#endif // include guard
