#ifndef GNURADIO_CIRCULARBUFFER_HPP
#define GNURADIO_CIRCULARBUFFER_HPP

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
#include <atomic>
#include <bit>
#include <cassert> // to assert if compiled for debugging
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <system_error>

#include <fmt/format.h>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __NR_memfd_create
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

#include "Buffer.hpp"
#include "ClaimStrategy.hpp"
#include "Sequence.hpp"
#include "WaitStrategy.hpp"

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
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {
        // the 2nd double mapped memory call mmap may fail and/or return an unsuitable return address which is unavoidable
        // this workaround retries to get a more favourable allocation up to three times before it throws the regular exception
        for (int retry_attempt = 0; retry_attempt < 3; retry_attempt++) {
            try {
                return do_allocate_internal(required_size, alignment);
            } catch (const std::system_error& e) { // explicitly caught for retry
                fmt::print("system-error: allocation failed (VERY RARE) '{}' - will retry, attempt: {}\n", e.what(), retry_attempt);
            } catch (const std::invalid_argument& e) { // explicitly caught for retry
                fmt::print("invalid_argument: allocation failed (VERY RARE) '{}' - will retry, attempt: {}\n", e.what(), retry_attempt);
            }
        }
        return do_allocate_internal(required_size, alignment);
    }
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] static void* do_allocate_internal(const std::size_t required_size, std::size_t alignment) { //NOSONAR

        const std::size_t size = 2 * required_size;
        if (size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::invalid_argument(fmt::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size/2;

        static std::size_t _counter;
        const auto buffer_name = fmt::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto memfd_create = [name = buffer_name.c_str()](unsigned int flags) {
            return syscall(__NR_memfd_create, name, flags);
        };
        auto shm_fd = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::system_error(errno, std::system_category(), fmt::format("{} - memfd_create error {}: {}",  buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, fmt::format("{} - ftruncate {}: {}",  buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // Map the first half into the now available hole.
        // Note that the second_copy_addr mmap argument is only a hint and mmap might place the
        // mapping somewhere else: "If addr is not NULL, then the kernel takes it as  a hint about
        // where to place the mapping". The returned pointer therefore must equal second_copy_addr
        // for our contiguous mapping to work as intended.
        void* second_copy_addr = static_cast<char*> (first_copy) + size_half;
        if (const void* result = mmap(second_copy_addr, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); result != second_copy_addr) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            if (result == MAP_FAILED) {
                throw std::system_error(errorCode, fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
            } else {
                ptrdiff_t diff2 = static_cast<const char*>(result) - static_cast<char*>(second_copy_addr);
                ptrdiff_t diff1 = static_cast<const char*>(result) - static_cast<char*>(first_copy);
                throw std::system_error(errorCode, fmt::format("{} - failed mmap for second copy: mismatching address -- result {} first_copy {} second_copy_addr {} - diff result-2nd {} diff result-1st {} size {}",
                                                     buffer_name, fmt::ptr(result), fmt::ptr(first_copy), fmt::ptr(second_copy_addr), diff2, diff1, 2*size_half));
            }
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
}
#else
    [[nodiscard]] static void* do_allocate_internal(const std::size_t, std::size_t) { //NOSONAR
        throw std::invalid_argument("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif

#ifdef HAS_POSIX_MAP_INTERFACE
    void  do_deallocate(void* p, std::size_t size, std::size_t alignment) override { //NOSONAR

        if (munmap(p, size) == -1) {
            throw std::system_error(errno, std::system_category(), fmt::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
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
    static inline std::pmr::polymorphic_allocator<T> allocator()
    {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};

/*
 * The enum determines how to handle samples if consume() or publish() was not called by the user in `processBulk` function.
 */
enum class SpanReleasePolicy {
    Terminate, //  terminates the program
    ProcessAll, // consume/publish all samples
    ProcessNone // consume/publish zero samples
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
class CircularBuffer
{
    using Allocator         = std::pmr::polymorphic_allocator<T>;
    using BufferType        = CircularBuffer<T, SIZE, producer_type, WAIT_STRATEGY>;
    using ClaimType         = detail::producer_type_v<SIZE, producer_type, WAIT_STRATEGY>;
    using DependendsType    = std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>;
    using signed_index_type = Sequence::signed_index_type;

    struct buffer_impl {
        Sequence                    _cursor;
        Allocator                   _allocator{};
        const bool                  _isMmapAllocated;
        const std::size_t             _size; // pre-condition: std::has_single_bit(_size)
        std::vector<T, Allocator>   _data;
        WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        ClaimType                   _claimStrategy;
        // list of dependent reader indices
        DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };
        std::atomic<std::size_t>    _writer_count{0UZ};

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _isMmapAllocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(std::bit_ceil(min_size), _isMmapAllocated)), _data(buffer_size(_size, _isMmapAllocated), _allocator), _claimStrategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _isMmapAllocated) {
            if (_isMmapAllocated) {
                const std::size_t pageSize = static_cast<std::size_t>(getpagesize());
                const std::size_t elementSize = sizeof(T);
                // least common multiple (lcm) of elementSize and pageSize
                std::size_t lcmValue = elementSize * pageSize / std::gcd(elementSize, pageSize);

                // adjust lcmValue to be larger than min_size
                while (lcmValue < min_size) {
                    lcmValue += lcmValue;
                }
                return lcmValue;
            } else {
                return min_size;
            }
        }
#else
        static std::size_t align_with_page_size(const std::size_t min_size, bool) {
            return min_size; // mmap() & getpagesize() not supported for non-POSIX OS
        }
#endif

        static std::size_t buffer_size(const std::size_t size, bool isMmapAllocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return isMmapAllocated ? size : 2 * size;
        }
    }; // struct buffer_impl

    template<typename U = T>
    class buffer_writer;

    template<typename U = T, SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
    class PublishableOutputRange {
        buffer_writer<U>* _parent = nullptr;

    public:
    using element_type = T;
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<T>::iterator;
    using reverse_iterator = typename std::span<T>::reverse_iterator;
    using pointer = typename std::span<T>::reverse_iterator;

    PublishableOutputRange() = delete;
    explicit PublishableOutputRange(buffer_writer<U>* parent) noexcept : _parent(parent) {
        _parent->_index = 0UZ;
        _parent->_offset = 0;
        _parent->_internalSpan = std::span<T>();
    #ifndef NDEBUG
        _parent->_rangesCounter++;
    #endif
    };
    explicit constexpr PublishableOutputRange(buffer_writer<U>* parent, std::size_t index, signed_index_type sequence, std::size_t nSlotsToClaim) noexcept :
        _parent(parent) {
        _parent->_index = index;
        _parent->_offset = sequence - static_cast<signed_index_type>(nSlotsToClaim);
        _parent->_internalSpan = std::span<T>(&_parent->_buffer->_data.data()[index], nSlotsToClaim);
    #ifndef NDEBUG
        _parent->_rangesCounter++;
    #endif
    }
    PublishableOutputRange(const PublishableOutputRange& other):_parent(other._parent) {
    #ifndef NDEBUG
        _parent->_rangesCounter++;
    #endif
    }
    PublishableOutputRange& operator=(const PublishableOutputRange& other) {
        if (this != &other) {
            _parent = other._parent;
    #ifndef NDEBUG
            _parent->_rangesCounter++;
    #endif
        }
       return *this;
    }

    ~PublishableOutputRange() {
    #ifndef NDEBUG
        _parent->_rangesCounter--;

        if constexpr (isMultiThreadedStrategy()) {
            if (_parent->_rangesCounter == 0 && !isFullyPublished()) {
                fmt::print(stderr, "CircularBuffer::multiple_writer::PublishableOutputRange() - did not publish {} samples\n", _parent->_internalSpan.size() - _parent->_nSamplesPublished);
                std::abort();
            }

        } else {
            if (_parent->_rangesCounter == 0 && !_parent->_internalSpan.empty() && !isPublished()) {
                fmt::print(stderr, "CircularBuffer::single_writer::PublishableOutputRange() - omitted publish call for {} reserved samples\n", _parent->_internalSpan.size());
                std::abort();
            }
        }
    #endif
    }

    [[nodiscard]] constexpr bool
    isPublished() const noexcept {
        return _parent->_isRangePublished;
    }

    [[nodiscard]] constexpr bool
    isFullyPublished() const noexcept {
        return _parent->_internalSpan.size() == _parent->_nSamplesPublished;
    }

    [[nodiscard]] constexpr static bool
    isMultiThreadedStrategy() noexcept {
        return std::is_base_of_v<MultiThreadedStrategy<SIZE, WAIT_STRATEGY>, ClaimType>;
    }

    [[nodiscard]] constexpr static SpanReleasePolicy
    spanReleasePolicy() noexcept {
        return policy;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return _parent->_internalSpan.size(); };
    [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return size() * sizeof(T); };
    [[nodiscard]] constexpr bool empty() const noexcept { return _parent->_internalSpan.empty(); }
    [[nodiscard]] constexpr iterator cbegin() const noexcept { return _parent->_internalSpan.cbegin(); }
    [[nodiscard]] constexpr iterator begin() const noexcept { return _parent->_internalSpan.begin(); }
    [[nodiscard]] constexpr iterator cend() const noexcept { return _parent->_internalSpan.cend(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return _parent->_internalSpan.end(); }
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _parent->_internalSpan.rbegin(); }
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _parent->_internalSpan.rend(); }
    [[nodiscard]] constexpr T* data() const noexcept { return _parent->_internalSpan.data(); }
    T& operator [](std::size_t i) const noexcept  {return _parent->_internalSpan[i]; }
    T& operator [](std::size_t i) noexcept { return _parent->_internalSpan[i]; }
    explicit(false) operator std::span<T>&() const noexcept { return _parent->_internalSpan; }
    operator std::span<T>&() noexcept { return _parent->_internalSpan; }

    constexpr void publish(std::size_t nSamplesToPublish) noexcept {
        assert(nSamplesToPublish <= _parent->_internalSpan.size() - _parent->_nSamplesPublished && "n_produced must be <= than unpublished samples");
        if (!_parent->_isMmapAllocated) {
            const std::size_t size = _parent->_size;
            // mirror samples below/above the buffer's wrap-around point
            const size_t nFirstHalf = std::min(size - _parent->_index, nSamplesToPublish);
            const size_t nSecondHalf = nSamplesToPublish - nFirstHalf;

            auto &data = _parent->_buffer->_data;
            std::copy(&data[_parent->_index], &data[_parent->_index + nFirstHalf], &data[_parent->_index + size]);
            std::copy(&data[size], &data[size + nSecondHalf], &data[0]);
        }
        _parent->_claimStrategy->publish(_parent->_offset, nSamplesToPublish);
        _parent->_offset += static_cast<signed_index_type>(nSamplesToPublish);
        _parent->_nSamplesPublished += nSamplesToPublish;
        _parent->_isRangePublished = true;
    }
    }; // class PublishableOutputRange

    static_assert(PublishableSpan<PublishableOutputRange<T>>);

    template <typename U>
    class buffer_writer {
        friend class PublishableOutputRange<U, SpanReleasePolicy::Terminate>;
        friend class PublishableOutputRange<U, SpanReleasePolicy::ProcessAll>;
        friend class PublishableOutputRange<U, SpanReleasePolicy::ProcessNone>;

        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        BufferTypeLocal _buffer; // controls buffer life-cycle, the rest are cache optimisations
        bool            _isMmapAllocated;
        std::size_t     _size;
        ClaimType*      _claimStrategy;

        // doesn't have to be atomic because this writer is accessed (by design) always by the same thread.
        // These are the parameters for PublishableOutputRange, only one PublishableOutputRange can be reserved per writer
        std::size_t       _nSamplesPublished {0UZ}; // controls how many samples were already published, multiple publish() calls are allowed
        bool              _isRangePublished {true};// controls if publish() was invoked
        std::size_t       _index {0UZ};
        signed_index_type _offset {0};
        std::span<T> _internalSpan{}; // internal span is managed by buffer_writer and is shared across all PublishableSpans reserved by this buffer_writer

#ifndef NDEBUG
        std::size_t _rangesCounter{0}; // this counter is used only in debug mode to check if publish() was invoked correctly
#endif

    public:
        buffer_writer() = delete;
        explicit buffer_writer(std::shared_ptr<buffer_impl> buffer) noexcept :
            _buffer(std::move(buffer)), _isMmapAllocated(_buffer->_isMmapAllocated),
            _size(_buffer->_size), _claimStrategy(std::addressof(_buffer->_claimStrategy)) { _buffer->_writer_count.fetch_add(1UZ, std::memory_order_relaxed); };

        buffer_writer(buffer_writer&& other) noexcept
            : _buffer(std::move(other._buffer))
            , _isMmapAllocated(_buffer->_isMmapAllocated)
            , _size(_buffer->_size)
            , _claimStrategy(std::addressof(_buffer->_claimStrategy))
            , _nSamplesPublished(std::exchange(other._nSamplesPublished, 0UZ))
            , _isRangePublished(std::exchange(other._isRangePublished, true))
            , _index(std::exchange(other._index, 0UZ))
            , _offset(std::exchange(other._offset, 0))
            , _internalSpan(std::exchange(other._internalSpan, std::span<T>{})) { };

        buffer_writer& operator=(buffer_writer tmp) noexcept {
            std::swap(_buffer, tmp._buffer);
            _isMmapAllocated = _buffer->_isMmapAllocated;
            _size = _buffer->_size;
            std::swap(_nSamplesPublished, tmp._nSamplesPublished);
            std::swap(_isRangePublished, tmp._isRangePublished);
            _claimStrategy = std::addressof(_buffer->_claimStrategy);
            std::swap(_index, tmp._index);
            std::swap(_offset, tmp._offset);
            std::swap(_internalSpan, tmp._internalSpan);

            return *this;
        }

        ~buffer_writer() {
            if (_buffer) {
                _buffer.get()->_writer_count.fetch_sub(1UZ, std::memory_order_relaxed);
            }
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return CircularBuffer(_buffer); };

        [[nodiscard]] constexpr std::size_t nSamplesPublished() const noexcept {return _nSamplesPublished;};

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto reserve(std::size_t nSamples) noexcept -> PublishableOutputRange<U, policy> {
            checkIfCanReserveAndAbortIfNeeded();
            _isRangePublished = false;
            _nSamplesPublished = 0UZ;

            if (nSamples == 0) {
                return PublishableOutputRange<U, policy>(this);
            }

            try {
                const auto sequence = _claimStrategy->next(*_buffer->_read_indices, nSamples); // alt: try_next
                const std::size_t index = (static_cast<std::size_t>(sequence) + _size - nSamples) % _size;
                return PublishableOutputRange<U, policy>(this, index, sequence, nSamples);
            } catch (const NoCapacityException &) {
                return PublishableOutputRange<U, policy>(this);
            }
        }

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void publish(Translator&& translator, std::size_t nSamples = 1, Args&&... args) {
            _isRangePublished = true;
            _nSamplesPublished += nSamples;
            if (nSamples <= 0 || _buffer->_read_indices->empty()) {
                return;
            }
            const auto sequence = _claimStrategy->next(*_buffer->_read_indices, nSamples);
            translate_and_publish(std::forward<Translator>(translator), nSamples, sequence, std::forward<Args>(args)...);
        } // blocks until elements are available

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr bool try_publish(Translator&& translator, std::size_t nSamples = 1, Args&&... args) {
            _isRangePublished = true;
            _nSamplesPublished += nSamples;
            if (nSamples <= 0 || _buffer->_read_indices->empty()) {
                return true;
            }
            try {
                const auto sequence = _claimStrategy->tryNext(*_buffer->_read_indices, nSamples);
                translate_and_publish(std::forward<Translator>(translator), nSamples, sequence, std::forward<Args>(args)...);
                return true;
            } catch (const NoCapacityException &) {
                return false;
            }
        }

        [[nodiscard]] constexpr signed_index_type position() const noexcept { return _buffer->_cursor.value(); }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return static_cast<std::size_t>(_claimStrategy->getRemainingCapacity(*_buffer->_read_indices));
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const signed_index_type publishSequence, const Args&... args) {
            try {
                auto& data = _buffer->_data;
                const std::size_t index = (static_cast<std::size_t>(publishSequence) + _size - n_slots_to_claim) % _size;
                std::span<U> writable_data(&data[index], n_slots_to_claim);
                if constexpr (std::is_invocable<Translator, std::span<T>&, signed_index_type, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), writable_data, publishSequence - static_cast<signed_index_type>(n_slots_to_claim), args...);
                } else if constexpr (std::is_invocable<Translator, std::span<T>&, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), writable_data, args...);
                } else {
                    static_assert(gr::meta::always_false<Translator>, "Translator does not provide a matching signature");
                }

                if (!_isMmapAllocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claimStrategy->publish(publishSequence - static_cast<signed_index_type>(n_slots_to_claim), n_slots_to_claim);
            } catch (const std::exception&) {
                throw;
            } catch (...) {
                throw std::runtime_error("CircularBuffer::translate_and_publish() - unknown user exception thrown");
            }
        }

        constexpr void checkIfCanReserveAndAbortIfNeeded() const noexcept {
            if constexpr (std::is_base_of_v<MultiThreadedStrategy<SIZE, WAIT_STRATEGY>, ClaimType>) {
                if (_internalSpan.size() - _nSamplesPublished != 0) {
                    fmt::print(stderr, "An error occurred: The method CircularBuffer::multiple_writer::reserve() was invoked for the second time in succession, "
                                    "a previous PublishableOutputRange was not fully published, {} samples remain unpublished.", _internalSpan.size() - _nSamplesPublished);
                    std::abort();
                }

            } else {
                if (!_internalSpan.empty() && not _isRangePublished) {
                    fmt::print(stderr, "An error occurred: The method CircularBuffer::single_writer::reserve() was invoked for the second time in succession "
                                    "without calling publish() for a previous PublishableOutputRange, {} samples was reserved.", _internalSpan.size());
                    std::abort();
                }
            }
        }
    }; // class buffer_writer
    //static_assert(BufferWriter<buffer_writer<T>>);

    template<typename U = T>
    class buffer_reader;

    template<typename U = T, SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
    class ConsumableInputRange {
        const buffer_reader<U>* _parent = nullptr;
        std::span<const T>      _internalSpan{};

    public:
    using element_type = T;
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<const T>::iterator;
    using reverse_iterator = typename std::span<const T>::reverse_iterator;
    using pointer = typename std::span<const T>::reverse_iterator;


    explicit ConsumableInputRange(const buffer_reader<U>* parent) noexcept : _parent(parent) {
        _parent->_rangesCounter++;
    }

    explicit constexpr ConsumableInputRange(const buffer_reader<U>* parent, std::size_t index, std::size_t nRequested) noexcept :
        _parent(parent), _internalSpan({ &_parent->_buffer->_data.data()[index], nRequested }) {
        _parent->_rangesCounter++;
    }

    ConsumableInputRange(const ConsumableInputRange& other)
        : _parent(other._parent),
          _internalSpan(other._internalSpan) {
        _parent->_rangesCounter++;
    }

    ConsumableInputRange& operator=(const ConsumableInputRange& other) {
        if (this != &other) {
            _parent = other._parent;
            _internalSpan = other._internalSpan;
            _parent->_rangesCounter++;
        }
        return *this;
    }

    ~ConsumableInputRange() {
        _parent->_rangesCounter--;

        if (_parent->_rangesCounter == 0) {
            if (_parent->isConsumeRequested()) {
                std::ignore = performConsume(_parent->_nSamplesToConsume);
            } else {
                if constexpr (spanReleasePolicy() == SpanReleasePolicy::Terminate) {
                     assert(false && "CircularBuffer::ConsumableInputRange() - omitted consume() call for SpanReleasePolicy::Terminate");
                     std::abort();
                 } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessAll) {
                     std::ignore = performConsume(_parent->_nSamplesFirstGet);
                 } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessNone){
                     std::ignore = performConsume(0UZ);
                 }
            }
        }
    }

    [[nodiscard]] constexpr static SpanReleasePolicy
    spanReleasePolicy() noexcept {
        return policy;
    }

    [[nodiscard]] constexpr bool
     isConsumeRequested() const noexcept {
         return _parent->isConsumeRequested();
     }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return _internalSpan.size(); }
    [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return size() * sizeof(T); }
    [[nodiscard]] constexpr bool empty() const noexcept { return _internalSpan.empty(); }
    [[nodiscard]] constexpr iterator cbegin() const noexcept { return _internalSpan.cbegin(); }
    [[nodiscard]] constexpr iterator begin() const noexcept { return _internalSpan.begin(); }
    [[nodiscard]] constexpr iterator cend() const noexcept { return _internalSpan.cend(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return _internalSpan.end(); }
    [[nodiscard]] constexpr const T& front() const noexcept { return _internalSpan.front(); }
    [[nodiscard]] constexpr const T& back() const noexcept { return _internalSpan.back(); }
    [[nodiscard]] constexpr auto first(std::size_t count) const noexcept { return _internalSpan.first(count); }
    [[nodiscard]] constexpr auto last(std::size_t count) const noexcept { return _internalSpan.last(count); }
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _internalSpan.rbegin(); }
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _internalSpan.rend(); }
    [[nodiscard]] constexpr const T* data() const noexcept { return _internalSpan.data(); }
    const T& operator [](std::size_t i) const noexcept  {return _internalSpan[i]; }
    const T& operator [](std::size_t i) noexcept { return _internalSpan[i]; }
    operator const std::span<const T>&() const noexcept { return _internalSpan; }
    operator std::span<const T>&() noexcept { return _internalSpan; }
    operator std::span<const T>&&() = delete;

    template <bool strict_check = true>
    [[nodiscard]] bool consume(std::size_t nSamples) const noexcept {
         if (isConsumeRequested()) {
            assert(false && "An error occurred: The method CircularBuffer::ConsumableInputRange::consume() was invoked for the second time in succession, a corresponding ConsumableInputRange was already consumed.");
        }
        return tryConsume<strict_check>(nSamples);
    }

    template <bool strict_check = true>
    [[nodiscard]] bool tryConsume(std::size_t nSamples) const noexcept {
        if (isConsumeRequested()) {
            return false;
        }
        if constexpr (strict_check) {
            if (nSamples > _parent->available()) {
                return false;
            }
        }
        _parent->_nSamplesToConsume = nSamples;
        return true;
    }

    private:
    template <bool strict_check = true>
    [[nodiscard]] bool performConsume(std::size_t nSamples) const noexcept {
        _parent->_nSamplesFirstGet = std::numeric_limits<std::size_t>::max();
        _parent->_nSamplesToConsume = std::numeric_limits<std::size_t>::max();
        if constexpr (strict_check) {
            if (nSamples <= 0) {
                return true;
            }

            if (nSamples > _parent->available()) {
                return false;
            }
        }
        _parent->_readIndexCached = _parent->_readIndex->addAndGet(static_cast<signed_index_type>(nSamples));
        _parent->_nSamplesConsumed = nSamples;
        return true;
    }

    }; // class ConsumableInputRange
    static_assert(ConsumableSpan<ConsumableInputRange<T>>);

    template<typename U>
    class buffer_reader
    {
        friend class ConsumableInputRange<U, SpanReleasePolicy::Terminate>;
        friend class ConsumableInputRange<U, SpanReleasePolicy::ProcessAll>;
        friend class ConsumableInputRange<U, SpanReleasePolicy::ProcessNone>;

        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        std::shared_ptr<Sequence>    _readIndex = std::make_shared<Sequence>();
        mutable signed_index_type    _readIndexCached;
        BufferTypeLocal              _buffer; // controls buffer life-cycle, the rest are cache optimisations
        std::size_t                  _size; // pre-condition: std::has_single_bit(_size)
        mutable std::size_t          _nSamplesFirstGet {std::numeric_limits<std::size_t>::max()}; // Maximum number of samples returned by the first call to get() (when reader is consumed). Subsequent calls to get(), without calling consume() again, will return up to _nSamplesFirstGet.
        mutable std::size_t          _rangesCounter {0UZ}; // reference counter for number of ConsumableSpanRanges

        // Samples are now consumed in a delayed manner. When the consume() method is called, the actual consumption does not happen immediately.
        // Instead, the real consume() operation is invoked in the destructor, when the last ConsumableInputRange is destroyed.
        mutable std::size_t          _nSamplesToConsume {std::numeric_limits<std::size_t>::max()}; // The number of samples requested for consumption by explicitly invoking the consume() method.
        mutable std::size_t          _nSamplesConsumed {0UZ}; // The number of samples actually consumed.

        std::size_t
        buffer_index() const noexcept {
            const auto bitmask = _size - 1;
            return static_cast<std::size_t>(_readIndexCached) & bitmask;
        }

    public:
        buffer_reader() = delete;
        explicit buffer_reader(std::shared_ptr<buffer_impl> buffer) noexcept :
            _buffer(buffer), _size(buffer->_size){
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_readIndex});
            _readIndexCached = _readIndex->value();
        }
        buffer_reader(buffer_reader&& other) noexcept
            : _readIndex(std::move(other._readIndex))
            , _readIndexCached(std::exchange(other._readIndexCached, _readIndex->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size)
            , _nSamplesFirstGet(std::move(other._nSamplesFirstGet))
            , _rangesCounter(std::move(other._rangesCounter))
            , _nSamplesToConsume(std::move(other._nSamplesToConsume))
            , _nSamplesConsumed(std::move(other._nSamplesConsumed)){
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_readIndex, tmp._readIndex);
            std::swap(_readIndexCached, tmp._readIndexCached);
            std::swap(_buffer, tmp._buffer);
            std::swap(_nSamplesFirstGet, tmp._nSamplesFirstGet);
            std::swap(_rangesCounter, tmp._rangesCounter);
            std::swap(_nSamplesToConsume, tmp._nSamplesToConsume);
            std::swap(_nSamplesConsumed, tmp._nSamplesConsumed);
            _size = _buffer->_size;
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _readIndex); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return CircularBuffer(_buffer); };

        [[nodiscard]] constexpr std::size_t nSamplesConsumed() const noexcept {return _nSamplesConsumed;};

        [[nodiscard]] constexpr bool isConsumeRequested() const noexcept {
            return _nSamplesToConsume != std::numeric_limits<std::size_t>::max();
        }

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto get(const std::size_t nRequested = std::numeric_limits<std::size_t>::max()) const noexcept -> ConsumableInputRange<U, policy> {
            if (isConsumeRequested()) {
                assert(false && "An error occurred: The method CircularBuffer::buffer_reader::get() was invoked after consume() methods was explicitly invoked.");
            }

            std::size_t nSamples { nRequested };
            if (nSamples == std::numeric_limits<std::size_t>::max()) {
                nSamples = available();
            } else {
                assert(nSamples <= available() && "Number of required samples is more than number of available samples.");
            }

           if (_nSamplesFirstGet == std::numeric_limits<std::size_t>::max() ) {
                _nSamplesFirstGet = nSamples;
                _nSamplesConsumed = 0UZ;
            } else {
                nSamples = std::min(nSamples, _nSamplesFirstGet);
            }
            return ConsumableInputRange<U, policy>(this, buffer_index(), nSamples);
        }

        [[nodiscard]] constexpr signed_index_type position() const noexcept { return _readIndexCached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            const auto last = _buffer->_claimStrategy.getHighestPublishedSequence(_readIndexCached + 1, _buffer->_cursor.value());
            return static_cast<std::size_t>(last - _readIndexCached);
        }
    }; // class buffer_reader

    //static_assert(BufferReader<buffer_reader<T>>);

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface && std::is_trivially_copyable_v<T>) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<buffer_impl> _shared_buffer_ptr;
    explicit CircularBuffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    CircularBuffer() = delete;
    explicit CircularBuffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~CircularBuffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] std::size_t n_writers() const { return _shared_buffer_ptr->_writer_count.load(std::memory_order_relaxed); }
    [[nodiscard]] std::size_t n_readers() const { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] const auto &claim_strategy()  { return _shared_buffer_ptr->_claimStrategy; }
    [[nodiscard]] const auto &wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] const auto &cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<CircularBuffer<int32_t>>);
// clang-format on

} // namespace gr
#endif // GNURADIO_CIRCULARBUFFER_HPP
