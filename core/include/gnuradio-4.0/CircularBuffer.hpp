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

#include <format>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>) && __has_include(<sys/syscall.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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
constexpr std::size_t round_up(std::size_t num_to_round, std::size_t multiple) noexcept {
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

class double_mapped_memory_resource : public std::pmr::memory_resource {
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {
        // the 2nd double mapped memory call mmap may fail and/or return an unsuitable return address which is unavoidable
        // this workaround retries to get a more favourable allocation up to three times before it throws the regular exception
        for (int retry_attempt = 0; retry_attempt < 3; retry_attempt++) {
            try {
                return do_allocate_internal(required_size, alignment);
            } catch (const std::system_error& e) { // explicitly caught for retry
                std::print("system-error: allocation failed (VERY RARE) '{}' - will retry, attempt: {}\n", e.what(), retry_attempt);
            } catch (const std::invalid_argument& e) { // explicitly caught for retry
                std::print("invalid_argument: allocation failed (VERY RARE) '{}' - will retry, attempt: {}\n", e.what(), retry_attempt);
            }
        }
        return do_allocate_internal(required_size, alignment);
    }
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] static void* do_allocate_internal(const std::size_t required_size, std::size_t alignment) { // NOSONAR

        const std::size_t size = 2 * required_size;
        if (size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::invalid_argument(std::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size / 2;

        static std::size_t _counter;
        const auto         buffer_name  = std::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto         memfd_create = [name = buffer_name.c_str()](unsigned int flags) { return syscall(__NR_memfd_create, name, flags); };
        auto               shm_fd       = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::system_error(errno, std::system_category(), std::format("{} - memfd_create error {}: {}", buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, std::format("{} - ftruncate {}: {}", buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, std::format("{} - failed munmap for first half {}: {}", buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            throw std::system_error(errorCode, std::format("{} - failed munmap for second half {}: {}", buffer_name, errno, strerror(errno)));
        }

        // Map the first half into the now available hole.
        // Note that the second_copy_addr mmap argument is only a hint and mmap might place the
        // mapping somewhere else: "If addr is not NULL, then the kernel takes it as  a hint about
        // where to place the mapping". The returned pointer therefore must equal second_copy_addr
        // for our contiguous mapping to work as intended.
        void* second_copy_addr = static_cast<char*>(first_copy) + size_half;
        if (const void* result = mmap(second_copy_addr, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0)); result != second_copy_addr) {
            std::error_code errorCode(errno, std::system_category());
            close(shm_fd);
            if (result == MAP_FAILED) {
                throw std::system_error(errorCode, std::format("{} - failed mmap for second copy {}: {}", buffer_name, errno, strerror(errno)));
            } else {
                ptrdiff_t diff2 = static_cast<const char*>(result) - static_cast<char*>(second_copy_addr);
                ptrdiff_t diff1 = static_cast<const char*>(result) - static_cast<char*>(first_copy);
                throw std::system_error(errorCode, std::format("{} - failed mmap for second copy: mismatching address -- result {} first_copy {} second_copy_addr {} - diff result-2nd {} diff result-1st {} size {}", buffer_name, gr::ptr(result), gr::ptr(first_copy), gr::ptr(second_copy_addr), diff2, diff1, 2 * size_half));
            }
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
    }
#else
    [[nodiscard]] static void* do_allocate_internal(const std::size_t, std::size_t) { // NOSONAR
        throw std::invalid_argument("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif

#ifdef HAS_POSIX_MAP_INTERFACE
    void do_deallocate(void* p, std::size_t size, std::size_t alignment) override { // NOSONAR

        if (munmap(p, size) == -1) {
            throw std::system_error(errno, std::system_category(), std::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
        }
    }
#else
    void do_deallocate(void*, std::size_t, size_t) override {}
#endif

    [[nodiscard]] bool do_is_equal(const memory_resource& other) const noexcept override { return this == &other; }

public:
    static inline double_mapped_memory_resource* defaultAllocator() {
        static auto instance = double_mapped_memory_resource();
        return &instance;
    }

    template<typename T>
    static inline std::pmr::polymorphic_allocator<T> allocator() {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};

/*
 * The enum determines how to handle samples if consume() or publish() was not called by the user in `processBulk` function.
 */
enum class SpanReleasePolicy {
    Terminate,  //  terminates the program
    ProcessAll, // consume/publish all samples
    ProcessNone // consume/publish zero samples
};

/*
 * The enum specifies whether to use the blocking reserve() method or the non-blocking tryReserve() method
 */
enum class WriterSpanReservePolicy {
    Reserve,    // use blocking reserve() method
    TryReserve, // use non-blocking tryReserve() method
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
template<typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producerType = ProducerType::Single, WaitStrategyLike TWaitStrategy = SleepingWaitStrategy>
class CircularBuffer {
    using Allocator  = std::pmr::polymorphic_allocator<T>;
    using BufferType = CircularBuffer<T, SIZE, producerType, TWaitStrategy>;
    using ClaimType  = detail::producer_type_v<SIZE, producerType, TWaitStrategy>;

    struct BufferImpl {
        Allocator                 _allocator{};
        const bool                _isMmapAllocated;
        const std::size_t         _size; // pre-condition: std::has_single_bit(_size)
        std::vector<T, Allocator> _data;
        ClaimType                 _claimStrategy;
        std::atomic<std::size_t>  _reader_count{0UZ};
        std::atomic<std::size_t>  _writer_count{0UZ};

        BufferImpl() = delete;
        BufferImpl(const std::size_t min_size, Allocator allocator)
            : _allocator(allocator),                                                                 //
              _isMmapAllocated(dynamic_cast<double_mapped_memory_resource*>(_allocator.resource())), //
              _size(align_with_page_size(std::bit_ceil(min_size), _isMmapAllocated)),                //
              _data(buffer_size(_size, _isMmapAllocated), _allocator),                               //
              _claimStrategy(ClaimType(_size)) {}

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _isMmapAllocated) {
            if (_isMmapAllocated) {
                const std::size_t pageSize    = static_cast<std::size_t>(getpagesize());
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
    }; // struct BufferImpl

    template<typename U = T>
    class Writer;

    template<typename U = T, SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
    class WriterSpan {
        Writer<U>* _parent = nullptr;

    public:
        using element_type     = T;
        using value_type       = typename std::remove_cv_t<T>;
        using iterator         = typename std::span<T>::iterator;
        using reverse_iterator = typename std::span<T>::reverse_iterator;
        using pointer          = typename std::span<T>::reverse_iterator;

        explicit WriterSpan(Writer<U>* parent) noexcept : _parent(parent) { _parent->_instanceCount++; };
        explicit constexpr WriterSpan(Writer<U>* parent, std::size_t index, std::size_t sequence, std::size_t nSlotsToClaim) noexcept : _parent(parent) {
            _parent->_index        = index;
            _parent->_offset       = sequence - nSlotsToClaim;
            _parent->_internalSpan = std::span<T>(&_parent->_buffer->_data.data()[index], nSlotsToClaim);
            _parent->_instanceCount++;
        }
        WriterSpan(const WriterSpan& other) : _parent(other._parent) { _parent->_instanceCount++; }
        WriterSpan& operator=(const WriterSpan& other) {
            if (this != &other) {
                _parent = other._parent;
                _parent->_instanceCount++;
            }
            return *this;
        }

        ~WriterSpan() {
            _parent->_instanceCount--;
            if (_parent->_instanceCount == 0) {
                if (!_parent->isPublishRequested()) {
                    if constexpr (spanReleasePolicy() == SpanReleasePolicy::Terminate) {
                        assert(false && "CircularBuffer::WriterSpan() - omitted publish() call for SpanReleasePolicy::Terminate");
                        std::abort();
                    } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessAll) {
                        publish(_parent->_internalSpan.size() - _parent->_nRequestedSamplesToPublish);
                    } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessNone) {
                        publish(0UZ);
                    }
                }

                if (!_parent->_buffer->_isMmapAllocated) {
                    const std::size_t size = _parent->_buffer->_size;
                    // mirror samples below/above the buffer's wrap-around point
                    const std::size_t nFirstHalf  = std::min(size - _parent->_index, _parent->_nRequestedSamplesToPublish);
                    const std::size_t nSecondHalf = _parent->_nRequestedSamplesToPublish - nFirstHalf;

                    auto& data = _parent->_buffer->_data;
                    std::copy(&data[_parent->_index], &data[_parent->_index + nFirstHalf], &data[_parent->_index + size]);
                    std::copy(&data[size], &data[size + nSecondHalf], &data[0]);
                }
                _parent->_buffer->_claimStrategy.publish(_parent->_offset, _parent->_nRequestedSamplesToPublish);
                _parent->_offset += _parent->_nRequestedSamplesToPublish;
#ifndef NDEBUG
                if constexpr (isMultiProducerStrategy()) {
                    if (!isFullyPublished()) {
                        std::print(stderr, "CircularBuffer::MultiWriter::WriterSpan() - did not publish {} samples\n", _parent->_internalSpan.size() - _parent->_nRequestedSamplesToPublish);
                        std::abort();
                    }

                } else {
                    if (!_parent->_internalSpan.empty() && !isPublishRequested()) {
                        std::print(stderr, "CircularBuffer::SingleWriter::WriterSpan() - omitted publish call for {} reserved samples\n", _parent->_internalSpan.size());
                        std::abort();
                    }
                }
#endif
                _parent->_nRequestedSamplesToPublish = 0;
                _parent->_internalSpan               = {};
            }
        }

        [[nodiscard]] constexpr static SpanReleasePolicy spanReleasePolicy() noexcept { return policy; }
        [[nodiscard]] constexpr std::size_t              nRequestedSamplesToPublish() const noexcept { return _parent->_nRequestedSamplesToPublish; }
        [[nodiscard]] constexpr bool                     isPublishRequested() const noexcept { return _parent->_isPublishRequested; }
        [[nodiscard]] constexpr bool                     isFullyPublished() const noexcept { return _parent->_internalSpan.size() == _parent->_nRequestedSamplesToPublish; }
        [[nodiscard]] constexpr static bool              isMultiProducerStrategy() noexcept { return std::is_base_of_v<MultiProducerStrategy<SIZE, TWaitStrategy>, ClaimType>; }
        [[nodiscard]] constexpr std::size_t              instanceCount() { return _parent->_instanceCount; }

        [[nodiscard]] constexpr std::size_t      size() const noexcept { return _parent->_internalSpan.size(); };
        [[nodiscard]] constexpr std::size_t      size_bytes() const noexcept { return size() * sizeof(T); };
        [[nodiscard]] constexpr bool             empty() const noexcept { return _parent->_internalSpan.empty(); }
        [[nodiscard]] constexpr iterator         cbegin() const noexcept { return _parent->_internalSpan.begin(); }
        [[nodiscard]] constexpr iterator         begin() const noexcept { return _parent->_internalSpan.begin(); }
        [[nodiscard]] constexpr iterator         cend() const noexcept { return _parent->_internalSpan.end(); }
        [[nodiscard]] constexpr iterator         end() const noexcept { return _parent->_internalSpan.end(); }
        [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _parent->_internalSpan.rbegin(); }
        [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _parent->_internalSpan.rend(); }
        [[nodiscard]] constexpr T*               data() const noexcept { return _parent->_internalSpan.data(); }
        T&                                       operator[](std::size_t i) const noexcept { return _parent->_internalSpan[i]; }
        T&                                       operator[](std::size_t i) noexcept { return _parent->_internalSpan[i]; }
        explicit(false) operator std::span<T>&() const noexcept { return _parent->_internalSpan; }
        explicit(false) operator std::span<T>&() noexcept { return _parent->_internalSpan; }

        constexpr void publish(std::size_t nSamplesToPublish) noexcept {
            assert(nSamplesToPublish <= _parent->_internalSpan.size() - _parent->_nRequestedSamplesToPublish && "n_produced must be <= than unpublished samples");
            _parent->_nRequestedSamplesToPublish += nSamplesToPublish;
            _parent->_isPublishRequested = true;
        }
    }; // class WriterSpan

    static_assert(WriterSpanLike<WriterSpan<T>>);

    template<typename U>
    class Writer {
        friend class WriterSpan<U, SpanReleasePolicy::Terminate>;
        friend class WriterSpan<U, SpanReleasePolicy::ProcessAll>;
        friend class WriterSpan<U, SpanReleasePolicy::ProcessNone>;

        using BufferTypeLocal = std::shared_ptr<BufferImpl>;

        BufferTypeLocal _buffer; // controls buffer life-cycle, the rest are cache optimisations

        // doesn't have to be atomic because this writer is accessed (by design) always by the same thread.
        // These are the parameters for WriterSpan, only one WriterSpan can be reserved per writer
        std::size_t  _nRequestedSamplesToPublish{0UZ}; // controls how many samples were already requested for publishing, multiple publish() calls are allowed
        bool         _isPublishRequested{true};        // controls if publish() was invoked
        std::size_t  _index{0UZ};
        std::size_t  _offset{0UZ};
        std::span<T> _internalSpan{};     // internal span is managed by Writer and is shared across all WriterSpans reserved by this Writer
        std::size_t  _instanceCount{0UZ}; // number of WriterSpan instances

    public:
        Writer() = delete;
        explicit Writer(std::shared_ptr<BufferImpl> buffer) noexcept : _buffer(std::move(buffer)) { _buffer->_writer_count.fetch_add(1UZ, std::memory_order_relaxed); };

        Writer(Writer&& other) noexcept
            : _buffer(std::move(other._buffer)),                                                  //
              _nRequestedSamplesToPublish(std::exchange(other._nRequestedSamplesToPublish, 0UZ)), //
              _isPublishRequested(std::exchange(other._isPublishRequested, true)),                //
              _index(std::exchange(other._index, 0UZ)),                                           //
              _offset(std::exchange(other._offset, 0)),                                           //
              _internalSpan(std::exchange(other._internalSpan, std::span<T>{})) {};

        Writer& operator=(Writer tmp) noexcept {
            std::swap(_buffer, tmp._buffer);
            std::swap(_nRequestedSamplesToPublish, tmp._nRequestedSamplesToPublish);
            std::swap(_isPublishRequested, tmp._isPublishRequested);
            std::swap(_index, tmp._index);
            std::swap(_offset, tmp._offset);
            std::swap(_internalSpan, tmp._internalSpan);

            return *this;
        }

        ~Writer() {
            if (_buffer) {
                _buffer->_writer_count.fetch_sub(1UZ, std::memory_order_relaxed);
            }
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return CircularBuffer(_buffer); };

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto tryReserve(std::size_t nSamples) noexcept -> WriterSpan<U, policy> {
            checkIfCanReserveAndAbortIfNeeded();
            _isPublishRequested         = false;
            _nRequestedSamplesToPublish = 0UZ;

            if (nSamples == 0) {
                return WriterSpan<U, policy>(this);
            }

            const std::optional<std::size_t> sequence = _buffer->_claimStrategy.tryNext(nSamples);
            if (sequence.has_value()) {
                const std::size_t index = (sequence.value() + _buffer->_size - nSamples) % _buffer->_size;
                return WriterSpan<U, policy>(this, index, sequence.value(), nSamples);
            } else {
                return WriterSpan<U, policy>(this);
            }
        }

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto reserve(std::size_t nSamples) noexcept -> WriterSpan<U, policy> {
            checkIfCanReserveAndAbortIfNeeded();
            _isPublishRequested         = false;
            _nRequestedSamplesToPublish = 0UZ;

            if (nSamples == 0) {
                return WriterSpan<U, policy>(this);
            }

            const auto        sequence = _buffer->_claimStrategy.next(nSamples);
            const std::size_t index    = (sequence + _buffer->_size - nSamples) % _buffer->_size;
            return WriterSpan<U, policy>(this, index, sequence, nSamples);
        }

        [[nodiscard]] constexpr std::size_t position() const noexcept { return _buffer->_claimStrategy._publishCursor.value(); }
        [[nodiscard]] constexpr std::size_t available() const noexcept { return _buffer->_claimStrategy.getRemainingCapacity(); }
        [[nodiscard]] constexpr bool        isPublishRequested() const noexcept { return _isPublishRequested; }
        [[nodiscard]] constexpr std::size_t nRequestedSamplesToPublish() const noexcept { return _nRequestedSamplesToPublish; };

    private:
        constexpr void checkIfCanReserveAndAbortIfNeeded() const noexcept {
            if constexpr (std::is_base_of_v<MultiProducerStrategy<SIZE, TWaitStrategy>, ClaimType>) {
                if (_internalSpan.size() - _nRequestedSamplesToPublish != 0) {
                    std::print(stderr,
                        "An error occurred: The method CircularBuffer::MultiWriter::reserve() was invoked for the second time in succession, "
                        "a previous WriterSpan was not fully published, {} samples remain unpublished.",
                        _internalSpan.size() - _nRequestedSamplesToPublish);
                    std::abort();
                }

            } else {
                if (!_internalSpan.empty() && !_isPublishRequested) {
                    std::print(stderr,
                        "An error occurred: The method CircularBuffer::SingleWriter::reserve() was invoked for the second time in succession "
                        "without calling publish() for a previous WriterSpan, {} samples was reserved.",
                        _internalSpan.size());
                    std::abort();
                }
            }
        }
    }; // class Writer
    // static_assert(BufferWriterLike<Writer<T>>);

    template<typename U = T>
    class Reader;

    template<typename U = T, SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
    class ReaderSpan {
        Reader<U>*         _parent = nullptr;
        std::span<const T> _internalSpan{};

    public:
        using element_type     = T;
        using value_type       = typename std::remove_cv_t<T>;
        using iterator         = typename std::span<const T>::iterator;
        using reverse_iterator = typename std::span<const T>::reverse_iterator;
        using pointer          = typename std::span<const T>::reverse_iterator;

        explicit ReaderSpan(const Reader<U>* parent) noexcept : _parent(parent) { _parent->_instanceCount++; }

        explicit constexpr ReaderSpan(Reader<U>* parent, std::size_t index, std::size_t nRequested) noexcept : _parent(parent), _internalSpan({&_parent->_buffer->_data.data()[index], nRequested}) { _parent->_instanceCount++; }

        ReaderSpan(const ReaderSpan& other) : _parent(other._parent), _internalSpan(other._internalSpan) { _parent->_instanceCount++; }

        ReaderSpan& operator=(const ReaderSpan& other) {
            if (this != &other) {
                _parent       = other._parent;
                _internalSpan = other._internalSpan;
                _parent->_rangesCounter++;
            }
            return *this;
        }

        virtual ~ReaderSpan() {
            _parent->_instanceCount--;

            if (_parent->_instanceCount == 0) {
                if (_parent->isConsumeRequested()) {
                    std::ignore = performConsume(_parent->_nRequestedSamplesToConsume);
                } else {
                    if constexpr (spanReleasePolicy() == SpanReleasePolicy::Terminate) {
                        assert(false && "CircularBuffer::ReaderSpan() - omitted consume() call for SpanReleasePolicy::Terminate");
                        std::abort();
                    } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessAll) {
                        std::ignore = performConsume(_parent->_nSamplesFirstGet);
                    } else if constexpr (spanReleasePolicy() == SpanReleasePolicy::ProcessNone) {
                        std::ignore = performConsume(0UZ);
                    }
                }
            }
        }

        [[nodiscard]] constexpr static SpanReleasePolicy spanReleasePolicy() noexcept { return policy; }
        [[nodiscard]] constexpr bool                     isConsumeRequested() const noexcept { return _parent->isConsumeRequested(); }
        [[nodiscard]] constexpr std::size_t              instanceCount() { return _parent->_instanceCount; }
        [[nodiscard]] constexpr std::size_t              nRequestedSamplesToConsume() const { return _parent->nRequestedSamplesToConsume(); }

        [[nodiscard]] constexpr std::size_t      size() const noexcept { return _internalSpan.size(); }
        [[nodiscard]] constexpr std::size_t      size_bytes() const noexcept { return size() * sizeof(T); }
        [[nodiscard]] constexpr bool             empty() const noexcept { return _internalSpan.empty(); }
        [[nodiscard]] constexpr iterator         cbegin() const noexcept { return _internalSpan.begin(); }
        [[nodiscard]] constexpr iterator         begin() const noexcept { return _internalSpan.begin(); }
        [[nodiscard]] constexpr iterator         cend() const noexcept { return _internalSpan.end(); }
        [[nodiscard]] constexpr iterator         end() const noexcept { return _internalSpan.end(); }
        [[nodiscard]] constexpr const T&         front() const noexcept { return _internalSpan.front(); }
        [[nodiscard]] constexpr const T&         back() const noexcept { return _internalSpan.back(); }
        [[nodiscard]] constexpr auto             first(std::size_t count) const noexcept { return _internalSpan.first(count); }
        [[nodiscard]] constexpr auto             last(std::size_t count) const noexcept { return _internalSpan.last(count); }
        [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _internalSpan.rbegin(); }
        [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _internalSpan.rend(); }
        [[nodiscard]] constexpr const T*         data() const noexcept { return _internalSpan.data(); }
        const T&                                 operator[](std::size_t i) const noexcept { return _internalSpan[i]; }
        const T&                                 operator[](std::size_t i) noexcept { return _internalSpan[i]; }
        explicit(false) operator const std::span<const T>&() const noexcept { return _internalSpan; }
        explicit(false) operator std::span<const T>&() noexcept { return _internalSpan; }
        // operator std::span<const T>&&() = delete;

        template<bool strict_check = true>
        [[nodiscard]] bool consume(std::size_t nSamples) noexcept {
            if (isConsumeRequested()) {
                assert(false && "An error occurred: The method CircularBuffer::ReaderSpan::consume() was invoked for the second time in succession, a corresponding ReaderSpan was already consumed.");
            }
            return tryConsume<strict_check>(nSamples);
        }

        template<bool strict_check = true>
        [[nodiscard]] bool tryConsume(std::size_t nSamples) noexcept {
            if (isConsumeRequested()) {
                return false;
            }
            if constexpr (strict_check) {
                if (nSamples > _parent->available()) {
                    return false;
                }
            }
            _parent->_nRequestedSamplesToConsume = nSamples;
            return true;
        }

    private:
        template<bool strict_check = true>
        [[nodiscard]] bool performConsume(std::size_t nSamples) noexcept {
            _parent->_nSamplesFirstGet           = std::numeric_limits<std::size_t>::max();
            _parent->_nRequestedSamplesToConsume = std::numeric_limits<std::size_t>::max();
            if constexpr (strict_check) {
                if (nSamples == 0) {
                    return true;
                }

                if (nSamples > _parent->available()) {
                    return false;
                }
            }
            _parent->_readIndexCached  = _parent->_readIndex->addAndGet(nSamples);
            _parent->_nSamplesConsumed = nSamples;
            return true;
        }

    }; // class ReaderSpan
    static_assert(ReaderSpanLike<ReaderSpan<T>>);

    template<typename U>
    class Reader {
        friend class ReaderSpan<U, SpanReleasePolicy::Terminate>;
        friend class ReaderSpan<U, SpanReleasePolicy::ProcessAll>;
        friend class ReaderSpan<U, SpanReleasePolicy::ProcessNone>;

        using BufferTypeLocal = std::shared_ptr<BufferImpl>;

        std::shared_ptr<Sequence> _readIndex = std::make_shared<Sequence>();
        std::size_t               _readIndexCached;
        BufferTypeLocal           _buffer;                                                    // controls buffer life-cycle, the rest are cache optimisations
        std::size_t               _nSamplesFirstGet{std::numeric_limits<std::size_t>::max()}; // Maximum number of samples returned by the first call to get() (when reader is consumed). Subsequent calls to get(), without calling consume() again, will return up to _nSamplesFirstGet.
        std::size_t               _instanceCount{0UZ};                                        // number of ReaderSpan instances

        // Samples are now consumed in a delayed manner. When the consume() method is called, the actual consumption does not happen immediately.
        // Instead, the real consume() operation is invoked in the destructor, when the last ReaderSpan is destroyed.
        std::size_t _nRequestedSamplesToConsume{std::numeric_limits<std::size_t>::max()}; // The number of samples requested for consumption by explicitly invoking the consume() method.
        std::size_t _nSamplesConsumed{0UZ};                                               // The number of samples actually consumed.

        std::size_t bufferIndex() const noexcept {
            const auto bitmask = _buffer->_size - 1;
            return _readIndexCached & bitmask;
        }

    public:
        Reader() = delete;
        explicit Reader(std::shared_ptr<BufferImpl> buffer) noexcept : _buffer(buffer) {
            gr::detail::addSequences(_buffer->_claimStrategy._readSequences, _buffer->_claimStrategy._publishCursor, {_readIndex});
            _buffer->_reader_count.fetch_add(1UZ, std::memory_order_relaxed);
            _readIndexCached = _readIndex->value();
        }

        Reader(Reader&& other) noexcept
            : _readIndex(std::move(other._readIndex)),                                      //
              _readIndexCached(std::exchange(other._readIndexCached, _readIndex->value())), //
              _buffer(other._buffer),                                                       //
              _nSamplesFirstGet(other._nSamplesFirstGet),                                   //
              _instanceCount(other._instanceCount),                                         //
              _nRequestedSamplesToConsume(other._nRequestedSamplesToConsume),               //
              _nSamplesConsumed(other._nSamplesConsumed) {}

        Reader& operator=(Reader tmp) noexcept {
            std::swap(_readIndex, tmp._readIndex);
            std::swap(_readIndexCached, tmp._readIndexCached);
            std::swap(_buffer, tmp._buffer);
            std::swap(_nSamplesFirstGet, tmp._nSamplesFirstGet);
            std::swap(_instanceCount, tmp._instanceCount);
            std::swap(_nRequestedSamplesToConsume, tmp._nRequestedSamplesToConsume);
            std::swap(_nSamplesConsumed, tmp._nSamplesConsumed);
            return *this;
        };
        ~Reader() {
            gr::detail::removeSequence(_buffer->_claimStrategy._readSequences, _readIndex);
            _buffer->_reader_count.fetch_sub(1UZ, std::memory_order_relaxed);
        }

        [[nodiscard]] constexpr BufferType  buffer() const noexcept { return CircularBuffer(_buffer); };
        [[nodiscard]] constexpr std::size_t nSamplesConsumed() const noexcept { return _nSamplesConsumed; };
        [[nodiscard]] constexpr bool        isConsumeRequested() const noexcept { return _nRequestedSamplesToConsume != std::numeric_limits<std::size_t>::max(); }
        [[nodiscard]] constexpr std::size_t nRequestedSamplesToConsume() const noexcept { return _nRequestedSamplesToConsume; }

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto get(const std::size_t nRequested = std::numeric_limits<std::size_t>::max()) noexcept -> ReaderSpan<U, policy> {
            if (isConsumeRequested()) {
                assert(false && "An error occurred: The method CircularBuffer::Reader::get() was invoked after consume() methods was explicitly invoked.");
            }

            std::size_t nSamples{nRequested};
            if (nSamples == std::numeric_limits<std::size_t>::max()) {
                nSamples = available();
            } else {
                assert(nSamples <= available() && "Number of required samples is more than number of available samples.");
            }
            if (_nSamplesFirstGet == std::numeric_limits<std::size_t>::max()) {
                _nSamplesFirstGet = nSamples;
                _nSamplesConsumed = 0UZ;
            } else {
                nSamples = std::min(nSamples, _nSamplesFirstGet);
            }
            return ReaderSpan<U, policy>(this, bufferIndex(), nSamples);
        }

        [[nodiscard]] constexpr std::size_t position() const noexcept { return _readIndexCached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept { return _buffer->_claimStrategy._publishCursor.value() - _readIndexCached; }
    }; // class Reader
    // static_assert(BufferReaderLike<Reader<T>>);

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface && std::is_trivially_copyable_v<T> && std::has_single_bit(sizeof(T))) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<BufferImpl> _sharedBufferPtr;
    explicit CircularBuffer(std::shared_ptr<BufferImpl> sharedBufferPtr) : _sharedBufferPtr(sharedBufferPtr) {}

public:
    CircularBuffer() = delete;
    explicit CircularBuffer(std::size_t minSize, Allocator allocator = DefaultAllocator()) : _sharedBufferPtr(std::make_shared<BufferImpl>(minSize, allocator)) {}
    ~CircularBuffer() = default;

    [[nodiscard]] std::size_t           size() const noexcept { return _sharedBufferPtr->_size; }
    [[nodiscard]] BufferWriterLike auto new_writer() { return Writer<T>(_sharedBufferPtr); }
    [[nodiscard]] BufferReaderLike auto new_reader() { return Reader<T>(_sharedBufferPtr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] std::size_t n_writers() const { return _sharedBufferPtr->_writer_count.load(std::memory_order_relaxed); }
    [[nodiscard]] std::size_t n_readers() const { return _sharedBufferPtr->_reader_count.load(std::memory_order_relaxed); }
    [[nodiscard]] const auto& claim_strategy() { return _sharedBufferPtr->_claimStrategy; }
    [[nodiscard]] const auto& wait_strategy() { return _sharedBufferPtr->_claimStrategy._wait_strategy; }
    [[nodiscard]] const auto& cursor_sequence() { return _sharedBufferPtr->_claimStrategy._publishCursor; }
};
static_assert(BufferLike<CircularBuffer<int32_t>>);

} // namespace gr
#endif // GNURADIO_CIRCULARBUFFER_HPP
