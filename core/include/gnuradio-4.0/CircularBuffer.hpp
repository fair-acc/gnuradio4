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
        for (int retry_attempt=0; retry_attempt < 3; retry_attempt++) {
            try {
                return do_allocate_internal(required_size, alignment);
            } catch (std::system_error& e) { // explicitly caught for retry
                fmt::print("system-error: allocation failed (VERY RARE) '{}' - will retry, attempt: {}\n", e.what(), retry_attempt);
            } catch (std::invalid_argument& e) { // explicitly caught for retry
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
        const bool                  _is_mmap_allocated;
        const std::size_t             _size; // pre-condition: std::has_single_bit(_size)
        std::vector<T, Allocator>   _data;
        WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        ClaimType                   _claim_strategy;
        // list of dependent reader indices
        DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _is_mmap_allocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(std::bit_ceil(min_size), _is_mmap_allocated)), _data(buffer_size(_size, _is_mmap_allocated), _allocator), _claim_strategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _is_mmap_allocated) {
            if (_is_mmap_allocated) {
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

        static std::size_t buffer_size(const std::size_t size, bool _is_mmap_allocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return _is_mmap_allocated ? size : 2 * size;
        }
    }; // struct buffer_impl

    template <typename U = T>
    class buffer_writer {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        bool                        _is_mmap_allocated;
        std::size_t                   _size;
        ClaimType*                  _claim_strategy;

    class ReservedOutputRange {
        buffer_writer<U>* _parent = nullptr;
        std::size_t       _index = 0;
        std::size_t       _n_slots_to_claim = 0;
        signed_index_type      _offset = 0;
        bool              _published_data = false;
        std::span<T>      _internal_span{};
    public:
    using element_type = T;
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<T>::iterator;
    using reverse_iterator = typename std::span<T>::reverse_iterator;
    using pointer = typename std::span<T>::reverse_iterator;

    explicit ReservedOutputRange(buffer_writer<U>* parent) noexcept : _parent(parent) {};
    explicit constexpr ReservedOutputRange(buffer_writer<U>* parent, std::size_t index, signed_index_type sequence, std::size_t n_slots_to_claim) noexcept :
        _parent(parent), _index(index), _n_slots_to_claim(n_slots_to_claim), _offset(sequence - static_cast<signed_index_type>(n_slots_to_claim)), _internal_span({ &_parent->_buffer->_data.data()[_index], _n_slots_to_claim }) { }
    ReservedOutputRange(const ReservedOutputRange&) = delete;
    ReservedOutputRange& operator=(const ReservedOutputRange&) = delete;
    ReservedOutputRange(ReservedOutputRange&& other) noexcept
        : _parent(std::exchange(other._parent, nullptr))
        , _index(std::exchange(other._index, 0))
        , _n_slots_to_claim(std::exchange(other._n_slots_to_claim, 0))
        , _offset(std::exchange(other._offset, 0))
        , _published_data(std::exchange(other._published_data, 0))
        , _internal_span(std::exchange(other._internal_span, std::span<T>{})) {
    };
    ReservedOutputRange& operator=(ReservedOutputRange&& other) noexcept {
        auto tmp = std::move(other);
        std::swap(_parent, tmp._parent);
        std::swap(_index, tmp._index);
        std::swap(_n_slots_to_claim, tmp._n_slots_to_claim);
        std::swap(_offset, tmp._offset);
        std::swap(_published_data, tmp._published_data);
        std::swap(_internal_span, tmp._internal_span);
        return *this;
    };
    ~ReservedOutputRange() {
        if constexpr (std::is_base_of_v<MultiThreadedStrategy<SIZE, WAIT_STRATEGY>, ClaimType>) {
            if (_n_slots_to_claim) {
                fmt::print(stderr, "circular_buffer::multiple_writer::ReservedOutputRange() - did not publish {} samples\n", _n_slots_to_claim);
                std::abort();
            }

        } else {
            if (_n_slots_to_claim && not _published_data) {
                fmt::print(stderr, "circular_buffer::single_writer::ReservedOutputRange() - omitted publish call for {} reserved samples\n", _n_slots_to_claim);
                std::abort();
            }
        }
    }

    [[nodiscard]] constexpr bool
    is_published() const noexcept {
        return _published_data;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return _n_slots_to_claim; };
    [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return _n_slots_to_claim * sizeof(T); };
    [[nodiscard]] constexpr bool empty() const noexcept { return _n_slots_to_claim == 0; }
    [[nodiscard]] constexpr iterator begin() const noexcept { return _internal_span.begin(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return _internal_span.end(); }
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _internal_span.rbegin(); }
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _internal_span.rend(); }
    [[nodiscard]] constexpr T* data() const noexcept { return _internal_span.data(); }

    T& operator [](std::size_t i) const noexcept  {return _parent->_buffer->_data.data()[_index + i]; }
    T& operator [](std::size_t i) noexcept { return _parent->_buffer->_data.data()[_index + i]; }
    operator std::span<T>&() const noexcept { return _internal_span; }
    operator std::span<T>&() noexcept { return _internal_span; }

    constexpr void publish(std::size_t n_produced) noexcept {
        assert(n_produced <= _n_slots_to_claim && "n_produced must be <= than claimed slots");
        if (!_parent->_is_mmap_allocated) {
            const std::size_t size = _parent->_size;
            // mirror samples below/above the buffer's wrap-around point
            const size_t nFirstHalf = std::min(size - _index, n_produced);
            const size_t nSecondHalf = n_produced - nFirstHalf;

            auto &data = _parent->_buffer->_data;
            std::copy(&data[_index], &data[_index + nFirstHalf], &data[_index + size]);
            std::copy(&data[size], &data[size + nSecondHalf], &data[0]);
        }
        _parent->_claim_strategy->publish(_offset + static_cast<signed_index_type>(n_produced));
        _n_slots_to_claim -= n_produced;
        _published_data = true;
    }
    }; // class ReservedOutputRange

    static_assert(PublishableSpan<ReservedOutputRange>);

    public:
        buffer_writer() = delete;
        explicit buffer_writer(std::shared_ptr<buffer_impl> buffer) noexcept :
            _buffer(std::move(buffer)), _is_mmap_allocated(_buffer->_is_mmap_allocated),
            _size(_buffer->_size), _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer(buffer_writer&& other) noexcept
            : _buffer(std::move(other._buffer))
            , _is_mmap_allocated(_buffer->_is_mmap_allocated)
            , _size(_buffer->_size)
            , _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer& operator=(buffer_writer tmp) noexcept {
            std::swap(_buffer, tmp._buffer);
            _is_mmap_allocated = _buffer->_is_mmap_allocated;
            _size = _buffer->_size;
            _claim_strategy = std::addressof(_buffer->_claim_strategy);

            return *this;
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return CircularBuffer(_buffer); };

        [[nodiscard]] constexpr auto reserve_output_range(std::size_t n_slots_to_claim) noexcept -> ReservedOutputRange {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (static_cast<std::size_t>(sequence) + _size - n_slots_to_claim) % _size;
                return ReservedOutputRange(this, index, sequence, n_slots_to_claim);
            } catch (const NoCapacityException &) {
                return ReservedOutputRange(this);
            }
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

        [[nodiscard]] constexpr signed_index_type position() const noexcept { return _buffer->_cursor.value(); }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return static_cast<std::size_t>(_claim_strategy->getRemainingCapacity(*_buffer->_read_indices));
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

                if (!_is_mmap_allocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claim_strategy->publish(publishSequence); // points at first non-writable index
            } catch (const std::exception&) {
                throw;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    }; // class buffer_writer
    //static_assert(BufferWriter<buffer_writer<T>>);

    template<typename U = T>
    class buffer_reader
    {
        class ConsumableInputRange;
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        std::shared_ptr<Sequence>    _readIndex = std::make_shared<Sequence>();
        mutable signed_index_type    _readIndexCached;
        BufferTypeLocal              _buffer; // controls buffer life-cycle, the rest are cache optimisations
        std::size_t                  _size; // pre-condition: std::has_single_bit(_size)
        // TODO: doesn't have to be atomic because this reader is (/must be) accessed (by design) always by the same thread.
        mutable std::atomic_bool     _isRangeConsumed {true}; // controls if consume() was invoked

        std::size_t
        buffer_index() const noexcept {
            const auto bitmask = _size - 1;
            return static_cast<std::size_t>(_readIndexCached) & bitmask;
        }

        class ConsumableInputRange {
            const buffer_reader<U>* _parent = nullptr;
            std::size_t             _index = 0;
            std::span<const T>      _internalSpan{};

        public:
        using element_type = T;
        using value_type = typename std::remove_cv_t<T>;
        using iterator = typename std::span<const T>::iterator;
        using reverse_iterator = typename std::span<const T>::reverse_iterator;
        using pointer = typename std::span<const T>::reverse_iterator;

        explicit ConsumableInputRange(const buffer_reader<U>* parent) noexcept : _parent(parent) {};
        explicit constexpr ConsumableInputRange(const buffer_reader<U>* parent, std::size_t index, std::size_t nRequested) noexcept :
            _parent(parent), _index(index), _internalSpan({ &_parent->_buffer->_data.data()[_index], nRequested }) { }

        ConsumableInputRange(const ConsumableInputRange& other)
            : _parent(other._parent),
              _index(other._index),
              _internalSpan(other._internalSpan) {
        }

        ConsumableInputRange& operator=(const ConsumableInputRange& other) {
            if (this != &other) {
                _parent = other._parent;
                _index = other._index;
                _internalSpan = other._internalSpan;
            }
            return *this;
        }

        ConsumableInputRange(ConsumableInputRange&& other) noexcept
            : _parent(std::exchange(other._parent, nullptr))
            , _index(std::exchange(other._index, 0))
            , _internalSpan(std::exchange(other._internalSpan, std::span<T>{})) {
        }
        ConsumableInputRange& operator=(ConsumableInputRange&& other) noexcept {
            auto tmp = std::move(other);
            std::swap(_parent, tmp._parent);
            std::swap(_index, tmp._index);
            std::swap(_internalSpan, tmp._internalSpan);
            return *this;
        }
        ~ConsumableInputRange() = default;

        [[nodiscard]] constexpr std::size_t size() const noexcept { return _internalSpan.size(); }
        [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return size() * sizeof(T); }
        [[nodiscard]] constexpr bool empty() const noexcept { return _internalSpan.empty(); }
        [[nodiscard]] constexpr iterator begin() const noexcept { return _internalSpan.begin(); }
        [[nodiscard]] constexpr iterator end() const noexcept { return _internalSpan.end(); }
        [[nodiscard]] constexpr const T& front() const noexcept { return _internalSpan.front(); }
        [[nodiscard]] constexpr const T& back() const noexcept { return _internalSpan.back(); }
        [[nodiscard]] constexpr auto first(std::size_t count) const noexcept { return _internalSpan.first(count); }
        [[nodiscard]] constexpr auto last(std::size_t count) const noexcept { return _internalSpan.last(count); }
        [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _internalSpan.rbegin(); }
        [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _internalSpan.rend(); }
        [[nodiscard]] constexpr const T* data() const noexcept { return _internalSpan.data(); }
        const T& operator [](std::size_t i) const noexcept  {return _parent->_buffer->_data.data()[_index + i]; }
        const T& operator [](std::size_t i) noexcept { return _parent->_buffer->_data.data()[_index + i]; }
        operator const std::span<const T>&() const noexcept { return _internalSpan; }
        operator std::span<const T>&() noexcept { return _internalSpan; }
        operator std::span<const T>&&() = delete;

        template <bool strict_check = true>
        [[nodiscard]] bool consume(std::size_t nSamples) const noexcept {
            if (std::atomic_load_explicit(&_parent->_isRangeConsumed, std::memory_order_acquire)) {
                fmt::println("An error occurred: The method CircularBuffer::buffer_reader::ConsumableInputRange::consume() was invoked for the second time in succession, a corresponding ConsumableInputRange was already consumed.");
                std::abort();
            }
            return tryConsume<strict_check>(nSamples);
        }

        template <bool strict_check = true>
        [[nodiscard]] bool tryConsume(std::size_t nSamples) const noexcept {
            if (std::atomic_load_explicit(&_parent->_isRangeConsumed, std::memory_order_acquire)) {
                return false;
            }
            std::atomic_store_explicit(&_parent->_isRangeConsumed, true, std::memory_order_release);
            _parent->_isRangeConsumed.notify_all();
            if constexpr (strict_check) {
                if (nSamples <= 0) {
                    return true;
                }

                if (nSamples > std::min(_internalSpan.size(), _parent->available())) {
                    return false;
                }
            }
            _parent->_readIndexCached = _parent->_readIndex->addAndGet(static_cast<signed_index_type>(nSamples));
            return true;
        }

        }; // class ConsumableInputRange
        static_assert(ConsumableSpan<ConsumableInputRange>);


    public:
        buffer_reader() = delete;
        explicit buffer_reader(std::shared_ptr<buffer_impl> buffer) noexcept :
            _buffer(buffer), _size(buffer->_size) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_readIndex});
            _readIndexCached = _readIndex->value();
        }
        buffer_reader(buffer_reader&& other) noexcept
            : _readIndex(std::move(other._readIndex))
            , _readIndexCached(std::exchange(other._readIndexCached, _readIndex->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_readIndex, tmp._readIndex);
            std::swap(_readIndexCached, tmp._readIndexCached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _readIndex); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return CircularBuffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr auto get(const std::size_t nRequested = 0UZ) const noexcept -> ConsumableInputRange {
            std::size_t n = nRequested > 0 ? nRequested : available();
            if constexpr (strict_check) {
                n = nRequested > 0 ? std::min(nRequested, available()) : available();
            }
            std::atomic_store_explicit(&_isRangeConsumed, false, std::memory_order_release);
            _isRangeConsumed.notify_all();
            return ConsumableInputRange(this, buffer_index(), n);
        }

        template <bool strict_check = true>
        [[nodiscard]] constexpr bool consume(const std::size_t nSamples = 1) noexcept {
            if constexpr (strict_check) {
                if (nSamples <= 0) {
                    std::atomic_store_explicit(&_isRangeConsumed, true, std::memory_order_release); // TODO: remove consume method
                    _isRangeConsumed.notify_all();
                    return true;
                }
                if (nSamples > available()) {
                    std::atomic_store_explicit(&_isRangeConsumed, true, std::memory_order_release); // TODO: remove consume method
                   _isRangeConsumed.notify_all();
                    return false;
                }
            }
            _readIndexCached = _readIndex->addAndGet(static_cast<signed_index_type>(nSamples));
            std::atomic_store_explicit(&_isRangeConsumed, true, std::memory_order_release); // TODO: remove consume method
            _isRangeConsumed.notify_all();
            return true;
        }

        [[nodiscard]] constexpr signed_index_type position() const noexcept { return _readIndexCached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return static_cast<std::size_t>(_buffer->_cursor.value() - _readIndexCached);
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
    [[nodiscard]] auto n_readers()              { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] const auto &claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] const auto &wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] const auto &cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<CircularBuffer<int32_t>>);
// clang-format on

} // namespace gr
#endif // GNURADIO_CIRCULARBUFFER_HPP
