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

#include "claim_strategy.hpp"
#include "wait_strategy.hpp"
#include "sequence.hpp"
#include "buffer.hpp"

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
        if (size % static_cast<std::size_t>(getpagesize()) != 0LU) {
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

        void* first_copy = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // Map the first half into the now available hole.
        // Note that the second_copy_addr mmap argument is only a hint and mmap might place the
        // mapping somewhere else: "If addr is not NULL, then the kernel takes it as  a hint about
        // where to place the mapping". The returned pointer therefore must equal second_copy_addr
        // for our contiguous mapping to work as intended.
        void* second_copy_addr = static_cast<char*> (first_copy) + size_half;
        if (const void* result = mmap(second_copy_addr, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); result != second_copy_addr) {
            close(shm_fd);
            if (result == MAP_FAILED) {
                throw std::runtime_error(fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
            } else {
                throw std::runtime_error(fmt::format("{} - failed mmap for second copy: mismatching address", buffer_name));
            }
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
        using size_type = std::int32_t;

        Sequence                    _cursor;
        Allocator                   _allocator{};
        const bool                  _is_mmap_allocated;
        const size_type             _size; // pre-condition: std::has_single_bit(_size)
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
        using size_type = typename buffer_impl::size_type;

        BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        bool                        _is_mmap_allocated;
        size_type                   _size;
        ClaimType*                  _claim_strategy;

    class ReservedOutputRange {
        buffer_writer<U>* _parent = nullptr;
        std::size_t       _index = 0;
        std::size_t       _n_slots_to_claim = 0;
        std::int64_t      _offset = 0;
        bool              _published_data = false;
        std::span<T>      _internal_span{};
    public:
    using element_type = T;
    using value_type = typename std::remove_cv_t<T>;
    using iterator = typename std::span<T>::iterator;
    using reverse_iterator = typename std::span<T>::reverse_iterator;
    using pointer = typename std::span<T>::reverse_iterator;

    explicit ReservedOutputRange(buffer_writer<U>* parent) noexcept : _parent(parent) {};
    explicit constexpr ReservedOutputRange(buffer_writer<U>* parent, std::size_t index, std::int64_t sequence, std::size_t n_slots_to_claim) noexcept :
        _parent(parent), _index(index), _n_slots_to_claim(n_slots_to_claim), _offset(sequence - n_slots_to_claim), _internal_span({ &_parent->_buffer->_data[_index], _n_slots_to_claim }) { }
    ReservedOutputRange(const ReservedOutputRange&) = delete;
    ReservedOutputRange& operator=(const ReservedOutputRange&) = delete;
    explicit ReservedOutputRange(ReservedOutputRange&& other) noexcept
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
                fmt::print(stderr, "circular_buffer::multiple_writer::ReservedOutputRange() - did not publish {} samples", _n_slots_to_claim);
                std::terminate();
            }

        } else {
            if (_n_slots_to_claim && not _published_data) {
                fmt::print(stderr, "circular_buffer::single_writer::ReservedOutputRange() - omitted publish call for {} reserved samples", _n_slots_to_claim);
                std::terminate();
            }
        }
    }

    constexpr std::size_t size() const noexcept { return _n_slots_to_claim; };
    constexpr std::size_t size_bytes() const noexcept { return _n_slots_to_claim * sizeof(T); };
    constexpr bool empty() const noexcept { return _n_slots_to_claim == 0; }
    constexpr iterator begin() const noexcept { return _internal_span.begin(); }
    constexpr iterator end() const noexcept { return _internal_span.end(); }
    constexpr reverse_iterator rbegin() const noexcept { return _internal_span.rbegin(); }
    constexpr reverse_iterator rend() const noexcept { return _internal_span.rend(); }
    constexpr T* data() const noexcept { return _internal_span.data(); }

    T& operator [](std::size_t i) const noexcept  {return _parent->_buffer->_data[_index + i]; }
    T& operator [](std::size_t i) noexcept { return _parent->_buffer->_data[_index + i]; }
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
        _parent->_claim_strategy->publish(_offset + n_produced);
        _n_slots_to_claim -= n_produced;
        _published_data = true;
    }
    };

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

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        [[nodiscard]] constexpr auto reserve_output_range(std::size_t n_slots_to_claim) noexcept -> ReservedOutputRange {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (sequence + _size - n_slots_to_claim) % _size;
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

        [[nodiscard]] constexpr std::int64_t position() const noexcept { return _buffer->_cursor.value(); }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _claim_strategy->getRemainingCapacity(*_buffer->_read_indices);
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const std::int64_t publishSequence, const Args&... args) {
            try {
                auto& data = _buffer->_data;
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
            } catch (const std::exception&) {
                throw;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    };

    template<typename U = T>
    class buffer_reader
    {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;
        using size_type = typename buffer_impl::size_type;

        std::shared_ptr<Sequence>   _read_index = std::make_shared<Sequence>();
        std::int64_t                _read_index_cached;
        BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        size_type                   _size; // pre-condition: std::has_single_bit(_size)

        std::size_t
        buffer_index() const noexcept {
            const auto bitmask = _size - 1;
            return static_cast<std::size_t>(_read_index_cached & bitmask);
        }

    public:
        buffer_reader() = delete;
        buffer_reader(std::shared_ptr<buffer_impl> buffer) noexcept :
            _buffer(buffer), _size(buffer->_size) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_read_index});
            _read_index_cached = _read_index->value();
        }
        buffer_reader(buffer_reader&& other) noexcept
            : _read_index(std::move(other._read_index))
            , _read_index_cached(std::exchange(other._read_index_cached, _read_index->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_read_index, tmp._read_index);
            std::swap(_read_index_cached, tmp._read_index_cached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _read_index); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr std::span<const U> get(const std::size_t n_requested = 0) const noexcept {
            const auto& data = _buffer->_data;
            if constexpr (strict_check) {
                const std::size_t n = n_requested > 0 ? std::min(n_requested, available()) : available();
                return { &data[buffer_index()], n };
            }
            const std::size_t n = n_requested > 0 ? n_requested : available();
            return { &data[buffer_index()], n };
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
    explicit circular_buffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    circular_buffer() = delete;
    explicit circular_buffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~circular_buffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] auto n_readers()              { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] const auto &claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] const auto &wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] const auto &cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<circular_buffer<int32_t>>);
// clang-format on

} // namespace gr

#endif // GNURADIO_CIRCULAR_BUFFER_HPP
