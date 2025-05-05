#ifndef GNURADIO_HISTORYBUFFER_HPP
#define GNURADIO_HISTORYBUFFER_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <format>

namespace gr {

/**
 * @brief A single-threaded circular history buffer with a double-mapped array,
 * allowing either newest-at-[0] (`push_back`) or oldest-at-[0] (`push_front`) usage.
 *
 * ### Storage
 * - Dynamic (`N == std::dynamic_extent`): uses `std::vector<T>` (size = 2 * capacity).
 * - Fixed (`N != std::dynamic_extent`): uses `std::array<T, N * 2>`.
 *
 * ### Key Operations
 * - `push_front(const T&)`: Add new item, drop oldest if full, index `[0]` is newest.
 * - `push_back(const T&)`: Add new item, drop oldest if full, index `[0]` is oldest.
 * - `pop_front()`, `pop_back()`: Remove from logical front/back of the ring.
 * - `front()`, `back()`: Returns the first/last item in logical order.
 * - `operator[](i)`, `at(i)`: Unchecked/checked access.
 * - `resize(...)` (dynamic-only): Adjust capacity, preserving existing data.
 * - `get_span(...)`: Obtain a contiguous view across wrap boundaries.
 *
 * ### Examples
 * \code{.cpp}
 * gr::history_buffer<int, 5> hb_newest;   // use push_front -> index[0] is newest
 * hb_newest.push_front(1); // [1]
 * hb_newest.push_front(2); // [2, 1]
 * hb_newest.push_front(3); // [3, 2, 1]
 * // => index[0] == 3, newest item
 *
 * gr::history_buffer<int, 5> hb_oldest;   // use push_back -> index[0] is oldest
 * hb_oldest.push_back(10); // [10]
 * hb_oldest.push_back(20); // [10, 20]
 * hb_oldest.push_back(30); // [10, 20, 30]
 * // => index[0] == 10, oldest item
 *
 * hb_newest.pop_front();  // remove newest => now hb_newest: [2, 1]
 * hb_oldest.pop_front();  // remove oldest => now hb_oldest: [20, 30]
 *
 * auto val = hb_newest.front();  // val == 2
 * auto last = hb_newest.back();  // last == 1
 * \endcode
 *
 * This class is not thread-safe. For concurrency, use external synchronization.
 */
template<typename T, std::size_t N = std::dynamic_extent, typename Allocator = std::allocator<T>>
class HistoryBuffer {
    using signed_index_type = std::make_signed_t<std::size_t>;
    using buffer_type       = typename std::conditional_t<N == std::dynamic_extent, std::vector<T, Allocator>, std::array<T, N * 2>>;

    buffer_type _buffer{};
    std::size_t _capacity = N;
    std::size_t _write_position{0UZ};
    std::size_t _size{0UZ};

    /**
     * @brief maps the logical index to the physical index in the buffer.
     */
    [[nodiscard]] constexpr inline std::size_t map_index(std::size_t index) const noexcept {
        if constexpr (N == std::dynamic_extent) { // runtime checks
            if (std::has_single_bit(_capacity)) { // _capacity is a power of two
                return (_write_position + index) & (_capacity - 1UZ);
            } else { // fallback
                return (_write_position + index) % _capacity;
            }
        } else {                                    // compile-time checks
            if constexpr (std::has_single_bit(N)) { // N is a power of two
                return (_write_position + index) & (N - 1UZ);
            } else { // fallback to modulo operation if not a power of two
                return (_write_position + index) % N;
            }
        }
    }

public:
    using value_type = T;

    constexpr explicit HistoryBuffer() noexcept { static_assert(N != std::dynamic_extent, "need to specify capacity"); }

    constexpr explicit HistoryBuffer(std::size_t capacity) : _buffer(capacity * 2), _capacity(capacity) {
        if (capacity == 0) {
            throw std::out_of_range("capacity is zero");
        }
        static_assert(N == std::dynamic_extent, "incompatible fixed capacity and using capacity argument");
    }

    /**
     * @brief Adds an element to the end expiring the oldest element beyond the buffer's capacities.
     */
    constexpr void push_front(const T& value) noexcept {
        if (_size < _capacity) [[unlikely]] {
            ++_size;
        }
        if (_write_position == 0) [[unlikely]] {
            _write_position = _capacity;
        }
        --_write_position;
        _buffer[_write_position]             = value;
        _buffer[_write_position + _capacity] = value;
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest elements beyond the buffer's capacities.
     */
    template<std::input_iterator Iter>
    constexpr void push_front(Iter begin, Iter end) noexcept {
        const auto nSamplesToCopy = std::distance(begin, end);
        Iter       optimizedBegin = static_cast<std::size_t>(nSamplesToCopy) > _capacity ? std::prev(end, static_cast<std::ptrdiff_t>(_capacity)) : begin;
        for (auto it = optimizedBegin; it != end; ++it) {
            push_front(*it);
        }
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest elements beyond the buffer's capacities.
     */
    template<std::ranges::range Range>
    constexpr void push_front(const Range& range) noexcept {
        push_front(std::ranges::begin(range), std::ranges::end(range));
    }

    /**
     * @brief Adds an element to the front expiring the oldest element beyond the buffer's capacities.
     */
    constexpr void push_back(const T& value) noexcept {
        if (_size == _capacity) {
            if (++_write_position == _capacity) {
                _write_position = 0U;
            }
        } else {
            ++_size;
        }

        std::size_t insertPos = _write_position + (_size - 1);
        if (insertPos >= _capacity) {
            insertPos -= _capacity;
        }

        _buffer[insertPos]             = value;
        _buffer[insertPos + _capacity] = value;
    }

    /**
     * @brief Inserts a contiguous range [first, last) so that the new items become
     *        the "newest" in the ring, with operator[](0) always the oldest.
     */
    template<std::input_iterator Iter>
    constexpr void push_back(Iter first, Iter last) noexcept {
        std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        if (n == 0) {
            return;
        }

        if (n >= _capacity) { // input range is larger than capacity -> keep only trailing bit
            first = last - static_cast<std::ptrdiff_t>(_capacity);
            n     = _capacity;
        }

        const std::size_t needed = _size + n;
        if (needed > _capacity) { // wrap-around detected
            std::size_t discardCount = needed - _capacity;
            _size -= discardCount;

            std::size_t newPos = _write_position + discardCount;
            if (newPos >= _capacity) {
                newPos -= _capacity; // single-step wrap-around
            }
            _write_position = newPos;
        }

        _size += n;

        std::size_t startPos = _write_position + (_size - n);
        if (startPos >= _capacity) {
            startPos -= _capacity; // single-step wrap-around
        }

        const std::ptrdiff_t chunk1 = static_cast<std::ptrdiff_t>(std::min(n, _capacity - startPos));
        std::copy(first, first + chunk1, _buffer.data() + startPos);
        std::copy(first, first + chunk1, _buffer.data() + startPos + _capacity);

        const std::ptrdiff_t chunk2 = static_cast<std::ptrdiff_t>(n) - chunk1;
        if (chunk2 > 0) { // copy the remainder (if needed)
            std::copy(first + chunk1, last, _buffer.data());
            std::copy(first + chunk1, last, _buffer.data() + _capacity);
        }
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest element so that the new items become
     *        the "newest" in the ring, with operator[](0) always the oldest.
     */
    template<std::ranges::range Range>
    constexpr void push_back(const Range& r) noexcept {
        push_back(std::ranges::begin(r), std::ranges::end(r));
    }

    /**
     * @brief Removes the logical front element (i.e. `operator[](0)`) from the buffer, decreasing the size by one.
     *
     * @throws std::out_of_range if the buffer is empty.
     *
     * @note
     * - If you only call `push_front(...)`, `operator[](0)` is the newest element; hence `pop_front()` removes the newest.
     * - If you only call `push_back(...)`, `operator[](0)` is the oldest element; hence `pop_front()` removes the oldest.
     * - Mixing both push modes can lead to non-intuitive behavior for front/back usage.
     */
    constexpr void pop_front() {
        if (empty()) {
            throw std::out_of_range("pop_front() called on empty HistoryBuffer");
        }
        if constexpr (N == std::dynamic_extent) {
            if (++_write_position == _capacity) {
                _write_position = 0U;
            }
        } else {
            if (_write_position == N - 1U) {
                _write_position = 0U;
            } else {
                ++_write_position;
            }
        }
        --_size;
    }

    /**
     * @brief Removes the logical back element (i.e. `operator[](size() - 1)`) from the buffer, decreasing the size by one.
     *
     * @throws std::out_of_range if the buffer is empty.
     *
     * @note
     * - If you only call `push_front(...)`, `operator[](size() - 1)` is the oldest element; hence `pop_back()` removes the oldest.
     * - If you only call `push_back(...)`, `operator[](size() - 1)` is the newest element; hence `pop_back()` removes the newest.
     * - Mixing both push modes can lead to non-intuitive behavior for front/back usage.
     */
    constexpr void pop_back() {
        if (empty()) {
            throw std::out_of_range("pop_back() called on empty HistoryBuffer");
        }
        // The item at [size()-1] is physically "after" the ring’s start—no need to shift _write_position
        --_size;
    }

    constexpr void resize(std::size_t newCapacity)
    requires(N == std::dynamic_extent)
    {
        if (newCapacity == 0) {
            throw std::out_of_range("new capacity is zero");
        }

        std::vector<T, Allocator> newBuf(newCapacity * 2);

        const std::size_t copyCount = std::min(_size, newCapacity);
        const auto        oldFirst  = cbegin();
        std::copy(oldFirst, oldFirst + static_cast<std::ptrdiff_t>(copyCount), newBuf.begin());                                                        // copy first half
        std::copy(newBuf.begin(), newBuf.begin() + static_cast<std::ptrdiff_t>(copyCount), newBuf.begin() + static_cast<std::ptrdiff_t>(newCapacity)); // mirror second half

        // update members
        std::swap(_buffer, newBuf);
        _capacity       = newCapacity;
        _size           = copyCount;
        _write_position = 0UZ; // implementation choice
    }

    /**
     * @brief unchecked accesses of the element at the specified logical index.
     */
    constexpr T& operator[](std::size_t index) noexcept { return _buffer[map_index(index)]; }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept { return _buffer[map_index(index)]; }

    [[nodiscard]] constexpr T& at(std::size_t index) noexcept(false) {
        if (index >= _size) {
            throw std::out_of_range(std::format("index {} out of range [0, {})", index, _size));
        }
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr const T& at(std::size_t index) const noexcept(false) {
        if (index >= _size) {
            throw std::out_of_range(std::format("index {} out of range [0, {})", index, _size));
        }
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr T& front() noexcept { return _buffer[map_index(0)]; }

    [[nodiscard]] constexpr const T& front() const noexcept { return _buffer[map_index(0)]; }

    [[nodiscard]] constexpr T& back() noexcept { return _buffer[map_index(_size - 1)]; }

    [[nodiscard]] constexpr const T& back() const noexcept { return _buffer[map_index(_size - 1)]; }

    [[nodiscard]] constexpr size_t size() const noexcept { return _size; }

    [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0; }

    [[nodiscard]] constexpr size_t capacity() const noexcept { return _capacity; }

    [[nodiscard]] constexpr T* data() noexcept { return _buffer.data(); }

    [[nodiscard]] constexpr const T* data() const noexcept { return _buffer.data(); }

    /**
     * @brief Returns a span of elements with given (optional) length with the last element being the newest
     */
    [[nodiscard]] constexpr std::span<const T> get_span(std::size_t index, std::size_t length = std::dynamic_extent) const {
        length = std::clamp(length, 0LU, std::min(_size - index, length));
        return std::span<const T>(&_buffer[map_index(index)], length);
    }

    [[nodiscard]] auto begin() noexcept { return std::next(_buffer.begin(), static_cast<signed_index_type>(_write_position)); }

    constexpr void reset(T defaultValue = T()) {
        _size           = 0UZ;
        _write_position = 0UZ;
        std::fill(_buffer.begin(), _buffer.end(), defaultValue);
    }

    [[nodiscard]] constexpr auto begin() const noexcept { return std::next(_buffer.begin(), static_cast<signed_index_type>(_write_position)); }

    [[nodiscard]] constexpr auto cbegin() const noexcept { return std::next(_buffer.begin(), static_cast<signed_index_type>(_write_position)); }

    [[nodiscard]] auto end() noexcept { return std::next(_buffer.begin(), static_cast<signed_index_type>(_write_position + _size)); }

    [[nodiscard]] constexpr auto end() const noexcept { return std::next(_buffer.begin(), static_cast<signed_index_type>(_write_position + _size)); }

    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

    [[nodiscard]] auto rbegin() noexcept { return std::make_reverse_iterator(end()); }

    [[nodiscard]] constexpr auto rbegin() const noexcept { return std::make_reverse_iterator(cend()); }

    [[nodiscard]] constexpr auto crbegin() const noexcept { return rbegin(); }

    [[nodiscard]] auto rend() noexcept { return std::make_reverse_iterator(begin()); }

    [[nodiscard]] constexpr auto rend() const noexcept { return std::make_reverse_iterator(cbegin()); }

    [[nodiscard]] constexpr auto crend() const noexcept { return rend(); }
};

} // namespace gr

#endif // GNURADIO_HISTORYBUFFER_HPP
