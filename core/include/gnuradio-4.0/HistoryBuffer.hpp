#ifndef GNURADIO_HISTORYBUFFER_HPP
#define GNURADIO_HISTORYBUFFER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

namespace gr {

/**
 * @brief A simple circular history buffer that supports contiguous ranged access across the wrap-around point.
 *
 * This buffer is similar to the `circular_buffer<T>` and uses a double-mapped memory approach.
 * It is optimised for single-threaded use and does not provide thread-safety or multi-writer/multi-reader support.
 *
 * Example usage:
 * gr::history_buffer<int, 5> buffer;
 * buffer.push_back(1);  // buffer: [1] (size: 1, capacity: 5)
 * buffer.push_back(2);  // buffer: [2, 1] (size: 2, capacity: 5)
 * buffer.push_back(3);  // buffer: [3, 2, 1] (size: 3, capacity: 5)
 * buffer.push_back(4);  // buffer: [4, 3, 2, 1] (size: 4, capacity: 5)
 * buffer.push_back(5);  // buffer: [5, 4, 3, 2, 1] (size: 5, capacity: 5)
 * buffer.push_back(6);  // buffer: [6, 5, 4, 3, 2] (size: 5, capacity: 5)
 *
 * // :
 * buffer[0];     // value: 6 - unchecked access of last/actual sample
 * buffer[1];     // value: 5 - unchecked access of previous sample
 * ...
 * buffer.at(0);  // value: 6 - checked access of last/actual sample
 * buffer.at(1); // value: 5 - checked access of last/actual sample
 * ...
 * buffer.get_span(0, 3);  // span: [6, 5, 4]
 * buffer.get_span(1, 3);  // span: [6, 3]
 * buffer.get_span(0);     // span: [6, 5, 4, 3, 2]
 * buffer.get_span(1);     // span: [5, 4, 3, 2s]
 */
template<typename T, std::size_t N = std::dynamic_extent, typename Allocator = std::allocator<T>>
class HistoryBuffer {
    using signed_index_type = std::make_signed_t<std::size_t>;
    using buffer_type       = typename std::conditional_t<N == std::dynamic_extent, std::vector<T, Allocator>, std::array<T, N * 2>>;

    buffer_type _buffer{};
    std::size_t _capacity = N;
    std::size_t _write_position{0};
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
    constexpr void push_back(const T& value) noexcept {
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
    template<typename Iter>
    constexpr void push_back_bulk(Iter cbegin, Iter cend) noexcept {
        const auto nSamplesToCopy = std::distance(cbegin, cend);
        Iter       optimizedBegin = static_cast<std::size_t>(nSamplesToCopy) > _capacity ? std::prev(cend, static_cast<std::ptrdiff_t>(_capacity)) : cbegin;
        for (auto it = optimizedBegin; it != cend; ++it) {
            push_back(*it);
        }
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest elements beyond the buffer's capacities.
     */
    template<typename Range>
    constexpr void push_back_bulk(const Range& range) noexcept {
        push_back_bulk(range.cbegin(), range.cend());
    }

    /**
     * @brief unchecked accesses of the element at the specified logical index.
     */
    constexpr T& operator[](std::size_t index) noexcept { return _buffer[map_index(index)]; }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept { return _buffer[map_index(index)]; }

    [[nodiscard]] constexpr T& at(std::size_t index) noexcept(false) {
        if (index >= _size) {
            throw std::out_of_range(fmt::format("index {} out of range [0, {})", index, _size));
        }
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr const T& at(std::size_t index) const noexcept(false) {
        if (index >= _size) {
            throw std::out_of_range(fmt::format("index {} out of range [0, {})", index, _size));
        }
        return _buffer[map_index(index)];
    }

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
