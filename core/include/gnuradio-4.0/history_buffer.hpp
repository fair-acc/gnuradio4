#ifndef GRAPH_PROTOTYPE_HISTORY_BUFFER_HPP
#define GRAPH_PROTOTYPE_HISTORY_BUFFER_HPP

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
 * buffer.push_back(2);  // buffer: [1, 2] (size: 2, capacity: 5)
 * buffer.push_back(3);  // buffer: [1, 2, 3] (size: 3, capacity: 5)
 * buffer.push_back(4);  // buffer: [1, 2, 3, 4] (size: 4, capacity: 5)
 * buffer.push_back(5);  // buffer: [1, 2, 3, 4, 5] (size: 5, capacity: 5)
 * buffer.push_back(6);  // buffer: [2, 3, 4, 5, 6] (size: 5, capacity: 5)
 *
 * // :
 * buffer[0];     // value: 6 - unchecked access of last/actual sample
 * buffer[-1];    // value: 5 - unchecked access of previous sample
 * ...
 * buffer.at(0);  // value: 6 - checked access of last/actual sample
 * buffer.at(-1); // value: 5 - checked access of last/actual sample
 * ...
 * buffer.get_span(0, 3);  // span: [4, 5, 6]
 * buffer.get_span(1, 3);  // span: [3, 4, 5]
 * buffer.get_span(0);     // span: [2, 3, 4, 5, 6]
 * buffer.get_span(1);     // span: [2, 3, 4, 5]
 */
template<typename T, std::size_t N = std::dynamic_extent, typename Allocator = std::allocator<T>>
class history_buffer {
    using signed_index_type = std::make_signed_t<std::size_t>;
    using buffer_type       = typename std::conditional_t<N == std::dynamic_extent, std::vector<T, Allocator>, std::array<T, N * 2>>;
    using size_type         = typename std::conditional_t<N == std::dynamic_extent, std::size_t, std::size_t>;

    buffer_type _buffer;
    std::size_t _capacity = N;
    std::size_t _write_position{ 0 };
    std::size_t _size{ 0 };

    /**
     * @brief maps the logical index to the physical index in the buffer.
     */
    constexpr inline std::size_t
    map_index(signed_index_type index) const noexcept {
        if constexpr (N == std::dynamic_extent) {
            return static_cast<std::size_t>((static_cast<signed_index_type>(_write_position + _capacity - 1) + index)) % _capacity;
        } else {
            return static_cast<std::size_t>((static_cast<signed_index_type>(_write_position + N - 1) + index)) % N;
        }
    }

public:
    constexpr explicit history_buffer() noexcept { static_assert(N != std::dynamic_extent, "need to specify capacity"); }

    constexpr explicit history_buffer(size_t capacity) : _buffer(capacity * 2), _capacity(capacity) {
        if (capacity == 0) {
            throw std::out_of_range("capacity is zero");
        }
        static_assert(N == std::dynamic_extent, "incompatible fixed capacity and using capacity argument");
    }

    /**
     * @brief Adds an element to the end expiring the oldest element beyond the buffer's capacities.
     */
    constexpr void
    push_back(const T &value) noexcept {
        _buffer[_write_position]             = value;
        _buffer[_write_position + _capacity] = _buffer[_write_position];
        ++_write_position;
        [[unlikely]] if (_write_position == _capacity) { _write_position = 0; }
        _size = std::min(_size + 1, _capacity);
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest elements beyond the buffer's capacities.
     */
    template<typename Iter>
    constexpr void
    push_back_bulk(Iter cbegin, Iter cend) noexcept {
        for (auto it = cbegin; it != cend; ++it) {
            push_back(*it);
        }
    }

    /**
     * @brief Adds a range of elements the end expiring the oldest elements beyond the buffer's capacities.
     */
    template<typename Range>
    constexpr void
    push_back_bulk(const Range &range) noexcept {
        for (const auto &item : range) {
            push_back(item);
        }
    }

    /**
     * @brief unchecked accesses of the element at the specified logical index.
     */
    constexpr T &
    operator[](signed_index_type index) noexcept {
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr const T &
    operator[](signed_index_type index) const noexcept {
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr T &
    at(signed_index_type index) {
        const auto signed_size = static_cast<signed_index_type>(_size);
        if (index > 0 || index <= -signed_size) {
            throw std::out_of_range(fmt::format("index {} out of range ]{}, {}]", index, -signed_size, 0));
        }
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr const T &
    at(signed_index_type index) const {
        const auto signed_size = static_cast<signed_index_type>(_size);
        if (index > 0 || index <= -signed_size) {
            throw std::out_of_range(fmt::format("index {} out of range ]{}, {}]", index, -signed_size, 0));
        }
        return _buffer[map_index(index)];
    }

    [[nodiscard]] constexpr size_t
    size() const noexcept {
        return _size;
    }

    [[nodiscard]] constexpr size_t
    capacity() const noexcept {
        return _capacity;
    }

    /**
     * @brief Returns a span of elements with given (optional) length with the last element being the newest
     */
    [[nodiscard]] constexpr std::span<const T>
    get_span(signed_index_type index, std::size_t length = std::dynamic_extent) const /*noexcept*/ {
        length = std::clamp(length, 0LU, std::max(0UL, static_cast<std::size_t>(static_cast<signed_index_type>(size()) + index)));
        return std::span<const T>(cend() + static_cast<typename buffer_type::difference_type>(index - static_cast<signed_index_type>(length)), length);
    }

    [[nodiscard]] auto
    begin() noexcept {
        return _buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity - _size);
    }

    [[nodiscard]] constexpr auto
    begin() const {
        return _buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity - _size);
    }

    [[nodiscard]] constexpr auto
    cbegin() const {
        return _buffer.cbegin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity - _size);
    }

    [[nodiscard]] auto
    end() noexcept {
        return _buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity);
    }

    [[nodiscard]] constexpr auto
    end() const noexcept {
        return _buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity);
    }

    [[nodiscard]] constexpr auto
    cend() const noexcept {
        return _buffer.cbegin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity);
    }

    [[nodiscard]] auto
    rbegin() noexcept {
        return std::make_reverse_iterator(_buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity));
    }

    [[nodiscard]] constexpr auto
    rbegin() const noexcept {
        return std::make_reverse_iterator(_buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity));
    }

    [[nodiscard]] constexpr auto
    crbegin() const noexcept {
        return rbegin();
    }

    [[nodiscard]] auto
    rend() noexcept {
        return std::make_reverse_iterator(_buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity - _size));
    }

    [[nodiscard]] constexpr auto
    rend() const noexcept {
        return std::make_reverse_iterator(_buffer.begin() + static_cast<typename buffer_type::difference_type>(_write_position + _capacity - _size));
    }

    [[nodiscard]] constexpr auto
    crend() const noexcept {
        return rend();
    }
};

} // namespace gr

#endif // GRAPH_PROTOTYPE_HISTORY_BUFFER_HPP
