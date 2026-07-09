#ifndef GNURADIO_DEVICE_SPANS_HPP
#define GNURADIO_DEVICE_SPANS_HPP

#include <cstddef>
#include <span>

namespace gr::device {

/**
 * @brief Spans a kernel body can be handed, in place of the port spans.
 *
 * Pure views over device-resident samples: pointer, size, no accounting. A kernel body cannot cross a launch to
 * reach the host ring, so these satisfy `InputViewLike`/`OutputViewLike` and never `ReaderSpanLike`/
 * `WriterSpanLike`; consuming, publishing and tags stay the framework's job on this path.
 *
 * The tag and accounting members are absent, not present-and-refusing, so a duck-typed body that feature-tests
 * before reading tags silently skips that branch on a device. Constrain such a body to the view concepts instead.
 */
template<typename T>
struct DeviceInputSpan {
    using value_type = T;

    const T*    _data = nullptr;
    std::size_t _size = 0UZ;

    [[nodiscard]] constexpr const T*    begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T*    end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T*    data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr const T&    operator[](std::size_t index) const noexcept { return _data[index]; }
    constexpr                           operator std::span<const T>() const noexcept { return {_data, _size}; }
};

template<typename T>
struct DeviceOutputSpan {
    using value_type = T;

    T*          _data = nullptr;
    std::size_t _size = 0UZ;

    [[nodiscard]] constexpr T*          begin() const noexcept { return _data; }
    [[nodiscard]] constexpr T*          end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr T*          data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr T&          operator[](std::size_t index) const noexcept { return _data[index]; }
    constexpr                           operator std::span<T>() const noexcept { return {_data, _size}; }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_SPANS_HPP
