#ifndef GNURADIO_AUDIO_COMMON_HPP
#define GNURADIO_AUDIO_COMMON_HPP

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#include <miniaudio.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <concepts>
#include <cstdint>
#include <format>
#include <source_location>
#include <string_view>

namespace gr::audio {

namespace tag {
inline constexpr DefaultTag<"audio_channels", gr::Size_t, "", "interleaved audio channel count"> AUDIO_CHANNELS;
} // namespace tag

namespace detail {

template<typename T>
concept AudioSample = std::same_as<T, float> || std::same_as<T, std::int16_t>;

template<AudioSample T>
[[nodiscard]] constexpr ma_format maFormatFor();

template<>
[[nodiscard]] constexpr ma_format maFormatFor<float>() {
    return ma_format_f32;
}

template<>
[[nodiscard]] constexpr ma_format maFormatFor<std::int16_t>() {
    return ma_format_s16;
}

inline gr::Error makeMiniaudioError(std::string_view operation, ma_result result, std::source_location location = std::source_location::current()) { return gr::Error(std::format("{}: {}", operation, ma_result_description(result)), location); }

} // namespace detail

} // namespace gr::audio

#endif // GNURADIO_AUDIO_COMMON_HPP
