#ifndef GNURADIO_DEVICE_LOG_HPP
#define GNURADIO_DEVICE_LOG_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/AtomicRef.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/ValueMap.hpp>

namespace gr::log {

// wire keys of a device-emitted record, short enough to stay inline in the packed entry
inline constexpr std::string_view                kKeyLevel   = "lvl";
inline constexpr std::string_view                kKeyLine    = "ln";
inline constexpr std::string_view                kKeyFile    = "fl";
inline constexpr std::string_view                kKeyMessage = "msg";
inline constexpr std::array<std::string_view, 6> kKeyArgs    = {"a0", "a1", "a2", "a3", "a4", "a5"};

inline constexpr std::size_t   kDeviceLogMaxArgs       = kKeyArgs.size();
inline constexpr std::uint32_t kDeviceLogEntryCapacity = 12U;   // 4 fixed fields + 6 arguments + sentinel + slack
inline constexpr std::uint32_t kDeviceLogSlotBytes     = 1024U; // multiple of gr::pmt::kBlobAlignment

[[nodiscard]] constexpr std::uint32_t deviceLogPayloadCapacity(std::uint32_t slotBytes, std::uint32_t entryCapacity = kDeviceLogEntryCapacity) noexcept {
    const auto fixed = static_cast<std::uint32_t>(sizeof(gr::pmt::Header)) + entryCapacity * static_cast<std::uint32_t>(sizeof(gr::pmt::PackedEntry));
    return slotBytes > fixed ? slotBytes - fixed : 0U;
}

/**
 * @brief POD handle over host-owned slab storage — the only state a kernel captures.
 *
 * Each slot holds one packed `gr::pmt::ValueMap` record. A kernel claims a slot with a single atomic
 * bump, never allocates and never blocks. The device produces and the host drains only after a
 * synchronisation barrier: cross-device atomics are not coherent, so the two must never write the
 * slab concurrently.
 */
struct DeviceLogSlab {
    std::byte*     slots{};
    std::uint64_t* cursor{};
    std::uint64_t* dropped{};   // lost because the slab was full
    std::uint64_t* truncated{}; // published, but a field did not fit the slot
    std::uint32_t  slotCount{};
    std::uint32_t  slotBytes{};

    [[nodiscard]] constexpr bool valid() const noexcept { return slots != nullptr && cursor != nullptr && dropped != nullptr && truncated != nullptr && slotCount != 0U && slotBytes >= sizeof(gr::pmt::Header); }

    [[nodiscard]] std::uint64_t droppedRecords() const noexcept { return valid() ? gr::atomic_ref(*dropped).load_acquire() : 0ULL; }
    [[nodiscard]] std::uint64_t truncatedRecords() const noexcept { return valid() ? gr::atomic_ref(*truncated).load_acquire() : 0ULL; }

    [[nodiscard]] std::uint32_t published() const noexcept {
        if (!valid()) {
            return 0U;
        }
        const std::uint64_t claimed = gr::atomic_ref(*cursor).load_acquire();
        return claimed < slotCount ? static_cast<std::uint32_t>(claimed) : slotCount;
    }

    [[nodiscard]] std::span<const std::byte> slot(std::uint32_t index) const noexcept { return {slots + static_cast<std::size_t>(index) * slotBytes, slotBytes}; }

    // claim-or-drop: an empty span means the slab is exhausted
    [[nodiscard]] std::span<std::byte> claim() const noexcept {
        if (!valid()) {
            return {};
        }
        const std::uint64_t index = gr::atomic_ref(*cursor).fetch_add(1ULL);
        if (index >= slotCount) {
            gr::atomic_ref(*dropped).fetch_add(1ULL);
            return {};
        }
        return {slots + static_cast<std::size_t>(index) * slotBytes, slotBytes};
    }

    // host-side: invalidate every slot header and rewind the cursor; the drop counters persist
    void recycle() const noexcept {
        if (!valid()) {
            return;
        }
        for (std::uint32_t i = 0U; i < slotCount; ++i) {
            slots[static_cast<std::size_t>(i) * slotBytes] = std::byte{0};
        }
        gr::atomic_ref(*cursor).store_release(0ULL);
    }

    void reset() const noexcept {
        if (!valid()) {
            return;
        }
        recycle();
        gr::atomic_ref(*dropped).store_release(0ULL);
        gr::atomic_ref(*truncated).store_release(0ULL);
    }
};

/// slab storage for targets without an allocator
template<std::size_t slotCount, std::size_t slotBytes = kDeviceLogSlotBytes>
struct StaticDeviceLogSlab {
    static_assert(slotBytes % gr::pmt::kBlobAlignment == 0UZ, "slot size must keep every slot blob-aligned");
    static_assert(slotBytes > sizeof(gr::pmt::Header), "slot must hold a wire header");

    alignas(gr::pmt::kBlobAlignment) std::array<std::byte, slotCount * slotBytes> storage{};
    std::uint64_t cursor{};
    std::uint64_t dropped{};
    std::uint64_t truncated{};

    StaticDeviceLogSlab() noexcept { view().reset(); }

    [[nodiscard]] DeviceLogSlab view() noexcept { return DeviceLogSlab{.slots = storage.data(), .cursor = &cursor, .dropped = &dropped, .truncated = &truncated, .slotCount = static_cast<std::uint32_t>(slotCount), .slotBytes = static_cast<std::uint32_t>(slotBytes)}; }
};

/**
 * @brief compile-time-checked format string that also captures the call site.
 *
 * Uses `__builtin_FILE()`/`__builtin_LINE()` rather than `std::source_location`: both reduce to a
 * string literal and an int, which are addressable in device code.
 */
template<typename... Args>
struct DeviceFormatString {
    std::string_view text{};
    std::string_view file{};
    std::uint32_t    line{};

    template<std::size_t N>
    consteval DeviceFormatString(const char (&format)[N], const char* callerFile = __builtin_FILE(), int callerLine = __builtin_LINE()) : text(format, N - 1UZ), line(static_cast<std::uint32_t>(callerLine)) {
        const std::string_view path{callerFile};
        const std::size_t      slash = path.find_last_of('/');
        file                         = slash == std::string_view::npos ? path : path.substr(slash + 1UZ);

        std::size_t placeholders = 0UZ;
        for (std::size_t i = 0UZ; i + 1UZ < N; ++i) {
            if (format[i] != '{') {
                continue;
            }
            if (format[i + 1UZ] == '{') {
                ++i;
                continue;
            }
            ++placeholders;
        }
        if (placeholders != sizeof...(Args)) {
            throw "gr::log device format string: placeholder count does not match the argument count";
        }
    }
};

namespace detail {

template<typename T>
[[nodiscard]] inline bool writeArg(gr::pmt::ValueMapView& map, std::string_view key, T value) noexcept {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::same_as<U, bool> || std::same_as<U, std::string_view>) {
        return map.try_emplace(key, value);
    } else if constexpr (std::floating_point<U>) {
        return map.try_emplace(key, static_cast<double>(value));
    } else if constexpr (std::signed_integral<U>) {
        return map.try_emplace(key, static_cast<std::int64_t>(value));
    } else if constexpr (std::unsigned_integral<U>) {
        return map.try_emplace(key, static_cast<std::uint64_t>(value));
    } else {
        static_assert(sizeof(U) == 0UZ, "gr::log device arguments must be arithmetic, bool, or std::string_view");
        return false;
    }
}

template<std::size_t... Is, typename... Args>
[[nodiscard]] inline bool writeArgs(gr::pmt::ValueMapView& map, std::index_sequence<Is...>, Args... args) noexcept {
    bool complete = true;
    ((complete = writeArg(map, kKeyArgs[Is], args) && complete), ...);
    return complete;
}

} // namespace detail

/**
 * @brief kernel-facing logger handle with the same call syntax as the host front end.
 *
 * @code
 * struct Kernel {
 *     gr::log::DeviceLogger log;
 *     void operator()(std::size_t i) const { log.warning("processed {} items", i); }
 * };
 * @endcode
 *
 * Arguments must be arithmetic, `bool`, or `std::string_view`; a `const char*` would be a device
 * pointer the host cannot dereference. The text is rendered host-side by `drainDeviceLog()`.
 */
struct DeviceLogger {
    DeviceLogSlab slab{};

    [[nodiscard]] constexpr bool valid() const noexcept { return slab.valid(); }

    template<typename... Args>
    void fatal(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        emit(Level::fatal, fmt, args...);
    }
    template<typename... Args>
    void failure(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        emit(Level::failure, fmt, args...);
    }
    template<typename... Args>
    void error(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        emit(Level::error, fmt, args...);
    }
    template<typename... Args>
    void warning(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        emit(Level::warning, fmt, args...);
    }
    template<typename... Args>
    void info(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        if constexpr (gr::meta::kDebugBuild) {
            emit(Level::info, fmt, args...);
        }
    }
    template<typename... Args>
    void debug(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        if constexpr (gr::meta::kDebugBuild) {
            emit(Level::debug, fmt, args...);
        }
    }
    template<typename... Args>
    void trace(DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) const noexcept {
        if constexpr (gr::meta::kDebugBuild) {
            emit(Level::trace, fmt, args...);
        }
    }

private:
    template<typename... Args>
    void emit(Level level, const DeviceFormatString<std::type_identity_t<Args>...>& fmt, Args... args) const noexcept {
        static_assert(sizeof...(Args) <= kDeviceLogMaxArgs, "gr::log device records carry at most kDeviceLogMaxArgs arguments");

        const std::span<std::byte> slot = slab.claim();
        if (slot.empty()) {
            return; // claim() already counted the drop
        }
        gr::pmt::ValueMapView map = gr::pmt::ValueMapView::formatAt(slot, deviceLogPayloadCapacity(slab.slotBytes), kDeviceLogEntryCapacity);
        if (map._header == nullptr) {
            return;
        }

        bool complete = map.try_emplace(kKeyLevel, static_cast<std::uint64_t>(std::to_underlying(level)));
        complete      = map.try_emplace(kKeyLine, static_cast<std::uint64_t>(fmt.line)) && complete;
        complete      = map.try_emplace(kKeyFile, fmt.file) && complete;
        complete      = map.try_emplace(kKeyMessage, fmt.text) && complete;
        complete      = detail::writeArgs(map, std::index_sequence_for<Args...>{}, args...) && complete;
        if (!complete) {
            gr::atomic_ref(*slab.truncated).fetch_add(1ULL);
        }
    }
};

namespace detail {

[[nodiscard]] inline bool hasBlobMagic(std::span<const std::byte> bytes) noexcept {
    const std::span<const std::byte> magic = std::as_bytes(std::span{gr::pmt::kBlobMagic});
    return bytes.size() >= sizeof(gr::pmt::Header) && std::ranges::equal(bytes.first(magic.size()), magic);
}

[[nodiscard]] inline std::uint64_t readScalar(const gr::pmt::ValueMapView& map, std::string_view key) noexcept {
    const auto iterator = map.find(key);
    if (iterator == map.end()) {
        return 0ULL;
    }
    gr::pmt::ValueView value = iterator->second;
    const auto*        found = value.get_if<std::uint64_t>();
    return found != nullptr ? *found : 0ULL;
}

[[nodiscard]] inline std::string_view readString(const gr::pmt::ValueMapView& map, std::string_view key) noexcept {
    const auto iterator = map.find(key);
    if (iterator == map.end()) {
        return {};
    }
    gr::pmt::ValueView value = iterator->second;
    const auto         found = value.get_if<std::string_view>();
    return found.has_value() ? *found : std::string_view{};
}

inline void appendArg(std::string& out, std::string_view specification, const gr::pmt::ValueMapView& map, std::size_t argIndex) {
    const auto iterator = argIndex < kDeviceLogMaxArgs ? map.find(kKeyArgs[argIndex]) : map.end();
    if (iterator == map.end()) {
        out += "{?}";
        return;
    }

    const auto render = [&out, specification](auto& value) {
        if (specification.empty()) {
            out += std::format("{}", value);
            return;
        }
        const std::string pattern = std::format("{{{}}}", specification);
#if __cpp_exceptions
        try {
            out += std::vformat(pattern, std::make_format_args(value));
        } catch (...) {
            out += std::format("{}", value); // malformed specification: fall back to the default rendering
        }
#else
        out += std::vformat(pattern, std::make_format_args(value));
#endif
    };

    gr::pmt::ValueView value = iterator->second;
    if (auto* asDouble = value.get_if<double>()) {
        render(*asDouble);
    } else if (auto* asSigned = value.get_if<std::int64_t>()) {
        render(*asSigned);
    } else if (auto* asUnsigned = value.get_if<std::uint64_t>()) {
        render(*asUnsigned);
    } else if (auto* asBool = value.get_if<bool>()) {
        render(*asBool);
    } else if (auto asString = value.get_if<std::string_view>()) {
        render(*asString);
    } else {
        out += "{?}";
    }
}

// substitutes the device-emitted arguments into the device-emitted format string
[[nodiscard]] inline std::string renderMessage(const gr::pmt::ValueMapView& map) {
    const std::string_view format = readString(map, kKeyMessage);
    std::string            out;
    out.reserve(format.size() + 32UZ);

    std::size_t argIndex = 0UZ;
    for (std::size_t i = 0UZ; i < format.size(); ++i) {
        const char character = format[i];
        const bool doubled   = i + 1UZ < format.size() && format[i + 1UZ] == character;
        if ((character == '{' || character == '}') && doubled) {
            out += character; // "{{" / "}}" escape
            ++i;
        } else if (character != '{') {
            out += character;
        } else if (const std::size_t close = format.find('}', i + 1UZ); close == std::string_view::npos) {
            out += format.substr(i);
            break;
        } else {
            appendArg(out, format.substr(i + 1UZ, close - i - 1UZ), map, argIndex++);
            i = close;
        }
    }
    return out;
}

template<std::size_t N>
inline bool storeBounded(char (&destination)[N], std::uint16_t& length, std::string_view text) noexcept {
    const std::size_t n = std::min(text.size(), N - 1UZ);
    std::copy_n(text.data(), n, destination);
    destination[n] = '\0';
    length         = static_cast<std::uint16_t>(n);
    return text.size() > n;
}

} // namespace detail

// decodes every published slot, renders it host-side, and publishes it as a canonical LogRecord
[[nodiscard]] inline std::size_t drainDeviceLog(const DeviceLogSlab& slab, Backend& backend) {
    if (!slab.valid()) {
        return 0UZ;
    }

    const std::uint32_t claimed   = slab.published();
    std::size_t         published = 0UZ;
    for (std::uint32_t i = 0U; i < claimed; ++i) {
        const std::span<const std::byte> bytes = slab.slot(i);
        if (!detail::hasBlobMagic(bytes)) {
            continue;
        }
        const gr::pmt::ValueMap map = gr::pmt::ValueMap::makeView(bytes);

        LogRecord record{};
        record.level             = static_cast<Level>(static_cast<std::uint8_t>(detail::readScalar(map, kKeyLevel)));
        record.line              = static_cast<std::uint32_t>(detail::readScalar(map, kKeyLine));
        record.droppedBefore     = slab.droppedRecords();
        record.locationTruncated = detail::storeBounded(record.location, record.locationLength, detail::readString(map, kKeyFile));
        record.textTruncated     = detail::storeBounded(record.text, record.textLength, detail::renderMessage(map));
        if (backend.publish(record)) {
            ++published;
        }
    }
    return published;
}

} // namespace gr::log

#endif // GNURADIO_DEVICE_LOG_HPP
