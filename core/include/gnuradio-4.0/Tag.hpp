#ifndef GNURADIO_TAG_HPP
#define GNURADIO_TAG_HPP

#include <cstddef>
#include <exception> // for std::terminate
#include <memory>    // for std::construct_at / std::destroy_at / std::uses_allocator
#include <span>
#include <type_traits>

#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

#ifdef __EMSCRIPTEN__
// constexpr for cases where emscripten does not yet support constexpr and has to fall back to static const or nothing
#define EM_CONSTEXPR
#define EM_CONSTEXPR_STATIC static const
#else
#define EM_CONSTEXPR        constexpr
#define EM_CONSTEXPR_STATIC constexpr
#endif

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

inline std::pmr::string operator""_spmr(const char* str, std::size_t len) { return std::pmr::string(str, len); }

// `property_map` is the public alias used throughout GR4 for tag and settings dictionaries —
// prefer it over the underlying `gr::pmt::ValueMap` to communicate the intent (tag / setting
// payload) and to keep call sites refactor-stable if the underlying type ever changes again.
// Direct `gr::pmt::ValueMap` references are reserved for the Value/ValueMap unit tests,
// micro-benchmarks, and the std::hash specialisation.
using property_map = pmt::ValueMap;
using Value        = pmt::Value;
using ValueView    = pmt::ValueView;
using ValueMap     = pmt::ValueMap;
using ValueMapView = pmt::ValueMapView;

inline auto convert_string_domain(const std::pmr::string& s) { return std::string(s); }
inline auto convert_string_domain(const std::string& s) { return std::pmr::string(s); }
inline auto convert_string_domain(const std::string_view& s) { return std::pmr::string(s); }

template<typename T, bool not_null = true>
struct checked_access_ptr {
    T* ptr = nullptr;

    checked_access_ptr(T* _ptr) : ptr(_ptr) {
        if (not_null && ptr == nullptr) {
            std::terminate();
        }
    }

    bool operator==(std::nullptr_t) const { return ptr == nullptr; }
    bool operator!=(std::nullptr_t) const { return ptr != nullptr; }
    T&   operator*() const {
        if (not_null && ptr == nullptr) {
            std::terminate();
        }
        return *ptr;
    }
    T* operator->() const {
        if (not_null && ptr == nullptr) {
            std::terminate();
        }
        return ptr;
    }
};

template<typename T>
concept PropertyMapType = std::same_as<std::decay_t<T>, property_map>;

template<typename T>
concept WireMapLike = std::same_as<std::decay_t<T>, property_map> || std::same_as<std::decay_t<T>, ValueMapView>;

/**
 * @brief 'Tag' is metadata attached to a stream of samples carrying extra information (a specific time, parameters,
 * gains, sampling frequency, annotations, or events that trigger downstream actions). Tags carry the index of the
 * stream sample they are attached to; blocks may insert or consume them anywhere in the flow. Block implementations
 * may chunk on the MIN_SAMPLES/MAX_SAMPLES criteria only, or additionally break the stream so there is at most one
 * tag per scheduler iteration. Multiple tags on the same sample are merged into one.
 *
 * `Tag` is a single trivially-copyable, non-owning value: a `std::size_t index` (the stream sample it is attached to)
 * and a non-owning `ValueMapView map` over an externally-owned wire blob — suitable for device/USM by-value transfer
 * and the zero-copy read/merge path. The `map` MUST NOT outlive the blob it views (the tag buffer's pool chunk, or the
 * `property_map` it was built from). Owning callers keep the `std::size_t` index and a `property_map` (payload)
 * separately (e.g. `std::pair<std::size_t, property_map>`) — there is no owning `Tag` type.
 */
struct Tag {
    std::size_t  index{0UZ};
    ValueMapView map{};

    GR_MAKE_REFLECTABLE(Tag, index, map);

    [[nodiscard]] bool operator==(const Tag& other) const { return index == other.index && map == other.map; }
};

static_assert(std::is_trivially_copyable_v<Tag>, "Tag must stay trivially copyable for device/USM by-value transport");
using meta::fixed_string;

template<WireMapLike TPropertyMap>
[[nodiscard]] std::span<const std::byte> tagPayloadBlob(TPropertyMap&& tagData) noexcept {
    return tagData.blob();
}

[[nodiscard]] inline Tag makeStoredTag(std::size_t index, std::span<const std::byte> storedBlob) noexcept { return Tag{index, ValueMap::makeView(storedBlob)}; }

// Shallow merge — insert_or_assign per source entry. ValueMap entries are byte-blob-resident,
// so nested-map mutation in the destination requires an explicit extract → mutate → reassign
// pattern; callers needing recursive merge semantics must implement that themselves.
inline void updateMaps(const ValueMap& src, ValueMap& dest) {
    for (const auto& [key, value] : src) {
        dest.insert_or_assign(std::string_view{key}, value);
    }
}

constexpr fixed_string GR_TAG_PREFIX = "gr:";

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit = "", fixed_string Description = "">
class DefaultTag {
    constexpr static fixed_string _key = GR_TAG_PREFIX + Key;

public:
    using value_type = PMT_TYPE;

    [[nodiscard]] constexpr const char*  key() const noexcept { return std::string_view(_key).data(); }
    [[nodiscard]] constexpr const char* shortKey() const noexcept { return std::string_view(Key).data(); }
    [[nodiscard]] constexpr const char* unit() const noexcept { return std::string_view(Unit).data(); }
    [[nodiscard]] constexpr const char* description() const noexcept { return std::string_view(Description).data(); }

    [[nodiscard]] EM_CONSTEXPR explicit(false) operator std::string_view() const noexcept { return std::string_view(_key); }

    template<typename T>
    requires std::is_same_v<value_type, T>
    [[nodiscard]] std::pair<std::string_view, pmt::Value> operator()(const T& newValue) const noexcept {
        return {std::string_view(_key), static_cast<pmt::Value>(PMT_TYPE(newValue))};
    }
};

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt, const TOtherString& str) noexcept {
    return std::string_view(dt) <=> str;
}

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const TOtherString& str, const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt) noexcept {
    return str <=> std::string_view(dt);
}

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr bool operator==(const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt, const TOtherString& str) noexcept {
    return (dt <=> std::string_view(str)) == 0;
}

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr bool operator==(const TOtherString& str, const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt) noexcept {
    return (std::string_view(str) <=> dt) == 0;
}

namespace tag { // definition of default tags and names
inline EM_CONSTEXPR_STATIC DefaultTag<"sample_rate", float, "Hz", "signal sample rate"> SAMPLE_RATE;
inline EM_CONSTEXPR_STATIC DefaultTag<"sample_rate", float, "Hz", "signal sample rate"> SIGNAL_RATE;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_name", std::string, "", "signal name"> SIGNAL_NAME;
inline EM_CONSTEXPR_STATIC DefaultTag<"num_channels", gr::Size_t, "", "interleaved channel count"> NUM_CHANNELS;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_quantity", std::string, "", "signal quantity"> SIGNAL_QUANTITY;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_unit", std::string, "", "signal's physical SI unit"> SIGNAL_UNIT;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_min", float, "a.u.", "signal physical max. (e.g. DAQ) limit"> SIGNAL_MIN;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_max", float, "a.u.", "signal physical max. (e.g. DAQ) limit"> SIGNAL_MAX;
inline EM_CONSTEXPR_STATIC DefaultTag<"n_dropped_samples", gr::Size_t, "", "number of dropped samples"> N_DROPPED_SAMPLES;
inline EM_CONSTEXPR_STATIC DefaultTag<"frequency", double, "Hz", "signal center frequency"> FREQUENCY;
inline EM_CONSTEXPR_STATIC DefaultTag<"rx_overflow", bool, "", "RX overflow indicator"> RX_OVERFLOW;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_name", std::string> TRIGGER_NAME;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_time", uint64_t, "ns", "UTC-based time-stamp"> TRIGGER_TIME;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_offset", float, "s", "sample delay w.r.t. the trigger (e.g.compensating analog group delays)"> TRIGGER_OFFSET;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_meta_info", property_map, "", "maps containing additional trigger information"> TRIGGER_META_INFO;
inline EM_CONSTEXPR_STATIC DefaultTag<"user_data", property_map, "", "opaque user-provided metadata, not interpreted as settings"> USER_DATA;
inline EM_CONSTEXPR_STATIC DefaultTag<"local_time", uint64_t, "ns", "UTC-based time-stamp (host)"> LOCAL_TIME; // should be only in 'TRIGGER_META_INFO', used for metering sample vs. time propagation delays
inline EM_CONSTEXPR_STATIC DefaultTag<"context", std::string, "", "multiplexing key to orchestrate node settings/behavioural changes"> CONTEXT;
inline EM_CONSTEXPR_STATIC DefaultTag<"ctx_time", std::uint64_t, "", "multiplexing UTC-time in [ns] when ctx should be applied"> CONTEXT_TIME;
inline EM_CONSTEXPR_STATIC DefaultTag<"reset_default", bool, "", "reset block state to stored default"> RESET_DEFAULTS;
inline EM_CONSTEXPR_STATIC DefaultTag<"store_default", bool, "", "store block settings as default"> STORE_DEFAULTS;
inline EM_CONSTEXPR_STATIC DefaultTag<"end_of_stream", bool, "", "end of stream, receiver should change to DONE state"> END_OF_STREAM;

[[nodiscard]] constexpr std::string_view settingsKey(std::string_view key) noexcept {
    constexpr std::string_view prefix = GR_TAG_PREFIX;
    if (key.starts_with(prefix)) {
        return key.substr(prefix.size());
    }
    return key;
}

inline constexpr std::array<std::string_view, 21> kDefaultTags = {"sample_rate", "frequency", "signal_name", "num_channels", "signal_quantity", "signal_unit", "signal_min", "signal_max", "n_dropped_samples", "rx_overflow", "trigger_name", "trigger_time", "trigger_offset", "trigger_meta_info", "user_data", "context", "ctx_time", "local_time", "reset_default", "store_default", "end_of_stream"};

template<typename T>
inline void put(property_map& map, std::string_view key, T&& value) {
    auto* res = map.resource();
    map.insert_or_assign(key, pmt::Value(std::forward<T>(value), res));
}

template<typename T, fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description>
inline void put(property_map& map, const DefaultTag<Key, PMT_TYPE, Unit, Description>& /*tag*/, T&& value) {
    auto* res = map.resource();
    map.insert_or_assign(Key.view(), pmt::Value(std::forward<T>(value), res));
}

} // namespace tag

} // namespace gr

#endif // GNURADIO_TAG_HPP
