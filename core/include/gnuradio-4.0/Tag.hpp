#ifndef GNURADIO_TAG_HPP
#define GNURADIO_TAG_HPP

#include <exception> // for std::terminate

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

using property_map = pmt::Value::Map;

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

/**
 * @brief 'Tag' is a metadata structure that can be attached to a stream of data to carry extra information about that data.
 * A tag can describe a specific time, parameter or meta-information (e.g. sampling frequency, gains, ...), provide annotations,
 * or indicate events that blocks may trigger actions in downstream blocks. Tags can be inserted or consumed by blocks at
 * any point in the signal processing flow, allowing for flexible and customisable data processing.
 *
 * Tags contain the index ID of the sending/receiving stream sample <T> they are attached to. Block implementations
 * may choose to chunk the data based on the MIN_SAMPLES/MAX_SAMPLES criteria only, or in addition break-up the stream
 * so that there is only one tag per scheduler iteration. Multiple tags on the same sample shall be merged to one.
 */
struct alignas(kCacheLine) Tag {
    std::size_t  index{0UZ};
    property_map map{};

    GR_MAKE_REFLECTABLE(Tag, index, map);

    bool operator==(const Tag& other) const = default;

    // TODO: do we need the convenience methods below?
    void reset() noexcept {
        index = 0;
        map.clear();
    }

    [[nodiscard]] pmt::Value& at(const std::string& key) { return map.at(convert_string_domain(key)); }

    [[nodiscard]] const pmt::Value& at(const std::string& key) const { return map.at(convert_string_domain(key)); }

    [[nodiscard]] std::optional<std::reference_wrapper<const pmt::Value>> get(const std::string& key) const noexcept {
        try {
            return map.at(convert_string_domain(key));
        } catch (const std::out_of_range& e) {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<std::reference_wrapper<pmt::Value>> get(const std::string& key) noexcept {
        try {
            return map.at(convert_string_domain(key));
        } catch (const std::out_of_range&) {
            return std::nullopt;
        }
    }

    void insert_or_assign(const std::pair<std::string, pmt::Value>& value) { map[convert_string_domain(value.first)] = value.second; }

    void insert_or_assign(const std::string& key, const pmt::Value& value) { map[convert_string_domain(key)] = value; }
};

} // namespace gr

namespace gr {
using meta::fixed_string;

inline void updateMaps(const pmt::Value::Map& src, pmt::Value::Map& dest) {
    for (const auto& [key, value] : src) {
        if (auto nested_map = checked_access_ptr<const pmt::Value::Map, false>{value.get_if<pmt::Value::Map>()}; nested_map != nullptr) {
            // If it's a nested map
            if (auto it = dest.find(key); it != dest.end()) {
                // If the key exists in the destination map
                auto dest_nested_map = checked_access_ptr<pmt::Value::Map, false>{it->second.get_if<pmt::Value::Map>()};
                if (dest_nested_map != nullptr) {
                    // Merge the nested maps recursively
                    updateMaps(*nested_map, *dest_nested_map);
                } else {
                    // Key exists but not a map, replace it
                    dest[key] = value;
                }
            } else {
                // If the key doesn't exist, just insert
                dest.insert({key, value});
            }
        } else {
            // If it's not a nested map, insert/replace the value
            dest[key] = value;
        }
    }
}

constexpr fixed_string GR_TAG_PREFIX = "gr:";

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit = "", fixed_string Description = "">
class DefaultTag {
    constexpr static fixed_string _key = GR_TAG_PREFIX + Key;

public:
    using value_type = PMT_TYPE;

    [[nodiscard]] constexpr const char* key() const noexcept { return std::string_view(_key).data(); }
    [[nodiscard]] constexpr const char* shortKey() const noexcept { return std::string_view(Key).data(); }
    [[nodiscard]] constexpr const char* unit() const noexcept { return std::string_view(Unit).data(); }
    [[nodiscard]] constexpr const char* description() const noexcept { return std::string_view(Description).data(); }

    [[nodiscard]] EM_CONSTEXPR explicit(false) operator std::string() const noexcept { return std::string(_key); }
    [[nodiscard]] EM_CONSTEXPR explicit(false) operator std::pmr::string() const noexcept { return std::pmr::string(_key); }

    template<typename T>
    requires std::is_same_v<value_type, T>
    [[nodiscard]] std::pair<std::string, pmt::Value> operator()(const T& newValue) const noexcept {
        return {std::string(_key), static_cast<pmt::Value>(PMT_TYPE(newValue))};
    }
};

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt, const TOtherString& str) noexcept {
    if ((dt.shortKey() <=> str) == 0) {
        return std::strong_ordering::equal; // shortKeys are equal
    } else {
        return dt.key() <=> str; // compare key()
    }
}

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const TOtherString& str, const DefaultTag<Key, PMT_TYPE, Unit, Description>& dt) noexcept {
    if ((str <=> dt.shortKey()) == std::strong_ordering::equal) {
        return std::strong_ordering::equal; // shortKeys are equal
    } else {
        return str <=> dt.key(); // compare key()
    }
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
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_quantity", std::string, "", "signal quantity"> SIGNAL_QUANTITY;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_unit", std::string, "", "signal's physical SI unit"> SIGNAL_UNIT;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_min", float, "a.u.", "signal physical max. (e.g. DAQ) limit"> SIGNAL_MIN;
inline EM_CONSTEXPR_STATIC DefaultTag<"signal_max", float, "a.u.", "signal physical max. (e.g. DAQ) limit"> SIGNAL_MAX;
inline EM_CONSTEXPR_STATIC DefaultTag<"n_dropped_samples", gr::Size_t, "", "number of dropped samples"> N_DROPPED_SAMPLES;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_name", std::string> TRIGGER_NAME;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_time", uint64_t, "ns", "UTC-based time-stamp"> TRIGGER_TIME;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_offset", float, "s", "sample delay w.r.t. the trigger (e.g.compensating analog group delays)"> TRIGGER_OFFSET;
inline EM_CONSTEXPR_STATIC DefaultTag<"trigger_meta_info", property_map, "", "maps containing additional trigger information"> TRIGGER_META_INFO;
inline EM_CONSTEXPR_STATIC DefaultTag<"local_time", uint64_t, "ns", "UTC-based time-stamp (host)"> LOCAL_TIME; // should be only in 'TRIGGER_META_INFO', used for metering sample vs. time propagation delays
inline EM_CONSTEXPR_STATIC DefaultTag<"context", std::string, "", "multiplexing key to orchestrate node settings/behavioural changes"> CONTEXT;
inline EM_CONSTEXPR_STATIC DefaultTag<"time", std::uint64_t, "", "multiplexing UTC-time in [ns] when ctx should be applied"> CONTEXT_TIME; // TODO: for backward compatibility -> rename to `ctx_time'
inline EM_CONSTEXPR_STATIC DefaultTag<"reset_default", bool, "", "reset block state to stored default"> RESET_DEFAULTS;
inline EM_CONSTEXPR_STATIC DefaultTag<"store_default", bool, "", "store block settings as default"> STORE_DEFAULTS;
inline EM_CONSTEXPR_STATIC DefaultTag<"end_of_stream", bool, "", "end of stream, receiver should change to DONE state"> END_OF_STREAM;

inline constexpr std::array<std::string_view, 16> kDefaultTags = {"sample_rate", "signal_name", "signal_quantity", "signal_unit", "signal_min", "signal_max", "n_dropped_samples", "trigger_name", "trigger_time", "trigger_offset", "trigger_meta_info", "context", "time", "reset_default", "store_default", "end_of_stream"};

} // namespace tag

} // namespace gr

#endif // GNURADIO_TAG_HPP
