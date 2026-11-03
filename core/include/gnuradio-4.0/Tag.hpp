#ifndef GNURADIO_TAG_HPP
#define GNURADIO_TAG_HPP

#include <algorithm>
#include <cstddef>
#include <exception> // for std::terminate
#include <limits>
#include <memory> // for std::construct_at / std::destroy_at / std::uses_allocator
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/formatter/ValueFormatter.hpp>

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
// non-owning counterpart: what a reader of a tag's payload should take, so that binding one does not copy it
using property_map_view = pmt::ValueMapView;
using Value             = pmt::Value;
using ValueView         = pmt::ValueView;
using ValueMap          = pmt::ValueMap;
using ValueMapView      = pmt::ValueMapView;

inline auto convert_string_domain(const std::pmr::string& s) { return std::string(s); }
inline auto convert_string_domain(const std::string& s) { return std::pmr::string(s); }
inline auto convert_string_domain(const std::string_view& s) { return std::pmr::string(s); }

template<typename T, bool notNull = true>
struct checked_access_ptr {
    T* ptr = nullptr;

    checked_access_ptr(T* ptr_) : ptr(ptr_) {
        if (notNull && ptr == nullptr) {
            std::terminate();
        }
    }

    bool operator==(std::nullptr_t) const { return ptr == nullptr; }
    bool operator!=(std::nullptr_t) const { return ptr != nullptr; }
    T&   operator*() const {
        if (notNull && ptr == nullptr) {
            std::terminate();
        }
        return *ptr;
    }
    T* operator->() const {
        if (notNull && ptr == nullptr) {
            std::terminate();
        }
        return ptr;
    }
};

/**
 * @brief 'Tag' is metadata attached to a stream of samples carrying extra information (a specific time, parameters,
 * gains, sampling frequency, annotations, or events that trigger downstream actions). Tags carry the index of the
 * stream sample they are attached to; blocks may insert or consume them anywhere in the flow. Block implementations
 * may chunk on the MIN_SAMPLES/MAX_SAMPLES criteria only, or additionally break the stream so there is at most one
 * tag per scheduler iteration. The tag ring keeps one entry per publish, so several tags on one sample stay separate entries.
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

[[nodiscard]] inline std::span<const std::byte> tagPayloadBlob(const pmt::ValueMapView& tagData) noexcept { return tagData.blob(); }

[[nodiscard]] inline Tag makeStoredTag(std::size_t index, std::span<const std::byte> storedBlob) noexcept { return Tag{index, ValueMapView::viewOf(storedBlob)}; }

// Shallow merge — insert_or_assign per source entry. ValueMap entries are byte-blob-resident,
// so nested-map mutation in the destination requires an explicit extract → mutate → reassign
// pattern; callers needing recursive merge semantics must implement that themselves.
inline void updateMaps(const ValueMap& src, ValueMap& dest) {
    for (const auto& [key, value] : src) {
        dest.insert_or_assign(std::string_view{key}, value);
    }
}

constexpr fixed_string kTagPrefix = "gr:";

/** a well-known tag key with its value type, unit and description; each has one setting alias of the same name in CamelCase (`gr::SampleRate`) */
template<fixed_string Key, typename TPmt, fixed_string Unit = "", fixed_string Description = "">
class DefaultTag {
    constexpr static fixed_string kKey = kTagPrefix + Key;

public:
    using value_type = TPmt;

    [[nodiscard]] constexpr const char* key() const noexcept { return std::string_view(kKey).data(); }
    [[nodiscard]] constexpr const char* shortKey() const noexcept { return std::string_view(Key).data(); }
    [[nodiscard]] constexpr const char* unit() const noexcept { return std::string_view(Unit).data(); }
    [[nodiscard]] constexpr const char* description() const noexcept { return std::string_view(Description).data(); }

    [[nodiscard]] constexpr explicit(false) operator std::string_view() const noexcept { return std::string_view(kKey); }

    template<typename T>
    requires std::is_same_v<value_type, T>
    [[nodiscard]] std::pair<std::string_view, pmt::Value> operator()(const T& newValue) const noexcept {
        return {std::string_view(kKey), static_cast<pmt::Value>(TPmt(newValue))};
    }
};

template<fixed_string Key, typename TPmt, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const DefaultTag<Key, TPmt, Unit, Description>& dt, const TOtherString& str) noexcept {
    return std::string_view(dt) <=> str;
}

template<fixed_string Key, typename TPmt, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr std::strong_ordering operator<=>(const TOtherString& str, const DefaultTag<Key, TPmt, Unit, Description>& dt) noexcept {
    return str <=> std::string_view(dt);
}

template<fixed_string Key, typename TPmt, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr bool operator==(const DefaultTag<Key, TPmt, Unit, Description>& dt, const TOtherString& str) noexcept {
    return (dt <=> std::string_view(str)) == 0;
}

template<fixed_string Key, typename TPmt, fixed_string Unit, fixed_string Description, gr::meta::string_like TOtherString>
constexpr bool operator==(const TOtherString& str, const DefaultTag<Key, TPmt, Unit, Description>& dt) noexcept {
    return (std::string_view(str) <=> dt) == 0;
}

namespace tag { // definition of default tags and names
inline constexpr DefaultTag<"sample_rate", float, "Hz", "sampling rate fₛ: samples per second on this edge">                               kSampleRate;
inline constexpr DefaultTag<"frame_rate", float, "Hz", "frames (items) per second on an edge, where it differs from the sampling rate">    kFrameRate;
inline constexpr DefaultTag<"est_sample_rate", float, "Hz", "measured sampling rate">                                                      kEstSampleRate;
inline constexpr DefaultTag<"est_frame_rate", float, "Hz", "measured frame rate">                                                          kEstFrameRate;
inline constexpr DefaultTag<"signal_name", std::string, "", "signal name">                                                                 kSignalName;
inline constexpr DefaultTag<"num_channels", gr::Size_t, "", "number of interleaved channels">                                              kNumChannels;
inline constexpr DefaultTag<"signal_quantity", std::string, "", "physical quantity of the signal, e.g. voltage (ISO 80000-1)">             kSignalQuantity;
inline constexpr DefaultTag<"signal_unit", std::string, "", "SI unit of the signal, e.g. V">                                               kSignalUnit;
inline constexpr DefaultTag<"signal_min", float, "a.u.", "lower physical limit of the signal, e.g. the DAQ range">                         kSignalMin;
inline constexpr DefaultTag<"signal_max", float, "a.u.", "upper physical limit of the signal, e.g. the DAQ range">                         kSignalMax;
inline constexpr DefaultTag<"n_dropped_samples", gr::Size_t, "", "number of dropped samples">                                              kNDroppedSamples;
inline constexpr DefaultTag<"frequency", double, "Hz", "signal or band centre frequency">                                                  kFrequency;
inline constexpr DefaultTag<"bandwidth", double, "Hz", "occupied bandwidth of the signal">                                                 kBandwidth;
inline constexpr DefaultTag<"frequency_min", double, "Hz", "lower edge of the occupied band">                                              kFrequencyMin;
inline constexpr DefaultTag<"frequency_max", double, "Hz", "upper edge of the occupied band">                                              kFrequencyMax;
inline constexpr DefaultTag<"rx_overflow", bool, "", "receiver overflow indicator">                                                        kRxOverflow;
inline constexpr DefaultTag<"trigger_name", std::string, "", "trigger name">                                                               kTriggerName;
inline constexpr DefaultTag<"trigger_time", uint64_t, "ns", "UTC time-stamp of the trigger">                                               kTriggerTime;
inline constexpr DefaultTag<"trigger_time_error", uint64_t, "ns", "1σ uncertainty of trigger_time">                                        kTriggerTimeError;
inline constexpr DefaultTag<"trigger_offset", float, "s", "sample delay relative to the trigger, e.g. compensating analogue group delays"> kTriggerOffset;
inline constexpr DefaultTag<"trigger_meta_info", property_map, "", "map with additional trigger information">                              kTriggerMetaInfo;
inline constexpr DefaultTag<"trigger_stop_time", uint64_t, "ns", "absolute UTC end of the window">                                         kTriggerStopTime;
inline constexpr DefaultTag<"trigger_pre", uint64_t, "ns", "window extension before trigger_time">                                         kTriggerPre;
inline constexpr DefaultTag<"trigger_post", uint64_t, "ns", "window extension after trigger_stop_time">                                    kTriggerPost;
inline constexpr DefaultTag<"acquisition_flags", property_map, "", "flag name → bool, inside trigger_meta_info; empty = clean">            kAcquisitionFlags;
inline constexpr DefaultTag<"user_data", property_map, "", "opaque user-provided metadata, not interpreted as settings">                   kUserData;
inline constexpr DefaultTag<"local_time", uint64_t, "ns", "host UTC time-stamp inside trigger_meta_info, for metering propagation delays"> kLocalTime;
inline constexpr DefaultTag<"context", std::string, "", "multiplexing key selecting a settings context">                                   kContext;
inline constexpr DefaultTag<"ctx_time", std::uint64_t, "ns", "UTC time at which the context applies">                                      kContextTime;
inline constexpr DefaultTag<"reset_default", bool, "", "reset block state to the stored default">                                          kResetDefaults;
inline constexpr DefaultTag<"store_default", bool, "", "store block settings as the default">                                              kStoreDefaults;
inline constexpr DefaultTag<"end_of_stream", bool, "", "end of stream: the receiver changes to DONE">                                      kEndOfStream;
inline constexpr DefaultTag<"axis_origin", Tensor<double>, "", "per tensor dimension: the coordinate of cell 0, in its axis_unit">         kAxisOrigin;
inline constexpr DefaultTag<"axis_step", Tensor<double>, "", "per tensor dimension: the coordinate step from cell to cell">                kAxisStep;
inline constexpr DefaultTag<"axis_periodic", Tensor<bool>, "", "per tensor dimension: its two ends are neighbours (a full turn)">          kAxisPeriodic;
inline constexpr DefaultTag<"axis_unit", Tensor<pmt::Value>, "", "per tensor dimension: the unit of its coordinates, e.g. rad or m">       kAxisUnit;
inline constexpr DefaultTag<"axis_name", Tensor<pmt::Value>, "", "per tensor dimension: what it spans, e.g. azimuth or x">                 kAxisName;

[[nodiscard]] constexpr std::string_view settingsKey(std::string_view key) noexcept {
    constexpr std::string_view kPrefix = kTagPrefix;
    if (key.starts_with(kPrefix)) {
        return key.substr(kPrefix.size());
    }
    return key;
}

inline constexpr auto kDefaultTagList = std::make_tuple(kSampleRate, kFrameRate, kFrequency, kBandwidth, kFrequencyMin, kFrequencyMax, kEstSampleRate, kEstFrameRate, kSignalName, kNumChannels, kSignalQuantity, kSignalUnit, kSignalMin, kSignalMax, kNDroppedSamples, kRxOverflow, kTriggerName, kTriggerTime, kTriggerTimeError, kTriggerOffset, kTriggerMetaInfo, kTriggerStopTime, kTriggerPre, kTriggerPost, kAcquisitionFlags, kUserData, kContext, kContextTime, kLocalTime, kResetDefaults, kStoreDefaults, kEndOfStream, kAxisOrigin, kAxisStep, kAxisPeriodic, kAxisUnit, kAxisName);
inline constexpr auto kDefaultTags    = std::apply([](const auto&... tags) { return std::array<std::string_view, sizeof...(tags)>{tags.shortKey()...}; }, kDefaultTagList);
static_assert(std::apply(
                  [](const auto&... tags) {
                      const auto matchesCanonical = []<typename TTag>(const TTag& tag) {
                          using T                  = typename TTag::value_type;
                          constexpr auto valueType = std::is_same_v<T, std::string> ? pmt::Value::ValueType::String : (pmt::Value::get_value_type<T>() == pmt::Value::ValueType::Monostate ? pmt::Value::ValueType::Value : pmt::Value::get_value_type<T>());
                          const auto     it        = std::ranges::find(pmt::keys::kCanonical, std::string_view(tag), &pmt::keys::CanonicalKey::name);
                          return it != pmt::keys::kCanonical.end() && it->unit == std::string_view(tag.unit()) && it->boundType == valueType;
                      };
                      return (matchesCanonical(tags) && ...);
                  },
                  kDefaultTagList),
    "every DefaultTag has a canonical key of the same unit and wire type");
static_assert(std::ranges::is_permutation(kDefaultTags, pmt::keys::kCanonical | std::views::transform([](const pmt::keys::CanonicalKey& key) { return settingsKey(key.name); })), "every standard key has a compact id");

template<typename T>
inline void put(property_map& map, std::string_view key, T&& value) {
    auto* res = map.resource();
    map.insert_or_assign(key, pmt::Value(std::forward<T>(value), res));
}

template<typename T, fixed_string Key, typename TPmt, fixed_string Unit, fixed_string Description>
inline void put(property_map& map, const DefaultTag<Key, TPmt, Unit, Description>& /*tag*/, T&& value) {
    constexpr fixed_string kKey = kTagPrefix + Key;
    auto*                  res  = map.resource();
    map.insert_or_assign(kKey.view(), pmt::Value(std::forward<T>(value), res));
}

} // namespace tag

/** the setting of a block that carries a DefaultTag: its type, unit and documentation are the tag's, so each is defined once; `gr::SampleRate sample_rate;` */
template<typename TTag, typename... TExtra>
struct SettingOf;

template<fixed_string Key, typename T, fixed_string UnitText, fixed_string DescriptionText, typename... TExtra>
struct SettingOf<const DefaultTag<Key, T, UnitText, DescriptionText>, TExtra...> {
    using type = Annotated<std::conditional_t<std::is_same_v<T, std::string>, std::pmr::string, T>, Key, Unit<UnitText>, Doc<DescriptionText>, TExtra...>;
};

namespace tag {
template<typename T>
[[nodiscard]] consteval bool hasCanonicalSettingType(std::string_view name) {
    return std::apply([name](const auto&... tags) { return ((name != tags.shortKey() || std::is_same_v<T, typename SettingOf<const std::remove_cvref_t<decltype(tags)>>::type::value_type> || std::is_same_v<T, std::vector<typename SettingOf<const std::remove_cvref_t<decltype(tags)>>::type::value_type>>) && ...); }, kDefaultTagList);
}
} // namespace tag

using SampleRate      = SettingOf<decltype(tag::kSampleRate), Visible, Limits<0.f, std::numeric_limits<float>::max()>>::type;
using FrameRate       = SettingOf<decltype(tag::kFrameRate), Visible, Limits<0.f, std::numeric_limits<float>::max()>>::type;
using EstSampleRate   = SettingOf<decltype(tag::kEstSampleRate), Limits<0.f, std::numeric_limits<float>::max()>>::type;
using SignalName      = SettingOf<decltype(tag::kSignalName)>::type;
using NumChannels     = SettingOf<decltype(tag::kNumChannels)>::type;
using SignalQuantity  = SettingOf<decltype(tag::kSignalQuantity)>::type;
using SignalUnit      = SettingOf<decltype(tag::kSignalUnit)>::type;
using SignalMin       = SettingOf<decltype(tag::kSignalMin)>::type;
using SignalMax       = SettingOf<decltype(tag::kSignalMax)>::type;
using Frequency       = SettingOf<decltype(tag::kFrequency), Visible>::type;
using TriggerName     = SettingOf<decltype(tag::kTriggerName)>::type;
using TriggerTime     = SettingOf<decltype(tag::kTriggerTime)>::type;
using TriggerOffset   = SettingOf<decltype(tag::kTriggerOffset)>::type;
using TriggerMetaInfo = SettingOf<decltype(tag::kTriggerMetaInfo)>::type;
using TriggerPre      = SettingOf<decltype(tag::kTriggerPre)>::type;
using TriggerPost     = SettingOf<decltype(tag::kTriggerPost)>::type;
using Context         = SettingOf<decltype(tag::kContext)>::type;
using AxisUnit        = SettingOf<decltype(tag::kAxisUnit)>::type;
using AxisName        = SettingOf<decltype(tag::kAxisName)>::type;

} // namespace gr

#endif // GNURADIO_TAG_HPP
