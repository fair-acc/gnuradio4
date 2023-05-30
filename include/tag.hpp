#ifndef GRAPH_PROTOTYPE_TAG_HPP
#define GRAPH_PROTOTYPE_TAG_HPP

#include <map>
#include <pmtv/pmt.hpp>
#include <reflection.hpp>
#include <utils.hpp>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t hardware_destructive_interference_size  = 64;
inline constexpr std::size_t hardware_constructive_interference_size = 64;
#endif

#ifdef __EMSCRIPTEN__
// constexpr for cases where emscripten does not yet support constexpr and has to fall back to static const or nothing
#define EM_CONSTEXPR
#define EM_CONSTEXPR_STATIC static const
#else
#define EM_CONSTEXPR constexpr
#define EM_CONSTEXPR_STATIC constexpr
#endif

namespace fair::graph {

enum class tag_propagation_policy_t {
    TPP_DONT = 0,       /*!< Scheduler doesn't propagate tags from in- to output. The
                       block itself is free to insert tags. */
    TPP_ALL_TO_ALL = 1, /*!< Propagate tags from all in- to all outputs. The
                       scheduler takes care of that. */
    TPP_ONE_TO_ONE = 2, /*!< Propagate tags from n. input to n. output. Requires
                       same number of in- and outputs */
    TPP_CUSTOM = 3      /*!< Like TPP_DONT, but signals the block it should implement
                       application-specific forwarding behaviour. */
};

/**
 * @brief 'tag_t' is a metadata structure that can be attached to a stream of data to carry extra information about that data.
 * A tag can describe a specific time, parameter or meta-information (e.g. sampling frequency, gains, ...), provide annotations,
 * or indicate events that blocks may trigger actions in downstream blocks. Tags can be inserted or consumed by blocks at
 * any point in the signal processing flow, allowing for flexible and customisable data processing.
 *
 * Tags contain the index ID of the sending/receiving stream sample <T> they are attached to. Node implementations
 * may choose to chunk the data based on the MIN_SAMPLES/MAX_SAMPLES criteria only, or in addition break-up the stream
 * so that there is only one tag per scheduler iteration. Multiple tags on the same sample shall be merged to one.
 */
struct alignas(hardware_constructive_interference_size) tag_t {
    using map_type = std::map<std::string, pmtv::pmt, std::less<>>;
    int64_t  index = 0;
    map_type map;

    // TODO: do we need the convenience methods below?
    [[nodiscard]] pmtv::pmt &
    at(const std::string &key) {
        return map.at(key);
    }

    [[nodiscard]] const pmtv::pmt &
    at(const std::string &key) const {
        return map.at(key);
    }

    [[nodiscard]] std::optional<std::reference_wrapper<const pmtv::pmt>>
    get(const std::string &key) const noexcept {
        try {
            return map.at(key);
        } catch (std::out_of_range &e) {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<std::reference_wrapper<pmtv::pmt>>
    get(const std::string &key) noexcept {
        try {
            return map.at(key);
        } catch (std::out_of_range &) {
            return std::nullopt;
        }
    }

    void
    insert_or_assign(const std::pair<std::string, pmtv::pmt> &value) {
        map[value.first] = value.second;
    }

    void
    insert_or_assign(const std::string &key, const pmtv::pmt &value) {
        map[key] = value;
    }
};
} // namespace fair::graph

ENABLE_REFLECTION(fair::graph::tag_t, index, map);

namespace fair::graph {
using meta::fixed_string;

constexpr fixed_string GR_TAG_PREFIX = "gr:";

template<fixed_string Key, typename PMT_TYPE, fixed_string Unit = "", fixed_string Description = "">
class default_tag {
    constexpr static fixed_string _key = GR_TAG_PREFIX + Key;

public:
    using value_type = PMT_TYPE;

    [[nodiscard]] constexpr std::string_view
    key() const noexcept {
        return std::string_view{ _key };
    }

    [[nodiscard]] constexpr std::string_view
    shortKey() const noexcept {
        return std::string_view(Key);
    }

    [[nodiscard]] constexpr std::string_view
    unit() const noexcept {
        return std::string_view(Unit);
    }

    [[nodiscard]] constexpr std::string_view
    description() const noexcept {
        return std::string_view(Description);
    }

    [[nodiscard]] EM_CONSTEXPR explicit(false) operator std::string() const noexcept { return std::string(_key); }

    template<typename T>
        requires std::is_same_v<value_type, T>
    [[nodiscard]] std::pair<std::string, pmtv::pmt>
    operator()(const T &newValue) const noexcept {
        return { std::string(_key), static_cast<pmtv::pmt>(PMT_TYPE(newValue)) };
    }
};

namespace tag { // definition of default tags and names
inline EM_CONSTEXPR_STATIC default_tag<"sample_rate", float, "Hz", "signal sample rate">                                                       SAMPLE_RATE;
inline EM_CONSTEXPR_STATIC default_tag<"sample_rate", float, "Hz", "signal sample rate">                                                       SIGNAL_RATE;
inline EM_CONSTEXPR_STATIC default_tag<"signal_name", std::string, "", "signal name">                                                          SIGNAL_NAME;
inline EM_CONSTEXPR_STATIC default_tag<"signal_unit", std::string, "", "signal's physical SI unit">                                            SIGNAL_UNIT;
inline EM_CONSTEXPR_STATIC default_tag<"signal_min", float, "a.u.", "signal physical max. (e.g. DAQ) limit">                                   SIGNAL_MIN;
inline EM_CONSTEXPR_STATIC default_tag<"signal_max", float, "a.u.", "signal physical max. (e.g. DAQ) limit">                                   SIGNAL_MAX;
inline EM_CONSTEXPR_STATIC default_tag<"trigger_name", std::string>                                                                            TRIGGER_NAME;
inline EM_CONSTEXPR_STATIC default_tag<"trigger_time", uint64_t, "ns", "UTC-based time-stamp">                                                 TRIGGER_TIME;
inline EM_CONSTEXPR_STATIC default_tag<"trigger_offset", float, "s", "sample delay w.r.t. the trigger (e.g.compensating analog group delays)"> TRIGGER_OFFSET;
inline EM_CONSTEXPR_STATIC default_tag<"context", std::string, "", "multiplexing key to orchestrate node settings/behavioural changes">        CONTEXT;

inline constexpr std::tuple DEFAULT_TAGS = {SAMPLE_RATE, SIGNAL_NAME, SIGNAL_UNIT, SIGNAL_MIN, SIGNAL_MAX, TRIGGER_NAME, TRIGGER_TIME, TRIGGER_OFFSET, CONTEXT};
} // namespace tag

} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_TAG_HPP
