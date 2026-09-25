#ifndef GNURADIO_TRIGGER_CONDITIONSOURCE_HPP
#define GNURADIO_TRIGGER_CONDITIONSOURCE_HPP

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>

namespace gr::blocks::trigger::detail {

inline void mergeInto(gr::property_map& target, const auto& source) {
    for (const auto& key : source.keys()) {
        if (auto value = source.find_value(key)) {
            target.insert_or_assign(key, gr::pmt::Value(value.value()));
        }
    }
}

struct ConditionSource {
    std::map<std::size_t, gr::property_map> tagsAt;

    [[nodiscard]] const gr::property_map* tagAt(std::size_t offset) const {
        const auto found = tagsAt.find(offset);
        return found == tagsAt.end() ? nullptr : &found->second;
    }

    [[nodiscard]] bool endsStream() const {
        return std::ranges::any_of(tagsAt, [](const auto& entry) {
            const auto& map   = entry.second;
            const auto  found = map.find(static_cast<std::string_view>(gr::tag::END_OF_STREAM));
            return found != map.end() && (*found).second == true;
        });
    }
};

[[nodiscard]] inline ConditionSource collectConditions(auto& inSpan, std::size_t nSamples) {
    ConditionSource conditions;
    for (const auto& tag : inSpan.rawTags()) {
        if (tag.index < inSpan.streamIndex) {
            continue;
        }
        if (const std::size_t offset = tag.index - inSpan.streamIndex; offset < nSamples) {
            mergeInto(conditions.tagsAt[offset], tag.map);
        }
    }
    return conditions;
}

} // namespace gr::blocks::trigger::detail

#endif // GNURADIO_TRIGGER_CONDITIONSOURCE_HPP
