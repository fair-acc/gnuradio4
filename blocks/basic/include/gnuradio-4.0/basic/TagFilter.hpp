#ifndef GNURADIO_BASIC_TAG_FILTER_HPP
#define GNURADIO_BASIC_TAG_FILTER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::TagFilter, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct TagFilter : Block<TagFilter<T>, NoTagPropagation> {
    using Description = Doc<"Copies samples and forwards only the tag keys matching key_filter; '*' matches any text, '!' negates, empty passes every key.">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<std::string, "key_filter", Doc<"glob over tag keys, e.g. 'gr:trigger_*' or '!gr:*'; empty: every key">> key_filter;

    GR_MAKE_REFLECTABLE(TagFilter, in, out, key_filter);

    work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t count = std::min(input.size(), output.size());
        std::ranges::copy(std::span<const T>{input.data(), count}, output.begin());
        for (const auto& [index, map] : input.tags()) {
            const std::size_t offset = index < 0 ? 0UZ : static_cast<std::size_t>(index);
            if (offset >= count) {
                break;
            }
            property_map accepted;
            for (const auto& [key, value] : map.get()) {
                if (key_filter.value.empty() || gr::trigger::detail::nameMatches(key_filter.value, std::string_view(key))) {
                    std::ignore = accepted.insert_or_assign(key, value);
                }
            }
            if (!accepted.empty()) {
                output.publishTag(accepted, offset);
            }
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return work::Status::OK;
    }
};

} // namespace gr::basic
#endif // GNURADIO_BASIC_TAG_FILTER_HPP
