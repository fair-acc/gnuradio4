#ifndef GRAPH_PROTOTYPE_TAG_MONITORS_HPP
#define GRAPH_PROTOTYPE_TAG_MONITORS_HPP

#include <node.hpp>
#include <reflection.hpp>
#include <tag.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace fair::graph::tag_test {

enum class ProcessFunction {
    USE_PROCESS_ONE  = 0, ///
    USE_PROCESS_BULK = 1  ///
};

void
print_tag(const tag_t &tag, std::string_view prefix = {}) {
    fmt::print("{} @index= {}: {{", prefix, tag.index);
    if (tag.map.empty()) {
        fmt::print("}}\n");
        return;
    }
    const auto &map = tag.map;
    for (const auto &[key, value] : map) {
        // workaround for:
        // fmt/core.h:1674:10: warning: possibly dangling reference to a temporary [-Wdangling-reference]
        // 1674 |   auto&& arg = arg_mapper<Context>().map(FMT_FORWARD(val));
        //      |          ^~~
        // #pragma GCC diagnostic push
        // #pragma GCC diagnostic ignored "-Wdangling-reference"
        fmt::print(" {:>5}: {} ", key, value);
        // #pragma GCC diagnostic pop
    }
    fmt::print("}}\n");
}

template<typename T, ProcessFunction UseProcessOne>
struct TagSource : public node<TagSource<T, UseProcessOne>> {
    OUT<T>             out;
    std::vector<tag_t> tags{};
    std::size_t        next_tag{ 0 };
    std::uint64_t      n_samples_max = 1024;
    std::uint64_t      n_samples_produced{ 0 };

    constexpr std::make_signed_t<std::size_t>
    available_samples(const TagSource &) noexcept {
        if constexpr (UseProcessOne == ProcessFunction::USE_PROCESS_ONE) {
            return n_samples_produced < n_samples_max ? 1 : -1; // '-1' -> DONE, produced enough samples
        } else if constexpr (UseProcessOne == ProcessFunction::USE_PROCESS_BULK) {
            std::make_signed_t<std::size_t> nextTagIn = next_tag < tags.size() ? tags[next_tag].index - static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)
                                                                               : static_cast<std::make_signed_t<std::size_t>>(n_samples_max - n_samples_produced);
            return n_samples_produced < n_samples_max ? std::max(1L, nextTagIn) : -1L; // '-1L' -> DONE, produced enough samples
        } else {
            static_assert(fair::meta::always_false<T>, "ProcessFunction-type not handled");
        }
    }

    T
    process_one() noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_ONE)
    {
        if (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)) {
            print_tag(tags[next_tag], fmt::format("{}::process_one(...)\t publish tag at  {:6}", this->name.value, n_samples_produced));
            tag_t &out_tag = this->output_tags()[0];
            out_tag        = tags[next_tag];
            out_tag.index  = 0; // indices > 0 write tags in the future ... handle with care
            this->forward_tags();
            next_tag++;
            n_samples_produced++;
            return static_cast<T>(1);
        }

        n_samples_produced++;
        return static_cast<T>(0);
    }

    work_return_status_t
    process_bulk(std::span<T> output) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_BULK)
    {
        if (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)) {
            print_tag(tags[next_tag], fmt::format("{}::process_one(...)\t publish tag at  {:6}", this->name, n_samples_produced));
            tag_t &out_tag = this->output_tags()[0];
            out_tag        = tags[next_tag];
            out_tag.index  = 0; // indices > 0 write tags in the future ... handle with care
            this->forward_tags();
            next_tag++;
        }

        n_samples_produced += output.size();
        return n_samples_produced < n_samples_max ? work_return_status_t::OK : work_return_status_t::DONE;
    }
};

static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);

template<typename T, ProcessFunction UseProcessOne>
struct TagMonitor : public node<TagMonitor<T, UseProcessOne>> {
    IN<T>              in;
    OUT<T>             out;
    std::vector<tag_t> tags{};
    std::uint64_t      n_samples_produced{ 0 };

    constexpr T
    process_one(const T &input) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_one(...)\t received tag at {:6}", this->name, n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }
        n_samples_produced++;
        return input;
    }

    constexpr work_return_status_t
    process_bulk(std::span<const T> input, std::span<T> output) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_bulk(...)\t received tag at {:6}", this->name, n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }

        n_samples_produced += input.size();
        std::memcpy(output.data(), input.data(), input.size() * sizeof(T));

        return work_return_status_t::OK;
    }
};

static_assert(HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(not HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);

template<typename T, ProcessFunction UseProcessOne>
struct TagSink : public node<TagSink<T, UseProcessOne>> {
    IN<T>              in;
    std::vector<tag_t> tags{};
    std::uint64_t      n_samples_produced{ 0 };

    // template<fair::meta::t_or_simd<T> V>
    constexpr void
    process_one(const T &) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_one(...)\t received tag at {:6}", this->name, n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }
        n_samples_produced++;
    }

    // template<fair::meta::t_or_simd<T> V>
    constexpr work_return_status_t
    process_bulk(std::span<const T> input) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_bulk(...)\t received tag at {:6}", this->name, n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }

        n_samples_produced += input.size();

        return work_return_status_t::OK;
    }
};

static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>);

} // namespace fair::graph::tag_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagSource<T, b>), out, n_samples_max);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagMonitor<T, b>), in, out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagSink<T, b>), in);

#endif // GRAPH_PROTOTYPE_TAG_MONITORS_HPP
