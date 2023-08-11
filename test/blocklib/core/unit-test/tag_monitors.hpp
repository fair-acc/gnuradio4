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

template<typename MapType>
void
map_diff_report(const MapType &map1, const MapType &map2, const std::string &name1, const std::string &name2) {
    for (const auto &[key, value] : map1) {
        if (!map2.contains(key)) {
            fmt::print("    key '{}' is present in {} but not in {}\n", key, name1, name2);
        } else if (map2.at(key) != value) {
            fmt::print("    key '{}' has different values ('{}' vs '{}')\n", key, value, map2.at(key));
        }
    }
}

template<typename IterType>
void
mismatch_report(const IterType &mismatchedTag1, const IterType &mismatchedTag2, const IterType &tags1_begin) {
    const auto index = static_cast<size_t>(std::distance(tags1_begin, mismatchedTag1));
    fmt::print("mismatch at index {}", index);
    if (mismatchedTag1->index != mismatchedTag2->index) {
        fmt::print("  - different index: {} vs {}\n", mismatchedTag1->index, mismatchedTag2->index);
    }

    if (mismatchedTag1->map != mismatchedTag2->map) {
        fmt::print("  - different map content:\n");
        map_diff_report(mismatchedTag1->map, mismatchedTag2->map, "the first map", "the second");
        map_diff_report(mismatchedTag2->map, mismatchedTag1->map, "the second map", "the first");
    }
}

bool
equal_tag_lists(const std::vector<tag_t> &tags1, const std::vector<tag_t> &tags2) {
    if (tags1.size() != tags2.size()) {
        fmt::print("vectors have different sizes ({} vs {})\n", tags1.size(), tags2.size());
        return false;
    }

    auto [mismatchedTag1, mismatchedTag2] = std::mismatch(tags1.begin(), tags1.end(), tags2.begin(), std::equal_to<>());
    if (mismatchedTag1 != tags1.end()) {
        mismatch_report(mismatchedTag1, mismatchedTag2, tags1.begin());
        return false;
    }
    return true;
}

template<typename T, ProcessFunction UseProcessOne>
struct TagSource : public node<TagSource<T, UseProcessOne>> {
    OUT<T>             out;
    std::vector<tag_t> tags{};
    std::size_t        next_tag{ 0 };
    std::int64_t       n_samples_max = 1024;
    std::int64_t       n_samples_produced{ 0 };

    constexpr std::make_signed_t<std::size_t>
    available_samples(const TagSource &) noexcept {
        if constexpr (UseProcessOne == ProcessFunction::USE_PROCESS_ONE) {
            // '-1' -> DONE, produced enough samples
            return n_samples_max == n_samples_produced ? -1 : n_samples_max - n_samples_produced;
        } else if constexpr (UseProcessOne == ProcessFunction::USE_PROCESS_BULK) {
            tag_t::signed_index_type nextTagIn = next_tag < tags.size() ? tags[next_tag].index - n_samples_produced : n_samples_max - n_samples_produced;
            return n_samples_produced < n_samples_max ? std::max(1L, nextTagIn) : -1L; // '-1L' -> DONE, produced enough samples
        } else {
            static_assert(fair::meta::always_false<T>, "ProcessFunction-type not handled");
        }
    }

    T
    process_one(std::size_t offset) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_ONE)
    {
        if (next_tag < tags.size() && tags[next_tag].index <= n_samples_produced) {
            print_tag(tags[next_tag], fmt::format("{}::process_one(...)\t publish tag at  {:6}", this->name.value, n_samples_produced));
            tag_t &out_tag = this->output_tags()[0];
            out_tag        = tags[next_tag];
            out_tag.index  = offset;
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
        if (next_tag < tags.size() && tags[next_tag].index <= n_samples_produced) {
            print_tag(tags[next_tag], fmt::format("{}::process_bulk(...{})\t publish tag at  {:6}", this->name, output.size(), n_samples_produced));
            tag_t &out_tag = this->output_tags()[0];
            out_tag        = tags[next_tag];
            out_tag.index  = 0; // indices > 0 write tags in the future ... handle with care
            this->forward_tags();
            next_tag++;
        }

        n_samples_produced += static_cast<std::int64_t>(output.size());
        return n_samples_produced < n_samples_max ? work_return_status_t::OK : work_return_status_t::DONE;
    }
};

template<typename T, ProcessFunction UseProcessOne>
struct TagMonitor : public node<TagMonitor<T, UseProcessOne>> {
    IN<T>              in;
    OUT<T>             out;
    std::vector<tag_t> tags{};
    std::int64_t       n_samples_produced{ 0 };

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
            print_tag(tag, fmt::format("{}::process_bulk(...{}, ...{})\t received tag at {:6}", this->name, input.size(), output.size(), n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }

        n_samples_produced += static_cast<std::int64_t>(input.size());
        std::memcpy(output.data(), input.data(), input.size() * sizeof(T));

        return work_return_status_t::OK;
    }
};

template<typename T, ProcessFunction UseProcessOne>
struct TagSink : public node<TagSink<T, UseProcessOne>> {
    using ClockSourceType = std::chrono::system_clock;
    IN<T>                                    in;
    std::vector<tag_t>                       tags{};
    std::int64_t                             n_samples_produced{ 0 };
    std::chrono::time_point<ClockSourceType> timeFirstSample = ClockSourceType::now();
    std::chrono::time_point<ClockSourceType> timeLastSample  = ClockSourceType::now();

    // template<fair::meta::t_or_simd<T> V>
    constexpr void
    process_one(const T &) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_ONE)
    {
        if (n_samples_produced == 0) {
            timeFirstSample = ClockSourceType::now();
        }
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_one(...1)    \t received tag at {:6}", this->name, n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }
        n_samples_produced++;
        timeLastSample = ClockSourceType::now();
    }

    // template<fair::meta::t_or_simd<T> V>
    constexpr work_return_status_t
    process_bulk(std::span<const T> input) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_BULK)
    {
        if (n_samples_produced == 0) {
            timeFirstSample = ClockSourceType::now();
        }
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("{}::process_bulk(...{})\t received tag at {:6}", this->name, input.size(), n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }

        n_samples_produced += static_cast<std::int64_t>(input.size());
        timeLastSample = ClockSourceType::now();
        return work_return_status_t::OK;
    }

    float
    effective_sample_rate() const {
        const auto total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(timeLastSample - timeFirstSample).count();
        return total_elapsed_time == 0 ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(n_samples_produced) * 1e6f / static_cast<float>(total_elapsed_time);
    }
};

} // namespace fair::graph::tag_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagSource<T, b>), out, n_samples_max);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagMonitor<T, b>), in, out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagSink<T, b>), in);

namespace fair::graph::tag_test {
// the concepts can only work as expected after ENABLE_REFLECTION_FOR_TEMPLATE_FULL
static_assert(HasProcessOneFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessBulkFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasProcessBulkFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>);

static_assert(HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(not HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);

static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>);
} // namespace fair::graph::tag_test

#endif // GRAPH_PROTOTYPE_TAG_MONITORS_HPP
