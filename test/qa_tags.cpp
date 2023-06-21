#include <boost/ut.hpp>

#include <buffer.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>
#include <tag.hpp>

#include <fmt/format.h>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;

    "TagReflection"_test = [] {
        static_assert(sizeof(tag_t) % 64 == 0, "needs to meet L1 cache size");
        static_assert(refl::descriptor::type_descriptor<fair::graph::tag_t>::name == "fair::graph::tag_t");
        static_assert(refl::member_list<tag_t>::size == 2, "index and map bein declared");
        static_assert(refl::trait::get_t<0, refl::member_list<tag_t>>::name == "index", "class field index is public API");
        static_assert(refl::trait::get_t<1, refl::member_list<tag_t>>::name == "map", "class field map is public API");
    };

    "DefaultTags"_test = [] {
        tag_t testTag;

        testTag.insert_or_assign(tag::SAMPLE_RATE, pmtv::pmt(3.0f));
        testTag.insert_or_assign(tag::SAMPLE_RATE(4.0f));
        // testTag.insert_or_assign(tag::SAMPLE_RATE(5.0)); // type-mismatch -> won't compile
        expect(testTag.at(tag::SAMPLE_RATE) == 4.0f);
        expect(tag::SAMPLE_RATE.shortKey() == "sample_rate");
        expect(tag::SAMPLE_RATE.key() == std::string{ GR_TAG_PREFIX }.append("sample_rate"));

        expect(testTag.get(tag::SAMPLE_RATE).has_value());
        static_assert(!std::is_const_v<decltype(testTag.get(tag::SAMPLE_RATE).value())>);
        expect(not testTag.get(tag::SIGNAL_NAME).has_value());

        static_assert(std::is_same_v<decltype(tag::SAMPLE_RATE), decltype(tag::SIGNAL_RATE)>);
        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.shortKey() == "signal_unit");
        static_assert(tag::SIGNAL_MIN.shortKey() == "signal_min");
        static_assert(tag::SIGNAL_MAX.shortKey() == "signal_max");
        static_assert(tag::TRIGGER_NAME.shortKey() == "trigger_name");
        static_assert(tag::TRIGGER_TIME.shortKey() == "trigger_time");
        static_assert(tag::TRIGGER_OFFSET.shortKey() == "trigger_offset");
    };
};

namespace fair::graph::tag_test {

void
print_tag(const tag_t &tag, std::string_view prefix = {}) {
    fmt::print("{} @index={}: {{", prefix, tag.index);
    if (tag.map.empty()) {
        fmt::print("}}\n");
        return;
    }
    for (const auto &[key, value] : tag.map) {
        fmt::print(" {:>5}: {} ", key, value);
    }
    fmt::print("}}\n");
}

template<typename T>
struct TagSource : public node<TagSource<T>> {
    OUT<T, 0, 1>       out;
    std::vector<tag_t> tags{ //
                             { 0, { { "key", "value@0" } } },
                             { 100, { { "key", "value@100" } } },
                             { 150, { { "key", "value@150" } } },
                             { 1000, { { "key", "value@1000" } } }
    };
    std::size_t   next_tag{ 0 };
    std::uint64_t n_samples_max = 1024;
    std::uint64_t n_samples_produced{ 0 };

    constexpr std::make_signed_t<std::size_t>
    available_samples(const TagSource &) noexcept {
        const auto ret = static_cast<std::make_signed_t<std::size_t>>(n_samples_max - n_samples_produced);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    T
    process_one() noexcept {
        if (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)) {
            fmt::print("publish sample at {} - {}\n", n_samples_produced, tags[next_tag].index);
            tag_t &out_tag = this->output_tags()[0];
            out_tag        = tags[next_tag];
            this->forward_tags();
            next_tag++;
            n_samples_produced++;
            return static_cast<T>(1);
        }

        n_samples_produced++;
        return static_cast<T>(0);
    }
};

static_assert(HasRequiredProcessFunction<TagSource<int>>);

enum class ProcessFunction {
    USE_PROCESS_ONE  = 0, ///
    USE_PROCESS_BULK = 1  ///
};

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
            print_tag(tag, fmt::format("monitor::process_one received tag at {}", n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }
        n_samples_produced++;
        return input;
    }

    constexpr work_return_t
    process_bulk(std::span<const T> input, std::span<T> output) noexcept
        requires(UseProcessOne == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("monitor::process_bulk received tag at {}", n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }

        n_samples_produced += input.size();
        std::memcpy(output.data(), input.data(), input.size() * sizeof(T));

        return work_return_t::OK;
    }
};

static_assert(HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(not HasProcessOneFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(not HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasProcessBulkFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_ONE>>);
static_assert(HasRequiredProcessFunction<TagMonitor<int, ProcessFunction::USE_PROCESS_BULK>>);

template<typename T>
struct TagSink : public node<TagSink<T>> {
    IN<T>              in;
    std::vector<tag_t> tags{};
    std::uint64_t      n_samples_produced{ 0 };

    // template<fair::meta::t_or_simd<T> V>
    constexpr void
    process_one(const T &) noexcept {
        if (this->input_tags_present()) {
            const tag_t &tag = this->input_tags()[0];
            print_tag(tag, fmt::format("sink received tag at {}", n_samples_produced));
            tags.emplace_back(n_samples_produced, tag.map);
            this->forward_tags();
        }
        n_samples_produced++;
    }
};

static_assert(HasRequiredProcessFunction<TagSink<int>>);

} // namespace fair::graph::tag_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::tag_test::TagSource<T>), out, n_samples_max);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, fair::graph::tag_test::ProcessFunction b), (fair::graph::tag_test::TagMonitor<T, b>), in, out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::tag_test::TagSink<T>), in);

const boost::ut::suite TagPropagation = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::tag_test;

    "tag_source"_test = [] {
        std::uint64_t n_samples = 1024;
        graph         flow_graph;
        auto         &src = flow_graph.make_node<TagSource<float>>({ { "n_samples_max", n_samples } });
        src.set_name("src");
        auto &monitor1 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>();
        monitor1.set_name("monitor1");
        auto &monitor2 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>();
        monitor2.set_name("monitor2");
        auto &sink = flow_graph.make_node<TagSink<float>>();
        sink.set_name("sink");
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(monitor1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor1).to<"in">(monitor2)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor2).to<"in">(sink)));

        scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(monitor1.n_samples_produced, n_samples)) << "monitor1 did not consume enough input samples";
        expect(eq(monitor2.n_samples_produced, n_samples)) << "monitor2 did not consume enough input samples";
        expect(eq(sink.n_samples_produced, n_samples)) << "sink did not consume enough input samples";

        auto equal_tag_lists = [](const std::vector<tag_t> &tags1, const std::vector<tag_t> &tags2) {
            if (tags1.size() != tags2.size()) {
                fmt::print("vectors have different sizes ({} vs {})\n", tags1.size(), tags2.size());
                return false;
            }

            auto result = std::mismatch(tags1.begin(), tags1.end(), tags2.begin(), [](const tag_t &tag1, const tag_t &tag2) { return tag1.index == tag2.index && tag1.map == tag2.map; });

            if (result.first != tags1.end()) {
                size_t       index = std::distance(tags1.begin(), result.first);
                const tag_t &tag1  = *result.first;
                const tag_t &tag2  = *result.second;
                fmt::print("mismatch at index {}\n", index);
                if (tag1.index != tag2.index) {
                    fmt::print("  - different index: {} vs {}\n", tag1.index, tag2.index);
                }
                if (tag1.map != tag2.map) {
                    fmt::print("  - different map content:\n");
                    for (const auto &[key, value] : tag1.map) {
                        if (tag2.map.find(key) == tag2.map.end()) {
                            fmt::print("    key '{}' is present in the first map but not in the second\n", key);
                        } else if (tag2.map.at(key) != value) {
                            fmt::print("    key '{}' has different values ('{}' vs '{}')\n", key, value, tag2.map.at(key));
                        }
                    }
                    for (const auto &[key, value] : tag2.map) {
                        if (tag1.map.find(key) == tag1.map.end()) {
                            fmt::print("    key '{}' is present in the second map but not in the first\n", key);
                        }
                    }
                }
                return false;
            }

            return true;
        };

        expect(equal_tag_lists(src.tags, monitor1.tags)) << "monitor1 did not receive the required tags";
        expect(equal_tag_lists(src.tags, monitor2.tags)) << "monitor2 did not receive the required tags";
        expect(equal_tag_lists(src.tags, sink.tags)) << "sink did not receive the required tags";
    };
};

int
main() { /* tests are statically executed */
}
