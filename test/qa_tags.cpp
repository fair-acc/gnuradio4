#include <boost/ut.hpp>

#include <buffer.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>
#include <tag.hpp>

#include <fmt/format.h>

#include "blocklib/core/unit-test/tag_monitors.hpp"

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
        static_assert(refl::member_list<tag_t>::size == 2, "index and map being declared");
        static_assert(refl::trait::get_t<0, refl::member_list<tag_t>>::name == "index", "class field index is public API");
        static_assert(refl::trait::get_t<1, refl::member_list<tag_t>>::name == "map", "class field map is public API");
    };

    "DefaultTags"_test = [] {
        tag_t testTag{};

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

const boost::ut::suite TagPropagation = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::tag_test;

    "tag_source"_test = [] {
        std::uint64_t n_samples = 1024;
        graph         flow_graph;
        auto         &src = flow_graph.make_node<TagSource<float>>({ { "n_samples_max", n_samples }, { "name", "TagSource" } });
        src.tags          = { // TODO: allow parameter settings to include maps?!?
            { 0, { { "key", "value@0" } } },
            { 100, { { "key", "value@100" } } },
            { 150, { { "key", "value@150" } } },
            { 1000, { { "key", "value@1000" } } }
        };
        src.set_name("TagSource"); // TODO: enable property_map to base-class parameter propagation
        auto &monitor1 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagMonitor1" } });
        monitor1.set_name("TagMonitor1");
        auto &monitor2 = flow_graph.make_node<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagMonitor2" } });
        monitor2.set_name("TagMonitor2");
        auto &sink1 = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({ { "name", "TagSink1" } });
        sink1.set_name("TagSink1");
        auto &sink2 = flow_graph.make_node<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({ { "name", "TagSink2" } });
        sink2.set_name("TagSink2");
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(monitor1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor1).to<"in">(monitor2)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor2).to<"in">(sink1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(monitor2).to<"in">(sink2)));

        scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        expect(eq(src.n_samples_produced, n_samples)) << "src did not produce enough output samples";
        expect(eq(monitor1.n_samples_produced, n_samples)) << "monitor1 did not consume enough input samples";
        expect(eq(monitor2.n_samples_produced, n_samples)) << "monitor2 did not consume enough input samples";
        expect(eq(sink1.n_samples_produced, n_samples)) << "sink1 did not consume enough input samples";
        expect(eq(sink2.n_samples_produced, n_samples)) << "sink2 did not consume enough input samples";

        auto equal_tag_lists = [](const std::vector<tag_t> &tags1, const std::vector<tag_t> &tags2) {
            if (tags1.size() != tags2.size()) {
                fmt::print("vectors have different sizes ({} vs {})\n", tags1.size(), tags2.size());
                return false;
            }

            if (const auto [mismatchedTag1, mismatchedTag2] = std::mismatch(tags1.begin(), tags1.end(), tags2.begin(), std::equal_to<>()); mismatchedTag1 != tags1.end()) {
                const auto index = static_cast<size_t>(std::distance(tags1.begin(), mismatchedTag1));
                fmt::print("mismatch at index {}", index);
                if (mismatchedTag1->index != mismatchedTag2->index) {
                    fmt::print("  - different index: {} vs {}\n", mismatchedTag1->index, mismatchedTag2->index);
                }
                if (mismatchedTag1->map != mismatchedTag2->map) {
                    fmt::print("  - different map content:\n");
                    for (const auto &[key, value] : mismatchedTag1->map) {
                        if (!mismatchedTag2->map.contains(key)) {
                            fmt::print("    key '{}' is present in the first map but not in the second\n", key);
                        } else if (mismatchedTag2->map.at(key) != value) {
                            fmt::print("    key '{}' has different values ('{}' vs '{}')\n", key, value, mismatchedTag2->map.at(key));
                        }
                    }
                    for (const auto &[key, value] : mismatchedTag2->map) {
                        if (!mismatchedTag1->map.contains(key)) {
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
        expect(equal_tag_lists(src.tags, sink1.tags)) << "sink1 did not receive the required tags";
        expect(equal_tag_lists(src.tags, sink2.tags)) << "sink1 did not receive the required tags";
    };
};

int
main() { /* tests are statically executed */
}
