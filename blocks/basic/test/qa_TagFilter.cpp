#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/TagFilter.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite<"TagFilter"> tagFilterTests = [] {
    using namespace boost::ut;
    "tag key filters leave every sample unchanged"_test = [] {
        for (const std::string pattern : {"", "keep*", "!drop*", "!*", "!gr:*"}) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{4U}}, {"values", gr::Tensor<float>(gr::data_from, {2.f, 3.f, 5.f, 7.f})}});
            source._tags                     = {{1U, {{"keep_me", 11}, {"drop_me", 22}, {gr::tag::kTriggerTime, std::uint64_t{123U}}}}};
            auto& filter                     = fixture.emplace<gr::basic::TagFilter<float>>({{"key_filter", pattern}});
            auto& sink                       = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(fixture.connect<"out", "in">(source, filter).has_value());
            expect(fixture.connect<"out", "in">(filter, sink).has_value());
            expect(fixture.run().has_value());
            expect(eq(sink._samples, std::vector<float>{2.f, 3.f, 5.f, 7.f}));
            const auto contains = [&](std::string_view key) { return std::ranges::any_of(sink._tags, [&](const auto& entry) { return entry.index == 1U && entry.map.contains(key); }); };
            expect(eq(contains("keep_me"), pattern != "!*")) << pattern;
            expect(eq(contains("drop_me"), pattern.empty() || pattern == "!gr:*")) << pattern;
            expect(eq(contains(gr::tag::kTriggerTime.key()), pattern.empty() || pattern == "!drop*")) << pattern;
        }
    };

    "a key filter longer than a trigger matcher token still selects its keys"_test = [] {
        const std::string         pattern = std::string("keep") + std::string(96UZ, '*');
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{2U}}});
        source._tags                     = {{1U, {{"keep_me", 11}, {"drop_me", 22}}}};
        auto& filter                     = fixture.emplace<gr::basic::TagFilter<float>>({{"key_filter", pattern}});
        auto& sink                       = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, filter).has_value());
        expect(fixture.connect<"out", "in">(filter, sink).has_value());
        expect(fixture.run().has_value());
        expect(std::ranges::any_of(sink._tags, [](const auto& entry) { return entry.map.contains("keep_me") && !entry.map.contains("drop_me"); }));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
