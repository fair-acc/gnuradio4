#include <boost/ut.hpp>
#include <gnuradio-4.0/basic/TagDispatch.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
[[nodiscard]] std::vector<int> originsOf(const auto& sink) {
    std::vector<int> origins;
    for (const auto& entry : sink._tags) {
        if (entry.map.contains("source")) {
            origins.push_back(entry.map.template value_or<int>("source", -1));
        }
    }
    return origins;
}

[[nodiscard]] std::size_t countKey(const auto& sink, std::string_view key) {
    return static_cast<std::size_t>(std::ranges::count_if(sink._tags, [key](const auto& entry) { return entry.map.contains(key); }));
}
} // namespace

const boost::ut::suite<"TagDispatch"> tagDispatchTests = [] {
    using namespace boost::ut;
    using Dispatch = gr::basic::TagDispatch<float>;

    "dispatch broadcasts selected input tags while sample streams remain separate"_test = [] {
        for (const gr::Size_t selected : {gr::Size_t{0U}, Dispatch::kAllInputs}) {
            gr::testing::GraphFixture fixture;
            auto&                     first  = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{4U}}, {"values", gr::Tensor<float>(gr::data_from, {1.f})}});
            auto&                     second = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{4U}}, {"values", gr::Tensor<float>(gr::data_from, {2.f})}});
            first._tags                      = {{0U, {{"source", 0}}}, {2U, {{"source", 10}}}};
            second._tags                     = {{0U, {{"source", 1}}}, {1U, {{"source", 11}}}, {2U, {{std::string(gr::tag::kTriggerName.shortKey()), std::string("short")}}}, {3U, {{std::string(gr::tag::kTriggerName.key()), std::string("second")}, {"source", 13}}}};
            auto& router                     = fixture.emplace<Dispatch>({{"n_ports", gr::Size_t{2U}}, {"input_selector", selected}});
            auto& firstSink                  = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            auto& secondSink                 = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(fixture.graph.connect(first, "out", router, "in#0").has_value());
            expect(fixture.graph.connect(second, "out", router, "in#1").has_value());
            expect(fixture.graph.connect(router, "out#0", firstSink, "in").has_value());
            expect(fixture.graph.connect(router, "out#1", secondSink, "in").has_value());
            expect(fixture.run().has_value());
            expect(eq(firstSink._samples, std::vector<float>(4UZ, 1.f)));
            expect(eq(secondSink._samples, std::vector<float>(4UZ, 2.f)));
            for (const auto* sink : {&firstSink, &secondSink}) {
                expect(eq(originsOf(*sink), selected == 0U ? std::vector<int>{0, 10} : std::vector<int>{0, 1, 11, 10, 13}));
            }
            const std::size_t triggerTagsOnFirst = selected == 0U ? 0UZ : 1UZ;
            expect(eq(countKey(firstSink, gr::tag::kTriggerName.key()), triggerTagsOnFirst)) << "an unselected input's control tag stays on its own stream";
            expect(eq(countKey(secondSink, gr::tag::kTriggerName.key()), 1UZ));
            expect(eq(countKey(firstSink, gr::tag::kTriggerName.shortKey()), triggerTagsOnFirst)) << "a short canonical key counts as a control tag too";
            expect(eq(countKey(secondSink, gr::tag::kTriggerName.shortKey()), 1UZ));
        }
    };

    "a single port passes its samples and tags"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TagSource<float>>({{"n_samples_max", gr::Size_t{3U}}, {"values", gr::Tensor<float>(gr::data_from, {5.f})}});
        source._tags                     = {{1U, {{"source", 7}}}};
        auto& router                     = fixture.emplace<Dispatch>({{"n_ports", gr::Size_t{1U}}, {"input_selector", gr::Size_t{0U}}});
        auto& sink                       = fixture.emplace<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.graph.connect(source, "out", router, "in#0").has_value());
        expect(fixture.graph.connect(router, "out#0", sink, "in").has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples, std::vector<float>(3UZ, 5.f)));
        expect(eq(originsOf(sink), std::vector<int>{7}));
    };

    "an input_selector naming no input is refused and keeps the previous selection"_test = [] {
        Dispatch      router;
        gr::MsgPortIn errors;
        expect(router.msgOut.connect(errors).has_value());
        router.settings().init();
        std::ignore = router.settings().applyStagedParameters();
        expect(router.settings().set({{"n_ports", gr::Size_t{2U}}, {"input_selector", gr::Size_t{1U}}}).empty());
        std::ignore = router.settings().activateContext();
        std::ignore = router.settings().applyStagedParameters();
        expect(eq(router.in.size(), 2UZ));
        expect(router.settings().set({{"input_selector", gr::Size_t{2U}}}).empty());
        std::ignore = router.settings().activateContext();
        std::ignore = router.settings().applyStagedParameters();
        expect(eq(router.input_selector.value, gr::Size_t{1U}));
        expect(eq(errors.streamReader().available(), 1UZ));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
