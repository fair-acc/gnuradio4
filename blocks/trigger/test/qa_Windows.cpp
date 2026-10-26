#include <boost/ut.hpp>

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Buffer.hpp>
#include <gnuradio-4.0/trigger/EventReduce.hpp>
#include <gnuradio-4.0/trigger/StreamOps.hpp>
#include <gnuradio-4.0/trigger/TagToEvent.hpp>

#include "Marble.hpp"
#include "TriggerTest.hpp"

namespace qa_buffer {
using namespace gr::blocks::trigger;
using gr::testing::TagSource;
using gr::trigger_test::CollectingSink;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"a", 1.f}};
    return map;
}
[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("cut")}}}, {"L", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("open")}, {std::string(gr::tag::TRIGGER_META_INFO.key()), gr::property_map{{std::string("closing_samples"), std::uint64_t{2}}}}}}};
    return map;
}

[[nodiscard]] std::vector<double> lengthsOf(const std::vector<gr::DataSet<float>>& windows) {
    std::vector<double> found;
    for (const gr::DataSet<float>& window : windows) {
        found.push_back(static_cast<double>(window.signal_values.size()));
    }
    return found;
}

const boost::ut::suite<"BufferCount"> _bufferCount = [] {
    using namespace boost::ut;

    "a window per n samples, each one holding exactly n"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<TagSource<float>>({{"n_samples_max", 9U}, {"values", std::vector<float>{1.f, 2.f, 3.f}}});
        auto&                     windows = fixture.emplace<BufferCount<float>>({{"n_count", 3U}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::Tensor<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._collected.size(), 3UZ)) << "nine samples in windows of three";
        for (const gr::Tensor<float>& window : sink._collected) {
            expect(eq(window.size(), 3UZ)) << "a count-defined window travels as a tensor, and its length is the count";
        }
    };

    "a skip smaller than the count overlaps the windows"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<TagSource<float>>({{"n_samples_max", 9U}, {"values", std::vector<float>{1.f, 2.f, 3.f}}});
        auto&                     windows = fixture.emplace<BufferCount<float>>({{"n_count", 4U}, {"n_skip", 2U}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::Tensor<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(ge(sink._collected.size(), 3UZ)) << "openings at 0, 2, 4 and 6, of which those that fill up are emitted";
        for (const gr::Tensor<float>& window : sink._collected) {
            expect(eq(window.size(), 4UZ));
        }
    };

    "a window of no samples is refused"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     windows = fixture.emplace<BufferCount<float>>({{"n_count", 5U}});
        windows.settings().init();
        std::ignore = windows.settings().applyStagedParameters();
        expect(eq(windows.n_count.value, 5U));

        expect(windows.settings().set({{"n_count", gr::Size_t(0)}}).empty());
        std::ignore = windows.settings().activateContext();
        std::ignore = windows.settings().applyStagedParameters();
        expect(eq(windows.n_count.value, 5U)) << "a window of no samples is not a window";
    };
};

const boost::ut::suite<"BufferTime"> _bufferTime = [] {
    using namespace boost::ut;

    "a duration in seconds becomes a window of samples where the rate is known"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<TagSource<float>>({{"n_samples_max", 12U}, {"values", std::vector<float>{1.f}}});
        auto&                     windows = fixture.emplace<BufferTime<float>>({{"timeout", 0.004f}, {"sample_rate", 1000.f}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._collected.size(), 3UZ)) << "4 ms at 1 kHz is four samples, and twelve samples make three windows";
        if (!sink._collected.empty()) {
            expect(eq(sink._collected[0].signal_values.size(), 4UZ));
            expect(eq(std::string_view{sink._collected[0].axis_names[0]}, std::string_view{"time"})) << "the axis says where in the stream the window sat";
        }
    };

    "with no duration at all nothing is gathered"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<TagSource<float>>({{"n_samples_max", 8U}, {"values", std::vector<float>{1.f}}});
        auto&                     windows = fixture.emplace<BufferTime<float>>({});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._collected.size(), 0UZ)) << "an unset duration is not an infinite one";
    };
};

const boost::ut::suite<"BufferToggle"> _bufferToggle = [] {
    using namespace boost::ut;

    "a trigger opens a window whose length that trigger carries"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a L:a a a a L:a a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     windows = fixture.emplace<BufferToggle<float>>({{"opening_filter", std::string("open")}, {"n_samples", 4U}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(windows.n_openings.value, 2U));
        expect(eq(sink._collected.size(), 2UZ));
        expect(lengthsOf(sink._collected) == std::vector<double>{2., 2.}) << "the event's own closing_samples, not the block's fallback of four";
    };

    "a trigger that says nothing takes the block's duration"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a T:a a a a a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     windows = fixture.emplace<BufferToggle<float>>({{"opening_filter", std::string("cut")}, {"n_samples", 3U}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._collected.size(), 1UZ));
        expect(lengthsOf(sink._collected) == std::vector<double>{3.});
    };
};

const boost::ut::suite<"BufferWhen"> _bufferWhen = [] {
    using namespace boost::ut;

    "the windows tile the stream, one closing where the next begins"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a T:a a a T:a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     windows = fixture.emplace<BufferWhen<float>>({{"filter", std::string("cut")}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(eq(sink._collected.size(), 2UZ)) << "two triggers close two windows; the third is still open when the stream ends";
        expect(lengthsOf(sink._collected) == std::vector<double>{2., 3.}) << "two samples before the first cut, three between the cuts";
    };

    "two triggers on one sample leave an empty window between them"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a T+T:a a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     windows = fixture.emplace<BufferWhen<float>>({{"filter", std::string("cut")}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        expect(ge(sink._collected.size(), 1UZ));
        expect(ge(windows.n_windows.value, 1U));
    };

    "the windows a run produced, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a T:a a a a T:a a a |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     windows = fixture.emplace<BufferWhen<float>>({{"filter", std::string("cut")}, {"sample_rate", 1000.f}});
        auto&                     sink    = fixture.emplace<CollectingSink<gr::DataSet<float>>>();
        expect(fixture.connect<"out", "in">(source, windows).has_value());
        expect(fixture.connect<"out", "in">(windows, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"BufferWhen: a trigger closes a window and opens the next, so they tile the stream"};
        diagram.unit = "sample";
        diagram.row("in").at(2U, "cut").at(6U, "cut").completes();
        diagram.condition(std::format("BufferWhen(filter = \"cut\"), {} windows", windows.n_windows.value));
        std::size_t at = 0UZ;
        for (std::size_t i = 0UZ; i < sink._collected.size(); ++i) {
            auto& row = diagram.row(std::format("out[{}]", i));
            for (std::size_t j = 0UZ; j < sink._collected[i].signal_values.size(); ++j) {
                row.at(at + j, "sample");
            }
            row.completes();
            at += sink._collected[i].signal_values.size();
        }
        diagram.print();

        expect(lengthsOf(sink._collected) == std::vector<double>{2., 4.});
    };
};
} // namespace qa_buffer

namespace qa_accumulate {
using namespace gr::blocks::trigger;
using gr::testing::ProcessFunction;
using gr::testing::TagSink;

/** 1 2 3 | 4 5 | 6, where the bars are the segment-ending triggers at samples 3 and 5 */
[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}, {"5", 5.f}, {"6", 6.f}};
    return map;
}
[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("cut")}}}};
    return map;
}
constexpr std::string_view kScript = "1 2 3 T:4 5 T:6 |";

template<Accumulation mode, typename TOut = float>
[[nodiscard]] std::vector<TOut> segmentsOf(gr::property_map settings) {
    settings[std::string("segment_filter")] = std::string("cut");
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     fold   = fixture.emplace<Accumulate<float, mode>>(std::move(settings));
    auto&                     sink   = fixture.emplace<TagSink<TOut, ProcessFunction::USE_PROCESS_ONE>>();

    boost::ut::expect(fixture.connect<"out", "in">(source, fold).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(fold, sink).has_value());
    boost::ut::expect(fixture.run().has_value());

    return std::vector<TOut>(sink._samples.begin(), sink._samples.end());
}

const boost::ut::suite<"Accumulate"> _accumulate = [] {
    using namespace boost::ut;

    "the last sample of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::last>({});
        expect(found == std::vector<float>{3.f, 5.f}) << "the sample before each cut; the third segment never ended";
    };

    "an unknown predicate is refused with an error message"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::last>>({{"segment_filter", std::string("cut")}, {"predicate", std::string("sideways")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        gr::trigger_test::ErrorMessages sent{fixture.scheduler()};
        expect(fixture.run().has_value());
        expect(eq(sent.countContaining("sideways"), 1UZ));
    };

    "the sum of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("sum")}});
        expect(found == std::vector<float>{6.f, 9.f}) << "1+2+3, then 4+5";
    };

    "the mean of each segment"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("mean")}});
        expect(found == std::vector<float>{2.f, 4.5f});
    };

    "the largest and the smallest of each segment"_test = [] {
        expect(segmentsOf<Accumulation::automatic>({{"operation", std::string("maximum")}}) == std::vector<float>{3.f, 5.f});
        expect(segmentsOf<Accumulation::automatic>({{"operation", std::string("minimum")}}) == std::vector<float>{1.f, 4.f});
    };

    "how many samples each segment held"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("count")}});
        expect(found == std::vector<float>{3.f, 2.f}) << "three samples before the first cut, two between the cuts";
    };

    "whether every sample of the segment compared true"_test = [] {
        const auto all = segmentsOf<Accumulation::all, std::uint8_t>({{"predicate", std::string("less")}, {"threshold", 4.f}});
        expect(all == std::vector<std::uint8_t>{1U, 0U}) << "1 2 3 are all below 4; the segment holding 4 and 5 is not";
    };

    "whether any sample of the segment compared true"_test = [] {
        const auto any = segmentsOf<Accumulation::any, std::uint8_t>({{"predicate", std::string("greater")}, {"threshold", 2.5f}});
        expect(any == std::vector<std::uint8_t>{1U, 1U}) << "3 in the first segment, 4 and 5 in the second";
    };

    "with no trigger named the cadence ends the segments"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"operation", std::string("sum")}, {"n_samples", 2U}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 7.f, 11.f}) << "1+2, 3+4, 5+6: a block with no trigger source is not a silent one";
    };

    "a cadence in seconds is the same limit where the rate is known"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"operation", std::string("count")}, {"n_samples", 0U}, {"timeout", 0.003f}, {"sample_rate", 1000.f}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        const std::vector<float> found(sink._samples.begin(), sink._samples.end());
        expect(found == std::vector<float>{3.f, 3.f}) << "3 ms at 1 kHz is three samples, twice over";
    };

    "a trigger ends a segment before the cadence does"_test = [] {
        const auto found = segmentsOf<Accumulation::automatic>({{"operation", std::string("count")}, {"n_samples", 100U}});
        expect(found == std::vector<float>{3.f, 2.f}) << "whichever comes first, and here it is the trigger";
    };

    "the segments and what each amounted to, drawn"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string(kScript)}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     fold   = fixture.emplace<Accumulate<float, Accumulation::automatic>>({{"segment_filter", std::string("cut")}, {"operation", std::string("sum")}});
        auto&                     sink   = fixture.emplace<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>();
        expect(fixture.connect<"out", "in">(source, fold).has_value());
        expect(fixture.connect<"out", "in">(fold, sink).has_value());
        expect(fixture.run().has_value());

        gr::testing::MarbleDiagram diagram{"Reduce(sum): a trigger ends a segment, and the sum is published where it sat"};
        diagram.unit  = "sample";
        auto& samples = diagram.row("in");
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            samples.at(i, std::format("{}", i + 1UZ));
        }
        samples.completes();
        diagram.condition("Reduce(operation = sum, segment_filter = \"cut\")");
        auto&                            out = diagram.row("out");
        const std::vector<std::uint64_t> cuts{3U, 5U};
        for (std::size_t i = 0UZ; i < sink._samples.size() && i < cuts.size(); ++i) {
            out.at(cuts[i], std::format("{:.0f}", sink._samples[i]));
        }
        out.completes();
        diagram.print();

        expect(eq(sink._samples.size(), 2UZ));
    };
};
} // namespace qa_accumulate

namespace qa_scan {
using namespace gr::blocks::trigger;

[[nodiscard]] const gr::property_map& values() {
    static const gr::property_map map{{"1", 1.f}, {"2", 2.f}, {"3", 3.f}, {"4", 4.f}};
    return map;
}

[[nodiscard]] const gr::property_map& tags() {
    static const gr::property_map map{{"R", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("restart")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}, //
        {"X", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("other")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};
    return map;
}

struct Run {
    std::vector<float> values;
    gr::Size_t         resets = 0U;
};

[[nodiscard]] Run scanned(std::string script, gr::property_map settings) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", values()}, {"sample_tags", tags()}});
    auto&                     block  = fixture.emplace<Scan<float>>(std::move(settings));
    auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
    boost::ut::expect(fixture.connect<"out", "in">(source, block).has_value());
    boost::ut::expect(fixture.connect<"out", "in">(block, sink).has_value());
    boost::ut::expect(fixture.run().has_value());
    return Run{sink._collected, block.n_resets.value};
}

const boost::ut::suite<"Scan"> _scan = [] {
    using namespace boost::ut;

    "a running sum publishes on every sample, not once per segment"_test = [] {
        const Run found = scanned("1 2 3 |", {{"operation", std::string("sum")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f}) << "the running value, which is what separates scan from reduce";
    };

    "every operation the family spells"_test = [] {
        expect(scanned("1 2 3 |", {{"operation", std::string("product")}}).values == std::vector<float>{1.f, 2.f, 6.f});
        expect(scanned("2 1 3 |", {{"operation", std::string("minimum")}}).values == std::vector<float>{2.f, 1.f, 1.f});
        expect(scanned("2 1 3 |", {{"operation", std::string("maximum")}}).values == std::vector<float>{2.f, 2.f, 3.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("count")}}).values == std::vector<float>{1.f, 2.f, 3.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("mean")}}).values == std::vector<float>{1.f, 1.5f, 2.f});
        expect(scanned("1 2 3 |", {{"operation", std::string("last")}}).values == std::vector<float>{1.f, 2.f, 3.f}) << "last is the identity, which is the honest answer for a running 'last'";
    };

    "a matching trigger restarts the accumulation at the sample it sits on"_test = [] {
        const Run found = scanned("1 2 3 R:1 2 |", {{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f, 1.f, 3.f}) << "the tagged sample starts the new total rather than ending the old one";
        expect(eq(found.resets, gr::Size_t{1}));
    };

    "the restart counter is a lifetime statistic that only the transition to INITIALISED zeroes"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 3 R:1 2 |")}, {"sample_values", values()}, {"sample_tags", tags()}});
        auto&                     block  = fixture.emplace<Scan<float>>({{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
        auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(block.n_resets.value, gr::Size_t{1}));
        expect(block.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(eq(block.n_resets.value, gr::Size_t{0})) << "reset() zeroes the lifetime counters";
    };

    "a trigger the filter does not name changes nothing"_test = [] {
        const Run found = scanned("1 2 X:3 |", {{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f});
        expect(eq(found.resets, gr::Size_t{0}));
    };

    "with no filter named nothing restarts it"_test = [] {
        const Run found = scanned("1 2 R:3 |", {{"operation", std::string("sum")}});
        expect(found.values == std::vector<float>{1.f, 3.f, 6.f});
        expect(eq(found.resets, gr::Size_t{0}));
    };

    "a single sample is its own running value"_test = [] { //
        expect(scanned("3 |", {{"operation", std::string("sum")}}).values == std::vector<float>{3.f});
    };

    "the answer does not depend on where the stream was cut"_test = [] {
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ, 5UZ}) {
            gr::testing::GraphFixture fixture;
            auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::string("1 2 3 R:1 2 |")}, {"sample_values", values()}, {"sample_tags", tags()}});
            auto&                     block  = fixture.emplace<Scan<float>>({{"operation", std::string("sum")}, {"reset_filter", std::string("restart")}});
            auto&                     sink   = fixture.emplace<gr::trigger_test::CollectingSink<float>>();
            block.in.max_samples             = chunk;
            expect(fixture.connect<"out", "in">(source, block).has_value());
            expect(fixture.connect<"out", "in">(block, sink).has_value());
            expect(fixture.run().has_value());
            expect(sink._collected == std::vector<float>{1.f, 3.f, 6.f, 1.f, 3.f}) << std::format("at a work-call size of {}", chunk);
        }
    };
};
} // namespace qa_scan

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
