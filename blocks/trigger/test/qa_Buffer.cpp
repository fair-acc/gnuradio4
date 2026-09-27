#include <boost/ut.hpp>

#include <numeric>
#include <string>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/trigger/Buffer.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::testing::TagSource;
using gr::trigger_test::CollectingSink;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.f}};
    return map;
}
[[nodiscard]] const gr::property_map& kTags() {
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
} // namespace

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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a L:a a a a L:a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a T:a a a a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a T:a a a T:a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a T+T:a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
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
        auto&                     source  = fixture.emplace<MarbleSource<float>>({{"script", std::string("a a T:a a a a T:a a a |")}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
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

int main() { /* tests are statically executed */ }
