#include <boost/ut.hpp>

#include <string>

#include <gnuradio-4.0/test/GraphFixture.hpp>

#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/SampleAndHold.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::completed;

const boost::ut::suite<"SampleAndHold"> _sampleAndHold = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    auto play = [&](std::string_view script, std::string_view emitted, gr::property_map settings) {
        gr::testing::GraphFixture fixture;
        settings[std::string("filter")] = std::string("start");
        auto& source                    = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto& block                     = fixture.emplace<SampleAndHold<float>>(std::move(settings));
        auto& sink                      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink.script(), completed(emitted)));
        gr::trigger_test::drawSubset("SampleAndHold: what the settings let through", std::format("the sample the trigger captured, held"), script, sink.script());
    };

    "the initial value holds until the first match, then the captured sample does"_test = [&] { //
        play("a b T:c d e |", "a a T:c c c", {{"initial_value", 1.0f}});
    };

    "a second match captures again"_test = [&] { //
        play("T:a b T:c d |", "T:a a T:c c", {{"initial_value", 1.0f}});
    };

    "tag_output=false withholds the capture tag"_test = [&] { //
        play("a T:c d |", "a c c", {{"initial_value", 1.0f}, {"tag_output", false}});
    };
};

int main() { /* tests are statically executed */ }
