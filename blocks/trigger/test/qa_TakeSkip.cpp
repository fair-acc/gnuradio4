#include <boost/ut.hpp>

#include <string>

#include <gnuradio-4.0/test/GraphFixture.hpp>

#include <gnuradio-4.0/trigger/Marble.hpp>
#include <gnuradio-4.0/trigger/TakeSkip.hpp>

#include "TriggerTest.hpp"

using namespace gr::blocks::trigger;
using gr::trigger_test::completed;

const boost::ut::suite<"TakeN and SkipN"> _takeSkip = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}, {"d", 4.0f}, {"e", 5.0f}, {"f", 6.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    auto play = [&]<typename TBlock>(std::string_view script, std::string_view emitted, gr::property_map settings) {
        gr::testing::GraphFixture fixture;
        settings[std::string("filter")] = std::string("start");
        auto& source                    = fixture.emplace<MarbleSource<float>>({{"script", std::string(script)}, {"sample_values", values}, {"sample_tags", tags}});
        auto& block                     = fixture.emplace<TBlock>(std::move(settings));
        auto& sink                      = fixture.emplace<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(fixture.connect<"out", "in">(source, block).has_value());
        expect(fixture.connect<"out", "in">(block, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink.script(), completed(emitted)));
        gr::trigger_test::drawSubset("TakeN/SkipN: what the settings let through", std::format("the run the trigger asked for"), script, sink.script());
    };

    "TakeN forwards n samples from the matching tag"_test = [&] { //
        play.template operator()<TakeN<float>>("a b T:c d e f |", "T:c d", {{"n", 2U}});
    };

    "SkipN suppresses n samples from the matching tag"_test = [&] { //
        play.template operator()<SkipN<float>>("a b T:c d e f |", "a b e f", {{"n", 2U}});
    };

    "a retrigger restarts the run by default"_test = [&] { //
        play.template operator()<TakeN<float>>("a T:b c T:d e f |", "T:b c T:d e", {{"n", 2U}});
    };

    "extend adds to the run instead of restarting it"_test = [&] { //
        play.template operator()<TakeN<float>>("a T:b T:c d e f |", "T:b T:c d e", {{"n", 2U}, {"retrigger", std::string("extend")}});
    };

    "hold is refused where no asynchronous input could release a held sample"_test = [&] {
        gr::testing::GraphFixture fixture;
        auto&                     takeN = fixture.emplace<TakeN<float>>({{"filter", std::string("start")}, {"n", 2U}, {"policy", std::string("hold")}});
        auto&                     skipN = fixture.emplace<SkipN<float>>({{"filter", std::string("start")}, {"n", 2U}, {"policy", std::string("hold")}}); // not `skip`, which is a boost::ut name
        expect(eq(std::string_view{takeN.policy.value}, std::string_view("drop")));
        expect(eq(std::string_view{skipN.policy.value}, std::string_view("drop")));
    };
};

int main() { /* tests are statically executed */ }
