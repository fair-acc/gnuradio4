#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <gnuradio-4.0/trigger/Events.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

using namespace gr::blocks::trigger;

namespace {
const auto timeBase = [] {
    using namespace boost::ut;

    "with neither an anchor nor a rate nothing can be dated"_test = [] {
        TimeBase time;
        expect(!time.measuresDuration());
        expect(!time.datesSamples());
        expect(!time.seconds(1000UZ).has_value());
        expect(!time.at(0UZ).has_value());
    };

    "a rate alone measures a duration but dates no sample"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        expect(time.measuresDuration());
        expect(!time.datesSamples()) << "without an origin there is no absolute time";
        expect(approx(time.seconds(500UZ).value(), 0.5, 1e-12));
        expect(eq(time.samples(0.25).value(), 250UZ));
    };

    "a duration is the shorter of its sample count and its seconds, unset ones ignored"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        expect(!time.samples(-1.).has_value()) << "a negative duration has no sample count";
        expect(!TimeBase{}.samples(1.).has_value()) << "without a rate seconds cannot be counted";

        struct Case {
            gr::Size_t  samples;
            float       seconds;
            std::size_t expected;
        };
        for (const auto& [samples, seconds, expected] : {Case{0U, 0.f, 0UZ}, Case{40U, 0.f, 40UZ}, Case{0U, 0.05f, 50UZ}, Case{40U, 0.05f, 40UZ}, Case{80U, 0.05f, 50UZ}}) {
            const auto resolved = resolveDuration(time, samples, seconds);
            expect(eq(resolved.value_or(0UZ), expected)) << samples << seconds;
            expect(eq(resolved.has_value(), expected != 0UZ));
        }
        expect(!resolveDuration(TimeBase{}, 0U, 0.05f).has_value()) << "seconds need a rate";

        expect(withinDuration(time, 30UZ, 20U, 0.f, 0U, 0.05f) && withinDuration(time, 20UZ, 20U, 0.f, 0U, 0.f) && withinDuration(time, 50UZ, 0U, 0.f, 0U, 0.05f));
        expect(!withinDuration(time, 19UZ, 20U, 0.f, 0U, 0.f) && !withinDuration(time, 51UZ, 0U, 0.f, 0U, 0.05f));
        expect(withinDuration(time, 123456UZ, 0U, 0.f, 0U, 0.f)) << "no bound accepts any width";
    };

    "one anchor plus the rate setting dates every sample"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 10UZ);
        expect(time.datesSamples());
        expect(eq(time.at(10UZ).value(), std::uint64_t{1'000'000'000U}));
        expect(eq(time.at(11UZ).value(), std::uint64_t{1'001'000'000U}));
        expect(eq(time.at(0UZ).value(), std::uint64_t{990'000'000U})) << "ten samples before the anchor is ten milliseconds earlier";
    };

    "a second anchor re-anchors, and the rate stays the nominal one"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(1'003'000'000U, 0.f, 2UZ);
        expect(approx(time.rate(), 1000., 1e-9)) << "a rate measured between anchors would take the jitter for a rate";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'004'000'000U})) << "counted on from the newest anchor";
    };

    "the anchor offset shifts the origin, in seconds"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.5e-3f, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'500'000U}));
    };

    "a rate that does not divide a nanosecond does not drift"_test = [] {
        TimeBase time;
        time.setRate(3000.);
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.at(3'000'000UZ).value(), std::uint64_t{1'000'000'000'000U})) << "1e6 samples at 3 kHz is exactly 1000 s";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'000'000U})) << "and three samples are exactly one millisecond";
    };

    "a time near the present epoch finds its sample to the nanosecond"_test = [] {
        constexpr std::uint64_t kAnchorNs = 1'800'000'000'000'000'000U;
        TimeBase                time;
        time.setRate(1e8);
        time.anchorAt(kAnchorNs, 0.f, 100UZ);
        expect(eq(time.indexAt(kAnchorNs + 130U).value(), 113UZ)) << "130 ns at 100 MHz is 13 samples; a double at 1.8e18 ns resolves only 256 ns";
        expect(eq(time.indexAt(kAnchorNs - 130U).value(), 87UZ)) << "and as many before the anchor";
    };

    "an anchor tag with gr:trigger_offset -0.5 s dates sample 0 half a second earlier"_test = [] {
        TimeBase               time;
        const gr::property_map anchorTag{{std::string(gr::tag::SAMPLE_RATE.key()), 1000.f}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), -0.5f}};
        std::ignore = time.adopt(gr::property_map_view{anchorTag}, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'500'000'000U})) << "2.0 s - 0.5 s";
        expect(eq(time.at(1UZ).value(), std::uint64_t{1'501'000'000U})) << "one sample at 1 kHz later";
        expect(eq(time.indexAt(1'500'000'000U).value(), 0UZ));
    };

    "without a dated rate indexAt has no answer"_test = [] {
        TimeBase anchorOnly;
        anchorOnly.anchorAt(1'000'000'000U, 0.f, 0UZ);
        expect(!anchorOnly.indexAt(1'000'000'000U).has_value()) << "an anchor without a rate cannot convert time to samples";

        TimeBase rateOnly;
        rateOnly.setRate(1000.);
        expect(!rateOnly.indexAt(1'000'000'000U).has_value()) << "a rate without an anchor has no origin";
    };

    "a reset forgets the anchor and the rate it derived, not the setting"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.reset();
        expect(!time.datesSamples());
        expect(time.measuresDuration()) << "the sample_rate setting survives, an anchor does not";
    };

    "a sample_rate tag is adopted under its wire key gr:sample_rate and its setting key sample_rate"_test = [] {
        TimeBase               wire;
        const gr::property_map wireTag{{std::string(gr::tag::SAMPLE_RATE.key()), 48'000.f}};
        std::ignore = wire.adopt(gr::property_map_view{wireTag}, 0UZ);
        expect(approx(wire.rate(), 48'000., 1e-9)) << "the wire key is how a source publishes the rate";

        TimeBase               setting;
        const gr::property_map settingTag{{std::string(gr::tag::SAMPLE_RATE.shortKey()), 12'000.f}};
        std::ignore = setting.adopt(gr::property_map_view{settingTag}, 0UZ);
        expect(approx(setting.rate(), 12'000., 1e-9));
    };

    "the merged condition of several tags carries the latest gr:sample_rate"_test = [] {
        const gr::property_map                                                            first{{std::string(gr::tag::SAMPLE_RATE.key()), 1'000.f}};
        const gr::property_map                                                            second{{std::string(gr::tag::SAMPLE_RATE.key()), 2'000.f}, {std::string(gr::tag::TRIGGER_NAME.key()), std::string("edge")}};
        const std::vector<gr::Tag>                                                        tags{gr::Tag{3UZ, gr::property_map_view{first}}, gr::Tag{3UZ, gr::property_map_view{second}}};
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> scratch{};

        const gr::property_map_view merged = gr::blocks::trigger::detail::mergedAt(tags.begin(), tags.end(), scratch);
        const auto                  rate   = merged.get_if<float>(gr::tag::SAMPLE_RATE.key());
        expect(rate != nullptr) << "a rate published under the wire key must not be dropped";
        if (rate != nullptr) {
            expect(eq(*rate, 2'000.f));
        }
    };

    "a dated trigger carries its event as trigger_time, the tag sitting on the first sample at or after it"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        const auto                  edge = datedTrigger(time, 4UZ, -0.5, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto                  offset = view.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'003'500'000U})) << "sample 4 is at 1.004 s; half a sample (0.5 ms at 1 kHz) earlier";
        expect(eq(*offset, 0.f)) << "a whole number of ns leaves no residual";
    };

    "trigger_offset holds the sub-ns residual of the event time"_test = [] {
        TimeBase time;
        time.setRate(3000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        // sample 4: 4e9/3000 = 1'333'333.33 ns after the anchor, rounded by at() to 1'333'333; half a sample is 1e9/6000 = 166'666.667 ns
        const auto                  edge = datedTrigger(time, 4UZ, -0.5, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto                  offset = view.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'001'166'666U})) << "1'001'333'333 - 166'666.667 = 1'001'166'666.333, floored";
        expect(approx(*offset, 0.333333e-9f, 1e-12f)) << "the third of a nanosecond that the floor dropped, in seconds";
    };

    "a time base anchors on the event time of its first tag, up to a sample before the tag's sample"_test = [] {
        TimeBase source;
        source.setRate(1000.);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);
        const auto pulse = datedTrigger(source, 40UZ, -0.5, 0.f, "pulse", "");

        TimeBase downstream;
        downstream.setRate(1000.);
        expect(downstream.adopt(gr::property_map_view{pulse}, 40UZ) == std::optional{AnchorChange::first});
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'039'500'000U}) << "the first anchor is adopted as it is: the sample is 40 ms after the source's origin, less the half sample";
        expect(downstream.at(0UZ) == std::optional<std::uint64_t>{999'500'000U});
    };

    "a derived trigger within one sample does not move the anchor, a jump of one sample or more does"_test = [] {
        TimeBase source;
        source.setRate(1000.);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);

        TimeBase downstream;
        downstream.setRate(1000.);
        const gr::property_map origin{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{origin}, 0UZ) == std::optional{AnchorChange::first});

        const auto derived = datedTrigger(source, 40UZ, -0.5, 0.f, "derived", ""); // event at 1.0395 s, the running time at sample 40 is 1.040 s: 0.5 ms apart
        expect(!downstream.adopt(gr::property_map_view{derived}, 40UZ).has_value()) << "0.5 sample apart is the tag's own sample lying after its event";
        expect(eq(downstream.anchorIndex, 0UZ));
        expect(eq(downstream.anchorTimeNs.value_or(0U), std::uint64_t{1'000'000'000U})) << "unmoved";

        const gr::property_map exactlyOneSample{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'041'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{exactlyOneSample}, 40UZ) == std::optional{AnchorChange::forward}) << "1.041 s against the running 1.040 s is 1 ms = one sample period";
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'041'000'000U});

        const gr::property_map jumpBack{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1'030'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{jumpBack}, 50UZ) == std::optional{AnchorChange::backward}) << "running time at 50 is 1.051 s, the tag says 1.030 s";
        expect(downstream.at(50UZ) == std::optional<std::uint64_t>{1'030'000'000U});
    };

    "a tag dates its sample at trigger_time + trigger_offset"_test = [] {
        const gr::property_map hardware{{std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 1e-3f}};
        TimeBase               time;
        time.setRate(1000.);
        std::ignore = time.adopt(gr::property_map_view{hardware}, 10UZ);
        expect(time.at(10UZ) == std::optional<std::uint64_t>{2'001'000'000U}) << "the sample is 1 ms after the hardware trigger";
    };

    "an undatable trigger carries a name and no time at all"_test = [] {
        const TimeBase time;
        const auto     found = datedTrigger(time, 0UZ, 0., 0.f, "edge", "ctx");
        expect(!gr::property_map_view{found}.contains(gr::tag::TRIGGER_TIME.key())) << "a fabricated time would be worse than none";
        expect(gr::property_map_view{found}.contains(gr::tag::TRIGGER_NAME.key()));
    };
};

} // namespace

int main() {
    using namespace boost::ut;
    "TimeBase"_test = timeBase;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
