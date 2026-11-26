#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/TimeBase.hpp>

namespace {
using gr::AnchorChange;
using gr::datedTrigger;
using gr::resolveDuration;
using gr::TimeBase;
using gr::withinDuration;

const boost::ut::suite<"TimeBase"> _timeBase = [] {
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
        time.setRate(1000.f);
        expect(time.measuresDuration());
        expect(!time.datesSamples()) << "without an origin there is no absolute time";
        expect(approx(time.seconds(500UZ).value(), 0.5, 1e-12));
        expect(eq(time.samples(0.25).value(), 250UZ));
    };

    "a duration is the shorter of its sample count and its seconds, unset ones ignored"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
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
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 10UZ);
        expect(time.datesSamples());
        expect(eq(time.at(10UZ).value(), std::uint64_t{1'000'000'000U}));
        expect(eq(time.at(11UZ).value(), std::uint64_t{1'001'000'000U}));
        expect(eq(time.at(0UZ).value(), std::uint64_t{990'000'000U})) << "ten samples before the anchor is ten milliseconds earlier";
    };

    "a second anchor re-anchors, and the rate stays the nominal one"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(1'003'000'000U, 0.f, 2UZ);
        expect(eq(time.rate(), 1000.f)) << "a rate measured between anchors would take the jitter for a rate";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'004'000'000U})) << "counted on from the newest anchor";
    };

    "a rate at or above 2^41 Hz, infinite or not a number is no rate"_test = [] {
        TimeBase time;
        time.setRate(0x1p40f);
        expect(time.measuresDuration()) << "2^40 Hz is still exact";
        for (const float rate : {0x1p41f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            time.setRate(rate);
            expect(!time.measuresDuration()) << rate;
            expect(eq(time.rate(), 0.f)) << rate;
        }
    };

    "an offset that is not finite or beyond 2^33 s counts as none"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        for (const float offset : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 0x1p33f}) {
            time.anchorAt(1'000'000'000U, offset, 0UZ);
            expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'000'000U})) << offset;
        }
        time.anchorAt(1'000'000'000U, 0x1p32f, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'000'000U} + std::uint64_t{4'294'967'296U} * 1'000'000'000U)) << "2^32 s, within the range";
    };

    "the anchor offset shifts the origin, in seconds"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.5e-3f, 0UZ);
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'500'000U}));
    };

    "a rate that does not divide a nanosecond does not drift"_test = [] {
        TimeBase time;
        time.setRate(3000.f);
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.at(3'000'000UZ).value(), std::uint64_t{1'000'000'000'000U})) << "1e6 samples at 3 kHz is exactly 1000 s";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'000'000U})) << "and three samples are exactly one millisecond";
    };

    "mulDiv is exact where the product exceeds 64 bits"_test = [] {
        const auto [quotient, remainder] = gr::detail::mulDiv(1'000'000'000'000'000'000U, 1'000'000'000U, 300'000'000U);
        expect(eq(quotient, std::uint64_t{3'333'333'333'333'333'333U})) << "10^27 / (3 · 10^8) = 10^19 / 3";
        expect(eq(remainder, std::uint64_t{100'000'000U})) << "10^27 - 9'999'999'999'999'999'999 · 10^8";
        constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
        expect(gr::detail::mulDiv(kMax, kMax, kMax) == std::pair{kMax, std::uint64_t{0U}});
        expect(gr::detail::mulDiv(kMax, 2U, kMax - 1U) == std::pair{std::uint64_t{2U}, std::uint64_t{2U}}) << "(2^64 - 1) · 2 = 2 · (2^64 - 2) + 2";
    };

    "61.44 MHz dates 2^50 samples to the nanosecond"_test = [] {
        TimeBase time;
        time.setRate(61.44e6f); // 1875 · 2^15: one sample is 3125/192 ns
        time.anchorAt(0U, 0.f, 0UZ);
        if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) { // a stream index on wasm32 stays below 2^32
            expect(eq(time.at(std::size_t{1} << 50U).value(), std::uint64_t{18'325'193'796'266'667U})) << "2^50 · 3125/192 = 18'325'193'796'266'666.67 ns; a double rounds to 4 ns there, ...668";
        }
        expect(eq(time.at(192UZ).value(), std::uint64_t{3'125U}));
    };

    "a rate that is not a whole number of Hz is exact as the float it is"_test = [] {
        TimeBase time;
        time.setRate(1e6f / 3.f); // 333'333.34375 Hz = 10'666'667 / 32
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.rate(), 333'333.34375f));
        expect(eq(time.at(10'666'667UZ).value(), std::uint64_t{32'000'000'000U})) << "10'666'667 samples at 10'666'667/32 Hz are 32 s";
        for (const std::size_t sample : {1UZ, 3UZ, 1'000UZ, 10'666'667UZ}) {
            expect(eq(time.indexAt(time.at(sample).value()).value(), sample)) << "a sample's own stamp is on that sample" << sample;
        }
    };

    "half a nanosecond rounds away from zero, either side of the anchor"_test = [] {
        TimeBase time;
        time.setRate(2e9f);
        time.anchorAt(1'000'000'000U, 0.f, 10UZ);
        expect(eq(time.at(11UZ).value(), std::uint64_t{1'000'000'001U})) << "0.5 ns after, as std::round(0.5)";
        expect(eq(time.at(9UZ).value(), std::uint64_t{999'999'999U})) << "0.5 ns before, as std::round(-0.5)";
    };

    "a duration in float seconds counts the samples of the exact product"_test = [] {
        TimeBase time;
        time.setRate(3.f);
        expect(eq(time.samples(0.5f).value(), 2UZ)) << "1.5 samples round away from zero";
        time.setRate(3e7f);
        expect(eq(time.samples(0.1f).value(), 3'000'000UZ)) << "0.1f = 13'421'773 / 2^27: 3'000'000.0447 samples";
        expect(eq(time.samples(0.1f).value(), time.samples(static_cast<double>(0.1f)).value())) << "as the double product of the two floats, which is exact";
    };

    "a time near the present epoch finds its sample to the nanosecond"_test = [] {
        constexpr std::uint64_t kAnchorNs = 1'800'000'000'000'000'000U;
        TimeBase                time;
        time.setRate(1e8f);
        time.anchorAt(kAnchorNs, 0.f, 100UZ);
        expect(eq(time.indexAt(kAnchorNs + 130U).value(), 113UZ)) << "130 ns at 100 MHz is 13 samples; a double at 1.8e18 ns resolves only 256 ns";
        expect(eq(time.indexAt(kAnchorNs - 130U).value(), 87UZ)) << "and as many before the anchor";
    };

    "a sample before the latest anchor is dated by the anchor it followed, both ways"_test = [] {
        static_assert(std::is_trivially_copyable_v<TimeBase>, "a kernel carries the time base by value");
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(2'012'000'000U, 0.f, 1000UZ);
        expect(eq(time.at(500UZ).value(), std::uint64_t{1'500'000'000U})) << "half a second after the first anchor, not 12 ms later as the newest would extrapolate";
        expect(eq(time.indexAt(1'500'000'000U).value(), 500UZ)) << "and back";
        expect(eq(time.at(1500UZ).value(), std::uint64_t{2'512'000'000U})) << "after the jump the newest anchor dates";
        expect(eq(time.indexAt(2'512'000'000U).value(), 1500UZ));
        expect(eq(time.indexAt(2'005'000'000U).value(), 1000UZ)) << "a time the 12 ms jump skipped is on the jump's sample";
    };

    "at 2 GS/s a sample's own time is on that sample, not a nanosecond's worth of samples earlier"_test = [] {
        TimeBase time;
        time.setRate(2e9f);
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.indexAt(time.at(10UZ).value()).value(), 10UZ)) << "sample 10 is at 5 ns";
        expect(eq(time.indexAt(6U).value(), 12UZ)) << "6 ns is sample 12; sample 11 at 5.5 ns has no whole-ns stamp of its own";
    };

    "a time a backward jump repeats is on the newest anchor's sample"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(1'994'000'000U, 0.f, 1000UZ);
        expect(eq(time.at(999UZ).value(), std::uint64_t{1'999'000'000U})) << "the sample before the jump keeps the first anchor's time";
        expect(eq(time.indexAt(1'997'000'000U).value(), 1003UZ)) << "seen at 997 and at 1003; the newest anchor wins";
        expect(eq(time.indexAt(1'500'000'000U).value(), 500UZ)) << "a time only the first anchor saw";
    };

    "beyond the anchors kept, the oldest kept dates the earlier samples"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        for (std::size_t k = 0UZ; k < TimeBase::kSupersededAnchors + 2UZ; ++k) {
            time.anchorAt(1'000'000'000U + k * 105'000'000U, 0.f, k * 100UZ); // each 100 samples, a 5 ms jump forward
        }
        expect(eq(time.at(150UZ).value(), std::uint64_t{1'155'000'000U})) << "the second anchor, still kept";
        expect(eq(time.at(50UZ).value(), std::uint64_t{1'055'000'000U})) << "the first is dropped: extrapolated back from the second, 5 ms off";
        time.reset();
        time.anchorAt(5'000'000'000U, 0.f, 1000UZ);
        expect(eq(time.at(500UZ).value(), std::uint64_t{4'500'000'000U})) << "a reset forgets the anchors it replaced";
    };

    "an anchor tag with gr:trigger_offset -0.5 s dates sample 0 half a second earlier"_test = [] {
        TimeBase               time;
        const gr::property_map anchorTag{{std::string(gr::tag::kSampleRate.key()), 1000.f}, {std::string(gr::tag::kTriggerTime.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), -0.5f}};
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
        rateOnly.setRate(1000.f);
        expect(!rateOnly.indexAt(1'000'000'000U).has_value()) << "a rate without an anchor has no origin";
    };

    "a reset forgets the anchor and the rate it derived, not the setting"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.reset();
        expect(!time.datesSamples());
        expect(time.measuresDuration()) << "the sample_rate setting survives, an anchor does not";
    };

    "a sample_rate tag is adopted under its wire key gr:sample_rate and its setting key sample_rate"_test = [] {
        TimeBase               wire;
        const gr::property_map wireTag{{std::string(gr::tag::kSampleRate.key()), 48'000.f}};
        std::ignore = wire.adopt(gr::property_map_view{wireTag}, 0UZ);
        expect(eq(wire.rate(), 48'000.f)) << "the wire key is how a source publishes the rate";

        TimeBase               setting;
        const gr::property_map settingTag{{std::string(gr::tag::kSampleRate.shortKey()), 12'000.f}};
        std::ignore = setting.adopt(gr::property_map_view{settingTag}, 0UZ);
        expect(eq(setting.rate(), 12'000.f));
    };

    "a dated trigger carries its event as trigger_time, the tag sitting on the first sample at or after it"_test = [] {
        TimeBase time;
        time.setRate(1000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        const auto                  edge = datedTrigger(time, 4UZ, -0.5f, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        const auto                  offset = view.get_if<float>(gr::tag::kTriggerOffset.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'003'500'000U})) << "sample 4 is at 1.004 s; half a sample (0.5 ms at 1 kHz) earlier";
        expect(eq(*offset, 0.f)) << "a whole number of ns leaves no residual";
    };

    "trigger_offset holds the sub-ns residual of the event time"_test = [] {
        TimeBase time;
        time.setRate(3000.f);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        // sample 4: 4e9/3000 = 1'333'333.33 ns after the anchor, rounded by at() to 1'333'333; half a sample is 1e9/6000 = 166'666.667 ns
        const auto                  edge = datedTrigger(time, 4UZ, -0.5f, 0.f, "edge", "ctx");
        const gr::property_map_view view{edge};
        const auto                  stamp  = view.get_if<std::uint64_t>(gr::tag::kTriggerTime.key());
        const auto                  offset = view.get_if<float>(gr::tag::kTriggerOffset.key());
        expect(stamp != nullptr and offset != nullptr) << fatal;
        expect(eq(*stamp, std::uint64_t{1'001'166'666U})) << "1'001'333'333 - 166'666.667 = 1'001'166'666.333, floored";
        expect(approx(*offset, 0.333333e-9f, 1e-12f)) << "the third of a nanosecond that the floor dropped, in seconds";
    };

    "a time base anchors on the event time of its first tag, up to a sample before the tag's sample"_test = [] {
        TimeBase source;
        source.setRate(1000.f);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);
        const auto pulse = datedTrigger(source, 40UZ, -0.5f, 0.f, "pulse", "");

        TimeBase downstream;
        downstream.setRate(1000.f);
        expect(downstream.adopt(gr::property_map_view{pulse}, 40UZ) == std::optional{AnchorChange::first});
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'039'500'000U}) << "the first anchor is adopted as it is: the sample is 40 ms after the source's origin, less the half sample";
        expect(downstream.at(0UZ) == std::optional<std::uint64_t>{999'500'000U});
    };

    "a derived trigger within one sample does not move the anchor, a jump of one sample or more does"_test = [] {
        TimeBase source;
        source.setRate(1000.f);
        source.anchorAt(1'000'000'000U, 0.f, 0UZ);

        TimeBase downstream;
        downstream.setRate(1000.f);
        const gr::property_map origin{{std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'000'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{origin}, 0UZ) == std::optional{AnchorChange::first});

        const auto derived = datedTrigger(source, 40UZ, -0.5f, 0.f, "derived", ""); // event at 1.0395 s, the running time at sample 40 is 1.040 s: 0.5 ms apart
        expect(!downstream.adopt(gr::property_map_view{derived}, 40UZ).has_value()) << "0.5 sample apart is the tag's own sample lying after its event";
        expect(eq(downstream.anchorIndex, 0UZ));
        expect(eq(downstream.anchorTimeNs.value_or(0U), std::uint64_t{1'000'000'000U})) << "unmoved";

        const gr::property_map exactlyOneSample{{std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'041'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{exactlyOneSample}, 40UZ) == std::optional{AnchorChange::forward}) << "1.041 s against the running 1.040 s is 1 ms = one sample period";
        expect(downstream.at(40UZ) == std::optional<std::uint64_t>{1'041'000'000U});

        const gr::property_map jumpBack{{std::string(gr::tag::kTriggerTime.key()), std::uint64_t{1'030'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), 0.f}};
        expect(downstream.adopt(gr::property_map_view{jumpBack}, 50UZ) == std::optional{AnchorChange::backward}) << "running time at 50 is 1.051 s, the tag says 1.030 s";
        expect(downstream.at(50UZ) == std::optional<std::uint64_t>{1'030'000'000U});
    };

    "a tag dates its sample at trigger_time + trigger_offset"_test = [] {
        const gr::property_map hardware{{std::string(gr::tag::kTriggerTime.key()), std::uint64_t{2'000'000'000U}}, {std::string(gr::tag::kTriggerOffset.key()), 1e-3f}};
        TimeBase               time;
        time.setRate(1000.f);
        std::ignore = time.adopt(gr::property_map_view{hardware}, 10UZ);
        expect(time.at(10UZ) == std::optional<std::uint64_t>{2'001'000'000U}) << "the sample is 1 ms after the hardware trigger";
    };

    "an undatable trigger carries a name and no time at all"_test = [] {
        const TimeBase time;
        const auto     found = datedTrigger(time, 0UZ, 0.f, 0.f, "edge", "ctx");
        expect(!gr::property_map_view{found}.contains(gr::tag::kTriggerTime.key())) << "a fabricated time would be worse than none";
        expect(gr::property_map_view{found}.contains(gr::tag::kTriggerName.key()));
    };
};
} // namespace

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
