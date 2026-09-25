#include <boost/ut.hpp>

#include <array>
#include <cstdint>
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

    "a dated trigger keeps whole nanoseconds in the time and only the remainder in the offset"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);

        const auto found  = datedTrigger(time, 4UZ, gr::UncertainValue<float>{0.5f, 0.f}, "edge", "ctx");
        const auto stamp  = gr::property_map_view{found}.get_if<std::uint64_t>(gr::tag::TRIGGER_TIME.key());
        const auto offset = gr::property_map_view{found}.get_if<float>(gr::tag::TRIGGER_OFFSET.key());
        expect(stamp != nullptr and offset != nullptr);
        if (stamp != nullptr and offset != nullptr) {
            expect(eq(*stamp, std::uint64_t{1'004'500'000U})) << "four samples plus half a sample, all of it in the time";
            expect(lt(std::abs(*offset), 1e-9f)) << "the offset is the sub-nanosecond remainder, not the sub-sample shift";
        }
    };

    "an undatable trigger carries a name and no time at all"_test = [] {
        const TimeBase time;
        const auto     found = datedTrigger(time, 0UZ, gr::UncertainValue<float>{0.f, 0.f}, "edge", "ctx");
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
