#include <boost/ut.hpp>

#include <cstdint>

#include <gnuradio-4.0/trigger/TimeBase.hpp>

using namespace gr::blocks::trigger;

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
        time.setRate(1000.);
        expect(time.measuresDuration());
        expect(!time.datesSamples()) << "without an origin there is no absolute time";
        expect(approx(time.seconds(500UZ).value(), 0.5, 1e-12));
        expect(eq(time.samples(0.25).value(), 250UZ));
    };

    "one anchor plus the rate setting dates every sample"_test = [] {
        TimeBase time;
        time.setRate(1000.); // 1 ms per sample
        time.anchorAt(1'000'000'000U, 0.f, 10UZ);
        expect(time.datesSamples());
        expect(eq(time.at(10UZ).value(), std::uint64_t{1'000'000'000U}));
        expect(eq(time.at(11UZ).value(), std::uint64_t{1'001'000'000U}));
        expect(eq(time.at(0UZ).value(), std::uint64_t{990'000'000U})) << "ten samples before the anchor is ten milliseconds earlier";
    };

    "two anchors fix the rate, and override the setting"_test = [] {
        TimeBase time;
        time.setRate(1.); // deliberately wrong
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.anchorAt(1'002'000'000U, 0.f, 2UZ); // 2 samples in 2 ms = 1 kHz
        expect(approx(time.rate(), 1000., 1e-9)) << "the stream's own timestamps win over the setting";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'003'000'000U}));
    };

    "the anchor offset shifts the origin, in seconds"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.5e-3f, 0UZ); // half a sample later
        expect(eq(time.at(0UZ).value(), std::uint64_t{1'000'500'000U}));
    };

    "a rate that does not divide a nanosecond does not drift"_test = [] {
        TimeBase time;
        time.setRate(3000.); // 333.333... ns per sample, unrepresentable as integer ns
        time.anchorAt(0U, 0.f, 0UZ);
        expect(eq(time.at(3'000'000UZ).value(), std::uint64_t{1'000'000'000'000U})) << "1e6 samples at 3 kHz is exactly 1000 s";
        expect(eq(time.at(3UZ).value(), std::uint64_t{1'000'000U})) << "and three samples are exactly one millisecond";
    };

    "a reset forgets the anchor and the rate it derived, not the setting"_test = [] {
        TimeBase time;
        time.setRate(1000.);
        time.anchorAt(1'000'000'000U, 0.f, 0UZ);
        time.reset();
        expect(!time.datesSamples());
        expect(time.measuresDuration()) << "the sample_rate setting survives, an anchor does not";
    };

    "a dated trigger keeps whole nanoseconds in the time and only the remainder in the offset"_test = [] {
        TimeBase time;
        time.setRate(1000.); // 1 ms per sample, so half a sample is 500 000 ns
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
        const TimeBase time; // no anchor, no rate
        const auto     found = datedTrigger(time, 0UZ, gr::UncertainValue<float>{0.f, 0.f}, "edge", "ctx");
        expect(!gr::property_map_view{found}.contains(gr::tag::TRIGGER_TIME.key())) << "a fabricated time would be worse than none";
        expect(gr::property_map_view{found}.contains(gr::tag::TRIGGER_NAME.key()));
    };
};

int main() { /* tests are statically executed */ }
