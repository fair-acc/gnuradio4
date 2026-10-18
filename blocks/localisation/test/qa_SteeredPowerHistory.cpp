#include <boost/ut.hpp>

#include <array>
#include <limits>

#include "../src/SteeredPowerMapExample.hpp"

const boost::ut::suite<"SteeredPowerHistory"> historyTests = [] {
    using namespace boost::ut;

    "a batch retains every map with the newest frame at age zero"_test = [] {
        gr::examples::SteeredPowerHistory history;
        for (const float value : {1.f, 2.f, 3.f}) {
            const std::array row{value, -value};
            expect(history.append(row, 4.f).has_value());
        }
        expect(eq(history.rows.size(), 3UZ));
        expect(eq(history.rows[0UZ][0UZ], 1.f));
        expect(eq(history.rows[1UZ][0UZ], 2.f));
        expect(eq(history.rows[2UZ][0UZ], 3.f));
        expect(eq(history.ageSeconds(0UZ), -0.5));
        expect(eq(history.ageSeconds(1UZ), -0.25));
        expect(eq(history.ageSeconds(2UZ), 0.0));
    };

    "capacity rollover discards only the oldest map"_test = [] {
        gr::examples::SteeredPowerHistory history;
        for (std::size_t frame = 0UZ; frame <= gr::examples::SteeredPowerHistory::kMaxFrames; ++frame) {
            const std::array row{static_cast<float>(frame)};
            expect(history.append(row, 4.f).has_value());
        }
        expect(eq(history.rows.size(), gr::examples::SteeredPowerHistory::kMaxFrames));
        expect(eq(history.rows.front().front(), 1.f));
        expect(eq(history.rows.back().front(), static_cast<float>(gr::examples::SteeredPowerHistory::kMaxFrames)));
        expect(eq(history.ageSeconds(0UZ), -63.75));
        expect(eq(history.ageSeconds(history.rows.size() - 1UZ), 0.0));
    };

    "rate or shape changes begin a fresh history and invalid rows leave it intact"_test = [] {
        gr::examples::SteeredPowerHistory history;
        constexpr std::array              kRow{1.f, 2.f};
        expect(history.append(kRow, 4.f).has_value());
        expect(history.append(kRow, 4.f).has_value());
        expect(history.append(kRow, 2.f).has_value());
        expect(eq(history.rows.size(), 1UZ));
        expect(eq(history.frameRate, 2.f));
        expect(eq(history.ageSeconds(0UZ), 0.0));
        constexpr std::array kNarrow{3.f};
        expect(history.append(kNarrow, 2.f).has_value());
        expect(eq(history.rows.size(), 1UZ));
        expect(eq(history.rows.front().size(), 1UZ));
        expect(!history.append({}, 2.f));
        expect(!history.append(kRow, 0.f));
        expect(!history.append(kRow, -1.f));
        expect(!history.append(kRow, std::numeric_limits<float>::quiet_NaN()));
        expect(!history.append(kRow, std::numeric_limits<float>::infinity()));
        expect(eq(history.rows.size(), 1UZ));
        expect(eq(history.rows.front().front(), 3.f));
        expect(eq(history.frameRate, 2.f));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
