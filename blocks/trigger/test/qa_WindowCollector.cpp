#include <boost/ut.hpp>

#include <array>
#include <vector>

#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

using gr::blocks::trigger::WindowCollector;

namespace {
[[nodiscard]] std::vector<std::vector<float>> drain(WindowCollector<float>& windows) {
    std::vector<std::vector<float>> found;
    while (auto window = windows.take()) {
        found.push_back(std::move(*window));
    }
    return found;
}
} // namespace

const boost::ut::suite<"WindowCollector"> _windowCollector = [] {
    using namespace boost::ut;

    const std::array<float, 9> ramp{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};

    "windows that follow one another hold every sample once"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening < 9UZ; opening += 3UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 3UZ));
        if (found.size() == 3UZ) {
            expect(eq(found[0].size(), 3UZ));
            expect(approx(found[1][0], 3.f, 1e-6f)) << "the second window starts where the first ended";
            expect(approx(found[2][2], 8.f, 1e-6f));
        }
    };

    "overlapping windows share the samples they both cover"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening + 3UZ <= 9UZ; opening += 2UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 4UZ)) << "openings at 0, 2, 4 and 6";
        if (found.size() == 4UZ) {
            expect(approx(found[0][2], 2.f, 1e-6f));
            expect(approx(found[1][0], 2.f, 1e-6f)) << "sample 2 belongs to two windows at once";
        }
    };

    "a window that closes where it opened is delivered empty, not skipped"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 0UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ)) << "a quiet interval is a window of length zero, and a consumer counts windows";
        if (!found.empty()) {
            expect(found[0].empty());
        }
    };

    "windows closing out of the order they opened still come out in closing order"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 6UZ)); // opened first, closes last
        expect(windows.open(0UZ, 2UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 2UZ));
        if (found.size() == 2UZ) {
            expect(eq(found[0].size(), 2UZ)) << "the one that closed first comes first";
            expect(eq(found[1].size(), 6UZ));
        }
    };

    "a collector already full refuses a window and counts it"_test = [&] {
        WindowCollector<float> windows; // a designated initialiser skipping a later field is an error under clang
        windows.maxOpen = 2UZ;
        expect(windows.open(0UZ, 100UZ));
        expect(windows.open(0UZ, 100UZ));
        expect(!windows.open(0UZ, 100UZ)) << "a notifier that never closes anything would otherwise grow without limit";
        expect(eq(windows.refused, 1U));
        expect(eq(windows.nOpen(), 2UZ));
    };

    "closing everything ends the open windows where the stream did"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.openHere());
        windows.push(ramp);
        expect(eq(windows.nClosed(), 0UZ)) << "a window with no closing index waits for a notifier";
        windows.closeAll();

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ));
        if (!found.empty()) {
            expect(eq(found[0].size(), 9UZ));
        }
    };

    "a window planned for a later position takes only the samples from there"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(4UZ, 7UZ)); // planned before its samples exist, which is how a block plans ahead
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ));
        if (!found.empty()) {
            expect(eq(found[0].size(), 3UZ));
            expect(approx(found[0][0], 4.f, 1e-6f)) << "it starts where it was told to, not where the stream was";
        }
    };

    "what the windows held, drawn"_test = [&] {
        WindowCollector<float> windows;
        for (std::size_t opening = 0UZ; opening + 3UZ <= 9UZ; opening += 2UZ) {
            expect(windows.open(opening, opening + 3UZ));
        }
        windows.push(ramp);
        const auto found = drain(windows);

        gr::testing::MarbleDiagram diagram{"WindowCollector: length 3 every 2 samples, so each sample is in two windows"};
        diagram.unit = "sample";
        for (std::size_t i = 0UZ; i < found.size(); ++i) {
            auto& row = diagram.row(std::format("w{}", i));
            for (std::size_t j = 0UZ; j < found[i].size(); ++j) {
                row.at(2UZ * i + j, std::format("{:.0f}", found[i][j]));
            }
            row.completes();
        }
        diagram.print();
        expect(eq(found.size(), 4UZ));
    };
};

int main() { /* tests are statically executed */ }
