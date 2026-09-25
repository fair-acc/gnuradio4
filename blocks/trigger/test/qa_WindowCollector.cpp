#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/trigger/WindowCollector.hpp>

using gr::blocks::trigger::WindowCollector;

namespace {
[[nodiscard]] std::vector<std::vector<float>> drain(WindowCollector<float>& windows) {
    std::vector<std::vector<float>> found;
    while (auto window = windows.take()) {
        found.push_back(std::move(window->samples));
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
        expect(windows.open(0UZ, 6UZ));
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
        WindowCollector<float> windows;
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
        expect(windows.open(4UZ, 7UZ));
        windows.push(ramp);

        const auto found = drain(windows);
        expect(eq(found.size(), 1UZ));
        if (!found.empty()) {
            expect(eq(found[0].size(), 3UZ));
            expect(approx(found[0][0], 4.f, 1e-6f)) << "it starts where it was told to, not where the stream was";
        }
    };

    "a window reaches back before its opening into the history and forward past it"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(16UZ);
        windows.push(std::span{ramp}.first(4UZ));
        expect(windows.open(1UZ, 7UZ)) << "two samples before the decision at sample 3, three after";
        expect(eq(windows.nClosed(), 0UZ)) << "the samples after it have not arrived yet";

        windows.push(std::span{ramp}.subspan(4UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{1.f, 2.f, 3.f, 4.f, 5.f, 6.f})) << "the window spans the decision, not just what followed it";
            expect(!window->partial);
        }
    };

    "a window reaching back further than the history is cut where the history ends and marked partial"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(2UZ);
        windows.push(std::span{ramp}.first(6UZ));
        expect(windows.open(1UZ, 6UZ)) << "samples 1 to 3 have left a history of two";

        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{4.f, 5.f})) << "only what the history still held, never padding";
            expect(eq(window->startsAt, 4UZ)) << "the window says where it really starts";
            expect(window->partial) << "and the caller is told it is short";
        }
    };

    "what travels with a window comes back with it, in the order the windows closed"_test = [&] {
        WindowCollector<float, std::uint64_t> windows;
        windows.reachBack(16UZ);
        windows.push(std::span{ramp}.first(3UZ));
        expect(windows.open(0UZ, 3UZ, 111U)) << "already complete when opened";
        expect(windows.open(1UZ, 4UZ, 222U));

        windows.push(std::span{ramp}.subspan(3UZ));
        const auto first  = windows.take();
        const auto second = windows.take();
        expect(first.has_value() and second.has_value());
        if (first and second) {
            expect(eq(first->info, std::uint64_t{111U})) << "each window carries the decision it belongs to";
            expect(eq(second->info, std::uint64_t{222U}));
            expect(std::ranges::equal(first->samples, std::vector{0.f, 1.f, 2.f}));
            expect(std::ranges::equal(second->samples, std::vector{1.f, 2.f, 3.f}));
        }
    };

    "a window keeps its samples however long it waits to be read"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(2UZ);
        windows.push(std::span{ramp}.first(2UZ));
        expect(windows.open(1UZ, 3UZ));
        windows.push(std::span{ramp}.subspan(2UZ));
        windows.push(std::span{ramp});

        const auto window = windows.take();
        expect(window.has_value()) << "the history of two has turned over many times";
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{1.f, 2.f}));
        }
    };

    "a closing position moved before the current one shortens the window to it"_test = [&] {
        WindowCollector<float> windows;
        expect(windows.open(0UZ, 100UZ));
        windows.push(std::span{ramp}.first(6UZ));
        windows.openWindows.front().closesAt = 2UZ;

        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{0.f, 1.f}));
        }
    };

    "a reset forgets the windows and the history"_test = [&] {
        WindowCollector<float> windows;
        windows.reachBack(4UZ);
        windows.push(std::span{ramp}.first(3UZ));
        expect(windows.open(1UZ));
        windows.reset();
        expect(eq(windows.nOpen(), 0UZ));
        expect(eq(windows.streamIndex, 0UZ));

        windows.push(std::span{ramp}.subspan(8UZ));
        expect(windows.open(0UZ, 1UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (window) {
            expect(std::ranges::equal(window->samples, std::vector{8.f})) << "the history holds only what came after the reset";
        }
    };

    "what a window cuts out of the history, drawn"_test = [] {
        constexpr std::size_t kLength = 60UZ;
        constexpr std::size_t kPeak   = 30UZ;
        std::vector<float>    stream(kLength, 0.f);
        for (std::size_t i = kPeak; i < kLength && i - kPeak < 15UZ; ++i) {
            stream[i] = 5.f * std::exp(-static_cast<float>(i - kPeak) / 4.f);
        }

        WindowCollector<float> windows;
        windows.reachBack(kLength);
        windows.push(stream);
        expect(windows.open(kPeak - 10UZ, kPeak + 21UZ));
        const auto window = windows.take();
        expect(window.has_value());
        if (!window) {
            return;
        }
        expect(eq(window->samples.size(), 31UZ)) << "ten before the decision, the decision, twenty after";
        expect(eq(window->samples[10UZ], 5.f)) << "the decision's own sample";

        std::vector<double> xStream(kLength);
        std::vector<double> yStream(kLength);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            xStream[i] = static_cast<double>(i);
            yStream[i] = static_cast<double>(stream[i]);
        }
        auto whole = gr::graphs::ImChart<90, 12>({{0., static_cast<double>(kLength - 1UZ)}, {-0.5, 6.}});
        whole.draw(xStream, yStream, "stream");
        std::println("\nthe stream, with the decision at sample {} of {}:", kPeak, kLength);
        whole.draw();

        std::vector<double> xCut(window->samples.size());
        std::vector<double> yCut(window->samples.size());
        for (std::size_t i = 0UZ; i < window->samples.size(); ++i) {
            xCut[i] = static_cast<double>(i) - 10.;
            yCut[i] = static_cast<double>(window->samples[i]);
        }
        auto cut = gr::graphs::ImChart<90, 12>({{xCut.front(), xCut.back()}, {-0.5, 6.}});
        cut.draw(xCut, yCut, "window");
        std::println("\nand the window it cut, zero at the decision -- the ten samples left of it are the history:");
        cut.draw();
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
