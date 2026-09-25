#include <boost/ut.hpp>

#include <string>
#include <string_view>
#include <tuple>

#include <gnuradio-4.0/test/EventMarbles.hpp>

namespace {
/// the diagram is written for a terminal, so the assertions read it with the colour stripped out
[[nodiscard]] std::string plain(std::string_view rendered) {
    std::string out;
    for (std::size_t i = 0UZ; i < rendered.size(); ++i) {
        if (rendered[i] == '\x1B') {
            while (i < rendered.size() && rendered[i] != 'm') {
                ++i;
            }
            continue;
        }
        out += rendered[i];
    }
    return out;
}

[[nodiscard]] std::vector<std::string> linesOf(std::string_view text) {
    std::vector<std::string> found;
    std::size_t              start = 0UZ;
    for (std::size_t at = text.find('\n'); at != std::string_view::npos; at = text.find('\n', start)) {
        found.emplace_back(text.substr(start, at - start));
        start = at + 1UZ;
    }
    return found;
}

[[nodiscard]] std::size_t countOf(std::string_view text, std::string_view needle) {
    std::size_t found = 0UZ;
    for (std::size_t at = text.find(needle); at != std::string_view::npos; at = text.find(needle, at + needle.size())) {
        ++found;
    }
    return found;
}
} // namespace

const boost::ut::suite<"EventMarbles"> _eventMarbles = [] {
    using namespace boost::ut;

    "each condition keeps one letter throughout"_test = [] {
        gr::testing::MarbleDiagram diagram{"letters"};
        diagram.row("in").at(0U, "start").at(1000U, "stop").at(2000U, "start");
        const std::string drawn = plain(diagram.render());

        expect(countOf(drawn, "A") >= 2UZ) << "the first condition seen is A, and stays A";
        expect(countOf(drawn, "B") >= 1UZ) << "the second is B";
        expect(drawn.contains("A start")) << "and the legend says which is which";
        expect(drawn.contains("B stop"));
    };

    "events sharing a column stack above the line"_test = [] {
        gr::testing::MarbleDiagram diagram{"stacking"};
        diagram.width = 8UZ;
        diagram.row("in").at(0U, "a").at(0U, "b").at(0U, "c"); // all at one instant on one row
        const std::string drawn = plain(diagram.render());

        const std::vector<std::string> lines   = linesOf(drawn);
        const bool                     topTier = std::ranges::any_of(lines, [](const std::string& line) { //
            return !line.starts_with("in") && !line.starts_with("time") && line.contains("C");
        });
        expect(topTier) << "the third sits two tiers up, on a line of its own with no label";
        const bool baseline = std::ranges::any_of(lines, [](const std::string& line) { return line.starts_with("in") && line.contains("A"); });
        expect(baseline) << "and the first stays on the stream's own line";
    };

    "past four in one column, three are drawn and the rest elided"_test = [] {
        gr::testing::MarbleDiagram diagram{"elision"};
        diagram.width = 6UZ;
        auto& row     = diagram.row("in");
        for (const std::string_view name : {"a", "b", "c", "d", "e", "f"}) {
            row.at(0U, std::string(name));
        }
        const std::string drawn = plain(diagram.render());

        expect(drawn.contains("…")) << "six in one column cannot all be drawn, and the drawing says so";
        expect(countOf(drawn, "…") == 1UZ) << "one ellipsis, standing for everything above the third";
        expect(drawn.contains("A")) << "the earliest three are still shown";
        expect(drawn.contains("B"));
        expect(drawn.contains("C"));
    };

    "an operator row names the condition between the streams"_test = [] {
        gr::testing::MarbleDiagram diagram{"operator"};
        diagram.row("in").at(0U, "start");
        diagram.condition("Gate(once)");
        diagram.row("out").at(0U, "opened");
        const std::string drawn = plain(diagram.render());

        expect(drawn.contains("Gate(once)"));
        expect(drawn.contains("┌")) << "drawn in a box, as rxmarbles does";
        expect(drawn.find("in ") < drawn.find("Gate(once)")) << "inputs above";
        expect(drawn.find("Gate(once)") < drawn.find("out ")) << "and the output below";
    };

    "a finished stream ends in a bar, an unfinished one in an arrow"_test = [] {
        gr::testing::MarbleDiagram diagram{"completion"};
        diagram.row("ends").at(0U, "a").completes();
        diagram.row("runs").at(0U, "a");
        const std::string drawn = plain(diagram.render());

        expect(drawn.contains("│")) << "the completion bar";
        expect(drawn.contains("▶")) << "and time carrying on";
    };

    "a stream that failed ends in a cross, with what it failed at"_test = [] {
        gr::testing::MarbleDiagram diagram{"failure"};
        diagram.row("fails").at(0U, "a").fails("overflow");
        diagram.row("ends").at(1U, "b").completes();
        const std::string drawn = plain(diagram.render());

        expect(drawn.contains("\u2716")) << "the error terminator rxmarbles draws as a cross";
        expect(drawn.contains("overflow")) << "and the reason beside it";
        expect(drawn.contains("\u2502")) << "a clean finish still ends in a bar";
    };

    "everything at one moment is labelled as such rather than spread"_test = [] {
        gr::testing::MarbleDiagram diagram{"instant"};
        diagram.row("in").at(500U, "a");
        diagram.row("out").at(500U, "b");
        expect(plain(diagram.render()).contains("one instant, 500 ns")) << "a spread axis would imply a duration that was never measured";
    };

    "a diagram with no events says so"_test = [] {
        gr::testing::MarbleDiagram diagram{"empty"};
        std::ignore = diagram.row("in");
        expect(plain(diagram.render()).contains("no events"));
    };
};

int main() { /* tests are statically executed */ }
