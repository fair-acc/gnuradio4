#include <boost/ut.hpp>

#include "Ccd.hpp"
#include "Figures.hpp"

#include <cstddef>
#include <map>
#include <ranges>
#include <string>
#include <vector>

using namespace boost::ut;
using namespace devtools;

// Ground truth: CCD counted by hand, as J. Lakos defines it ("Large-Scale C++ Software Design", 1996, ch. 4): the sum
// over the components of how many components each one needs, itself included; the fixture under test/code-metrics checks
// the tool's other figures end to end
const suite<"CodeMetrics"> codeMetricsTests = [] {
    "a balanced binary tree of 1, 3 and 7 components has a CCD of 1, 5 and 17"_test = [] {
        // seven: the root needs all 7, its two children 3 each, the four leaves 1 each
        expect(approx(balancedTreeCcd(1UZ), 1.0, 1e-12));
        expect(approx(balancedTreeCcd(3UZ), 3.0 + 1.0 + 1.0, 1e-12));
        expect(approx(balancedTreeCcd(7UZ), 7.0 + 2.0 * 3.0 + 4.0 * 1.0, 1e-12));
    };

    "a balanced tree's NCCD is 1, a flat design's below, a chain's above"_test = [] {
        expect(approx(normalisedCcd(17UZ, 7UZ), 1.0, 1e-12));
        // seven independent components need only themselves: 7
        expect(approx(normalisedCcd(7UZ, 7UZ), 7.0 / 17.0, 1e-12));
        // a chain of seven, each needing all below it: 7 + 6 + … + 1 = 28
        expect(approx(normalisedCcd(28UZ, 7UZ), 28.0 / 17.0, 1e-12));
        expect(gt(normalisedCcd(28UZ, 7UZ), 1.0) && lt(normalisedCcd(7UZ, 7UZ), 1.0));
    };

    // Ground truth: the comparison's rules, applied by hand to the figures of an earlier report
    "a figure that should fall and rose is worse, by its share of the earlier value"_test = [] {
        expect(verdictOf(Better::lower, 2.68, 2.74) == Verdict::worse);
        expect(verdictOf(Better::lower, 2.74, 2.68) == Verdict::better);
        expect(verdictOf(Better::higher, 60.0, 70.0) == Verdict::better);
        expect(approx(*percentChange(2.68, 2.74), 2.2388, 1e-4)) << "(2.74 - 2.68) / 2.68 = 2.24 %";
        expect(!percentChange(0.0, 3.0).has_value()) << "no share of nothing";
    };

    "a figure without a better direction only changes; one not measured now or before is said so"_test = [] {
        expect(verdictOf(Better::neither, 187.0, 190.0) == Verdict::changed);
        expect(verdictOf(Better::lower, 5.0, 5.0) == Verdict::unchanged);
        expect(verdictOf(Better::lower, std::nullopt, 5.0) == Verdict::appeared);
        expect(verdictOf(Better::lower, 5.0, std::nullopt) == Verdict::notMeasured) << "not measured is not zero, which would read as an improvement";
    };

    "a report's figures are read back by name; any other line is not a figure"_test = [] {
        const std::map<std::string, double> figures = figuresIn("translation units parsed: 2 of 2\n  NCCD: 2.74\n      CCD over a balanced binary tree's\n  calls across components: 4.0 %\n  cognitive complexity: not measured (code-metrics.sh --cognitive)\nNCCD: 9\n  unknown: 3\n");
        expect(figures == std::map<std::string, double>{{"NCCD", 2.74}, {"calls across components", 4.0}});
    };

    // Ground truth: the places of two hand-written --list excerpts, and their differences worked out by hand
    "a --list report's places are read back under their figure, keyed without line numbers"_test = [] {
        const Places places = placesIn("  lone callees: 2 (one call site)\n    1 | src/app/A.cpp drawPlot(const Plot &) <- DocumentView | line 12, 4 lines\n    1 | src/app/B.cpp a | b <- C | line 3, 1 lines\n"
                                       "  highest: x 3\n    7 | not a place: no figure above | -\n  cognitive complexity: 9 (2 functions scored)\n    6 | src/app/A.cpp f() | line 40\n");
        expect(places.size() == 2UZ);
        expect(places.at("lone callees").size() == 2UZ);
        expect(places.at("lone callees").at("src/app/A.cpp drawPlot(const Plot &) <- DocumentView").where == "line 12, 4 lines");
        expect(places.at("lone callees").contains("src/app/B.cpp a | b <- C")) << "a key may hold the separator: the value ends at the first, the where begins at the last";
        expect(eq(places.at("cognitive complexity").at("src/app/A.cpp f()").value, 6.0));
        expect(eq(placeLine("src/app/A.cpp f()", Place{.value = 6.0, .where = "line 40"}), std::string{"    6 | src/app/A.cpp f() | line 40"}));
    };

    "places that are new, gone, risen or fallen are named, the largest change first; unchanged ones are not"_test = [] {
        using PlaceMap         = std::map<std::string, Place, std::less<>>;
        const PlaceMap before  = {{"f", {.value = 10.0, .where = "line 1"}}, {"g", {.value = 4.0, .where = "line 9"}}, {"h", {.value = 2.0, .where = "line 20"}}, {"same", {.value = 5.0, .where = "line 30"}}};
        const PlaceMap after   = {{"f", {.value = 13.0, .where = "line 2"}}, {"g", {.value = 1.0, .where = "line 9"}}, {"k", {.value = 7.0, .where = "line 40"}}, {"same", {.value = 5.0, .where = "line 31"}}};
        const auto     changes = placeChanges(before, after);
        const auto     keys    = changes | std::views::transform(&PlaceChange::key) | std::ranges::to<std::vector>();
        expect(keys == std::vector<std::string>{"k", "f", "g", "h"}) << "k new by 7, f rose by 3, g fell by 3 (ties by name), h gone by 2";
        expect(!changes[0].before.has_value() && *changes[0].after == 7.0);
        expect(*changes[1].before == 10.0 && *changes[1].after == 13.0 && changes[1].where == "line 2") << "where is the place now";
        expect(changes[3].where == "line 20" && !changes[3].after.has_value()) << "a gone place keeps where it was";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
