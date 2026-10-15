#ifndef GNURADIO_DEVTOOLS_FIGURES_HPP
#define GNURADIO_DEVTOOLS_FIGURES_HPP

// one registry of the figures devtools/code-metrics reports, read by every output mode

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace devtools {

enum class Better { lower, higher, neither };

struct Figure {
    std::string_view name; // as the report prints it, and as --compare finds it again
    int              group;
    std::string_view unit;
    Better           better;
    std::string_view meaning;
    std::string_view placesOf; // the figure whose places --list prints and --compare diffs to say where this one moved
    std::string_view remedy;   // printed by --compare beside a figure that moved the worse way: "do:" when mechanical, "usual way:" when it needs judgement
};

// (4) naming, (5) hot paths and (6) compile time are measured elsewhere, or not yet (see kGroups)
constexpr std::string_view kNarrowIncludes = "usual way: a header that uses a type only by reference or pointer declares it (struct Theme;), and its .cpp includes it";

constexpr std::array<Figure, 24> kFigures{{
    {"components", 1, "", Better::neither, "file stems per root (X.hpp and X.cpp are one)", "", ""},
    {"include edges", 1, "", Better::neither, "#include edges between components under the source roots", "include edges", kNarrowIncludes},
    {"components in a cycle", 1, "", Better::lower, "components that reach themselves through #include", "components in a cycle", "do: one of the two headers declares the other's type instead of including it"},
    {"CCD", 1, "", Better::lower, "per component, the components its #includes reach, itself counted, summed", "include edges", kNarrowIncludes},
    {"NCCD", 1, "", Better::lower, "CCD over a balanced binary tree's; 1 is balanced, below flatter, above more entangled", "include edges", kNarrowIncludes},
    {"calls across components", 1, "%", Better::lower, "share of calls to own functions that leave the caller's component", "calls across components", "usual way: a function with one caller in another file moves next to that caller; a call into src/core or a system's entry point is the design"},
    {"lone callees", 1, "", Better::lower, "functions called from one place, in another component, and by no test", "lone callees", "do: inline it at its one caller, or give it a qa_ test; a system's entry point called from the frame loop is exempt (CLAUDE.md §2)"},
    {"unused functions", 1, "", Better::lower, "functions defined under the source roots that no own code names, tests included", "unused functions", "do: delete it; dead API goes (CLAUDE.md §4)"},
    {"functions without ambient reach", 1, "%", Better::higher, "share of source functions that depend on nothing their signature does not show", "functions without ambient reach", "usual way: read the clock, environment or global once where the frame starts and pass the value down; drawing code may draw"},
    {"core functions with ambient reach", 1, "", Better::lower, "core functions that reach a clock, drawing context, environment, I/O or global", "core functions with ambient reach", "do: the core reads nothing ambient: the caller passes the value as a parameter"},
    {"cognitive complexity", 2, "", Better::lower, "Campbell's score summed over the sources (needs code-metrics.sh --cognitive)", "cognitive complexity", "usual way: return early instead of else; join nested ifs with &&; a chain of ifs on one enum becomes a switch"},
    {"split-invariant complexity", 2, "", Better::lower, "the same with single-use, untested callees folded into their caller (needs --cognitive)", "split-invariant complexity", "do: a single-use piece split off gets a qa_ test or a second caller, or is folded back into its caller"},
    {"live mutable locals", 2, "", Better::lower, "locals written in more than one top-level statement of a function", "live mutable locals", "do: one const local per value, or a block that owns the writes; a scanner's cursor and a state machine's state are exempt"},
    {"write-only fields", 2, "", Better::lower, "fields under the source roots that no own code reads, tests included", "write-only fields", "do: delete the field and its writes"},
    {"nested ternaries", 2, "", Better::lower, "?: holding another ?: in the same expression", "nested ternaries", "do: name the inner choice as a const local (CLAUDE.md §2: one expression per idea)"},
    {"flag arguments", 2, "", Better::lower, "parameters of own functions that some call fills with a literal true or false", "flag arguments", "usual way: an enum class naming the modes, or a const bool local named for what it says at the call"},
    {"same argument everywhere", 2, "", Better::lower, "parameters of own functions that every call (two or more) fills with the same constant", "same argument everywhere", "do: drop the parameter and use the value inside the function"},
    {"unscoped pairs", 2, "", Better::lower, "push and pop, begin and end, lock and unlock written by hand in one function, +1 per exit or branch between them", "unscoped pairs", "do: a scope object closes it (a c_resource alias, ScopedFont), so no exit or branch can skip the closing"},
    {"tokens + comment words", 3, "", Better::lower, "everything there is to read", "tokens + comment words", "usual way: delete what restates the code; reuse what the codebase has instead of writing it again"},
    {"SLOC", 3, "", Better::neither, "lines holding code; reported, not optimised", "", ""},
    {"repeated expressions", 3, "", Better::lower, "expressions of 8 tokens or more written 3 times or more in one file, counted beyond their first writing", "repeated expressions", "do: give the value a name: a const local, or a member function when the expression reads one type's fields"},
    {"restating comments", 3, "", Better::lower, "comments whose words the next lines' identifiers already say", "restating comments", "do: delete it, unless it names a unit, a range, an empty value's meaning or a source"},
    {"lines over 160 characters", 3, "", Better::lower, "lines a reader must scroll or wrap", "lines over 160 characters", "do: split the line into named const locals"},
    {"lines over 250 characters", 3, "", Better::lower, "lines no one reads whole", "lines over 250 characters", "do: split the line into named const locals"},
}};

struct Group {
    int              number;
    std::string_view title;
    std::string_view measuredBy; // empty: by code-metrics itself
};

constexpr std::array<Group, 6> kGroups{{
    {1, "composition: how far a change can reach (includes, calls, ambient dependencies)", ""},
    {2, "cognitive complexity: how much a reader must hold", ""},
    {3, "reading volume: how much there is to read", ""},
    {4, "naming and API consistency", "devtools/naming-check"},
    {5, "hot paths: instructions per item, binary and .wasm size", "devtools/code-metrics.sh --hot-paths"},
    {6, "compile time", "not measured yet"},
}};

using Measured = std::optional<double>;

enum class Verdict { better, worse, unchanged, changed, appeared, notMeasured };

/// how a figure moved from `before` to `after`: changed when it has no better direction, appeared when the earlier
/// report lacks it, not measured when this run lacks it
[[nodiscard]] inline Verdict verdictOf(Better better, Measured before, Measured after) noexcept {
    if (!after) {
        return Verdict::notMeasured;
    }
    if (!before) {
        return Verdict::appeared;
    }
    if (*after == *before) {
        return Verdict::unchanged;
    }
    if (better == Better::neither) {
        return Verdict::changed;
    }
    return (*after < *before) == (better == Better::lower) ? Verdict::better : Verdict::worse;
}

[[nodiscard]] constexpr std::string_view verdictName(Verdict verdict) noexcept {
    switch (verdict) {
    case Verdict::better: return "better";
    case Verdict::worse: return "worse";
    case Verdict::unchanged: return "unchanged";
    case Verdict::changed: return "changed";
    case Verdict::appeared: return "new";
    case Verdict::notMeasured: return "not measured";
    }
    return {};
}

[[nodiscard]] constexpr std::string_view betterName(Better better) noexcept {
    switch (better) {
    case Better::lower: return "lower";
    case Better::higher: return "higher";
    case Better::neither: return "neither";
    }
    return {};
}

/// the change from `before` to `after` in percent of `before`; nothing when there is no base to relate it to
[[nodiscard]] inline std::optional<double> percentChange(double before, double after) noexcept { return before == 0.0 ? std::nullopt : std::optional{100.0 * (after - before) / std::abs(before)}; }

template<typename TOnLine>
void forEachLine(std::string_view report, TOnLine&& onLine) {
    while (!report.empty()) {
        const std::size_t end = report.find('\n');
        onLine(report.substr(0, end));
        report = end == std::string_view::npos ? std::string_view{} : report.substr(end + 1);
    }
}

/// the registry figure whose line `line` is (`  <name>: <number>…`), or nullptr
[[nodiscard]] inline const Figure* figureOfLine(std::string_view line) noexcept {
    for (const Figure& figure : kFigures) {
        if (line.starts_with("  ") && line.substr(2).starts_with(figure.name) && line.substr(2 + figure.name.size()).starts_with(": ")) {
            return &figure;
        }
    }
    return nullptr;
}

/// the figures a report states, by name: lines `  <name>: <number>…`; any other line is not a figure
[[nodiscard]] inline std::map<std::string, double> figuresIn(std::string_view report) {
    std::map<std::string, double> figures;
    forEachLine(report, [&figures](std::string_view line) {
        if (const Figure* figure = figureOfLine(line)) {
            const std::string_view valueText = line.substr(4 + figure->name.size());
            double                 value     = 0.0;
            if (std::from_chars(valueText.data(), valueText.data() + valueText.size(), value).ec == std::errc{}) {
                figures[std::string{figure->name}] = value;
            }
        }
    });
    return figures;
}

/// one place behind a figure; its key names it without a line number, so it is found again after edits above it
struct Place {
    double      value = 0.0; // its share of the figure: 1 for a listed item, else a count or a score
    std::string where;       // for the reader: line, and what else the item says
};
using Places = std::map<std::string, std::map<std::string, Place, std::less<>>, std::less<>>; // figure name -> key -> place

[[nodiscard]] inline std::string placeLine(std::string_view key, const Place& place) { return std::format("    {:.0f} | {} | {}", place.value, key, place.where); }

/// the places a --list report prints: lines `    <value> | <key> | <where>` under the line of the figure they belong to
[[nodiscard]] inline Places placesIn(std::string_view report) {
    Places      places;
    std::string figureName;
    forEachLine(report, [&](std::string_view line) {
        if (const Figure* figure = figureOfLine(line)) {
            figureName = figure->name;
            return;
        }
        const std::size_t first = line.find(" | ");
        const std::size_t last  = line.rfind(" | ");
        if (!line.starts_with("    ") || first == std::string_view::npos || first == last) {
            figureName.clear();
            return;
        }
        double value = 0.0;
        if (!figureName.empty() && std::from_chars(line.data() + 4, line.data() + first, value).ec == std::errc{}) {
            places[figureName][std::string{line.substr(first + 3, last - first - 3)}] = Place{.value = value, .where = std::string{line.substr(last + 3)}};
        }
    });
    return places;
}

struct PlaceChange {
    std::string key;
    Measured    before; // nothing: new
    Measured    after;  // nothing: gone
    std::string where;  // now, or before when gone
};

/// the places that are new, gone or changed in value, the largest change first
[[nodiscard]] inline std::vector<PlaceChange> placeChanges(const std::map<std::string, Place, std::less<>>& before, const std::map<std::string, Place, std::less<>>& after) {
    std::vector<PlaceChange> changes;
    for (const auto& [key, place] : after) {
        const auto earlier = before.find(key);
        if (earlier == before.end()) {
            changes.push_back(PlaceChange{.key = key, .before = std::nullopt, .after = place.value, .where = place.where});
        } else if (earlier->second.value != place.value) {
            changes.push_back(PlaceChange{.key = key, .before = earlier->second.value, .after = place.value, .where = place.where});
        }
    }
    for (const auto& [key, place] : before) {
        if (!after.contains(key)) {
            changes.push_back(PlaceChange{.key = key, .before = place.value, .after = std::nullopt, .where = place.where});
        }
    }
    const auto size = [](const PlaceChange& change) { return std::abs(change.after.value_or(0.0) - change.before.value_or(0.0)); };
    std::ranges::sort(changes, [&size](const PlaceChange& a, const PlaceChange& b) { return std::tuple{size(b), a.key} < std::tuple{size(a), b.key}; });
    return changes;
}

} // namespace devtools

#endif // GNURADIO_DEVTOOLS_FIGURES_HPP
