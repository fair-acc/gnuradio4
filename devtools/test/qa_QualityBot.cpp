#include <boost/ut.hpp>

#include "QualityBot.hpp"

using namespace boost::ut;
using namespace devtools::bot;

const suite<"QualityBot"> tests = [] {
    "a quoted CSV field keeps its comma, quote and line break, and a table reads back as written"_test = [] {
        const Table       written{.columns = {"commit", "note", "total"}, .rows = {Row{{"commit", "a1"}, {"note", "say \"hi\", then\nleave"}, {"total", "76.9"}}, Row{{"commit", "b2"}, {"total", ""}}}};
        const std::string text = csvText(written);
        expect(eq(text, std::string{"commit,note,total\na1,\"say \"\"hi\"\", then\nleave\",76.9\nb2,,\n"}));
        const Table read = csvTable(text);
        expect(read.columns == written.columns);
        expect(eq(read.rows.size(), 2UZ));
        expect(eq(read.rows[0].at("note"), std::string{"say \"hi\", then\nleave"}));
        expect(eq(read.rows[1].at("note"), std::string{})) << "a missing value is empty, never zero";
    };

    "a number keeps its value through percent signs and thousands separators, and text is no number"_test = [] {
        expect(eq(numberIn("76.9 %").value_or(-1.0), 76.9));
        expect(eq(numberIn("1,234,567").value_or(-1.0), 1234567.0));
        expect(!numberIn("").has_value());
        expect(!numberIn("n/a").has_value());
        expect(!numberIn("12abc").has_value());
    };

    "thousands are grouped by three from the right"_test = [] {
        expect(eq(withThousands(0), std::string{"0"}));
        expect(eq(withThousands(999), std::string{"999"}));
        expect(eq(withThousands(1000), std::string{"1,000"}));
        expect(eq(withThousands(71839253671), std::string{"71,839,253,671"}));
        expect(eq(withThousands(-1234), std::string{"-1,234"}));
    };

    "a change within half a tenth of a point reads ±0.0, a missing side n/a"_test = [] {
        expect(eq(signedChange(76.94, 76.9), std::string{"±0.0"}));
        expect(eq(signedChange(77.0, 76.9), std::string{"+0.1"}));
        expect(eq(signedChange(75.0, 76.9), std::string{"-1.9"}));
        expect(eq(signedChange(std::nullopt, 76.9), std::string{"n/a"}));
    };

    "a cut never splits a UTF-8 character"_test = [] {
        const std::string text = "ab🟢"; // 🟢 is four bytes
        expect(eq(cutUtf8(text, 3), std::string{"ab"}));
        expect(eq(cutUtf8(text, 5), std::string{"ab"}));
        expect(eq(cutUtf8(text, 6), text));
    };

    "a JSON string escapes quotes, backslashes and control characters"_test = [] { expect(eq(jsonQuoted("a\"b\\c\nd\x01"), std::string{R"("a\"b\\c\nd\u0001")"})); };

    "the changed lines of a diff are the added hunks' lines, a deleted file contributes none"_test = [] {
        const std::string diff    = "diff --git a/x.cpp b/x.cpp\n--- a/x.cpp\n+++ b/x.cpp\n@@ -3,0 +4,2 @@\n+a\n+b\n@@ -9 +11 @@\n-c\n+d\n"
                                    "diff --git a/gone.cpp b/gone.cpp\n--- a/gone.cpp\n+++ /dev/null\n@@ -1,3 +0,0 @@\n";
        const auto        changed = changedLinesIn(diff);
        expect(eq(changed.size(), 1UZ));
        expect(changed.at("x.cpp") == std::set<unsigned>{4U, 5U, 11U});
    };

    "line numbers read as ranges"_test = [] {
        expect(eq(rangesOf({3U, 5U, 6U, 7U, 9U}), std::string{"L3, L5–L7, L9"}));
        expect(eq(rangesOf({}), std::string{}));
    };

    "the base is the recorded commit, else its nearest recorded ancestor, never compared across tools but charted across them"_test = [] {
        const std::vector<Row> history{Row{{"commit", "c1aaaaaaaa"}, {"tool", "t1"}, {"total", "70"}}, Row{{"commit", "c2bbbbbbbb"}, {"tool", "t1"}, {"total", "71"}}};
        const Base             exact = baseFrom("c2bbbbbbbb", {"c2bbbbbbbb", "c1aaaaaaaa"}, history, "t1");
        expect(eq(exact.row.at("total"), std::string{"71"}));
        expect(eq(exact.rows.size(), 2UZ));
        expect(exact.note.empty());
        const Base nearest = baseFrom("c3cccccccc", {"c3cccccccc", "c1aaaaaaaa"}, history, "t1");
        expect(eq(nearest.row.at("total"), std::string{"70"}));
        expect(eq(nearest.rows.size(), 1UZ)) << "the chart ends at the stand-in";
        expect(eq(nearest.note, std::string{"not recorded; compared with `c1aaaaaa`"}));
        const Base otherTools = baseFrom("c2bbbbbbbb", {"c2bbbbbbbb"}, history, "t2");
        expect(otherTools.row.empty()) << "no comparison across tools";
        expect(eq(otherTools.rows.size(), 2UZ)) << "the history still charts";
        expect(eq(otherTools.note, std::string{"tools changed since `c2bbbbbb`: not comparable"}));
        const Base none = baseFrom("c9", {"c9"}, history, "t1");
        expect(eq(none.note, std::string{"not recorded yet"}));
    };

    "a report's figures are its two-space lines, not the places indented beneath them"_test = [] {
        const std::string report  = "(3) reading volume\n  repeated expressions: 2837 (written three times or more)\n    14 | blocks/basic/ConverterBlocks.hpp: 0.5 * x | line 3\n  SLOC: 71,150\n";
        const auto        figures = figureLines(report);
        expect(eq(figures.size(), 2UZ));
        expect(eq(figures[0].first, std::string{"repeated expressions"}));
        expect(eq(figures[0].second, std::string{"2837"}));
        expect(eq(figures[1].second, std::string{"71,150"}));
    };

    "an ISO date with an offset is the same instant as its UTC spelling"_test = [] {
        expect(eq(epochSecondsOf("1970-01-01T00:00:00+00:00").value_or(-1), 0LL));
        expect(eq(epochSecondsOf("2026-10-03T08:25:32+02:00").value_or(-1), epochSecondsOf("2026-10-03T06:25:32+00:00").value_or(-2)));
        expect(!epochSecondsOf("yesterday").has_value());
    };

    "a chart point is labelled by its PR, else its short commit, and carries the change against the point before"_test = [] {
        const std::vector<Row>        rows{Row{{"commit", "aaaaaaa111"}, {"date", "2026-10-01T08:00:00+02:00"}, {"pr", "850"}, {"tool", "t1"}, {"total", "70"}}, Row{{"commit", "bbbbbbb222"}, {"date", "2026-10-02T09:30:00+02:00"}, {"pr", ""}, {"tool", "t1"}, {"total", "71.5"}}};
        const std::vector<ChartPoint> drawn = chartPoints(rows, "total", 71.0, "2026-10-03T10:00:00+02:00", "t1", 100, 365LL * 86'400);
        expect(eq(drawn.size(), 3UZ));
        expect(eq(drawn[0].label, std::string{"#850"}));
        expect(eq(drawn[1].label, std::string{"bbbbbbb"}));
        expect(eq(drawn[2].label, std::string{"this"}));
        expect(eq(drawn[0].date, std::string{"2026-10-01 06:00"})) << "in UTC, so committers of different offsets share one axis";
        expect(!drawn[0].change.has_value()) << "the first point has nothing before it";
        expect(eq(drawn[1].change.value_or(0.0), 1.5));
        expect(eq(drawn[2].change.value_or(0.0), -0.5)) << "this PR's change against the newest main point";
        expect(!drawn[1].newTools && !drawn[2].newTools);
    };

    "a change between two recorded figures is their decimal difference, not its binary rounding"_test = [] {
        const std::vector<Row>        rows{Row{{"commit", "c1"}, {"date", "2026-10-03T08:25:32+02:00"}, {"tool", "t"}, {"total", "73.81"}}, Row{{"commit", "c2"}, {"date", "2026-10-05T14:39:12+02:00"}, {"tool", "t"}, {"total", "74.01"}}};
        const std::vector<ChartPoint> drawn = chartPoints(rows, "total", std::nullopt, "", "t", 100, 365LL * 86'400);
        expect(eq(drawn[1].change.value_or(0.0), 0.2));
        expect(chart("line coverage", drawn, "%").contains("bar [0, 0.2]"));
    };

    "a change of tools marks the first point measured with the new ones and draws no change across it"_test = [] {
        const std::vector<Row>        rows{Row{{"commit", "c1"}, {"date", "2026-10-01T00:00:00+00:00"}, {"tool", "llvm-cov-20/1"}, {"total", "73.8"}}, Row{{"commit", "c2"}, {"date", "2026-10-02T00:00:00+00:00"}, {"tool", "llvm-cov-20/1"}, {"total", "74.0"}}};
        const std::vector<ChartPoint> drawn = chartPoints(rows, "total", 76.1, "2026-10-03T00:00:00+00:00", "9e1f0c2a7b", 100, 365LL * 86'400);
        expect(eq(drawn.size(), 3UZ)) << "the line continues across the change";
        expect(!drawn[1].newTools);
        expect(drawn[2].newTools) << "this run measures with other tools";
        expect(!drawn[2].change.has_value()) << "a change of method is no change of the code";
    };

    "a chart shows at most the given number of points, this run included, and nothing older than its window"_test = [] {
        std::vector<Row> rows;
        for (int day = 1; day <= 9; ++day) {
            rows.push_back(Row{{"commit", std::format("c{}", day)}, {"date", std::format("2026-01-0{}T00:00:00+00:00", day)}, {"tool", "t"}, {"total", std::to_string(day)}});
        }
        const std::vector<ChartPoint> lastThree = chartPoints(rows, "total", 10.0, "2026-01-10T00:00:00+00:00", "t", 3, 365LL * 86'400);
        expect(eq(lastThree.size(), 3UZ));
        expect(eq(lastThree.front().value, 8.0)) << "the two newest commits, then this run";
        const std::vector<ChartPoint> lastFourDays = chartPoints(rows, "total", 10.0, "2026-01-10T00:00:00+00:00", "t", 100, 4LL * 86'400);
        expect(eq(lastFourDays.front().value, 6.0)) << "day 6 is four days before day 10";
        expect(eq(lastFourDays.size(), 5UZ));
        expect(chartPoints({}, "total", std::nullopt, "", "t", 100, 365LL * 86'400).empty());
    };

    "a chart draws the figure and the change per merge, each by commit and by commit date, collapsed below the table"_test = [] {
        const std::vector<ChartPoint> drawn{{.label = "#850", .date = "2026-10-01 08:00", .value = 70.0, .change = std::nullopt, .newTools = false}, {.label = "this", .date = "2026-10-02 09:30", .value = 71.5, .change = 1.5, .newTools = true}};
        const std::string             text = chart("line coverage", drawn, "%");
        expect(text.starts_with("<details><summary>📈 line coverage</summary>"));
        expect(eq(std::ranges::distance(std::views::split(text, std::string_view{"```mermaid"})) - 1, 4)) << "four charts";
        expect(!text.contains("``````")) << "each chart its own fenced block";
        expect(text.contains(R"(x-axis ["#850", "this †"])")) << "by commit, the new tools marked";
        expect(text.contains(R"(x-axis ["2026-10-01 08:00", "2026-10-02 09:30 †"])")) << "by commit date";
        expect(text.contains("line [70, 71.5]"));
        expect(text.contains("bar [0, 1.5]")) << "the first point has no change before it";
        expect(text.contains("y-axis \"Δ %\""));
        expect(text.contains("† measured with other tools"));
        expect(text.contains("    width: 1200\n    height: 400\n")) << "3:1, wider than 16:9";
        expect(chart("t", {}, "%").empty());
    };

    "a settings file's lines become words per key, comments and blank lines are dropped, a later key wins"_test = [] {
        const std::expected<SettingWords, std::string> read = settingsIn("# downstream\nmarker = gr-digitizers\n\nnamingArguments = --root  blocklib\t--exclude blocklib/test # trailing\nwasmModules =\nmarker = again\n");
        expect(read.has_value());
        expect(eq(read->size(), 3UZ));
        expect(read->at("marker") == std::vector<std::string>{"again"});
        expect(read->at("namingArguments") == std::vector<std::string>{"--root", "blocklib", "--exclude", "blocklib/test"});
        expect(read->at("wasmModules").empty()) << "an empty list is a setting, not an absent one";
        expect(settingsIn("").value().empty());
    };

    "a settings line without a key or an equals sign is refused with its line number"_test = [] {
        expect(eq(settingsIn("marker = x\nnot a setting\n").error(), std::string{"line 2: 'not a setting' is no 'key = value'"}));
        expect(!settingsIn("= value").has_value());
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
