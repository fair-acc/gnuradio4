#ifndef GNURADIO_DEVTOOLS_QUALITY_BOT_HPP
#define GNURADIO_DEVTOOLS_QUALITY_BOT_HPP

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// the quality bots' pure parts (devtools/qualitybot.cpp), each with its ground truth in qa_QualityBot.cpp
namespace devtools::bot {

using Row = std::map<std::string, std::string, std::less<>>;

inline constexpr std::string_view kGreen  = "🟢";
inline constexpr std::string_view kYellow = "🟡";
inline constexpr std::string_view kRed    = "🔴";

[[nodiscard]] inline std::vector<std::string> linesOf(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t              start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        lines.emplace_back(text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return lines;
}

[[nodiscard]] inline std::string trimmed(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::string{text.substr(first, last - first + 1)};
}

[[nodiscard]] inline std::string joined(std::ranges::input_range auto&& words, std::string_view separator) { return words | std::views::join_with(separator) | std::ranges::to<std::string>(); }

// "76.9 %", "1,234" -> the number; anything else -> none, so a missing value never reads as zero
[[nodiscard]] inline std::optional<double> numberIn(std::string_view text) {
    std::string digits;
    for (const char character : text) {
        if (character != '%' && character != ',' && character != ' ') {
            digits += character;
        }
    }
    double value            = 0.0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (digits.empty() || error != std::errc{} || end != digits.data() + digits.size()) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] inline std::string shortest(double value) { return std::format("{:.15g}", value); }

[[nodiscard]] inline std::string withThousands(long long value) {
    std::string       digits = std::to_string(value < 0 ? -value : value);
    std::string       grouped;
    const std::size_t lead = digits.size() % 3U == 0U ? 3U : digits.size() % 3U;
    for (std::size_t i = 0; i < digits.size(); ++i) {
        if (i >= lead && (i - lead) % 3U == 0U) {
            grouped += ',';
        }
        grouped += digits[i];
    }
    return value < 0 ? "-" + grouped : grouped;
}

[[nodiscard]] inline std::string plural(std::size_t count, std::string_view noun) { return std::format("{} {}{}", count, noun, count == 1U ? "" : "s"); }

// a change in percentage points: ±0.0 within ±0.05, so rounding noise never reads as a change
[[nodiscard]] inline std::string signedChange(std::optional<double> now, std::optional<double> before) {
    if (!now || !before) {
        return "n/a";
    }
    const double change = *now - *before;
    return std::abs(change) < 0.05 ? "±0.0" : std::format("{:+.1f}", change);
}

[[nodiscard]] inline std::string cutUtf8(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) {
        return std::string{text};
    }
    std::size_t end = limit;
    while (end > 0U && (static_cast<unsigned char>(text[end]) & 0xC0U) == 0x80U) {
        --end;
    }
    return std::string{text.substr(0, end)};
}

[[nodiscard]] inline std::string jsonQuoted(std::string_view text) {
    std::string quoted = "\"";
    for (const char character : text) {
        switch (character) {
        case '"': quoted += "\\\""; break;
        case '\\': quoted += "\\\\"; break;
        case '\n': quoted += "\\n"; break;
        case '\t': quoted += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20U) {
                quoted += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(character)));
            } else {
                quoted += character;
            }
        }
    }
    return quoted + "\"";
}

[[nodiscard]] inline std::vector<std::vector<std::string>> csvRecords(std::string_view text) {
    std::vector<std::vector<std::string>> records;
    std::vector<std::string>              record;
    std::string                           field;
    bool                                  quoted = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char character = text[i];
        if (quoted) {
            if (character == '"' && i + 1 < text.size() && text[i + 1] == '"') {
                field += '"';
                ++i;
            } else if (character == '"') {
                quoted = false;
            } else {
                field += character;
            }
        } else if (character == '"') {
            quoted = true;
        } else if (character == ',') {
            record.push_back(std::move(field));
            field.clear();
        } else if (character == '\n') {
            record.push_back(std::move(field));
            field.clear();
            records.push_back(std::move(record));
            record.clear();
        } else if (character != '\r') {
            field += character;
        }
    }
    if (!field.empty() || !record.empty()) {
        record.push_back(std::move(field));
        records.push_back(std::move(record));
    }
    return records;
}

struct Table {
    std::vector<std::string> columns;
    std::vector<Row>         rows;
};

[[nodiscard]] inline Table csvTable(std::string_view text) {
    std::vector<std::vector<std::string>> records = csvRecords(text);
    Table                                 table;
    if (records.empty()) {
        return table;
    }
    table.columns = records.front();
    for (const std::vector<std::string>& record : records | std::views::drop(1)) {
        Row row;
        for (std::size_t k = 0; k < table.columns.size(); ++k) {
            row[table.columns[k]] = k < record.size() ? record[k] : std::string{};
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

[[nodiscard]] inline std::string csvField(std::string_view value) {
    if (value.find_first_of(",\"\n\r") == std::string_view::npos) {
        return std::string{value};
    }
    std::string quoted = "\"";
    for (const char character : value) {
        quoted += character == '"' ? std::string{"\"\""} : std::string(1, character);
    }
    return quoted + "\"";
}

[[nodiscard]] inline std::string csvText(const Table& table) {
    std::string text;
    const auto  line = [&text](const auto& fields) {
        bool first = true;
        for (const auto& field : fields) {
            text += (first ? "" : ",") + csvField(field);
            first = false;
        }
        text += '\n';
    };
    line(table.columns);
    for (const Row& row : table.rows) {
        std::vector<std::string> fields;
        for (const std::string& column : table.columns) {
            const auto value = row.find(column);
            fields.push_back(value == row.end() ? std::string{} : value->second);
        }
        line(fields);
    }
    return text;
}

// from `git diff -U0`
[[nodiscard]] inline std::map<std::string, std::set<unsigned>, std::less<>> changedLinesIn(std::string_view diff) {
    std::map<std::string, std::set<unsigned>, std::less<>> changed;
    std::optional<std::string>                             current;
    for (const std::string& line : linesOf(diff)) {
        if (line.starts_with("+++ ")) {
            current = line.starts_with("+++ b/") ? std::optional<std::string>{line.substr(6)} : std::nullopt;
            continue;
        }
        if (!line.starts_with("@@") || !current) {
            continue;
        }
        const std::size_t plus = line.find(" +");
        if (plus == std::string::npos) {
            continue;
        }
        unsigned    first  = 0;
        unsigned    count  = 1;
        const char* begin  = line.data() + plus + 2;
        const char* end    = line.data() + line.size();
        const auto  parsed = std::from_chars(begin, end, first);
        if (parsed.ptr != end && *parsed.ptr == ',') {
            std::from_chars(parsed.ptr + 1, end, count);
        }
        for (unsigned number = first; number < first + count; ++number) {
            changed[*current].insert(number);
        }
    }
    return changed;
}

// `L3, L5–L9` for a set of line numbers
[[nodiscard]] inline std::string rangesOf(const std::set<unsigned>& numbers) {
    std::vector<std::pair<unsigned, unsigned>> spans;
    for (const unsigned number : numbers) {
        if (!spans.empty() && number == spans.back().second + 1U) {
            spans.back().second = number;
        } else {
            spans.emplace_back(number, number);
        }
    }
    std::string text;
    for (const auto& [low, high] : spans) {
        text += std::format("{}{}", text.empty() ? "" : ", ", low == high ? std::format("L{}", low) : std::format("L{}–L{}", low, high));
    }
    return text;
}

struct Base {
    std::string      commit; // the main commit the change lands on, empty when there is none
    Row              row;    // the bot's figures there, empty when not recorded or not comparable
    std::vector<Row> rows;   // main's history up to it, oldest first, of every tool
    std::string      note;
};

// the nearest recorded first-parent ancestor of `commit` stands in for it, named in the note; figures of other tools do not
// compare, but their history still charts
[[nodiscard]] inline Base baseFrom(std::string_view commit, const std::vector<std::string>& firstParentAncestors, const std::vector<Row>& history, std::string_view toolVersion) {
    std::map<std::string, std::size_t, std::less<>> indexOf;
    for (std::size_t k = 0; k < history.size(); ++k) {
        indexOf[history[k].at("commit")] = k;
    }
    const auto recorded = std::ranges::find_if(firstParentAncestors, [&indexOf](const std::string& sha) { return indexOf.contains(sha); });
    if (recorded == firstParentAncestors.end()) {
        return {.commit = std::string{commit}, .row = {}, .rows = {}, .note = "not recorded yet"};
    }
    const std::size_t index  = indexOf.at(*recorded);
    const Row&        row    = history[index];
    const std::string short8 = recorded->substr(0, 8);
    std::vector<Row>  rows(history.begin(), history.begin() + static_cast<std::ptrdiff_t>(index + 1U));
    if (row.at("tool") != toolVersion) {
        return {.commit = std::string{commit}, .row = {}, .rows = std::move(rows), .note = std::format("tools changed since `{}`: not comparable", short8)};
    }
    return {.commit = std::string{commit}, .row = row, .rows = std::move(rows), .note = *recorded == commit ? std::string{} : std::format("not recorded; compared with `{}`", short8)};
}

// `  name: value (meaning)` lines of a code-metrics --list report, in their order; the places listed beneath a figure
// are indented further and are not figures, even when their key holds a `: <number>`
[[nodiscard]] inline std::vector<std::pair<std::string, std::string>> figureLines(std::string_view report) {
    std::vector<std::pair<std::string, std::string>> figures;
    const std::regex                                 pattern{R"(^  ([^ :][^:]*): ([0-9][0-9.,]*(?: %)?))"};
    for (const std::string& line : linesOf(report)) {
        std::smatch found;
        if (std::regex_search(line, found, pattern)) {
            figures.emplace_back(found[1].str(), found[2].str());
        }
    }
    return figures;
}

[[nodiscard]] inline std::optional<long long> epochSecondsOf(std::string_view iso) {
    int        year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, offsetHours = 0, offsetMinutes = 0;
    const auto field = [iso](std::size_t at, std::size_t width, int& value) { return at + width <= iso.size() && std::from_chars(iso.data() + at, iso.data() + at + width, value).ec == std::errc{}; };
    if (!field(0, 4, year) || !field(5, 2, month) || !field(8, 2, day) || !field(11, 2, hour) || !field(14, 2, minute) || !field(17, 2, second)) {
        return std::nullopt;
    }
    int sign = 0;
    if (iso.size() >= 25U && (iso[19] == '+' || iso[19] == '-') && field(20, 2, offsetHours) && field(23, 2, offsetMinutes)) {
        sign = iso[19] == '+' ? 1 : -1;
    }
    const std::chrono::sys_days days{std::chrono::year{year} / std::chrono::month{static_cast<unsigned>(month)} / std::chrono::day{static_cast<unsigned>(day)}};
    return static_cast<long long>(days.time_since_epoch().count()) * 86400LL + hour * 3600LL + minute * 60LL + second - sign * (offsetHours * 3600LL + offsetMinutes * 60LL);
}

[[nodiscard]] inline std::string fenced(std::string_view text, std::string_view language = "") {
    const auto last = text.find_last_not_of(" \t\r\n");
    return std::format("```{}\n{}\n```", language, last == std::string_view::npos ? std::string_view{} : text.substr(0, last + 1));
}

[[nodiscard]] inline std::string details(std::string_view summary, std::string_view body) { return std::format("<details><summary>{}</summary>\n\n{}\n\n</details>\n", summary, trimmed(body)); }

struct ChartPoint {
    std::string           label; // `#<pr>` where the history names one, else the short commit; `this` for this run
    std::string           date;  // the commit's date and time in UTC, `YYYY-MM-DD hh:mm`
    double                value;
    std::optional<double> change; // against the point before, none for the first and across a change of tools
    bool                  newTools;
};

[[nodiscard]] inline std::vector<ChartPoint> chartPoints(const std::vector<Row>& rows, std::string_view figure, std::optional<double> current, std::string_view currentDate, std::string_view currentTools, std::size_t points, long long maxAgeSeconds) {
    const auto valueOf = [figure](const Row& row) -> std::optional<double> {
        const auto found = row.find(figure);
        return found == row.end() ? std::nullopt : numberIn(found->second);
    };
    const auto fieldOf = [](const Row& row, std::string_view key) {
        const auto found = row.find(key);
        return found == row.end() ? std::string{} : found->second;
    };
    const auto minuteOf = [](std::string_view iso) {
        const std::optional<long long> at = epochSecondsOf(iso);
        return at ? std::format("{:%Y-%m-%d %H:%M}", std::chrono::sys_seconds{std::chrono::seconds{*at}}) : std::string{};
    };
    struct Measured {
        const Row* row;
        double     value;
    };
    std::vector<Measured> measured;
    for (const Row& row : rows) {
        if (const std::optional<double> value = valueOf(row)) {
            measured.push_back({&row, *value});
        }
    }
    const std::string              newestDate = current ? std::string{currentDate} : measured.empty() ? std::string{} : fieldOf(*measured.back().row, "date");
    const std::optional<long long> newest     = epochSecondsOf(newestDate);
    const auto                     tooOld     = [&](const Measured& entry) {
        const std::optional<long long> at = epochSecondsOf(fieldOf(*entry.row, "date"));
        return newest && at && *newest - *at > maxAgeSeconds;
    };
    std::erase_if(measured, tooOld);
    const std::size_t historyPoints = current && points > 0U ? points - 1U : points;
    if (measured.size() > historyPoints) {
        measured.erase(measured.begin(), measured.end() - static_cast<std::ptrdiff_t>(historyPoints));
    }

    // recorded figures carry a few decimals; their difference would print binary noise (74.01 - 73.81 = 0.200000000000003)
    constexpr double        kStepsPerUnit = 1e9;
    const auto              changeOf      = [](double now, double before) { return std::round((now - before) * kStepsPerUnit) / kStepsPerUnit; };
    std::vector<ChartPoint> drawn;
    std::string             toolsBefore;
    for (const Measured& entry : measured) {
        const std::string           pr       = fieldOf(*entry.row, "pr");
        const std::string           tools    = fieldOf(*entry.row, "tool");
        const bool                  newTools = !drawn.empty() && tools != toolsBefore;
        const std::optional<double> change   = drawn.empty() || newTools ? std::nullopt : std::optional<double>{changeOf(entry.value, drawn.back().value)};
        drawn.push_back({.label = pr.empty() ? fieldOf(*entry.row, "commit").substr(0, 7) : std::format("#{}", pr), .date = minuteOf(fieldOf(*entry.row, "date")), .value = entry.value, .change = change, .newTools = newTools});
        toolsBefore = tools;
    }
    if (current) {
        const bool                  newTools = !drawn.empty() && currentTools != toolsBefore;
        const std::optional<double> change   = drawn.empty() || newTools ? std::nullopt : std::optional<double>{changeOf(*current, drawn.back().value)};
        drawn.push_back({.label = "this", .date = minuteOf(currentDate), .value = *current, .change = change, .newTools = newTools});
    }
    return drawn;
}

// four charts, not two with a second y-axis: GitHub's Mermaid draws one y-axis per chart
inline constexpr int             kChartWidth  = 1200; // px
inline constexpr int             kChartHeight = 400;  // px
[[nodiscard]] inline std::string chart(std::string_view title, const std::vector<ChartPoint>& drawn, std::string_view unit) {
    if (drawn.empty()) {
        return {};
    }
    const auto               quoted = [](std::string_view text) { return std::format("\"{}\"", text); };
    std::vector<std::string> byCommit;
    std::vector<std::string> byDate;
    std::vector<std::string> values;
    std::vector<std::string> changes;
    for (const ChartPoint& point : drawn) {
        const std::string_view marker = point.newTools ? " †" : "";
        byCommit.push_back(quoted(std::format("{}{}", point.label, marker)));
        byDate.push_back(quoted(std::format("{}{}", point.date, marker)));
        values.push_back(shortest(point.value));
        changes.push_back(shortest(point.change.value_or(0.0)));
    }
    const auto        mermaid    = [&](std::string_view chartTitle, const std::vector<std::string>& labels, std::string_view axis, std::string_view mark, const std::vector<std::string>& data) { return fenced(std::format("---\nconfig:\n  xyChart:\n    width: {}\n    height: {}\n---\nxychart-beta\n    title \"{}\"\n    x-axis [{}]\n    y-axis \"{}\"\n    {} [{}]", kChartWidth, kChartHeight, chartTitle, joined(labels, ", "), axis, mark, joined(data, ", ")), "mermaid") + "\n\n"; };
    const std::string changeAxis = std::format("Δ {}", unit);
    std::string       body       = mermaid(std::format("{} on main, by commit", title), byCommit, unit, "line", values);
    body += mermaid(std::format("change per merge, by commit (this PR last)"), byCommit, changeAxis, "bar", changes);
    body += mermaid(std::format("{} on main, by commit date (UTC)", title), byDate, unit, "line", values);
    body += mermaid(std::format("change per merge, by commit date (UTC, this PR last)"), byDate, changeAxis, "bar", changes);
    if (std::ranges::any_of(drawn, &ChartPoint::newTools)) {
        body += "\n† measured with other tools than the point before: a change of method, not of the code, so no change is drawn there\n";
    }
    return details(std::format("📈 {}", title), body);
}

} // namespace devtools::bot

#endif // GNURADIO_DEVTOOLS_QUALITY_BOT_HPP
