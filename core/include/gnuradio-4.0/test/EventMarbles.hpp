#ifndef GNURADIO_TEST_EVENTMARBLES_HPP
#define GNURADIO_TEST_EVENTMARBLES_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>

namespace gr::testing {

/**
 * A marble diagram for event streams, drawn the way rxmarbles.com draws one: a row per stream on a shared time axis shows when a condition fired and how the inputs stood to the output.
 * Events are letters coloured and lettered by their condition, so the drawing survives a log without colour; events sharing a column stack upwards (past four: the three earliest, then an ellipsis).
 * An operator row names the condition between the inputs above it and the output below.
 *
 *   MarbleDiagram diagram{"coincidence"}; diagram.row("ch0").at(1'000'000'000U, "start"); diagram.condition("Coincidence(all, +/-50 ns)"); diagram.print();
 */
struct MarbleDiagram {
    /** the ANSI codes of `gr::graphs::Color`, repeated here so a core test header need not depend on `algorithm`; the order matches ImChart's cycle, which starts at blue */
    constexpr static std::array<std::string_view, 8UZ> kPalette{"\x1B[34m", "\x1B[31m", "\x1B[32m", "\x1B[33m", "\x1B[35m", "\x1B[36m", "\x1B[94m", "\x1B[91m"};
    constexpr static std::string_view                  kReset = "\x1B[39m";
    constexpr static std::string_view                  kFaint = "\x1B[90m";
    [[nodiscard]] constexpr static char                letterFor(std::size_t index) noexcept { return static_cast<char>('A' + static_cast<char>(index % 26UZ)); }
    constexpr static std::size_t                       kStackDepth = 4UZ;
    constexpr static std::string_view                  kEllipsis   = "…";
    constexpr static std::string_view                  kCompletion = "│";
    constexpr static std::string_view                  kError      = "✖";

    struct Mark {
        std::uint64_t at;
        std::string   name;
    };

    struct Row {
        std::string       label;
        std::vector<Mark> marks;
        std::string       condition;
        bool              ended   = false;
        bool              errored = false;
        std::string       reason;

        Row& at(std::uint64_t timeNs, std::string name) {
            marks.push_back(Mark{.at = timeNs, .name = std::move(name)});
            return *this;
        }

        Row& all(std::span<const std::pair<std::uint64_t, std::string>> events) {
            for (const auto& [timeNs, name] : events) {
                at(timeNs, name);
            }
            return *this;
        }

        Row& completes() {
            ended = true;
            return *this;
        }

        Row& fails(std::string why = {}) {
            errored = true;
            reason  = std::move(why);
            return *this;
        }
    };

    std::string title;
    std::size_t width  = 72UZ;
    bool        colour = true;
    std::string unit   = "ns"; /**< what the marks are placed by: nanoseconds, or a sample index where a stream is undated */
    /** a deque, not a vector: `row()` hands out a reference that a caller keeps while adding further rows, and a vector would move the rows out from under it */
    std::deque<Row> rows;

    /** a constructor rather than aggregate initialisation: `{title}` alone is a missing-field initialiser, which clang rejects under -Werror */
    explicit MarbleDiagram(std::string diagramTitle) : title(std::move(diagramTitle)) {}

    [[nodiscard]] Row& row(std::string label) {
        rows.push_back(Row{.label = std::move(label), .marks = {}, .condition = {}, .ended = false, .errored = false, .reason = {}});
        return rows.back();
    }

    void condition(std::string text) { rows.push_back(Row{.label = {}, .marks = {}, .condition = std::move(text), .ended = false, .errored = false, .reason = {}}); }

    [[nodiscard]] std::string render() const {
        std::uint64_t first      = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t last       = 0U;
        std::size_t   labelWidth = 4UZ;
        for (const Row& line : rows) {
            labelWidth = std::max(labelWidth, line.label.size());
            for (const Mark& mark : line.marks) {
                first = std::min(first, mark.at);
                last  = std::max(last, mark.at);
            }
        }
        if (first > last) {
            return std::format("{}: no events\n", title);
        }
        const bool          instant = first == last;
        const std::uint64_t span    = instant ? 1U : last - first;

        std::vector<std::string> names; // conditions in first-seen order: the index is the letter and the colour
        for (const Row& line : rows) {
            for (const Mark& mark : line.marks) {
                if (!std::ranges::contains(names, mark.name)) {
                    names.push_back(mark.name);
                }
            }
        }

        const auto axis = [&] {
            const std::string rule = repeat("\u2500", width);
            if (instant) {
                return std::format("{:<{}} {}{}{} one instant, {} {}\n", "time", labelWidth, faint(), rule, untint(), first, unit);
            }
            return std::format("{:<{}} {}{}{} {} .. {} {}\n", "time", labelWidth, faint(), rule, untint(), first, last, unit);
        };
        const auto box = [&](std::string_view text) {
            const std::size_t inner = std::min(width, text.size() + 2UZ);
            const std::string rule  = repeat("─", inner);
            std::string       lines = std::format("{:<{}} {}┌{}┐{}\n", "", labelWidth, faint(), rule, untint());
            lines += std::format("{:<{}} {}│{} {} {}│{}\n", "", labelWidth, faint(), untint(), text, faint(), untint());
            lines += std::format("{:<{}} {}└{}┘{}\n", "", labelWidth, faint(), rule, untint());
            return lines;
        };
        const auto legend = [&] {
            if (names.empty()) {
                return std::string{};
            }
            std::string lines = std::format("{:<{}} ", "", labelWidth);
            for (std::size_t index = 0UZ; index < names.size(); ++index) {
                lines += std::format("{}{}{} {}   ", tint(index), letterFor(index), untint(), names[index]);
            }
            return lines + "\n";
        };

        std::string out = std::format("\n{}\n", title);
        out += axis();
        for (const Row& line : rows) {
            out += line.condition.empty() ? lane(line, labelWidth, first, span, instant, names) : box(line.condition);
        }
        out += legend();
        return out;
    }

    void print() const { std::print("{}", render()); }

private:
    [[nodiscard]] std::string_view tint(std::size_t index) const noexcept { return colour ? kPalette[index % kPalette.size()] : std::string_view{}; }
    [[nodiscard]] std::string_view untint() const noexcept { return colour ? kReset : std::string_view{}; }
    [[nodiscard]] std::string_view faint() const noexcept { return colour ? kFaint : std::string_view{}; }

    /** `std::string`'s fill constructor counts bytes, and a box-drawing character is three of them */
    [[nodiscard]] static std::string repeat(std::string_view glyph, std::size_t times) { return std::views::repeat(glyph, times) | std::views::join | std::ranges::to<std::string>(); }

    [[nodiscard]] std::string lane(const Row& line, std::size_t labelWidth, std::uint64_t first, std::uint64_t span, bool instant, const std::vector<std::string>& names) const {
        std::vector<std::vector<std::size_t>> stack(width);
        for (const Mark& mark : line.marks) {
            const double      fraction = static_cast<double>(mark.at - first) / static_cast<double>(span);
            const std::size_t column   = instant ? 0UZ : static_cast<std::size_t>(fraction * static_cast<double>(width - 2UZ));
            const std::size_t index    = static_cast<std::size_t>(std::ranges::find(names, mark.name) - names.begin());
            stack[std::min(column, width - 1UZ)].push_back(index);
        }

        std::size_t depth = 1UZ;
        for (const std::vector<std::size_t>& column : stack) {
            depth = std::max(depth, std::min(column.size(), kStackDepth));
        }

        std::string out;
        for (std::size_t level = depth; level-- > 0UZ;) {
            out += tier(line, stack, level, labelWidth);
        }
        return out;
    }

    [[nodiscard]] std::string tier(const Row& line, const std::vector<std::vector<std::size_t>>& stack, std::size_t level, std::size_t labelWidth) const {
        constexpr std::size_t kNothing   = std::numeric_limits<std::size_t>::max();
        constexpr std::size_t kElided    = kNothing - 1UZ;
        const bool            isBaseline = level == 0UZ;

        std::string lane;
        std::size_t active = kNothing - 2UZ;
        for (const std::vector<std::size_t>& column : stack) {
            std::size_t wanted = kNothing;
            if (column.size() > kStackDepth && level == kStackDepth - 1UZ) {
                wanted = kElided;
            } else if (level < column.size()) {
                wanted = column[level];
            }

            if (wanted != active) {
                lane += wanted == kNothing || wanted == kElided ? faint() : tint(wanted);
                active = wanted;
            }
            if (wanted == kElided) {
                lane += kEllipsis;
            } else if (wanted == kNothing) {
                lane += isBaseline ? "─" : " ";
            } else {
                lane += letterFor(wanted);
            }
        }
        if (isBaseline) {
            lane += line.errored ? tint(1UZ) : faint();
            lane += line.errored ? kError : line.ended ? kCompletion : std::string_view{"▶"};
            if (line.errored && !line.reason.empty()) {
                lane += std::format(" {}", line.reason);
            }
        }
        lane += untint();
        return std::format("{:<{}} {}\n", isBaseline ? line.label : std::string{}, labelWidth, lane);
    }
};

} // namespace gr::testing

#endif // GNURADIO_TEST_EVENTMARBLES_HPP
