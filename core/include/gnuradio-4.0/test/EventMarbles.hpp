#ifndef GNURADIO_TEST_EVENTMARBLES_HPP
#define GNURADIO_TEST_EVENTMARBLES_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>

namespace gr::testing {

/**
 * A marble diagram for event streams, drawn the way rxmarbles.com draws one.
 *
 * An assertion count says a trigger fired; it does not show *when*, nor how the inputs stood in relation to the
 * output. A row per stream on a shared time axis does, and it is the first thing worth looking at when a condition
 * behaves unexpectedly — so a unit test prints one beside the assertions it explains.
 *
 * Time runs left to right. Each event is a letter, coloured and lettered by the condition it carries, so the drawing
 * survives being copied into a log where the colour is lost. Events sharing a column stack upwards; past four, the
 * three earliest are drawn and the rest become an ellipsis. An operator row names the condition between the inputs
 * above it and the output below, as the website does with its box.
 *
 * @code
 * MarbleDiagram diagram{"coincidence, 50 ns window"};
 * diagram.row("ch0").at(1'000'000'000U, "start");
 * diagram.row("ch1").at(1'000'000'030U, "start");
 * diagram.condition("Coincidence(all, +/-50 ns)");
 * diagram.row("out").at(1'000'000'000U, "coincidence");
 * diagram.print();
 * @endcode
 */
struct MarbleDiagram {
    /// the ANSI codes of `gr::graphs::Color`, repeated here so a core test header need not depend on `algorithm`;
    /// the order matches ImChart's cycle, which starts at blue
    constexpr static std::array<std::string_view, 8UZ> kPalette{"\x1B[34m", "\x1B[31m", "\x1B[32m", "\x1B[33m", "\x1B[35m", "\x1B[36m", "\x1B[94m", "\x1B[91m"};
    constexpr static std::string_view                  kReset = "\x1B[39m";
    constexpr static std::string_view                  kFaint = "\x1B[90m";
    /// a letter per condition as well as a colour, so the diagram still reads where colour is stripped
    [[nodiscard]] constexpr static char letterFor(std::size_t index) noexcept { return static_cast<char>('A' + static_cast<char>(index % 26UZ)); }
    /// how many events sharing one column are drawn before the rest are elided
    constexpr static std::size_t      kStackDepth = 4UZ;
    constexpr static std::string_view kEllipsis   = "…";
    /// what rxmarbles puts at the end of a stream: a bar for a clean finish, a cross for one that failed
    constexpr static std::string_view kCompletion = "│";
    constexpr static std::string_view kError      = "✖";

    struct Mark {
        std::uint64_t at;
        std::string   name;
    };

    struct Row {
        std::string       label;
        std::vector<Mark> marks;
        std::string       condition; // set on an operator row, which carries no marks
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

        /// draws the completion bar rxmarbles puts at the end of a finished stream
        Row& completes() {
            ended = true;
            return *this;
        }

        /// draws the cross rxmarbles puts where a stream ends in an error, with what it failed at
        Row& fails(std::string why = {}) {
            errored = true;
            reason  = std::move(why);
            return *this;
        }
    };

    std::string title;
    std::size_t width  = 72UZ;
    bool        colour = true;
    std::string unit   = "ns"; // what the marks are placed by: nanoseconds, or a sample index where a stream is undated
    /// a deque, not a vector: `row()` hands out a reference that a caller keeps while adding further rows, and a
    /// vector would move the rows out from under it
    std::deque<Row> rows;

    /// a constructor rather than aggregate initialisation: `{title}` alone is a missing-field initialiser, which
    /// clang rejects under -Werror
    explicit MarbleDiagram(std::string diagramTitle) : title(std::move(diagramTitle)) {}

    [[nodiscard]] Row& row(std::string label) {
        rows.push_back(Row{.label = std::move(label), .marks = {}, .condition = {}, .ended = false, .errored = false, .reason = {}});
        return rows.back();
    }

    /// the operator between the streams above and below it
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
        const bool          instant = first == last; // everything at one moment: a spread axis would be a lie
        const std::uint64_t span    = instant ? 1U : last - first;

        std::map<std::string, std::size_t> order; // a condition keeps one colour and shape throughout
        for (const Row& line : rows) {
            for (const Mark& mark : line.marks) {
                order.emplace(mark.name, order.size());
            }
        }

        std::string out = std::format("\n{}\n", title);
        out += axis(labelWidth, first, last, instant);
        for (const Row& line : rows) {
            out += line.condition.empty() ? lane(line, labelWidth, first, span, instant, order) : box(line.condition, labelWidth);
        }
        out += legend(labelWidth, order);
        return out;
    }

    void print() const { std::print("{}", render()); }

private:
    [[nodiscard]] std::string_view tint(std::size_t index) const noexcept { return colour ? kPalette[index % kPalette.size()] : std::string_view{}; }
    [[nodiscard]] std::string_view untint() const noexcept { return colour ? kReset : std::string_view{}; }
    [[nodiscard]] std::string_view faint() const noexcept { return colour ? kFaint : std::string_view{}; }

    /// `std::string`'s fill constructor counts bytes, and a box-drawing character is three of them
    [[nodiscard]] static std::string repeat(std::string_view glyph, std::size_t times) {
        std::string out;
        out.reserve(glyph.size() * times);
        for (std::size_t i = 0UZ; i < times; ++i) {
            out += glyph;
        }
        return out;
    }

    [[nodiscard]] std::string axis(std::size_t labelWidth, std::uint64_t first, std::uint64_t last, bool instant) const {
        if (instant) {
            return std::format("{:<{}} {}{}{} one instant, {} {}\n", "time", labelWidth, faint(), repeat("\u2500", width), untint(), first, unit);
        }
        return std::format("{:<{}} {}{}{} {} .. {} {}\n", "time", labelWidth, faint(), repeat("\u2500", width), untint(), first, last, unit);
    }

    /// Events sharing a column stack upwards, the line itself carrying the first. Beyond `kStackDepth` the three
    /// earliest are drawn and the remainder become an ellipsis, so a busy instant says how busy it was.
    [[nodiscard]] std::string lane(const Row& line, std::size_t labelWidth, std::uint64_t first, std::uint64_t span, bool instant, const std::map<std::string, std::size_t>& order) const {
        std::vector<std::vector<std::size_t>> stack(width);
        for (const Mark& mark : line.marks) {
            const std::size_t column = instant ? 0UZ : static_cast<std::size_t>((static_cast<double>(mark.at - first) / static_cast<double>(span)) * static_cast<double>(width - 2UZ));
            stack[std::min(column, width - 1UZ)].push_back(order.at(mark.name));
        }

        std::size_t depth = 1UZ;
        for (const std::vector<std::size_t>& column : stack) {
            depth = std::max(depth, std::min(column.size(), kStackDepth));
        }

        std::string out;
        for (std::size_t level = depth; level-- > 0UZ;) { // topmost first, the line itself last
            out += tier(line, stack, level, labelWidth);
        }
        return out;
    }

    [[nodiscard]] std::string tier(const Row& line, const std::vector<std::vector<std::size_t>>& stack, std::size_t level, std::size_t labelWidth) const {
        constexpr std::size_t kNothing   = std::numeric_limits<std::size_t>::max();
        constexpr std::size_t kElided    = kNothing - 1UZ;
        const bool            isBaseline = level == 0UZ;

        std::string lane;
        std::size_t active = kNothing - 2UZ; // neither a condition's colour nor the line's
        for (const std::vector<std::size_t>& column : stack) {
            std::size_t wanted = kNothing;
            if (column.size() > kStackDepth && level == kStackDepth - 1UZ) {
                wanted = kElided; // the three below this are drawn; everything above them is one mark
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
                lane += isBaseline ? "─" : " "; // only the baseline draws the stream's own line
            } else {
                lane += letterFor(wanted);
            }
        }
        if (isBaseline) {
            lane += line.errored ? tint(1UZ) : faint(); // the palette's second colour is red, as ImChart cycles it
            lane += line.errored ? kError : line.ended ? kCompletion : std::string_view{"▶"};
            if (line.errored && !line.reason.empty()) {
                lane += std::format(" {}", line.reason);
            }
        }
        lane += untint();
        return std::format("{:<{}} {}\n", isBaseline ? line.label : std::string{}, labelWidth, lane);
    }

    [[nodiscard]] std::string box(std::string_view text, std::size_t labelWidth) const {
        const std::size_t inner = std::min(width, text.size() + 2UZ);
        const std::string rule  = repeat("─", inner);
        std::string       out   = std::format("{:<{}} {}┌{}┐{}\n", "", labelWidth, faint(), rule, untint());
        out += std::format("{:<{}} {}│{} {} {}│{}\n", "", labelWidth, faint(), untint(), text, faint(), untint());
        out += std::format("{:<{}} {}└{}┘{}\n", "", labelWidth, faint(), rule, untint());
        return out;
    }

    [[nodiscard]] std::string legend(std::size_t labelWidth, const std::map<std::string, std::size_t>& order) const {
        if (order.empty()) {
            return {};
        }
        std::vector<std::pair<std::size_t, std::string>> byOrder;
        byOrder.reserve(order.size());
        for (const auto& [name, index] : order) {
            byOrder.emplace_back(index, name);
        }
        std::ranges::sort(byOrder);

        std::string out = std::format("{:<{}} ", "", labelWidth);
        for (const auto& [index, name] : byOrder) {
            out += std::format("{}{}{} {}   ", tint(index), letterFor(index), untint(), name);
        }
        return out + "\n";
    }
};

} // namespace gr::testing

#endif // GNURADIO_TEST_EVENTMARBLES_HPP
