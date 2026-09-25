#ifndef GNURADIO_TRIGGER_MARBLE_HPP
#define GNURADIO_TRIGGER_MARBLE_HPP

#include <algorithm>
#include <expected>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/trigger/Events.hpp>

namespace gr::blocks::trigger {

inline constexpr std::string_view kMarbleTagSeparator   = "+";
inline constexpr char             kMarbleTagValueMarker = ':';
inline constexpr char             kMarbleEndOfStream    = '|';
inline constexpr char             kMarbleError          = '#'; // rxmarbles' error terminator: the stream ends badly
inline constexpr char             kMarbleNoSample       = '.';
inline constexpr char             kMarbleRowLabel       = '@';

GR_REGISTER_BLOCK(gr::blocks::trigger::MarbleSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK(gr::blocks::trigger::MarbleSink, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

struct Marble {
    std::pmr::string         value;
    std::vector<std::string> tags;
    bool                     endOfStream = false;
    bool                     error       = false;
    std::pmr::string         reason;
};

struct MarbleRow {
    std::pmr::string    label;
    std::vector<Marble> marbles;
};

struct MarbleScript {
    std::vector<MarbleRow> rows;

    [[nodiscard]] static std::expected<MarbleScript, Error> parse(std::string_view text) {
        MarbleScript score;
        for (const auto line : std::views::split(text, '\n')) {
            std::string_view row = trim(std::string_view{line.begin(), line.end()});
            if (row.empty()) {
                continue;
            }

            MarbleRow parsed;
            if (row.front() == kMarbleRowLabel) {
                const std::size_t afterLabel = row.find(' ');
                parsed.label                 = std::pmr::string(row.substr(1UZ, afterLabel == std::string_view::npos ? std::string_view::npos : afterLabel - 1UZ));
                if (parsed.label.empty()) {
                    return std::unexpected(Error(std::format("row '{}' opens with '{}' but names nothing", row, kMarbleRowLabel)));
                }
                row = afterLabel == std::string_view::npos ? std::string_view{} : row.substr(afterLabel + 1UZ);
            }
            if (auto marbles = parseRow(row); marbles) {
                parsed.marbles = std::move(marbles.value());
            } else {
                return std::unexpected(marbles.error());
            }
            score.rows.push_back(std::move(parsed));
        }
        if (score.rows.empty()) {
            score.rows.push_back(MarbleRow{});
        }
        return score;
    }

    [[nodiscard]] const std::vector<Marble>& firstRow() const noexcept { return rows.front().marbles; }

    [[nodiscard]] const MarbleRow* find(std::string_view label) const noexcept {
        const auto found = std::ranges::find_if(rows, [label](const MarbleRow& row) { return row.label == label; });
        return found == rows.end() ? nullptr : std::addressof(*found);
    }

    [[nodiscard]] std::string toString() const {
        std::string text;
        for (const MarbleRow& row : rows) {
            if (!text.empty()) {
                text.push_back('\n');
            }
            if (!row.label.empty()) {
                text.push_back(kMarbleRowLabel);
                text.append(row.label);
                text.push_back(' ');
            }
            text.append(spell(row.marbles));
        }
        return text;
    }

    [[nodiscard]] static std::string spell(std::span<const Marble> marbles) {
        std::string text;
        for (const Marble& marble : marbles) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            if (marble.endOfStream) {
                text.push_back(kMarbleEndOfStream);
                continue;
            }
            if (marble.error) {
                text.push_back(kMarbleError);
                if (!marble.reason.empty()) {
                    text.push_back(kMarbleTagValueMarker);
                    text.append(marble.reason);
                }
                continue;
            }
            for (std::size_t i = 0UZ; i < marble.tags.size(); ++i) {
                text.append(marble.tags[i]);
                text.push_back(i + 1UZ == marble.tags.size() ? kMarbleTagValueMarker : kMarbleTagSeparator[0]);
            }
            text.append(marble.value);
        }
        return text;
    }

private:
    [[nodiscard]] static std::expected<std::vector<Marble>, Error> parseRow(std::string_view text) {
        std::vector<Marble> marbles;
        for (const auto token : std::views::split(text, ' ')) {
            const std::string_view word{token.begin(), token.end()};
            if (word.empty()) {
                continue;
            }
            if (word.size() == 1UZ && word.front() == kMarbleEndOfStream) {
                Marble terminator;
                terminator.endOfStream = true;
                marbles.push_back(std::move(terminator));
                continue;
            }
            if (word.front() == kMarbleError) {
                const std::string_view reason = word.size() > 1UZ && word[1UZ] == kMarbleTagValueMarker ? word.substr(2UZ) : std::string_view{};
                if (word.size() > 1UZ && word[1UZ] != kMarbleTagValueMarker) {
                    return std::unexpected(Error(std::format("marble '{}' must be '{}' alone or '{}{}reason'", word, kMarbleError, kMarbleError, kMarbleTagValueMarker)));
                }
                Marble failure;
                failure.error  = true;
                failure.reason = std::pmr::string(reason);
                marbles.push_back(std::move(failure));
                continue;
            }

            Marble                 marble;
            const auto             marker    = word.find(kMarbleTagValueMarker);
            const std::string_view tagPart   = marker == std::string_view::npos ? std::string_view{} : word.substr(0UZ, marker);
            const std::string_view valuePart = marker == std::string_view::npos ? word : word.substr(marker + 1UZ);

            if (valuePart.empty()) {
                return std::unexpected(Error(std::format("marble '{}' names tags but no sample", word)));
            }
            if (valuePart.find(kMarbleTagValueMarker) != std::string_view::npos) {
                return std::unexpected(Error(std::format("marble '{}' carries more than one '{}'", word, kMarbleTagValueMarker)));
            }
            marble.value = std::pmr::string(valuePart);

            for (const auto tagToken : std::views::split(tagPart, kMarbleTagSeparator[0])) {
                const std::string_view tagSymbol{tagToken.begin(), tagToken.end()};
                if (tagSymbol.empty()) {
                    if (tagPart.empty()) {
                        continue;
                    }
                    return std::unexpected(Error(std::format("marble '{}' has an empty tag symbol", word)));
                }
                marble.tags.emplace_back(tagSymbol);
            }
            marbles.push_back(std::move(marble));
        }
        return marbles;
    }

    [[nodiscard]] static std::string_view trim(std::string_view text) noexcept {
        constexpr auto isSpace = [](char c) noexcept { return c == ' ' || c == '\r' || c == '\t'; };
        const auto     first   = std::ranges::find_if_not(text, isSpace);
        if (first == text.end()) {
            return {};
        }
        const auto last = std::ranges::find_if_not(text | std::views::reverse, isSpace).base();
        return std::string_view{first, last};
    }
};

template<typename T>
struct MarbleSource : gr::Block<MarbleSource<T>> {
    using Description = Doc<R"(@brief emit the samples a marble script describes, tagging them where it says to

    script  "@ch0 a b T:c d |"       the notation RxMarbles draws, written down
    out     ─a──b──T:c──d──│         '|' ends the stream, '#' ends it in an error

One line per stream, each nameable with `@`, and `row` picks the line this source plays -- so one script describes every input
of a combination operator.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};
    gr::PortOut<T>   out;

    A<std::pmr::string, "script", Doc<"marble notation: 'a T:b c |'">>            script;
    A<std::pmr::string, "row", Doc<"which row to play, empty = first">>           row;
    A<gr::property_map, "sample values", Doc<"value symbol -> sample">>           sample_values;
    A<gr::property_map, "sample tags", Doc<"tag symbol -> tag published there">>  sample_tags;
    A<gr::Size_t, "repeat", Doc<"how many times the script is played, 0 = once">> repeat = 1U;

    GR_MAKE_REFLECTABLE(MarbleSource, evtOut, out, script, row, sample_values, sample_tags, repeat);

    std::vector<Marble>      _marbles;
    std::size_t              _position        = 0UZ;
    gr::Size_t               _playsLeft       = 1U;
    std::size_t              _eventsPublished = 0UZ;
    std::vector<std::string> _pendingErrors;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& /*newSettings*/) {
        _marbles.clear();
        if (auto parsed = MarbleScript::parse(script); !parsed) {
            _pendingErrors.push_back(parsed.error().message);
            gr::log::warning("MarbleSource: {}", parsed.error().message);
        } else if (row.value.empty()) {
            _marbles = parsed->firstRow();
        } else if (const MarbleRow* named = parsed->find(row.value); named != nullptr) {
            _marbles = named->marbles;
        } else {
            _pendingErrors.push_back(std::format("script names no row '{}'", row.value));
            gr::log::warning("MarbleSource: {}", _pendingErrors.back());
        }
        _position  = 0UZ;
        _playsLeft = repeat == 0U ? 1U : repeat.value;
    }

    gr::work::Status processBulk(gr::OutputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) {
        std::size_t produced = 0UZ;
        drainPendingErrors(evtSpan);
        while (produced < outSpan.size() && _position < _marbles.size()) {
            const Marble& marble = _marbles[_position];
            if (marble.endOfStream || marble.error) {
                if (marble.error) {
                    report(evtSpan, marble.reason.empty() ? std::pmr::string("the script ends in an error") : marble.reason);
                }
                ++_position;
                outSpan.publish(produced);
                evtSpan.publish(_eventsPublished);
                this->requestStop();
                return gr::work::Status::DONE;
            }

            const auto sample = sample_values.value.template get_if<T>(std::string_view{marble.value});
            if (!sample) {
                report(evtSpan, std::format("marble '{}' names a sample the settings do not define", marble.value));
                ++_position;
                continue;
            }
            for (const std::string& tagSymbol : marble.tags) {
                const auto tag = sample_tags.value.template get_if<gr::property_map>(std::string_view{tagSymbol});
                if (!tag) {
                    report(evtSpan, std::format("marble tag '{}' is not defined", tagSymbol));
                    continue;
                }
                outSpan.publishTag(*tag, produced);
            }
            outSpan[produced] = *sample;
            ++produced;
            ++_position;
        }

        if (_position >= _marbles.size()) {
            if (_playsLeft > 1U) {
                --_playsLeft;
                _position = 0UZ;
            } else {
                outSpan.publish(produced);
                evtSpan.publish(_eventsPublished);
                this->requestStop();
                return gr::work::Status::DONE;
            }
        }
        outSpan.publish(produced);
        evtSpan.publish(_eventsPublished);
        return gr::work::Status::OK;
    }

    void report(gr::OutputSpanLike auto& evtSpan, std::string_view reason) {
        gr::log::warning("MarbleSource: {}", reason);
        if (_eventsPublished >= evtSpan.size()) {
            return;
        }
        if (gr::emitEvent(evtSpan, _eventsPublished, detail::makeErrorEvent(std::move(reason), this->unique_name.value()))) {
            ++_eventsPublished;
        }
    }

    void drainPendingErrors(gr::OutputSpanLike auto& evtSpan) {
        _eventsPublished = 0UZ;
        for (std::string& reason : _pendingErrors) {
            report(evtSpan, std::move(reason));
        }
        _pendingErrors.clear();
    }
};

template<typename T>
struct MarbleSink : gr::Block<MarbleSink<T>> {
    using Description = Doc<R"(@brief record samples and tags, and render them back as a marble script

    in      ─a──b──T:c──d──│
    script()  "a b T:c d |"          the terminator says how the stream ended: '|' or '#:reason'

What makes a test a comparison rather than a count: the recorded script is compared against the one that was played.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>   in;
    gr::EventPortIn evtIn;

    A<gr::property_map, "sample values", Doc<"value symbol -> sample">>         sample_values;
    A<gr::property_map, "sample tags", Doc<"tag symbol -> tag recorded there">> sample_tags;

    GR_MAKE_REFLECTABLE(MarbleSink, in, evtIn, sample_values, sample_tags);

    std::vector<T>                                        _samples;
    std::vector<std::pair<std::size_t, gr::property_map>> _tags;
    bool                                                  _completed = false;
    bool                                                  _errored   = false;
    std::pmr::string                                      _reason;

    void stop() { _completed = true; }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& evtSpan) {
        for (const gr::property_map_view& event : evtSpan) {
            noteError(event);
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        for (const auto& tag : inSpan.rawTags()) {
            if (tag.map.contains(std::string_view{gr::tag::END_OF_STREAM.key()})) {
                _completed = true;
                continue;
            }
            _tags.emplace_back(_samples.size() + (tag.index - inSpan.streamIndex), gr::property_map(tag.map));
        }
        _samples.insert(_samples.end(), std::ranges::begin(inSpan), std::ranges::end(inSpan));
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] std::string script() const {
        std::string text;
        for (std::size_t i = 0UZ; i < _samples.size(); ++i) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            for (const auto& [index, map] : _tags) {
                if (index == i) {
                    text.append(lookUpTagSymbol(map));
                    text.push_back(kMarbleTagValueMarker);
                }
            }
            text.append(lookUpSampleSymbol(_samples[i]));
        }
        if (_errored || _completed) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            text.push_back(_errored ? kMarbleError : kMarbleEndOfStream);
            if (_errored && !_reason.empty()) {
                text.push_back(kMarbleTagValueMarker);
                text.append(_reason);
            }
        }
        return text;
    }

private:
    void noteError(const gr::property_map_view& event) {
        if (event.empty()) {
            return;
        }
        const auto name = event.template get_if<std::string_view>(std::string_view{gr::tag::TRIGGER_NAME.key()});
        if (!name.has_value() || *name != std::string_view{"error"}) {
            return;
        }
        _errored = true;
        if (const auto details = event.template get_if<gr::property_map>(std::string_view{gr::tag::TRIGGER_META_INFO.key()})) {
            if (const auto reason = details->template get_if<std::string_view>(std::string_view{"reason"})) {
                _reason = std::pmr::string(*reason);
            }
        }
    }

    [[nodiscard]] std::string lookUpSampleSymbol(const T& sample) const {
        const auto keys  = sample_values.value.keys();
        const auto found = std::ranges::find_if(keys, [this, &sample](const auto& key) {
            const auto candidate = sample_values.value.template get_if<T>(key);
            return candidate && *candidate == sample;
        });
        return found == keys.end() ? std::format("{}", sample) : std::string(*found);
    }

    [[nodiscard]] std::string lookUpTagSymbol(const gr::property_map& recorded) const {
        const auto keys  = sample_tags.value.keys();
        const auto found = std::ranges::find_if(keys, [this, &recorded](const auto& key) {
            const auto candidate = sample_tags.value.template get_if<gr::property_map>(key);
            return candidate && *candidate == recorded;
        });
        return found == keys.end() ? std::string("?") : std::string(*found);
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_MARBLE_HPP
