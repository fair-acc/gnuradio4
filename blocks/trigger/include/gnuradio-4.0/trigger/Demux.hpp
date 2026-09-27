#ifndef GNURADIO_TRIGGER_DEMUX_HPP
#define GNURADIO_TRIGGER_DEMUX_HPP

#include <algorithm>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::Demux, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct Demux : gr::Block<Demux<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief route a stream to the output its context tag names, sample-exact

    in      ─a─a─R:a─a─a─F:b─b─▶    (no RxMarbles equivalent: Rx routes by predicate, not by tag)
    out#0   ──────R:a─a─a──────▶    contexts = RAMP, FLATTOP
    out#1   ─────────────F:b─b─▶

One chain usually carries several machine states in turn -- a ramp, a flat top, a calibration pulse -- and what happens next
differs for each.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::EventPortIn                        evtIn;
    gr::PortIn<T>                          in;
    std::vector<gr::PortOut<T, gr::Async>> out;

    A<std::vector<std::string>, "contexts", Doc<"one context per output, in output order">> contexts;

    A<gr::Size_t, "n switches", Doc<"context changes acted on">>                     n_switches  = 0U;
    A<gr::Size_t, "n unmatched", Doc<"samples with no output, or before the first">> n_unmatched = 0U;

    GR_MAKE_REFLECTABLE(Demux, evtIn, in, out, contexts, n_switches, n_unmatched);

    std::size_t                                           _streamIndex = 0UZ;
    std::size_t                                           _selected    = kNothingSelected;
    std::vector<std::pair<std::size_t, gr::property_map>> _switches;
    std::vector<std::size_t>                              _published;

    constexpr static std::size_t kNothingSelected = std::numeric_limits<std::size_t>::max();

    void start() {
        _streamIndex = 0UZ;
        _selected    = kNothingSelected;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("contexts") && oldSettings.find_value("contexts") != newSettings.find_value("contexts")) {
            out.resize(contexts.value.size());
            _selected = kNothingSelected;
        }
    }

    template<gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(gr::InputSpanLike auto& evtSpan, gr::InputSpanLike auto& inSpan, std::span<TOutput>& outs) {
        gather(evtSpan, inSpan);
        _published.assign(outs.size(), 0UZ);

        std::size_t consumed   = 0UZ;
        std::size_t nextSwitch = 0UZ;
        while (consumed < inSpan.size()) {
            const std::size_t absolute = _streamIndex + consumed;
            while (nextSwitch < _switches.size() && _switches[nextSwitch].first <= absolute) {
                select(_switches[nextSwitch].second, outs);
                ++nextSwitch;
            }

            const std::size_t until = nextSwitch < _switches.size() ? _switches[nextSwitch].first : _streamIndex + inSpan.size();
            std::size_t       run   = until - absolute;
            if (_selected >= outs.size()) {
                n_unmatched = n_unmatched + static_cast<gr::Size_t>(run);
                consumed += run;
                continue;
            }
            run = std::min(run, outs[_selected].size() - _published[_selected]);
            if (run == 0UZ) {
                break;
            }
            std::ranges::copy(inSpan | std::views::drop(consumed) | std::views::take(run), outs[_selected].begin() + static_cast<std::ptrdiff_t>(_published[_selected]));
            _published[_selected] += run;
            consumed += run;
        }

        for (std::size_t channel = 0UZ; channel < outs.size(); ++channel) {
            outs[channel].publish(_published[channel]);
        }
        _streamIndex += consumed;
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void gather(gr::InputSpanLike auto& evtSpan, const gr::InputSpanLike auto& inSpan) {
        _switches.clear();
        for (const gr::property_map_view& event : evtSpan) {
            if (!event.empty() && event.contains(std::string_view{gr::tag::CONTEXT.key()})) {
                _switches.emplace_back(_streamIndex, gr::property_map{event});
            }
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        for (const auto& tag : inSpan.rawTags()) {
            if (tag.map.contains(std::string_view{gr::tag::CONTEXT.key()})) {
                _switches.emplace_back(_streamIndex + (tag.index - inSpan.streamIndex), gr::property_map{gr::property_map_view{tag.map}});
            }
        }
        std::ranges::stable_sort(_switches, {}, [](const auto& entry) { return entry.first; });
    }

    template<gr::OutputSpanLike TOutput>
    void select(const gr::property_map& switching, std::span<TOutput>& outs) {
        const gr::property_map_view view{switching};
        const auto                  named = view.template get_if<std::string_view>(std::string_view{gr::tag::CONTEXT.key()});
        const std::size_t           found = named.has_value() ? indexOf(*named) : kNothingSelected;
        if (found != _selected) {
            n_switches = n_switches + 1U;
        }
        _selected = found;
        if (_selected < outs.size()) {
            outs[_selected].publishTag(switching, _published[_selected]);
        }
    }

    [[nodiscard]] std::size_t indexOf(std::string_view context) const {
        const auto found = std::ranges::find(contexts.value, context);
        return found == contexts.value.end() ? kNothingSelected : static_cast<std::size_t>(std::ranges::distance(contexts.value.begin(), found));
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_DEMUX_HPP
