#ifndef GNURADIO_BASIC_TAG_DISPATCH_HPP
#define GNURADIO_BASIC_TAG_DISPATCH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>

#include <format>
#include <limits>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::TagDispatch, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T, std::size_t nPorts = 0UZ>
requires(nPorts <= detail::kMaxPorts)
struct TagDispatch : Block<TagDispatch<T, nPorts>, NoTagPropagation> {
    using Description = Doc<"Copies each input to its output; the selected input's tags reach every output, kAllInputs: all; canonical tags (gr: or short key) stay on their stream.">;

    static constexpr gr::Size_t kAllInputs = std::numeric_limits<gr::Size_t>::max();

    detail::PortsOf<PortIn<T>, nPorts>  in;
    detail::PortsOf<PortOut<T>, nPorts> out;

    Annotated<gr::Size_t, "n_ports", Doc<"inputs and outputs; nPorts when that is not 0">, Limits<1U, static_cast<gr::Size_t>(detail::kMaxPorts)>> n_ports        = nPorts == 0UZ ? 1U : static_cast<gr::Size_t>(nPorts);
    Annotated<gr::Size_t, "input_selector", Doc<"input whose tags reach every output; kAllInputs: every input's">>                                 input_selector = kAllInputs;

    GR_MAKE_REFLECTABLE(TagDispatch, in, out, n_ports, input_selector);

    struct PendingTag {
        std::size_t  index;
        std::size_t  port;
        bool         broadcast;
        property_map map;
    };
    std::vector<PendingTag> _pending; // reused across calls

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        detail::refuseNewPortCount<nPorts>(*this, newSettings, in);
        if constexpr (nPorts == 0UZ) {
            if (newSettings.contains("n_ports") || in.size() != n_ports.value) {
                in.resize(n_ports.value);
                out.resize(n_ports.value);
            }
        }
        if (input_selector != kAllInputs && input_selector >= in.size()) {
            const gr::Size_t previous = oldSettings.value_or<gr::Size_t>("input_selector", kAllInputs);
            this->emitErrorMessage("settingsChanged()", std::format("input_selector {} names no input of {}: keeps {}", input_selector.value, in.size(), previous));
            input_selector = previous;
        }
    }

    template<InputSpanLike TIn, OutputSpanLike TOut>
    work::Status processBulk(std::span<TIn> inputs, std::span<TOut> outputs) {
        std::size_t count = std::numeric_limits<std::size_t>::max();
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            count = std::min({count, inputs[port].size(), outputs[port].size()});
        }
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            std::ranges::copy(std::span<const T>{inputs[port].data(), count}, outputs[port].begin());
        }
        _pending.clear();
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            const bool broadcast = input_selector == kAllInputs || input_selector == port;
            for (const auto& [index, map] : inputs[port].tags()) {
                const std::size_t offset = index < 0 ? 0UZ : static_cast<std::size_t>(index);
                if (offset >= count) {
                    break;
                }
                if (broadcast) {
                    _pending.push_back({offset, port, true, property_map(map.get())});
                } else {
                    property_map control;
                    for (const auto& [key, value] : map.get()) {
                        const std::string_view name(key);
                        if (gr::pmt::keys::lookupId(name) != gr::pmt::keys::kIdUnknown || gr::pmt::keys::lookupShortId(name) != gr::pmt::keys::kIdUnknown) {
                            std::ignore = control.insert_or_assign(key, value);
                        }
                    }
                    if (!control.empty()) {
                        _pending.push_back({offset, port, false, std::move(control)});
                    }
                }
            }
        }
        std::ranges::stable_sort(_pending, {}, &PendingTag::index);
        for (const auto& tag : _pending) {
            if (tag.broadcast) {
                for (auto& output : outputs) {
                    output.publishTag(tag.map, tag.index);
                }
            } else {
                outputs[tag.port].publishTag(tag.map, tag.index);
            }
        }
        for (std::size_t port = 0UZ; port < inputs.size(); ++port) {
            std::ignore = inputs[port].consume(count);
            outputs[port].publish(count);
        }
        return work::Status::OK;
    }
};

} // namespace gr::basic
#endif // GNURADIO_BASIC_TAG_DISPATCH_HPP
