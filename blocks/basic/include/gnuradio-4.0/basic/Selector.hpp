#ifndef GNURADIO_SELECTOR_HPP
#define GNURADIO_SELECTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic {
using namespace gr;

template<typename T>
struct Selector : Block<Selector<T>> {
    using Description = Doc<R""(
@brief basic multiplexing class to route arbitrary inputs to outputs

See https://wiki.gnuradio.org/index.php/Selector

The selector block allows arbitrary mapping between the input and output ports.

Selector has an arbitrary number of input and output ports, defined by
the `n_inputs` and `n_outputs` properties.

The mapping is defined by a pair of vectors `map_in` and `map_out`, the
corresponding indices from these two vectors define the
(input port index, output port index pairs).

For example, for arrays:

    map_in  = {2, 2, 3, 3}
    map_out = {1, 2, 3, 4}

The mapping is as follows:

     +--------------+
    -|-1   /----> 1-|-
    -|-2 >------> 2-|-
    -|-3 >------> 3-|-
    -|-4   \----> 4-|-
     +--------------+

And for arrays:

    map_in  = {1, 2, 3, 4}
    map_out = {2, 2, 3, 3}

The mapping is as follows:

     +--------------+
    -|-1 >-----\  1-|-
    -|-2 >------> 2-|-
    -|-3 >------> 3-|-
    -|-4 >-----/  4-|-
     +--------------+

It also contains two additional ports -- `selectOut` and `monitorOut`. Port
`selectOut` which can be used to define which input port is passed on to the
`monitorOut` port.

For uses where all input ports should be read even if they are not connected
to any output port (thus reading and ignoring all the values from the input),
you can set the `backPressure` property to false.

)"">;
    // optional shortening
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    PortIn<gr::Size_t, Async, Optional> select{};  // optional select input
    PortOut<T, Async, Optional>         monitor{}; // optional monitor output (for diagnostics and debugging purposes)
    std::vector<PortIn<T, Async>>       inputs{};  // TODO: need to add exception to pmt_t that this isn't interpreted as a settings type
    std::vector<PortOut<T, Async>>      outputs{};

    // settings
    A<gr::Size_t, "n_inputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>       n_inputs  = 0U;
    A<gr::Size_t, "n_outputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>      n_outputs = 0U;
    A<std::vector<gr::Size_t>, "map_in", Visible, Doc<"input port index to route from">>        map_in{}; // N.B. need two vectors since pmt_t doesn't support pairs (yet!?!)
    A<std::vector<gr::Size_t>, "map_out", Visible, Doc<"output port index to route to">>        map_out{};
    A<bool, "back_pressure", Visible, Doc<"true: do not consume samples from un-routed ports">> back_pressure = false;

    std::map<gr::Size_t, std::vector<gr::Size_t>> _internalMapping{};
    gr::Size_t                                    _selectedSrc = -1U;

    void
    settingsChanged(const gr::property_map &old_settings, const gr::property_map &new_settings) {
        if (new_settings.contains("n_inputs") || new_settings.contains("n_outputs")) {
            fmt::print("{}: configuration changed: n_inputs {} -> {}, n_outputs {} -> {}\n", this->name, old_settings.at("n_inputs"),
                       new_settings.contains("n_inputs") ? new_settings.at("n_inputs") : "same", old_settings.at("n_outputs"),
                       new_settings.contains("n_outputs") ? new_settings.at("n_outputs") : "same");
            inputs.resize(n_inputs);
            outputs.resize(n_outputs);
        }
        if (new_settings.contains("map_in") || new_settings.contains("map_out")) {
            assert(map_in.value.size() == map_out.value.size() && "map_in and map_out must have the same length");
            _internalMapping.clear();

            if (map_in.value.size() != map_out.value.size()) {
                throw std::invalid_argument("Input and output map need to have the same number of elements");
            }

            for (std::size_t i = 0U; i < map_out.value.size(); ++i) {
                _internalMapping[map_in.value[i]].push_back(map_out.value[i]);
            }
        }
    }

    template<gr::ConsumableSpan TInSpan, gr::PublishableSpan TOutSpan>
    gr::work::Status
    processBulk(const ConsumableSpan auto &selectSpan, const std::span<TInSpan> &ins, PublishableSpan auto &monOut, std::span<TOutSpan> &outs) {
        if (_internalMapping.empty()) {
            if (back_pressure) {
                std::for_each(ins.begin(), ins.end(), [](auto &input) { std::ignore = input.consume(0UZ); });
            } else {
                // make the implicit consume all available behaviour explicit
                std::for_each(ins.begin(), ins.end(), [](auto &input) { std::ignore = input.consume(input.size()); });
            }
            return work::Status::OK;
        }

        std::set<std::size_t> usedInputs;

        if (const auto selectAvailable = selectSpan.size(); selectAvailable > 0) {
            _selectedSrc = selectSpan.back();
            std::ignore  = selectSpan.consume(selectAvailable); // consume all samples on the 'select' streaming input port
        }

        std::vector<int> outOffsets(outs.size(), 0U);

        auto copyToOutput = [&outOffsets](auto inputAvailable, auto &inputSpan, auto &outputSpan, int outIndex) {
            const auto offset = (outIndex < 0) ? 0 : outOffsets[static_cast<std::size_t>(outIndex)];
            std::copy_n(inputSpan.begin(), inputAvailable, std::next(outputSpan.begin(), offset));
            if (outIndex >= 0) {
                outOffsets[static_cast<std::size_t>(outIndex)] += static_cast<int>(inputAvailable);
            }
            outputSpan.publish(inputAvailable);
        };

        bool monitorOutProcessed = false;
        for (const auto &[inIndex, outIndices] : _internalMapping) {
            ConsumableSpan auto inputSpan = ins[inIndex];
            auto                available = inputSpan.size();

            for (const auto outIndex : outIndices) {
                const auto remainingSize = outs[static_cast<std::size_t>(outIndex)].size() - static_cast<std::size_t>(outOffsets[static_cast<std::size_t>(outIndex)]);
                if (available > remainingSize) {
                    available = remainingSize;
                }
            }

            if (_selectedSrc == inIndex) {
                if (available > monOut.size()) {
                    available = monOut.size();
                }
            }

            if (available == 0) {
                continue;
            }

            for (const auto outIndex : outIndices) {
                copyToOutput(available, inputSpan, outs[outIndex], static_cast<int>(outIndex));
            }

            if (_selectedSrc == inIndex) {
                monitorOutProcessed = true;
                copyToOutput(available, inputSpan, monOut, -1);
            }

            std::ignore = inputSpan.consume(available);
            usedInputs.insert(inIndex);
        }

        if (!monitorOutProcessed && _selectedSrc < ins.size()) {
            ConsumableSpan auto inputSpan = ins[_selectedSrc];
            auto                available = std::min(inputSpan.size(), monOut.size());
            copyToOutput(available, inputSpan, monOut, -1);
            std::ignore = inputSpan.consume(available);
        }

        for (auto iPort = 0U; iPort < ins.size(); ++iPort) {
            if (usedInputs.contains(iPort)) continue;

            if (back_pressure) {
                std::ignore = ins[iPort].consume(0UZ);
            } else {
                // make the implicit consume all available behaviour explicit
                std::ignore = ins[iPort].consume(ins[iPort].size());
            }
        }
        return work::Status::OK;
    }
};
} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::Selector, select, inputs, monitor, outputs, n_inputs, n_outputs, map_in, map_out, back_pressure)
auto registerSelector = gr::registerBlock<gr::basic::Selector, float, double>(gr::globalBlockRegistry());
static_assert(gr::HasProcessBulkFunction<gr::basic::Selector<double>>);

#endif // include guard
