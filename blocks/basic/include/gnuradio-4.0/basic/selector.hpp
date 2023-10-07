#ifndef GNURADIO_SELECTOR_HPP
#define GNURADIO_SELECTOR_HPP

#include <gnuradio-4.0/node.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic {
using namespace gr;

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A           = Annotated<T, description, Arguments...>;

using SelectorDoc = Doc<R""(
@brief basic multiplexing class to route arbitrary inputs to outputs

See https://wiki.gnuradio.org/index.php/Selector

The selector block allows arbitrary mapping between the input and output ports.

Selector has an arbitrary number of input and output ports, defined by
the `nInputs` and `nOutputs` properties.

The mapping is defined by a pair of vectors `mapIn` and `mapOut`, the
corresponding indices from these two vectors define the
(input port index, output port index pairs).

For example, for arrays:

    mapIn  = {2, 2, 3, 3}
    mapOut = {1, 2, 3, 4}

The mapping is as follows:

     +--------------+
    -|-1   /----> 1-|-
    -|-2 >------> 2-|-
    -|-3 >------> 3-|-
    -|-4   \----> 4-|-
     +--------------+

And for arrays:

    mapIn  = {1, 2, 3, 4}
    mapOut = {2, 2, 3, 3}

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

template<typename T>
struct Selector : node<Selector<T>, SelectorDoc> {
    // port definitions
    PortIn<std::uint32_t, Async, Optional> selectOut;
    PortOut<T, Async, Optional>            monitorOut; // optional monitor output (more for demo/API purposes than actual need)
    std::vector<PortIn<T, Async>>          inputs;     // TODO: need to add exception to pmt_t that this isn't interpreted as a settings type
    std::vector<PortOut<T, Async>>         outputs;

    // settings
    A<std::uint32_t, "nInputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>    nInputs  = 0U;
    A<std::uint32_t, "nOutputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>   nOutputs = 0U;
    A<std::vector<std::uint32_t>, "mapIn", Visible, Doc<"input port index to route from">>     mapIn; // N.B. need two vectors since pmt_t doesn't support pairs (yet!?!)
    A<std::vector<std::uint32_t>, "mapOut", Visible, Doc<"output port index to route to">>     mapOut;
    A<bool, "backPressure", Visible, Doc<"true: do not consume samples from un-routed ports">> backPressure = false;
    std::map<std::uint32_t, std::vector<std::uint32_t>>                                        _internalMapping;
    std::uint32_t                                                                              _selectedSrc = -1U;

    void
    settings_changed(const gr::property_map &old_settings, const gr::property_map &new_settings) {
        if (new_settings.contains("nInputs") || new_settings.contains("nOutputs")) {
            fmt::print("{}: configuration changed: nInputs {} -> {}, nOutputs {} -> {}\n", static_cast<void *>(this), old_settings.at("nInputs"),
                       new_settings.contains("nInputs") ? new_settings.at("nInputs") : "same", old_settings.at("nOutputs"), new_settings.contains("nOutputs") ? new_settings.at("nOutputs") : "same");
            inputs.resize(nInputs);
            outputs.resize(nOutputs);
        }
        if (new_settings.contains("mapIn") || new_settings.contains("mapOut")) {
            assert(mapIn.value.size() == mapOut.value.size() && "mapIn and mapOut must have the same length");
            _internalMapping.clear();

            if (mapIn.value.size() != mapOut.value.size()) {
                throw std::invalid_argument("Input and output map need to have the same number of elements");
            }

            for (std::size_t i = 0U; i < mapOut.value.size(); ++i) {
                _internalMapping[mapIn.value[i]].push_back(mapOut.value[i]);
            }
        }
    }

    using select_reader_t  = typename PortIn<std::uint32_t, Async, Optional>::ReaderType;
    using monitor_writer_t = typename PortOut<T, Async, Optional>::WriterType;
    using input_reader_t   = typename PortIn<T, Async>::ReaderType;
    using output_writer_t  = typename PortOut<T, Async>::WriterType;

    gr::work_return_status_t
    process_bulk(select_reader_t                      *select, //
                 const std::vector<input_reader_t *>  &ins,
                 monitor_writer_t                     *monOut, //
                 const std::vector<output_writer_t *> &outs) {
        if (_internalMapping.empty()) {
            if (backPressure) {
                std::for_each(ins.begin(), ins.end(), [](auto *input) { std::ignore = input->consume(0_UZ); });
            } else {
                // make the implicit consume all available behaviour explicit
                std::for_each(ins.begin(), ins.end(), [](auto *input) { std::ignore = input->consume(input->available()); });
            }
            return work_return_status_t::OK;
        }

        std::set<std::size_t> used_inputs;

        if (const auto select_available = select->available(); select_available > 0) {
            auto select_span = select->get(select_available);
            _selectedSrc     = select_span.back();
            std::ignore      = select->consume(select_available); // consume all samples on the 'select' streaming input port
        }

        auto copy_to_output = [](auto input_available, auto &input_span, auto *output_writer) {
            auto output_span = output_writer->reserve_output_range(input_available);

            for (std::size_t i = 0; i < input_span.size(); ++i) {
                output_span[i] = input_span[i];
            }

            output_span.publish(input_available);
        };

        bool monitor_out_processed = false;

        for (const auto &[input_index, output_indices] : _internalMapping) {
            auto *input_reader    = ins[input_index];
            auto  input_available = input_reader->available();

            for (const auto output_index : output_indices) {
                auto *writer = outs[output_index];
                if (input_available > writer->available()) {
                    input_available = writer->available();
                }
            }

            if (_selectedSrc == input_index) {
                if (input_available > monOut->available()) {
                    input_available = monOut->available();
                }
            }

            if (input_available == 0) {
                continue;
            }

            auto input_span = input_reader->get(input_available);
            for (const auto output_index : output_indices) {
                auto *output_writer = outs[output_index];
                copy_to_output(input_available, input_span, output_writer);
            }

            if (_selectedSrc == input_index) {
                monitor_out_processed = true;
                copy_to_output(input_available, input_span, monOut);
            }

            std::ignore = input_reader->consume(input_available);
            used_inputs.insert(input_index);
        }

        if (!monitor_out_processed && _selectedSrc < ins.size()) {
            auto *input_reader    = ins[_selectedSrc];
            auto input_available = std::min(input_reader->available(), monOut->available());
            auto input_span = input_reader->get(input_available);
            copy_to_output(input_available, input_span, monOut);
            std::ignore = input_reader->consume(input_available);
        }

        for (auto src_port = 0U; src_port < ins.size(); ++src_port) {
            if (used_inputs.contains(src_port)) continue;

            if (backPressure) {
                std::ignore = ins[src_port]->consume(0_UZ);

            } else {
                // make the implicit consume all available behaviour explicit
                std::ignore = ins[src_port]->consume(ins[src_port]->available());
            }
        }

        return work_return_status_t::OK;
    }
};
} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::Selector, selectOut, inputs, monitorOut, outputs, nInputs, nOutputs, mapIn, mapOut, backPressure);
static_assert(gr::HasProcessBulkFunction<gr::basic::Selector<double>>);

#endif // include guard
