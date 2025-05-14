#ifndef GNURADIO_SELECTOR_HPP
#define GNURADIO_SELECTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic {
using namespace gr;

GR_REGISTER_BLOCK(gr::basic::Selector, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T>
struct Selector : Block<Selector<T>, NoDefaultTagForwarding> {
    using Description = Doc<R""(@brief basic multiplexing class to route arbitrary inputs to outputs

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


When multiple input ports are mapped to a single output port, their samples are synchronized.
This means that all input ports are aligned to have the same number of samples.
It is assumed that these ports should have the same sample rate; otherwise, the buffer may fill up quickly.

Example: Assuming there are 3 input ports with sample counts of 1, 2, and 3, and these are mapped to one output, the output would look like:
`1, 2, 3, 1, 2, 3, 1, 2, 3...`

If you want to change this behavior, set `sync_combined_ports = false;`.
In this case, samples are taken in order from each input, resulting in a sequence that copies samples sequentially from the first input, then the second, and so on.


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
    A<gr::Size_t, "n_inputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>                      n_inputs  = 0U;
    A<gr::Size_t, "n_outputs", Visible, Doc<"variable number of inputs">, Limits<1U, 32U>>                     n_outputs = 0U;
    A<std::vector<gr::Size_t>, "map_in", Visible, Doc<"input port index to route from">>                       map_in{}; // N.B. need two vectors since pmt_t doesn't support pairs (yet!?!)
    A<std::vector<gr::Size_t>, "map_out", Visible, Doc<"output port index to route to">>                       map_out{};
    A<bool, "back_pressure", Visible, Doc<"true: do not consume samples from un-routed ports">>                back_pressure       = false;
    A<bool, "sync combined port", Doc<"true: input ports connected to the same output port are synchronised">> sync_combined_ports = true;

    GR_MAKE_REFLECTABLE(Selector, select, inputs, monitor, outputs, n_inputs, n_outputs, map_in, map_out, back_pressure, sync_combined_ports);

    std::map<std::size_t, std::vector<std::size_t>> _internalMappingInOut{};
    std::map<std::size_t, std::vector<std::size_t>> _internalMappingOutIn{};

    std::size_t _selectedSrc = 0UZ;

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("n_inputs") || newSettings.contains("n_outputs")) {
            std::print("{}: configuration changed: n_inputs {} -> {}, n_outputs {} -> {}\n", this->name, oldSettings.at("n_inputs"), newSettings.contains("n_inputs") ? newSettings.at("n_inputs") : "same", oldSettings.at("n_outputs"), newSettings.contains("n_outputs") ? newSettings.at("n_outputs") : "same");
            inputs.resize(n_inputs);
            outputs.resize(n_outputs);
        }
        if (newSettings.contains("map_in") || newSettings.contains("map_out")) {
            assert(map_in.value.size() == map_out.value.size() && "map_in and map_out must have the same length");
            _internalMappingInOut.clear();
            _internalMappingOutIn.clear();

            if (map_in.value.size() != map_out.value.size()) {
                throw std::invalid_argument("Input and output map need to have the same number of elements");
            }

            std::set<std::pair<gr::Size_t, gr::Size_t>> duplicateSet{};
            for (auto i : std::views::iota(static_cast<std::size_t>(0), map_in.value.size())) {
                gr::Size_t inIdx  = map_in.value[i];
                gr::Size_t outIdx = map_out.value[i];

                _internalMappingInOut[inIdx].push_back(outIdx);
                _internalMappingOutIn[outIdx].push_back(inIdx);

                if (!duplicateSet.insert({inIdx, outIdx}).second) { // check for duplicates
                    throw std::invalid_argument(std::format("Duplicate pair (in:{}, out:{}) at i={}", inIdx, outIdx, i));
                }

                if (inIdx >= n_inputs) { // range checks
                    throw std::invalid_argument(std::format("map_in[{}] = {} is >= n_inputs ({})", i, inIdx, n_inputs));
                }
                if (outIdx >= n_outputs) {
                    throw std::invalid_argument(std::format("map_out[{}] = {} is >= n_outputs ({})", i, outIdx, n_outputs));
                }
            }
        }
    }

    template<gr::InputSpanLike TInSpan, gr::OutputSpanLike TOutSpan>
    gr::work::Status processBulk(InputSpanLike auto& selectSpan, std::span<TInSpan>& ins, OutputSpanLike auto& monOut, std::span<TOutSpan>& outs) {
        if (_internalMappingInOut.empty()) {
            std::ranges::for_each(ins, [this](auto& input) { std::ignore = input.consume(back_pressure ? 0UZ : input.size()); });
            return work::Status::OK;
        }

        if (const auto selectAvailable = selectSpan.size(); selectAvailable > 0) {
            _selectedSrc = selectSpan.back();
            std::ignore  = selectSpan.consume(selectAvailable); // consume all samples on the 'select' streaming input port
        } else {
            _selectedSrc = std::numeric_limits<std::size_t>::max();
        }

        std::vector<std::size_t> outOffsets(outs.size(), 0UZ);
        auto                     copyToOutput = [&outOffsets](std::size_t nSamplesToCopy, auto& inputSpan, auto& outputSpan, std::size_t outIndex) {
            const auto diffCount  = static_cast<std::ptrdiff_t>(nSamplesToCopy);
            const auto diffOffset = static_cast<std::ptrdiff_t>((outIndex == std::numeric_limits<std::size_t>::max()) ? 0U : outOffsets[outIndex]);

            std::copy_n(inputSpan.begin(), diffCount, std::next(outputSpan.begin(), diffOffset));

            if (outIndex != std::numeric_limits<std::size_t>::max()) {
                outOffsets[outIndex] += nSamplesToCopy;
            }

            const auto tags = inputSpan.tags(); // Tag handling
            for (const auto& [normalisedTagIndex, tagMap] : tags) {
                if (normalisedTagIndex < static_cast<std::ptrdiff_t>(nSamplesToCopy)) {
                    outputSpan.publishTag(tagMap, std::max(static_cast<std::size_t>(normalisedTagIndex) + static_cast<std::size_t>(diffOffset), 0UZ));
                }
            }
            outputSpan.publish(nSamplesToCopy);
        };

        std::vector<std::size_t> nSamplesToConsume(ins.size(), std::numeric_limits<std::size_t>::max());
        if (sync_combined_ports && std::ranges::any_of(_internalMappingOutIn, [](const auto& pair) { return pair.second.size() > 1; })) {
            for (const auto& [outIndex, inIndices] : _internalMappingOutIn) {
                std::size_t nInSamplesAvailable = std::ranges::min(inIndices | std::views::transform([&ins](std::size_t i) { return ins[i].size(); }));
                nInSamplesAvailable             = std::min(nInSamplesAvailable, outs[outIndex].size() / inIndices.size());
                if (std::ranges::find(inIndices, _selectedSrc) != inIndices.end()) {
                    nInSamplesAvailable = std::min(nInSamplesAvailable, monOut.size());
                }
                for (const std::size_t inIndex : inIndices) {
                    nSamplesToConsume[inIndex] = std::min(nSamplesToConsume[inIndex], nInSamplesAvailable);
                }
            }

            // we need to iterate until no changes occur to include all dependencies -> Think about alternative approach
            bool changed = true;
            while (changed) {
                changed = false;
                for (const auto& [outIndex, inIndices] : _internalMappingOutIn) {
                    const std::size_t currentMin = std::ranges::min(inIndices | std::views::transform([&nSamplesToConsume](std::size_t i) { return nSamplesToConsume[i]; }));
                    for (const std::size_t inIndex : inIndices) {
                        if (nSamplesToConsume[inIndex] > currentMin) {
                            changed = true;
                        }
                        nSamplesToConsume[inIndex] = currentMin;
                    }
                }
            }

            for (const auto& [outIndex, inIndices] : _internalMappingOutIn) {
                if (inIndices.size() == 1) {
                    const std::size_t inIndex = inIndices[0];
                    copyToOutput(nSamplesToConsume[inIndex], ins[inIndex], outs[outIndex], std::numeric_limits<std::size_t>::max());
                } else if (inIndices.size() > 1) {
                    auto&       outSpan           = outs[outIndex];
                    std::size_t nSamplesToPublish = 0UZ;
                    for (std::size_t iS = 0UZ; iS < nSamplesToConsume[0]; iS++) {
                        for (const std::size_t inIndex : inIndices) {
                            outSpan[nSamplesToPublish] = ins[inIndex][iS];
                            for (const auto& tag : ins[inIndex].rawTags) {
                                const auto relIndex = tag.index >= ins[inIndex].streamIndex ? static_cast<std::ptrdiff_t>(tag.index - ins[inIndex].streamIndex) : -static_cast<std::ptrdiff_t>(ins[inIndex].streamIndex - tag.index);
                                if (relIndex == static_cast<std::ptrdiff_t>(iS)) {
                                    outSpan.publishTag(tag.map, nSamplesToPublish);
                                }
                            }
                            nSamplesToPublish++;
                        }
                    }
                    outSpan.publish(nSamplesToPublish);
                }
            }
        } else {
            for (const auto& [inIndex, outIndices] : _internalMappingInOut) {
                InputSpanLike auto inSpan         = ins[inIndex];
                std::size_t        monOutSize     = _selectedSrc == inIndex ? monOut.size() : std::numeric_limits<std::size_t>::max();
                std::size_t        nSamplesToCopy = std::min({inSpan.size(), monOutSize, //
                           std::ranges::min(outIndices | std::views::transform([&](std::size_t outIndex) { return outs[outIndex].size() - outOffsets[outIndex]; }))});

                for (const std::size_t outIndex : outIndices) {
                    copyToOutput(nSamplesToCopy, inSpan, outs[outIndex], outIndex);
                }
                nSamplesToConsume[inIndex] = nSamplesToCopy;
            }
        }

        if (_selectedSrc < ins.size()) {
            InputSpanLike auto inSpan         = ins[_selectedSrc];
            std::size_t        nSamplesToCopy = std::min({inSpan.size(), monOut.size(), nSamplesToConsume[_selectedSrc]});
            copyToOutput(nSamplesToCopy, inSpan, monOut, std::numeric_limits<std::size_t>::max());
            nSamplesToConsume[_selectedSrc] = nSamplesToCopy;
        }

        for (std::size_t inIndex = 0UZ; inIndex < ins.size(); inIndex++) {
            const std::size_t nBackPressure = back_pressure ? 0UZ : ins[inIndex].size();
            const std::size_t nFinal        = nSamplesToConsume[inIndex] == std::numeric_limits<std::size_t>::max() ? nBackPressure : nSamplesToConsume[inIndex];
            std::ignore                     = ins[inIndex].consume(nFinal);
            ins[inIndex].consumeTags(nFinal);
        }
        return work::Status::OK;
    }
};
} // namespace gr::basic

#endif // include guard
