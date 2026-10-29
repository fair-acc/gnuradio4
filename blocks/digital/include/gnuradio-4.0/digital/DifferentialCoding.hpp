#ifndef GNURADIO_DIGITAL_DIFFERENTIALCODING_HPP
#define GNURADIO_DIGITAL_DIFFERENTIALCODING_HPP

#include <algorithm>
#include <cstdint>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::digital {

GR_REGISTER_BLOCK(gr::digital::DifferentialEncoder)

struct DifferentialEncoder : Block<DifferentialEncoder> {
    using Description = Doc<R""(@brief Differential encoder: out[n] = in[n] XOR out[n-1], so an input bit of 1 flips the output and 0 keeps it.

The information then lies in the changes between bits.
DifferentialDecoder recovers it even when the receiver inverts all bits, as a BPSK receiver locked with the opposite sign does.)"">;

    PortIn<std::uint8_t>  in;
    PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(DifferentialEncoder, in, out);

    bool _previousOutput = false;

    void reset() noexcept { _previousOutput = false; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        bool previousOutput = _previousOutput;
        std::ranges::transform(input, output.begin(), [&previousOutput](std::uint8_t bit) -> std::uint8_t {
            const bool flip = bit != 0U;
            previousOutput  = previousOutput != flip;
            return previousOutput;
        });
        _previousOutput = previousOutput;
        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK(gr::digital::DifferentialDecoder)

struct DifferentialDecoder : Block<DifferentialDecoder> {
    using Description = Doc<R""(@brief Differential decoder: out[n] = in[n] XOR in[n-1], so the output is 1 where the input bit changes.

It undoes DifferentialEncoder.
Because only changes count, inverting all input bits, as a BPSK receiver locked with the opposite sign does, changes only the first output bit.)"">;

    PortIn<std::uint8_t>  in;
    PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(DifferentialDecoder, in, out);

    bool _previousInput = false;

    void reset() noexcept { _previousInput = false; }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        bool previousInput = _previousInput;
        std::ranges::transform(input, output.begin(), [&previousInput](std::uint8_t bit) -> std::uint8_t {
            const bool current = bit != 0U;
            const bool changed = current != previousInput;
            previousInput      = current;
            return changed;
        });
        _previousInput = previousInput;
        return gr::work::Status::OK;
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_DIFFERENTIALCODING_HPP
