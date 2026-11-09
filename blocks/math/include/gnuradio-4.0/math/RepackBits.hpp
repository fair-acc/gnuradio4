#ifndef GNURADIO_MATH_REPACK_BITS_HPP
#define GNURADIO_MATH_REPACK_BITS_HPP

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::blocks::math {

enum class BitOrder : std::uint8_t { lsbFirst, msbFirst };

GR_REGISTER_BLOCK(gr::blocks::math::RepackBits, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<std::unsigned_integral T>
struct RepackBits : Block<RepackBits<T>, Resampling<1UZ, 1UZ, false>> {
    using Description = Doc<R""(@brief regroups the low `bits_in` bits of each input word into output words of `bits_out` bits (GR3: Repack Bits).

`bit_order` names the end of a word that goes first, on both sides; bits above `bits_in` are ignored.
Bits regroup in whole chunks of lcm(`bits_in`, `bits_out`) bits; a remainder waits for more input.
Unpacking k bits per word is `bits_in = k, bits_out = 1, bit_order = msbFirst` (GR3: Unpack K Bits, Packed to Unpacked); packing swaps the two widths (GR3: Pack K Bits, Unpacked to Packed).)"">;

    static constexpr gr::Size_t kMaxBits = std::numeric_limits<T>::digits;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "bits_in", Doc<"significant low bits per input word">, Limits<1U, kMaxBits>> bits_in   = 1U;
    Annotated<gr::Size_t, "bits_out", Doc<"bits per output word">, Limits<1U, kMaxBits>>               bits_out  = 8U;
    Annotated<BitOrder, "bit_order", Doc<"which end of a word goes first">>                            bit_order = BitOrder::lsbFirst;

    GR_MAKE_REFLECTABLE(RepackBits, in, out, bits_in, bits_out, bit_order);

    void start() {
        const std::size_t bitsPerChunk = std::lcm(bits_in.value, bits_out.value);
        this->input_chunk_size         = bitsPerChunk / bits_in.value;
        this->output_chunk_size        = bitsPerChunk / bits_out.value;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t bitsIn       = bits_in.value;
        const std::size_t bitsOut      = bits_out.value;
        const std::size_t bitsPerChunk = std::lcm(bitsIn, bitsOut);
        const std::size_t wordsIn      = bitsPerChunk / bitsIn;
        const std::size_t wordsOut     = bitsPerChunk / bitsOut;
        const std::size_t nChunks      = std::min(input.size() / wordsIn, output.size() / wordsOut);
        const bool        msbFirst     = bit_order.value == BitOrder::msbFirst;

        std::fill_n(output.begin(), nChunks * wordsOut, T{0});
        for (std::size_t bit = 0UZ; bit < nChunks * bitsPerChunk; ++bit) {
            const std::size_t inShift  = msbFirst ? bitsIn - 1UZ - bit % bitsIn : bit % bitsIn;
            const std::size_t outShift = msbFirst ? bitsOut - 1UZ - bit % bitsOut : bit % bitsOut;
            const T           value    = static_cast<T>(input[bit / bitsIn] >> inShift) & T{1};
            output[bit / bitsOut] |= static_cast<T>(value << outShift);
        }
        std::ignore = input.consume(nChunks * wordsIn);
        output.publish(nChunks * wordsOut);
        if (nChunks > 0UZ) {
            return gr::work::Status::OK;
        }
        return input.size() < wordsIn ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (!newSettings.contains("bits_in") && !newSettings.contains("bits_out")) {
            return;
        }
        const std::size_t bitsPerChunk = std::lcm(bits_in.value, bits_out.value);
        this->input_chunk_size         = bitsPerChunk / bits_in.value;
        this->output_chunk_size        = bitsPerChunk / bits_out.value;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_REPACK_BITS_HPP
