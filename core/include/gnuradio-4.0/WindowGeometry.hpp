#ifndef GNURADIO_WINDOW_GEOMETRY_HPP
#define GNURADIO_WINDOW_GEOMETRY_HPP

#include <algorithm>
#include <cstddef>

namespace gr {

/**
 * How a span decomposes into the windows a block declared, `nWindows == 0` when it declared none.
 *
 * This is the one formula the framework plans, consumes and publishes by, and the one a batched body walks its
 * frames by. Keeping a second copy anywhere lets the two drift, and the drift publishes samples nothing wrote.
 */
struct WindowGeometry {
    std::size_t nWindows = 0UZ;
    std::size_t hop      = 0UZ;
    std::size_t inChunk  = 0UZ;
    std::size_t outChunk = 0UZ;
};

template<typename TBlock>
[[nodiscard]] WindowGeometry windowGeometry(const TBlock& block, std::size_t nIn, std::size_t nOut) {
    const std::size_t inChunk  = static_cast<std::size_t>(block.input_chunk_size);
    const std::size_t outChunk = static_cast<std::size_t>(block.output_chunk_size);
    if ((inChunk <= 1UZ && outChunk <= 1UZ) || inChunk == 0UZ || outChunk == 0UZ) {
        return {}; // a 1:1 block is the auto-parallel tier's business, not this one
    }
    const std::size_t hop      = block.stride == 0U ? inChunk : static_cast<std::size_t>(block.stride);
    const std::size_t byOutput = nOut / outChunk;
    const std::size_t byInput  = inChunk > nIn ? 0UZ : 1UZ + (nIn - inChunk) / hop;
    return {.nWindows = std::min(byOutput, byInput), .hop = hop, .inChunk = inChunk, .outChunk = outChunk};
}

} // namespace gr

#endif // GNURADIO_WINDOW_GEOMETRY_HPP
