#ifndef GNURADIO_DEVICE_WINDOW_GEOMETRY_HPP
#define GNURADIO_DEVICE_WINDOW_GEOMETRY_HPP

#include <algorithm>
#include <cstddef>

namespace gr::device {

/**
 * @brief How a span decomposes into the windows a block declared, `nWindows == 0` when it declared none.
 *
 * The framework's window tier and a block's own `processBulk_sycl` need the same answer. A hatch receives the
 * whole batched span -- that is what lets it pay for one kernel launch instead of one per frame -- so it has to
 * walk the frames itself, and this is the arithmetic it walks them by.
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

} // namespace gr::device

#endif // GNURADIO_DEVICE_WINDOW_GEOMETRY_HPP
