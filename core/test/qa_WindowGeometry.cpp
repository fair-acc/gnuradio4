#include <boost/ut.hpp>

#include <gnuradio-4.0/WindowGeometry.hpp>

#include <cstdint>

namespace {
/// the three settings `windowGeometry` reads, without dragging a whole Block in to carry them
struct WindowedBlockStub {
    std::uint32_t input_chunk_size  = 1U;
    std::uint32_t output_chunk_size = 1U;
    std::uint32_t stride            = 0U;
};
} // namespace

const boost::ut::suite<"WindowGeometry"> _windowGeometryTests = [] {
    using namespace boost::ut;

    "non-overlapping chunks tile the span"_test = [] {
        const gr::WindowGeometry geometry = gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 4U, .output_chunk_size = 1U}, 16UZ, 4UZ);
        expect(eq(geometry.nWindows, 4UZ));
        expect(eq(geometry.hop, 4UZ)) << "an unset stride hops by a whole chunk";
    };

    "overlapping windows hop by the stride"_test = [] {
        const gr::WindowGeometry geometry = gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 100U, .output_chunk_size = 100U, .stride = 50U}, 1000UZ, 1900UZ);
        expect(eq(geometry.nWindows, 19UZ));
        expect(eq(geometry.hop, 50UZ));
    };

    "a stride wider than the chunk leaves gaps"_test = [] {
        const gr::WindowGeometry geometry = gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 50U, .output_chunk_size = 50U, .stride = 100U}, 1000UZ, 1000UZ);
        expect(eq(geometry.nWindows, 10UZ));
    };

    "the scarcer side bounds the count"_test = [] {
        const WindowedBlockStub block{.input_chunk_size = 10U, .output_chunk_size = 10U, .stride = 10U};
        expect(eq(gr::windowGeometry(block, 100UZ, 30UZ).nWindows, 3UZ)) << "output-bound";
        expect(eq(gr::windowGeometry(block, 40UZ, 100UZ).nWindows, 4UZ)) << "input-bound";
    };

    "too little input yields no window at all"_test = [] { expect(eq(gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 100U, .output_chunk_size = 1U}, 99UZ, 100UZ).nWindows, 0UZ)); };

    "a 1:1 block declares no window"_test = [] { expect(eq(gr::windowGeometry(WindowedBlockStub{}, 1024UZ, 1024UZ).nWindows, 0UZ)); };

    "a zero chunk size is refused rather than divided by"_test = [] {
        expect(eq(gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 0U, .output_chunk_size = 8U}, 1024UZ, 1024UZ).nWindows, 0UZ));
        expect(eq(gr::windowGeometry(WindowedBlockStub{.input_chunk_size = 8U, .output_chunk_size = 0U}, 1024UZ, 1024UZ).nWindows, 0UZ));
    };

    // Block::computeResampling reserves (nWindows - 1) * hop + inChunk input and nWindows * outChunk output, then the body
    // asks this same function how many windows it was handed. Both sides only agree while that round trip is a fixed point.
    "the span the framework reserves decomposes back into the windows it planned"_test = [] {
        for (std::uint32_t inChunk : {1U, 2U, 7U, 64U, 100U}) {
            for (std::uint32_t outChunk : {1U, 3U, 64U, 100U}) {
                for (std::uint32_t stride : {0U, 1U, 3U, 33U, 50U, 100U, 250U}) {
                    const WindowedBlockStub block{.input_chunk_size = inChunk, .output_chunk_size = outChunk, .stride = stride};

                    const gr::WindowGeometry planned = gr::windowGeometry(block, 4096UZ, 4096UZ);
                    if (planned.nWindows == 0UZ) {
                        continue;
                    }
                    const std::size_t        reservedIn  = (planned.nWindows - 1UZ) * planned.hop + planned.inChunk;
                    const std::size_t        reservedOut = planned.nWindows * planned.outChunk;
                    const gr::WindowGeometry handed      = gr::windowGeometry(block, reservedIn, reservedOut);

                    expect(eq(handed.nWindows, planned.nWindows)) << std::format("in_chunk {} out_chunk {} stride {}", inChunk, outChunk, stride);
                    expect(eq(handed.nWindows * handed.outChunk, reservedOut)) << "every published sample belongs to a window the body was handed";
                }
            }
        }
    };
};

int main() { /* tests are statically registered */ }
