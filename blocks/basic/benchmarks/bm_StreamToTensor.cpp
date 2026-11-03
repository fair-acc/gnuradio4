#include <benchmark.hpp>

#include <format>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace {
constexpr gr::Size_t kSamples = 1U << 20U;

void runCopy() {
    gr::Graph graph;
    auto&     source = graph.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", kSamples}});
    auto&     copy   = graph.emplaceBlock<gr::testing::Copy<float>>();
    auto&     sink   = graph.emplaceBlock<gr::testing::CountingSink<float>>();
    std::ignore      = graph.connect<"out", "in">(source, copy);
    std::ignore      = graph.connect<"out", "in">(copy, sink);
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}

[[nodiscard]] gr::property_map sized(gr::Size_t size, gr::Size_t stride = 0U) { return {{"size", size}, {"stride", stride}}; }

constexpr std::size_t kStreams = 4UZ;

template<std::size_t nStreams>
void runStreams(gr::property_map framing) {
    gr::Graph graph;
    auto&     toTensor  = graph.emplaceBlock<gr::basic::StreamsToTensor<float, nStreams>>(std::move(framing));
    auto&     toStreams = graph.emplaceBlock<gr::basic::TensorToStreams<float, nStreams>>();
    std::ignore         = graph.connect<"out", "in">(toTensor, toStreams);
    for (std::size_t port = 0UZ; port < nStreams; ++port) {
        auto& source = graph.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", kSamples}});
        auto& sink   = graph.emplaceBlock<gr::testing::CountingSink<float>>();
        std::ignore  = graph.connect(source, "out", toTensor, std::format("in#{}", port));
        std::ignore  = graph.connect(toStreams, std::format("out#{}", port), sink, "in");
    }
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}
} // namespace

inline const boost::ut::suite<"StreamToTensor"> _streamToTensor = [] {
    using namespace benchmark; // figures per input sample, stream -> tensor -> stream
    "control: Copy<float>"_benchmark.repeat<5>(kSamples)                                                = [] { runCopy(); };
    "one port, size 8"_benchmark.repeat<5>(kSamples)                                                    = [] { runStreams<1UZ>(sized(8U)); };
    "one port, size 1024"_benchmark.repeat<5>(kSamples)                                                 = [] { runStreams<1UZ>(sized(1024U)); };
    "one port, size 1024, stride 512: overlapping, twice the samples out"_benchmark.repeat<5>(kSamples) = [] { runStreams<1UZ>(sized(1024U, 512U)); };
    "one port, size 8, stride 64: skipping, an eighth of the samples out"_benchmark.repeat<5>(kSamples) = [] { runStreams<1UZ>(sized(8U, 64U)); };
    "4 streams -> [4] per sample -> 4 streams"_benchmark.repeat<5>(kSamples)                            = [] { runStreams<kStreams>({}); };
    "4 streams -> [4,1024] frames -> 4 streams"_benchmark.repeat<5>(kSamples)                           = [] { runStreams<kStreams>(sized(1024U)); };
};

int main() {}
