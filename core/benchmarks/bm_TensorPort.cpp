#include <benchmark.hpp>

#include <array>
#include <format>
#include <functional>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/meta/CacheLineSize.hpp>

namespace {
constexpr gr::Size_t kFrames = 20000U;

template<bool cacheLineAligned>
struct FrameSource : gr::Block<FrameSource<cacheLineAligned>> {
    gr::TensorPortOut<float> out;
    gr::Size_t               frame_size = 1024U;
    GR_MAKE_REFLECTABLE(FrameSource, out, frame_size);

    gr::Size_t _sent = 0U;

    gr::work::Status processBulk(gr::OutputSpanLike auto& output) {
        if (_sent >= kFrames) {
            output.publish(0UZ);
            return gr::work::Status::DONE;
        }
        const std::size_t                n = std::min<std::size_t>(output.size(), kFrames - _sent);
        const std::array<std::size_t, 1> extents{frame_size};
        for (std::size_t k = 0UZ; k < n; ++k) {
            if constexpr (cacheLineAligned) {
                const std::span<std::byte> stored = output.reserveBlob(k, frame_size * sizeof(float), gr::kCacheLine);
                output[k]                         = gr::TensorView<float>(reinterpret_cast<float*>(stored.data()), extents);
            } else {
                std::ignore = gr::emit(output, k, extents);
            }
            std::ranges::fill(std::span(output[k].data(), frame_size), static_cast<float>(k));
        }
        output.publish(n);
        _sent += static_cast<gr::Size_t>(n);
        return gr::work::Status::OK;
    }
};

struct FrameSum : gr::Block<FrameSum> {
    gr::TensorPortIn<float> in;
    GR_MAKE_REFLECTABLE(FrameSum, in);

    float _total = 0.f;

    gr::work::Status processBulk(gr::InputSpanLike auto& input) {
        for (const auto& frame : input) {
            _total += std::ranges::fold_left(std::span(frame.data(), frame.size()), 0.f, std::plus{});
        }
        return gr::work::Status::OK;
    }
};

template<bool cacheLineAligned>
void runGraph(gr::Size_t frameSize) {
    gr::Graph graph;
    auto&     source = graph.emplaceBlock<FrameSource<cacheLineAligned>>({{"frame_size", frameSize}});
    auto&     sum    = graph.emplaceBlock<FrameSum>();
    std::ignore      = graph.connect<"out", "in">(source, sum);
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}
} // namespace

inline const boost::ut::suite<"TensorPort"> _tensorPort = [] {
    for (const gr::Size_t frameSize : {256U, 4096U}) {
        const std::size_t nFloats     = kFrames * frameSize;
        const std::string typeAligned = std::format("TensorView<float> port, alignof(float), {} floats per tensor", frameSize);
        const std::string lineAligned = std::format("TensorView<float> port, cache-line aligned, {} floats per tensor", frameSize);

        ::benchmark::benchmark<5>(typeAligned, nFloats) = [frameSize] { runGraph<false>(frameSize); };
        ::benchmark::benchmark<5>(lineAligned, nFloats) = [frameSize] { runGraph<true>(frameSize); };
    }
};

int main() {}
