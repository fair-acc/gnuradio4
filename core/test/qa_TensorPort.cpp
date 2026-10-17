#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;

template<typename T>
[[nodiscard]] GR_DEVICE_FN constexpr T frameElement(std::size_t frame, std::size_t j) noexcept {
    return T(static_cast<decltype(std::real(T{}))>(frame % 7UZ + j));
}

/** frame k has rank 1 or 2 and holds (k % 7) + j at element j, so its sum is exact in float; on a device the views are reserved on the host and filled by a kernel */
template<typename T>
struct FrameSource : gr::Block<FrameSource<T>> {
    gr::TensorPortOut<T> out;
    gr::Size_t           n_frames   = 100U;
    gr::Size_t           frame_size = 64U;
    bool                 by_copy    = false; /**< copies an owning Tensor into the reserved elements instead of filling them in place */
    GR_MAKE_REFLECTABLE(FrameSource, out, n_frames, frame_size, by_copy);

    gr::Size_t  _sent        = 0U;
    std::size_t _deviceCalls = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& output) {
        if (_sent >= n_frames) {
            output.publish(0UZ);
            return gr::work::Status::DONE;
        }
        const std::size_t n = std::min<std::size_t>(output.size(), n_frames - _sent);
        for (std::size_t k = 0UZ; k < n; ++k) {
            const std::size_t                  frame = _sent + k;
            const std::size_t                  size  = frame_size + frame % 3UZ;
            const std::array<std::size_t, 2>   shape2d{2UZ, size / 2UZ};
            const std::array<std::size_t, 1>   shape1d{size};
            const std::span<const std::size_t> extents = size % 2UZ == 0UZ ? std::span<const std::size_t>(shape2d) : std::span<const std::size_t>(shape1d);
            if (by_copy) {
                gr::Tensor<T> owned(gr::extents_from, extents);
                for (std::size_t j = 0UZ; j < owned.size(); ++j) {
                    owned.data()[j] = frameElement<T>(frame, j);
                }
                auto view = gr::emit(output, k, extents);
                expect(view.has_value());
                std::ranges::copy(owned, view->begin());
            } else {
                auto view = gr::emit(output, k, extents);
                expect(view.has_value());
                for (std::size_t j = 0UZ; j < view->size(); ++j) {
                    view->data()[j] = frameElement<T>(frame, j);
                }
            }
        }
        output.publish(n);
        _sent += static_cast<gr::Size_t>(n);
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::device::DeviceContext& ctx, gr::OutputSpanLike auto& output) {
        ++_deviceCalls;
        if (_sent >= n_frames) {
            output.publish(0UZ);
            return gr::work::Status::DONE;
        }
        const std::size_t n = std::min<std::size_t>(output.size(), n_frames - _sent);
        for (std::size_t k = 0UZ; k < n; ++k) {
            const std::array<std::size_t, 1> extents{frame_size + (_sent + k) % 3UZ};
            if (!gr::emit(output, k, extents)) {
                output.publish(0UZ);
                return gr::work::Status::ERROR;
            }
        }
        gr::device::parallelFor(ctx, n, [frames = output.data(), first = static_cast<std::size_t>(_sent)] GR_DEVICE_LAMBDA(std::size_t k) {
            T* const data = frames[k].data();
            for (std::size_t j = 0UZ; j < frames[k].size(); ++j) {
                data[j] = frameElement<T>(first + k, j);
            }
        });
        output.publish(n);
        _sent += static_cast<gr::Size_t>(n);
        return gr::work::Status::OK;
    }
};

[[nodiscard]] std::size_t expectedSum(std::size_t frame, std::size_t frameSize) {
    const std::size_t size = frameSize + frame % 3UZ;
    return (frame % 7UZ) * size + size * (size - 1UZ) / 2UZ;
}

/** the sum of each frame's elements, on the host or in a kernel; records each view's alignment on the host path */
template<typename T>
struct FrameSum : gr::Block<FrameSum<T>> {
    gr::TensorPortIn<T> in;
    gr::PortOut<T>      out;
    GR_MAKE_REFLECTABLE(FrameSum, in, out);

    std::size_t _deviceCalls = 0UZ;
    bool        _aligned     = true;

    gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        for (std::size_t k = 0UZ; k < input.size(); ++k) {
            _aligned  = _aligned && reinterpret_cast<std::uintptr_t>(input[k].data()) % alignof(T) == 0UZ;
            output[k] = std::ranges::fold_left(input[k], T{}, std::plus<>{});
        }
        return gr::work::Status::OK;
    }

    gr::work::Status processBulk(gr::device::DeviceContext& ctx, gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        ++_deviceCalls;
        gr::device::parallelFor(ctx, input.size(), [frames = input.data(), sums = output.data()] GR_DEVICE_LAMBDA(std::size_t k) {
            const T*          data = frames[k].data();
            const std::size_t n    = frames[k].size();
            T                 sum{};
            for (std::size_t j = 0UZ; j < n; ++j) {
                sum += data[j];
            }
            sums[k] = sum;
        });
        return gr::work::Status::OK;
    }
};

template<typename T>
[[nodiscard]] bool sumsCorrect(const auto& samples, std::size_t frameSize) {
    return std::ranges::all_of(std::views::iota(0UZ, samples.size()), [&](std::size_t k) { return samples[k] == T(static_cast<decltype(std::real(T{}))>(expectedSum(k, frameSize))); });
}

template<typename T>
using Sink = gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_BULK>;

const auto host = [] {
    "tensors of varying rank and size, written in place or copied in, arrive whole, aligned for their element type"_test = []<typename T>() {
        for (const bool byCopy : {false, true}) {
            gr::testing::GraphFixture<> fixture;
            const gr::property_map      frames{{"n_frames", gr::Size_t(300)}, {"frame_size", gr::Size_t(64)}, {"by_copy", byCopy}};
            auto&                       source = fixture.emplace<FrameSource<T>>(frames);
            auto&                       sum    = fixture.emplace<FrameSum<T>>();
            auto&                       sink   = fixture.emplace<Sink<T>>({{"log_samples", true}});
            expect(fixture.connect<"out", "in">(source, sum).has_value());
            expect(fixture.connect<"out", "in">(sum, sink).has_value());
            expect(fixture.run().has_value());
            expect(eq(sink._samples.size(), 300UZ)) << (byCopy ? "a copied tensor" : "a tensor filled in place");
            expect(sumsCorrect<T>(sink._samples, 64UZ)) << (byCopy ? "a copied tensor" : "a tensor filled in place");
            expect(sum._aligned) << "every view's data is aligned to alignof(T)";
        }
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "tensors smaller or larger than a chunk publish without an allocation per tensor"_test = [] {
        const auto edgeAllocations = [](gr::Size_t nFrames, gr::Size_t frameSize) {
            gr::allocator::pmr::CountingResource edge;
            gr::testing::GraphFixture<>          fixture;
            const gr::property_map               frames{{"n_frames", nFrames}, {"frame_size", frameSize}};
            auto&                                source = fixture.emplace<FrameSource<float>>(gr::ResourceProfile{.data = &edge}, frames);
            auto&                                sum    = fixture.emplace<FrameSum<float>>();
            auto&                                sink   = fixture.emplace<Sink<float>>();
            expect(fixture.connect<"out", "in">(source, sum).has_value());
            expect(fixture.connect<"out", "in">(sum, sink).has_value());
            expect(fixture.run().has_value());
            return edge.allocCount;
        };
        for (const gr::Size_t frameSize : {gr::Size_t(256), gr::Size_t(4096)}) {
            const std::size_t fewer = edgeAllocations(2000U, frameSize);
            const std::size_t more  = edgeAllocations(20000U, frameSize);
            expect(le(more - fewer, 18UZ)) << "18000 more tensors of" << frameSize << "floats cost fewer than one allocation per 1000 (sizes vary by 2 floats)";
        }
    };

    "20000 tensors recycle the buffer's chunks without loss"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<FrameSource<float>>({{"n_frames", gr::Size_t(20000)}, {"frame_size", gr::Size_t(256)}});
        auto&                       sum    = fixture.emplace<FrameSum<float>>();
        auto&                       sink   = fixture.emplace<Sink<float>>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, sum).has_value());
        expect(fixture.connect<"out", "in">(sum, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._samples.size(), 20000UZ));
        expect(sumsCorrect<float>(sink._samples, 256UZ));
    };
};

struct FrameChain {
    FrameSource<float>& source;
    FrameSum<float>&    sum;
};

/** 500 frames of 64 floats through source -> sum -> sink on the fixture, checking the sums; the settings place the blocks */
template<typename TFixture>
[[nodiscard]] FrameChain runFrameChain(TFixture& fixture, std::string_view domain, const gr::property_map& sourceSettings, const gr::property_map& sumSettings) {
    auto& source = fixture.template emplace<FrameSource<float>>(sourceSettings);
    auto& sum    = fixture.template emplace<FrameSum<float>>(sumSettings);
    auto& sink   = fixture.template emplace<Sink<float>>({{"log_samples", true}});
    expect(fixture.template connect<"out", "in">(source, sum).has_value());
    expect(fixture.template connect<"out", "in">(sum, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(eq(sink._samples.size(), 500UZ)) << domain;
    expect(sumsCorrect<float>(sink._samples, 64UZ)) << domain;
    return {source, sum};
}

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "a kernel reads every tensor through its view on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto chain = runFrameChain(fixture, domain, {{"n_frames", gr::Size_t(500)}, {"frame_size", gr::Size_t(64)}}, {{"gr:compute_domain", std::string(domain) + "!"}});
        expect(domain == "host" || chain.sum._deviceCalls > 0UZ) << domain << "the device overload ran";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "a kernel fills tensors reserved on the host, which a host block reads on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const gr::property_map frames{{"n_frames", gr::Size_t(500)}, {"frame_size", gr::Size_t(64)}, {"gr:compute_domain", std::string(domain) + "!"}};
        const auto             chain = runFrameChain(fixture, domain, frames, gr::property_map{});
        expect(domain == "host" || chain.source._deviceCalls > 0UZ) << domain << "the device overload ran";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "tensors between two blocks on one device stay there"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::string      placed = std::string(domain) + "!";
        const gr::property_map frames{{"n_frames", gr::Size_t(500)}, {"frame_size", gr::Size_t(64)}, {"gr:compute_domain", placed}};
        const auto             chain = runFrameChain(fixture, domain, frames, {{"gr:compute_domain", placed}});
        expect(domain == "host" || (chain.source._deviceCalls > 0UZ && chain.sum._deviceCalls > 0UZ)) << domain << "both device overloads ran";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "tensor ports"_test                        = host;
    "tensor ports on every served domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
