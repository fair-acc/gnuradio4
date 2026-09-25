#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <format>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/basic/TensorMonitor.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::basic::TensorMonitor;

const std::string kTriggerKey{gr::tag::TRIGGER_NAME.key()};

using Source = gr::testing::TensorSource<float>;
template<typename T, typename... Attributes>
using Collector = gr::testing::TensorCollector<T, Attributes...>;

constexpr std::size_t kShapeWidth = TensorMonitor<float>::kShapeSize;
static_assert(kShapeWidth == 1UZ + gr::detail::kMaxRank);

/** the `shape` tensor for a report: the rank, then the extents, zero-padded to `kShapeWidth` */
[[nodiscard]] std::vector<gr::Size_t> paddedShape(std::vector<gr::Size_t> rankAndExtents) {
    rankAndExtents.resize(kShapeWidth, 0U);
    return rankAndExtents;
}

template<typename T>
[[nodiscard]] T valueAt(std::size_t frame, std::size_t j) {
    if constexpr (gr::meta::complex_like<T>) {
        return T(static_cast<typename T::value_type>(frame * 1000UZ + j), -static_cast<typename T::value_type>(j));
    } else {
        return static_cast<T>(frame * 1000UZ + j);
    }
}

[[nodiscard]] std::vector<Source::Frame> framesOf(const std::vector<std::vector<std::size_t>>& shapes, std::size_t nFrames) {
    std::vector<Source::Frame> frames(nFrames);
    for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
        frames[frame].extents  = shapes[frame % shapes.size()];
        const std::size_t size = std::ranges::fold_left(frames[frame].extents, 1UZ, std::multiplies<>{});
        for (std::size_t j = 0UZ; j < size; ++j) {
            frames[frame].elements.push_back(static_cast<float>(frame * 1000UZ + j));
        }
    }
    return frames;
}

[[nodiscard]] std::vector<std::size_t> triggerAt(const auto& collector) {
    std::vector<std::size_t> indices;
    for (const auto& [index, map] : collector._tags) {
        if (map.contains(kTriggerKey)) {
            indices.push_back(index);
        }
    }
    return indices;
}

[[nodiscard]] bool contentsIntact(const auto& collector, const std::vector<std::vector<std::size_t>>& shapes, std::size_t nFrames) {
    if (collector._elements.size() != nFrames) {
        return false;
    }
    for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
        const auto& shape = shapes[frame % shapes.size()];
        if (collector._shapes[frame] != shape) {
            return false;
        }
        for (std::size_t j = 0UZ; j < collector._elements[frame].size(); ++j) {
            if (collector._elements[frame][j] != static_cast<float>(frame * 1000UZ + j)) {
                return false;
            }
        }
    }
    return true;
}

const auto host = [] {
    "20000 tensors of three shapes pass through whole, their contents intact after the pool recycles"_test = [] {
        const std::vector<std::vector<std::size_t>> shapes{{64UZ}, {8UZ, 8UZ}, {65UZ}};
        gr::testing::GraphFixture<>                 fixture;
        auto&                                       source  = fixture.emplace<Source>();
        auto&                                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                                       sink    = fixture.emplace<Collector<float>>();
        source._frames                                      = framesOf(shapes, 20000UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.run().has_value());
        expect(contentsIntact(sink, shapes, 20000UZ));
    };

    "each change of shape goes out once, as the rank and the zero-padded extents"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source  = fixture.emplace<Source>();
        auto&                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                       sink    = fixture.emplace<Collector<float>>();
        auto&                       shapes  = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                      = framesOf({{4UZ}, {4UZ}, {2UZ, 8UZ}, {2UZ, 8UZ}, {2UZ, 8UZ}, {3UZ}}, 6UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, shapes).has_value());
        expect(fixture.run().has_value());

        expect(eq(shapes._elements.size(), 3UZ)) << "[4] once, [2,8] once, [3] once";
        expect(shapes._elements.size() == 3UZ && shapes._elements[0] == paddedShape({1U, 4U}));
        expect(shapes._elements.size() == 3UZ && shapes._elements[1] == paddedShape({2U, 2U, 8U}));
        expect(shapes._elements.size() == 3UZ && shapes._elements[2] == paddedShape({1U, 3U}));
        expect(eq(sink._elements.size(), 6UZ)) << "every tensor passes through";
    };

    "a hundred shape changes through a small shape edge never hold the tensors back; the reports that arrive are in order"_test = [] {
        std::vector<std::vector<std::size_t>> shapes;
        for (std::size_t i = 0UZ; i < 100UZ; ++i) {
            shapes.push_back({4UZ + i});
        }
        gr::testing::GraphFixture<> fixture;
        auto&                       source  = fixture.emplace<Source>();
        auto&                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                       sink    = fixture.emplace<Collector<float>>();
        auto&                       reports = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                      = framesOf(shapes, 100UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.graph.connect(monitor, "shape", reports, "in", gr::EdgeParameters{.minBufferSize = 2UZ}).has_value());
        expect(fixture.run().has_value());
        expect(contentsIntact(sink, shapes, 100UZ)) << "every tensor passes, however full `shape` is";
        expect(!reports._elements.empty()) << "the first change finds `shape` empty";
        const bool inOrder = std::ranges::is_sorted(reports._elements, {}, [](const auto& report) { return report[1]; });
        const bool rank1   = std::ranges::all_of(reports._elements, [](const auto& report) { return report[0] == 1U && report[1] >= 4U && report[1] <= 103U; });
        expect(inOrder && rank1) << "each report is a shape that occurred, in order";
    };

    "a tag travels with its tensor to out"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source  = fixture.emplace<Source>();
        auto&                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                       sink    = fixture.emplace<Collector<float>>();
        auto&                       shapes  = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                      = framesOf({{4UZ}, {6UZ}}, 20UZ);
        source._frameTags                   = {{7UZ, gr::property_map{{kTriggerKey, "here"}}}};
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, shapes).has_value());
        expect(fixture.run().has_value());
        expect(triggerAt(sink) == std::vector<std::size_t>{7UZ}) << "tensor 7 carries the tag";
        expect(triggerAt(shapes).empty()) << "the shape port counts shape changes, not tensors";
    };

    "a tensor of rank 3 reports [3, extents, 0, ...]; one of the highest rank reports every extent, with no padding left"_test = [] {
        constexpr std::size_t                       kMaxRank = gr::detail::kMaxRank;
        gr::testing::GraphFixture<>                 fixture;
        auto&                                       source  = fixture.emplace<Source>();
        auto&                                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                                       sink    = fixture.emplace<Collector<float>>();
        auto&                                       reports = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        const std::vector<std::vector<std::size_t>> shapes{{2UZ, 3UZ, 4UZ}, std::vector<std::size_t>(kMaxRank, 2UZ)};
        source._frames = framesOf(shapes, 2UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, reports).has_value());
        expect(fixture.run().has_value());
        expect(contentsIntact(sink, shapes, 2UZ)) << "24 and 2^kMaxRank elements pass through";
        std::vector<gr::Size_t> highestRank{static_cast<gr::Size_t>(kMaxRank)};
        highestRank.insert(highestRank.end(), kMaxRank, 2U);
        expect(eq(reports._elements.size(), 2UZ));
        expect(reports._elements.size() == 2UZ && reports._elements[0] == paddedShape({3U, 2U, 3U, 4U}));
        expect(reports._elements.size() == 2UZ && reports._elements[1] == highestRank);
    };

    "a shape that returns after another is reported again; the same shape twice in a row is not"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source  = fixture.emplace<Source>();
        auto&                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                       sink    = fixture.emplace<Collector<float>>();
        auto&                       reports = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                      = framesOf({{4UZ}, {4UZ}, {4UZ}, {2UZ, 2UZ}, {4UZ}}, 5UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, reports).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 5UZ));
        expect(eq(reports._elements.size(), 3UZ)) << "[4], then [2,2], then [4] again";
        expect(reports._elements.size() == 3UZ && reports._elements[0] == paddedShape({1U, 4U}));
        expect(reports._elements.size() == 3UZ && reports._elements[1] == paddedShape({2U, 2U, 2U}));
        expect(reports._elements.size() == 3UZ && reports._elements[2] == paddedShape({1U, 4U}));
    };

    "a thousand tensors of one shape report it exactly once"_test = [] {
        const std::vector<std::vector<std::size_t>> shapes{{3UZ, 5UZ}};
        gr::testing::GraphFixture<>                 fixture;
        auto&                                       source  = fixture.emplace<Source>();
        auto&                                       monitor = fixture.emplace<TensorMonitor<float>>();
        auto&                                       sink    = fixture.emplace<Collector<float>>();
        auto&                                       reports = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                                      = framesOf(shapes, 1000UZ);
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, reports).has_value());
        expect(fixture.run().has_value());
        expect(contentsIntact(sink, shapes, 1000UZ));
        expect(eq(reports._elements.size(), 1UZ));
        expect(reports._elements.size() == 1UZ && reports._elements[0] == paddedShape({2U, 3U, 5U}));
    };

    "tensors of double and of complex elements pass through intact, with their shape changes reported"_test = []<typename T>() {
        const std::vector<std::vector<std::size_t>> shapes{{5UZ}, {2UZ, 3UZ}, {2UZ, 3UZ}};
        gr::testing::GraphFixture<>                 fixture;
        auto&                                       source  = fixture.emplace<gr::testing::TensorSource<T>>();
        auto&                                       monitor = fixture.emplace<TensorMonitor<T>>();
        auto&                                       sink    = fixture.emplace<Collector<T>>();
        auto&                                       reports = fixture.emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames.resize(shapes.size());
        for (std::size_t frame = 0UZ; frame < shapes.size(); ++frame) {
            source._frames[frame].extents = shapes[frame];
            const std::size_t size        = std::ranges::fold_left(shapes[frame], 1UZ, std::multiplies<>{});
            for (std::size_t j = 0UZ; j < size; ++j) {
                source._frames[frame].elements.push_back(valueAt<T>(frame, j));
            }
        }
        expect(fixture.connect<"out", "in">(source, monitor).has_value());
        expect(fixture.connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.connect<"shape", "in">(monitor, reports).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), shapes.size()));
        for (std::size_t frame = 0UZ; frame < std::min(sink._elements.size(), shapes.size()); ++frame) {
            expect(sink._shapes[frame] == shapes[frame]) << "frame" << frame;
            for (std::size_t j = 0UZ; j < sink._elements[frame].size(); ++j) {
                expect(sink._elements[frame][j] == valueAt<T>(frame, j)) << "frame" << frame << "element" << j;
            }
        }
        expect(eq(reports._elements.size(), 2UZ)) << "[5], then [2,3] once";
        expect(reports._elements.size() == 2UZ && reports._elements[0] == paddedShape({1U, 5U}));
        expect(reports._elements.size() == 2UZ && reports._elements[1] == paddedShape({2U, 2U, 3U}));
    } | std::tuple<double, std::complex<float>, std::complex<double>>{};

    "start forgets the last shape and any report still waiting, so the next tensor is reported again"_test = [] {
        using Monitor = TensorMonitor<float>;
        Monitor monitor;
        monitor._lastShape = Monitor::Shape{1U, 4U};
        monitor._pending   = Monitor::Shape{2U, 2U, 2U};
        monitor.start();
        expect(monitor._lastShape == Monitor::kNoShape);
        expect(!monitor._pending.has_value());
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "tensors pass through a monitor placed on each served domain, contents intact"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::vector<std::vector<std::size_t>> shapes{{32UZ}, {4UZ, 8UZ}};
        auto&                                       source  = fixture.template emplace<Source>();
        auto&                                       monitor = fixture.template emplace<TensorMonitor<float>>({{"gr:compute_domain", std::string(domain) + "!"}});
        auto&                                       sink    = fixture.template emplace<Collector<float>>();
        auto&                                       reports = fixture.template emplace<Collector<gr::Size_t, gr::Async>>();
        source._frames                                      = framesOf(shapes, 500UZ);
        expect(fixture.template connect<"out", "in">(source, monitor).has_value());
        expect(fixture.template connect<"out", "in">(monitor, sink).has_value());
        expect(fixture.template connect<"shape", "in">(monitor, reports).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        expect(contentsIntact(sink, shapes, 500UZ)) << domain;
        expect(eq(reports._elements.size(), 500UZ)) << domain << "the shape alternates, so every tensor changes it";
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
} // namespace

int main() {
    "TensorMonitor"_test                        = host;
    "TensorMonitor on every served domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
