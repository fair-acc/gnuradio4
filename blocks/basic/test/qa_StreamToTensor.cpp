#include <boost/ut.hpp>

#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/basic/StreamToTensor.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::basic::StreamsToTensor;
using gr::basic::TensorToStreams;

using gr::testing::kFrameRateKey;
using gr::testing::kProbeRate;
using gr::testing::kRateKey;
using gr::testing::kTriggerKey;
using gr::testing::ProbeSink;
using gr::testing::ProbeSource;
using gr::testing::ramp;
using gr::testing::samplesOf;
using gr::testing::sourceOf;
using gr::testing::tagAt;

using TensorSink = gr::testing::TensorCollector<float>;

const std::string kTimeKey{gr::tag::kTriggerTime.key()};
const std::string kOffsetKey{gr::tag::kTriggerOffset.key()};

[[nodiscard]] std::vector<std::pair<std::size_t, std::uint64_t>> triggerTimesByTensor(const TensorSink& sink) {
    return sink._tags | std::views::filter([](const std::pair<std::size_t, gr::property_map>& tag) { return tag.second.contains(kTimeKey); }) //
           | std::views::transform([](const std::pair<std::size_t, gr::property_map>& tag) { return std::pair{tag.first, tag.second.value_or<std::uint64_t>(kTimeKey, std::uint64_t{0})}; }) | std::ranges::to<std::vector<std::pair<std::size_t, std::uint64_t>>>();
}

[[nodiscard]] std::vector<std::size_t> tensorsTagged(const TensorSink& sink, const std::string& key) {
    return sink._tags | std::views::filter([&](const auto& tag) { return tag.second.contains(key); }) | std::views::keys | std::ranges::to<std::vector<std::size_t>>();
}

[[nodiscard]] float lastTagged(const TensorSink& sink, const std::string& key) {
    const auto latestFirst = sink._tags | std::views::reverse;
    const auto tag         = std::ranges::find_if(latestFirst, [&](const auto& t) { return t.second.contains(key); });
    return tag == latestFirst.end() ? 0.f : tag->second.value_or<float>(key, 0.f);
}

using WideTensorSource = gr::testing::TensorSource<float>;

[[nodiscard]] WideTensorSource& wideSourceOf(auto& fixture, const std::vector<std::size_t>& widths, std::size_t wrongKAt = std::numeric_limits<std::size_t>::max()) {
    auto&       source = fixture.template emplace<WideTensorSource>();
    std::size_t sample = 0UZ;
    for (std::size_t i = 0UZ; i < widths.size(); ++i) {
        const std::size_t width = std::max(widths[i], 1UZ);
        const std::size_t rows  = i == wrongKAt ? 2UZ : 3UZ;
        auto&             frame = source._frames.emplace_back();
        frame.extents           = widths[i] == 0UZ ? std::vector<std::size_t>{rows} : std::vector<std::size_t>{rows, width};
        for (std::size_t port = 0UZ; port < rows; ++port) {
            for (std::size_t j = 0UZ; j < width; ++j) {
                frame.elements.push_back(static_cast<float>(100UZ * port + sample + j));
            }
        }
        sample += width;
    }
    source._sampleRate = kProbeRate;
    return source;
}

const auto definitions = [] {
    "firstWindowHolding counts windows as written out by hand"_test = [] {
        using gr::basic::firstWindowHolding;
        // size 4, stride 2: windows [0,4) [2,6) [4,8) [6,10) ...
        expect(eq(firstWindowHolding(3UZ, 4UZ, 2UZ), 0UZ));
        expect(eq(firstWindowHolding(5UZ, 4UZ, 2UZ), 1UZ)) << "[2,6) is the first to hold 5";
        expect(eq(firstWindowHolding(9UZ, 4UZ, 2UZ), 3UZ)) << "[6,10)";
        // size 2, stride 4: windows [0,2) [4,6) [8,10) ...; 2, 3, 6, 7 fall between
        expect(eq(firstWindowHolding(3UZ, 2UZ, 4UZ), 1UZ)) << "skipped: the next window, [4,6)";
        expect(eq(firstWindowHolding(5UZ, 2UZ, 4UZ), 1UZ));
        expect(eq(firstWindowHolding(9UZ, 2UZ, 4UZ), 2UZ));
        for (std::size_t i = 0UZ; i < 64UZ; ++i) {
            const std::size_t w = firstWindowHolding(i, 4UZ, 2UZ);
            expect(i < w * 2UZ + 4UZ && (w == 0UZ || i >= (w - 1UZ) * 2UZ + 4UZ)) << "window" << w << "is the first to hold" << i;
        }
    };

    "timeOfFrameStart moves a time by the samples after the frame start, as computed by hand"_test = [] {
        using gr::basic::timeOfFrameStart;
        constexpr std::uint64_t kT      = 1'700'000'000'000'000'000ULL;
        constexpr std::uint64_t kLatest = std::numeric_limits<std::uint64_t>::max();
        expect(eq(timeOfFrameStart(kT, 232, 1e6f), kT - 232'000ULL)) << "232 samples at 1 MS/s: 232 us earlier";
        expect(eq(timeOfFrameStart(kT, -424, 1e6f), kT + 424'000ULL)) << "a frame starting 424 samples after the sample: 424 us later";
        expect(eq(timeOfFrameStart(kT, 0, 1e6f), kT)) << "on the frame start";
        expect(eq(timeOfFrameStart(kT, 3, 48000.f), kT - 62'500ULL)) << "3 / 48 kHz = 62.5 us";
        expect(eq(timeOfFrameStart(kT, 1, 48000.f), kT - 20'833ULL)) << "1 / 48 kHz = 20833.3 ns, rounded";
        expect(eq(timeOfFrameStart(kT, 232, 0.f), kT)) << "unknown rate: unchanged";
        expect(eq(timeOfFrameStart(100ULL, 1, 1e6f), 0ULL)) << "1000 ns before 100 ns saturates at 0";
        expect(eq(timeOfFrameStart(kLatest - 10ULL, -1, 1e6f), kLatest)) << "1000 ns after 10 ns before the end saturates at the end";
    };

    "a tensor of size samples every stride samples, at a frame rate of input rate / stride; a tag on the first tensor holding its sample"_test = [] {
        struct Case {
            gr::Size_t               size;
            gr::Size_t               stride;
            std::size_t              nTensors;
            std::vector<std::size_t> triggerAt;
            std::size_t              nSamples = 20UZ;
        };
        for (const Case& c : std::vector<Case>{{4U, 2U, 9UZ, {1UZ, 3UZ, 6UZ}}, {4U, 4U, 5UZ, {1UZ, 2UZ, 3UZ}}, {2U, 4U, 5UZ, {1UZ, 2UZ, 4UZ}}, {2U, 4U, 5UZ, {1UZ, 2UZ, 4UZ, 4UZ}, 18UZ}, {4U, 0U, 5UZ, {1UZ, 2UZ, 3UZ}}, {1U, 0U, 20UZ, {5UZ, 9UZ, 14UZ}}}) {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = sourceOf(fixture, ramp(c.nSamples, 0.f));
            source._tags                       = {{5UZ, {{kTriggerKey, "five"}}}, {9UZ, {{kTriggerKey, "nine"}, {kRateKey, 2.f * kProbeRate}}}, {14UZ, {{kTriggerKey, "fourteen, skipped at size 2, stride 4"}}}};
            if (c.nSamples == 18UZ) {
                source._tags.push_back({17UZ, {{kTriggerKey, "seventeen, on the last tensor, short of its stride"}}});
            }
            auto& toTensor = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", c.size}, {"stride", c.stride}});
            auto& sink     = fixture.emplace<TensorSink>();
            expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
            expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
            expect(fixture.run().has_value());
            const std::size_t              hop   = c.stride == 0U ? c.size : c.stride;
            const std::vector<std::size_t> shape = c.size == 1U ? std::vector<std::size_t>{1UZ} : std::vector<std::size_t>{1UZ, c.size};
            expect(eq(sink._elements.size(), c.nTensors)) << "size" << c.size << "stride" << c.stride << ":" << c.nSamples << "samples, trailing ones short of a tensor stay";
            for (std::size_t w = 0UZ; w < sink._elements.size(); ++w) {
                expect(sink._elements[w] == ramp(c.size, static_cast<float>(w * hop))) << "tensor" << w;
                expect(sink._shapes[w] == shape) << "tensor" << w << "of one port is [1, size], or [1] for size 1";
            }
            expect(eq(lastTagged(sink, kRateKey), 2.f * kProbeRate)) << "the signal's 2 kHz from sample 9 on, unchanged by the framing";
            const float frameRate = hop == 1UZ ? 0.f : 2.f * kProbeRate / static_cast<float>(hop);
            expect(eq(lastTagged(sink, kFrameRateKey), frameRate)) << "tensors per second: 2 kHz /" << hop;
            expect(tensorsTagged(sink, kTriggerKey) == c.triggerAt) << "size" << c.size << "stride" << c.stride << std::format("got {}", tensorsTagged(sink, kTriggerKey));
        }
    };

    "a stride beyond 2^24 is refused and the previous one kept; one beyond the input buffer stops the stream"_test = [] {
        StreamsToTensor<float, 1UZ> toTensor({{"size", gr::Size_t(4)}, {"stride", gr::Size_t(3)}});
        toTensor.init(toTensor.progress);
        std::ignore = toTensor.settings().setStaged({{"stride", (1U << 24U) + 1U}});
        std::ignore = toTensor.settings().applyStagedParameters();
        expect(eq(toTensor.stride.value, 3U));

        gr::testing::GraphFixture<> fixture;
        auto&                       source  = sourceOf(fixture, ramp(64UZ, 0.f));
        auto&                       tooLong = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(2)}, {"stride", 1U << 22U}});
        auto&                       sink    = fixture.emplace<TensorSink>();
        expect(fixture.graph.connect(source, "out", tooLong, "in#0").has_value());
        expect(fixture.connect<"out", "in">(tooLong, sink).has_value());
        std::ignore = fixture.run();
        expect(sink._elements.empty());
    };

    "stream to tensor to stream is the identity; a tag moves to its tensor's first element"_test = []<typename T>() {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = sourceOf<T>(fixture, ramp<T>(64UZ, 0.f));
        source._tags                       = {{13UZ, {{kTriggerKey, "thirteen"}}}};
        auto& toTensor                     = fixture.emplace<StreamsToTensor<T, 1UZ>>({{"size", gr::Size_t(8)}});
        auto& toStream                     = fixture.emplace<TensorToStreams<T, 1UZ>>();
        auto& sink                         = fixture.emplace<ProbeSink<T>>({{"log_samples", true}});
        expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
        expect(fixture.connect<"out", "in">(toTensor, toStream).has_value());
        expect(fixture.graph.connect(toStream, "out#0", sink, "in").has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink) == ramp<T>(64UZ, 0.f));
        expect(eq(sink.sample_rate, kProbeRate));
        expect(tagAt(sink, 8UZ)) << "sample 13 is in tensor 1, whose first element is 8";
    } | std::tuple<float, std::complex<double>, std::uint8_t>{};

    "a tensor of another shape stops the N -> 1 merge"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       toTensor = fixture.emplace<StreamsToTensor<float, 2UZ>>({{"size", gr::Size_t(4)}});
        auto&                       toStream = fixture.emplace<TensorToStreams<float, 1UZ>>();
        auto&                       sink     = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        for (std::size_t port = 0UZ; port < 2UZ; ++port) {
            auto& source = sourceOf<float>(fixture, ramp<float>(32UZ, static_cast<float>(100UZ * port)));
            expect(fixture.graph.connect(source, "out", toTensor, std::format("in#{}", port)).has_value());
        }
        expect(fixture.connect<"out", "in">(toTensor, toStream).has_value());
        expect(fixture.graph.connect(toStream, "out#0", sink, "in").has_value());
        std::ignore = fixture.run();
        expect(sink._samples.empty());
    };

    "N streams to tensors of shape [N] and back is the identity; a tag goes to output 0, its sample_rate to all"_test = []<typename T>() {
        gr::testing::GraphFixture<>  fixture;
        std::vector<ProbeSource<T>*> sources;
        std::vector<ProbeSink<T>*>   sinks;
        auto&                        toTensor  = fixture.emplace<StreamsToTensor<T>>({{"n_ports", gr::Size_t(3)}});
        auto&                        toStreams = fixture.emplace<TensorToStreams<T, 3UZ>>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sources.push_back(&sourceOf<T>(fixture, ramp<T>(40UZ, static_cast<float>(50UZ * port))));
            sinks.push_back(&fixture.emplace<ProbeSink<T>>({{"log_samples", true}}));
            expect(fixture.graph.connect(*sources.back(), "out", toTensor, std::format("in#{}", port)).has_value());
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        sources[1]->_tags = {{5UZ, {{kTriggerKey, "five"}}}};
        expect(fixture.connect<"out", "in">(toTensor, toStreams).has_value());
        expect(fixture.run().has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<T>(40UZ, static_cast<float>(50UZ * port))) << "output" << port;
            expect(eq(sinks[port]->sample_rate, kProbeRate)) << "output" << port;
        }
        expect(tagAt(*sinks[0], 5UZ)) << "input 1 sample 5 is tensor 5, whose first element goes to output 0";
        expect(!tagAt(*sinks[1], 5UZ) && !tagAt(*sinks[2], 5UZ));
    } | std::tuple<float, std::complex<double>, std::uint8_t>{};

    "a tensor of another shape stops the 1 -> N split"_test = [] {
        gr::testing::GraphFixture<>    fixture;
        auto&                          source    = sourceOf(fixture, ramp(32UZ, 0.f));
        auto&                          toTensor  = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(2)}});
        auto&                          toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
        std::vector<ProbeSink<float>*> sinks;
        expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
        expect(fixture.connect<"out", "in">(toTensor, toStreams).has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        std::ignore = fixture.run();
        expect(std::ranges::all_of(sinks, [](const auto* sink) { return sink->_samples.empty(); }));
    };

    "M streams to [M,N] frames, channel-major, every stride samples: the frame rate is the input rate / stride; a tag on the first frame holding its sample"_test = [] {
        struct Case {
            gr::Size_t               stride;
            std::vector<std::size_t> starts;
            std::size_t              taggedFrame;
        };
        // 20 samples per input, size 4; a tag on input 2, sample 9 -- held first by [8,12), [6,10) and [6,10)
        for (const auto& [stride, starts, taggedFrame] : {Case{0U, {0UZ, 4UZ, 8UZ, 12UZ, 16UZ}, 2UZ}, Case{2U, {0UZ, 2UZ, 4UZ, 6UZ, 8UZ, 10UZ, 12UZ, 14UZ, 16UZ}, 3UZ}, Case{6U, {0UZ, 6UZ, 12UZ}, 1UZ}}) {
            gr::testing::GraphFixture<>      fixture;
            std::vector<ProbeSource<float>*> sources;
            auto&                            toTensor = fixture.emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(3)}, {"size", gr::Size_t(4)}, {"stride", stride}});
            auto&                            sink     = fixture.emplace<TensorSink>();
            for (std::size_t port = 0UZ; port < 3UZ; ++port) {
                sources.push_back(&sourceOf<float>(fixture, ramp<float>(20UZ, static_cast<float>(100UZ * port))));
                expect(fixture.graph.connect(*sources.back(), "out", toTensor, std::format("in#{}", port)).has_value());
            }
            sources[2]->_tags = {{9UZ, {{kTriggerKey, "nine"}}}};
            expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
            expect(fixture.run().has_value());

            expect(eq(sink._elements.size(), starts.size())) << "stride" << stride;
            for (std::size_t w = 0UZ; w < std::min(starts.size(), sink._elements.size()); ++w) {
                std::vector<float> expected;
                for (std::size_t port = 0UZ; port < 3UZ; ++port) {
                    for (std::size_t j = 0UZ; j < 4UZ; ++j) {
                        expected.push_back(static_cast<float>(100UZ * port + starts[w] + j));
                    }
                }
                expect(sink._elements[w] == expected) << "stride" << stride << "frame" << w;
            }
            const float hop = static_cast<float>(stride == 0U ? 4U : stride);
            expect(eq(lastTagged(sink, kRateKey), kProbeRate)) << "stride" << stride;
            expect(eq(lastTagged(sink, kFrameRateKey), kProbeRate / hop)) << "stride" << stride;
            expect(tensorsTagged(sink, kTriggerKey) == std::vector<std::size_t>{taggedFrame}) << "stride" << stride << ": the first frame holding sample 9";
        }
    };

    "M streams to [M,N] frames and back to M streams is the identity"_test = [] {
        gr::testing::GraphFixture<>      fixture;
        std::vector<ProbeSource<float>*> sources;
        std::vector<ProbeSink<float>*>   sinks;
        auto&                            toTensor  = fixture.emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(3)}, {"size", gr::Size_t(8)}});
        auto&                            toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sources.push_back(&sourceOf<float>(fixture, ramp<float>(64UZ, static_cast<float>(100UZ * port))));
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(*sources.back(), "out", toTensor, std::format("in#{}", port)).has_value());
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.connect<"out", "in">(toTensor, toStreams).has_value());
        expect(fixture.run().has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<float>(64UZ, static_cast<float>(100UZ * port))) << "output" << port;
            expect(eq(sinks[port]->sample_rate, kProbeRate)) << "output" << port << ": the signal's rate throughout";
        }
    };

    "[K,N] tensors with N changing go out in order, at the signal's sample rate"_test = [] {
        gr::testing::GraphFixture<>    fixture;
        std::vector<ProbeSink<float>*> sinks;
        auto&                          source    = wideSourceOf(fixture, {4UZ, 1UZ, 7UZ, 4UZ, 2UZ});
        auto&                          toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.connect<"out", "in">(source, toStreams).has_value());
        expect(fixture.run().has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<float>(18UZ, static_cast<float>(100UZ * port))) << "output" << port;
            expect(eq(sinks[port]->sample_rate, kProbeRate)) << "output" << port;
        }
    };

    "size 1 with stride 3: one [M] frame every 3 samples, at a frame rate of input rate / 3"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       toTensor = fixture.emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(2)}, {"stride", gr::Size_t(3)}});
        auto&                       sink     = fixture.emplace<TensorSink>();
        for (std::size_t port = 0UZ; port < 2UZ; ++port) {
            auto& source = sourceOf<float>(fixture, ramp<float>(30UZ, static_cast<float>(100UZ * port)));
            expect(fixture.graph.connect(source, "out", toTensor, std::format("in#{}", port)).has_value());
        }
        expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 10UZ)) << "frames at 0, 3, ..., 27";
        for (std::size_t w = 0UZ; w < sink._elements.size(); ++w) {
            expect(sink._elements[w] == std::vector<float>{float(3UZ * w), float(100UZ + 3UZ * w)}) << "frame" << w;
        }
        expect(eq(lastTagged(sink, kRateKey), kProbeRate));
        expect(eq(lastTagged(sink, kFrameRateKey), kProbeRate / 3.f));
    };

    "the same tag on every input reaches its frame once"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       toTensor = fixture.emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(3)}, {"size", gr::Size_t(4)}});
        auto&                       sink     = fixture.emplace<TensorSink>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            auto& source = sourceOf<float>(fixture, ramp<float>(16UZ, 0.f));
            expect(fixture.graph.connect(source, "out", toTensor, std::format("in#{}", port)).has_value());
        }
        expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(tensorsTagged(sink, kRateKey).size(), 1UZ)) << "three inputs, one sample_rate tag";
    };

    "tags on different inputs arrive in frame order, a skipped one at the next frame, one after the last frame nowhere"_test = [] {
        // size 4, stride 6, 14 samples: frames [0,4) and [6,10); 12 and 13 stay behind
        for (const bool single : {false, true}) {
            gr::testing::GraphFixture<>      fixture;
            std::vector<ProbeSource<float>*> sources;
            auto&                            sink = fixture.emplace<TensorSink>();
            for (std::size_t port = 0UZ; port < (single ? 1UZ : 2UZ); ++port) {
                sources.push_back(&sourceOf<float>(fixture, ramp<float>(14UZ, static_cast<float>(100UZ * port))));
            }
            sources[0]->_tags = {{5UZ, {{kTriggerKey, "skipped"}}}, {11UZ, {{kTriggerKey, "after the last frame"}}}};
            if (single) {
                auto& toTensor = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(4)}, {"stride", gr::Size_t(6)}});
                expect(fixture.graph.connect(*sources[0], "out", toTensor, "in#0").has_value());
                expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
            } else {
                sources[1]->_tags = {{1UZ, {{kTriggerKey, "first frame"}}}};
                auto& toTensor    = fixture.emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(2)}, {"size", gr::Size_t(4)}, {"stride", gr::Size_t(6)}});
                for (std::size_t port = 0UZ; port < 2UZ; ++port) {
                    expect(fixture.graph.connect(*sources[port], "out", toTensor, std::format("in#{}", port)).has_value());
                }
                expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
            }
            expect(fixture.run().has_value());
            expect(eq(sink._elements.size(), 2UZ)) << (single ? "single" : "two inputs");
            const std::vector<std::size_t> expected = single ? std::vector<std::size_t>{1UZ} : std::vector<std::size_t>{0UZ, 1UZ};
            expect(tensorsTagged(sink, kTriggerKey) == expected) << (single ? "single" : "two inputs") << ": sample 5 goes to frame 1, sample 11 nowhere";
        }
    };

    "a frame carries the time of its first sample: a trigger_time on overlapping frames moves earlier by its offset into the frame"_test = [] {
        // 1 MS/s, size 1024, stride 256, 4096 samples: frames start at 0, 256, ..., 3072 (13 frames)
        constexpr std::uint64_t     kT1 = 1'700'000'000'000'000'000ULL;
        constexpr std::uint64_t     kT2 = 1'700'000'000'001'000'000ULL;
        gr::testing::GraphFixture<> fixture;
        auto&                       source = sourceOf(fixture, ramp(4096UZ, 0.f), 1e6f);
        source._tags                       = {{1000UZ, {{kTriggerKey, "thousand"}, {kTimeKey, kT1}}}, {1500UZ, {{kTriggerKey, "no time"}}}, {2000UZ, {{kTriggerKey, "two thousand"}, {kTimeKey, kT2}, {kOffsetKey, 0.25f}}}};
        auto& toTensor                     = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(1024)}, {"stride", gr::Size_t(256)}});
        auto& sink                         = fixture.emplace<TensorSink>();
        expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
        expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 13UZ));
        expect(tensorsTagged(sink, kTriggerKey) == std::vector<std::size_t>{0UZ, 2UZ, 4UZ}) << std::format("got {}", tensorsTagged(sink, kTriggerKey));
        const std::vector<std::pair<std::size_t, std::uint64_t>> expected{{0UZ, kT1 - 1'000'000ULL}, {4UZ, kT2 - 976'000ULL}};
        expect(triggerTimesByTensor(sink) == expected) << std::format("got {}", triggerTimesByTensor(sink));
        expect(eq(lastTagged(sink, kOffsetKey), 0.25f)) << "trigger_offset unchanged";
    };

    "a frame carries the time of its first sample: a trigger_time on a skipped sample moves later by the gap, one on a frame start stays"_test = [] {
        // 1 MS/s, size 256, stride 1024, 4096 samples: frames [0,256), [1024,1280), [2048,2304), [3072,3328)
        constexpr std::uint64_t     kT1 = 1'700'000'000'000'000'000ULL;
        constexpr std::uint64_t     kT2 = 1'700'000'000'002'000'000ULL;
        constexpr std::uint64_t     kT3 = 1'700'000'000'003'000'000ULL;
        gr::testing::GraphFixture<> fixture;
        auto&                       source = sourceOf(fixture, ramp(4096UZ, 0.f), 1e6f);
        source._tags                       = {{600UZ, {{kTimeKey, kT1}}}, {2100UZ, {{kTimeKey, kT2}}}, {3072UZ, {{kTimeKey, kT3}}}};
        auto& toTensor                     = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(256)}, {"stride", gr::Size_t(1024)}});
        auto& sink                         = fixture.emplace<TensorSink>();
        expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
        expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 4UZ));
        const std::vector<std::pair<std::size_t, std::uint64_t>> expected{{1UZ, kT1 + 424'000ULL}, {2UZ, kT2 - 52'000ULL}, {3UZ, kT3}};
        expect(triggerTimesByTensor(sink) == expected) << std::format("got {}", triggerTimesByTensor(sink));
    };

    "a trigger_time stays unchanged on its frame while the sample rate is unknown"_test = [] {
        // size 1024, stride 256: sample 2000 is first held by frame 4 [1024,2048)
        constexpr std::uint64_t     kT = 1'700'000'000'000'000'000ULL;
        gr::testing::GraphFixture<> fixture;
        auto&                       source = sourceOf(fixture, ramp(4096UZ, 0.f), 0.f);
        source._tags                       = {{2000UZ, {{kTimeKey, kT}}}};
        auto& toTensor                     = fixture.emplace<StreamsToTensor<float, 1UZ>>({{"size", gr::Size_t(1024)}, {"stride", gr::Size_t(256)}});
        auto& sink                         = fixture.emplace<TensorSink>();
        expect(fixture.graph.connect(source, "out", toTensor, "in#0").has_value());
        expect(fixture.connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value());
        const std::vector<std::pair<std::size_t, std::uint64_t>> expected{{4UZ, kT}};
        expect(triggerTimesByTensor(sink) == expected) << std::format("got {}", triggerTimesByTensor(sink));
    };

    "[K] and [K,N] tensors mixed, N = 1 included, go out in order"_test = [] {
        gr::testing::GraphFixture<>    fixture;
        std::vector<ProbeSink<float>*> sinks;
        auto&                          source    = wideSourceOf(fixture, {0UZ, 2UZ, 1UZ, 0UZ, 3UZ});
        auto&                          toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.connect<"out", "in">(source, toStreams).has_value());
        expect(fixture.run().has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<float>(8UZ, static_cast<float>(100UZ * port))) << "output" << port;
        }
    };

    "a wrong K after [K,N] tensors stops after them; so does a tensor wider than the output buffer"_test = [] {
        for (const bool wide : {false, true}) {
            gr::testing::GraphFixture<>    fixture;
            std::vector<ProbeSink<float>*> sinks;
            auto&                          source    = wideSourceOf(fixture, {4UZ, 2UZ, wide ? 1UZ << 20U : 4UZ, 4UZ}, wide ? std::numeric_limits<std::size_t>::max() : 2UZ);
            auto&                          toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
            for (std::size_t port = 0UZ; port < 3UZ; ++port) {
                sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
                expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
            }
            expect(fixture.connect<"out", "in">(source, toStreams).has_value());
            std::ignore = fixture.run();
            for (std::size_t port = 0UZ; port < 3UZ; ++port) {
                expect(samplesOf(*sinks[port]) == ramp<float>(6UZ, static_cast<float>(100UZ * port))) << (wide ? "wide" : "wrong K") << "output" << port;
            }
        }
    };

    "a sample_rate tag on a tensor that does not fit the first call arrives once, unchanged, without a frame rate"_test = [] {
        gr::testing::GraphFixture<>    fixture;
        std::vector<ProbeSink<float>*> sinks;
        auto&                          source    = wideSourceOf(fixture, {600UZ, 3UZ});
        auto&                          toStreams = fixture.emplace<TensorToStreams<float, 3UZ>>();
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.connect<"out", "in">(source, toStreams).has_value());
        expect(fixture.run().has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            const auto nRateTags = std::ranges::count_if(sinks[port]->_tags, [](const auto& tag) { return tag.map.contains(kRateKey); });
            expect(eq(nRateTags, 1)) << "output" << port;
            expect(eq(sinks[port]->sample_rate, kProbeRate)) << "output" << port << ": the signal's rate, not scaled";
            const auto nFrameRateTags = std::ranges::count_if(sinks[port]->_tags, [](const auto& tag) { return tag.map.contains(kFrameRateKey); });
            expect(eq(nFrameRateTags, 0)) << "output" << port << ": a stream again";
            expect(samplesOf(*sinks[port]) == ramp<float>(603UZ, static_cast<float>(100UZ * port))) << "output" << port;
        }
    };

    "a tag on a later tensor lands on that tensor's first sample of output 0, after the samples of the tensors before it; n_ports sizes the outputs of the dynamic variant"_test = [] {
        gr::testing::GraphFixture<>    fixture;
        std::vector<ProbeSink<float>*> sinks;
        auto&                          source = wideSourceOf(fixture, {4UZ, 1UZ, 7UZ, 4UZ, 2UZ});
        source._frameTags                     = {{2UZ, {{kTriggerKey, "second-but-one"}}}, {4UZ, {{kTriggerKey, "last"}}}};
        auto& toStreams                       = fixture.emplace<TensorToStreams<float>>({{"n_ports", gr::Size_t(3)}});
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.connect<"out", "in">(source, toStreams).has_value());
        expect(fixture.run().has_value());
        expect(eq(toStreams.out.size(), 3UZ));
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<float>(18UZ, static_cast<float>(100UZ * port))) << "output" << port;
        }
        expect(tagAt(*sinks[0], 5UZ)) << "tensors of 4 and 1 samples precede the third";
        expect(tagAt(*sinks[0], 16UZ)) << "tensors of 4, 1, 7 and 4 samples precede the fifth";
        expect(!tagAt(*sinks[1], 5UZ) && !tagAt(*sinks[2], 16UZ));
    };

    "n_ports follows nPorts when that is fixed at compile time"_test = [] {
        StreamsToTensor<float, 3UZ> toTensor({{"n_ports", gr::Size_t(2)}});
        toTensor.init(toTensor.progress);
        expect(eq(toTensor.n_ports.value, 3U));
        TensorToStreams<float, 1UZ> single;
        single.init(single.progress);
        expect(eq(single.n_ports.value, 1U));
    };
};

} // namespace

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "M streams to [M,N] frames and back is the identity with both blocks on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::string              placed = std::string(domain) + "!";
        std::vector<ProbeSink<float>*> sinks;
        auto&                          toTensor  = fixture.template emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(3)}, {"size", gr::Size_t(8)}, {"gr:compute_domain", placed}});
        auto&                          toStreams = fixture.template emplace<TensorToStreams<float, 3UZ>>({{"gr:compute_domain", placed}});
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            auto& source = sourceOf<float>(fixture, ramp<float>(256UZ, static_cast<float>(1000UZ * port)));
            sinks.push_back(&fixture.template emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(source, "out", toTensor, std::format("in#{}", port)).has_value());
            expect(fixture.graph.connect(toStreams, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.template connect<"out", "in">(toTensor, toStreams).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            expect(samplesOf(*sinks[port]) == ramp<float>(256UZ, static_cast<float>(1000UZ * port))) << domain << "output" << port;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "overlapping [M,N] frames are cut on each served domain as on the host"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& toTensor = fixture.template emplace<StreamsToTensor<float>>({{"n_ports", gr::Size_t(2)}, {"size", gr::Size_t(4)}, {"stride", gr::Size_t(2)}, {"gr:compute_domain", std::string(domain) + "!"}});
        auto& sink     = fixture.template emplace<TensorSink>();
        for (std::size_t port = 0UZ; port < 2UZ; ++port) {
            auto& source = sourceOf<float>(fixture, ramp<float>(64UZ, static_cast<float>(100UZ * port)));
            expect(fixture.graph.connect(source, "out", toTensor, std::format("in#{}", port)).has_value());
        }
        expect(fixture.template connect<"out", "in">(toTensor, sink).has_value());
        expect(fixture.run().has_value()) << domain;
        expect(eq(sink._elements.size(), 31UZ)) << domain << "frames start at 0, 2, ..., 60";
        for (std::size_t w = 0UZ; w < sink._elements.size(); ++w) {
            const std::vector<float> expected{float(2UZ * w), float(2UZ * w + 1UZ), float(2UZ * w + 2UZ), float(2UZ * w + 3UZ), float(100UZ + 2UZ * w), float(101UZ + 2UZ * w), float(102UZ + 2UZ * w), float(103UZ + 2UZ * w)};
            expect(sink._elements[w] == expected) << domain << "frame" << w;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

int main() {
    "Stream <-> Tensor"_test                        = definitions;
    "Stream <-> Tensor on every served domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
