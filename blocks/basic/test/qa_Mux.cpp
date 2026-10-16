#include <boost/ut.hpp>

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/basic/Mux.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::basic::Demux;
using gr::basic::Mux;
using gr::testing::kProbeRate;
using gr::testing::kRateKey;
using gr::testing::kTriggerKey;
using gr::testing::ProbeSink;
using gr::testing::ProbeSource;
using gr::testing::ramp;
using gr::testing::samplesOf;
using gr::testing::sourceOf;
using gr::testing::tagAt;

const auto turnLayout = [] {
    "a turn's layout places and finds samples as written out by hand"_test = [] {
        // lengths 2, 3, 1: a0 a1 b0 b1 b2 c0 | a2 a3 b3 b4 b5 c1 | ...
        constexpr gr::basic::TurnLayout kLayout = gr::basic::turnLayoutOf(std::array{2U, 3U, 1U});
        static_assert(kLayout.turn() == 6UZ && kLayout.length(1UZ) == 3UZ);
        expect(eq(kLayout.muxedIndex(0UZ, 0UZ), 0UZ));
        expect(eq(kLayout.muxedIndex(1UZ, 4UZ), 9UZ)) << "b4";
        expect(eq(kLayout.muxedIndex(2UZ, 1UZ), 11UZ)) << "c1";
        expect(eq(kLayout.portOf(9UZ), 1UZ));
        expect(eq(kLayout.portSamplesBefore(9UZ, 0UZ), 4UZ)) << "a0..a3 precede b4";
        expect(eq(kLayout.portSamplesBefore(9UZ, 1UZ), 4UZ)) << "b0..b3 precede b4";
        expect(eq(kLayout.portSamplesBefore(9UZ, 2UZ), 1UZ)) << "c0 precedes b4";
        for (std::size_t i = 0UZ; i < 60UZ; ++i) {
            const std::size_t port = kLayout.portOf(i);
            expect(eq(kLayout.muxedIndex(port, kLayout.indexInPort(i, port)), i)) << "the inverse at" << i;
            expect(eq(kLayout.indexInPort(i, port), kLayout.portSamplesBefore(i, port))) << "its own port's samples before" << i;
        }
    };
};

const auto equalLengths = [] {
    "equal lengths take turns from each input, at three times the rate; a tag lands on its sample"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a = sourceOf(fixture, ramp(4UZ, 0.f));
        auto&                       b = sourceOf(fixture, ramp(4UZ, 10.f));
        auto&                       c = sourceOf(fixture, ramp(4UZ, 20.f));
        b._tags                       = {{3UZ, {{kTriggerKey, "b3"}, {"local_key", 1}}}};
        auto& mux                     = fixture.emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 2U, 2U}}});
        auto& sink                    = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.graph.connect(c, "out", mux, "in#2").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink) == std::vector<float>{0.f, 1.f, 10.f, 11.f, 20.f, 21.f, 2.f, 3.f, 12.f, 13.f, 22.f, 23.f});
        expect(eq(sink.sample_rate, 3.f * kProbeRate));
        expect(tagAt(sink, 9UZ)) << "input 1, sample 3: (3 / 2) * 6 + 2 + 3 % 2 = 9, where 13 is";
        expect(tagAt(sink, 9UZ, "local_key")) << "a key without gr: goes on with the rest of its tag";
    };

    "tags on different inputs and in different turns each land on their own output sample, in output order"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a = sourceOf(fixture, ramp(8UZ, 0.f));
        auto&                       b = sourceOf(fixture, ramp(8UZ, 10.f));
        a._tags                       = {{5UZ, {{kTriggerKey, "late"}}}};
        b._tags                       = {{1UZ, {{kTriggerKey, "early"}}}};
        auto& mux                     = fixture.emplace<Mux<float, 2UZ>>({{"lengths", std::vector<gr::Size_t>{4U, 4U}}});
        auto& sink                    = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(tagAt(sink, 5UZ)) << "input 1, sample 1: 0 * 8 + 4 + 1 = 5";
        expect(tagAt(sink, 9UZ)) << "input 0, sample 5: 1 * 8 + 0 + 1 = 9";
    };

    "equal lengths deal turns to each output, at a third of the rate; a tag follows its sample, its sample_rate goes to every output"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source   = sourceOf(fixture, ramp(18UZ, 0.f));
        source._tags                         = {{9UZ, {{kTriggerKey, "nine"}, {kRateKey, 3.f * kProbeRate}}}};
        auto&                          demux = fixture.emplace<Demux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 2U, 2U}}});
        std::vector<ProbeSink<float>*> sinks;
        expect(fixture.connect<"out", "in">(source, demux).has_value());
        for (std::size_t port = 0UZ; port < 3UZ; ++port) {
            sinks.push_back(&fixture.emplace<ProbeSink<float>>({{"log_samples", true}}));
            expect(fixture.graph.connect(demux, std::format("out#{}", port), *sinks.back(), "in").has_value());
        }
        expect(fixture.run().has_value());
        expect(samplesOf(*sinks[0]) == std::vector<float>{0.f, 1.f, 6.f, 7.f, 12.f, 13.f});
        expect(samplesOf(*sinks[1]) == std::vector<float>{2.f, 3.f, 8.f, 9.f, 14.f, 15.f});
        expect(samplesOf(*sinks[2]) == std::vector<float>{4.f, 5.f, 10.f, 11.f, 16.f, 17.f});
        expect(std::ranges::all_of(sinks, [](const auto* sink) { return tagAt(*sink, 0UZ, kRateKey); })) << "the source's 1 kHz, as 1/3 kHz";
        expect(tagAt(*sinks[1], 3UZ)) << "input 9 goes to output 1 (9 % 6 = 3 lies in run 2..3) at 1 * 2 + 3 - 2 = 3";
        expect(std::ranges::none_of(sinks, [](const auto* sink) { return tagAt(*sink, 2UZ) || tagAt(*sink, 4UZ); }) && !tagAt(*sinks[0], 3UZ)) << "the trigger goes nowhere else";
        expect(tagAt(*sinks[0], 4UZ, kRateKey)) << "output 0 has a2, a3 before input 9";
        expect(tagAt(*sinks[2], 2UZ, kRateKey)) << "output 2 has c0, c1 before input 9";
        expect(eq(sinks[2]->sample_rate, kProbeRate)) << "3 kHz / 3";
    };

    "trailing samples short of a full turn stay behind, as in GR3's Interleave"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a    = sourceOf(fixture, ramp(5UZ, 0.f));
        auto&                       b    = sourceOf(fixture, ramp(5UZ, 10.f));
        auto&                       mux  = fixture.emplace<Mux<float, 2UZ>>({{"lengths", std::vector<gr::Size_t>{2U, 2U}}});
        auto&                       sink = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink) == std::vector<float>{0.f, 1.f, 10.f, 11.f, 2.f, 3.f, 12.f, 13.f});
    };

    "muxing then demuxing with equal lengths is the identity, with compile-time ports too"_test = []<typename T>() {
        gr::testing::GraphFixture<> fixture;
        auto&                       a     = sourceOf<T>(fixture, ramp<T>(64UZ, 0.f));
        auto&                       b     = sourceOf<T>(fixture, ramp<T>(64UZ, 100.f));
        auto&                       mux   = fixture.emplace<Mux<T, 2UZ>>({{"lengths", std::vector<gr::Size_t>{4U, 4U}}});
        auto&                       demux = fixture.emplace<Demux<T, 2UZ>>({{"lengths", std::vector<gr::Size_t>{4U, 4U}}});
        auto&                       sinkA = fixture.emplace<ProbeSink<T>>({{"log_samples", true}});
        auto&                       sinkB = fixture.emplace<ProbeSink<T>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, demux).has_value());
        expect(fixture.graph.connect(demux, "out#0", sinkA, "in").has_value());
        expect(fixture.graph.connect(demux, "out#1", sinkB, "in").has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sinkA) == ramp<T>(64UZ, 0.f));
        expect(samplesOf(sinkB) == ramp<T>(64UZ, 100.f));
        expect(eq(sinkA.sample_rate, kProbeRate)) << "the rate comes back too";
    } | std::tuple<float, std::complex<double>, std::uint8_t>{};
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "muxing and demuxing unequal lengths are the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const std::string placed = std::string(domain) + "!";
        auto&             a      = sourceOf(fixture, ramp(300UZ, 0.f), 300.f);
        auto&             b      = sourceOf(fixture, ramp(500UZ, 5000.f), 500.f);
        auto&             mux    = fixture.template emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{3U, 5U}}, {"gr:compute_domain", placed}});
        auto&             demux  = fixture.template emplace<Demux<float>>({{"lengths", std::vector<gr::Size_t>{3U, 5U}}, {"gr:compute_domain", placed}});
        auto&             sinkA  = fixture.template emplace<ProbeSink<float>>({{"log_samples", true}});
        auto&             sinkB  = fixture.template emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.template connect<"out", "in">(mux, demux).has_value());
        expect(fixture.graph.connect(demux, "out#0", sinkA, "in").has_value());
        expect(fixture.graph.connect(demux, "out#1", sinkB, "in").has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        expect(samplesOf(sinkA) == ramp(300UZ, 0.f)) << domain;
        expect(samplesOf(sinkB) == ramp(500UZ, 5000.f)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

template<typename T>
[[nodiscard]] float rateAt(const ProbeSink<T>& sink, std::size_t index) {
    const auto tag = std::ranges::find_if(sink._tags, [&](const auto& t) { return t.index == index && t.map.contains(kRateKey); });
    return tag == sink._tags.end() ? 0.f : tag->map.template value_or<float>(kRateKey, 0.f);
}

const auto unequalLengths = [] {
    "runs of lengths[p] from each input in turn; the output rate is input 0's times sum / lengths[0]"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a = sourceOf(fixture, ramp(6UZ, 0.f));
        auto&                       b = sourceOf(fixture, ramp(9UZ, 10.f), 1.5f * kProbeRate);
        b._tags                       = {{1UZ, {{kTriggerKey, "b1"}}}};
        auto& mux                     = fixture.emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
        auto& sink                    = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink) == std::vector<float>{0.f, 1.f, 10.f, 11.f, 12.f, 2.f, 3.f, 13.f, 14.f, 15.f, 4.f, 5.f, 16.f, 17.f, 18.f});
        expect(eq(sink.sample_rate, 2.5f * kProbeRate)) << "1 kHz * 5 / 2";
        expect(std::ranges::none_of(sink._tags, [](const auto& tag) { return tag.map.contains(kRateKey) && tag.map.template value_or<float>(kRateKey, 0.f) != 2.5f * kProbeRate; })) << "input 1's 1.5 kHz is dropped";
        expect(tagAt(sink, 3UZ, kTriggerKey)) << "input 1 sample 1 is output 3, where 11 is";
    };

    "a sample_rate tag on input 0 mid-stream announces the new output rate on its sample's place"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a = sourceOf(fixture, ramp(6UZ, 0.f));
        auto&                       b = sourceOf(fixture, ramp(9UZ, 10.f), 1.5f * kProbeRate);
        a._tags                       = {{4UZ, {{kRateKey, 2.f * kProbeRate}, {kTriggerKey, "a4"}}}};
        auto& mux                     = fixture.emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
        auto& sink                    = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(rateAt(sink, 10UZ), 5.f * kProbeRate)) << "input 0 sample 4 is output 10; 2 kHz * 5 / 2";
        expect(tagAt(sink, 10UZ, kTriggerKey)) << "the rest of the tag goes with it, in the same tag";
        expect(eq(sink.sample_rate, 5.f * kProbeRate));
    };

    "the stream ends when an ended input cannot fill its next run, though another input still holds samples"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a    = sourceOf(fixture, ramp(3UZ, 0.f));
        auto&                       b    = sourceOf(fixture, ramp(10UZ, 10.f));
        auto&                       mux  = fixture.emplace<Mux<float, 2UZ>>({{"lengths", std::vector<gr::Size_t>{2U, 2U}}});
        auto&                       sink = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink) == std::vector<float>{0.f, 1.f, 10.f, 11.f}) << "input 0 holds one sample of its second run; the partial turn stays behind";
    };

    "runs of lengths[p] to each output in turn, each at its share of the rate; a sample_rate tag reaches every output"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = sourceOf(fixture, ramp(15UZ, 0.f));
        source._tags                       = {{8UZ, {{kTriggerKey, "eight"}, {kRateKey, 2.f * kProbeRate}}}};
        auto& demux                        = fixture.emplace<Demux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
        auto& sink0                        = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        auto& sink1                        = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.connect<"out", "in">(source, demux).has_value());
        expect(fixture.graph.connect(demux, "out#0", sink0, "in").has_value());
        expect(fixture.graph.connect(demux, "out#1", sink1, "in").has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sink0) == std::vector<float>{0.f, 1.f, 5.f, 6.f, 10.f, 11.f});
        expect(samplesOf(sink1) == std::vector<float>{2.f, 3.f, 4.f, 7.f, 8.f, 9.f, 12.f, 13.f, 14.f});
        expect(eq(rateAt(sink0, 0UZ), 0.4f * kProbeRate)) << "1 kHz * 2 / 5";
        expect(eq(rateAt(sink1, 0UZ), 0.6f * kProbeRate)) << "1 kHz * 3 / 5";
        expect(tagAt(sink1, 4UZ, kTriggerKey)) << "input 8 is output 1 sample 4";
        expect(eq(rateAt(sink1, 4UZ), 1.2f * kProbeRate)) << "2 kHz * 3 / 5, in the same tag";
        expect(eq(rateAt(sink0, 4UZ), 0.8f * kProbeRate)) << "output 0 has written 0, 1, 5, 6 when input 8 arrives";
        expect(!tagAt(sink0, 4UZ, kTriggerKey));
    };

    "muxing then demuxing with the same lengths is the identity, rates included"_test = []<typename T>() {
        gr::testing::GraphFixture<> fixture;
        auto&                       a     = sourceOf<T>(fixture, ramp<T>(300UZ, 0.f), 300.f);
        auto&                       b     = sourceOf<T>(fixture, ramp<T>(700UZ, 1000.f), 700.f);
        auto&                       mux   = fixture.emplace<Mux<T, 2UZ>>({{"lengths", std::vector<gr::Size_t>{3U, 7U}}});
        auto&                       demux = fixture.emplace<Demux<T, 2UZ>>({{"lengths", std::vector<gr::Size_t>{3U, 7U}}});
        auto&                       sinkA = fixture.emplace<ProbeSink<T>>({{"log_samples", true}});
        auto&                       sinkB = fixture.emplace<ProbeSink<T>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, demux).has_value());
        expect(fixture.graph.connect(demux, "out#0", sinkA, "in").has_value());
        expect(fixture.graph.connect(demux, "out#1", sinkB, "in").has_value());
        expect(fixture.run().has_value());
        expect(samplesOf(sinkA) == ramp<T>(300UZ, 0.f));
        expect(samplesOf(sinkB) == ramp<T>(700UZ, 1000.f));
        expect(eq(sinkA.sample_rate, 300.f));
        expect(eq(sinkB.sample_rate, 700.f));
    } | std::tuple<float, std::complex<double>, std::uint8_t>{};

    "dense tags on long inputs all arrive, each on its sample, and the stream still ends"_test = [] {
        constexpr std::size_t       kSamplesPerInput = 1UZ << 17U;
        constexpr std::size_t       kTagPeriod       = 64UZ;
        gr::testing::GraphFixture<> fixture;
        auto&                       a = fixture.emplace<ProbeSource<float>>({{"n_samples_max", static_cast<gr::Size_t>(kSamplesPerInput)}, {"mark_tag", false}});
        auto&                       b = fixture.emplace<ProbeSource<float>>({{"n_samples_max", static_cast<gr::Size_t>(kSamplesPerInput)}, {"mark_tag", false}});
        for (std::size_t i = 0UZ; i < kSamplesPerInput; i += kTagPeriod) {
            a._tags.push_back({i, {{kTriggerKey, "a"}}});
            a._tags.push_back({i + kTagPeriod - 1UZ, {{"local_key", 1}}});
            b._tags.push_back({i, {{kTriggerKey, "b"}}});
            b._tags.push_back({i + kTagPeriod - 1UZ, {{"local_key", 1}}});
        }
        auto& mux  = fixture.emplace<Mux<float, 2UZ>>({{"lengths", std::vector<gr::Size_t>{1U, 1U}}});
        auto& sink = fixture.emplace<ProbeSink<float>>({{"log_samples", false}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._nSamplesProduced, static_cast<gr::Size_t>(2UZ * kSamplesPerInput)));
        const auto triggers = std::ranges::count_if(sink._tags, [](const auto& tag) { return tag.map.contains(kTriggerKey); });
        expect(eq(triggers, static_cast<std::ptrdiff_t>(2UZ * kSamplesPerInput / kTagPeriod)));
        expect(std::ranges::all_of(sink._tags, [](const auto& tag) { return !tag.map.contains(kTriggerKey) || tag.index % (2UZ * kTagPeriod) < 2UZ; })) << "input p sample k is output 2k + p";
    };

    "lengths with a zero run, or more runs than ports, are refused and the previous ones kept"_test = [] {
        Mux<float> mux({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
        mux.init(mux.progress);
        std::ignore = mux.settings().set({{"lengths", std::vector<gr::Size_t>{2U, 0U}}});
        std::ignore = mux.settings().applyStagedParameters();
        expect(std::ranges::equal(mux.lengths.value, std::vector<gr::Size_t>{2U, 3U}));
        expect(eq(mux.in.size(), 2UZ));

        Demux<float, 2UZ> demux({{"lengths", std::vector<gr::Size_t>{4U, 5U, 6U}}});
        demux.init(demux.progress);
        expect(std::ranges::equal(demux.lengths.value, std::vector<gr::Size_t>{1U, 1U})) << "three runs for two outputs";
    };

    "an input with a known sample_rate announces the multiplied (Mux) or divided (Demux) rate once, on the first output sample"_test = [] {
        const auto rateTagsOf = [](const auto& sink) { return std::ranges::count_if(sink._tags, [](const auto& tag) { return tag.map.contains(kRateKey); }); };
        {
            gr::testing::GraphFixture<> fixture;
            auto&                       a    = sourceOf(fixture, ramp(40UZ, 0.f), 1000.f);
            auto&                       b    = sourceOf(fixture, ramp(60UZ, 0.f), 1000.f);
            auto&                       mux  = fixture.emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
            auto&                       sink = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
            expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
            expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
            expect(fixture.connect<"out", "in">(mux, sink).has_value());
            expect(fixture.run().has_value());
            expect(eq(rateTagsOf(sink), std::ptrdiff_t{1}));
            expect(eq(rateAt(sink, 0UZ), 2500.f)) << "1 kHz * (2 + 3) / 2";
        }
        {
            gr::testing::GraphFixture<> fixture;
            auto&                       source = sourceOf(fixture, ramp(60UZ, 0.f), 1000.f);
            auto&                       demux  = fixture.emplace<Demux<float>>({{"lengths", std::vector<gr::Size_t>{2U, 3U}}});
            auto&                       sink0  = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
            auto&                       sink1  = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
            expect(fixture.connect<"out", "in">(source, demux).has_value());
            expect(fixture.graph.connect(demux, "out#0", sink0, "in").has_value());
            expect(fixture.graph.connect(demux, "out#1", sink1, "in").has_value());
            expect(fixture.run().has_value());
            expect(eq(rateTagsOf(sink0), std::ptrdiff_t{1}));
            expect(eq(rateTagsOf(sink1), std::ptrdiff_t{1}));
            expect(eq(rateAt(sink0, 0UZ), 400.f)) << "1 kHz * 2 / 5";
            expect(eq(rateAt(sink1, 0UZ), 600.f)) << "1 kHz * 3 / 5";
        }
    };

    "less than one complete turn of input publishes nothing"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a    = sourceOf(fixture, ramp(3UZ, 0.f));
        auto&                       b    = sourceOf(fixture, ramp(3UZ, 10.f));
        auto&                       mux  = fixture.emplace<Mux<float, 2UZ>>({{"lengths", std::vector<gr::Size_t>{4U, 4U}}});
        auto&                       sink = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.graph.connect(b, "out", mux, "in#1").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        expect(fixture.run().has_value());
        expect(sink._samples.empty()) << "3 samples per input is short of a turn of 4 + 4";
    };

    "an unconnected input stops the mux before it stalls"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       a    = sourceOf(fixture, ramp(4UZ, 0.f));
        auto&                       mux  = fixture.emplace<Mux<float>>({{"lengths", std::vector<gr::Size_t>{1U, 1U}}});
        auto&                       sink = fixture.emplace<ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.graph.connect(a, "out", mux, "in#0").has_value());
        expect(fixture.connect<"out", "in">(mux, sink).has_value());
        std::ignore = fixture.run();
        expect(sink._samples.empty()) << "no whole turn without input 1";
    };
};

} // namespace

int main() {
    "TurnLayout"_test                       = turnLayout;
    "Mux and Demux of equal lengths"_test   = equalLengths;
    "Mux and Demux on every domain"_test    = domains;
    "Mux and Demux of unequal lengths"_test = unequalLengths;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
