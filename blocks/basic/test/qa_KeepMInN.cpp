#include <boost/ut.hpp>

#include <algorithm>
#include <format>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/basic/KeepMInN.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::basic::countKeptBefore;
using gr::basic::KeepMInN;
using gr::testing::kProbeRate;
using gr::testing::kRateKey;
using gr::testing::kTriggerKey;
using gr::testing::ramp;

using Kept = gr::testing::ProbedChain<KeepMInN<float>>;

constexpr float kRateOf3In5 = kProbeRate * (3.f / 5.f);

struct Selection {
    gr::Size_t m;
    gr::Size_t n;
    gr::Size_t offset;
};

[[nodiscard]] gr::property_map settingsOf(Selection selection) { return {{"m", selection.m}, {"n", selection.n}, {"offset", selection.offset}}; }

[[nodiscard]] std::vector<float> keptOfRamp(std::size_t nChunks, Selection selection) {
    const auto isKept   = [selection](std::size_t i) { return (i + selection.n - selection.offset) % selection.n < selection.m; };
    const auto asSample = [](std::size_t i) { return static_cast<float>(i); };
    return std::views::iota(0UZ, nChunks * selection.n) | std::views::filter(isKept) | std::views::transform(asSample) | std::ranges::to<std::vector<float>>();
}

[[nodiscard]] std::vector<std::size_t> indicesWith(const gr::testing::ProbeSink<float>& sink, const std::string& key) {
    const auto carriesKey = [&key](const auto& tag) { return tag.map.contains(key); };
    return sink._tags | std::views::filter(carriesKey) | std::views::transform(&gr::testing::OwningTag::index) | std::ranges::to<std::vector<std::size_t>>();
}

const auto definitions = [] {
    "maps an input index to its output index"_test = [] {
        const auto outputIndices = [](Selection selection) {
            const auto outputIndex = [selection](std::size_t i) { return countKeptBefore(i, selection.m, selection.n, selection.offset); };
            return std::views::iota(0UZ, 10UZ) | std::views::transform(outputIndex) | std::ranges::to<std::vector<std::size_t>>();
        };
        expect(outputIndices({3, 5, 0}) == std::vector<std::size_t>{0, 1, 2, 3, 3, 3, 4, 5, 6, 6}) << "kept 0 1 2, 5 6 7";
        expect(outputIndices({3, 5, 4}) == std::vector<std::size_t>{0, 1, 2, 2, 2, 3, 4, 5, 5, 5}) << "kept 0 1 4, 5 6 9";
        expect(outputIndices({1, 4, 3}) == std::vector<std::size_t>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2}) << "kept 3, 7";
    };

    "keeps sample i when (i - offset) mod n < m"_test = [] {
        const std::vector<Selection> selections{{1, 2, 0}, {1, 3, 1}, {1, 7, 2}, {3, 5, 0}, {3, 5, 1}, {3, 5, 2}, {3, 5, 3}, {3, 5, 4}, {4, 4, 0}, {1, 6, 5}};
        for (const Selection selection : selections) {
            constexpr std::size_t kChunks = 20UZ;
            Kept                  kept(ramp(kChunks * selection.n), settingsOf(selection));
            expect(kept.run() == keptOfRamp(kChunks, selection)) << std::format("m {} n {} offset {}", selection.m, selection.n, selection.offset);
        }
    };

    // input     0 1 2 3 4 | 5 6 7 8 9
    // offset 0  ■ ■ ■ · · | ■ ■ ■ · ·
    // offset 1  · ■ ■ ■ · | · ■ ■ ■ ·
    // offset 2  · · ■ ■ ■ | · · ■ ■ ■
    // offset 3  ■ · · ■ ■ | ■ · · ■ ■
    // offset 4  ■ ■ · · ■ | ■ ■ · · ■
    "offset starts the kept run, which wraps within the chunk"_test = [] {
        const std::vector<std::vector<float>> expected{{0, 1, 2, 5, 6, 7}, {1, 2, 3, 6, 7, 8}, {2, 3, 4, 7, 8, 9}, {0, 3, 4, 5, 8, 9}, {0, 1, 4, 5, 6, 9}};
        for (gr::Size_t offset = 0; offset < 5; offset++) {
            Kept kept(ramp(10UZ), settingsOf({3, 5, offset}));
            expect(kept.run() == expected[offset]) << std::format("offset {}", offset);
        }
    };

    "drops an incomplete last chunk"_test = [] {
        Kept kept(ramp(12UZ), settingsOf({1, 5, 0}));
        expect(kept.run() == std::vector<float>{0.f, 5.f});
    };

    "scales the rate by m/n"_test = [] {
        Kept kept(ramp(50UZ), settingsOf({3, 5, 0}));
        std::ignore = kept.run();
        expect(eq(kept.sink->sample_rate, kRateOf3In5));
    };

    "moves a tag on a dropped sample to the next kept one"_test = [] {
        Kept kept(ramp(20UZ), settingsOf({3, 5, 0}));
        kept.source->_tags = {{2UZ, {{kTriggerKey, "kept"}}}, {4UZ, {{kTriggerKey, "dropped, last of a chunk"}}}, {8UZ, {{kTriggerKey, "dropped"}}}};
        std::ignore        = kept.run();
        expect(indicesWith(*kept.sink, kTriggerKey) == std::vector<std::size_t>{2, 3, 6}) << "inputs 2, 4 and 8 go to the outputs of inputs 2, 5 and 10";
    };

    "applies a new m and its rate from the next chunk"_test = [] {
        Kept kept(ramp(30UZ), {{"n", gr::Size_t(5)}}); // m left at its default: a setting given here ignores tags
        kept.source->_tags = {{12UZ, {{"m", gr::Size_t(3)}}}};
        expect(kept.run() == std::vector<float>{0.f, 5.f, 10.f, 15.f, 16.f, 17.f, 20.f, 21.f, 22.f, 25.f, 26.f, 27.f});
        expect(eq(kept.sink->sample_rate, kRateOf3In5));
        const auto announced = std::ranges::find_if(kept.sink->_tags, [](const auto& tag) { return tag.index == 3UZ && tag.map.contains(kRateKey); });
        expect(fatal(announced != kept.sink->_tags.end())) << "input 15 is output 3";
        expect(eq(announced->map.template value_or<float>(kRateKey, 0.f), kRateOf3In5));
    };

    "refuses m > n or offset >= n"_test = [] {
        KeepMInN<float> block(settingsOf({2, 4, 1}));
        block.settings().init();
        std::ignore = block.settings().applyStagedParameters();
        const std::vector<gr::property_map> refusals{{{"m", gr::Size_t(5)}}, {{"offset", gr::Size_t(4)}}, {{"n", gr::Size_t(1)}}};
        for (const gr::property_map& refused : refusals) {
            std::ignore = block.settings().set(refused);
            std::ignore = block.settings().applyStagedParameters();
            expect(eq(block.m.value, 2U) && eq(block.n.value, 4U) && eq(block.offset.value, 1U));
            expect(eq(block.input_chunk_size.value, 4U) && eq(block.output_chunk_size.value, 2U));
        }
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "same samples and tags on every domain"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr Selection   kSelection{3, 5, 4};
        constexpr std::size_t kChunks = 64UZ;
        auto&                 source  = gr::testing::sourceOf(fixture, ramp(kChunks * kSelection.n));
        source._tags                  = {{2UZ, {{kTriggerKey, "dropped"}}}, {9UZ, {{kTriggerKey, "kept"}}}};
        gr::property_map settings     = settingsOf(kSelection);
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
        auto& keep = fixture.template emplace<KeepMInN<float>>(std::move(settings));
        auto& sink = fixture.template emplace<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(fixture.template connect<"out", "in">(source, keep).has_value());
        expect(fixture.template connect<"out", "in">(keep, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
        expect(gr::testing::samplesOf(sink) == keptOfRamp(kChunks, kSelection)) << domain;
        expect(indicesWith(sink, kTriggerKey) == std::vector<std::size_t>{2, 5}) << domain << ": inputs 2 and 9 go to the outputs of inputs 4 and 9";
        expect(eq(sink.sample_rate, kRateOf3In5)) << domain;
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

} // namespace

int main() {
    "KeepMInN"_test                 = definitions;
    "KeepMInN on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
