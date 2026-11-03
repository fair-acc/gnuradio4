#ifndef GNURADIO_TESTING_STREAM_PROBES_HPP
#define GNURADIO_TESTING_STREAM_PROBES_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace gr::testing {

inline constexpr float kProbeRate = 1000.f;

template<typename T>
using ProbeSource = TagSource<T, ProcessFunction::USE_PROCESS_BULK>;
template<typename T>
using ProbeSink = TagSink<T, ProcessFunction::USE_PROCESS_BULK>;

inline const std::string kRateKey{gr::tag::kSampleRate.key()};
inline const std::string kTriggerKey{gr::tag::kTriggerName.key()};
inline const std::string kFrameRateKey{gr::tag::kFrameRate.key()};

template<typename T = float>
[[nodiscard]] std::vector<T> ramp(std::size_t n, float first = 0.f) {
    const auto at = [first](std::size_t i) -> T {
        if constexpr (std::is_integral_v<T>) { // wraps modulo 2^N: a floating value outside T's range would convert with undefined behaviour
            return static_cast<T>(static_cast<std::int64_t>(first) + static_cast<std::int64_t>(i));
        } else {
            return static_cast<T>(static_cast<double>(first) + static_cast<double>(i));
        }
    };
    return std::views::iota(0UZ, n) | std::views::transform(at) | std::ranges::to<std::vector<T>>();
}

template<typename T = float>
[[nodiscard]] auto& sourceOf(auto& fixture, const std::vector<T>& values, float rate = kProbeRate) {
    return fixture.template emplace<ProbeSource<T>>({{"values", values}, {"n_samples_max", static_cast<gr::Size_t>(values.size())}, {"mark_tag", false}, {"sample_rate", rate}});
}

template<typename T>
[[nodiscard]] std::vector<T> samplesOf(const ProbeSink<T>& sink) {
    return {sink._samples.begin(), sink._samples.end()};
}

template<typename TBlock, typename T = float>
struct ProbedChain {
    gr::testing::GraphFixture<> fixture;
    ProbeSource<T>*             source = nullptr;
    ProbeSink<T>*               sink   = nullptr;

    ProbedChain(const std::vector<T>& values, gr::property_map blockSettings) {
        source      = &sourceOf(fixture, values);
        auto& block = fixture.template emplace<TBlock>(std::move(blockSettings));
        sink        = &fixture.template emplace<ProbeSink<T>>({{"log_samples", true}});
        boost::ut::expect(fixture.template connect<"out", "in">(*source, block).has_value());
        boost::ut::expect(fixture.template connect<"out", "in">(block, *sink).has_value());
    }

    [[nodiscard]] std::vector<T> run() {
        const auto ran = fixture.run();
        boost::ut::expect(ran.has_value()) << (ran.has_value() ? std::string{} : ran.error().message);
        return samplesOf(*sink);
    }
};

template<typename T>
[[nodiscard]] bool tagAt(const ProbeSink<T>& sink, std::size_t index, const std::string& key = kTriggerKey) {
    return std::ranges::any_of(sink._tags, [&](const auto& tag) { return tag.index == index && tag.map.contains(key); });
}

} // namespace gr::testing

#endif // GNURADIO_TESTING_STREAM_PROBES_HPP
