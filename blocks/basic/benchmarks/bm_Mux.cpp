#include <benchmark.hpp>

#include <format>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/Mux.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
constexpr gr::Size_t  kSamples   = 1U << 20U;
constexpr std::size_t kTagPeriod = 64UZ;

[[nodiscard]] std::string portName(std::string_view name, std::size_t nPorts, std::size_t port) { return nPorts == 1UZ ? std::string(name) : std::format("{}#{}", name, port); }

template<typename TBlock>
void runGraph(gr::property_map settings, std::size_t nIn, std::size_t nOut, bool tagged = false) {
    gr::Graph  graph;
    auto&      block = graph.emplaceBlock<TBlock>(std::move(settings));
    const auto perIn = static_cast<gr::Size_t>(kSamples / nIn);
    for (std::size_t port = 0UZ; port < nIn; ++port) {
        if (tagged) {
            auto& source = graph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", perIn}, {"mark_tag", false}, {"repeat_tags", true}});
            source._tags = {{kTagPeriod - 1UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "tick"}}}};
            std::ignore  = graph.connect(source, "out", block, portName("in", nIn, port));
        } else {
            auto& source = graph.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", perIn}});
            std::ignore  = graph.connect(source, "out", block, portName("in", nIn, port));
        }
    }
    for (std::size_t port = 0UZ; port < nOut; ++port) {
        auto& sink  = graph.emplaceBlock<gr::testing::CountingSink<float>>();
        std::ignore = graph.connect(block, portName("out", nOut, port), sink, "in");
    }
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}

[[nodiscard]] gr::property_map runs(gr::Size_t first, gr::Size_t second) { return {{"lengths", std::vector<gr::Size_t>{first, second}}}; }
} // namespace

inline const boost::ut::suite<"Mux and Demux"> _mux = [] {
    using namespace benchmark;
    "control: Copy<float>, one source and sink"_benchmark.repeat<5>(kSamples)   = [] { runGraph<gr::testing::Copy<float>>({}, 1UZ, 1UZ); };
    "Mux<float, 2>, lengths 1, 1"_benchmark.repeat<5>(kSamples)                 = [] { runGraph<gr::basic::Mux<float, 2UZ>>(runs(1U, 1U), 2UZ, 1UZ); };
    "Mux<float, 2>, lengths 64, 64"_benchmark.repeat<5>(kSamples)               = [] { runGraph<gr::basic::Mux<float, 2UZ>>(runs(64U, 64U), 2UZ, 1UZ); };
    "Mux<float>, lengths 64, 64"_benchmark.repeat<5>(kSamples)                  = [] { runGraph<gr::basic::Mux<float>>(runs(64U, 64U), 2UZ, 1UZ); };
    "tagged control: Copy<float>, a tag every 64"_benchmark.repeat<5>(kSamples) = [] { runGraph<gr::testing::Copy<float>>({}, 1UZ, 1UZ, true); };
    "Mux<float, 2>, lengths 1, 1, a tag every 64"_benchmark.repeat<5>(kSamples) = [] { runGraph<gr::basic::Mux<float, 2UZ>>(runs(1U, 1U), 2UZ, 1UZ, true); };
    "Demux<float, 2>, lengths 1, 1"_benchmark.repeat<5>(kSamples)               = [] { runGraph<gr::basic::Demux<float, 2UZ>>(runs(1U, 1U), 1UZ, 2UZ); };
    "Demux<float, 2>, lengths 64, 64"_benchmark.repeat<5>(kSamples)             = [] { runGraph<gr::basic::Demux<float, 2UZ>>(runs(64U, 64U), 1UZ, 2UZ); };
};

int main() {}
