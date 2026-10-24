#include <benchmark.hpp>

#include <string>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
constexpr gr::Size_t  kSamples = 1U << 20U;
constexpr std::size_t kCopies  = 4UZ;

void runGraph(std::size_t tagEvery) {
    gr::Graph graph;
    auto&     source = graph.emplaceBlock<gr::testing::TagSource<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", kSamples}, {"mark_tag", false}, {"repeat_tags", true}});
    source._tags     = {{tagEvery - 1UZ, {{std::string(gr::tag::TRIGGER_NAME.key()), "tick"}, {"local_note", 1.f}}}};
    auto& first      = graph.emplaceBlock<gr::testing::Copy<float>>();
    std::ignore      = graph.connect<"out", "in">(source, first);
    auto* last       = &first;
    for (std::size_t i = 1UZ; i < kCopies; ++i) {
        auto& next  = graph.emplaceBlock<gr::testing::Copy<float>>();
        std::ignore = graph.connect<"out", "in">(*last, next);
        last        = &next;
    }
    auto& sink  = graph.emplaceBlock<gr::testing::CountingSink<float>>();
    std::ignore = graph.connect<"out", "in">(*last, sink);
    gr::scheduler::Simple<> scheduler;
    std::ignore = scheduler.exchange(std::move(graph));
    std::ignore = scheduler.runAndWait();
}
} // namespace

inline const boost::ut::suite<"TagForwarding"> _tagForwarding = [] {
    using namespace benchmark;
    "no tags, 4 copies"_benchmark.repeat<5>(kSamples)                = [] { runGraph(kSamples); };
    "a tag every 16 samples, 4 copies"_benchmark.repeat<5>(kSamples) = [] { runGraph(16UZ); };
};

int main() {}
