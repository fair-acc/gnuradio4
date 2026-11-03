#include <benchmark.hpp>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <format>
#include <memory_resource>
#include <string>
#include <string_view>
#include <thread>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/PerformanceMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace std::chrono_literals;

constexpr gr::Size_t                kBenchmarkSamples = 1'000'000U;
constexpr std::uint64_t             kTriggerTimeNs    = 1'000'000'000U;
[[nodiscard]] constexpr std::size_t samplesPerTag(int useCase) noexcept { return useCase == 2 ? 10'000UZ : useCase == 3 ? 1UZ : 0UZ; }
/** `nSamples == 0` runs until `runTime` elapses */
void runGraph(gr::Size_t nSamples, int useCase, std::chrono::seconds runTime, const std::string& csvPath) {
    using namespace gr::testing;
    const gr::property_map kNoLogging{{"log_samples", false}, {"log_tags", false}};
    /** static: singletons first allocated during a run keep their memory in these pools until the process ends */
    static std::pmr::unsynchronized_pool_resource poolData;
    static std::pmr::unsynchronized_pool_resource poolTag;
    static std::pmr::synchronized_pool_resource   poolMechanics;
    std::pmr::memory_resource* const              previousDefault = std::pmr::set_default_resource(&poolMechanics);
    {
        gr::Graph              graph(gr::ResourceProfile{.data = &poolData, .tag = &poolTag, .mechanics = &poolMechanics});
        const gr::property_map sourceSettings{{"n_samples_max", nSamples}, {"repeat_tags", true}, {"verbose_console", false}};
        auto&                  source = graph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>(sourceSettings);
        if (const std::size_t every = samplesPerTag(useCase); every > 0UZ) {
            using namespace gr::tag;
            const auto             key = [](const auto& tag) { return std::string(tag.key()); };
            const gr::property_map metaInfo{{key(kLocalTime), kTriggerTimeNs}};
            const gr::property_map trigger{{key(kTriggerName), "tick"}, {key(kTriggerTime), kTriggerTimeNs}, {key(kTriggerOffset), 0.f}, {key(kTriggerMetaInfo), metaInfo}};
            source._tags = {{every - 1UZ, trigger}};
        }
        auto& monitorBulk = graph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>(kNoLogging);
        auto& monitorOne  = graph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>(kNoLogging);
        auto& monitor     = graph.emplaceBlock<PerformanceMonitor<float>>({{"evaluate_perf_rate", gr::Size_t(100'000)}, {"output_csv_file_path", csvPath}});
        auto& sinkRes     = graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>(kNoLogging);
        auto& sinkRate    = graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>(kNoLogging);
        std::ignore       = graph.connect<"out", "in">(source, monitorBulk);
        std::ignore       = graph.connect<"out", "in">(monitorBulk, monitorOne);
        std::ignore       = graph.connect<"out", "in">(monitorOne, monitor);
        std::ignore       = graph.connect<"outRes", "in">(monitor, sinkRes);
        std::ignore       = graph.connect<"outRate", "in">(monitor, sinkRate);

        gr::scheduler::Simple<> scheduler;
        std::ignore = scheduler.exchange(std::move(graph));
        std::jthread stopAfterRunTime;
        if (nSamples == 0U) {
            stopAfterRunTime = std::jthread([&scheduler, runTime](std::stop_token stop) {
                const auto deadline = std::chrono::steady_clock::now() + runTime;
                while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline && scheduler.state() != gr::lifecycle::State::STOPPED) {
                    std::this_thread::sleep_for(40ms);
                }
                scheduler.requestStop();
            });
        }
        std::ignore = scheduler.runAndWait();
    }
    std::pmr::set_default_resource(previousDefault);
}
} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        const auto useCase3 = [] { runGraph(kBenchmarkSamples, 3, 0s, ""); };

        ::benchmark::benchmark<3>("use case 3: trigger name, time, offset and meta info on every sample", kBenchmarkSamples) = useCase3;
        return 0;
    }
    const auto argument = [&](int index, int fallback) {
        int value = fallback;
        if (index < argc) {
            const std::string_view text(argv[index]);
            std::ignore = std::from_chars(text.data(), text.data() + text.size(), value);
        }
        return value;
    };
    const int         runTime = argument(1, 5);
    const int         useCase = argument(2, 3);
    const std::string csvPath = argc >= 4 ? std::string(argv[3]) : std::string{};
    std::println("bm_PerformanceMonitor <run_time>[s] <use_case>[1: no tags, 2: a tag every 10000 samples, 3: a tag on every sample] <output_csv_file_path>");
    std::println("run_time: {} s, use_case: {}, output_csv_file_path: '{}'", runTime, useCase, csvPath);
    runGraph(0U, useCase, std::chrono::seconds(runTime), csvPath);
    return 0;
}
