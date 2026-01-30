#include <boost/ut.hpp>
#include <format>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/PerformanceMonitor.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <string>

namespace {
using namespace std::chrono_literals;
template<typename Scheduler>
auto createWatchdog(Scheduler& sched, std::chrono::seconds timeOut = 2s, std::chrono::milliseconds pollingPeriod = 40ms) {
    auto externalInterventionNeeded = std::make_shared<std::atomic_bool>(false);

    std::thread watchdogThread([&sched, externalInterventionNeeded, timeOut, pollingPeriod]() {
        auto timeout = std::chrono::steady_clock::now() + timeOut;
        while (std::chrono::steady_clock::now() < timeout) {
            if (sched.state() == gr::lifecycle::State::STOPPED) {
                return;
            }
            std::this_thread::sleep_for(pollingPeriod);
        }
        std::println("watchdog kicked in");
        externalInterventionNeeded->store(true, std::memory_order_relaxed);
        sched.requestStop();
        std::println("requested scheduler to stop");
    });

    return std::make_pair(std::move(watchdogThread), externalInterventionNeeded);
}
} // namespace

int main(int argc, char* argv[]) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;

    int         runTime     = 5; // in seconds
    int         testCaseId  = 1;
    std::string outFilePath = "";

    if (argc >= 2) {
        runTime = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        testCaseId = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        outFilePath = std::string(argv[3]);
    }
    std::println("3 optional settings are available: qa_PerformanceMonitor <run_time>[in sec] <test_case_id>[1:no tags,2:moderate,3:1-to-1] <output_file_path>");
    std::println("<run_time>:{} s, <test_case_id>:{}, <output_file_path>:{}", runTime, testCaseId, outFilePath);

    gr::Size_t         nSamples         = 0U;
    gr::Size_t         evaluatePerfRate = 100'000;
    Graph              testGraph;
    const property_map srcParameter = {{"n_samples_max", nSamples}, {"name", "TagSource"}, {"verbose_console", false}, {"repeat_tags", true}};
    auto&              src          = testGraph.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>(srcParameter);

    // parameters of generated Tags
    std::size_t nSamplesPerTag    = 10000UZ;
    bool        tagWithAutoUpdate = false;
    std::string tagName           = tagWithAutoUpdate ? gr::tag::SAMPLE_RATE.shortKey() : "some_random_name_1234";
    std::string outputCsvFilePath = "";

    if (outFilePath != "") {
        outputCsvFilePath = outFilePath;
    }
    if (testCaseId == 1) {
        nSamplesPerTag = 0;
    } else if (testCaseId == 2) {
        nSamplesPerTag = 10000;
    } else if (testCaseId == 3) {
        nSamplesPerTag = 1;
    }

    if (nSamplesPerTag > 0) {
        src._tags = {gr::Tag(nSamplesPerTag - 1, gr::property_map{{convert_string_domain(tagName), 2000.f}})};
    };

    auto& monitorBulk        = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagMonitorBulk"}, {"log_samples", false}, {"log_tags", false}});
    auto& monitorOne         = testGraph.emplaceBlock<TagMonitor<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TagMonitorOne"}, {"log_samples", false}, {"log_tags", false}});
    auto& monitorPerformance = testGraph.emplaceBlock<PerformanceMonitor<float>>( //
        {{"name", "PerformanceMonitor"}, {"evaluate_perf_rate", evaluatePerfRate}, {"output_csv_file_path", outputCsvFilePath}});

    // performance statistics outputs
    auto& sinkRes  = testGraph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkRes"}, {"log_samples", false}, {"log_tags", false}});
    auto& sinkRate = testGraph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>({{"name", "TagSinkRate"}, {"log_samples", false}, {"log_tags", false}});

    // src -> monitorBulk -> monitorOne -> monitorPerformance
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(monitorBulk)));
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorBulk).to<"in">(monitorOne)));
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(monitorOne).to<"in">(monitorPerformance)));
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"outRes">(monitorPerformance).to<"in">(sinkRes)));
    expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"outRate">(monitorPerformance).to<"in">(sinkRate)));

    gr::scheduler::Simple<> sched;
    if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, runTime > 0 ? std::chrono::seconds(runTime) : 2s);
    expect(sched.runAndWait().has_value());

    if (watchdogThread.joinable()) {
        watchdogThread.join();
    }

    expect(approx(monitorPerformance.n_updates_res, sinkRes._nSamplesProduced, 2U));
    expect(approx(monitorPerformance.n_updates_rate, sinkRate._nSamplesProduced, 2U));
    //   expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
}
