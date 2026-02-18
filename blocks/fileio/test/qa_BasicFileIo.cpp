#include <boost/ut.hpp>

#include <gnuradio-4.0/fileio/BasicFileIo.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <format>

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

template<typename DataType>
void runTest(const gr::blocks::fileio::Mode mode) {
    using namespace boost::ut;
    using namespace gr::blocks::fileio;
    using namespace gr::testing;
    using scheduler = gr::scheduler::Simple<>;

    constexpr gr::Size_t nSamples    = 1024U;
    const gr::Size_t     maxFileSize = mode == gr::blocks::fileio::Mode::multi ? 256U : 0U;
    std::string          modeName{magic_enum::enum_name(mode)};
    std::string          fileName = std::format("/tmp/gr4_file_sink_test/TestFileName_{}.bin", modeName);
    gr::blocks::fileio::detail::deleteFilesContaining(fileName);

    "BasicFileSink"_test = [&] { // NOSONAR capture all
        std::string testCaseName = std::format("BasicFileSink: failed for type '{}' and '{}", gr::meta::type_name<DataType>(), modeName);
        gr::Graph   flow;

        auto& source   = flow.emplaceBlock<ConstantSource<DataType>>({{"n_samples_max", nSamples}});
        auto& fileSink = flow.emplaceBlock<BasicFileSink<DataType>>({{"file_name", fileName}, {"mode", modeName}, {"max_bytes_per_file", maxFileSize}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, fileSink, fileSink.in)));

        scheduler sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value()) << testCaseName;

        if (watchdogThread.joinable()) {
            watchdogThread.join();
        }
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed)) << testCaseName;
        expect(eq(source.count, nSamples)) << testCaseName;
        expect(eq(fileSink._totalBytesWritten / sizeof(DataType), nSamples)) << testCaseName;

        std::vector<std::filesystem::path> files = gr::blocks::fileio::detail::getSortedFilesContaining(fileName);
        if (mode == gr::blocks::fileio::Mode::multi) {
            // greater-equal 'ge' because files can be legitimally zero-sized
            expect(ge(files.size(), (nSamples * sizeof(DataType)) / maxFileSize)) << testCaseName;
        } else {
            expect(eq(files.size(), 1U)) << testCaseName;
        }
        for (const auto& file : files) {
            auto fileSize = gr::blocks::fileio::detail::getFileSize(file);
            if (mode == gr::blocks::fileio::Mode::multi) {
                // less-equal 'le' because files can be legitimally zero-sized
                expect(le(fileSize, maxFileSize)) << testCaseName;
            } else {
                expect(eq(fileSize, nSamples * sizeof(DataType))) << testCaseName;
            }
        }
    };

    // N.B. test directory contains the output files from the previous sink test
    "BasicFileSource"_test = [&] { // NOSONAR capture all
        std::string testCaseName = std::format("BasicFileSource: failed for type '{}' and '{}", gr::meta::type_name<DataType>(), modeName);
        gr::Graph   flow;
        auto&       fileSource = flow.emplaceBlock<BasicFileSource<DataType>>({{"file_name", fileName}, {"mode", modeName}});
        auto&       sink       = flow.emplaceBlock<CountingSink<DataType>>();

        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(fileSource, fileSource.out, sink, sink.in)));

        scheduler schedRead;
        if (auto ret = schedRead.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto [watchdogThreadRead, externalInterventionNeededRead] = createWatchdog(schedRead, 2s);
        expect(schedRead.runAndWait().has_value()) << testCaseName;

        if (watchdogThreadRead.joinable()) {
            watchdogThreadRead.join();
        }
        expect(!externalInterventionNeededRead->load(std::memory_order_relaxed)) << testCaseName;
        expect(eq(sink.count, nSamples)) << testCaseName;
        expect(eq(fileSource._totalBytesRead, nSamples * sizeof(DataType))) << testCaseName;
    };

    // Test for `offset` and `length` parameters
    "BasicFileSource with offset and length"_test = [&] { // NOSONAR capture all
        constexpr gr::Size_t offsetSamples = 8U;
        constexpr gr::Size_t lengthSamples = 8U;
        std::string          testCaseName  = std::format("BasicFileSource with offset and length: failed for type '{}' and '{}", gr::meta::type_name<DataType>(), modeName);
        gr::Graph            flow;
        auto&                fileSource = flow.emplaceBlock<BasicFileSource<DataType>>({{"file_name", fileName}, {"mode", modeName}, {"offset", offsetSamples}, {"length", lengthSamples}});
        auto&                sink       = flow.emplaceBlock<CountingSink<DataType>>();

        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(fileSource, fileSource.out, sink, sink.in)));

        scheduler schedRead;
        if (auto ret = schedRead.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto [watchdogThreadRead, externalInterventionNeededRead] = createWatchdog(schedRead, 2s);
        expect(schedRead.runAndWait().has_value()) << testCaseName;

        if (watchdogThreadRead.joinable()) {
            watchdogThreadRead.join();
        }
        expect(!externalInterventionNeededRead->load(std::memory_order_relaxed)) << testCaseName;

        auto nonEmptyFileCount = static_cast<gr::Size_t>(std::ranges::count_if(gr::blocks::fileio::detail::getSortedFilesContaining(fileName), [](const auto& file) { return std::filesystem::file_size(file) > 0; }));
        expect(eq(sink.count, nonEmptyFileCount * lengthSamples)) << testCaseName;
        expect(eq(fileSource._totalBytesRead, nonEmptyFileCount * lengthSamples * sizeof(DataType))) << testCaseName;
    };

    expect(!gr::blocks::fileio::detail::deleteFilesContaining(fileName).empty());
}

} // anonymous namespace

const boost::ut::suite<"basic file IO tests"> basicFileIOTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;

    constexpr auto kArithmeticTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>>();

    using enum gr::blocks::fileio::Mode;
    "overwrite mode"_test = []<typename T>(const T&) { runTest<T>(overwrite); } | kArithmeticTypes;

    "append mode"_test = []<typename T>(const T&) { runTest<T>(append); } | kArithmeticTypes;

    "create new mode"_test = []<typename T>(const T&) { runTest<T>(multi); } | kArithmeticTypes;
};

int main() { /* not needed for UT */ }
