#include <boost/ut.hpp>

#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <tuple>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/fileio/BasicFileIo.hpp>
#include <gnuradio-4.0/soapy/Soapy.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

gr::Graph createGraph(std::string fileName1, std::string fileName2, gr::Size_t maxFileSize, float sampleRate, double rxCenterFrequency, double bandwidth, double rxGains) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::soapy;
    using namespace gr::blocks::fileio;

    Graph flow;
    using TDataType = std::complex<float>;

    auto& source = flow.emplaceBlock<SoapyBlock<TDataType, 2UZ>>({
        {"device", "lime"},                                                                 //
        {"sample_rate", sampleRate},                                                        //
        {"rx_channels", std::vector<gr::Size_t>{0U, 1U}},                                   //
        {"rx_antennae", std::vector<std::string>{"LNAW", "LNAW"}},                          //
        {"rx_center_frequency", std::vector<double>{rxCenterFrequency, rxCenterFrequency}}, //
        {"rx_bandwdith", std::vector<double>{bandwidth, bandwidth}},                        //
        {"rx_gains", std::vector<double>{rxGains, rxGains}},
    });
    fmt::println("set parameter:\n   sample_rate: {} SP/s\n   rx_center_frequency: {} Hz\n   rx_bandwdith: {} Hz\n   rx_gains: {} [dB]", //
        sampleRate, rxCenterFrequency, bandwidth, rxGains);

    if (fileName1.contains("null")) {
        fmt::println("write channel0 to NullSink");
        auto& fileSink1 = flow.emplaceBlock<testing::NullSink<TDataType>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out0">(source).to<"in">(fileSink1))) << "error connecting NullSink1";
    } else {
        fmt::println("write to fileName1: {}", fileName1);
        auto& fileSink1 = flow.emplaceBlock<BasicFileSink<TDataType>>({{"file_name", fileName1}, {"mode", "multi"}, {"max_bytes_per_file", maxFileSize}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out0">(source).to<"in">(fileSink1))) << "error connecting BasicFileSink1";
    }

    if (fileName2.contains("null")) {
        fmt::println("write channel1 to NullSink");
        auto& fileSink2 = flow.emplaceBlock<testing::NullSink<TDataType>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out1">(source).to<"in">(fileSink2))) << "error connecting NullSink2";
    } else {
        fmt::println("write to fileName2: {}", fileName2);
        auto& fileSink2 = flow.emplaceBlock<BasicFileSink<TDataType>>({{"file_name", fileName2}, {"mode", "multi"}, {"max_bytes_per_file", maxFileSize}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out1">(source).to<"in">(fileSink2))) << "error connecting BasicFileSink2";
    }

    return flow;
}

std::optional<std::tuple<std::string, std::string, gr::Size_t, float, double, double, double>> parseArguments(int argc, char* argv[], const std::string& defaultFileName1, const std::string& defaultFileName2, gr::Size_t defaultMaxFileSize, float defaultSampleRate, double defaultRxCenterFrequency, double defaultBandwidth, double defaultRxGains);

int main(int argc, char* argv[]) {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks::soapy;
    using namespace gr::blocks::fileio;

    constexpr gr::Size_t defaultMaxFileSize       = 100 * 1UZ << 20; // 100 MB
    constexpr float      defaultSampleRate        = 2'000'000.f;     // 2 MHz
    constexpr double     defaultRxCenterFrequency = 107e6;           // 107 MHz
    constexpr double     defaultBandwidth         = 5e6;             // 5 MHz
    constexpr double     defaultRxGains           = 10.0;            // 10 dB

    std::filesystem::path exePath    = std::filesystem::current_path();
    auto                  parsedArgs = parseArguments(argc, argv, exePath / "test_ch0.bin", exePath / "test_ch1.bin", defaultMaxFileSize, defaultSampleRate, defaultRxCenterFrequency, defaultBandwidth, defaultRxGains);
    if (!parsedArgs) {
        fmt::println(R"(
Usage:
  {0} <baseName>
    Sets fileName1 to <baseName>_ch0.bin and fileName2 to <baseName>_ch1.bin

  {0} <fileName1> <fileName2>
  {0} <fileName1> <fileName2> <maxFileSize [bytes]>
  {0} <fileName1> <fileName2> <maxFileSize [bytes]> <sample_rate [SP/s]> <rx_center_frequency [Hz]> <BW [Hz]> <rx_gains [dB]>

Default:
  If no arguments are provided, fileName1 defaults to test_ch0.bin and fileName2 defaults to test_ch1.bin
  If the file name contains 'null' then the data is written to the NullSource (for testing/rate checks etc.)
)",
            argv[0]);
        return 1;
    }

    auto [fileName1, fileName2, maxFileSize, sampleRate, rxCenterFrequency, bandwidth, rxGains] = *parsedArgs;

    Graph flow = createGraph(fileName1, fileName2, maxFileSize, sampleRate, rxCenterFrequency, bandwidth, rxGains);

    auto threadPool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 10UZ);
    auto sched      = gr::scheduler::Simple<>{std::move(flow), threadPool};
    auto retVal     = sched.runAndWait();
    expect(retVal.has_value()) << fmt::format("scheduler execution error: {}", retVal.error());

    return 0;
}

bool isNumber(const std::string& str) { return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit); }

std::optional<std::tuple<std::string, std::string, gr::Size_t, float, double, double, double>> parseArguments(int argc, char* argv[], const std::string& defaultFileName1, const std::string& defaultFileName2, gr::Size_t defaultMaxFileSize, float defaultSampleRate, double defaultRxCenterFrequency, double defaultBandwidth, double defaultRxGains) {
    std::string fileName1         = defaultFileName1;
    std::string fileName2         = defaultFileName2;
    gr::Size_t  maxFileSize       = defaultMaxFileSize;
    float       sampleRate        = defaultSampleRate;
    double      rxCenterFrequency = defaultRxCenterFrequency;
    double      bandwidth         = defaultBandwidth;
    double      rxGains           = defaultRxGains;

    switch (argc) {
    case 2:
        if (std::string(argv[1]).starts_with("-")) {
            return std::nullopt;
        }
        fileName1 = std::string(argv[1]) + "_ch0.bin";
        fileName2 = std::string(argv[1]) + "_ch1.bin";
        break;
    case 3:
        fileName1 = argv[1];
        fileName2 = argv[2];
        break;
    case 4:
        fileName1 = argv[1];
        fileName2 = argv[2];
        if (!isNumber(argv[3])) {
            std::cerr << "Error: maxFileSize must be a number." << std::endl;
            return std::nullopt;
        }
        maxFileSize = static_cast<gr::Size_t>(std::stoull(argv[3]));
        break;
    case 8:
        fileName1 = argv[1];
        fileName2 = argv[2];
        if (!isNumber(argv[3])) {
            std::cerr << "Error: maxFileSize must be a number." << std::endl;
            return std::nullopt;
        }
        maxFileSize       = static_cast<gr::Size_t>(std::stoull(argv[3]));
        sampleRate        = std::stof(argv[4]);
        rxCenterFrequency = std::stod(argv[5]);
        bandwidth         = std::stod(argv[6]);
        rxGains           = std::stod(argv[7]);
        break;
    case 1: break; // Use defaults
    default: std::cerr << "Usage: " << argv[0] << " <baseName> | <fileName1> <fileName2> | <fileName1> <fileName2> <maxFileSize> | <fileName1> <fileName2> <maxFileSize> <sample_rate> <rx_center_frequency> <BW> <rx_gains>" << std::endl; return std::nullopt;
    }

    if (fileName1 == fileName2) {
        std::cerr << "Error: fileName1 and fileName2 must be different." << std::endl;
        return std::nullopt;
    }

    auto makeAbsoluteIfNeeded = [&](std::string fileName) {
        std::filesystem::path exePath = std::filesystem::current_path();
        std::filesystem::path filePath(fileName);
        if (!filePath.is_absolute() && filePath.string().find("./") != 0 && filePath.string().find("../") != 0) {
            fileName = (exePath / filePath).string();
        }
        return fileName;
    };

    return std::make_tuple(makeAbsoluteIfNeeded(fileName1), makeAbsoluteIfNeeded(fileName2), maxFileSize, sampleRate, rxCenterFrequency, bandwidth, rxGains);
}
