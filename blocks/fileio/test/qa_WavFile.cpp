#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/fileio/WavBlocks.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <httplib.h>

using namespace boost::ut;
using namespace std::chrono_literals;

void appendLe16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void appendLe32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void appendText(std::vector<std::uint8_t>& bytes, std::string_view text) { bytes.insert(bytes.end(), text.begin(), text.end()); }

void appendChunk(std::vector<std::uint8_t>& bytes, std::string_view id, std::span<const std::uint8_t> chunkBytes) {
    appendText(bytes, id);
    appendLe32(bytes, static_cast<std::uint32_t>(chunkBytes.size()));
    bytes.insert(bytes.end(), chunkBytes.begin(), chunkBytes.end());
    if ((chunkBytes.size() & 1U) != 0U) {
        bytes.push_back(0U);
    }
}

void patchLe32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    bytes[offset + 0U] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    bytes[offset + 2U] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    bytes[offset + 3U] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

std::vector<std::uint8_t> encodePcm8(const std::vector<std::uint8_t>& samples) { return samples; }

std::vector<std::uint8_t> encodePcm16(const std::vector<std::int16_t>& samples) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(samples.size() * sizeof(std::int16_t));
    for (const auto sample : samples) {
        appendLe16(bytes, static_cast<std::uint16_t>(sample));
    }
    return bytes;
}

std::vector<std::uint8_t> encodePcm24(const std::vector<std::int32_t>& samples) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(samples.size() * 3U);
    for (const auto sample : samples) {
        const auto value = static_cast<std::uint32_t>(sample);
        bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
        bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
        bytes.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    }
    return bytes;
}

std::vector<std::uint8_t> encodePcm32(const std::vector<std::int32_t>& samples) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(samples.size() * sizeof(std::int32_t));
    for (const auto sample : samples) {
        appendLe32(bytes, static_cast<std::uint32_t>(sample));
    }
    return bytes;
}

std::vector<std::uint8_t> encodeFloat32(const std::vector<float>& samples) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(samples.size() * sizeof(float));
    for (const auto sample : samples) {
        appendLe32(bytes, std::bit_cast<std::uint32_t>(sample));
    }
    return bytes;
}

std::vector<std::uint8_t> makeWav(std::uint16_t formatTag, std::uint16_t channels, std::uint16_t bitsPerSample, std::uint32_t sampleRate, const std::vector<std::uint8_t>& dataBytes, bool addJunkChunk = false) {
    std::vector<std::uint8_t> bytes;
    appendText(bytes, "RIFF");
    appendLe32(bytes, 0U);
    appendText(bytes, "WAVE");

    std::vector<std::uint8_t> fmt;
    const std::uint32_t       byteRate   = sampleRate * channels * (bitsPerSample / 8U);
    const std::uint16_t       blockAlign = static_cast<std::uint16_t>(channels * (bitsPerSample / 8U));
    appendLe16(fmt, formatTag);
    appendLe16(fmt, channels);
    appendLe32(fmt, sampleRate);
    appendLe32(fmt, byteRate);
    appendLe16(fmt, blockAlign);
    appendLe16(fmt, bitsPerSample);
    appendChunk(bytes, "fmt ", fmt);

    if (addJunkChunk) {
        static constexpr std::array<std::uint8_t, 5U> junk{{'h', 'e', 'l', 'l', 'o'}};
        appendChunk(bytes, "JUNK", junk);
    }

    appendChunk(bytes, "data", dataBytes);
    patchLe32(bytes, 4U, static_cast<std::uint32_t>(bytes.size() - 8U));
    return bytes;
}

struct TempFile {
    std::filesystem::path path;
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

std::string writeTempAudioFile(std::span<const std::uint8_t> bytes) {
    const auto    path = std::filesystem::temp_directory_path() / std::format("gr4-wav-{}.wav", std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    return path.string();
}

template<typename T>
struct WavSourceTestCase {
    std::string_view          name;
    std::vector<std::uint8_t> wavBytes;
    std::vector<T>            expectedSamples;
    float                     sampleRate;
    gr::Size_t                numChannels;
};

void expectSingleFormatTag(const std::vector<gr::Tag>& tags, float sampleRate, gr::Size_t numChannels, std::string_view caseName) {
    expect(ge(tags.size(), 1U)) << caseName;
    if (tags.empty()) {
        return;
    }
    expect(eq(gr::test::get_value_or_fail<float>(tags[0].map.at(gr::tag::SAMPLE_RATE.shortKey())), sampleRate)) << caseName;
    expect(eq(gr::test::get_value_or_fail<gr::Size_t>(tags[0].map.at(gr::tag::NUM_CHANNELS.shortKey())), numChannels)) << caseName;
}

template<typename TSource, typename T, typename TSampleCheck>
void runLocalSourceCases(const std::vector<WavSourceTestCase<T>>& cases, TSampleCheck&& sampleCheck) {
    for (const auto& testCase : cases) {
        const auto caseName = std::format("{} / {}", gr::meta::type_name<TSource>(), testCase.name);
        TempFile   file{writeTempAudioFile(testCase.wavBytes)};

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<TSource>({{"uri", file.path.string()}});
        auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        sampleCheck(std::vector<T>(sink._samples.begin(), sink._samples.end()), testCase.expectedSamples, caseName);
        expectSingleFormatTag(sink._tags, testCase.sampleRate, testCase.numChannels, caseName);
    }
}

std::expected<void, gr::Error> runSchedulerFor(gr::scheduler::Simple<>& sched, std::chrono::milliseconds duration) {
    std::optional<std::expected<void, gr::Error>> result;
    auto                                          schedThread = std::thread([&sched, &result] { result = sched.runAndWait(); });
    std::this_thread::sleep_for(duration);
    sched.requestStop();
    schedThread.join();
    return std::move(*result);
}

const boost::ut::suite<"WAV file blocks"> _wavFileTests = [] {
    using namespace boost::ut;

    "Local WAV sources"_test = [] {
        constexpr std::string_view caseName = "gr::blocks::fileio::WavSource<std::int16_t>";

        const std::vector<std::int16_t> reference{0, 16384, -16384, 32767};
        const std::uint32_t             sampleRate = 8000U;

        const auto wavBytes = makeWav(1U, 1U, 16U, sampleRate, encodePcm16(reference));
        TempFile   file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", file.path.string()}});
        auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<std::int16_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        expect(eq(std::vector<std::int16_t>(sink._samples.begin(), sink._samples.end()), reference)) << caseName;
        expectSingleFormatTag(sink._tags, static_cast<float>(sampleRate), 1U, caseName);
    };

    "PCM16 sources handle channels and pre-data chunks"_test = [] {
        const std::vector<WavSourceTestCase<std::int16_t>> cases{
            {"stereo pcm16", makeWav(1U, 2U, 16U, 22050U, encodePcm16(std::vector<std::int16_t>{0, 1000, -1000, 2000, -2000, 3000})), {0, 1000, -1000, 2000, -2000, 3000}, 22050.f, 2U},
            {"three channel pcm16", makeWav(1U, 3U, 16U, 32000U, encodePcm16(std::vector<std::int16_t>{100, 200, 300, -100, -200, -300})), {100, 200, 300, -100, -200, -300}, 32000.f, 3U},
            {"junk chunk before data", makeWav(1U, 1U, 16U, 44100U, encodePcm16(std::vector<std::int16_t>{0, 2000, -2000, 4000}), true), {0, 2000, -2000, 4000}, 44100.f, 1U},
        };

        runLocalSourceCases<gr::blocks::fileio::WavSource<std::int16_t>>(cases, [](const auto& actual, const auto& expected, std::string_view name) { expect(eq(actual, expected)) << name; });
    };

    "WavSource normalizes non-16-bit WAV variants to float"_test = [] {
        const std::vector<WavSourceTestCase<float>> cases{
            {"pcm8 mono with data padding", makeWav(1U, 1U, 8U, 8000U, encodePcm8(std::vector<std::uint8_t>{0U, 128U, 255U})), {-1.0f, 0.0f, 127.0f / 128.0f}, 8000.f, 1U},
            {"pcm24 mono", makeWav(1U, 1U, 24U, 16000U, encodePcm24(std::vector<std::int32_t>{0, 4'194'304, -4'194'304})), {0.0f, 0.5f, -0.5f}, 16000.f, 1U},
            {"pcm32 mono", makeWav(1U, 1U, 32U, 48000U, encodePcm32(std::vector<std::int32_t>{0, 1'073'741'824, -1'073'741'824})), {0.0f, 0.5f, -0.5f}, 48000.f, 1U},
            {"float32 mono", makeWav(3U, 1U, 32U, 9600U, encodeFloat32(std::vector<float>{-1.0f, 0.0f, 0.5f, 1.0f})), {-1.0f, 0.0f, 0.5f, 1.0f}, 9600.f, 1U},
        };

        runLocalSourceCases<gr::blocks::fileio::WavSource<float>>(cases, [](const auto& actual, const auto& expected, std::string_view name) {
            expect(eq(actual.size(), expected.size())) << name;
            for (std::size_t i = 0U; i < expected.size(); ++i) {
                expect(approx(actual[i], expected[i], 1e-6f)) << name;
            }
        });
    };

    "WavSink writes int16 WAV file and WavSource reads it back"_test = [] {
        constexpr std::string_view caseName = "WavSink int16 roundtrip";

        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000};
        const auto                      wavBytes = makeWav(1U, 2U, 16U, 22050U, encodePcm16(reference));
        TempFile                        inputFile{writeTempAudioFile(wavBytes)};

        const auto outputPath = std::filesystem::temp_directory_path() / std::format("gr4-wavsink-{}.wav", std::chrono::steady_clock::now().time_since_epoch().count());
        TempFile   outputFile{outputPath};

        // write: WavSource -> WavSink
        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", inputFile.path.string()}});
            auto&     sink   = graph.emplaceBlock<gr::blocks::fileio::WavSink<std::int16_t>>({{"uri", outputPath.string()}, {"sample_rate", 22050.f}, {"num_channels", gr::Size_t(2)}});
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;
            expect(eq(sink.total_samples_written.value, static_cast<gr::Size_t>(reference.size()))) << caseName;
        }

        // read back and verify
        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", outputPath.string()}});
            auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<std::int16_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;

            expect(eq(std::vector<std::int16_t>(sink._samples.begin(), sink._samples.end()), reference)) << caseName;
            expectSingleFormatTag(sink._tags, 22050.f, 2U, caseName);
        }
    };

    "WavSink writes float32 WAV file"_test = [] {
        constexpr std::string_view caseName = "WavSink float32";

        const auto outputPath = std::filesystem::temp_directory_path() / std::format("gr4-wavsink-f32-{}.wav", std::chrono::steady_clock::now().time_since_epoch().count());
        TempFile   outputFile{outputPath};

        const std::vector<float> reference{-1.0f, 0.0f, 0.5f, 1.0f};
        const auto               wavBytes = makeWav(3U, 1U, 32U, 9600U, encodeFloat32(reference));
        TempFile                 inputFile{writeTempAudioFile(wavBytes)};

        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", inputFile.path.string()}});
            auto&     sink   = graph.emplaceBlock<gr::blocks::fileio::WavSink<float>>({{"uri", outputPath.string()}, {"sample_rate", 9600.f}, {"num_channels", gr::Size_t(1)}});
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;
        }

        // read back
        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", outputPath.string()}});
            auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;

            const auto samples = std::vector<float>(sink._samples.begin(), sink._samples.end());
            expect(eq(samples.size(), reference.size())) << caseName;
            for (std::size_t i = 0U; i < reference.size(); ++i) {
                expect(approx(samples[i], reference[i], 1e-6f)) << caseName;
            }
        }
    };

    "WavSink writes multi-channel (4ch) WAV file"_test = [] {
        constexpr std::string_view caseName = "WavSink 4-channel";

        const auto outputPath = std::filesystem::temp_directory_path() / std::format("gr4-wavsink-4ch-{}.wav", std::chrono::steady_clock::now().time_since_epoch().count());
        TempFile   outputFile{outputPath};

        const std::vector<std::int16_t> reference{100, 200, 300, 400, -100, -200, -300, -400};
        const auto                      wavBytes = makeWav(1U, 4U, 16U, 44100U, encodePcm16(reference));
        TempFile                        inputFile{writeTempAudioFile(wavBytes)};

        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", inputFile.path.string()}});
            auto&     sink   = graph.emplaceBlock<gr::blocks::fileio::WavSink<std::int16_t>>({{"uri", outputPath.string()}, {"sample_rate", 44100.f}, {"num_channels", gr::Size_t(4)}});
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;
        }

        {
            gr::Graph graph;
            auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", outputPath.string()}});
            auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<std::int16_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;

            expect(eq(std::vector<std::int16_t>(sink._samples.begin(), sink._samples.end()), reference)) << caseName;
            expectSingleFormatTag(sink._tags, 44100.f, 4U, caseName);
        }
    };

    "WavSource loop restarts from beginning"_test = [] {
        constexpr std::string_view caseName = "WavSource loop";

        const std::vector<std::int16_t> reference{1000, 2000, 3000};
        const auto                      wavBytes = makeWav(1U, 1U, 16U, 8000U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<std::int16_t>>({{"uri", file.path.string()}, {"repeat", true}});
        auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<std::int16_t, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 200ms).has_value()) << caseName;

        // should have produced more samples than one pass through the file
        expect(gt(sink._nSamplesProduced, reference.size())) << caseName;
        // first three samples should match the reference
        const auto samples = std::vector<std::int16_t>(sink._samples.begin(), sink._samples.begin() + static_cast<std::ptrdiff_t>(reference.size()));
        expect(eq(samples, reference)) << caseName;
    };

#ifndef __EMSCRIPTEN__
    "HTTP WAV sources"_test = [] {
        constexpr std::string_view      caseName = "gr::blocks::fileio::WavSource<float>";
        const std::vector<std::int16_t> reference{0, 8192, -8192, 16384};
        const auto                      wavBytes = makeWav(1U, 1U, 16U, 11025U, encodePcm16(reference));
        const std::string               body(reinterpret_cast<const char*>(wavBytes.data()), wavBytes.size());

        httplib::Server server;
        server.Get("/tone.wav", [&body](const httplib::Request&, httplib::Response& res) { res.set_content(body, "audio/wav"); });
        const int port = server.bind_to_any_port("localhost");
        expect(port > 0) << caseName;

        auto thread = std::thread{[&server] { server.listen_after_bind(); }};
        server.wait_until_ready();

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", std::format("http://localhost:{}/tone.wav", port)}});
        auto&     sink   = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        server.stop();
        thread.join();

        const auto samples = std::vector<float>(sink._samples.begin(), sink._samples.end());
        expect(eq(samples.size(), reference.size())) << caseName;
        expect(approx(samples[1], 0.25f, 1e-3f)) << caseName;
        expect(approx(samples[2], -0.25f, 1e-3f)) << caseName;
        expectSingleFormatTag(sink._tags, 11025.f, 1U, caseName);
    };

#endif
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
