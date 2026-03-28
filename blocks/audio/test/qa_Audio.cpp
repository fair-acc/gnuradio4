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
#include <gnuradio-4.0/audio/AudioBlocks.hpp>
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
    const auto    path = std::filesystem::temp_directory_path() / std::format("gr4-audio-{}.wav", std::chrono::steady_clock::now().time_since_epoch().count());
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
    gr::Size_t                channels;
};

void expectSingleFormatTag(const std::vector<gr::Tag>& tags, float sampleRate, gr::Size_t channels, std::string_view caseName) {
    expect(eq(tags.size(), 1U)) << caseName;
    expect(eq(gr::test::get_value_or_fail<float>(tags[0].map.at(gr::tag::SAMPLE_RATE.shortKey())), sampleRate)) << caseName;
    expect(eq(gr::test::get_value_or_fail<gr::Size_t>(tags[0].map.at(gr::tag::NUM_CHANNELS.shortKey())), channels)) << caseName;
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
        expectSingleFormatTag(sink._tags, testCase.sampleRate, testCase.channels, caseName);
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

const boost::ut::suite audioTests = [] {
    using namespace boost::ut;

    "Local WAV sources"_test = [] {
        constexpr std::string_view caseName = "gr::audio::WavSource<std::int16_t>";

        const std::vector<std::int16_t> reference{0, 16384, -16384, 32767};
        const std::uint32_t             sampleRate = 8000U;

        const auto wavBytes = makeWav(1U, 1U, 16U, sampleRate, encodePcm16(reference));
        TempFile   file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::audio::WavSource<std::int16_t>>({{"uri", file.path.string()}});
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

        runLocalSourceCases<gr::audio::WavSource<std::int16_t>>(cases, [](const auto& actual, const auto& expected, std::string_view name) { expect(eq(actual, expected)) << name; });
    };

    "WavSource normalizes non-16-bit WAV variants to float"_test = [] {
        const std::vector<WavSourceTestCase<float>> cases{
            {"pcm8 mono with data padding", makeWav(1U, 1U, 8U, 8000U, encodePcm8(std::vector<std::uint8_t>{0U, 128U, 255U})), {-1.0f, 0.0f, 127.0f / 128.0f}, 8000.f, 1U},
            {"pcm24 mono", makeWav(1U, 1U, 24U, 16000U, encodePcm24(std::vector<std::int32_t>{0, 4'194'304, -4'194'304})), {0.0f, 0.5f, -0.5f}, 16000.f, 1U},
            {"pcm32 mono", makeWav(1U, 1U, 32U, 48000U, encodePcm32(std::vector<std::int32_t>{0, 1'073'741'824, -1'073'741'824})), {0.0f, 0.5f, -0.5f}, 48000.f, 1U},
            {"float32 mono", makeWav(3U, 1U, 32U, 9600U, encodeFloat32(std::vector<float>{-1.0f, 0.0f, 0.5f, 1.0f})), {-1.0f, 0.0f, 0.5f, 1.0f}, 9600.f, 1U},
        };

        runLocalSourceCases<gr::audio::WavSource<float>>(cases, [](const auto& actual, const auto& expected, std::string_view name) {
            expect(eq(actual.size(), expected.size())) << name;
            for (std::size_t i = 0U; i < expected.size(); ++i) {
                expect(approx(actual[i], expected[i], 1e-6f)) << name;
            }
        });
    };

#ifndef __EMSCRIPTEN__
    "HTTP WAV sources"_test = [] {
        constexpr std::string_view      caseName = "gr::audio::WavSource<float>";
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
        auto&     source = graph.emplaceBlock<gr::audio::WavSource<float>>({{"uri", std::format("http://localhost:{}/tone.wav", port)}});
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

#ifndef __EMSCRIPTEN__
    "AudioSink plays PCM with soundio dummy backend"_test = [] {
        constexpr std::string_view      caseName = "AudioSink soundio dummy backend";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000};
        const auto                      wavBytes = makeWav(1U, 2U, 16U, 22050U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::audio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"buffer_frames", gr::Size_t(256)}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;
        expect(sched.state() != gr::lifecycle::State::ERROR) << caseName;

        expect(sink._useDummyBackendForTests) << caseName;
        expect(eq(sink.num_channels.value, 2U)) << caseName;
        expect(eq(sink.sample_rate.value, 22050.f)) << caseName;
    };

    "AudioSource captures PCM with soundio dummy backend"_test = [] {
        constexpr std::string_view caseName = "AudioSource soundio dummy backend";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(2)}, {"buffer_frames", gr::Size_t(256)}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 200ms).has_value()) << caseName;
        expect(sched.state() != gr::lifecycle::State::ERROR) << caseName;

        expect(source._useDummyBackendForTests) << caseName;
        expect(gt(source.sample_rate.value, 0.0f)) << caseName;
        expect(gt(source.num_channels.value, 0U)) << caseName;
        expect(gt(sink._nSamplesProduced, 0UZ)) << caseName;
        expectSingleFormatTag(sink._tags, source.sample_rate.value, source.num_channels.value, caseName);
    };

    "AudioSource loops back into AudioSink with soundio dummy backend"_test = [] {
        constexpr std::string_view caseName = "AudioSource to AudioSink soundio dummy backend";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(2)}, {"buffer_frames", gr::Size_t(256)}});
        auto&     sink                  = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"buffer_frames", gr::Size_t(256)}});
        source._useDummyBackendForTests = true;
        sink._useDummyBackendForTests   = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 200ms).has_value()) << caseName;
        expect(sched.state() != gr::lifecycle::State::ERROR) << caseName;

        expect(source._useDummyBackendForTests) << caseName;
        expect(sink._useDummyBackendForTests) << caseName;
        expect(gt(source.sample_rate.value, 0.0f)) << caseName;
        expect(gt(source.num_channels.value, 0U)) << caseName;
        expect(eq(sink.sample_rate.value, source.sample_rate.value)) << caseName;
        expect(eq(sink.num_channels.value, source.num_channels.value)) << caseName;
    };

    "available_devices is populated after start"_test = [] {
        constexpr std::string_view caseName = "available_devices populated";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"buffer_frames", gr::Size_t(256)}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 200ms).has_value()) << caseName;

        expect(!source.available_devices.value.empty()) << caseName;
        for (const auto& entry : source.available_devices.value) {
            expect(entry.find('[') != std::string::npos) << "device entry should contain '[': " << entry;
            expect(entry.find(']') != std::string::npos) << "device entry should contain ']': " << entry;
        }
    };

    "AudioSink available_devices is populated after start"_test = [] {
        constexpr std::string_view      caseName = "AudioSink available_devices populated";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000};
        const auto                      wavBytes = makeWav(1U, 1U, 16U, 22050U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::audio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"buffer_frames", gr::Size_t(256)}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        expect(!sink.available_devices.value.empty()) << caseName;
    };

#endif
};

const boost::ut::suite deviceResolutionTests = [] {
    using namespace boost::ut;
    using gr::audio::detail::AudioDeviceInfo;
    using gr::audio::detail::resolveDeviceIndex;

    const std::vector<AudioDeviceInfo> devices{
        {.name = "Built-in Audio Output", .id = "hw:0,0"},
        {.name = "USB Headset", .id = "hw:1,0"},
        {.name = "HDMI Output", .id = "hw:2,0"},
    };

    "empty spec returns nullopt (system default)"_test = [&] { expect(!resolveDeviceIndex("", devices).has_value()); };

    "'default' returns nullopt"_test = [&] { expect(!resolveDeviceIndex("default", devices).has_value()); };

    "'Default' returns nullopt (case-insensitive)"_test = [&] { expect(!resolveDeviceIndex("Default", devices).has_value()); };

    "substring match on name"_test = [&] {
        auto result = resolveDeviceIndex("usb", devices);
        expect(result.has_value()) << "should match 'USB Headset'";
        expect(eq(*result, 1UZ));
    };

    "substring match is case-insensitive"_test = [&] {
        auto result = resolveDeviceIndex("hdmi", devices);
        expect(result.has_value()) << "should match 'HDMI Output'";
        expect(eq(*result, 2UZ));
    };

    "exact ID match with @id: prefix"_test = [&] {
        auto result = resolveDeviceIndex("@id:hw:1,0", devices);
        expect(result.has_value()) << "should match hw:1,0";
        expect(eq(*result, 1UZ));
    };

    "unmatched name returns nullopt"_test = [&] { expect(!resolveDeviceIndex("NonExistent", devices).has_value()); };

    "unmatched @id: returns nullopt"_test = [&] { expect(!resolveDeviceIndex("@id:hw:99,0", devices).has_value()); };

    "first match wins for ambiguous substring"_test = [&] {
        auto result = resolveDeviceIndex("output", devices);
        expect(result.has_value());
        expect(eq(*result, 0UZ)) << "should match 'Built-in Audio Output' first";
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
