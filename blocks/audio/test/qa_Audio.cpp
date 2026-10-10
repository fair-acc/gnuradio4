#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/algorithm/ImGraph.hpp>
#include <gnuradio-4.0/audio/AudioBlocks.hpp>
#include <gnuradio-4.0/fileio/WavBlocks.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <httplib.h>

using namespace boost::ut;
using namespace std::chrono_literals;
using gr::audio::AudioPortMode;
using gr::audio::ChannelPadding;

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
    gr::Size_t                numChannels;
};

void expectSingleFormatTag(const std::vector<gr::testing::OwningTag>& tags, float sampleRate, gr::Size_t numChannels, std::string_view caseName) {
    expect(ge(tags.size(), 1U)) << caseName;
    if (tags.empty()) {
        return;
    }
    expect(eq(gr::test::get_value_or_fail<float>(tags[0].map.find_value(gr::tag::SAMPLE_RATE).value()), sampleRate)) << caseName;
    expect(eq(gr::test::get_value_or_fail<gr::Size_t>(tags[0].map.find_value(gr::tag::NUM_CHANNELS).value()), numChannels)) << caseName;
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

struct RateTaker : gr::Block<RateTaker> {
    gr::PortIn<float> in;

    gr::Annotated<float, "sample_rate", gr::Unit<"Hz">> sample_rate = 0.f;

    GR_MAKE_REFLECTABLE(RateTaker, in, sample_rate);

    constexpr void processOne(float) const noexcept {}
};

const boost::ut::suite<"audio device tests"> _audioTests = [] {
    using namespace boost::ut;

#ifndef __EMSCRIPTEN__
    "AudioSink plays PCM with soundio dummy backend"_test = [] {
        constexpr std::string_view      caseName = "AudioSink soundio dummy backend";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000};
        const auto                      wavBytes = makeWav(1U, 2U, 16U, 22050U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
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
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(2)}, {"io_buffer_size", 0.1f}});
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

    "AudioSource's format tag updates a downstream sample_rate"_test = [] {
        constexpr std::string_view caseName = "AudioSource format tag downstream";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<RateTaker>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 200ms).has_value()) << caseName;
        expect(sched.state() != gr::lifecycle::State::ERROR) << caseName;

        expect(gt(source.sample_rate.value, 0.0f)) << caseName;
        expect(eq(sink.sample_rate.value, source.sample_rate.value)) << caseName;
    };

    "AudioSource loops back into AudioSink with soundio dummy backend"_test = [] {
        constexpr std::string_view caseName = "AudioSource to AudioSink soundio dummy backend";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(2)}, {"io_buffer_size", 0.1f}});
        auto&     sink                  = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
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
        expect(gt(sink.sample_rate.value, 0.0f)) << caseName;
        expect(gt(sink.num_channels.value, 0U)) << caseName;
    };

    "available_devices is populated after start"_test = [] {
        constexpr std::string_view caseName = "available_devices populated";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}});
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
        auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        expect(!sink.available_devices.value.empty()) << caseName;
    };

    "AudioSink publishes its active format and permission to the settings"_test = [] {
        constexpr std::string_view      caseName = "AudioSink active settings";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000};
        TempFile                        file{writeTempAudioFile(makeWav(1U, 2U, 16U, 22050U, encodePcm16(reference)))};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        const auto active = sink.settings().get();
        expect(eq(active.value_or<float>("sample_rate", 0.f), 22050.f)) << caseName;
        expect(eq(active.value_or<gr::Size_t>("num_channels", 0U), 2U)) << caseName;
        expect(active.value_or<bool>("permission", false)) << caseName;
    };

    "a sink rebuilt from its saved settings still adopts the first tagged format"_test = [] {
        constexpr std::string_view      caseName = "AudioSink saved settings";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000};
        TempFile                        file{writeTempAudioFile(makeWav(1U, 2U, 16U, 22050U, encodePcm16(reference)))};
        const auto                      playWav = [&](gr::property_map sinkSettings) {
            gr::Graph graph;
            auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
            auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>(std::move(sinkSettings));
            sink._useDummyBackendForTests = true;
            expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;
            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(graph)).has_value()) << caseName;
            expect(sched.runAndWait().has_value()) << caseName;
            expect(sched.state() != gr::lifecycle::State::ERROR) << caseName;
            expect(eq(sink.sample_rate.value, 22050.f)) << caseName;
            expect(eq(sink.num_channels.value, 2U)) << caseName;
            expect(eq(sink.req_sample_rate.value, 48000.f)) << caseName;
            expect(eq(sink.n_inputs.value, 1U)) << caseName;
            gr::property_map saved;
            for (const std::string_view key : {"req_sample_rate", "sample_rate", "n_inputs", "num_channels", "io_buffer_size"}) {
                expect(sink.settings().get().contains(key)) << caseName << key;
                saved.insert_or_assign(key, gr::Value(*sink.settings().get().find_value(key)));
            }
            return saved;
        };
        std::ignore = playWav(playWav({{"io_buffer_size", 0.1f}}));
    };

#endif
};

const boost::ut::suite<"audio device resolution"> _deviceResolutionTests = [] {
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

    "a fragment of 'default' selects a matching device rather than the default"_test = [&] { expect(eq(resolveDeviceIndex("au", devices).value_or(99UZ), 0UZ)); };

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

#ifndef __EMSCRIPTEN__
const boost::ut::suite<"audio timing drift"> _timingAndDriftTests = [] {
    using namespace boost::ut;

    "AudioSource emits timing tags with dummy backend"_test = [] {
        constexpr std::string_view caseName = "AudioSource timing tags";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"tag_interval", 0.0f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 500ms).has_value()) << caseName;

        expect(gt(sink._nSamplesProduced, 0UZ)) << caseName;
        expect(gt(sink._tags.size(), 1UZ)) << caseName;

        bool foundTimingTag = false;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME)) {
                foundTimingTag = true;
                expect(sinkTag.map.contains(gr::tag::TRIGGER_NAME)) << caseName;
                expect(sinkTag.map.contains(gr::tag::TRIGGER_OFFSET)) << caseName;

                expect(sinkTag.map.contains(gr::tag::TRIGGER_META_INFO)) << caseName;
                break;
            }
        }
        expect(foundTimingTag) << "should have at least one timing tag";
    };

    "DriftCompensator inserts sample when source is fast"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 10>                  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        std::size_t nProduced = 4U;
        for (int i = 0; i < 300; ++i) {
            nProduced = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.0001, 48000.0, 1U);
        }

        expect(ge(nProduced, 4UZ)) << "should insert or maintain sample count";
    };

    "DriftCompensator drops sample when source is slow"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 10>                  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        std::size_t nProduced = 4U;
        for (int i = 0; i < 300; ++i) {
            nProduced = comp.compensateSource(std::span(buf), 4U, 48000.0 * 0.9999, 48000.0, 1U);
        }

        expect(le(nProduced, 4UZ)) << "should drop or maintain sample count";
    };

    "DriftCompensator interpolation produces smooth values"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 10>                  buf{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        comp.fractionalAccumulator = 0.99;
        buf[0]                     = 1.0f;
        buf[1]                     = 3.0f;

        auto n = comp.compensateSource(std::span(buf), 2U, 48000.0 * 1.01, 48000.0, 1U);
        if (n == 3U) {
            expect(approx(buf[2], 2.0f, 0.5f)) << "interpolated sample should be between neighbours";
        }
    };

    "DriftCompensator stereo insert preserves channel interleaving"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 20>                  buf{};
        buf[0] = 1.f;
        buf[1] = 2.f;
        buf[2] = 3.f;
        buf[3] = 4.f;

        comp.fractionalAccumulator = 0.99;
        auto n                     = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.01, 48000.0, 2U);
        if (n == 6U) {
            expect(approx(buf[4], 2.0f, 0.5f)) << "inserted L should interpolate between 1 and 3";
            expect(approx(buf[5], 3.0f, 0.5f)) << "inserted R should interpolate between 2 and 4";
        }
    };

    "DriftCompensator stereo drop preserves frame alignment"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 10>                  buf{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 0.f, 0.f, 0.f, 0.f};

        comp.fractionalAccumulator = -0.99;
        auto n                     = comp.compensateSource(std::span(buf), 6U, 48000.0 * 0.99, 48000.0, 2U);
        if (n == 4U) {
            expect(eq(n % 2UZ, 0UZ)) << "dropped result should be frame-aligned";
        }
    };

    "emit_timing_tags=false suppresses timing tags"_test = [] {
        constexpr std::string_view caseName = "timing tags disabled";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", false}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 300ms).has_value()) << caseName;

        expect(gt(sink._nSamplesProduced, 0UZ)) << caseName;
        bool foundTimingTag = false;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME)) {
                foundTimingTag = true;
            }
        }
        expect(!foundTimingTag) << "no timing tags should be emitted when disabled";
    };

    "emit_meta_info=false omits TRIGGER_META_INFO"_test = [] {
        constexpr std::string_view caseName = "meta info disabled";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"emit_meta_info", false}, {"tag_interval", 0.0f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 300ms).has_value()) << caseName;

        bool foundTimingTag = false;
        bool foundMetaInfo  = false;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME)) {
                foundTimingTag = true;
                if (sinkTag.map.contains(gr::tag::TRIGGER_META_INFO)) {
                    foundMetaInfo = true;
                }
            }
        }
        expect(foundTimingTag) << "should have timing tags";
        expect(!foundMetaInfo) << "TRIGGER_META_INFO should be absent when emit_meta_info=false";
    };

    "tag_interval throttles timing tag emission"_test = [] {
        constexpr std::string_view caseName = "tag interval throttling";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"tag_interval", 10.0f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 300ms).has_value()) << caseName;

        std::size_t timingTagCount = 0U;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME)) {
                ++timingTagCount;
            }
        }
        expect(le(timingTagCount, 1UZ)) << "tag_interval=10s should heavily throttle emission";
    };

    "AudioSource rate estimator converges"_test = [] {
        constexpr std::string_view caseName = "rate estimator convergence";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"ppm_estimator_cutoff", 0.5f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 500ms).has_value()) << caseName;

        const double estimated = source._rateEstimator.estimatedRate();
        expect(gt(estimated, 0.0)) << "rate estimator should have a positive estimate";
        expect(gt(static_cast<float>(estimated), 1000.f)) << "estimated rate should be above 1 kHz";
        expect(lt(static_cast<float>(estimated), 200000.f)) << "estimated rate should be below 200 kHz";
    };

    "an upstream est_sample_rate tag cannot overwrite the sink's own measurement"_test = [] {
        gr::Graph graph;
        auto&     sink = graph.emplaceBlock<gr::audio::AudioSink<float>>();
        sink._measuredSampleRate.store(47999.f);
        sink.est_sample_rate = 12345.f;
        sink.settingsChanged({}, {{"est_sample_rate", 12345.f}});
        expect(eq(sink.est_sample_rate.value, 47999.f));
    };

    "clk_in forwards clock offset and trigger name"_test = [] {
        constexpr std::string_view caseName = "clk_in forwarding";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"tag_interval", 0.0f}});
        source._useDummyBackendForTests = true;

        auto& clkSource = graph.emplaceBlock<gr::testing::TagSource<std::uint8_t, gr::testing::ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_max", gr::Size_t(0)}, {"mark_tag", false}});

        constexpr std::uint64_t kFakeUtcNs = 1700000000'000000000ULL; // a fixed UTC timestamp
        gr::property_map        clkTagMap;
        gr::tag::put(clkTagMap, gr::tag::TRIGGER_TIME, kFakeUtcNs);
        gr::tag::put(clkTagMap, gr::tag::TRIGGER_NAME, std::string("GPS:TEST"));
        clkSource._tags = {{0U, std::move(clkTagMap)}};

        auto& sink = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "clk_in">(clkSource, source).has_value()) << caseName;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 500ms).has_value()) << caseName;

        expect(gt(sink._nSamplesProduced, 0UZ)) << caseName;

        expect(source._clockOffsetValid) << "clock offset should be valid after receiving clk_in tag";

        bool foundGpsTrigger = false;
        for (const auto& sinkTag : sink._tags) {
            if (auto it = sinkTag.map.find(gr::tag::TRIGGER_NAME); it != sinkTag.map.end()) {
                const gr::Value nameEntry = (*it).second;
                if (auto name = nameEntry.get_if<std::string_view>()) {
                    if (*name == "GPS:TEST") {
                        foundGpsTrigger = true;
                        break;
                    }
                }
            }
        }
        expect(foundGpsTrigger) << "timing tags should forward the clock trigger name from clk_in";
    };

    "AudioSource permission setting is readable"_test = [] {
        constexpr std::string_view caseName = "AudioSource permission";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 300ms).has_value()) << caseName;

        expect(static_cast<bool>(source.permission.value)) << "source permission should be true after start";

        const auto activeParams = source.settings().getStored().value_or(gr::property_map{});
        expect(activeParams.contains("permission")) << "permission should be in active parameters";
    };

    "AudioSink permission setting is readable"_test = [] {
        constexpr std::string_view      caseName = "AudioSink permission";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000, -3000, 4000};
        const auto                      wavBytes = makeWav(1U, 1U, 16U, 22050U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        expect(static_cast<bool>(sink.permission.value)) << "sink permission should be true after start";

        const auto activeParams = sink.settings().getStored().value_or(gr::property_map{});
        expect(activeParams.contains("permission")) << "permission should be in active parameters";
    };

    "DriftCompensator sink insert and drop"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;

        std::array<float, 10> input{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        std::array<float, 12> adjusted{};

        comp.fractionalAccumulator = 0.99;
        auto n                     = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 0.99, 48000.0, 1U);
        expect(ge(n, 4UZ)) << "sink compensator should insert when sink is faster";

        comp.fractionalAccumulator = -0.99;
        n                          = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 1.01, 48000.0, 1U);
        expect(le(n, 4UZ)) << "sink compensator should drop when sink is slower";
    };
};
#endif

template<typename T>
struct AudioInputFixture : gr::Block<AudioInputFixture<T>> {
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(AudioInputFixture, out);
    using gr::Block<AudioInputFixture<T>>::Block;
    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto&) { return gr::work::Status::INSUFFICIENT_INPUT_ITEMS; }
};

template<typename T>
void queueAudio(AudioInputFixture<T>& producer, std::span<const T> samples, float rate = 48000.f) {
    const gr::property_map tags{gr::tag::SAMPLE_RATE(rate), gr::tag::NUM_CHANNELS(gr::Size_t{1U})};
    producer.out.publishTag(tags, 0UZ);
    auto span = producer.out.streamWriter().reserve(samples.size());
    std::ranges::copy(samples, span.begin());
    span.publish(samples.size());
}

template<typename TSource>
auto& prepareSourceWithoutDevice(TSource& source) {
    source.applyNegotiatedFormat({.sampleRate = 48000U, .numChannels = 1U});
    source._captureSamples.resize(32UZ);
    source._publishSpans.reserve(gr::audio::detail::kMaxAudioChannels);
    auto& state = source._backendImpl.state();
    state.recreateBuffer(32UZ);
    return state;
}

template<typename TSink>
void prepareSinkWithoutDevice(TSink& sink, std::size_t deviceChannels) {
    sink._activeConfig = {.sampleRate = 48000U, .numChannels = static_cast<std::uint32_t>(deviceChannels)};
    sink.sample_rate   = 48000.f;
    sink._staging.recreateBuffer(32UZ * deviceChannels);
    sink._inputRates.assign(sink._logicalChannels, 0U);
    if constexpr (requires { sink.in.size(); }) {
        sink._connectedInputs.clear();
        for (const auto& port : sink.in) {
            sink._connectedInputs.push_back(port.isConnected());
        }
    }
}

template<typename TSink>
void startSink(TSink& sink) {
#if defined(__EMSCRIPTEN__)
    prepareSinkWithoutDevice(sink, sink._logicalChannels);
#else
    sink._useDummyBackendForTests = true;
    sink.start();
    expect(!sink._failed.load()) << fatal;
#endif
}

template<typename TSink>
void stopSink(TSink& sink) {
#if !defined(__EMSCRIPTEN__)
    sink.stop();
#else
    std::ignore = sink;
#endif
}

template<typename T, AudioPortMode mode = AudioPortMode::interleaved>
struct SinkTestGraph {
    gr::Graph                          graph;
    std::vector<AudioInputFixture<T>*> producers;
    gr::audio::AudioSink<T, mode>&     sink;
    bool                               started = false;

    SinkTestGraph(gr::property_map settings, std::initializer_list<std::size_t> connectedInputs = {0UZ}) : sink(graph.emplaceBlock<gr::audio::AudioSink<T, mode>>(std::move(settings))) {
        for (const std::size_t input : connectedInputs) {
            auto& producer = graph.emplaceBlock<AudioInputFixture<T>>();
            producers.push_back(&producer);
            expect(graph.connect(producer, "out", sink, mode == AudioPortMode::interleaved ? std::string("in") : std::format("in#{}", input)).has_value()) << fatal;
        }
        expect(graph.connectPendingEdges()) << fatal;
    }

    ~SinkTestGraph() {
        if (started) {
            stopSink(sink);
        }
    }

    void start() {
        startSink(sink);
        started = true;
    }

    AudioInputFixture<T>& producer(std::size_t index = 0UZ) { return *producers[index]; }
};

template<typename TSink>
auto sinkInputSpans(TSink& sink, std::size_t frames) {
    using Span = decltype(sink.in[0].template get<gr::SpanReleasePolicy::ProcessAll>(0UZ));
    std::vector<Span> spans;
    spans.reserve(sink.in.size());
    for (auto& port : sink.in) {
        spans.push_back(port.template get<gr::SpanReleasePolicy::ProcessAll>(std::min(frames, port.streamReader().available())));
    }
    return spans;
}

template<typename T>
void verifySparsePlayback(ChannelPadding padding) {
    SinkTestGraph<T, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{3U}}, {"req_sample_rate", 48000.f}, {"channel_padding", padding == ChannelPadding::cyclic ? "cyclic" : "zero"}}, {0UZ, 2UZ});
    auto&                                        sink  = fixture.sink;
    auto&                                        first = fixture.producer(0UZ);
    auto&                                        third = fixture.producer(1UZ);
    prepareSinkWithoutDevice(sink, 4UZ);
    const std::array<T, 2UZ> left{T{1}, T{2}};
    const std::array<T, 2UZ> right{T{3}, T{4}};
    queueAudio<T>(first, left);
    queueAudio<T>(third, right);
    {
        auto      spans = sinkInputSpans(sink, 2UZ);
        std::span inputs(spans);
        expect(sink.processBulk(inputs) == gr::work::Status::OK);
    }
    expect(eq(sink.in[0].streamReader().available(), 0UZ));
    expect(eq(sink.in[2].streamReader().available(), 0UZ));
    auto                     staged = sink._staging.reader.get(8UZ);
    const std::array<T, 8UZ> reference{T{1}, T{}, T{3}, padding == ChannelPadding::cyclic ? T{1} : T{}, T{2}, T{}, T{4}, padding == ChannelPadding::cyclic ? T{2} : T{}};
    expect(std::ranges::equal(staged, reference));
}

template<typename T, AudioPortMode mode>
void verifyCaptureFallback(ChannelPadding padding) {
    using Source   = gr::audio::AudioSource<T, mode>;
    using Consumer = gr::testing::TagSink<T, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
    gr::Graph              graph;
    auto&                  source = graph.emplaceBlock<Source>({{"req_sample_rate", 96000.f}, {"n_outputs", gr::Size_t{2U}}, {"emit_timing_tags", false}, {"drift_correction", "None"}});
    std::vector<Consumer*> consumers;
    const std::size_t      nPorts = mode == AudioPortMode::interleaved ? 1UZ : 2UZ;
    for (std::size_t channel = 0UZ; channel < nPorts; ++channel) {
        auto& consumer = graph.emplaceBlock<Consumer>();
        consumers.push_back(&consumer);
        expect(graph.connect(source, mode == AudioPortMode::interleaved ? "out" : std::format("out#{}", channel), consumer, "in").has_value()) << fatal;
    }
    expect(graph.connectPendingEdges()) << fatal;
    auto& state = prepareSourceWithoutDevice(source);
    state.channelPadding.store(padding);
    const std::array<float, 3UZ> input{0.25f, -0.5f, 0.75f};
    expect(eq(state.writePlanarFloat(input.data(), input.size(), 1UZ, 2UZ), 6UZ));
    std::array<T, 6UZ> reference{};
    if constexpr (std::same_as<T, float>) {
        reference = {0.25f, 0.f, -0.5f, 0.f, 0.75f, 0.f};
    } else {
        reference = {8192, 0, -16384, 0, 24575, 0};
    }
    if (padding == ChannelPadding::cyclic) {
        for (std::size_t frame = 0UZ; frame < input.size(); ++frame) {
            reference[frame * 2UZ + 1UZ] = reference[frame * 2UZ];
        }
    }
    source.publishSamples(6UZ, 2UZ);
    expect(eq(source.req_sample_rate.value, 96000.f));
    expect(eq(source.sample_rate.value, 48000.f));
    expect(eq(source.n_outputs.value, 2U));
    for (std::size_t channel = 0UZ; channel < nPorts; ++channel) {
        auto& consumer  = *consumers[channel];
        auto  inputSpan = consumer.in.template get<gr::SpanReleasePolicy::ProcessAll>(mode == AudioPortMode::interleaved ? 6UZ : 3UZ);
        expect(consumer.processBulk(inputSpan) == gr::work::Status::OK);
        expect(eq(consumer._samples.size(), mode == AudioPortMode::interleaved ? 6UZ : 3UZ));
        for (std::size_t index = 0UZ; index < consumer._samples.size(); ++index) {
            const std::size_t referenceIndex = mode == AudioPortMode::interleaved ? index : index * 2UZ + channel;
            expect(eq(consumer._samples[index], reference[referenceIndex]));
        }
        expect(eq(consumer._tags.size(), 1UZ)) << fatal;
        expect(eq(gr::test::get_value_or_fail<float>(consumer._tags[0].map.find_value(gr::tag::SAMPLE_RATE).value()), 48000.f));
        expect(eq(gr::test::get_value_or_fail<gr::Size_t>(consumer._tags[0].map.find_value(gr::tag::NUM_CHANNELS).value()), mode == AudioPortMode::interleaved ? 2U : 1U));
    }
}

const boost::ut::suite<"audio format adaptation"> _adaptationTests = [] {
    "requested 96 kHz capture tags delivered 48 kHz on every output without resampling"_test = [] {
        for (const auto padding : {ChannelPadding::zero, ChannelPadding::cyclic}) {
            verifyCaptureFallback<float, AudioPortMode::interleaved>(padding);
            verifyCaptureFallback<std::int16_t, AudioPortMode::interleaved>(padding);
            verifyCaptureFallback<float, AudioPortMode::multiplexed>(padding);
            verifyCaptureFallback<std::int16_t, AudioPortMode::multiplexed>(padding);
        }
    };

    "interleaved audio uses one edge, multiplexed audio one edge per channel"_test = [] {
        const auto drawnEdges = [](gr::Graph& graph) {
            const std::string drawing = gr::graph::draw(graph);
            std::println("{}", drawing);
            return std::ranges::count_if(drawing | std::views::split('\n'), [](auto line) { return std::string_view(line).contains("─▶"); });
        };
        gr::Graph interleaved;
        auto&     stereoSource = interleaved.emplaceBlock<gr::audio::AudioSource<float>>({{"name", "AudioSource"}, {"n_outputs", gr::Size_t{2U}}});
        auto&     stereoSink   = interleaved.emplaceBlock<gr::audio::AudioSink<float>>({{"name", "AudioSink"}, {"n_inputs", gr::Size_t{2U}}});
        expect(interleaved.connect<"out", "in">(stereoSource, stereoSink).has_value()) << fatal;
        expect(interleaved.connectPendingEdges()) << fatal;
        expect(eq(drawnEdges(interleaved), 1));

        gr::Graph multiplexed;
        auto&     monoSources = multiplexed.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"name", "AudioSourceMultiplexed"}, {"n_outputs", gr::Size_t{2U}}});
        auto&     monoSinks   = multiplexed.emplaceBlock<gr::audio::AudioSinkMultiplexed<float>>({{"name", "AudioSinkMultiplexed"}, {"n_inputs", gr::Size_t{2U}}});
        expect(multiplexed.connect(monoSources, "out#0", monoSinks, "in#0").has_value()) << fatal;
        expect(multiplexed.connect(monoSources, "out#1", monoSinks, "in#1").has_value()) << fatal;
        expect(multiplexed.connectPendingEdges()) << fatal;
        expect(eq(drawnEdges(multiplexed), 2));
    };

    "stereo maps cyclically to eight channels without per-frame channel remapping"_test = [] {
        const std::array<float, 4UZ> input{1.f, 2.f, 3.f, 4.f};
        std::array<float, 16UZ>      output{};
        gr::audio::detail::adaptChannels<float>(output, 2UZ, 2UZ, 8UZ, ChannelPadding::cyclic, [&](std::size_t frame, std::size_t channel) { return input[frame * 2UZ + channel]; });
        for (std::size_t frame = 0UZ; frame < 2UZ; ++frame) {
            for (std::size_t channel = 0UZ; channel < 8UZ; ++channel) {
                expect(eq(output[frame * 8UZ + channel], input[frame * 2UZ + channel % 2UZ]));
            }
        }
    };

    "timing tags carry the measured rate at the top level, not in the trigger meta information"_test = [] {
        using Consumer = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
        gr::Graph graph;
        auto&     source   = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"emit_timing_tags", true}, {"tag_interval", 0.f}, {"drift_correction", "None"}});
        auto&     consumer = graph.emplaceBlock<Consumer>();
        expect(graph.connect<"out", "in">(source, consumer).has_value()) << fatal;
        expect(graph.connectPendingEdges()) << fatal;
        auto&                        state = prepareSourceWithoutDevice(source);
        const std::array<float, 2UZ> input{0.25f, 0.5f};
        expect(eq(state.writePlanarFloat(input.data(), 2UZ, 1UZ, 1UZ), 2UZ));
        source.publishSamples(2UZ, 1UZ);
        auto inputSpan = consumer.in.template get<gr::SpanReleasePolicy::ProcessAll>(consumer.in.streamReader().available());
        expect(consumer.processBulk(inputSpan) == gr::work::Status::OK);
        const auto timingTag = std::ranges::find_if(consumer._tags, [](const auto& candidate) { return candidate.map.contains(gr::tag::TRIGGER_TIME); });
        expect(timingTag != consumer._tags.end()) << fatal;
        expect(gt(gr::test::get_value_or_fail<float>(timingTag->map.find_value(gr::tag::EST_SAMPLE_RATE).value()), 0.f));
        expect(!gr::test::get_value_or_fail<gr::property_map>(timingTag->map.find_value(gr::tag::TRIGGER_META_INFO).value()).contains("sample_rate"));
    };

    "96 kHz capture fallback updates downstream settings to 48 kHz on every mono port"_test = [] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"req_sample_rate", 96000.f}, {"n_outputs", gr::Size_t{2U}}, {"emit_timing_tags", false}, {"drift_correction", "None"}});
        auto&     first  = graph.emplaceBlock<RateTaker>();
        auto&     second = graph.emplaceBlock<RateTaker>();
        expect(graph.connect(source, "out#0", first, "in").has_value()) << fatal;
        expect(graph.connect(source, "out#1", second, "in").has_value()) << fatal;
        expect(graph.connectPendingEdges()) << fatal;
        auto&                        state = prepareSourceWithoutDevice(source);
        const std::array<float, 2UZ> samples{0.25f, 0.5f};
        expect(eq(state.writePlanarFloat(samples.data(), 2UZ, 1UZ, 2UZ), 4UZ));
        source.publishSamples(4UZ, 2UZ);
        expect(first.work(2UZ).status == gr::work::Status::OK);
        expect(second.work(2UZ).status == gr::work::Status::OK);
        expect(eq(first.sample_rate.value, 48000.f));
        expect(eq(second.sample_rate.value, 48000.f));
    };

    "signed PCM boundaries are clipped and normalised consistently"_test = [] {
        gr::audio::detail::AudioSourceState<std::int16_t> source;
        source.recreateBuffer(16UZ);
        const std::array<float, 5UZ> input{-2.f, -1.f, 0.f, 1.f, 2.f};
        expect(eq(source.writePlanarFloat(input.data(), input.size(), 1UZ, 1UZ), input.size()));
        std::array<std::int16_t, 5UZ> captured{};
        expect(eq(source.readToOutput(captured, 1UZ), captured.size()));
        expect(captured == std::array<std::int16_t, 5UZ>{-32768, -32768, 0, 32767, 32767});
        gr::audio::detail::AudioSinkState<std::int16_t> sink;
        sink.recreateBuffer(16UZ);
        const std::array<std::int16_t, 2UZ> endpoints{-32768, 32767};
        expect(eq(sink.writeFromInput(endpoints, 1UZ), 2UZ));
        std::array<float, 2UZ> played{};
        sink.readPlanarFloat(played.data(), 2UZ, 1UZ);
        expect(played == std::array{-1.f, 32767.f / 32768.f});
    };

    "zero available capture channels publish no fabricated samples"_test = [] {
        gr::audio::detail::AudioSourceState<float> state;
        state.recreateBuffer(16UZ);
        const std::array<float, 1UZ> input{1.f};
        expect(eq(state.writePlanarFloat(input.data(), 1UZ, 0UZ, 2UZ), 0UZ));
        expect(eq(state.reader.available(), 0UZ));
    };

    "runtime padding changes apply to the next capture batch"_test = [] {
        gr::audio::detail::AudioSourceState<float> state;
        state.recreateBuffer(16UZ);
        const std::array<float, 1UZ> input{0.5f};
        expect(eq(state.writePlanarFloat(input.data(), 1UZ, 1UZ, 2UZ), 2UZ));
        state.channelPadding.store(ChannelPadding::cyclic);
        expect(eq(state.writePlanarFloat(input.data(), 1UZ, 1UZ, 2UZ), 2UZ));
        std::array<float, 4UZ> output{};
        expect(eq(state.readToOutput(output, 2UZ), 4UZ));
        expect(output == std::array{0.5f, 0.f, 0.5f, 0.5f});
    };

    "partial capture batches remain frame aligned and count overflow"_test = [] {
        gr::audio::detail::AudioSourceState<float> state;
        state.recreateBuffer(4UZ);
        if (state.writer.available() > 4UZ) {
            auto fill = state.writer.reserve(state.writer.available() - 4UZ);
            std::ranges::fill(fill, 0.f);
            fill.publish(fill.size());
        }
        const std::array<float, 3UZ> input{1.f, 2.f, 3.f};
        expect(eq(state.writePlanarFloat(input.data(), 3UZ, 1UZ, 2UZ), 4UZ));
        expect(eq(state.overflowCount.load(), 1UZ));
    };

    "playback callback bounds output width and discards non-playable channels"_test = [] {
        gr::audio::detail::AudioSinkState<float> state;
        state.recreateBuffer(16UZ);
        const std::array<float, 8UZ> input{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        expect(eq(state.writeFromInput(input, 8UZ), 8UZ));
        std::array<float, 4UZ> output{-1.f, 0.f, 0.f, -1.f};
        state.readPlanarFloat(output.data() + 1, 1UZ, 8UZ, 2UZ);
        expect(output == std::array{-1.f, 1.f, 2.f, -1.f});
    };

    "intentional disconnected silence does not count as an underrun"_test = [] {
        gr::audio::detail::AudioSinkState<float> state;
        state.intentionalSilence.store(true);
        std::array<float, 2UZ> output{1.f, 1.f};
        state.readPlanarFloat(output.data(), 1UZ, 2UZ);
        expect(output == std::array{0.f, 0.f});
        expect(eq(state.underrunCount.load(), 0UZ));
    };

    "a worklet ring hands captured blocks to the source in order and counts the ones the worklet dropped"_test = [] {
        using gr::audio::detail::WorkletBlockRing;
        gr::audio::detail::AudioSourceState<float> state;
        state.recreateBuffer(4096UZ);
        state.numChannels = 2UZ;
        state.channelPadding.store(ChannelPadding::cyclic);
        WorkletBlockRing ring(4UZ, 2UZ);
        for (std::uint32_t block = 0U; block < 2U; ++block) { // what the JS capture processor does with a mono input
            std::fill_n(ring.slotSamples(block), WorkletBlockRing::kBlockFrames, static_cast<float>(block + 1U));
            ring.blockFrames[block]   = WorkletBlockRing::kBlockFrames;
            ring.blockChannels[block] = 1U;
            ring.head.store(block + 1U);
        }
        ring.missedBlocks.store(3U);

        state.drainWorklet(ring);

        expect(eq(ring.filledBlocks(), 0UZ));
        expect(eq(state.overflowCount.load(), 3UZ)) << "blocks the worklet dropped on a full ring are overflows";
        expect(eq(state.reader.available(), 2UZ * WorkletBlockRing::kBlockFrames * 2UZ));
        const auto samples = state.reader.get(state.reader.available());
        expect(eq(samples[0], 1.f) && eq(samples[1], 1.f)) << "mono is padded cyclically to stereo";
        expect(eq(samples[2UZ * WorkletBlockRing::kBlockFrames], 2.f)) << "blocks arrive in order";
    };

    "the playback ring takes only whole blocks while audio streams and flushes the tail once the input ended"_test = [] {
        using gr::audio::detail::WorkletBlockRing;
        gr::audio::detail::AudioSinkState<float> state;
        state.recreateBuffer(4096UZ);
        state.numChannels = 2UZ;
        std::vector<float> interleaved(2UZ * 200UZ);
        for (std::size_t frame = 0UZ; frame < 200UZ; ++frame) {
            interleaved[2UZ * frame]       = static_cast<float>(frame);
            interleaved[2UZ * frame + 1UZ] = -static_cast<float>(frame);
        }
        expect(eq(state.writeFromInput(std::span<const float>(interleaved), 2UZ), interleaved.size())) << fatal;
        WorkletBlockRing ring(16UZ, 2UZ);

        state.feedWorklet(ring, 4UZ);
        expect(eq(ring.filledBlocks(), 1UZ)) << "200 staged frames make one whole block, the rest waits for more";
        expect(eq(ring.slotSamples(0U)[1], 1.f) && eq(ring.slotSamples(0U)[WorkletBlockRing::kBlockFrames + 1UZ], -1.f)) << "a block is planar per channel";

        ring.missedBlocks.store(2U);
        state.feedWorklet(ring, 4UZ);
        expect(eq(state.underrunCount.load(), 2UZ)) << "an empty ring while streaming is an underrun";
        expect(eq(ring.filledBlocks(), 1UZ));

        state.intentionalSilence.store(true);
        ring.missedBlocks.store(5U);
        state.feedWorklet(ring, 4UZ);
        expect(eq(ring.filledBlocks(), 2UZ)) << "once the input ended the partial tail is flushed";
        expect(eq(state.underrunCount.load(), 2UZ)) << "silence after the input ended is not an underrun";
        expect(eq(state.reader.available(), 0UZ));
    };

    "silence before the first samples is not an underrun, a gap after them is"_test = [] {
        gr::audio::detail::AudioSinkState<float> state;
        state.recreateBuffer(8UZ);
        std::array<float, 2UZ> output{};
        state.readPlanarFloat(output.data(), 2UZ, 1UZ);
        expect(eq(state.underrunCount.load(), 0UZ));
        const std::array<float, 1UZ> samples{1.f};
        expect(eq(state.writeFromInput(std::span<const float>(samples), 1UZ), 1UZ));
        state.readPlanarFloat(output.data(), 2UZ, 1UZ);
        expect(eq(state.underrunCount.load(), 1UZ));
    };

    "invalid rates fail validation while unusual positive preferences remain negotiable"_test = [] {
        using gr::audio::detail::validSampleRate;
        expect(eq(validSampleRate(441010.f), 441010U));
        expect(eq(validSampleRate(0.f), 0U));
        expect(eq(validSampleRate(-1.f), 0U));
        expect(eq(validSampleRate(std::numeric_limits<float>::infinity()), 0U));
        expect(eq(validSampleRate(std::numeric_limits<float>::quiet_NaN()), 0U));
        expect(eq(validSampleRate(std::numeric_limits<float>::max()), 0U));
    };

    "playback transfer bounds cover page-rounded buffers and adaptive expansion"_test = [] {
        for (const std::size_t channels : {1UZ, 2UZ, 8UZ}) {
            const std::size_t scratch  = 240001UZ * channels;
            const std::size_t space    = 240640UZ * channels;
            const std::size_t transfer = gr::audio::detail::playbackTransferSamples(space, space, scratch, channels);
            expect(transfer < scratch);
            expect(eq(transfer % channels, 0UZ));
            std::vector<float>                     input(transfer, 0.5f);
            std::vector<float>                     output(scratch, -1.f);
            gr::algorithm::DriftCompensator<float> drift;
            drift.mode                 = gr::algorithm::DriftCorrection::AdaptiveResampling;
            const std::size_t adjusted = drift.compensateSink(input, output, transfer, 48000.0 * 0.999, 48000.0, channels);
            expect(adjusted > transfer);
            expect(adjusted <= std::min(space, scratch));
        }
        expect(eq(gr::audio::detail::playbackTransferSamples(64UZ, 1UZ, 64UZ, 1UZ), 0UZ));
        expect(eq(gr::audio::detail::playbackTransferSamples(64UZ, 64UZ, 64UZ, 0UZ), 0UZ));
    };

    "legacy settings normalise without overwriting an explicit request or channel count"_test = [] {
        gr::Graph graph;
        auto&     legacy = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 44100.f}, {"num_channels", gr::Size_t{2U}}});
        expect(eq(legacy.req_sample_rate.value, 44100.f));
        expect(eq(legacy.sample_rate.value, 0.f));
        expect(eq(legacy.n_outputs.value, 2U));
        auto& explicitSource = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"req_sample_rate", 96000.f}, {"sample_rate", 44100.f}, {"n_outputs", gr::Size_t{8U}}, {"num_channels", gr::Size_t{2U}}});
        expect(eq(explicitSource.req_sample_rate.value, 96000.f));
        expect(eq(explicitSource.out.size(), 8UZ));
        expect(eq(explicitSource.num_channels.value, 8U));
    };

    "settings stored as other numeric types still configure rate and width"_test = [] {
        gr::Graph graph;
        auto&     sink   = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"sample_rate", 44100.0}, {"num_channels", 2}});
        auto&     source = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"n_outputs", 3}, {"req_sample_rate", 96000.0}});
        expect(eq(sink.req_sample_rate.value, 44100.f));
        expect(eq(sink.n_inputs.value, 2U));
        expect(eq(sink.num_channels.value, 2U));
        expect(eq(source.out.size(), 3UZ));
        expect(eq(source.req_sample_rate.value, 96000.f));
    };

    "a port collection follows its width until a port is connected"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{2U}}}, {0UZ, 1UZ});
        auto&                                            sink = fixture.sink;
        expect(sink.settings().set({{"n_inputs", gr::Size_t{3U}}}).empty());
        std::ignore = sink.settings().activateContext();
        std::ignore = sink.settings().applyStagedParameters();
        expect(eq(sink.n_inputs.value, 2U));
        expect(eq(sink.in.size(), 2UZ));

        gr::Graph unconnected;
        auto&     source = unconnected.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"n_outputs", gr::Size_t{8U}}});
        expect(source.settings().set({{"n_outputs", gr::Size_t{2U}}}).empty());
        std::ignore = source.settings().activateContext();
        std::ignore = source.settings().applyStagedParameters();
        expect(eq(source.n_outputs.value, 2U));
        expect(eq(source.out.size(), 2UZ));
    };

#ifdef GR_ENABLE_BLOCK_REGISTRY
    "legacy and multiplexed registry aliases create the expected port topology"_test = [] {
        gr::BlockRegistry registry;
        expect(eq(gr::registerBlock<gr::audio::AudioSource<float>, "gr::audio::AudioSource<float>">(registry), 0));
        expect(eq(gr::registerBlock<gr::audio::AudioSink<float>, "gr::audio::AudioSink<float>">(registry), 0));
        expect(eq(gr::registerBlock<gr::audio::AudioSourceMultiplexed<float>, "gr::audio::AudioSourceMultiplexed<float>">(registry), 0));
        expect(eq(gr::registerBlock<gr::audio::AudioSinkMultiplexed<float>, "gr::audio::AudioSinkMultiplexed<float>">(registry), 0));
        gr::Graph   graph;
        const auto& source            = graph.addBlock(registry.create("gr::audio::AudioSource<float32>", {{"num_channels", gr::Size_t{2U}}}));
        const auto& sink              = graph.addBlock(registry.create("gr::audio::AudioSink<float32>", {{"num_channels", gr::Size_t{2U}}}));
        const auto& multiplexedSource = graph.addBlock(registry.create("gr::audio::AudioSourceMultiplexed<float32>", {{"n_outputs", gr::Size_t{8U}}}));
        const auto& multiplexedSink   = graph.addBlock(registry.create("gr::audio::AudioSinkMultiplexed<float32>", {{"n_inputs", gr::Size_t{8U}}}));
        expect(source != nullptr && sink != nullptr && multiplexedSource != nullptr && multiplexedSink != nullptr) << fatal;
        expect(eq(source->dynamicOutputPortsSize(), 1UZ));
        expect(eq(sink->dynamicInputPortsSize(), 1UZ));
        expect(eq(multiplexedSource->dynamicOutputPortsSize(0UZ), 8UZ));
        expect(eq(multiplexedSink->dynamicInputPortsSize(0UZ), 8UZ));
    };

    "a saved multiplexed graph reloads with its port collections and connections"_test = [] {
        gr::BlockRegistry     registry;
        gr::SchedulerRegistry schedulers;
        expect(eq(gr::registerBlock<gr::audio::AudioSourceMultiplexed<float>, "gr::audio::AudioSourceMultiplexed<float>">(registry), 0));
        expect(eq(gr::registerBlock<gr::audio::AudioSinkMultiplexed<float>, "gr::audio::AudioSinkMultiplexed<float>">(registry), 0));
        gr::PluginLoader           loader(registry, schedulers, {});
        constexpr std::string_view stereoLoopback    = R"(blocks:
  - id: gr::audio::AudioSourceMultiplexed<float32>
    parameters:
      name: source
      n_outputs: 2
  - id: gr::audio::AudioSinkMultiplexed<float32>
    parameters:
      name: sink
      n_inputs: 2
connections:
  - [source, [0, 0], sink, [0, 0]]
  - [source, [0, 1], sink, [0, 1]]
)";
        const auto                 expectStereoPorts = [](const gr::Graph& graph) {
            for (const auto& block : graph.blocks()) {
                if (block->name() == "source") {
                    expect(eq(block->dynamicOutputPortsSize(0UZ), 2UZ));
                } else if (block->name() == "sink") {
                    expect(eq(block->dynamicInputPortsSize(0UZ), 2UZ));
                }
            }
            expect(eq(graph.edges().size(), 2UZ));
        };
        auto loaded = gr::loadGrc(loader, stereoLoopback);
        expect(loaded.has_value()) << fatal;
        expectStereoPorts(*loaded.value());
        auto reloaded = gr::loadGrc(loader, gr::saveGrc(loader, *loaded.value()));
        expect(reloaded.has_value()) << fatal;
        expectStereoPorts(*reloaded.value());
    };
#endif

    "a later PCM rate tag is checked only when its samples are staged"_test = [] {
        SinkTestGraph<float> fixture({{"req_sample_rate", 48000.f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        const std::array<float, 2UZ> samples{1.f, 2.f};
        queueAudio<float>(producer, samples);
        const gr::property_map rateChange{gr::tag::SAMPLE_RATE(96000.f)};
        producer.out.publishTag(rateChange, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(1UZ);
            output[0]   = 3.f;
            output.publish(1UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(2UZ);
            expect(sink.processBulk(input) == gr::work::Status::OK);
        }
        expect(eq(sink.in.streamReader().available(), 1UZ));
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(sink.processBulk(input) == gr::work::Status::ERROR);
        }
        expect(eq(sink.in.streamReader().available(), 1UZ));
    };

    "multiplexed inputs disagreeing on the first rate are rejected without consuming samples"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}}, {0UZ, 1UZ});
        auto&                                            sink   = fixture.sink;
        auto&                                            first  = fixture.producer(0UZ);
        auto&                                            second = fixture.producer(1UZ);
        fixture.start();
        const std::array<float, 1UZ> samples{1.f};
        queueAudio<float>(first, samples);
        queueAudio<float>(second, samples, 96000.f);
        {
            auto      spans = sinkInputSpans(sink, 1UZ);
            std::span inputs(spans);
            expect(sink.processBulk(inputs) == gr::work::Status::ERROR);
        }
        expect(eq(sink.in[0].streamReader().available(), 1UZ));
        expect(eq(sink.in[1].streamReader().available(), 1UZ));
    };

    "disconnected multiplexed slots stay zero under both device padding policies"_test = [] {
        for (const auto padding : {ChannelPadding::zero, ChannelPadding::cyclic}) {
            verifySparsePlayback<float>(padding);
            verifySparsePlayback<std::int16_t>(padding);
        }
    };

    "a starved connected input prevents other channels from advancing"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}}, {0UZ, 1UZ});
        auto&                                            sink  = fixture.sink;
        auto&                                            first = fixture.producer(0UZ);
        fixture.start();
        const std::array<float, 2UZ> samples{1.f, 2.f};
        queueAudio<float>(first, samples);
        {
            auto      spans = sinkInputSpans(sink, 2UZ);
            std::span inputs(spans);
            expect(sink.processBulk(inputs) == gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        }
        expect(eq(sink.in[0].streamReader().available(), 2UZ));
        expect(eq(sink._totalStagedSamples, 0UZ));
    };

    "a blocked multiplexed output drops capture for all ports together"_test = [] {
        using Consumer = gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>;
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"n_outputs", gr::Size_t{2U}}, {"emit_timing_tags", false}, {"drift_correction", "None"}});
        auto&     fast   = graph.emplaceBlock<Consumer>();
        auto&     slow   = graph.emplaceBlock<Consumer>();
        expect(graph.connect(source, "out#0", fast, "in").has_value()) << fatal;
        expect(graph.connect(source, "out#1", slow, "in").has_value()) << fatal;
        expect(graph.connectPendingEdges()) << fatal;
        auto& state = prepareSourceWithoutDevice(source);
        {
            auto full = source.out[1].streamWriter().reserve(source.out[1].streamWriter().available());
            std::ranges::fill(full, 0.f);
            full.publish(full.size());
        }
        const std::array<float, 2UZ> input{0.25f, 0.5f};
        expect(eq(state.writePlanarFloat(input.data(), 2UZ, 1UZ, 2UZ), 4UZ));
        source.publishSamples(4UZ, 2UZ);
        expect(eq(fast.in.streamReader().available(), 0UZ));
        expect(eq(state.reader.available(), 0UZ));
        expect(eq(state.overflowCount.load(), 1UZ));
        {
            auto discard = slow.in.streamReader().get();
            expect(discard.consume(discard.size()));
        }
        expect(eq(state.writePlanarFloat(input.data(), 2UZ, 1UZ, 2UZ), 4UZ));
        source.publishSamples(4UZ, 2UZ);
        expect(eq(fast.in.streamReader().available(), 2UZ));
        expect(eq(slow.in.streamReader().available(), 2UZ));
    };

#if !defined(__EMSCRIPTEN__)
    "a first tagged rate the device cannot play is rejected without consuming samples"_test = [] {
        SinkTestGraph<float> fixture({{"req_sample_rate", 48000.f}, {"io_buffer_size", 0.1f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        const std::array<float, 2UZ> samples{1.f, 2.f};
        queueAudio<float>(producer, samples, 4.f);
        {
            auto span = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(2UZ);
            expect(sink.processBulk(span) == gr::work::Status::ERROR);
        }
        expect(ge(sink.sample_rate.value, 8000.f));
        expect(eq(sink.req_sample_rate.value, 48000.f));
        expect(eq(sink.in.streamReader().available(), 2UZ));
        expect(eq(sink._totalStagedSamples, 0UZ));
        expect(sink._lastError.find("explicit resampler") != std::string::npos);
    };

    "multiplexed inputs agreeing on a first rate renegotiate playback without changing the request"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}, {"io_buffer_size", 0.1f}}, {0UZ, 1UZ});
        auto&                                            sink   = fixture.sink;
        auto&                                            first  = fixture.producer(0UZ);
        auto&                                            second = fixture.producer(1UZ);
        fixture.start();
        const std::array<float, 2UZ> samples{1.f, 2.f};
        queueAudio<float>(first, samples, 44100.f);
        queueAudio<float>(second, samples, 44100.f);
        {
            auto      spans = sinkInputSpans(sink, 2UZ);
            std::span inputs(spans);
            expect(sink.processBulk(inputs) == gr::work::Status::OK);
        }
        expect(eq(sink.sample_rate.value, 44100.f));
        expect(eq(sink.req_sample_rate.value, 48000.f));
        expect(eq(sink._totalStagedSamples, 2UZ * sink._activeConfig.numChannels));
    };
#endif

    "a format tag that splits an interleaved frame is rejected"_test = [] {
        SinkTestGraph<float> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        const gr::property_map stereo{gr::tag::SAMPLE_RATE(48000.f), gr::tag::NUM_CHANNELS(gr::Size_t{2U})};
        producer.out.publishTag(stereo, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(3UZ);
            std::ranges::copy(std::array{1.f, 2.f, 3.f}, output.begin());
            output.publish(3UZ);
        }
        producer.out.publishTag(stereo, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(1UZ);
            output[0]   = 4.f;
            output.publish(1UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(4UZ);
            expect(sink.processBulk(input) == gr::work::Status::OK);
        }
        expect(eq(sink._totalStagedSamples, sink._activeConfig.numChannels));
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(2UZ);
            expect(sink.processBulk(input) == gr::work::Status::ERROR);
        }
        expect(eq(sink.in.streamReader().available(), 2UZ));
        expect(sink._lastError.find("splits an interleaved frame") != std::string::npos);
    };

    "a trailing partial interleaved frame waits for more input"_test = [] {
        SinkTestGraph<float> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        producer.out.publishTag(gr::property_map{gr::tag::SAMPLE_RATE(48000.f), gr::tag::NUM_CHANNELS(gr::Size_t{2U})}, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(3UZ);
            std::ranges::copy(std::array{1.f, 2.f, 3.f}, output.begin());
            output.publish(3UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(3UZ);
            expect(sink.processBulk(input) == gr::work::Status::OK);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(sink.processBulk(input) == gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        }
        expect(!sink._failed.load());
        expect(eq(sink.in.streamReader().available(), 1UZ));
    };

    "an unconnected source discards capture without counting an overflow"_test = [] {
        gr::Graph                    graph;
        auto&                        source = graph.emplaceBlock<gr::audio::AudioSourceMultiplexed<float>>({{"n_outputs", gr::Size_t{2U}}, {"emit_timing_tags", false}, {"drift_correction", "None"}});
        auto&                        state  = prepareSourceWithoutDevice(source);
        const std::array<float, 2UZ> input{0.25f, 0.5f};
        expect(eq(state.writePlanarFloat(input.data(), 2UZ, 1UZ, 2UZ), 4UZ));
        source.publishSamples(4UZ, 2UZ);
        expect(eq(state.reader.available(), 0UZ));
        expect(eq(state.overflowCount.load(), 0UZ));
    };

    "buffers are sized from the duration at the negotiated rate"_test = [] {
        using gr::audio::detail::bufferFramesFor;
        using gr::audio::detail::kMinBufferFrames;
        expect(eq(bufferFramesFor(5.f, 48000U), 240000UZ));
        expect(eq(bufferFramesFor(5.f, 1U), kMinBufferFrames));
        expect(eq(bufferFramesFor(0.f, 48000U), kMinBufferFrames));
        expect(eq(bufferFramesFor(-1.f, 48000U), kMinBufferFrames));
    };

    "all disconnected inputs stage no fabricated frames"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{2U}}, {"req_sample_rate", 48000.f}}, {});
        auto&                                            sink = fixture.sink;
        fixture.start();
        auto      spans = sinkInputSpans(sink, 0UZ);
        std::span inputs(spans);
        expect(sink.processBulk(inputs) == gr::work::Status::INSUFFICIENT_INPUT_ITEMS);
        expect(eq(sink._totalStagedSamples, 0UZ));
    };

    "unequal multiplexed stream lengths stop at the earliest connected end-of-stream"_test = [] {
        SinkTestGraph<float, AudioPortMode::multiplexed> fixture({{"n_inputs", gr::Size_t{3U}}, {"req_sample_rate", 48000.f}}, {0UZ, 2UZ});
        auto&                                            sink   = fixture.sink;
        auto&                                            first  = fixture.producer(0UZ);
        auto&                                            second = fixture.producer(1UZ);
        fixture.start();
        const std::array<float, 2UZ> shortStream{1.f, 2.f};
        const std::array<float, 4UZ> longStream{3.f, 4.f, 5.f, 6.f};
        queueAudio<float>(first, shortStream);
        queueAudio<float>(second, longStream);
        first.publishEoS();
        second.publishEoS();
        expect(sink.work().status == gr::work::Status::OK);
        expect(eq(sink._totalStagedSamples, 2UZ * sink._activeConfig.numChannels));
        expect(sink.work().status == gr::work::Status::DONE);
        expect(eq(sink._totalStagedSamples, 2UZ * sink._activeConfig.numChannels));
    };

#if !defined(__EMSCRIPTEN__)
    "native layout selection prefers the narrowest layout covering the request, else the widest below it"_test = [] {
        auto                   mono   = *soundio_channel_layout_get_default(1);
        auto                   stereo = *soundio_channel_layout_get_default(2);
        auto                   quad   = *soundio_channel_layout_get_default(4);
        std::array             layouts{quad, mono, stereo};
        SoundIoSampleRateRange rates{48000, 48000};
        SoundIoDevice          device{};
        device.layouts           = layouts.data();
        device.layout_count      = static_cast<int>(layouts.size());
        device.sample_rates      = &rates;
        device.sample_rate_count = 1;
        expect(eq(gr::audio::detail::selectSoundIoLayout(&device, 3U)->channel_count, 4));
        expect(eq(gr::audio::detail::selectSoundIoLayout(&device, 8U)->channel_count, 4));
        device.layouts      = &quad;
        device.layout_count = 1;
        expect(eq(gr::audio::detail::selectSoundIoLayout(&device, 1U)->channel_count, 4));
        device.layout_count = 0;
        expect(gr::audio::detail::selectSoundIoLayout(&device, 1U) == nullptr);
    };

    "native devices without sample-rate ranges fail rate selection"_test = [] {
        SoundIoSampleRateRange rates{44100, 48000};
        SoundIoDevice          device{};
        device.sample_rates      = &rates;
        device.sample_rate_count = 1;
        expect(eq(gr::audio::detail::selectSoundIoSampleRate(&device, 96000U).value_or(0), 48000));
        device.sample_rate_count = 0;
        expect(!gr::audio::detail::selectSoundIoSampleRate(&device, 48000U).has_value());
    };

    "a resampling sound server opens at the requested rate, not its current device rate"_test = [] {
        SoundIoSampleRateRange serverRate{96000, 96000};
        SoundIo                soundServer{};
        SoundIoDevice          device{};
        device.soundio              = &soundServer;
        device.sample_rates         = &serverRate;
        device.sample_rate_count    = 1;
        soundServer.current_backend = SoundIoBackendPulseAudio;
        expect(eq(gr::audio::detail::selectSoundIoSampleRate(&device, 48000U).value_or(0), 48000));
        soundServer.current_backend = SoundIoBackendAlsa;
        expect(eq(gr::audio::detail::selectSoundIoSampleRate(&device, 48000U).value_or(0), 96000));
    };

    "an external stop discards staged audio instead of playing it out"_test = [] {
        SinkTestGraph<float> fixture({{"req_sample_rate", 48000.f}, {"io_buffer_size", 2.f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer();
        fixture.start();
        const std::size_t        stagingCapacity = sink._staging.writer.available();
        const std::vector<float> chunk(1024UZ, 0.5f);
        for (std::size_t iteration = 0UZ; iteration < 1000UZ && sink._staging.reader.available() < stagingCapacity / 2UZ; ++iteration) {
            queueAudio<float>(producer, chunk);
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(sink.in.streamReader().available());
            expect(sink.processBulk(input) != gr::work::Status::ERROR);
        }
        expect(ge(sink._staging.reader.available(), stagingCapacity / 2UZ)) << fatal;
        stopSink(sink);
        fixture.started = false;
        expect(gt(sink._staging.reader.available(), 0UZ)) << "the stop played the staged audio out";
    };

    "a stop on upstream end-of-stream plays the staged audio out"_test = [] {
        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t{48000U}}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"req_sample_rate", 48000.f}, {"io_buffer_size", 0.2f}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << fatal;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << fatal;
        expect(sched.runAndWait().has_value());
        expect(eq(sink._totalStagedSamples, 48000UZ));
        expect(eq(sink._staging.reader.available(), 0UZ)) << "the end of the stream was discarded";
    };

    "the playback device buffer stays short whatever the staging buffer size"_test = [] {
        gr::audio::detail::SoundIoSinkBackend<float> backend;
        const auto                                   format = backend.start({.sampleRate = 48000U, .numChannels = 2U, .bufferSeconds = 5.f, .useDummyBackendForTests = true});
        expect(format.has_value()) << fatal;
        expect(le(backend.softwareLatency(), 2. * gr::audio::detail::kMaxDeviceLatencySeconds)) << "playback would start behind a staging buffer's worth of silence";
        backend.shutdown();
    };

    "a very low requested rate keeps native buffers sized by duration"_test = [] {
        gr::Graph graph;
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"req_sample_rate", 1.f}, {"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        sink._inputChannels           = 7UZ;
        sink.start();
        gr::on_scope_exit stopSinkOnExit{[&sink] { sink.stop(); }};
        expect(!sink._failed.load()) << fatal;
        const auto rate     = static_cast<std::uint32_t>(sink.sample_rate.value);
        const auto capacity = gr::audio::detail::bufferFramesFor(0.1f, rate) * sink._activeConfig.numChannels;
        expect(ge(rate, 8000U));
        expect(eq(sink._bufferCapacity, capacity));
        expect(le(sink._backendImpl.state().buffer.size(), 2UZ * capacity));
        expect(eq(sink._inputChannels, 0UZ));

        auto& source                    = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"req_sample_rate", 1.f}, {"io_buffer_size", 0.1f}});
        source._useDummyBackendForTests = true;
        source.start();
        gr::on_scope_exit stopSourceOnExit{[&source] { source.stop(); }};
        expect(!source._failed.load()) << fatal;
        expect(le(source._backendImpl.state().buffer.size(), 2UZ * gr::audio::detail::bufferFramesFor(0.1f, static_cast<std::uint32_t>(source.sample_rate.value))));
    };

    "native rate and layout helpers fall back from stereo 96 kHz to mono 48 kHz"_test = [] {
        auto                   mono = *soundio_channel_layout_get_default(1);
        SoundIoSampleRateRange rates{48000, 48000};
        SoundIoDevice          device{};
        device.layouts           = &mono;
        device.layout_count      = 1;
        device.sample_rates      = &rates;
        device.sample_rate_count = 1;
        expect(eq(soundio_device_nearest_sample_rate(&device, 96000), 48000));
        expect(eq(soundio_device_nearest_sample_rate(&device, 441010), 48000));
        const auto* selected = gr::audio::detail::selectSoundIoLayout(&device, 2U);
        expect(selected != nullptr) << fatal;
        expect(eq(selected->channel_count, 1));
    };

    "a backend restart within a run does not reopen the first-format window"_test = [] {
        SinkTestGraph<float> fixture({{"req_sample_rate", 48000.f}, {"io_buffer_size", 0.1f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        {
            auto output = producer.out.streamWriter().reserve(2UZ);
            std::ranges::copy(std::array{1.f, 2.f}, output.begin());
            output.publish(2UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(2UZ);
            expect(sink.processBulk(input) == gr::work::Status::OK);
        }
        sink._restartPending = true;
        producer.out.publishTag(gr::property_map{gr::tag::SAMPLE_RATE(44100.f)}, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(1UZ);
            output[0]   = 3.f;
            output.publish(1UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(sink.processBulk(input) == gr::work::Status::ERROR);
        }
        expect(eq(sink.sample_rate.value, 48000.f));
        expect(eq(sink.in.streamReader().available(), 1UZ));
    };

    "start clears a stale restart request and error, and multiplexed silence is intentional"_test = [] {
        gr::Graph graph;
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSinkMultiplexed<float>>({{"n_inputs", gr::Size_t{2U}}, {"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        sink._restartPending          = true;
        sink._lastError               = "stale";
        sink.start();
        gr::on_scope_exit stopOnExit{[&sink] { sink.stop(); }};
        expect(!sink._failed.load()) << fatal;
        expect(!sink._restartPending);
        expect(sink._lastError.empty());
        expect(sink._backendImpl.state().intentionalSilence.load());
    };

    "a source reconfiguration publishes the renegotiated rate to its settings"_test = [] {
        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"req_sample_rate", 22050.f}, {"io_buffer_size", 0.1f}});
        source._useDummyBackendForTests = true;
        source.start();
        gr::on_scope_exit stopOnExit{[&source] { source.stop(); }};
        expect(!source._failed.load()) << fatal;
        expect(eq(source.settings().get().value_or<float>("sample_rate", 0.f), 22050.f));
        source.req_sample_rate = 44100.f;
        source.reconfigure();
        expect(eq(source.settings().get().value_or<float>("sample_rate", 0.f), 44100.f));
    };

    "an invalid requested rate fails start with a clear error"_test = [] {
        gr::Graph graph;
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}});
        sink._useDummyBackendForTests = true;
        sink.req_sample_rate          = 0.f;
        sink.start();
        expect(sink._failed.load());
        expect(sink._lastError.find("req_sample_rate positive") != std::string::npos);
    };
#endif

    "format tags stored as other numeric types are converted safely"_test = [] {
        SinkTestGraph<float> fixture({{"req_sample_rate", 48000.f}});
        auto&                sink     = fixture.sink;
        auto&                producer = fixture.producer(0UZ);
        fixture.start();
        producer.out.publishTag(gr::property_map{{gr::tag::SAMPLE_RATE.shortKey(), 48000.0}, {gr::tag::NUM_CHANNELS.shortKey(), std::int64_t{1}}}, 0UZ);
        {
            auto output = producer.out.streamWriter().reserve(2UZ);
            std::ranges::copy(std::array{1.f, 2.f}, output.begin());
            output.publish(2UZ);
        }
        {
            auto input = sink.in.template get<gr::SpanReleasePolicy::ProcessAll>(2UZ);
            expect(sink.processBulk(input) == gr::work::Status::OK);
        }
        expect(eq(sink._totalStagedSamples, 2UZ * sink._activeConfig.numChannels));
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
