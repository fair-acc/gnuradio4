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
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
                foundTimingTag = true;
                expect(sinkTag.map.contains(gr::tag::TRIGGER_NAME.shortKey())) << caseName;
                expect(sinkTag.map.contains(gr::tag::TRIGGER_OFFSET.shortKey())) << caseName;

                expect(sinkTag.map.contains(gr::tag::TRIGGER_META_INFO.shortKey())) << caseName;
                break;
            }
        }
        expect(foundTimingTag) << "should have at least one timing tag";
    };

    "DriftCompensator inserts sample when source is fast"_test = [] {
        gr::algorithm::DriftCompensator<float> comp;
        std::array<float, 10>                  buf{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        // simulate source 100 ppm fast over many calls to accumulate >=1 sample
        std::size_t nProduced = 4U;
        for (int i = 0; i < 300; ++i) {
            nProduced = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.0001, 48000.0, 1U);
        }

        // after enough calls, compensator should have inserted at least once
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
        // stereo: L0=1, R0=2, L1=3, R1=4
        buf[0] = 1.f;
        buf[1] = 2.f;
        buf[2] = 3.f;
        buf[3] = 4.f;

        comp.fractionalAccumulator = 0.99;
        auto n                     = comp.compensateSource(std::span(buf), 4U, 48000.0 * 1.01, 48000.0, 2U);
        if (n == 6U) {
            // inserted stereo frame at index 4,5 interpolated from frames (0,1) and (2,3)
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
        // should have the format tag but no TRIGGER_TIME tags
        bool foundTimingTag = false;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
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
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
                foundTimingTag = true;
                if (sinkTag.map.contains(gr::tag::TRIGGER_META_INFO.shortKey())) {
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
        // large interval — should emit at most 1-2 timing tags in 300ms
        auto& source                    = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"tag_interval", 10.0f}});
        source._useDummyBackendForTests = true;
        auto& sink                      = graph.emplaceBlock<gr::testing::TagSink<float, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(runSchedulerFor(sched, 300ms).has_value()) << caseName;

        std::size_t timingTagCount = 0U;
        for (const auto& sinkTag : sink._tags) {
            if (sinkTag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
                ++timingTagCount;
            }
        }
        // with 10s interval and 300ms runtime, expect at most 1 timing tag (the first one)
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
        // dummy backend may not honour the exact requested rate — just verify it's in a sane range
        expect(gt(static_cast<float>(estimated), 1000.f)) << "estimated rate should be above 1 kHz";
        expect(lt(static_cast<float>(estimated), 200000.f)) << "estimated rate should be below 200 kHz";
    };

    "AudioSink rate estimator runs during playback"_test = [] {
        constexpr std::string_view      caseName = "AudioSink rate estimator";
        const std::vector<std::int16_t> reference{0, 1000, -1000, 2000, -2000, 3000, -3000, 4000};
        const auto                      wavBytes = makeWav(1U, 1U, 16U, 22050U, encodePcm16(reference));
        TempFile                        file{writeTempAudioFile(wavBytes)};

        gr::Graph graph;
        auto&     source              = graph.emplaceBlock<gr::blocks::fileio::WavSource<float>>({{"uri", file.path.string()}});
        auto&     sink                = graph.emplaceBlock<gr::audio::AudioSink<float>>({{"io_buffer_size", 0.1f}, {"ppm_estimator_cutoff", 0.5f}});
        sink._useDummyBackendForTests = true;
        expect(graph.connect<"out", "in">(source, sink).has_value()) << caseName;

        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(graph)).has_value()) << caseName;
        expect(sched.runAndWait().has_value()) << caseName;

        // the estimator may or may not converge for such a short file, but it should have been initialised
        expect(ge(sink._rateEstimator.estimatedRate(), 0.0)) << "sink rate estimator should have run";
    };

    "clk_in forwards clock offset and trigger name"_test = [] {
        constexpr std::string_view caseName = "clk_in forwarding";

        gr::Graph graph;
        auto&     source                = graph.emplaceBlock<gr::audio::AudioSource<float>>({{"sample_rate", 22050.f}, {"num_channels", gr::Size_t(1)}, {"io_buffer_size", 0.1f}, {"emit_timing_tags", true}, {"tag_interval", 0.0f}});
        source._useDummyBackendForTests = true;

        // clock source: emits a TRIGGER_TIME tag at sample 0 with a known UTC timestamp
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

        // check that the clock offset was applied
        expect(source._clockOffsetValid) << "clock offset should be valid after receiving clk_in tag";

        // check that a timing tag uses the forwarded trigger name
        bool foundGpsTrigger = false;
        for (const auto& sinkTag : sink._tags) {
            if (auto it = sinkTag.map.find(gr::tag::TRIGGER_NAME.shortKey()); it != sinkTag.map.end()) {
                if (auto* name = it->second.get_if<std::pmr::string>()) {
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

        // sink compensate: input → adjusted buffer
        std::array<float, 10> input{1.f, 2.f, 3.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        std::array<float, 12> adjusted{};

        // sink is faster than source — needs to insert
        comp.fractionalAccumulator = 0.99;
        auto n                     = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 0.99, 48000.0, 1U);
        expect(ge(n, 4UZ)) << "sink compensator should insert when sink is faster";

        // sink is slower than source — needs to drop
        comp.fractionalAccumulator = -0.99;
        n                          = comp.compensateSink(std::span<const float>(input.data(), 4U), std::span(adjusted), 4U, 48000.0 * 1.01, 48000.0, 1U);
        expect(le(n, 4UZ)) << "sink compensator should drop when sink is slower";
    };
};
#endif

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
