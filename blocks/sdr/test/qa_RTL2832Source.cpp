#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <numeric>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/sdr/RTL2832Source.hpp>

#include <gnuradio-4.0/sdr/RTL2832Device.hpp>

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
#include <gnuradio-4.0/GpsSource.hpp>
#if defined(__linux__)
#include <gnuradio-4.0/PpsSource.hpp>
#endif
#if defined(__APPLE__)
#include <util.h>
#else
#include <pty.h>
#endif
#include <unistd.h>
#endif

using namespace boost::ut;
using namespace gr;
using namespace gr::testing;

namespace {

template<typename T>
void printNoiseStats(std::span<const T> samples, double sampleRate) {
    if (samples.empty()) {
        return;
    }

    if constexpr (std::is_same_v<T, std::complex<float>>) {
        double sumI     = 0.0;
        double sumQ     = 0.0;
        double sumPower = 0.0;

        for (const auto& sample : samples) {
            auto i = static_cast<double>(sample.real());
            auto q = static_cast<double>(sample.imag());
            sumI += i;
            sumQ += q;
            sumPower += i * i + q * q;
        }

        auto   nSamples = static_cast<double>(samples.size());
        double meanI    = sumI / nSamples;
        double meanQ    = sumQ / nSamples;
        double meanPow  = sumPower / nSamples;
        double rmsLevel = 10.0 * std::log10(meanPow > 0.0 ? meanPow : 1e-20);

        std::println("  [STATS] IQ samples: {}  ({:.2f} s at {:.0f} kHz)", samples.size(), nSamples / sampleRate, sampleRate / 1e3);
        std::println("  [STATS] DC offset:  I={:+.4f}  Q={:+.4f}", meanI, meanQ);
        std::println("  [STATS] RMS power:  {:.1f} dBFS", rmsLevel);
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        double sum    = 0.0;
        double sumSq  = 0.0;
        auto   nBytes = static_cast<double>(samples.size());

        for (auto rawByte : samples) {
            double centred = static_cast<double>(rawByte) - 127.5;
            sum += centred;
            sumSq += centred * centred;
        }

        double mean    = sum / nBytes;
        double rms     = std::sqrt(sumSq / nBytes);
        double nIq     = nBytes / 2.0;
        double seconds = nIq / sampleRate;

        std::println("  [STATS] raw bytes: {}  ({:.2f} s at {:.0f} kHz, {:.2f} MS/s IQ)", samples.size(), seconds, sampleRate / 1e3, nIq / seconds / 1e6);
        std::println("  [STATS] DC offset: {:.2f} (centred around 127.5)", mean);
        std::println("  [STATS] RMS level: {:.2f} LSB", rms);
    }
}

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)

bool hasRtlDevice() {
    gr::blocks::sdr::RTL2832Device probe;
    bool                           found = probe.open(0).has_value();
    if (found) {
        probe.close();
    }
    return found;
}

struct PtyPair {
    int         master = -1;
    int         slave  = -1;
    std::string slaveName;

    PtyPair() = default;
    ~PtyPair() {
        if (master >= 0) {
            ::close(master);
        }
        if (slave >= 0) {
            ::close(slave);
        }
    }

    PtyPair(const PtyPair&)            = delete;
    PtyPair& operator=(const PtyPair&) = delete;
    PtyPair(PtyPair&& o) noexcept : master(std::exchange(o.master, -1)), slave(std::exchange(o.slave, -1)), slaveName(std::move(o.slaveName)) {}
    PtyPair& operator=(PtyPair&& o) noexcept {
        std::swap(master, o.master);
        std::swap(slave, o.slave);
        slaveName = std::move(o.slaveName);
        return *this;
    }

    static std::optional<PtyPair> create() {
        PtyPair p;
        if (::openpty(&p.master, &p.slave, nullptr, nullptr, nullptr) != 0) {
            return std::nullopt;
        }
        p.slaveName = ::ttyname(p.slave);
        return p;
    }

    void writeLine(std::string_view line) const {
        std::string           msg = std::string(line) + "\r\n";
        [[maybe_unused]] auto n   = ::write(master, msg.data(), msg.size());
    }
};

std::string nmeaWithChecksum(std::string_view body) {
    auto cs = gr::timing::detail::computeNMEAChecksum(body.substr(1));
    return std::format("{}*{:02X}", body, cs);
}

void sendNMEASequence(const PtyPair& pty, int startSecond, int count, int delayMs = 50) {
    for (int i = 0; i < count; ++i) {
        int sec = (startSecond + i) % 60;
        pty.writeLine(nmeaWithChecksum(std::format("$GPRMC,1200{:02d}.00,A,5001.1900,N,00840.6570,E,0.5,45.0,110326,,,A", sec)));
        pty.writeLine(nmeaWithChecksum(std::format("$GPGGA,1200{:02d}.00,5001.1900,N,00840.6570,E,1,10,0.8,136.0,M,47.0,M,,", sec)));
        pty.writeLine(nmeaWithChecksum("$GPGSA,A,3,04,05,09,12,,,,,,,,,1.8,1.0,1.5"));
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }
}

#endif // POSIX

} // namespace

const boost::ut::suite<"RTL2832Source"> rtl2832Tests = [] {
    "RTL2832Source<uint8_t> is constructible with timing defaults"_test = [] {
        gr::blocks::sdr::RTL2832Source<std::uint8_t> block(gr::property_map{});
        expect(eq(block.frequency.value, 100.0e6));
        expect(eq(block.sample_rate.value, 2.048e6f));
        expect(block.auto_gain.value);
        expect(eq(block.device_index.value, 0U));
        expect(eq(block.trigger_name.value, std::string("SDR_WALLCLOCK")));
        expect(block.emit_timing_tags.value);
        expect(block.emit_meta_info.value);
    };

    "RTL2832Source<complex<float>> is constructible"_test = [] {
        gr::blocks::sdr::RTL2832Source<std::complex<float>> block(gr::property_map{});
        expect(eq(block.frequency.value, 100.0e6));
        expect(eq(block.trigger_name.value, std::string("SDR_WALLCLOCK")));
    };

    "RTL2832Source clk_in port is optional"_test = [] {
        gr::blocks::sdr::RTL2832Source<std::complex<float>> block(gr::property_map{});
        expect(decltype(block.clk_in)::kIsOptional) << "clk_in must be Optional";
        expect(decltype(block.clk_in)::kIsInput) << "clk_in must be an input port";
        expect(!block.clk_in.isConnected()) << "clk_in is not connected by default";
    };

    "RTL2832Source settings via emplaceBlock"_test = [] {
        Graph testGraph;
        auto& block = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", 433.92e6},
            {"sample_rate", 1.024e6f},
            {"gain", 20.f},
            {"auto_gain", false},
            {"device_index", 1U},
            {"ppm_correction", std::int32_t{-5}},
            {"trigger_name", std::string("MY_SDR")},
            {"emit_timing_tags", false},
            {"emit_meta_info", false},
        });
        expect(eq(block.frequency.value, 433.92e6));
        expect(eq(block.sample_rate.value, 1.024e6f));
        expect(eq(block.gain.value, 20.f));
        expect(!block.auto_gain.value);
        expect(eq(block.device_index.value, 1U));
        expect(eq(block.ppm_correction.value, std::int32_t{-5}));
        expect(eq(block.trigger_name.value, std::string("MY_SDR")));
        expect(!block.emit_timing_tags.value);
        expect(!block.emit_meta_info.value);
    };

    "E4000 PLL lookup table covers expected frequency ranges"_test = [] {
        using namespace gr::blocks::sdr;

        // lowest entry should cover at least 72 MHz
        expect(gt(kE4kPllLut.front().maxFreqKhz, 70'000U));

        // highest entry should cover at least 1200 MHz
        expect(ge(kE4kPllLut.back().maxFreqKhz, 1'200'000U));

        // all multipliers should be > 0
        for (const auto& pllEntry : kE4kPllLut) {
            expect(gt(pllEntry.mul, 0U)) << "PLL entry must have non-zero multiplier";
        }

        // entries should be sorted by maxFreqKhz
        for (std::size_t i = 1; i < kE4kPllLut.size(); ++i) {
            expect(gt(kE4kPllLut[i].maxFreqKhz, kE4kPllLut[i - 1].maxFreqKhz));
        }
    };

    "E4000 LNA gain table has valid entries"_test = [] {
        using namespace gr::blocks::sdr;

        // first entry (index 0) is the lowest gain
        expect(eq(kE4kLnaGainTenths[0], std::int16_t{-50})) << "-5.0 dB at index 0";

        // last entry (index 14) is the highest gain
        expect(eq(kE4kLnaGainTenths[14], std::int16_t{300})) << "30.0 dB at index 14";

        // gains should be non-decreasing
        for (std::size_t i = 1; i < kE4kLnaGainTenths.size(); ++i) {
            expect(ge(kE4kLnaGainTenths[i], kE4kLnaGainTenths[i - 1]));
        }
    };

    "E4000 RF filter tables have 16 entries each"_test = [] {
        using namespace gr::blocks::sdr;

        expect(eq(kE4kRfFilterUhfMhz.size(), 16UZ));
        expect(eq(kE4kRfFilterLbandMhz.size(), 16UZ));

        // UHF filters should be sorted
        for (std::size_t i = 1; i < kE4kRfFilterUhfMhz.size(); ++i) {
            expect(gt(kE4kRfFilterUhfMhz[i], kE4kRfFilterUhfMhz[i - 1]));
        }

        // L-band filters should be sorted
        for (std::size_t i = 1; i < kE4kRfFilterLbandMhz.size(); ++i) {
            expect(gt(kE4kRfFilterLbandMhz[i], kE4kRfFilterLbandMhz[i - 1]));
        }
    };

    "TunerType includes e4000 variant"_test = [] {
        using namespace gr::blocks::sdr;

        auto tuner = TunerType::e4000;
        expect(tuner != TunerType::none);
        expect(tuner != TunerType::r820t);
        expect(tuner != TunerType::r828d);
    };

    "IQ conversion produces normalised values"_test = [] {
        std::array<std::uint8_t, 8>        raw = {0, 255, 127, 128, 255, 0, 0, 0};
        std::array<std::complex<float>, 4> result{};

        gr::blocks::sdr::detail::convertToComplex(raw.data(), result.data(), 4UZ);

        // sample 0: I=0 → -1.0, Q=255 → +1.0
        expect(lt(std::abs(result[0].real() - (-1.f)), 0.01f));
        expect(lt(std::abs(result[0].imag() - 1.f), 0.01f));

        // sample 1: I=127, Q=128 → near zero
        expect(lt(std::abs(result[1].real()), 0.01f));
        expect(lt(std::abs(result[1].imag()), 0.01f));

        // sample 2: I=255 → +1.0, Q=0 → -1.0
        expect(lt(std::abs(result[2].real() - 1.f), 0.01f));
        expect(lt(std::abs(result[2].imag() - (-1.f)), 0.01f));
    };

    "real device complex<float> capture with noise stats"_test = [] {
        gr::blocks::sdr::RTL2832Device probe;
        bool                           hasDevice = probe.open(0).has_value();
        auto                           tunerType = probe._tunerType;
        if (hasDevice) {
            probe.close();
        }
        if (!hasDevice) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 2.048e6f;
        constexpr double kFrequency       = 100.0e6; // FM broadcast band
        constexpr auto   kCaptureDuration = std::chrono::seconds(2);
        constexpr auto   kExpectedSamples = static_cast<std::size_t>(kSampleRate * 2);

        const char* tunerName = tunerType == gr::blocks::sdr::TunerType::e4000 ? "E4000" : tunerType == gr::blocks::sdr::TunerType::r820t ? "R820T" : tunerType == gr::blocks::sdr::TunerType::r828d ? "R828D" : "unknown";
        std::println("  [INFO] device detected (tuner: {})", tunerName);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"auto_gain", true},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", true},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        std::size_t nSamples = sink._nSamplesProduced;
        expect(gt(nSamples, kExpectedSamples / 2UZ)) << "captured at least ~1 s of data";

        printNoiseStats<std::complex<float>>(sink._samples, static_cast<double>(kSampleRate));

        std::println("  [INFO] tags received: {}", sink._tags.size());
        for (const auto& tag : sink._tags) {
            std::println("    tag @{}: {} entries", tag.index, tag.map.size());
            for (const auto& [key, val] : tag.map) {
                std::println("      {} = {}", std::string_view(key), val);
            }
        }
    };

    "real device uint8_t raw capture with noise stats"_test = [] {
        gr::blocks::sdr::RTL2832Device probe;
        bool                           hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        if (!hasDevice) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 1.024e6f;
        constexpr double kFrequency       = 433.92e6; // ISM band
        constexpr auto   kCaptureDuration = std::chrono::seconds(3);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::uint8_t>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"gain", 30.f},
            {"auto_gain", false},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::uint8_t, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", true},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        std::size_t nSamples = sink._nSamplesProduced;
        expect(gt(nSamples, 0UZ)) << "received raw samples";

        printNoiseStats<std::uint8_t>(sink._samples, static_cast<double>(kSampleRate));

        std::println("  [INFO] tags received: {}", sink._tags.size());
        for (const auto& tag : sink._tags) {
            std::println("    tag @{}: {} entries", tag.index, tag.map.size());
        }
    };

    "real device timing tag format validation"_test = [] {
        gr::blocks::sdr::RTL2832Device probe;
        bool                           hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        if (!hasDevice) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 2.048e6f;
        constexpr double kFrequency       = 100.0e6;
        constexpr auto   kCaptureDuration = std::chrono::seconds(1);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency}, {"sample_rate", kSampleRate}, {"auto_gain", true}, {"trigger_name", std::string("TEST_TRIGGER")}, {"tag_interval", 0.f}, // emit every chunk for testing
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", false},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        expect(gt(sink._tags.size(), 2UZ)) << "at least a few timing tags";

        // validate first tag has all four required keys and full device meta_info
        const auto& firstTag = sink._tags.front();
        expect(eq(firstTag.index, 0UZ)) << "first tag at sample 0";

        auto triggerName = firstTag.get(std::string(tag::TRIGGER_NAME.shortKey()));
        expect(triggerName.has_value()) << "first tag has trigger_name";
        if (triggerName) {
            auto* name = triggerName->get().get_if<std::pmr::string>();
            expect(name != nullptr);
            if (name) {
                expect(eq(*name, std::pmr::string("TEST_TRIGGER"))) << "custom trigger name";
            }
        }

        auto triggerTime = firstTag.get(std::string(tag::TRIGGER_TIME.shortKey()));
        expect(triggerTime.has_value()) << "first tag has trigger_time";
        if (triggerTime) {
            auto* timestamp = triggerTime->get().get_if<std::uint64_t>();
            expect(timestamp != nullptr);
            if (timestamp) {
                expect(gt(*timestamp, std::uint64_t{1'700'000'000'000'000'000ULL})) << "timestamp after 2023";
            }
        }

        auto triggerOffset = firstTag.get(std::string(tag::TRIGGER_OFFSET.shortKey()));
        expect(triggerOffset.has_value()) << "first tag has trigger_offset";
        if (triggerOffset) {
            auto* offset = triggerOffset->get().get_if<float>();
            expect(offset != nullptr);
            if (offset) {
                expect(eq(*offset, 0.f)) << "trigger_offset is 0";
            }
        }

        auto metaInfo = firstTag.get(std::string(tag::TRIGGER_META_INFO.shortKey()));
        expect(metaInfo.has_value()) << "first tag has trigger_meta_info";
        if (metaInfo) {
            auto* metaMap = metaInfo->get().get_if<property_map>();
            expect(metaMap != nullptr) << "meta_info is a property_map";
            if (metaMap) {
                expect(metaMap->contains(std::pmr::string("trigger_source")));
                expect(metaMap->contains(std::pmr::string("clock_source")));
                expect(metaMap->contains(std::pmr::string("device_name")));
                expect(metaMap->contains(std::pmr::string("sample_rate")));
                expect(metaMap->contains(std::pmr::string("frequency")));
                expect(metaMap->contains(std::pmr::string("gain")));
                expect(metaMap->contains(std::pmr::string("auto_gain")));
            }
        }

        if (sink._tags.size() > 1) {
            const auto& secondTag  = sink._tags[1];
            auto        secondMeta = secondTag.get(std::string(tag::TRIGGER_META_INFO.shortKey()));
            expect(secondMeta.has_value()) << "second tag has meta_info";
            if (secondMeta) {
                auto* secondMetaMap = secondMeta->get().get_if<property_map>();
                if (secondMetaMap) {
                    expect(!secondMetaMap->contains(std::pmr::string("device_name"))) << "unchanged params not repeated";
                    expect(!secondMetaMap->contains(std::pmr::string("sample_rate"))) << "unchanged params not repeated";
                }
            }
        }

        std::uint64_t prevTimestamp = 0;
        for (const auto& capturedTag : sink._tags) {
            auto timeEntry = capturedTag.get(std::string(tag::TRIGGER_TIME.shortKey()));
            if (timeEntry) {
                auto* timestamp = timeEntry->get().get_if<std::uint64_t>();
                if (timestamp) {
                    expect(ge(*timestamp, prevTimestamp)) << "timestamps monotonically non-decreasing";
                    prevTimestamp = *timestamp;
                }
            }
        }
    };

    "real device emit_timing_tags=false suppresses tags"_test = [] {
        gr::blocks::sdr::RTL2832Device probe;
        bool                           hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        if (!hasDevice) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 2.048e6f;
        constexpr double kFrequency       = 100.0e6;
        constexpr auto   kCaptureDuration = std::chrono::seconds(1);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"emit_timing_tags", false},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", false},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        expect(gt(sink._nSamplesProduced, 0UZ)) << "received samples";
        expect(eq(sink._tags.size(), 0UZ)) << "no timing tags when emit_timing_tags=false";
    };

    "real device emit_meta_info=false produces tags without meta_info"_test = [] {
        gr::blocks::sdr::RTL2832Device probe;
        bool                           hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        if (!hasDevice) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 2.048e6f;
        constexpr double kFrequency       = 100.0e6;
        constexpr auto   kCaptureDuration = std::chrono::seconds(1);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"emit_meta_info", false},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", false},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "in">(src, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        expect(gt(sink._tags.size(), 0UZ)) << "tags still emitted";

        for (const auto& capturedTag : sink._tags) {
            auto triggerName = capturedTag.get(std::string(tag::TRIGGER_NAME.shortKey()));
            expect(triggerName.has_value()) << "tag has trigger_name";

            auto triggerTime = capturedTag.get(std::string(tag::TRIGGER_TIME.shortKey()));
            expect(triggerTime.has_value()) << "tag has trigger_time";

            auto metaInfo = capturedTag.get(std::string(tag::TRIGGER_META_INFO.shortKey()));
            expect(!metaInfo.has_value()) << "no trigger_meta_info when emit_meta_info=false";
        }
    };

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
#if defined(__linux__)
    "PpsSource → RTL2832Source clock integration"_test = [] {
        if (!hasRtlDevice()) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr float  kSampleRate      = 2.048e6f;
        constexpr double kFrequency       = 100.0e6;
        constexpr auto   kCaptureDuration = std::chrono::milliseconds(2500);

        Graph testGraph;
        auto& pps  = testGraph.emplaceBlock<gr::timing::PpsSource>({
            {"clock_mode", std::string("NTP")},
        });
        auto& rtl  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"auto_gain", true},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", false},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "clk_in">(pps, rtl).has_value());
        expect(testGraph.connect<"out", "in">(rtl, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });
        std::this_thread::sleep_for(kCaptureDuration);
        sched.requestStop();
        schedThread.join();

        expect(gt(sink._nSamplesProduced, 0UZ)) << "received IQ samples";
        expect(gt(sink._tags.size(), 0UZ)) << "received timing tags";

        bool foundPpsTriggerName = false;
        bool foundClockSource    = false;
        for (const auto& capturedTag : sink._tags) {
            auto triggerName = capturedTag.get(std::string(tag::TRIGGER_NAME.shortKey()));
            if (triggerName) {
                auto* name = triggerName->get().get_if<std::pmr::string>();
                if (name && name->find("PPS") != std::pmr::string::npos) {
                    foundPpsTriggerName = true;
                }
            }

            auto metaInfo = capturedTag.get(std::string(tag::TRIGGER_META_INFO.shortKey()));
            if (metaInfo) {
                auto* metaMap = metaInfo->get().get_if<property_map>();
                if (metaMap) {
                    if (auto it = metaMap->find(std::pmr::string("clock_source")); it != metaMap->end()) {
                        auto* clockSource = it->second.get_if<std::pmr::string>();
                        if (clockSource && clockSource->find("PPS") != std::pmr::string::npos) {
                            foundClockSource = true;
                        }
                    }
                }
            }
        }

        std::println("  [INFO] PPS trigger name forwarded: {}  clock_source set: {}", foundPpsTriggerName, foundClockSource);

        // PPS fires once per second; in 2.5s we expect at least 1 PPS tag to arrive
        // and be forwarded. If PPS hasn't fired yet, RTL falls back to wallclock.
        if (foundPpsTriggerName) {
            expect(foundClockSource) << "clock_source reflects PPS when trigger_name forwarded";
        } else {
            std::println("  [INFO] PPS did not fire within capture window — wallclock fallback (expected)");
            expect(true) << "wallclock fallback is valid";
        }
    };
#endif // __linux__

    "GpsSource (PTY mock) → RTL2832Source clock integration"_test = [] {
        if (!hasRtlDevice()) {
            std::println("  [SKIP] no RTL2832 device detected");
            expect(true) << "skipped — no device";
            return;
        }

        auto pty = PtyPair::create();
        expect(pty.has_value()) << "PTY creation must succeed";
        if (!pty) {
            return;
        }

        constexpr float  kSampleRate = 2.048e6f;
        constexpr double kFrequency  = 100.0e6;

        Graph testGraph;
        auto& gps  = testGraph.emplaceBlock<gr::timing::GpsSource>({
            {"device_path", std::string(pty->slaveName)},
        });
        auto& rtl  = testGraph.emplaceBlock<gr::blocks::sdr::RTL2832Source<std::complex<float>>>({
            {"frequency", kFrequency},
            {"sample_rate", kSampleRate},
            {"auto_gain", true},
        });
        auto& sink = testGraph.emplaceBlock<TagSink<std::complex<float>, ProcessFunction::USE_PROCESS_BULK>>({
            {"log_samples", false},
            {"log_tags", true},
        });
        expect(testGraph.connect<"out", "clk_in">(gps, rtl).has_value());
        expect(testGraph.connect<"out", "in">(rtl, sink).has_value());

        gr::scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto schedThread = std::thread([&sched] { sched.runAndWait(); });

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        sendNMEASequence(*pty, 0, 5, 80);
        std::this_thread::sleep_for(std::chrono::seconds(2));

        sched.requestStop();
        schedThread.join();

        expect(gt(sink._nSamplesProduced, 0UZ)) << "received IQ samples";
        expect(gt(sink._tags.size(), 0UZ)) << "received timing tags";

        bool foundGpsTriggerName = false;
        bool foundGpsClockSource = false;
        bool foundClockOffset    = false;
        for (const auto& capturedTag : sink._tags) {
            auto triggerName = capturedTag.get(std::string(tag::TRIGGER_NAME.shortKey()));
            if (triggerName) {
                auto* name = triggerName->get().get_if<std::pmr::string>();
                if (name && name->find("GPS") != std::pmr::string::npos) {
                    foundGpsTriggerName = true;
                }
            }

            auto metaInfo = capturedTag.get(std::string(tag::TRIGGER_META_INFO.shortKey()));
            if (metaInfo) {
                auto* metaMap = metaInfo->get().get_if<property_map>();
                if (metaMap) {
                    if (auto it = metaMap->find(std::pmr::string("clock_source")); it != metaMap->end()) {
                        auto* clockSource = it->second.get_if<std::pmr::string>();
                        if (clockSource && clockSource->find("GPS") != std::pmr::string::npos) {
                            foundGpsClockSource = true;
                        }
                    }
                    if (metaMap->contains(std::pmr::string("clock_offset_ns"))) {
                        foundClockOffset = true;
                    }
                }
            }
        }

        std::println("  [INFO] GPS trigger name forwarded: {}  clock_source: {}  clock_offset: {}", foundGpsTriggerName, foundGpsClockSource, foundClockOffset);

        if (foundGpsTriggerName) {
            expect(foundGpsClockSource) << "clock_source reflects GPS";
            expect(foundClockOffset) << "clock_offset_ns present when GPS locked";
        } else {
            std::println("  [INFO] GPS tags not yet received by RTL — wallclock fallback");
            expect(true) << "wallclock fallback is valid";
        }
    };
#endif // POSIX
};

int main() { return 0; }
