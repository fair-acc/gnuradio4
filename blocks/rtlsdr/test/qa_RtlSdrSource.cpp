#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <numeric>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/rtlsdr/RtlSdrSource.hpp>

#include <gnuradio-4.0/rtlsdr/RtlSdrDevice.hpp>

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

        for (const auto& s : samples) {
            auto i = static_cast<double>(s.real());
            auto q = static_cast<double>(s.imag());
            sumI += i;
            sumQ += q;
            sumPower += i * i + q * q;
        }

        auto   n        = static_cast<double>(samples.size());
        double meanI    = sumI / n;
        double meanQ    = sumQ / n;
        double meanPow  = sumPower / n;
        double rmsLevel = 10.0 * std::log10(meanPow > 0.0 ? meanPow : 1e-20);

        std::println("  [STATS] IQ samples: {}  ({:.2f} s at {:.0f} kHz)", samples.size(), n / sampleRate, sampleRate / 1e3);
        std::println("  [STATS] DC offset:  I={:+.4f}  Q={:+.4f}", meanI, meanQ);
        std::println("  [STATS] RMS power:  {:.1f} dBFS", rmsLevel);
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        double sum    = 0.0;
        double sumSq  = 0.0;
        auto   nBytes = static_cast<double>(samples.size());

        for (auto v : samples) {
            double centred = static_cast<double>(v) - 127.5;
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

} // namespace

const boost::ut::suite<"RtlSdrSource"> rtlSdrTests = [] {
    "RtlSdrSource<uint8_t> is constructible"_test = [] {
        gr::blocks::rtlsdr::RtlSdrSource<std::uint8_t> block({});
        expect(eq(block.center_frequency.value, 100.0e6));
        expect(eq(block.sample_rate.value, 2.048e6));
        expect(block.auto_gain.value);
        expect(eq(block.device_index.value, 0U));
    };

    "RtlSdrSource<complex<float>> is constructible"_test = [] {
        gr::blocks::rtlsdr::RtlSdrSource<std::complex<float>> block({});
        expect(eq(block.center_frequency.value, 100.0e6));
    };

    "RtlSdrSource settings via emplaceBlock"_test = [] {
        Graph testGraph;
        auto& block = testGraph.emplaceBlock<gr::blocks::rtlsdr::RtlSdrSource<std::complex<float>>>({
            {"center_frequency", 433.92e6},
            {"sample_rate", 1.024e6},
            {"gain", 20.f},
            {"auto_gain", false},
            {"device_index", 1U},
            {"ppm_correction", std::int32_t{-5}},
        });
        expect(eq(block.center_frequency.value, 433.92e6));
        expect(eq(block.sample_rate.value, 1.024e6));
        expect(eq(block.gain.value, 20.f));
        expect(!block.auto_gain.value);
        expect(eq(block.device_index.value, 1U));
        expect(eq(block.ppm_correction.value, std::int32_t{-5}));
    };

    "IQ conversion produces normalised values"_test = [] {
        std::array<std::uint8_t, 8>        raw = {0, 255, 127, 128, 255, 0, 0, 0};
        std::array<std::complex<float>, 4> result{};

        gr::blocks::rtlsdr::detail::convertToComplex(raw.data(), result.data(), 4UZ);

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
        gr::blocks::rtlsdr::RtlSdrDevice probe;
        bool                             hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        std::uint32_t deviceCount = hasDevice ? 1U : 0U;
        if (deviceCount == 0U) {
            std::println("  [SKIP] no RTL-SDR device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr double kSampleRate      = 2.048e6;
        constexpr double kCenterFrequency = 100.0e6; // FM broadcast band
        constexpr auto   kCaptureDuration = std::chrono::seconds(2);
        constexpr auto   kExpectedSamples = static_cast<std::size_t>(kSampleRate * 2);

        std::println("  [INFO] device detected");

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::rtlsdr::RtlSdrSource<std::complex<float>>>({
            {"center_frequency", kCenterFrequency},
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

        printNoiseStats<std::complex<float>>(sink._samples, kSampleRate);

        std::println("  [INFO] tags received: {}", sink._tags.size());
        for (const auto& tag : sink._tags) {
            std::println("    tag @{}: {} entries", tag.index, tag.map.size());
            for (const auto& [key, val] : tag.map) {
                std::println("      {} = {}", std::string_view(key), val);
            }
        }
    };

    "real device uint8_t raw capture with noise stats"_test = [] {
        gr::blocks::rtlsdr::RtlSdrDevice probe;
        bool                             hasDevice = probe.open(0).has_value();
        if (hasDevice) {
            probe.close();
        }
        std::uint32_t deviceCount = hasDevice ? 1U : 0U;
        if (deviceCount == 0U) {
            std::println("  [SKIP] no RTL-SDR device detected");
            expect(true) << "skipped — no device";
            return;
        }

        constexpr double kSampleRate      = 1.024e6;
        constexpr double kCenterFrequency = 433.92e6; // ISM band
        constexpr auto   kCaptureDuration = std::chrono::seconds(3);

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<gr::blocks::rtlsdr::RtlSdrSource<std::uint8_t>>({
            {"center_frequency", kCenterFrequency},
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

        printNoiseStats<std::uint8_t>(sink._samples, kSampleRate);

        std::println("  [INFO] tags received: {}", sink._tags.size());
        for (const auto& tag : sink._tags) {
            std::println("    tag @{}: {} entries", tag.index, tag.map.size());
        }
    };
};

int main() { return 0; }
