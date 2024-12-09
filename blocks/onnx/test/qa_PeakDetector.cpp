#include <boost/ut.hpp>

#include <gnuradio-4.0/onnx/PeakDetector.hpp>

#include <cmath>
#include <numbers>
#include <print>

using namespace boost::ut;
using namespace gr::blocks::onnx;

namespace {

gr::DataSet<float> makeGaussianSpectrum(std::size_t n, std::span<const std::pair<float, float>> peaks, float noiseFloor = 0.1f) {
    gr::DataSet<float> ds;
    ds.signal_names      = {"Spectrum"};
    ds.signal_units      = {"a.u."};
    ds.signal_quantities = {""};
    ds.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
    ds.extents           = {static_cast<std::int32_t>(n)};
    ds.meta_information  = {{}};
    ds.timing_events     = {{}};

    ds.signal_values.resize(n, noiseFloor);
    for (const auto& [center, amplitude] : peaks) {
        float sigma = 3.0f;
        for (std::size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - center;
            ds.signal_values[i] += amplitude * std::exp(-0.5f * x * x / (sigma * sigma));
        }
    }

    return ds;
}

} // namespace

const boost::ut::suite<"PeakDetector"> peakDetectorTests = [] {
    "detects known peaks at high SNR"_test = [] {
        constexpr std::size_t   n       = 512;
        std::pair<float, float> peaks[] = {{100.f, 10.f}, {250.f, 8.f}, {400.f, 12.f}};

        auto input = makeGaussianSpectrum(n, peaks);

        PeakDetector detector;
        detector.prominence_threshold = 3.0f;
        detector.min_peak_distance    = 5;
        detector.max_peaks            = 8;

        auto output = detector.processOne(std::move(input));

        expect(eq(output.signal_names.size(), 2UZ));
        expect(eq(output.signal_names[0], std::string("Spectrum")));
        expect(eq(output.signal_names[1], std::string("Prominence")));
        expect(eq(output.signal_values.size(), 2 * n));

        expect(eq(output.timing_events.size(), 2UZ));
        const auto& peakEvents = output.timing_events[0];

        std::println("detected {} peaks (expected 3)", peakEvents.size());
        for (const auto& [idx, props] : peakEvents) {
            auto  confIt = props.find(std::pmr::string("confidence"));
            auto  sigIt  = props.find(std::pmr::string("sigma"));
            float conf   = confIt != props.end() ? confIt->second.value_or<float>(0.f) : 0.f;
            float sigma  = sigIt != props.end() ? sigIt->second.value_or<float>(0.f) : 0.f;
            std::println("  bin={} conf={:.2f} sigma={:.1f}", idx, conf, sigma);
        }

        expect(eq(peakEvents.size(), 3UZ)) << "should detect exactly 3 peaks";

        // verify positions are within 2 bins of the true centres
        for (const auto& [idx, props] : peakEvents) {
            bool nearExpected = false;
            for (const auto& [center, amp] : peaks) {
                if (std::abs(static_cast<float>(idx) - center) < 2.f) {
                    nearExpected = true;
                    break;
                }
            }
            expect(nearExpected) << "peak at bin " << idx << " is not near any expected position";
        }
    };

    "respects max_peaks limit"_test = [] {
        constexpr std::size_t   n       = 512;
        std::pair<float, float> peaks[] = {{50.f, 10.f}, {150.f, 8.f}, {250.f, 12.f}, {350.f, 6.f}, {450.f, 9.f}};

        auto input = makeGaussianSpectrum(n, peaks);

        PeakDetector detector;
        detector.prominence_threshold = 2.0f;
        detector.max_peaks            = 3;

        auto output = detector.processOne(std::move(input));
        expect(le(output.timing_events[0].size(), 3UZ)) << "should respect max_peaks=3";
    };

    "returns zero peaks for flat spectrum"_test = [] {
        constexpr std::size_t n = 256;
        gr::DataSet<float>    input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {""};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.extents           = {static_cast<std::int32_t>(n)};
        input.meta_information  = {{}};
        input.timing_events     = {{}};
        input.signal_values.assign(n, 1.0f);

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));

        expect(output.timing_events[0].empty()) << "flat spectrum should produce no peaks";
    };

    "detects single narrow peak"_test = [] {
        constexpr std::size_t   n       = 1024;
        std::pair<float, float> peaks[] = {{512.f, 20.f}};

        auto input = makeGaussianSpectrum(n, peaks, 0.05f);

        PeakDetector detector;
        detector.prominence_threshold = 3.0f;

        auto output = detector.processOne(std::move(input));
        expect(eq(output.timing_events[0].size(), 1UZ)) << "should detect exactly 1 peak";

        if (!output.timing_events[0].empty()) {
            auto pos = output.timing_events[0][0].first;
            expect(lt(std::abs(pos - 512), std::ptrdiff_t(3))) << "peak position should be near bin 512";
        }
    };

    "NMS suppresses nearby peaks"_test = [] {
        constexpr std::size_t n = 256;
        // two peaks separated by only 4 bins
        std::pair<float, float> peaks[] = {{100.f, 10.f}, {104.f, 8.f}};

        auto input = makeGaussianSpectrum(n, peaks);

        PeakDetector detector;
        detector.min_peak_distance = 10;

        auto output = detector.processOne(std::move(input));
        expect(eq(output.timing_events[0].size(), 1UZ)) << "NMS should suppress the weaker nearby peak";
    };

    "output DataSet preserves metadata"_test = [] {
        constexpr std::size_t   n       = 128;
        std::pair<float, float> peaks[] = {{64.f, 10.f}};

        auto input        = makeGaussianSpectrum(n, peaks);
        input.timestamp   = 12345;
        input.axis_names  = {"frequency"};
        input.axis_units  = {"Hz"};
        input.axis_values = {{0.f, 1.f}};

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));

        expect(eq(output.timestamp, std::int64_t(12345)));
        expect(eq(output.axis_names[0], std::string("frequency")));
        expect(eq(output.axis_units[0], std::string("Hz")));
    };

    "peak properties contain required fields"_test = [] {
        constexpr std::size_t   n       = 512;
        std::pair<float, float> peaks[] = {{256.f, 15.f}};

        auto input = makeGaussianSpectrum(n, peaks);

        PeakDetector detector;
        auto         output = detector.processOne(std::move(input));

        expect(eq(output.timing_events[0].size(), 1UZ));
        if (!output.timing_events[0].empty()) {
            const auto& props = output.timing_events[0][0].second;
            expect(props.contains(std::pmr::string("confidence")));
            expect(props.contains(std::pmr::string("sigma")));
            expect(props.contains(std::pmr::string("amplitude")));
            expect(props.contains(std::pmr::string("w68")));
            expect(props.contains(std::pmr::string("w96")));
        }
    };
};

int main() { /* boost::ut */ }
