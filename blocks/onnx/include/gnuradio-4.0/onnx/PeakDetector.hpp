#ifndef GR_ONNX_PEAK_DETECTOR_HPP
#define GR_ONNX_PEAK_DETECTOR_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/onnx/PeakExtraction.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gr::blocks::onnx {

struct PeakDetector : gr::Block<PeakDetector> {
    using Description = Doc<"Detects peaks in spectra using classical local-maxima and prominence.">;

    gr::PortIn<gr::DataSet<float>>  in;
    gr::PortOut<gr::DataSet<float>> out;

    Annotated<float, "prominence threshold">   prominence_threshold = 3.0f; // in noise sigma
    Annotated<gr::Size_t, "min peak distance"> min_peak_distance    = 5U;   // bins
    Annotated<gr::Size_t, "max peaks">         max_peaks            = 8U;

    GR_MAKE_REFLECTABLE(PeakDetector, in, out, prominence_threshold, min_peak_distance, max_peaks);

    [[nodiscard]] gr::DataSet<float> processOne(gr::DataSet<float> inData) noexcept {
        if (inData.signal_values.empty()) {
            return inData;
        }

        const std::size_t      nSignals = std::max(1UZ, inData.signal_names.size());
        const std::size_t      n        = inData.signal_values.size() / nSignals;
        std::span<const float> spectrum(inData.signal_values.data(), n);

        // estimate noise floor: median and MAD of the spectrum
        auto [median, noiseSigma] = estimateNoise(spectrum);

        float absThreshold = median + prominence_threshold * noiseSigma;

        // find local maxima above threshold
        struct Candidate {
            std::size_t index;
            float       prominence;
        };
        std::vector<Candidate> candidates;

        for (std::size_t i = 1; i < n - 1; ++i) {
            if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] >= absThreshold) {
                float prom = estimateProminence(spectrum, i);
                candidates.push_back({i, prom});
            }
        }

        // sort by prominence descending
        std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) { return a.prominence > b.prominence; });

        // NMS: suppress neighbours within min_peak_distance
        std::vector<bool>      suppressed(n, false);
        std::vector<Candidate> accepted;
        accepted.reserve(std::min(candidates.size(), static_cast<std::size_t>(max_peaks.value)));

        for (const auto& c : candidates) {
            if (suppressed[c.index] || accepted.size() >= max_peaks) {
                continue;
            }
            accepted.push_back(c);

            std::size_t lo = (c.index > min_peak_distance) ? c.index - min_peak_distance : 0;
            std::size_t hi = std::min(c.index + static_cast<std::size_t>(min_peak_distance) + 1, n);
            for (std::size_t s = lo; s < hi; ++s) {
                suppressed[s] = true;
            }
        }

        // build prominence curve (for heatmap-like visualisation)
        std::vector<float> prominenceCurve(n, 0.f);
        for (const auto& c : candidates) {
            prominenceCurve[c.index] = c.prominence / (noiseSigma + 1e-10f);
        }

        // build output DataSet
        gr::DataSet<float> output;
        output.timestamp   = inData.timestamp;
        output.axis_names  = inData.axis_names;
        output.axis_units  = inData.axis_units;
        output.axis_values = inData.axis_values;

        output.signal_names      = {"Spectrum", "Prominence"};
        output.signal_quantities = {"", ""};
        output.signal_units      = {inData.signal_units.empty() ? "" : inData.signal_units[0], ""};
        output.signal_ranges     = {gr::Range<float>{0.f, 0.f}, gr::Range<float>{0.f, 0.f}};

        output.extents = {static_cast<std::int32_t>(n)};

        output.signal_values.resize(2 * n);
        std::copy_n(spectrum.begin(), n, output.signal_values.begin());
        std::copy(prominenceCurve.begin(), prominenceCurve.end(), output.signal_values.begin() + static_cast<std::ptrdiff_t>(n));

        output.meta_information = {{}, {}};

        // timing_events: one vector per signal; peaks go into signal 0
        std::vector<gr::DataSet<float>::idx_pmt_map> peakEvents;
        for (const auto& c : accepted) {
            auto [halfLeft, halfRight] = estimateWidth(spectrum, c.index);
            float sigma                = (halfLeft + halfRight) * 0.5f;

            gr::property_map props{
                {std::pmr::string("confidence"), gr::pmt::Value(c.prominence / (noiseSigma + 1e-10f))},
                {std::pmr::string("sigma"), gr::pmt::Value(sigma)},
                {std::pmr::string("amplitude"), gr::pmt::Value(spectrum[c.index] - median)},
                {std::pmr::string("w68"), gr::pmt::Value(sigma * 2.0f)},
                {std::pmr::string("w96"), gr::pmt::Value(sigma * 4.0f)},
            };
            peakEvents.emplace_back(static_cast<std::ptrdiff_t>(c.index), std::move(props));
        }
        output.timing_events = {std::move(peakEvents), {}};

        return output;
    }
};

} // namespace gr::blocks::onnx

inline const auto registerPeakDetector = gr::registerBlock<gr::blocks::onnx::PeakDetector>(gr::globalBlockRegistry());

#endif // GR_ONNX_PEAK_DETECTOR_HPP
