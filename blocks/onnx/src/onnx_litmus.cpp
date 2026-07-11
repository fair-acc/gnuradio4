// MVP C++ integration for the peaks-tensor regressor: loads the model through
// OnnxPeakDetector and reproduces the Python litmus table on the SAME spectra
// (src/ex4_python/litmus.py --dump-bin), so any divergence is a C++ integration fault
// rather than a modelling difference.
//
//   onnx_litmus <model.onnx|model.ort> <litmus_cases.bin> [gate_threshold]

#include <gnuradio-4.0/onnx/OnnxPeakDetector.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <print>
#include <string>
#include <vector>

namespace {

struct Case {
    float              centre = 0.f;
    float              amp    = 0.f;
    float              sigmaL = 0.f;
    float              sigmaR = 0.f;
    std::vector<float> samples;
};

[[nodiscard]] std::vector<Case> readCases(const std::string& path, std::size_t& modelN) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::int32_t nCases = 0;
    std::int32_t n      = 0;
    in.read(reinterpret_cast<char*>(&nCases), sizeof(nCases));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    modelN = static_cast<std::size_t>(n);

    std::vector<Case> cases;
    cases.reserve(static_cast<std::size_t>(nCases));
    for (std::int32_t i = 0; i < nCases; ++i) {
        Case                 c;
        std::array<float, 4> gt{};
        in.read(reinterpret_cast<char*>(gt.data()), sizeof(gt));
        c.centre = gt[0];
        c.amp    = gt[1];
        c.sigmaL = gt[2];
        c.sigmaR = gt[3];
        c.samples.resize(modelN);
        in.read(reinterpret_cast<char*>(c.samples.data()), static_cast<std::streamsize>(modelN * sizeof(float)));
        if (!in) {
            return {};
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

struct Cell {
    std::vector<float> pos, amp, sig, asym;
    std::size_t        detected = 0;
    std::size_t        trials   = 0;
    std::size_t        extra    = 0;
};

[[nodiscard]] float median(std::vector<float> v) {
    if (v.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

// the grid is emitted width-major then S/N-major, TRIALS per cell — mirrors litmus.py
constexpr std::array kWidths{"narrow", "medium", "wide"};
constexpr std::array kSnrs{"small", "middle", "large"};

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::println("usage: onnx_litmus <model> <litmus_cases.bin> [gate_threshold] [max_peaks, 0=uncapped]");
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::string casePath  = argv[2];
    const float       theta     = (argc > 3) ? std::stof(argv[3]) : 0.5f;
    // 0 = uncapped: litmus.py applies NO host-side cap, so the default max_peaks=8 would make
    // the C++ table systematically under-count FP relative to the Python reference.
    const gr::Size_t maxPeaks = (argc > 4) ? static_cast<gr::Size_t>(std::stoul(argv[4])) : 0U;

    std::size_t modelN = 0;
    auto        cases  = readCases(casePath, modelN);
    if (cases.empty()) {
        std::println("failed to read cases from {}", casePath);
        return 1;
    }
    const std::size_t nCells = kWidths.size() * kSnrs.size();
    const std::size_t trials = cases.size() / nCells;

    gr::blocks::onnx::OnnxPeakDetector block;
    block.model_path     = modelPath;
    block.gate_threshold = theta;
    block.resample_mode  = gr::blocks::onnx::ResampleMode::None;
    // uncapped must EXCEED the model's K, otherwise the block's own default (8) silently applies
    block.max_peaks = (maxPeaks > 0U) ? maxPeaks : gr::Size_t{4096};
    block.start();
    if (!block.isModelLoaded()) {
        std::println("model not loadable: {}", modelPath);
        return 1;
    }
    std::println("model : {}", modelPath);
    std::println("layout: {}", block.peaks_layout.value);
    std::println("cases : {} ({} cells x {} trials), N={}\n", cases.size(), nCells, trials, modelN);

    std::vector<Cell> cells(nCells);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c    = cases[i];
        Cell&       cell = cells[i / trials];
        ++cell.trials;

        gr::DataSet<float> input;
        input.signal_names      = {"Spectrum"};
        input.signal_units      = {"a.u."};
        input.signal_quantities = {""};
        input.signal_ranges     = {gr::Range<float>{0.f, 0.f}};
        input.extents           = {static_cast<std::int32_t>(modelN)};
        input.meta_information  = {{}};
        input.timing_events     = {{}};
        input.signal_values     = c.samples;

        const auto  output = block.processOne(input);
        const auto& events = output.timing_events.empty() ? decltype(output.timing_events[0]){} : output.timing_events[0];
        if (events.empty()) {
            continue;
        }

        const float sigmaAvg = 0.5f * (c.sigmaL + c.sigmaR);
        const float tol      = std::max(sigmaAvg, 3.f);

        // nearest committed peak to the injected one — use the fractional `centre`
        // property, NOT the integer timing_events index, or sub-bin accuracy is lost
        auto centreOf = [](const gr::DataSet<float>::idx_pmt_map& ev) {
            auto it = ev.second.find(std::pmr::string("centre"));
            return it != ev.second.end() ? it->second.value_or<float>(0.f) : static_cast<float>(ev.first);
        };
        std::size_t best     = 0;
        float       bestDist = std::numeric_limits<float>::max();
        for (std::size_t k = 0; k < events.size(); ++k) {
            const float d = std::abs(centreOf(events[k]) - c.centre);
            if (d < bestDist) {
                bestDist = d;
                best     = k;
            }
        }
        if (bestDist > tol) {
            cell.extra += events.size();
            continue;
        }
        ++cell.detected;
        cell.extra += events.size() - 1;

        const auto& props = events[best].second;
        auto        prop  = [&props](const char* key) -> float {
            auto it = props.find(std::pmr::string(key));
            return it != props.end() ? it->second.value_or<float>(0.f) : 0.f;
        };
        const float sigL = prop("sigma_left");
        const float sigR = prop("sigma_right");

        cell.pos.push_back(bestDist / sigmaAvg);
        cell.amp.push_back(std::abs(prop("amplitude") - c.amp) / c.amp);
        cell.sig.push_back(std::abs(prop("sigma") - sigmaAvg) / sigmaAvg);
        cell.asym.push_back(std::abs((sigR - sigL) - (c.sigmaR - c.sigmaL)) / sigmaAvg);
    }

    std::println("{:>7} {:>7} | {:>9} {:>9} {:>9} {:>9} | {:>6} {:>6}", "width", "S/N", "pos/sig", "amp", "sig/sig", "asym", "det", "fp");
    std::println("{}", std::string(78, '-'));
    for (std::size_t w = 0; w < kWidths.size(); ++w) {
        for (std::size_t s = 0; s < kSnrs.size(); ++s) {
            const Cell& cell = cells[w * kSnrs.size() + s];
            std::println("{:>7} {:>7} | {:>8.1f}% {:>8.1f}% {:>8.1f}% {:>8.1f}% | {:>5.0f}% {:>6.2f}", kWidths[w], kSnrs[s], 100.f * median(cell.pos), 100.f * median(cell.amp), 100.f * median(cell.sig), 100.f * median(cell.asym), 100.f * static_cast<float>(cell.detected) / static_cast<float>(std::max(cell.trials, 1UZ)), static_cast<float>(cell.extra) / static_cast<float>(std::max(cell.trials, 1UZ)));
        }
        std::println("");
    }
    block.stop();
    return 0;
}
