#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <random>
#include <vector>

struct GenData {
    std::vector<float> signal;
    std::vector<float> positions;
    std::vector<float> fwhms;
};

static GenData generateSignal(std::size_t nSize, std::size_t nPeaks, float noiseLevel = 0.05f, float asymmetryFactor = 0.2f) {
    std::vector<float> x(nSize);
    std::iota(x.begin(), x.end(), 0.0f);

    std::vector<float>                    signal(nSize, 0.0f);
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> posDistribution(50.f, static_cast<float>(nSize) - 50.f);
    std::uniform_real_distribution<float> fwhmDistribution(10.0f, 50.0f);
    std::uniform_real_distribution<float> ampDistribution(0.5f, 2.0f);
    std::uniform_real_distribution<float> asymDistribution(-asymmetryFactor, asymmetryFactor);

    std::vector<float> peakPosition(nPeaks);
    std::vector<float> fwhms(nPeaks);
    std::vector<float> amps(nPeaks);

    for (std::size_t i = 0UZ; i < nPeaks; ++i) {
        peakPosition[i] = posDistribution(gen);
        fwhms[i]        = fwhmDistribution(gen);
        amps[i]         = ampDistribution(gen);
    }

    for (std::size_t i = 0UZ; i < nPeaks; ++i) {
        const float stddev  = fwhms[i] / (2.0f * std::sqrt(2.0f * std::log(2.0f)));
        const float stddev2 = stddev * stddev;
        const float asym    = asymDistribution(gen);
        for (std::size_t idx = 0UZ; idx < nSize; ++idx) {
            const float diffPeak = x[idx] - peakPosition[i];
            float       val      = amps[i] * std::exp(-diffPeak * diffPeak / (2.f * stddev2));
            val *= (1.0f + asym * (x[idx] - peakPosition[i]) / fwhms[i]);
            signal[idx] += std::max(0.0f, val);
        }
    }

    // add noise
    std::normal_distribution<float> noise_dist(0.0f, noiseLevel * (*std::max_element(signal.begin(), signal.end())));
    for (auto& val : signal) {
        val += noise_dist(gen);
    }

    // sort peaks by position
    std::vector<std::size_t> idxs(nPeaks);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) { return peakPosition[a] < peakPosition[b]; });

    std::vector<float> sortedPositions(nPeaks);
    std::vector<float> sortedFWHM(nPeaks);
    for (std::size_t i = 0UZ; i < nPeaks; ++i) {
        sortedPositions[i] = peakPosition[idxs[i]];
        sortedFWHM[i]      = fwhms[idxs[i]];
    }

    return {std::move(signal), std::move(sortedPositions), std::move(sortedFWHM)};
}

int main() {
    // initialize ONNX Runtime
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "PeakDetector");
    Ort::RunOptions     runOptions;
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // load the ONNX model
    auto         modelPath = "peak_detector.onnx";
    Ort::Session session(env, modelPath, sessionOptions);
    std::array   inName{session.GetInputNameAllocated(0UZ, Ort::AllocatorWithDefaultOptions())};
    std::array   outName{session.GetOutputNameAllocated(0UZ, Ort::AllocatorWithDefaultOptions())};

    std::array<const char*, 1> inNames{inName[0].get()};
    std::array<const char*, 1> outNames{outName[0].get()};

    const std::size_t signalLength          = 1024UZ;
    const std::size_t numPeaks              = 5UZ;
    auto [signal, truePositions, trueFWHMs] = generateSignal(signalLength, numPeaks);

    // Prepare input tensor
    std::vector<int64_t> inputShape{1UZ, signalLength, 1UZ};
    std::vector<float>   inputData    = signal; // already have data in 'signal'
    Ort::MemoryInfo      mem_info     = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value           inputTensors = Ort::Value::CreateTensor<float>(mem_info, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

    // run inference
    auto output_tensors = session.Run(runOptions, inNames.data(), &inputTensors, 1UZ, outNames.data(), 1UZ);

    const float*       outData = output_tensors[0].GetTensorData<float>();
    std::vector<float> predPositions(outData, outData + numPeaks);
    std::vector<float> predFWHMs(outData + numPeaks, outData + 2 * numPeaks);

    // sort predictions by position
    std::vector<std::size_t> idxs(numPeaks);
    std::iota(idxs.begin(), idxs.end(), 0UZ);
    std::ranges::sort(idxs, [&](size_t a, size_t b) { return predPositions[a] < predPositions[b]; });

    std::vector<float> sorted_pred_positions(numPeaks), sorted_pred_fwhms(numPeaks);
    for (std::size_t i = 0UZ; i < numPeaks; ++i) {
        sorted_pred_positions[i] = predPositions[idxs[i]];
        sorted_pred_fwhms[i]     = predFWHMs[idxs[i]];
    }

    // print results
    std::print("True Peak Positions:      {}\n", truePositions);
    std::print("True Peak FWHMs:          {}\n", trueFWHMs);
    std::print("Predicted Peak Positions: {}\n", sorted_pred_positions);
    std::print("Predicted Peak FWHMs:     {}\n", sorted_pred_fwhms);

    return 0;
}
