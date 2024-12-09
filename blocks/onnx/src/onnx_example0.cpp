#include <gnuradio-4.0/meta/formatter.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>

// Generate synthetic signal
static std::tuple<std::vector<float>, std::vector<int>, std::vector<float>>
generate_signal(int num_peaks=5, int signal_length=1024, float noise_level=0.05, float asymmetry_factor=0.2) {
    std::vector<float> x(signal_length);
    for (int i = 0; i < signal_length; ++i) x[i] = (float)i;

    std::vector<float> signal(signal_length, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> pos_dist(50, signal_length-50);
    std::uniform_real_distribution<float> fwhm_dist(10.0f,50.0f);
    std::uniform_real_distribution<float> amp_dist(0.5f,2.0f);
    std::uniform_real_distribution<float> asym_dist(-asymmetry_factor, asymmetry_factor);

    std::vector<int> peak_positions(num_peaks);
    std::vector<float> fwhms(num_peaks);
    std::vector<float> amps(num_peaks);

    for (int i=0; i<num_peaks; ++i) {
        peak_positions[i] = pos_dist(gen);
        fwhms[i] = fwhm_dist(gen);
        amps[i] = amp_dist(gen);
    }

    for (int i=0; i<num_peaks; ++i) {
        float stddev = fwhms[i] / (2 * std::sqrt(2 * std::log(2)));
        float asym = asym_dist(gen);
        for (int idx=0; idx<signal_length; ++idx) {
            float val = amps[i] * std::exp(-((x[idx]-peak_positions[i])*(x[idx]-peak_positions[i])) / (2*stddev*stddev));
            val *= (1.0f + asym * (x[idx]-peak_positions[i]) / fwhms[i]);
            if (val < 0.0f) val = 0.0f;
            signal[idx] += val;
        }
    }

    // Add noise
    std::normal_distribution<float> noise_dist(0.0f, noise_level * (*std::max_element(signal.begin(), signal.end())));
    for (auto &val : signal) {
        val += noise_dist(gen);
    }

    // Sort peaks by position
    std::vector<std::size_t> idxs(num_peaks);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b){
        return peak_positions[a] < peak_positions[b];
    });

    std::vector<int> sorted_pos(num_peaks);
    std::vector<float> sorted_fwhm(num_peaks);
    for (int i=0; i<num_peaks; ++i) {
        sorted_pos[i] = peak_positions[idxs[i]];
        sorted_fwhm[i] = fwhms[idxs[i]];
    }

    return {signal, sorted_pos, sorted_fwhm};
}

int main() {
    using namespace Ort;

    // Initialize ONNX Runtime
    Env env(ORT_LOGGING_LEVEL_WARNING, "PeakDetector");
    SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load the ONNX model
    Session session(env, "peak_detector.onnx", session_options);

    auto input_count = session.GetInputCount();
    auto output_count = session.GetOutputCount();

    char* input_name = session.GetInputNameAllocated(0, AllocatorWithDefaultOptions());
    char* output_name = session.GetOutputNameAllocated(0, AllocatorWithDefaultOptions());

    // Generate test signal
    int signal_length = 1024;
    int num_peaks = 5;
    auto [signal, true_positions, true_fwhms] = generate_signal(num_peaks, signal_length);

    // Prepare input tensor
    std::vector<int64_t> input_shape{1, signal_length, 1};
    std::vector<float> input_data = signal; // Already have data in 'signal'
    Value input_tensor = Value::CreateTensor<float>(AllocatorWithDefaultOptions(), input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    auto& output_tensor = output_tensors.front();

    float* out_data = output_tensor.GetTensorMutableData<float>();
    std::vector<float> pred_positions(out_data, out_data + num_peaks);
    std::vector<float> pred_fwhms(out_data + num_peaks, out_data + 2*num_peaks);

    // Sort predictions by position
    std::vector<std::size_t> idxs(num_peaks);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b){
        return pred_positions[a] < pred_positions[b];
    });

    std::vector<float> sorted_pred_positions(num_peaks), sorted_pred_fwhms(num_peaks);
    for (int i=0; i<num_peaks; ++i) {
        sorted_pred_positions[i] = pred_positions[idxs[i]];
        sorted_pred_fwhms[i] = pred_fwhms[idxs[i]];
    }

    // Print results using fmt
    fmt::print("True Peak Positions:    {}\n", true_positions);
    fmt::print("True Peak FWHMs:        {}\n", true_fwhms);
    fmt::print("Predicted Peak Positions: {}\n", sorted_pred_positions);
    fmt::print("Predicted Peak FWHMs:     {}\n", sorted_pred_fwhms);

    AllocatorFree(AllocatorWithDefaultOptions(), input_name);
    AllocatorFree(AllocatorWithDefaultOptions(), output_name);

    return 0;
}
