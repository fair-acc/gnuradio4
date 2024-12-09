#ifndef GR_ONNX_UTILS_HPP
#define GR_ONNX_UTILS_HPP

// OnnxUtils.hpp — backward-compatibility header.
// All functionality has been moved to OnnxPreprocess.hpp and PeakExtraction.hpp.
#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>
#include <gnuradio-4.0/onnx/PeakExtraction.hpp>

namespace gr::blocks::onnx {

// Backward-compatible free functions delegating to OnnxPreprocess<float>
inline void normalise(std::span<const float> raw, std::span<float> out) { OnnxPreprocess<float>::normaliseLogMAD(raw, out); }

inline void resample(std::span<const float> input, std::span<float> output) { OnnxPreprocess<float>::resample(input, output); }

} // namespace gr::blocks::onnx

#endif // GR_ONNX_UTILS_HPP
