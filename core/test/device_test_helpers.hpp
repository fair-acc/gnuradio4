#ifndef GNURADIO_DEVICE_TEST_HELPERS_HPP
#define GNURADIO_DEVICE_TEST_HELPERS_HPP

#include <cstddef>
#include <gnuradio-4.0/Complex.hpp>

namespace gr::test {

// device kernel code lives in device_test_helpers.cpp, separate from Boost.UT
// suite registration, to avoid AdaptiveCpp SSCP interference with global constructors.

void deviceParallelMultiply(const float* in, float* out, std::size_t N, float factor);
void deviceParallelComplexRotate(const gr::complex<float>* in, gr::complex<float>* out, std::size_t N, gr::complex<float> factor);

// GL compute shader test: compile + dispatch a multiply shader
bool glComputeAvailable();
void glShaderMultiply(const float* in, float* out, std::size_t N, float factor);

} // namespace gr::test

#endif // GNURADIO_DEVICE_TEST_HELPERS_HPP
