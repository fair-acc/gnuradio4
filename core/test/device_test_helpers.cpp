#include "device_test_helpers.hpp"
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::test {

void deviceParallelMultiply(const float* hostIn, float* hostOut, std::size_t N, float factor) {
    gr::device::DeviceContextCpu ctx;
    auto                         dIn  = ctx.allocateShared<float>(N);
    auto                         dOut = ctx.allocateShared<float>(N);
    ctx.copyHostToDevice(hostIn, dIn, N);
    float* pIn  = dIn.devicePointer<float>();
    float* pOut = dOut.devicePointer<float>();
    for (std::size_t i = 0; i < N; ++i) {
        pOut[i] = pIn[i] * factor;
    }
    ctx.copyDeviceToHost(dOut, hostOut, N);
    ctx.deallocate(dIn);
    ctx.deallocate(dOut);
}

void deviceParallelComplexRotate(const gr::complex<float>* hostIn, gr::complex<float>* hostOut, std::size_t N, gr::complex<float> factor) {
    gr::device::DeviceContextCpu ctx;
    auto                         dIn  = ctx.allocateShared<gr::complex<float>>(N);
    auto                         dOut = ctx.allocateShared<gr::complex<float>>(N);
    ctx.copyHostToDevice(hostIn, dIn, N);
    gr::complex<float>* pIn  = dIn.devicePointer<gr::complex<float>>();
    gr::complex<float>* pOut = dOut.devicePointer<gr::complex<float>>();
    for (std::size_t i = 0; i < N; ++i) {
        pOut[i] = pIn[i] * factor;
    }
    ctx.copyDeviceToHost(dOut, hostOut, N);
    ctx.deallocate(dIn);
    ctx.deallocate(dOut);
}

} // namespace gr::test
