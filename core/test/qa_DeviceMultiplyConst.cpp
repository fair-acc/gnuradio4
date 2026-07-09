#include <boost/ut.hpp>

#include <numeric>
#include <vector>

#include "device_test_helpers.hpp"

using namespace boost::ut;

const suite<"device::MultiplyConst CPU auto-parallel"> cpuTests =
    [] {
        "produces correct output"_test = [] {
            constexpr std::size_t N = 4096;
            std::vector<float>    input(N);
            std::iota(input.begin(), input.end(), 0.f);
            std::vector<float> output(N, 0.f);

            gr::test::deviceParallelMultiply(input.data(), output.data(), N, 2.5f);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(output[i], static_cast<float>(i) * 2.5f));
            }
        };

        "settings change reflected"_test = [] {
            constexpr std::size_t N = 256;
            std::vector<float>    input(N);
            std::iota(input.begin(), input.end(), 1.f);
            std::vector<float> output(N, 0.f);

            gr::test::deviceParallelMultiply(input.data(), output.data(), N, 3.f);
            expect(eq(output[0], 3.f));
            expect(eq(output[99], 300.f));

            gr::test::deviceParallelMultiply(input.data(), output.data(), N, 0.5f);
            expect(eq(output[0], 0.5f));
            expect(eq(output[99], 50.f));
        };

        "matches scalar CPU path"_test = [] {
            constexpr std::size_t N      = 1024;
            constexpr float       factor = 7.f;

            std::vector<float> input(N);
            std::iota(input.begin(), input.end(), 0.f);

            std::vector<float> cpuOutput(N);
            for (std::size_t i = 0; i < N; ++i) {
                cpuOutput[i] = input[i] * factor;
            }

            std::vector<float> deviceOutput(N, 0.f);
            gr::test::deviceParallelMultiply(input.data(), deviceOutput.data(), N, factor);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(deviceOutput[i], cpuOutput[i]));
            }
        };

        "gr::complex rotation by i"_test = [] {
            constexpr std::size_t           N = 128;
            std::vector<gr::complex<float>> input(N);
            for (std::size_t i = 0; i < N; ++i) {
                input[i] = {static_cast<float>(i), 0.f};
            }

            std::vector<gr::complex<float>> output(N);
            gr::test::deviceParallelComplexRotate(input.data(), output.data(), N, {0.f, 1.f});

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(output[i].re, 0.f));
                expect(eq(output[i].im, static_cast<float>(i)));
            }
        };
    };

const suite<"device::MultiplyConst GLSL shader"> glslTests = [] {
    "shader multiply produces correct output"_test = [] {
            if (!gr::test::glComputeAvailable()) {
                return;
            }

            constexpr std::size_t N = 512;
            std::vector<float>    input(N);
            std::iota(input.begin(), input.end(), 0.f);
            std::vector<float> output(N, 0.f);

            gr::test::glShaderMultiply(input.data(), output.data(), N, 3.f);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(output[i], static_cast<float>(i) * 3.f)) << "at index " << i;
            }
        };

        "shader matches CPU path"_test = [] {
            if (!gr::test::glComputeAvailable()) {
                return;
            }

            constexpr std::size_t N      = 256;
            constexpr float       factor = 0.5f;

            std::vector<float> input(N);
            std::iota(input.begin(), input.end(), 1.f);

            std::vector<float> cpuOutput(N);
            for (std::size_t i = 0; i < N; ++i) {
                cpuOutput[i] = input[i] * factor;
            }

            std::vector<float> shaderOutput(N, 0.f);
            gr::test::glShaderMultiply(input.data(), shaderOutput.data(), N, factor);

            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(shaderOutput[i], cpuOutput[i])) << "at index " << i;
            }
    };
};

int main() { /* not needed for UT */ }
