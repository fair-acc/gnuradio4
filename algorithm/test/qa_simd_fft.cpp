#include <gnuradio-4.0/algorithm/fourier/simd_fft.hpp>

#include <c++/15/numbers>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

int main() {
    using T            = float;
    constexpr size_t N = 1024;

    // Create FFT object
    simd_fft::SimdFFT_float fft(N);

    // Complex transform example
    {
        std::vector<std::complex<float>> data(N);
        for (std::size_t i = 0; i < N; ++i) {
            T t     = static_cast<T>(i) / static_cast<T>(N);
            data[i] = std::complex<float>(std::cos(2.0f * std::numbers::pi_v<T> * t), std::sin(2.0f * std::numbers::pi_v<T> * t));
        }

        // Forward transform
        fft.transform_complex(data, simd_fft::SimdFFT_float::Direction::Forward);

        // Process spectrum...

        // Backward transform
        fft.transform_complex(data, simd_fft::SimdFFT_float::Direction::Backward);
    }

    // Real transform example
    {
        std::vector<float>               real_data(N);
        std::vector<std::complex<float>> freq_data(N / 2 + 1);

        // Initialize real data...
        for (size_t i = 0; i < N; ++i) {
            T t     = static_cast<T>(i) / static_cast<T>(N);
            real_data[i] = std::cos(2.0f * std::numbers::pi_v<T> * 440.0f * t);
        }

        // Real-to-complex FFT
        fft.transform_real(real_data, freq_data, simd_fft::SimdFFT_float::Direction::Forward);

        // Process spectrum...

        // Complex-to-real IFFT
        fft.transform_real(real_data, freq_data, simd_fft::SimdFFT_float::Direction::Backward);
    }

    return 0;
}