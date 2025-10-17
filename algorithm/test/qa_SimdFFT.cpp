#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>

template<gr::algorithm::Transform transform_, gr::algorithm::Order ordering_, std::size_t N>
struct Flags {
    constexpr static gr::algorithm::Transform transform = transform_;
    constexpr static gr::algorithm::Order     ordering  = ordering_;
    constexpr static std::size_t              size      = N;
};

template<std::floating_point T>
auto generate_chirp(std::size_t N, T f_start, T f_end) {
    std::vector<T> signal(N);
    for (std::size_t n = 0; n < N; ++n) {
        T t       = T(n) / T(N);
        T freq    = f_start + (f_end - f_start) * t;
        signal[n] = std::sin(T{2} * std::numbers::pi_v<T> * freq * T(n));
    }
    return signal;
}

boost::ut::suite<"SimdFFT Comprehensive"> _ = [] {
    using namespace boost::ut;
    using namespace gr::algorithm;

    "SimdFFT<T> tests"_test = []<typename T> {
        "invalid transform size detection"_test = [](std::size_t invalidN) {
            expect(!SimdFFT<T, Transform::Real>::canProcessSize(invalidN, Order::Ordered));
            expect(!SimdFFT<T, Transform::Complex>::canProcessSize(invalidN, Order::Ordered) || invalidN == 48); // 48 is OK for complex

            // verify exception thrown
            expect(throws<gr::exception>([&invalidN] {
                SimdFFT<T, Transform::Real>               setup(invalidN);
                std::vector<T, gr::allocator::Aligned<T>> in(invalidN);
                std::vector<T, gr::allocator::Aligned<T>> out(invalidN);
                setup.transform(forward, ordered, in, out);
            })) << std::format("N={} should throw", invalidN);
        } | std::array{7UZ, 11UZ, 14UZ, 17UZ, 31UZ, 35UZ, 48UZ, 49UZ}; // primes and non-factorable

        using enum Transform;
        using enum Order;
        "sine-wave detection"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;
                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} sine-wave FFT test", N, Flag::transform, Flag::ordering)) = [] {};
                    return; // skip
                }

                constexpr auto valid_frequencies = []<std::size_t N>() {
                    std::array<T, 4> freqs{};
                    std::size_t      idx = 0;
                    // choose k/N where k creates integer periods
                    for (auto k : {N / 8, N / 4, 3 * N / 8, N / 2 - 1}) {
                        if (k > 0 && k < N / 2) {
                            freqs[idx++] = static_cast<T>(k) / T(N);
                        }
                    }
                    return std::pair{freqs, idx};
                };

                constexpr bool                                       is_real = (Flag::transform == Real);
                auto [freqs, num_freqs]                                      = valid_frequencies.template operator()<N>();

                for (std::size_t f_idx = 0; f_idx < num_freqs; ++f_idx) {
                    T           freq         = T(freqs[f_idx]);
                    std::size_t expected_bin = static_cast<std::size_t>(freq * N);

                    std::vector<T, gr::allocator::Aligned<T>> input(is_real ? N : 2 * N);
                    std::vector<T, gr::allocator::Aligned<T>> output(is_real ? N : 2 * N);

                    // generate sine wave
                    for (std::size_t n = 0; n < N; ++n) {
                        T phase = T{2} * std::numbers::pi_v<T> * freq * T(n);
                        if constexpr (is_real) {
                            input[n] = std::sin(phase);
                        } else {
                            input[2 * n]     = std::cos(phase);
                            input[2 * n + 1] = std::sin(phase);
                        }
                    }

                    SimdFFT<T, Flag::transform> setup(N);
                    setup.template transform<Direction::Forward, Flag::ordering>(input, output);

                    // compute magnitudes
                    const std::size_t num_bins = is_real ? (N / 2 + 1) : N;
                    std::vector<T>    magnitudes(num_bins);

                    if constexpr (is_real) {
                        magnitudes[0]     = std::abs(output[0]) / T(N);
                        magnitudes[N / 2] = std::abs(output[1]) / T(N);
                        for (std::size_t k = 1; k < N / 2; ++k) {
                            magnitudes[k] = std::hypot(output[2 * k], output[2 * k + 1]) / T(N);
                        }
                    } else {
                        for (std::size_t k = 0; k < N; ++k) {
                            magnitudes[k] = std::hypot(output[2 * k], output[2 * k + 1]) / T(N);
                        }
                    }

                    const auto        peak_it  = std::max_element(magnitudes.begin(), magnitudes.end());
                    const std::size_t peak_bin = static_cast<std::size_t>(std::distance(magnitudes.begin(), peak_it));
                    expect(eq(peak_bin, expected_bin)) << std::format("N={}, f={:.3f}, bin {}, expected {}", N, freq, peak_bin, expected_bin);
                }
            } |
            std::tuple{Flags<Real, Ordered, 32UZ>{}, Flags<Real, Ordered, 48UZ>{} /* should be impossible for real-valued FFT */, Flags<Real, Ordered, 64UZ>{}, Flags<Real, Ordered, 128UZ>{}, // real valued tests
                Flags<Real, Ordered, 160UZ>{}, Flags<Real, Ordered, 512UZ>{}, Flags<Real, Ordered, 1024UZ>{}, Flags<Complex, Ordered, 16UZ>{},                                                 //
                Flags<Complex, Ordered, 32UZ>{}, Flags<Complex, Ordered, 48UZ>{}, Flags<Complex, Ordered, 64UZ>{}, Flags<Complex, Ordered, 128UZ>{},                                           //
                Flags<Complex, Ordered, 140UZ>{}, Flags<Complex, Ordered, 1024UZ>{}};                                                                                                          // complex tests

        "sine round-trip identity"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;
                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} sine round-trip identity FFT test", N, Flag::transform, Flag::ordering)) = [] {};
                    return; // skip
                }

                const std::size_t nSamples  = Flag::transform == Real ? N : 2 * N;
                const T           tolerance = T(1e-5f) * T(N);

                std::vector<T, gr::allocator::Aligned<T>> input(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> spectrum(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> reconstructed(nSamples);

                // multi-tone signal
                for (std::size_t i = 0; i < N; ++i) {
                    T val = std::sin(T{2} * std::numbers::pi_v<T> * T(i) / T(8)) + T{0.5} * std::cos(T{2} * std::numbers::pi_v<T> * T(i) / T(4));
                    if constexpr (Flag::transform == Real) {
                        input[i] = val;
                    } else {
                        input[2 * i]     = val;
                        input[2 * i + 1] = std::sin(T{2} * std::numbers::pi_v<T> * T(i) / T(6));
                    }
                }

                SimdFFT<T, Flag::transform> setup(N);
                setup.template transform<Direction::Forward, Flag::ordering>(input, spectrum);
                T spectrum_energy = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    spectrum_energy += std::abs(spectrum[i]);
                }

                std::string spectrumE = std::format("spectrum energy: {}, first few values: {} {} {} {}", //
                    spectrum_energy, spectrum[0], spectrum[1], spectrum[2], spectrum[3]);
                setup.template transform<Direction::Backward, Flag::ordering>(spectrum, reconstructed);

                for (auto& val : reconstructed) {
                    val /= N;
                }

                T max_error = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    max_error = std::max(max_error, std::abs(input[i] - reconstructed[i]));
                    expect(approx(input[i], reconstructed[i], tolerance)) << std::format("index: {}, N={}, {}, {}, error={} - energy={}", i, N, Flag::transform, Flag::ordering, max_error, spectrumE);
                    if (std::abs(input[i] - reconstructed[i]) > tolerance) {
                        return;
                    }
                }

                expect(le(max_error, tolerance)) << std::format("N={}, error={} - energy={}", N, max_error, spectrumE);
            } |
            std::tuple{
                Flags<Real, Ordered, 32UZ>{}, Flags<Real, Ordered, 64UZ>{}, Flags<Real, Ordered, 128UZ>{}, Flags<Real, Ordered, 256UZ>{}, Flags<Real, Ordered, 512UZ>{}, Flags<Real, Ordered, 1024UZ>{}, Flags<Real, Ordered, 2048UZ>{},                                                                                   // real-valued, ordered
                Flags<Real, Unordered, 32UZ>{}, Flags<Real, Unordered, 48UZ>{}, Flags<Real, Unordered, 64UZ>{}, Flags<Real, Unordered, 128UZ>{}, Flags<Real, Unordered, 256UZ>{}, Flags<Real, Unordered, 512UZ>{}, Flags<Real, Unordered, 1024UZ>{}, Flags<Real, Unordered, 2048UZ>{},                                     // real-valued, unordered
                Flags<Complex, Ordered, 32UZ>{}, Flags<Complex, Ordered, 64UZ>{}, Flags<Complex, Ordered, 48UZ>{}, Flags<Complex, Ordered, 128UZ>{}, Flags<Complex, Ordered, 1024UZ>{}, Flags<Complex, Ordered, 2048UZ>{},                                                                                                 // complex ordered
                Flags<Complex, Unordered, 32UZ>{}, Flags<Complex, Unordered, 64UZ>{}, Flags<Complex, Unordered, 48UZ>{}, Flags<Complex, Unordered, 128UZ>{}, Flags<Complex, Unordered, 1024UZ>{}, Flags<Complex, Unordered, 2048UZ>{},                                                                                     // complex unordered
                Flags<Real, Unordered, 48UZ>{} /* 2^4 × 3 */, Flags<Real, Unordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Real, Unordered, 80UZ>{} /* 2^4 × 5 */, Flags<Real, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Real, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Real, Unordered, 160UZ>{}, /* 2^5 × 5 */                   // real-valued, radix 3 & 6
                Flags<Complex, Unordered, 48UZ>{} /* 2^4 × 3 */, Flags<Complex, Unordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Complex, Unordered, 80UZ>{} /* 2^4 × 5 */, Flags<Complex, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Unordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Unordered, 160UZ>{}, /* 2^5 × 5 */ // complex unordered, radix 3 & 6
                Flags<Complex, Ordered, 48UZ>{} /* 2^4 × 3 */, Flags<Complex, Ordered, 60UZ>{} /* 2^4 × 3 * 5 */, Flags<Complex, Ordered, 80UZ>{} /* 2^4 × 5 */, Flags<Complex, Ordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Ordered, 96UZ>{} /* 2^5 × 3 */, Flags<Complex, Ordered, 160UZ>{} /* 2^5 × 5 */              // complex ordered, radix 3 & 6
            };

        // , 192UZ /* 2^6 × 3 */, 240UZ /* 2^4 × 3 × 5 */, 256UZ /* 2^8 */, 320UZ /* 2^6 × 5 */,     //
        // 384UZ /* 2^7 × 3 */, 480UZ /* 2^5 × 3 × 5 */, 512UZ /* 2^9 */, 1024UZ /* 2^10 */, 2048UZ /* 2^11 */          //

        "chirp round-trip identity"_test =
            []<typename Args>() {
                constexpr std::size_t N = std::remove_reference_t<Args>::size;
                using Flag              = std::remove_reference_t<Args>;

                if (!SimdFFT<T, Flag::transform>::canProcessSize(N, Flag::ordering)) {
                    skip / test(std::format("unsupported - N={} {} {} FFT chirp round-trip identity test", N, Flag::transform, Flag::ordering)) = [] {};
                    return;
                }

                const std::size_t nSamples  = Flag::transform == Real ? N : 2 * N;
                const T           tolerance = T(1e-5f) * N;

                std::vector<T, gr::allocator::Aligned<T>> input(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> spectrum(nSamples);
                std::vector<T, gr::allocator::Aligned<T>> reconstructed(nSamples);

                // chirp from f=0.05 to f=0.45
                auto chirp = generate_chirp<T>(N, T(0.05f), T(0.45f));

                for (std::size_t n = 0; n < N; ++n) {
                    if constexpr (Flag::transform == Real) {
                        input[n] = chirp[n];
                    } else {
                        input[2 * n]     = chirp[n];
                        input[2 * n + 1] = generate_chirp<T>(N, T(0.1f), T(0.4))[n];
                    }
                }

                SimdFFT<T, Flag::transform> setup(N);
                setup.template transform<Direction::Forward, Flag::ordering>(input, spectrum);
                setup.template transform<Direction::Backward, Flag::ordering>(spectrum, reconstructed);

                for (auto& val : reconstructed) {
                    val /= N;
                }

                T max_error = T{0};
                for (std::size_t i = 0; i < nSamples; ++i) {
                    max_error = std::max(max_error, std::abs(input[i] - reconstructed[i]));
                }

                expect(max_error < tolerance) << std::format("chirp N={}, error={}", N, max_error);
            } |
            std::tuple{Flags<Real, Ordered, 32>{}, Flags<Real, Ordered, 64>{}, Flags<Real, Unordered, 32>{}, Flags<Real, Unordered, 64>{},           //
                Flags<Complex, Unordered, 32>{}, Flags<Complex, Unordered, 64>{}, Flags<Complex, Unordered, 96>{}, Flags<Complex, Unordered, 512>{}, // complex unordered
                Flags<Complex, Ordered, 32>{}, Flags<Complex, Ordered, 64>{}, Flags<Complex, Unordered, 96>{}, Flags<Complex, Ordered, 512>{}};      // complex ordered

        "radix"_test = [](Transform transform) {
            constexpr static std::size_t L = vec<T, 4>::size();

            "radix"_test = [&transform]<typename Args>() {
                constexpr std::size_t Radix = 2UZ;
                test(std::format("{} radix-{}", transform, Radix)) =
                    [&transform, &Radix](std::size_t N) {
                        if (N % Radix != 0) {
                            return; // skip if N is not divisible by radix
                        }

                        // Calculate parameters based on radix and transform type
                        const std::size_t stride = [&]() {
                            if (transform == Complex) {
                                return Radix;
                            }
                            // Real transforms: stride is 2*radix (Hermitian symmetry)
                            return 2 * Radix;
                        }();

                        const std::size_t nGroups = N / stride;

                        // Radix-r butterfly: needs (radix+1)×logical_size for safety margin
                        const std::size_t bufSize = (Radix + 1) * nGroups * stride * L;

                        std::vector<T, gr::allocator::Aligned<T>> input(bufSize);
                        std::iota(input.begin(), input.end(), T(0));

                        // Twiddle factors: (radix-1) complex values = 2*(radix-1) floats
                        std::vector<T> twiddles(2 * (Radix - 1));
                        for (std::size_t i = 0; i < Radix - 1; ++i) {
                            const T angle       = -2 * std::numbers::pi_v<T> * T(i + 1) / T(Radix);
                            twiddles[2 * i]     = std::cos(angle);
                            twiddles[2 * i + 1] = std::sin(angle);
                        }

                        std::vector<T, gr::allocator::Aligned<T>> fwdOutput(bufSize);
                        std::vector<T, gr::allocator::Aligned<T>> bwdOutput(bufSize);

                        if constexpr (Radix == 2UZ) {
                            if (transform == Complex) {
                                details::complexRadix2<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                details::realRadix2<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 3UZ) {
                            if (transform == Complex) {
                                details::complexRadix3<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                details::realRadix3<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 4UZ) {
                            if (transform == Complex) {
                                details::complexRadix4<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                details::realRadix4<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 5UZ) {
                            if (transform == Complex) {
                                details::complexRadix5<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            } else {
                                details::realRadix5<Direction::Forward, T>(stride, nGroups, input, fwdOutput, twiddles);
                            }
                        }

                        expect(std::all_of(fwdOutput.begin(), fwdOutput.end(), [](T x) { return std::isfinite(x); })) << std::format("forward radix-{}: invalid values detected", Radix);

                        if constexpr (Radix == 2) {
                            if (transform == Complex) {
                                details::complexRadix2<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                details::realRadix2<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 3) {
                            if (transform == Complex) {
                                details::complexRadix3<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                details::realRadix3<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 4) {
                            if (transform == Complex) {
                                details::complexRadix4<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                details::realRadix4<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        } else if constexpr (Radix == 5) {
                            if (transform == Complex) {
                                details::complexRadix5<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            } else {
                                details::realRadix5<Direction::Backward, T>(stride, nGroups, fwdOutput, bwdOutput, twiddles);
                            }
                        }

                        expect(std::all_of(bwdOutput.begin(), bwdOutput.end(), [](T x) { return std::isfinite(x); })) << std::format("backward radix-{}: invalid values detected", Radix);
                    } |
                    std::array{
                        32UZ /* 2^5 */, 48UZ /* 2^4 × 3 */, 64UZ /* 2^6 */, 80UZ /* 2^4 × 5 */, 96UZ /* 2^5 × 3 */, 128UZ /* 2^7 */, //
                        160UZ /* 2^5 × 5 */, 192UZ /* 2^6 × 3 */, 240UZ /* 2^4 × 3 × 5 */, 256UZ /* 2^8 */, 320UZ /* 2^6 × 5 */,     //
                        384UZ /* 2^7 × 3 */, 480UZ /* 2^5 × 3 × 5 */, 512UZ /* 2^9 */, 1024UZ /* 2^10 */, 2048UZ /* 2^11 */          //
                    };
            } | std::tuple{std::integral_constant<std::size_t, 2UZ>{}, std::integral_constant<std::size_t, 4UZ>{}, std::integral_constant<std::size_t, 3UZ>{}, std::integral_constant<std::size_t, 5UZ>{}};
        } | std::array{Complex, Real};

        /*
        skip / "hermitian symmetry for real signals"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N);
            std::vector<T, gr::allocator::Aligned<T>> output(N);

            std::mt19937                      rng(42);
            std::uniform_real_distribution<T> dist(T{-1}, T{1});
            for (auto& val : input) {
                val = dist(rng);
            }

            PFFFT_Setup<T, Real> setup(N);
            pffft_transform<Direction::Forward, Ordered>(setup, input, output);

            // check X[k] = conj(X[N-k])
            const T tolerance = T(1e-5f);
            for (std::size_t k = 1; k < N / 2; ++k) {
                T re_k  = output[2 * k];
                T im_k  = output[2 * k + 1];
                T re_nk = output[2 * (N - k)];
                T im_nk = output[2 * (N - k) + 1];

                expect(std::abs(re_k - re_nk) < tolerance) << std::format("Re symmetry at k={}", k);
                expect(std::abs(im_k + im_nk) < tolerance) << std::format("Im symmetry at k={}", k);
            }
        } | std::array{32UZ, 128UZ, 1024UZ};
        */

        "DC edge cases"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N, T(1.5f)); // constant DC
            std::vector<T, gr::allocator::Aligned<T>> output(N);

            SimdFFT<T, Real> setup(N);
            setup.template transform<Direction::Forward, Ordered>(input, output);

            // all energy should be at DC
            expect(le(std::abs(output[0] - T(1.5) * T(N)), T(1e-4f))) << "DC component";

            for (std::size_t k = 1; k < N / 2; ++k) {
                const T mag = std::hypot(output[2 * k], output[2 * k + 1]);
                expect(le(mag, T(1e-4))) << std::format("bin {} should be zero", k);
            }
        } | std::array{32UZ, 64UZ, 256UZ};

        "Nyquist signal edge case"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> input(N);
            for (std::size_t n = 0; n < N; ++n) {
                input[n] = (n % 2 == 0) ? T{1} : T{-1}; // alternating +/- at Nyquist
            }

            std::vector<T, gr::allocator::Aligned<T>> output(N);
            SimdFFT<T, Real>                          setup(N);
            setup.template transform<Direction::Forward, Ordered>(input, output);

            // all energy at Nyquist (bin N/2)
            expect(gt(std::abs(output[1]), T(0.9) * T(N))) << "Nyquist magnitude";
        } | std::array{32UZ, 64UZ, 128UZ, 256UZ};

        "linearity"_test = [](std::size_t N) {
            std::vector<T, gr::allocator::Aligned<T>> x(2 * N), y(2 * N);
            std::vector<T, gr::allocator::Aligned<T>> fx(2 * N), fy(2 * N), fsum(2 * N);
            std::vector<T, gr::allocator::Aligned<T>> sum(2 * N);

            std::mt19937                      rng(123);
            std::uniform_real_distribution<T> dist(T{-1}, T{1});

            for (std::size_t i = 0; i < 2 * N; ++i) {
                x[i]   = dist(rng);
                y[i]   = dist(rng);
                sum[i] = x[i] + y[i];
            }

            SimdFFT<T, Complex> setup(N);
            setup.transform(forward, ordered, x, fx);
            setup.transform(forward, ordered, y, fy);
            setup.transform(forward, ordered, sum, fsum);

            // check FFT(x+y) = FFT(x) + FFT(y)
            T max_error = T{0};
            for (std::size_t i = 0UZ; i < 2UZ * N; ++i) {
                max_error = std::max(max_error, std::abs(fsum[i] - (fx[i] + fy[i])));
            }

            expect(lt(max_error, T(1e-4) * T(N))) << std::format("linearity error={}", max_error);
        } | std::array{32UZ, 64UZ, 128UZ, 256UZ};
    } | std::tuple{float{} /*, double{}*/};
};

int main() { /* tests are auto-registered */ }
