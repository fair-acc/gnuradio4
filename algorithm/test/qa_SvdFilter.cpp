#include <boost/ut.hpp>

#include <cmath>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/filter/SvdFilter.hpp>

const boost::ut::suite<"SVD filter algorithm"> svdFilterTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::svd_filter;

    "low-rank approximation"_test = [] {
        "rank-1 matrix"_test = [] {
            gr::Tensor<double> A({4, 4});
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                for (std::size_t j = 0UZ; j < 4UZ; ++j) {
                    A[i, j] = static_cast<double>(i + 1) * static_cast<double>(j + 1);
                }
            }

            auto   A_approx = lowRankApproximation(A, Config<double>{.maxRank = 1UZ});
            double maxError = 0.0;
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                for (std::size_t j = 0UZ; j < 4UZ; ++j) {
                    maxError = std::max(maxError, std::abs(A[i, j] - A_approx[i, j]));
                }
            }
            expect(lt(maxError, 1e-10)) << "rank-1 approximation error: " << maxError;
        };

        "auto rank detection"_test = [] {
            gr::Tensor<double> A({5, 5});
            A.fill(0.0);
            for (std::size_t i = 0UZ; i < 5UZ; ++i) {
                for (std::size_t j = 0UZ; j < 5UZ; ++j) {
                    A[i, j] = static_cast<double>(i + 1) * static_cast<double>(j + 1) + 0.1 * static_cast<double>(i) * static_cast<double>(4 - j);
                }
            }

            auto A_approx = lowRankApproximation(A, Config<double>{.relativeThreshold = 0.01});
            expect(A_approx.extent(0) == A.extent(0));
            expect(A_approx.extent(1) == A.extent(1));
        };
    };

    "signal denoising"_test = [] {
        constexpr std::size_t    N  = 128UZ;
        constexpr double         fs = 1000.0;
        constexpr Config<double> config{.maxRank = 4UZ, .energyFraction = 0.95};

        "clean sinusoid preservation"_test = [&] {
            std::vector<double> clean(N);
            for (std::size_t i = 0UZ; i < N; ++i) {
                clean[i] = std::sin(2.0 * std::numbers::pi_v<double> * 50.0 * static_cast<double>(i) / fs);
            }

            auto   denoised = denoiseWindow<double>(clean, N / 2UZ, config);
            double maxError = 0.0;
            for (std::size_t i = 0UZ; i < N; ++i) {
                maxError = std::max(maxError, std::abs(clean[i] - denoised[i]));
            }
            expect(lt(maxError, 0.1)) << "clean signal distortion: " << maxError;
        };

        "noisy sinusoid denoising"_test = [&] {
            std::mt19937                     gen(42);
            std::normal_distribution<double> noise(0.0, 0.3);

            std::vector<double> noisy(N), clean(N);
            for (std::size_t i = 0UZ; i < N; ++i) {
                clean[i] = std::sin(2.0 * std::numbers::pi_v<double> * 50.0 * static_cast<double>(i) / fs);
                noisy[i] = clean[i] + noise(gen);
            }

            auto   denoised      = denoiseWindow<double>(noisy, N / 2UZ, config);
            double noisyError    = 0.0;
            double denoisedError = 0.0;
            for (std::size_t i = 0UZ; i < N; ++i) {
                noisyError += (noisy[i] - clean[i]) * (noisy[i] - clean[i]);
                denoisedError += (denoised[i] - clean[i]) * (denoised[i] - clean[i]);
            }
            noisyError    = std::sqrt(noisyError / static_cast<double>(N));
            denoisedError = std::sqrt(denoisedError / static_cast<double>(N));

            expect(lt(denoisedError, noisyError)) << "denoising should reduce error";
        };
    };

    "streaming denoiser BoundaryPolicy"_test = [] {
        "Default policy pre-fills with defaultValue"_test = [] {
            constexpr Config<double> config{.maxRank = 2UZ, .boundaryPolicy = BoundaryPolicy::Default, .defaultValue = 0.0};
            SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

            // Process a sinusoid - the filter should produce valid output from first sample
            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                double input = std::sin(2.0 * std::numbers::pi_v<double> * 0.1 * static_cast<double>(i));
                outputs.push_back(denoiser.processOne(input));
            }
            expect(eq(outputs.size(), 50UZ)) << "should produce output for all inputs";
            expect(std::isfinite(outputs.back())) << "output should be finite";
        };

        "ZeroOrderHold policy fills lazily"_test = [] {
            constexpr Config<double> config{.maxRank = 2UZ, .boundaryPolicy = BoundaryPolicy::ZeroOrderHold};
            SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

            // Process a sinusoid - first sample fills buffer, then denoising starts
            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                double input = std::sin(2.0 * std::numbers::pi_v<double> * 0.1 * static_cast<double>(i));
                outputs.push_back(denoiser.processOne(input));
            }
            expect(eq(outputs.size(), 50UZ)) << "should produce output for all inputs";
            expect(std::isfinite(outputs.back())) << "output should be finite";
        };

        "reset clears state"_test = [] {
            constexpr Config<double> config{.maxRank = 2UZ, .boundaryPolicy = BoundaryPolicy::Default};
            SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

            // Process some samples
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                double input = std::sin(2.0 * std::numbers::pi_v<double> * 0.1 * static_cast<double>(i));
                std::ignore  = denoiser.processOne(input);
            }
            denoiser.reset();

            // After reset, should still produce valid output
            double output = denoiser.processOne(std::sin(0.0));
            expect(std::isfinite(output)) << "output valid after reset";
        };
    };

    "streaming denoiser hopFraction"_test = [] {
        "custom hopFraction"_test = [] {
            constexpr Config<double> config{.maxRank = 2UZ, .hopFraction = 0.5};
            SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

            expect(eq(denoiser.hopSize(), 16UZ)) << "hopSize should be windowSize * hopFraction";
        };

        "small hopFraction"_test = [] {
            constexpr Config<double> config{.maxRank = 2UZ, .hopFraction = 0.1};
            SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

            expect(eq(denoiser.hopSize(), 3UZ)) << "hopSize = 32 * 0.1 = 3";
        };
    };

    "streaming denoiser state"_test = [] {
        constexpr Config<double> config{.maxRank = 2UZ, .energyFraction = 0.95};
        SvdDenoiser<double>      denoiser(32UZ, 16UZ, config);

        "processOne returns values"_test = [&] {
            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 100UZ; ++i) {
                double input = std::sin(2.0 * std::numbers::pi_v<double> * 0.1 * static_cast<double>(i));
                outputs.push_back(denoiser.processOne(input));
            }
            expect(eq(outputs.size(), 100UZ));
        };
    };

    "complex signal support"_test = [] {
        using C = std::complex<double>;

        "complex denoising"_test = [] {
            constexpr std::size_t    N = 64UZ;
            constexpr Config<double> config{.maxRank = 2UZ, .energyFraction = 0.95};

            std::vector<C> signal(N);
            for (std::size_t i = 0UZ; i < N; ++i) {
                double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 16.0;
                signal[i]    = C{std::cos(phase), std::sin(phase)};
            }

            auto denoised = denoiseWindow<C>(signal, N / 2UZ, config);
            expect(eq(denoised.size(), N));
        };
    };
};

const boost::ut::suite<"SVD edge cases"> svdEdgeCaseTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::svd_filter;

    "minimum window size"_test = [] {
        "windowSize = 2"_test = [] {
            SvdDenoiser<double> denoiser(2UZ, 1UZ, Config<double>{.maxRank = 1UZ});
            expect(eq(denoiser.windowSize(), 2UZ));

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 10UZ; ++i) {
                outputs.push_back(denoiser.processOne(static_cast<double>(i)));
            }
            expect(eq(outputs.size(), 10UZ));
            expect(std::isfinite(outputs.back()));
        };

        "denoiseWindow with 2 samples"_test = [] {
            std::vector<double> signal = {1.0, 2.0};
            auto                result = denoiseWindow<double>(signal, 1UZ, Config<double>{});
            expect(eq(result.size(), 2UZ));
        };

        "denoiseWindow with 1 sample returns copy"_test = [] {
            std::vector<double> signal = {42.0};
            auto                result = denoiseWindow<double>(signal, 1UZ, Config<double>{});
            expect(eq(result.size(), 1UZ));
            expect(eq(result[0], 42.0));
        };
    };

    "Config parameter edge cases"_test = [] {
        "energyFraction = 1.0 keeps all"_test = [] {
            std::vector<double> signal(32);
            for (std::size_t i = 0UZ; i < 32UZ; ++i) {
                signal[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 8.0);
            }

            Config<double> config{.energyFraction = 1.0};
            auto           denoised = denoiseWindow<double>(signal, 16UZ, config);
            expect(eq(denoised.size(), signal.size()));

            double maxError = 0.0;
            for (std::size_t i = 0UZ; i < signal.size(); ++i) {
                maxError = std::max(maxError, std::abs(signal[i] - denoised[i]));
            }
            expect(lt(maxError, 1e-10)) << "energyFraction=1 should preserve signal exactly";
        };

        "energyFraction = 0.0 keeps at least rank 1"_test = [] {
            std::vector<double> signal(16);
            for (std::size_t i = 0UZ; i < 16UZ; ++i) {
                signal[i] = static_cast<double>(i);
            }

            Config<double> config{.energyFraction = 0.0};
            auto           denoised = denoiseWindow<double>(signal, 8UZ, config);
            expect(eq(denoised.size(), signal.size()));
            expect(std::isfinite(denoised[0]));
        };

        "maxRank = 1 produces rank-1 approximation"_test = [] {
            std::vector<double> signal(32);
            for (std::size_t i = 0UZ; i < 32UZ; ++i) {
                signal[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 8.0) + 0.5 * std::cos(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 4.0);
            }

            Config<double> config{.maxRank = 1UZ};
            auto           denoised = denoiseWindow<double>(signal, 16UZ, config);
            expect(eq(denoised.size(), signal.size()));
        };

        "absoluteThreshold filters small singular values"_test = [] {
            std::vector<double> signal(32);
            for (std::size_t i = 0UZ; i < 32UZ; ++i) {
                signal[i] = 10.0 * std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 8.0);
            }

            Config<double> config{.absoluteThreshold = 1.0};
            auto           denoised = denoiseWindow<double>(signal, 16UZ, config);
            expect(eq(denoised.size(), signal.size()));
        };

        "relativeThreshold filters by ratio"_test = [] {
            std::vector<double> signal(32);
            for (std::size_t i = 0UZ; i < 32UZ; ++i) {
                signal[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 8.0);
            }

            Config<double> config{.relativeThreshold = 0.5};
            auto           denoised = denoiseWindow<double>(signal, 16UZ, config);
            expect(eq(denoised.size(), signal.size()));
        };
    };

    "streaming denoiser edge cases"_test = [] {
        "setParameters reconfigures"_test = [] {
            SvdDenoiser<double> denoiser(32UZ, 16UZ, Config<double>{});
            expect(eq(denoiser.windowSize(), 32UZ));

            denoiser.setParameters(64UZ, 32UZ, Config<double>{.hopFraction = 0.5});
            expect(eq(denoiser.windowSize(), 64UZ));
            expect(eq(denoiser.hankelRows(), 32UZ));
            expect(eq(denoiser.hopSize(), 32UZ));
        };

        "setConfig updates config only"_test = [] {
            Config<double>      config1{.maxRank = 2UZ, .hopFraction = 0.25};
            SvdDenoiser<double> denoiser(32UZ, 16UZ, config1);
            expect(eq(denoiser.hopSize(), 8UZ));

            Config<double> config2{.maxRank = 4UZ, .hopFraction = 0.5};
            denoiser.setConfig(config2);
            expect(eq(denoiser.hopSize(), 16UZ));
            expect(eq(denoiser.windowSize(), 32UZ)); // unchanged
        };

        "complex streaming denoiser"_test = [] {
            using C = std::complex<double>;
            SvdDenoiser<C> denoiser(16UZ, 8UZ, Config<double>{.maxRank = 2UZ});

            std::vector<C> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 8.0;
                outputs.push_back(denoiser.processOne(C{std::cos(phase), std::sin(phase)}));
            }
            expect(eq(outputs.size(), 50UZ));
            expect(std::isfinite(outputs.back().real()));
            expect(std::isfinite(outputs.back().imag()));
        };

        "delay calculation"_test = [] {
            SvdDenoiser<double> denoiser32(32UZ, 16UZ, Config<double>{});
            expect(eq(denoiser32.delay(), 15UZ)) << "(32-1)/2 = 15";

            SvdDenoiser<double> denoiser33(33UZ, 16UZ, Config<double>{});
            expect(eq(denoiser33.delay(), 16UZ)) << "(33-1)/2 = 16";
        };

        "very small hopFraction clamps to 1"_test = [] {
            Config<double>      config{.hopFraction = 0.001};
            SvdDenoiser<double> denoiser(32UZ, 16UZ, config);
            expect(ge(denoiser.hopSize(), 1UZ)) << "hopSize should be at least 1";
        };
    };

    "near-constant signal handling"_test = [] {
        "DC with small variation"_test = [] {
            // Pure DC creates degenerate Hankel matrix, add tiny variation
            std::vector<double> signal(64);
            for (std::size_t i = 0UZ; i < 64UZ; ++i) {
                signal[i] = 5.0 + 0.001 * std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 64.0);
            }

            Config<double> config{.maxRank = 2UZ};
            auto           result = denoiseWindow<double>(signal, 32UZ, config);

            expect(eq(result.size(), 64UZ));
            for (const auto& val : result) {
                expect(std::isfinite(val));
                expect(approx(val, 5.0, 0.1)) << "near-DC should be close to 5.0";
            }
        };

        "streaming with slowly varying input"_test = [] {
            SvdDenoiser<double> denoiser(16UZ, 8UZ, Config<double>{.maxRank = 2UZ});

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                // Small variation to avoid degenerate matrix
                double input = 3.0 + 0.01 * std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 50.0);
                outputs.push_back(denoiser.processOne(input));
            }

            for (std::size_t i = 30UZ; i < 50UZ; ++i) {
                expect(std::isfinite(outputs[i]));
            }
        };
    };
};

const boost::ut::suite<"mountain range visualisation"> mountainRangeTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;

    "basic mountain range"_test = [] {
        ImChart<180UZ, 30UZ> chart;

        constexpr std::size_t            nPoints = 1000;
        constexpr std::size_t            nTraces = 5;
        std::vector<double>              xValues(nPoints);
        std::vector<std::vector<double>> traces(nTraces);

        for (std::size_t i = 0; i < nPoints; ++i) {
            xValues[i] = static_cast<double>(i);
        }
        for (std::size_t t = 0; t < nTraces; ++t) {
            traces[t].resize(nPoints);
            double phase = static_cast<double>(t) * 0.5;
            for (std::size_t i = 0; i < nPoints; ++i) {
                traces[t][i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 200.0 + phase);
            }
        }

        expect(nothrow([&] { chart.drawMountainRange(xValues, traces, "wave"); }));
        expect(nothrow([&] { chart.draw(); }));
    };

    "mountain range with custom offsets"_test = [] {
        ImChart<180UZ, 60UZ> chart;

        constexpr std::size_t            nPoints     = 8000UZ;
        constexpr std::size_t            nTraces     = 24UZ;
        constexpr double                 sigma1      = 200.0;
        constexpr double                 centre1Base = 0.25 * static_cast<double>(nPoints);
        constexpr double                 centre2     = 0.75 * static_cast<double>(nPoints);
        constexpr double                 sigma2Base  = 150.0;
        std::vector<double>              xValues(nPoints);
        std::vector<std::vector<double>> traces(nTraces);

        for (std::size_t i = 0; i < nPoints; ++i) {
            xValues[i] = static_cast<double>(i);
        }
        for (std::size_t t = 0; t < nTraces; ++t) {
            traces[t].resize(nPoints);
            double phase   = 5.0 * std::numbers::pi_v<double> * static_cast<double>(t) / static_cast<double>(nTraces);
            double centre1 = centre1Base + sigma1 * std::sin(phase);
            double sigma2  = sigma2Base * (2.0 + 0.5 * std::sin(phase));

            for (std::size_t i = 0; i < nPoints; ++i) {
                double x1    = static_cast<double>(i) - centre1;
                double x2    = static_cast<double>(i) - centre2;
                traces[t][i] = std::exp(-x1 * x1 / (2.0 * sigma1 * sigma1)) + std::exp(-x2 * x2 / (2.0 * sigma2 * sigma2));
            }
        }

        expect(nothrow([&] { chart.drawMountainRange(xValues, traces, "peak", 1UZ, 1UZ, 0UZ); }));
        expect(nothrow([&] { chart.draw(); }));
    };
};

namespace {
void printStats(std::string_view label, double noisyRms, double lpRms, double svdRms) { std::println(stderr, "[{}] noisy={:.4f}, LP={:.4f} ({:.1f}%), SVD={:.4f} ({:.1f}%)", label, noisyRms, lpRms, 100.0 * (1.0 - lpRms / noisyRms), svdRms, 100.0 * (1.0 - svdRms / noisyRms)); }

auto computeRms(const std::vector<double>& filtered, const std::vector<double>& clean, std::size_t skipStart, std::size_t skipEnd, std::size_t delay) {
    double      rms   = 0.0;
    std::size_t count = 0UZ;
    for (std::size_t i = skipStart; i < filtered.size() - skipEnd; ++i) {
        if (i >= delay) {
            rms += std::pow(filtered[i] - clean[i - delay], 2.0);
            ++count;
        }
    }
    return count > 0UZ ? std::sqrt(rms / static_cast<double>(count)) : 0.0;
}
} // namespace

const boost::ut::suite<"SVD denoising comparison"> visualDemoTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;
    using namespace gr::algorithm::svd_filter;
    using namespace gr::filter;

    constexpr std::size_t kChartWidth  = 180UZ;
    constexpr std::size_t kChartHeight = 60UZ;

    "single sinusoid"_test = [] {
        constexpr std::size_t N       = 640UZ;
        constexpr double      fSignal = 2.5 / static_cast<double>(N);
        constexpr double      fc      = 0.01;

        std::mt19937                     gen(42);
        std::normal_distribution<double> noise(0.0, 0.5);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = std::sin(2.0 * std::numbers::pi_v<double> * fSignal * static_cast<double>(i));
            noisy[i]   = clean[i] + noise(gen);
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 40UZ;
        constexpr Config<double> svdConfig{.maxRank = 2UZ, .energyFraction = 0.80};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 10UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("sinusoid", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };

    "multi-tone"_test = [] {
        constexpr std::size_t N  = 640UZ;
        constexpr double      f1 = 3.0 / static_cast<double>(N);
        constexpr double      f2 = 6.0 / static_cast<double>(N);
        constexpr double      fc = 0.015;

        std::mt19937                     gen(123);
        std::normal_distribution<double> noise(0.0, 0.4);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = std::sin(2.0 * std::numbers::pi_v<double> * f1 * static_cast<double>(i)) + 0.5 * std::sin(2.0 * std::numbers::pi_v<double> * f2 * static_cast<double>(i));
            noisy[i]   = clean[i] + noise(gen);
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 112UZ;
        constexpr Config<double> svdConfig{.maxRank = 3UZ, .energyFraction = 0.80};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 10UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("multi-tone", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };

    "damped oscillation"_test = [] {
        constexpr std::size_t N  = 640UZ;
        constexpr double      fc = 0.015;

        std::mt19937                     gen(456);
        std::normal_distribution<double> noise(0.0, 0.3);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i]      = static_cast<double>(i);
            double t        = static_cast<double>(i) / static_cast<double>(N);
            double envelope = std::exp(-2.0 * t);
            clean[i]        = envelope * std::sin(2.0 * std::numbers::pi_v<double> * 4.0 * t);
            noisy[i]        = clean[i] + noise(gen);
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 104UZ;
        constexpr Config<double> svdConfig{.maxRank = 2UZ, .energyFraction = 0.80};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 10UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("damped", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };

    "DC + ramp"_test = [] {
        constexpr std::size_t N  = 2000UZ;
        constexpr double      fc = 0.005;

        std::mt19937                     gen(789);
        std::normal_distribution<double> noise(0.0, 0.4);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = 0.5 + 0.8 * static_cast<double>(i) / static_cast<double>(N);
            noisy[i]   = clean[i] + noise(gen);
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 128UZ;
        constexpr Config<double> svdConfig{.maxRank = 1UZ, .energyFraction = 0.70};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 50UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 50UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("DC+ramp", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };

    "Gaussian pulse"_test = [] {
        constexpr std::size_t N      = 640UZ;
        constexpr double      centre = static_cast<double>(N) / 2.0;
        constexpr double      width  = static_cast<double>(N) / 8.0;

        gr::DataSet<double> cleanDs;
        cleanDs.axis_names        = {"sample"};
        cleanDs.axis_units        = {""};
        cleanDs.axis_values       = {std::vector<double>(N)};
        cleanDs.signal_names      = {"Gaussian"};
        cleanDs.signal_quantities = {"amplitude"};
        cleanDs.signal_units      = {""};
        cleanDs.signal_values.resize(N);
        cleanDs.extents = {static_cast<std::int32_t>(N)};

        for (std::size_t i = 0UZ; i < N; ++i) {
            cleanDs.axis_values[0][i] = static_cast<double>(i);
            double x                  = static_cast<double>(i) - centre;
            cleanDs.signal_values[i]  = std::exp(-x * x / (2.0 * width * width));
        }

        auto noisyDs = gr::dataset::addNoise(cleanDs, 0.25, 0UZ, 321U);

        constexpr std::size_t    svdWindowSize = 71UZ;
        constexpr Config<double> svdConfig{.maxRank = 2UZ, .energyFraction = 0.99};
        auto                     svdFiltered = denoiseWindow<double>(noisyDs.signal_values, svdWindowSize, svdConfig);

        auto maDs = gr::dataset::filter::applyMovingAverage(noisyDs, svdWindowSize);

        const auto            cleanSignal = cleanDs.signalValues(0UZ);
        const auto            noisySignal = noisyDs.signalValues(0UZ);
        const auto            maSignal    = maDs.signalValues(0UZ);
        constexpr std::size_t skipEdge    = 20UZ;
        double                noisyRms = 0.0, maRms = 0.0, svdRms = 0.0;
        std::size_t           count = 0UZ;
        for (std::size_t i = skipEdge; i < N - skipEdge; ++i) {
            noisyRms += std::pow(noisySignal[i] - cleanSignal[i], 2.0);
            maRms += std::pow(maSignal[i] - cleanSignal[i], 2.0);
            svdRms += std::pow(svdFiltered[i] - cleanSignal[i], 2.0);
            ++count;
        }
        noisyRms = std::sqrt(noisyRms / static_cast<double>(count));
        maRms    = std::sqrt(maRms / static_cast<double>(count));
        svdRms   = std::sqrt(svdRms / static_cast<double>(count));

        printStats("Gaussian", noisyRms, maRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(cleanDs.axis_values[0], noisySignal, "noisy");
        chart.draw(cleanDs.axis_values[0], maSignal, "MA");
        chart.draw(cleanDs.axis_values[0], svdFiltered, "SVD");
        chart.draw(cleanDs.axis_values[0], cleanSignal, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(maRms, noisyRms)) << "MA should reduce noise";
    };

    "high noise (0 dB SNR)"_test = [] {
        constexpr std::size_t N       = 640UZ;
        constexpr double      fSignal = 2.5 / static_cast<double>(N);
        constexpr double      fc      = 0.01;

        std::mt19937                     gen(999);
        std::normal_distribution<double> noise(0.0, 1.0);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = std::sin(2.0 * std::numbers::pi_v<double> * fSignal * static_cast<double>(i));
            noisy[i]   = clean[i] + noise(gen);
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 88UZ;
        constexpr Config<double> svdConfig{.maxRank = 2UZ, .energyFraction = 0.80};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 10UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("high noise", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, noisyRms)) << "SVD should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };

    "chirp"_test = [] {
        constexpr std::size_t N    = 2000UZ;
        constexpr double      fMin = 0.002;
        constexpr double      fMax = 0.04;
        constexpr double      fc   = 0.009;

        std::mt19937                     gen(12345);
        std::normal_distribution<double> noise(0.0, 0.5);

        std::vector<double> xValues(N), clean(N), noisy(N);
        double              phase = 0.0;
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i]  = static_cast<double>(i);
            double t    = static_cast<double>(i) / static_cast<double>(N);
            double t8   = std::pow(t, 2.0);
            double freq = fMin + (fMax - fMin) * t8;
            clean[i]    = std::sin(phase);
            noisy[i]    = clean[i] + noise(gen);
            phase += 2.0 * std::numbers::pi_v<double> * freq;
        }

        constexpr FilterParameters firParams{.order = 2UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        constexpr std::size_t    svdWindowSize = 91UZ;
        constexpr Config<double> svdConfig{.maxRank = 3UZ, .energyFraction = 0.70};

        Filter<double>      firFilter(firCoeffs);
        SvdDenoiser<double> svdFilter(svdWindowSize, svdWindowSize / 2UZ, svdConfig);
        std::vector<double> lpFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]  = firFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        const std::size_t svdDelay  = svdFilter.delay();
        const std::size_t skipStart = std::max({svdDelay, lpDelay, svdWindowSize}) + 100UZ;
        const std::size_t skipEnd   = std::max(svdDelay, lpDelay) + 100UZ;

        double noisyRms = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms    = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double svdRms   = computeRms(svdFiltered, clean, skipStart, skipEnd, svdDelay);

        printStats("chirp", noisyRms, lpRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(svdRms, lpRms)) << "SVD should outperform LP on wideband signals";
    };
};

const boost::ut::suite<"SVD streaming steady state"> svdSteadyStateTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::svd_filter;

    "steady state noise reduction"_test = [] {
        constexpr std::size_t N       = 512UZ;
        constexpr double      fSignal = 8.0 / static_cast<double>(N);

        std::mt19937                     gen(123);
        std::normal_distribution<double> noise(0.0, 0.3);

        std::vector<double> clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            clean[i] = std::sin(2.0 * std::numbers::pi_v<double> * fSignal * static_cast<double>(i));
            noisy[i] = clean[i] + noise(gen);
        }

        constexpr std::size_t    windowSize = 48UZ;
        constexpr Config<double> config{.maxRank = 2UZ, .energyFraction = 0.90};

        SvdDenoiser<double> filter(windowSize, windowSize / 2UZ, config);
        std::vector<double> filtered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            filtered[i] = filter.processOne(noisy[i]);
        }

        const std::size_t svdDelay    = filter.delay();
        const std::size_t steadyStart = windowSize + svdDelay + 10UZ;

        auto computeRmsLocal = [&](std::size_t start, std::size_t end, std::size_t delay) {
            double      rms   = 0.0;
            std::size_t count = 0UZ;
            for (std::size_t i = std::max(start, delay); i < end; ++i) {
                rms += std::pow(filtered[i] - clean[i - delay], 2.0);
                ++count;
            }
            return count > 0UZ ? std::sqrt(rms / static_cast<double>(count)) : 0.0;
        };

        auto computeNoisyRms = [&](std::size_t start, std::size_t end) {
            double      rms   = 0.0;
            std::size_t count = 0UZ;
            for (std::size_t i = start; i < end; ++i) {
                rms += std::pow(noisy[i] - clean[i], 2.0);
                ++count;
            }
            return count > 0UZ ? std::sqrt(rms / static_cast<double>(count)) : 0.0;
        };

        double steadyRms      = computeRmsLocal(steadyStart, N, svdDelay);
        double steadyNoisyRms = computeNoisyRms(steadyStart, N);
        double improvement    = 100.0 * (1.0 - steadyRms / steadyNoisyRms);

        std::println(stderr, "steady state: RMS={:.4f}, noisy={:.4f}, improvement={:.1f}%", steadyRms, steadyNoisyRms, improvement);

        expect(gt(improvement, 50.0)) << "steady state should show >50% noise reduction";
    };
};

const boost::ut::suite<"visual spectrum analysis"> ssaVisualisationTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;
    using namespace gr::algorithm::svd_filter;

    "SSA decomposition"_test = [] {
        constexpr std::size_t N           = 1000UZ;
        constexpr std::size_t L           = N / 10UZ; // window size < N/2
        constexpr std::size_t nComponents = 5UZ;
        constexpr double      twoPi       = 2.0 * std::numbers::pi_v<double>;

        // generate synthetic signal: parabola + sine + cosine + noise
        std::mt19937                     gen(42);
        std::normal_distribution<double> noiseDist(0.0, 0.05);

        std::vector<double> xValues(N), signal(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            const double t     = static_cast<double>(i) / static_cast<double>(N);
            const double phase = twoPi * 10.0 * t;
            xValues[i]         = static_cast<double>(i);
            signal[i]          = 4.0 * std::pow(t - 0.5, 2) + 0.30 * std::sin(phase) + 0.15 * std::cos(2.0 * phase) + noiseDist(gen);
        }

        // SSA Step 1: Embedding - build Hankel matrix H[i,j] = signal[i+j]
        auto H                    = gr::math::hankel<double>(signal, L).value();
        const auto [nRows, nCols] = std::pair{H.extent(0), H.extent(1)}; // L, K = N - L + 1

        // SSA Step 2: SVD decomposition H = U · S · V^T
        gr::Tensor<double> U, S, V;
        auto               status = gr::math::gesvd(U, S, V, H);
        expect(status == gr::math::svd::Status::Success || status == gr::math::svd::Status::EarlyReturn);

        const std::size_t nSV = std::min({nComponents, S.size(), nRows, nCols});

        // SSA Step 3 & 4: Reconstruction via diagonal averaging
        // for component i: X_i = s_i · u_i · v_i^T, then average anti-diagonals
        auto reconstructComponent = [&](std::size_t i) -> std::vector<double> {
            gr::Tensor<double> Xi({nRows, nCols});
            for (std::size_t r = 0UZ; r < nRows; ++r) {
                for (std::size_t c = 0UZ; c < nCols; ++c) {
                    Xi[r, c] = S[i] * U[r, i] * V[c, i];
                }
            }
            return gr::math::hankelAverage(Xi).value();
        };

        std::vector<std::vector<double>> components(nSV);
        std::ranges::generate(components, [&, i = 0UZ]() mutable { return reconstructComponent(i++); });

        // singular values for chart
        std::vector<double> svIndices(S.size()), svValues(S.size());
        for (std::size_t i = 0UZ; i < S.size(); ++i) {
            svIndices[i] = static_cast<double>(i + 1UZ);
            svValues[i]  = S[i];
        }

        // Plot 1: Reconstructed SSA components

        ImChart<180UZ, 40UZ> chart({{0.0, static_cast<double>(N)}, {-0.2, 1.2}});
        chart.draw(xValues, signal, "F");
        for (std::size_t i = 0UZ; i < nSV; ++i) {
            chart.draw(xValues, components[i], "F" + std::to_string(i));
        }
        expect(nothrow([&] { chart.draw(); }));

        // Plot 2: Singular values (log Y-axis)

        ImChart<180UZ, 15UZ, LinearAxisTransform, LogAxisTransform> chartS( // s
            {{-1.0, static_cast<double>(std::min(2 * nComponents, S.size())) + 0.0}, {0.5, 200.0}});
        chartS.draw<Style::Marker>(svIndices, svValues, "S");
        expect(nothrow([&] { chartS.draw(); }));

        // sanity checks
        expect(gt(S[0], S[1])) << "singular values should be descending";
        expect(gt(S[0], 0.0)) << "first singular value should be positive";

        // verify reconstruction: Σ components ≈ original signal
        std::vector<double> reconstructed(N, 0.0);
        for (std::size_t i = 0UZ; i < S.size(); ++i) {
            auto comp = reconstructComponent(i);
            std::ranges::transform(reconstructed, comp, reconstructed.begin(), std::plus{});
        }

        double maxErr = 0.0;
        for (std::size_t i = 0UZ; i < N; ++i) {
            maxErr = std::max(maxErr, std::abs(signal[i] - reconstructed[i]));
        }
        expect(lt(maxErr, 1e-9)) << "reconstruction error: " << maxErr;
    };
};

int main() { /* tests are automatically registered and run */ }
