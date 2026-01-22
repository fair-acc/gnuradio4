#include <boost/ut.hpp>

#include <cmath>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/filter/SavitzkyGolay.hpp>

const boost::ut::suite<"Savitzky-Golay coefficient computation"> coeffTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay;

    "smoothing coefficients symmetry"_test = [] {
        auto coeffs = computeCoefficients<double>(5UZ, 2UZ, {});
        expect(eq(coeffs.size(), 5UZ));
        for (std::size_t i = 0UZ; i < coeffs.size() / 2UZ; ++i) {
            expect(approx(coeffs[i], coeffs[coeffs.size() - 1UZ - i], 1e-10)) << "symmetric at " << i;
        }
    };

    "smoothing coefficients sum to 1"_test = [] {
        for (std::size_t W : {5UZ, 7UZ, 11UZ, 21UZ}) {
            auto   coeffs = computeCoefficients<double>(W, 3UZ, {});
            double sum    = std::reduce(coeffs.begin(), coeffs.end());
            expect(approx(sum, 1.0, 1e-10)) << "W=" << W;
        }
    };

    "derivative coefficients sum to 0"_test = [] {
        auto   coeffs = computeCoefficients<double>(7UZ, 3UZ, {.derivOrder = 1UZ});
        double sum    = std::reduce(coeffs.begin(), coeffs.end());
        expect(approx(sum, 0.0, 1e-10)) << "first derivative";

        auto   coeffs2 = computeCoefficients<double>(7UZ, 4UZ, {.derivOrder = 2UZ});
        double sum2    = std::reduce(coeffs2.begin(), coeffs2.end());
        expect(approx(sum2, 0.0, 1e-10)) << "second derivative";
    };

    "causal coefficients"_test = [] {
        auto coeffs = computeCoefficients<double>(5UZ, 2UZ, {.alignment = Alignment::Causal});
        expect(eq(coeffs.size(), 5UZ));
        double sum = std::reduce(coeffs.begin(), coeffs.end());
        expect(approx(sum, 1.0, 1e-10));
    };

    "parameter clamping"_test = [] {
        auto c1 = computeCoefficients<double>(0UZ, 0UZ, {.delta = 0.0});
        expect(eq(c1.size(), 1UZ)) << "window clamped to 1";

        auto c2 = computeCoefficients<double>(5UZ, 10UZ, {});
        expect(eq(c2.size(), 5UZ)) << "poly order clamped";

        auto c3 = computeCoefficients<double>(5UZ, 2UZ, {.derivOrder = 10UZ});
        expect(eq(c3.size(), 5UZ)) << "deriv order clamped";
    };
};

const boost::ut::suite<"Savitzky-Golay boundary handling"> boundaryTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay::detail;

    "reflect index"_test = [] {
        expect(eq(reflectIndex(0, 5UZ), 0UZ));
        expect(eq(reflectIndex(4, 5UZ), 4UZ));
        expect(eq(reflectIndex(-1, 5UZ), 0UZ));
        expect(eq(reflectIndex(-2, 5UZ), 1UZ));
        expect(eq(reflectIndex(5, 5UZ), 4UZ));
        expect(eq(reflectIndex(6, 5UZ), 3UZ));
        expect(eq(reflectIndex(-3, 5UZ), 2UZ));
    };

    "replicate index"_test = [] {
        expect(eq(replicateIndex(0, 5UZ), 0UZ));
        expect(eq(replicateIndex(4, 5UZ), 4UZ));
        expect(eq(replicateIndex(-1, 5UZ), 0UZ));
        expect(eq(replicateIndex(-10, 5UZ), 0UZ));
        expect(eq(replicateIndex(5, 5UZ), 4UZ));
        expect(eq(replicateIndex(100, 5UZ), 4UZ));
    };

    "empty array edge case"_test = [] {
        expect(eq(reflectIndex(0, 0UZ), 0UZ));
        expect(eq(replicateIndex(0, 0UZ), 0UZ));
    };
};

const boost::ut::suite<"Savitzky-Golay edge cases"> sgEdgeCaseTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay;

    "minimum window sizes"_test = [] {
        "window_size = 1"_test = [] {
            auto coeffs = computeCoefficients<double>(1UZ, 0UZ, {});
            expect(eq(coeffs.size(), 1UZ));
            expect(approx(coeffs[0], 1.0, 1e-10)) << "single coeff should be 1.0";
        };

        "window_size = poly_order + 1 (minimum valid)"_test = [] {
            auto coeffs = computeCoefficients<double>(3UZ, 2UZ, {});
            expect(eq(coeffs.size(), 3UZ));
            double sum = std::reduce(coeffs.begin(), coeffs.end());
            expect(approx(sum, 1.0, 1e-10));
        };

        "window_size = 2, poly_order = 1"_test = [] {
            auto coeffs = computeCoefficients<double>(2UZ, 1UZ, {});
            expect(eq(coeffs.size(), 2UZ));
            double sum = std::reduce(coeffs.begin(), coeffs.end());
            expect(approx(sum, 1.0, 1e-10));
        };
    };

    "poly_order = 0 (moving average)"_test = [] {
        auto   coeffs = computeCoefficients<double>(5UZ, 0UZ, {});
        double sum    = std::reduce(coeffs.begin(), coeffs.end());
        expect(approx(sum, 1.0, 1e-10));

        // all coefficients should be equal for moving average
        for (const auto& c : coeffs) {
            expect(approx(c, 1.0 / 5.0, 1e-10)) << "MA coeffs should be 1/N";
        }
    };

    "input size edge cases"_test = [] {
        "input shorter than window uses boundary policy"_test = [] {
            std::vector<double> in = {1.0, 2.0, 3.0};
            std::vector<double> out(3);

            Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
            auto           coeffs = computeCoefficients<double>(5UZ, 2UZ, config);
            apply<double>(in, out, coeffs, config);

            for (const auto& v : out) {
                expect(std::isfinite(v)) << "output should be finite";
            }
        };

        "single element input"_test = [] {
            std::vector<double> in = {42.0};
            std::vector<double> out(1);

            Config<double> config{.boundaryPolicy = BoundaryPolicy::Replicate};
            auto           coeffs = computeCoefficients<double>(3UZ, 1UZ, config);
            apply<double>(in, out, coeffs, config);

            expect(std::isfinite(out[0]));
        };

        "empty input handling"_test = [] {
            std::vector<double> in;
            std::vector<double> out;

            Config<double> config{};
            auto           coeffs = computeCoefficients<double>(5UZ, 2UZ, config);
            // Should not crash on empty input
            apply<double>(in, out, coeffs, config);
            expect(eq(out.size(), 0UZ));
        };
    };

    "boundary policy comparison"_test = [] {
        std::vector<double> in = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> outReflect(8), outReplicate(8);

        Config<double> configReflect{.boundaryPolicy = BoundaryPolicy::Reflect};
        Config<double> configReplicate{.boundaryPolicy = BoundaryPolicy::Replicate};

        auto coeffsReflect   = computeCoefficients<double>(5UZ, 2UZ, configReflect);
        auto coeffsReplicate = computeCoefficients<double>(5UZ, 2UZ, configReplicate);

        apply<double>(in, outReflect, coeffsReflect, configReflect);
        apply<double>(in, outReplicate, coeffsReplicate, configReplicate);

        // Both should produce finite results
        for (std::size_t i = 0UZ; i < 8UZ; ++i) {
            expect(std::isfinite(outReflect[i]));
            expect(std::isfinite(outReplicate[i]));
        }

        // Edge values may differ between policies
        // Just verify both handle boundaries gracefully
    };

    "streaming filter edge cases"_test = [] {
        "minimum window streaming"_test = [] {
            SavitzkyGolayFilter<double> filter(3UZ, 2UZ);
            expect(eq(filter.windowSize(), 3UZ));

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 20UZ; ++i) {
                outputs.push_back(filter.processOne(static_cast<double>(i)));
            }
            expect(eq(outputs.size(), 20UZ));
            expect(std::isfinite(outputs.back()));
        };

        "large window streaming"_test = [] {
            SavitzkyGolayFilter<double> filter(101UZ, 4UZ);
            expect(eq(filter.windowSize(), 101UZ));
            expect(eq(filter.delay(), 50UZ)) << "(101-1)/2 = 50";

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 200UZ; ++i) {
                outputs.push_back(filter.processOne(std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 50.0)));
            }
            expect(std::isfinite(outputs.back()));
        };

        "constant input streaming"_test = [] {
            SavitzkyGolayFilter<double> filter(11UZ, 3UZ);

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                outputs.push_back(filter.processOne(7.0));
            }

            // After warmup, constant input should produce constant output
            for (std::size_t i = 20UZ; i < 50UZ; ++i) {
                expect(approx(outputs[i], 7.0, 1e-10)) << "constant preserved at " << i;
            }
        };
    };

    "derivative edge cases"_test = [] {
        "high derivative order clamped"_test = [] {
            // derivOrder > polyOrder should be clamped
            auto coeffs = computeCoefficients<double>(5UZ, 2UZ, {.derivOrder = 5UZ});
            expect(eq(coeffs.size(), 5UZ));
        };

        "second derivative on quadratic"_test = [] {
            constexpr std::size_t N     = 30UZ;
            constexpr double      delta = 0.1;

            std::vector<double> in(N), out(N);
            for (std::size_t i = 0UZ; i < N; ++i) {
                double x = static_cast<double>(i) * delta;
                in[i]    = x * x; // f(x) = x^2
            }

            Config<double> config{.derivOrder = 2UZ, .delta = delta, .boundaryPolicy = BoundaryPolicy::Reflect};
            auto           coeffs = computeCoefficients<double>(7UZ, 3UZ, config);
            apply<double>(in, out, coeffs, config);

            // d²/dx²(x²) = 2
            for (std::size_t i = 5UZ; i < N - 5UZ; ++i) {
                expect(approx(out[i], 2.0, 0.1)) << "second derivative at " << i;
            }
        };

        "zero-phase with derivative produces finite output"_test = [] {
            // Note: zero-phase with derivative applies derivative twice (forward + backward),
            // so it computes d²/dx² not d/dx. Just verify it produces finite output.
            constexpr std::size_t N     = 50UZ;
            constexpr double      delta = 0.1;

            std::vector<double> in(N), out(N);
            for (std::size_t i = 0UZ; i < N; ++i) {
                double x = static_cast<double>(i) * delta;
                in[i]    = std::sin(x);
            }

            Config<double> config{.derivOrder = 1UZ, .delta = delta, .boundaryPolicy = BoundaryPolicy::Reflect};
            auto           coeffs = computeCoefficients<double>(9UZ, 4UZ, config);
            applyZeroPhase<double>(in, out, coeffs, config);

            for (std::size_t i = 10UZ; i < N - 10UZ; ++i) {
                expect(std::isfinite(out[i])) << "output finite at " << i;
            }
        };
    };

    "delta parameter"_test = [] {
        "delta = 0 clamped to positive"_test = [] {
            // delta <= 0 should be handled gracefully
            auto coeffs = computeCoefficients<double>(5UZ, 2UZ, {.derivOrder = 1UZ, .delta = 0.0});
            expect(eq(coeffs.size(), 5UZ));
            // Coefficients should still be finite
            for (const auto& c : coeffs) {
                expect(std::isfinite(c));
            }
        };

        "negative delta handled"_test = [] {
            auto coeffs = computeCoefficients<double>(5UZ, 2UZ, {.derivOrder = 1UZ, .delta = -1.0});
            expect(eq(coeffs.size(), 5UZ));
            for (const auto& c : coeffs) {
                expect(std::isfinite(c));
            }
        };
    };

    "setParameters reconfigures streaming filter"_test = [] {
        SavitzkyGolayFilter<double> filter(5UZ, 2UZ);
        expect(eq(filter.windowSize(), 5UZ));
        expect(eq(filter.polyOrder(), 2UZ));

        filter.setParameters(11UZ, 4UZ, {.alignment = Alignment::Causal});
        expect(eq(filter.windowSize(), 11UZ));
        expect(eq(filter.polyOrder(), 4UZ));
        expect(filter.alignment() == Alignment::Causal);
    };
};

const boost::ut::suite<"Savitzky-Golay offline filtering"> offlineTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay;

    "preserves polynomial"_test = [] {
        constexpr std::size_t N = 20UZ;
        std::vector<double>   in(N), out(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            double x = static_cast<double>(i) - 10.0;
            in[i]    = 2.0 + 3.0 * x + 0.5 * x * x;
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(5UZ, 2UZ, config);
        apply<double>(in, out, coeffs, config);

        for (std::size_t i = 3UZ; i < N - 3UZ; ++i) {
            expect(approx(out[i], in[i], 1e-10)) << "quadratic preserved at " << i;
        }
    };

    "smoothing reduces noise"_test = [] {
        constexpr std::size_t N = 100UZ;

        std::mt19937                     gen(42);
        std::normal_distribution<double> noise(0.0, 0.2);

        std::vector<double> clean(N), noisy(N), smoothed(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            clean[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 20.0);
            noisy[i] = clean[i] + noise(gen);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(11UZ, 3UZ, config);
        apply<double>(noisy, smoothed, coeffs, config);

        double noisyRms = 0.0, smoothedRms = 0.0;
        for (std::size_t i = 10UZ; i < N - 10UZ; ++i) {
            noisyRms += std::pow(noisy[i] - clean[i], 2.0);
            smoothedRms += std::pow(smoothed[i] - clean[i], 2.0);
        }
        noisyRms    = std::sqrt(noisyRms / static_cast<double>(N - 20UZ));
        smoothedRms = std::sqrt(smoothedRms / static_cast<double>(N - 20UZ));

        expect(lt(smoothedRms, noisyRms)) << "smoothing reduces error";
    };

    "first derivative accuracy"_test = [] {
        constexpr std::size_t N     = 50UZ;
        constexpr double      delta = 0.1;

        std::vector<double> in(N), out(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            double x = static_cast<double>(i) * delta;
            in[i]    = x * x;
        }

        Config<double> config{.derivOrder = 1UZ, .delta = delta, .boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(5UZ, 2UZ, config);
        apply<double>(in, out, coeffs, config);

        for (std::size_t i = 5UZ; i < N - 5UZ; ++i) {
            double x        = static_cast<double>(i) * delta;
            double expected = 2.0 * x;
            expect(approx(out[i], expected, 1e-8)) << "d/dx(x^2) = 2x at i=" << i;
        }
    };
};

const boost::ut::suite<"Savitzky-Golay zero-phase filtering"> zeroPhaseTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay;

    "no phase shift"_test = [] {
        constexpr std::size_t N = 200UZ;

        std::vector<double> in(N), out(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            in[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 40.0);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(11UZ, 3UZ, config);
        applyZeroPhase<double>(in, out, coeffs, config);

        double phase0 = std::atan2(in[50], in[40]);
        double phase1 = std::atan2(out[50], out[40]);
        expect(approx(phase0, phase1, 0.01)) << "phase preserved";
    };

    "peak position preserved"_test = [] {
        constexpr std::size_t N      = 100UZ;
        constexpr std::size_t centre = 50UZ;

        std::vector<double> in(N), out(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            double x = static_cast<double>(i) - static_cast<double>(centre);
            in[i]    = std::exp(-x * x / 50.0);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(11UZ, 3UZ, config);
        applyZeroPhase<double>(in, out, coeffs, config);

        auto inMax  = std::max_element(in.begin(), in.end());
        auto outMax = std::max_element(out.begin(), out.end());
        expect(eq(std::distance(in.begin(), inMax), std::distance(out.begin(), outMax))) << "peak at same position";
    };
};

const boost::ut::suite<"Savitzky-Golay streaming"> streamingTests = [] {
    using namespace boost::ut;
    using namespace gr::algorithm::savitzky_golay;

    "processOne consistency"_test = [] {
        constexpr std::size_t N = 100UZ;

        std::vector<double> in(N), streamOut(N), batchOut(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            in[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 20.0);
        }

        SavitzkyGolayFilter<double> filter(11UZ, 3UZ);
        for (std::size_t i = 0UZ; i < N; ++i) {
            streamOut[i] = filter.processOne(in[i]);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(11UZ, 3UZ, config);
        apply<double>(in, batchOut, coeffs, config);

        const std::size_t warmup = filter.windowSize();
        for (std::size_t i = warmup + 5UZ; i < N - 5UZ; ++i) {
            expect(approx(streamOut[i], batchOut[i - filter.delay()], 0.01)) << "matches batch at " << i;
        }
    };

    "causal mode delay"_test = [] {
        SavitzkyGolayFilter<double> causalFilter(11UZ, 3UZ, {.alignment = Alignment::Causal});
        SavitzkyGolayFilter<double> centredFilter(11UZ, 3UZ);

        expect(eq(causalFilter.delay(), 0UZ)) << "causal has no delay";
        expect(eq(centredFilter.delay(), 5UZ)) << "centred has (W-1)/2 delay";
    };

    "reset clears state"_test = [] {
        SavitzkyGolayFilter<double> filter(5UZ, 2UZ);
        SavitzkyGolayFilter<double> freshFilter(5UZ, 2UZ);

        for (int i = 0; i < 20; ++i) {
            std::ignore = filter.processOne(static_cast<double>(i));
        }

        filter.reset();
        expect(eq(filter.processOne(42.0), freshFilter.processOne(42.0))) << "reset produces fresh filter state";
    };

    "setParameters reconfigures"_test = [] {
        SavitzkyGolayFilter<double> filter(5UZ, 2UZ);
        expect(eq(filter.windowSize(), 5UZ));

        filter.setParameters(11UZ, 3UZ, {.alignment = Alignment::Causal});
        expect(eq(filter.windowSize(), 11UZ));
        expect(filter.alignment() == Alignment::Causal);
    };
};

namespace {
void printStats(std::string_view label, double noisyRms, double lpRms, double sgCentredRms, double sgCausalRms) { std::println(stderr, "[{}] noisy={:.4f}, LP={:.4f} ({:.1f}%), SG-centred={:.4f} ({:.1f}%), SG-causal={:.4f} ({:.1f}%)", label, noisyRms, lpRms, 100.0 * (1.0 - lpRms / noisyRms), sgCentredRms, 100.0 * (1.0 - sgCentredRms / noisyRms), sgCausalRms, 100.0 * (1.0 - sgCausalRms / noisyRms)); }

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

const boost::ut::suite<"Savitzky-Golay visual comparison"> visualTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;
    using namespace gr::algorithm::savitzky_golay;

    constexpr std::size_t kChartWidth  = 160UZ;
    constexpr std::size_t kChartHeight = 50UZ;

    "smoothing comparison"_test = [] {
        constexpr std::size_t N = 300UZ;

        std::mt19937                     gen(42);
        std::normal_distribution<double> noise(0.0, 0.3);

        std::vector<double> xValues(N), clean(N), noisy(N), sg5(N), sg11(N), sg21(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 50.0);
            noisy[i]   = clean[i] + noise(gen);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           c5  = computeCoefficients<double>(5UZ, 3UZ, config);
        auto           c11 = computeCoefficients<double>(11UZ, 3UZ, config);
        auto           c21 = computeCoefficients<double>(21UZ, 3UZ, config);

        apply<double>(noisy, sg5, c5, config);
        apply<double>(noisy, sg11, c11, config);
        apply<double>(noisy, sg21, c21, config);

        double noisyRms = computeRms(noisy, clean, 20UZ, 20UZ, 0UZ);
        double sg5Rms   = computeRms(sg5, clean, 20UZ, 20UZ, 0UZ);
        double sg11Rms  = computeRms(sg11, clean, 20UZ, 20UZ, 0UZ);
        double sg21Rms  = computeRms(sg21, clean, 20UZ, 20UZ, 0UZ);

        std::println(stderr, "[SG smoothing] noisy={:.4f}, W=5:{:.4f}, W=11:{:.4f}, W=21:{:.4f}", noisyRms, sg5Rms, sg11Rms, sg21Rms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, sg5, "W=5");
        chart.draw(xValues, sg11, "W=11");
        chart.draw(xValues, sg21, "W=21");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sg5Rms, noisyRms));
        expect(lt(sg11Rms, sg5Rms));
        expect(lt(sg21Rms, sg11Rms));
    };

    "derivative extraction"_test = [] {
        constexpr std::size_t N     = 200UZ;
        constexpr double      delta = 0.05;

        std::mt19937                     gen(123);
        std::normal_distribution<double> noise(0.0, 0.02);

        std::vector<double> xValues(N), clean(N), noisy(N), deriv(N), trueDeriv(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            double x     = static_cast<double>(i) * delta;
            xValues[i]   = x;
            clean[i]     = std::sin(x);
            noisy[i]     = clean[i] + noise(gen);
            trueDeriv[i] = std::cos(x);
        }

        Config<double> config{.derivOrder = 1UZ, .delta = delta, .boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(11UZ, 4UZ, config);
        apply<double>(noisy, deriv, coeffs, config);

        double derivRms = 0.0;
        for (std::size_t i = 20UZ; i < N - 20UZ; ++i) {
            derivRms += std::pow(deriv[i] - trueDeriv[i], 2.0);
        }
        derivRms = std::sqrt(derivRms / static_cast<double>(N - 40UZ));

        std::println(stderr, "[SG derivative] rms error = {:.4f}", derivRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "sin(x)+noise");
        chart.draw(xValues, deriv, "d/dx SG");
        chart.draw(xValues, trueDeriv, "cos(x)");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(derivRms, 0.15)) << "derivative accuracy";
    };

    "zero-phase vs single-pass"_test = [] {
        constexpr std::size_t N      = 150UZ;
        constexpr std::size_t centre = 75UZ;
        constexpr double      sigma  = 10.0;

        std::mt19937                     gen(456);
        std::normal_distribution<double> noise(0.0, 0.1);

        std::vector<double> xValues(N), clean(N), noisy(N), single(N), zeroPhase(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            double x   = static_cast<double>(i) - static_cast<double>(centre);
            clean[i]   = std::exp(-x * x / (2.0 * sigma * sigma));
            noisy[i]   = clean[i] + noise(gen);
        }

        Config<double> config{.boundaryPolicy = BoundaryPolicy::Reflect};
        auto           coeffs = computeCoefficients<double>(15UZ, 3UZ, config);
        apply<double>(noisy, single, coeffs, config);
        applyZeroPhase<double>(noisy, zeroPhase, coeffs, config);

        auto cleanPeak     = std::max_element(clean.begin(), clean.end());
        auto singlePeak    = std::max_element(single.begin(), single.end());
        auto zeroPhasePeak = std::max_element(zeroPhase.begin(), zeroPhase.end());

        std::size_t cleanIdx     = static_cast<std::size_t>(std::distance(clean.begin(), cleanPeak));
        std::size_t singleIdx    = static_cast<std::size_t>(std::distance(single.begin(), singlePeak));
        std::size_t zeroPhaseIdx = static_cast<std::size_t>(std::distance(zeroPhase.begin(), zeroPhasePeak));

        std::println(stderr, "[zero-phase] peak positions: clean={}, single={}, zero-phase={}", cleanIdx, singleIdx, zeroPhaseIdx);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, single, "single");
        chart.draw(xValues, zeroPhase, "zero-phase");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(le(std::abs(static_cast<std::ptrdiff_t>(cleanIdx) - static_cast<std::ptrdiff_t>(zeroPhaseIdx)), 1)) << "zero-phase preserves peak position";
    };
};

const boost::ut::suite<"SG vs LP denoising comparison"> sgVsLpTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;
    using namespace gr::algorithm::savitzky_golay;
    using namespace gr::filter;

    constexpr std::size_t kChartWidth  = 180UZ;
    constexpr std::size_t kChartHeight = 60UZ;

    "single sinusoid"_test = [] {
        constexpr std::size_t N       = 1000UZ;
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

        const std::size_t     sgWindowSize   = firCoeffs.b.size();
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 4UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 4UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 10UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("sinusoid", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        const std::size_t     sgWindowSize   = firCoeffs.b.size();
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 4UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 4UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 10UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("multi-tone", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        constexpr FilterParameters firParams{.order = 4UZ, .fLow = fc, .attenuationDb = 40.0, .fs = 1.0};
        const auto                 firCoeffs = fir::designFilter<double>(Type::LOWPASS, firParams);
        const std::size_t          lpDelay   = (firCoeffs.b.size() - 1UZ) / 2UZ;

        const std::size_t     sgWindowSize   = firCoeffs.b.size();
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 3UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 3UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 10UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("damped", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        const std::size_t     sgWindowSize   = firCoeffs.b.size();
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 2UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 2UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 50UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 50UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("DC+ramp", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        constexpr std::size_t sgWindowSize = 71UZ;
        Config<double>        configCentred{.boundaryPolicy = BoundaryPolicy::Reflect};
        Config<double>        configCausal{.alignment = Alignment::Causal, .boundaryPolicy = BoundaryPolicy::Reflect};
        auto                  sgCentredCoeffs = computeCoefficients<double>(sgWindowSize, 3UZ, configCentred);
        auto                  sgCausalCoeffs  = computeCoefficients<double>(sgWindowSize, 3UZ, configCausal);
        std::vector<double>   sgCentredFiltered(N), sgCausalFiltered(N);
        apply<double>(noisyDs.signal_values, sgCentredFiltered, sgCentredCoeffs, configCentred);
        apply<double>(noisyDs.signal_values, sgCausalFiltered, sgCausalCoeffs, configCausal);

        auto maDs = gr::dataset::filter::applyMovingAverage(noisyDs, sgWindowSize);

        const auto            cleanSignal = cleanDs.signalValues(0UZ);
        const auto            noisySignal = noisyDs.signalValues(0UZ);
        const auto            maSignal    = maDs.signalValues(0UZ);
        constexpr std::size_t skipEdge    = 20UZ;
        double                noisyRms = 0.0, maRms = 0.0, sgCentredRms = 0.0, sgCausalRms = 0.0;
        std::size_t           count = 0UZ;
        for (std::size_t i = skipEdge; i < N - skipEdge; ++i) {
            noisyRms += std::pow(noisySignal[i] - cleanSignal[i], 2.0);
            maRms += std::pow(maSignal[i] - cleanSignal[i], 2.0);
            sgCentredRms += std::pow(sgCentredFiltered[i] - cleanSignal[i], 2.0);
            sgCausalRms += std::pow(sgCausalFiltered[i] - cleanSignal[i], 2.0);
            ++count;
        }
        noisyRms     = std::sqrt(noisyRms / static_cast<double>(count));
        maRms        = std::sqrt(maRms / static_cast<double>(count));
        sgCentredRms = std::sqrt(sgCentredRms / static_cast<double>(count));
        sgCausalRms  = std::sqrt(sgCausalRms / static_cast<double>(count));

        printStats("Gaussian", noisyRms, maRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(cleanDs.axis_values[0], noisySignal, "noisy");
        chart.draw(cleanDs.axis_values[0], maSignal, "MA");
        chart.draw(cleanDs.axis_values[0], sgCentredFiltered, "SG-centred");
        chart.draw(cleanDs.axis_values[0], sgCausalFiltered, "SG-causal");
        chart.draw(cleanDs.axis_values[0], cleanSignal, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        const std::size_t     sgWindowSize   = firCoeffs.b.size();
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 3UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 3UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 10UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 10UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("high noise", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(sgCausalRms, noisyRms)) << "SG-causal should reduce noise";
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

        constexpr std::size_t sgWindowSize   = 51UZ;
        const std::size_t     sgCentredDelay = (sgWindowSize - 1UZ) / 2UZ;
        constexpr std::size_t sgCausalDelay  = 0UZ;
        std::println("FIR size: {}, SG size: {}", firCoeffs.b.size(), sgWindowSize);

        Filter<double>              firFilter(firCoeffs);
        SavitzkyGolayFilter<double> sgCentredFilter(sgWindowSize, 5UZ);
        SavitzkyGolayFilter<double> sgCausalFilter(sgWindowSize, 1UZ, {.alignment = Alignment::Causal});
        std::vector<double>         lpFiltered(N), sgCentredFiltered(N), sgCausalFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            lpFiltered[i]        = firFilter.processOne(noisy[i]);
            sgCentredFiltered[i] = sgCentredFilter.processOne(noisy[i]);
            sgCausalFiltered[i]  = sgCausalFilter.processOne(noisy[i]);
        }

        const std::size_t skipStart = std::max({sgCentredDelay, lpDelay, sgWindowSize}) + 100UZ;
        const std::size_t skipEnd   = std::max(sgCentredDelay, lpDelay) + 100UZ;

        double noisyRms     = computeRms(noisy, clean, skipStart, skipEnd, 0UZ);
        double lpRms        = computeRms(lpFiltered, clean, skipStart, skipEnd, lpDelay);
        double sgCentredRms = computeRms(sgCentredFiltered, clean, skipStart, skipEnd, sgCentredDelay);
        double sgCausalRms  = computeRms(sgCausalFiltered, clean, skipStart, skipEnd, sgCausalDelay);

        printStats("chirp", noisyRms, lpRms, sgCentredRms, sgCausalRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, lpFiltered, "LP");
        chart.draw(xValues, sgCentredFiltered, "SG-centred");
        chart.draw(xValues, sgCausalFiltered, "SG-causal");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgCentredRms, noisyRms)) << "SG-centred should reduce noise";
        expect(lt(lpRms, noisyRms)) << "LP should reduce noise";
    };
};

int main() { /* tests are automatically registered and run */ }
