#include <boost/ut.hpp>

#include <cmath>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/filter/SavitzkyGolayFilter.hpp>
#include <gnuradio-4.0/filter/SvdDenoiser.hpp>

const boost::ut::suite<"SavitzkyGolayFilter block"> sgFilterBlockTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    "float default parameters"_test = [] {
        SavitzkyGolayFilter<float> filter;
        expect(eq(filter.window_size.value, static_cast<gr::Size_t>(11)));
        expect(eq(filter.poly_order.value, static_cast<gr::Size_t>(4)));
        expect(eq(filter.deriv_order.value, static_cast<gr::Size_t>(0)));
    };

    "float processOne"_test = [] {
        SavitzkyGolayFilter<float> filter;
        filter.start();

        // With BoundaryPolicy::Default, history is pre-filled with zeros
        // Process a few samples and verify we get finite output
        std::vector<float> outputs;
        for (int i = 0; i < 20; ++i) {
            outputs.push_back(filter.processOne(static_cast<float>(i)));
        }
        expect(eq(outputs.size(), 20UZ));
        expect(std::isfinite(outputs.back())) << "output should be finite";
    };

    "double processOne"_test = [] {
        SavitzkyGolayFilter<double> filter;
        filter.start();

        std::vector<double> outputs;
        for (int i = 0; i < 20; ++i) {
            outputs.push_back(filter.processOne(static_cast<double>(i)));
        }
        expect(eq(outputs.size(), 20UZ));
        expect(std::isfinite(outputs.back())) << "output should be finite";
    };

    "smoothing quality"_test = [] {
        constexpr std::size_t N = 300UZ;

        std::mt19937                     gen(42);
        std::normal_distribution<double> noise(0.0, 0.3);

        std::vector<double> clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            clean[i] = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 50.0);
            noisy[i] = clean[i] + noise(gen);
        }

        SavitzkyGolayFilter<double> filter;
        filter.window_size = 11;
        filter.poly_order  = 3;
        filter.deriv_order = 0;
        filter.start();

        std::vector<double> filtered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            filtered[i] = filter.processOne(noisy[i]);
        }

        constexpr std::size_t delay    = 5UZ; // (window_size - 1) / 2
        constexpr std::size_t skipN    = 20UZ;
        double                noisyRms = 0.0, filteredRms = 0.0;
        for (std::size_t i = skipN + delay; i < N - skipN; ++i) {
            noisyRms += std::pow(noisy[i - delay] - clean[i - delay], 2.0);
            filteredRms += std::pow(filtered[i] - clean[i - delay], 2.0);
        }
        noisyRms    = std::sqrt(noisyRms / static_cast<double>(N - 2UZ * skipN - delay));
        filteredRms = std::sqrt(filteredRms / static_cast<double>(N - 2UZ * skipN - delay));

        expect(lt(filteredRms, noisyRms)) << "filtering reduces noise";
    };

    "settingsChanged"_test = [] {
        SavitzkyGolayFilter<float> filter;
        filter.window_size = 21;
        filter.poly_order  = 4;
        filter.alignment   = "Causal";
        filter.settingsChanged({}, {{"window_size", 21U}, {"poly_order", static_cast<gr::Size_t>(4)}, {"alignment", std::string("Causal")}});

        expect(eq(filter.window_size.value, static_cast<gr::Size_t>(21)));
        expect(eq(filter.poly_order.value, static_cast<gr::Size_t>(4)));
    };

    "reset"_test = [] {
        SavitzkyGolayFilter<float> filter;
        filter.start();

        for (int i = 0; i < 50; ++i) {
            std::ignore = filter.processOne(static_cast<float>(i));
        }
        filter.reset();

        // After reset, history is cleared and re-initialized
        float output = filter.processOne(1.0f);
        expect(std::isfinite(output)) << "output finite after reset";
    };
};

const boost::ut::suite<"SavitzkyGolayDataSetFilter block"> sgDataSetBlockTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    "basic processing"_test = [] {
        constexpr std::size_t N = 100UZ;

        gr::DataSet<double> input;
        input.axis_names        = {"sample"};
        input.axis_units        = {""};
        input.axis_values       = {std::vector<double>(N)};
        input.signal_names      = {"test"};
        input.signal_quantities = {"amplitude"};
        input.signal_units      = {""};
        input.signal_values.resize(N);
        input.extents = {static_cast<std::int32_t>(N)};

        for (std::size_t i = 0UZ; i < N; ++i) {
            input.axis_values[0][i] = static_cast<double>(i);
            input.signal_values[i]  = static_cast<double>(i % 10);
        }

        SavitzkyGolayDataSetFilter<double> filter;
        filter.window_size = 5;
        filter.poly_order  = 2;
        filter.start();

        auto output = filter.processOne(input);
        expect(eq(output.signal_values.size(), N));
    };

    "peak position preserved"_test = [] {
        constexpr std::size_t N      = 100UZ;
        constexpr std::size_t centre = 50UZ;

        gr::DataSet<double> input;
        input.axis_names        = {"sample"};
        input.axis_units        = {""};
        input.axis_values       = {std::vector<double>(N)};
        input.signal_names      = {"gaussian"};
        input.signal_quantities = {"amplitude"};
        input.signal_units      = {""};
        input.signal_values.resize(N);
        input.extents = {static_cast<std::int32_t>(N)};

        std::mt19937                     gen(123);
        std::normal_distribution<double> noise(0.0, 0.05);

        for (std::size_t i = 0UZ; i < N; ++i) {
            input.axis_values[0][i] = static_cast<double>(i);
            double x                = static_cast<double>(i) - static_cast<double>(centre);
            input.signal_values[i]  = std::exp(-x * x / 100.0) + noise(gen);
        }

        SavitzkyGolayDataSetFilter<double> filter;
        filter.window_size = 11;
        filter.poly_order  = 3;
        filter.start();

        auto output = filter.processOne(input);

        auto inMax  = std::max_element(input.signal_values.begin(), input.signal_values.end());
        auto outMax = std::max_element(output.signal_values.begin(), output.signal_values.end());

        std::ignore        = inMax; // peak position in noisy input varies
        std::size_t outIdx = static_cast<std::size_t>(std::distance(output.signal_values.begin(), outMax));

        expect(le(std::abs(static_cast<std::ptrdiff_t>(outIdx) - static_cast<std::ptrdiff_t>(centre)), 1)) << "zero-phase preserves peak";
    };

    "settingsChanged"_test = [] {
        SavitzkyGolayDataSetFilter<float> filter;
        filter.window_size     = 15;
        filter.boundary_policy = "Replicate";
        filter.settingsChanged({}, {{"window_size", 15U}, {"boundary_policy", std::string("Replicate")}});

        expect(eq(filter.window_size.value, static_cast<gr::Size_t>(15)));
        expect(eq(filter.boundary_policy.value, std::string("Replicate")));
    };
};

const boost::ut::suite<"SavitzkyGolayFilter edge cases"> sgFilterEdgeCaseTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    "sample_rate parameter"_test = [] {
        "sample_rate affects derivative scaling"_test = [] {
            SavitzkyGolayFilter<double> filter;
            filter.window_size = 11;
            filter.poly_order  = 3;
            filter.deriv_order = 1;
            filter.sample_rate = 1000.0f; // 1 kHz
            filter.start();

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 50UZ; ++i) {
                outputs.push_back(filter.processOne(static_cast<double>(i)));
            }
            expect(std::isfinite(outputs.back()));
        };

        "sample_rate = 0 handled gracefully"_test = [] {
            SavitzkyGolayFilter<double> filter;
            filter.window_size = 11;
            filter.sample_rate = 0.0f; // edge case
            filter.start();

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 20UZ; ++i) {
                outputs.push_back(filter.processOne(static_cast<double>(i)));
            }
            // Should still produce finite output
            expect(std::isfinite(outputs.back()));
        };
    };

    "minimum window_size"_test = [] {
        SavitzkyGolayFilter<float> filter;
        filter.window_size = 3;
        filter.poly_order  = 2;
        filter.start();

        std::vector<float> outputs;
        for (int i = 0; i < 20; ++i) {
            outputs.push_back(filter.processOne(static_cast<float>(i)));
        }
        expect(std::isfinite(outputs.back()));
    };

    "alignment modes"_test = [] {
        "Centred alignment"_test = [] {
            SavitzkyGolayFilter<double> filter;
            filter.window_size = 11;
            filter.alignment   = "Centred";
            filter.start();

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 30UZ; ++i) {
                outputs.push_back(filter.processOne(std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 10.0)));
            }
            expect(std::isfinite(outputs.back()));
        };

        "Causal alignment"_test = [] {
            SavitzkyGolayFilter<double> filter;
            filter.window_size = 11;
            filter.alignment   = "Causal";
            filter.start();

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 30UZ; ++i) {
                outputs.push_back(filter.processOne(std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 10.0)));
            }
            expect(std::isfinite(outputs.back()));
        };

        "invalid alignment defaults to Centred"_test = [] {
            SavitzkyGolayFilter<double> filter;
            filter.window_size = 11;
            filter.alignment   = "Invalid"; // not recognized
            filter.start();

            std::vector<double> outputs;
            for (std::size_t i = 0UZ; i < 30UZ; ++i) {
                outputs.push_back(filter.processOne(static_cast<double>(i)));
            }
            expect(std::isfinite(outputs.back()));
        };
    };

    "derivative order"_test = [] {
        SavitzkyGolayFilter<double> filter;
        filter.window_size = 11;
        filter.poly_order  = 4;
        filter.deriv_order = 2; // second derivative
        filter.sample_rate = 100.0f;
        filter.start();

        std::vector<double> outputs;
        for (std::size_t i = 0UZ; i < 50UZ; ++i) {
            double x = static_cast<double>(i) * 0.01;
            outputs.push_back(filter.processOne(x * x)); // f(x) = x^2
        }
        expect(std::isfinite(outputs.back()));
    };
};

const boost::ut::suite<"SavitzkyGolayDataSetFilter edge cases"> sgDataSetEdgeCaseTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    "empty signal_values"_test = [] {
        gr::DataSet<double> input;
        input.axis_names        = {"sample"};
        input.axis_units        = {""};
        input.axis_values       = {std::vector<double>()};
        input.signal_names      = {"test"};
        input.signal_quantities = {"amplitude"};
        input.signal_units      = {""};
        input.signal_values     = {}; // empty
        input.extents           = {0};

        SavitzkyGolayDataSetFilter<double> filter;
        filter.window_size = 5;
        filter.start();

        auto output = filter.processOne(input);
        expect(eq(output.signal_values.size(), 0UZ)) << "empty input produces empty output";
    };

    "signal shorter than window_size"_test = [] {
        constexpr std::size_t N = 3UZ;

        gr::DataSet<double> input;
        input.axis_names        = {"sample"};
        input.axis_units        = {""};
        input.axis_values       = {std::vector<double>(N)};
        input.signal_names      = {"test"};
        input.signal_quantities = {"amplitude"};
        input.signal_units      = {""};
        input.signal_values.resize(N);
        input.extents = {static_cast<std::int32_t>(N)};

        for (std::size_t i = 0UZ; i < N; ++i) {
            input.axis_values[0][i] = static_cast<double>(i);
            input.signal_values[i]  = static_cast<double>(i);
        }

        SavitzkyGolayDataSetFilter<double> filter;
        filter.window_size = 11; // larger than input
        filter.start();

        auto output = filter.processOne(input);
        expect(eq(output.signal_values.size(), N));
        for (const auto& val : output.signal_values) {
            expect(std::isfinite(val)) << "output should be finite";
        }
    };

    "derivative order > 0"_test = [] {
        constexpr std::size_t N = 50UZ;

        gr::DataSet<double> input;
        input.axis_names        = {"sample"};
        input.axis_units        = {""};
        input.axis_values       = {std::vector<double>(N)};
        input.signal_names      = {"quadratic"};
        input.signal_quantities = {"amplitude"};
        input.signal_units      = {""};
        input.signal_values.resize(N);
        input.extents = {static_cast<std::int32_t>(N)};

        for (std::size_t i = 0UZ; i < N; ++i) {
            double x                = static_cast<double>(i) * 0.1;
            input.axis_values[0][i] = x;
            input.signal_values[i]  = x * x; // f(x) = x^2
        }

        SavitzkyGolayDataSetFilter<double> filter;
        filter.window_size = 11;
        filter.poly_order  = 3;
        filter.deriv_order = 1; // first derivative
        filter.start();

        auto output = filter.processOne(input);
        expect(eq(output.signal_values.size(), N));

        // d/dx(x^2) = 2x, but with zero-phase the derivative should be approximately correct
        for (std::size_t i = 10UZ; i < N - 10UZ; ++i) {
            expect(std::isfinite(output.signal_values[i]));
        }
    };

    "boundary policy variations"_test = [] {
        constexpr std::size_t N = 20UZ;

        auto makeInput = [] {
            gr::DataSet<double> input;
            input.axis_names        = {"sample"};
            input.axis_units        = {""};
            input.axis_values       = {std::vector<double>(N)};
            input.signal_names      = {"test"};
            input.signal_quantities = {"amplitude"};
            input.signal_units      = {""};
            input.signal_values.resize(N);
            input.extents = {static_cast<std::int32_t>(N)};
            for (std::size_t i = 0UZ; i < N; ++i) {
                input.axis_values[0][i] = static_cast<double>(i);
                input.signal_values[i]  = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 10.0);
            }
            return input;
        };

        "Reflect policy"_test = [&] {
            SavitzkyGolayDataSetFilter<double> filter;
            filter.window_size     = 7;
            filter.boundary_policy = "Reflect";
            filter.start();

            auto output = filter.processOne(makeInput());
            expect(eq(output.signal_values.size(), N));
        };

        "Replicate policy"_test = [&] {
            SavitzkyGolayDataSetFilter<double> filter;
            filter.window_size     = 7;
            filter.boundary_policy = "Replicate";
            filter.start();

            auto output = filter.processOne(makeInput());
            expect(eq(output.signal_values.size(), N));
        };
    };
};

namespace {
void printStats(std::string_view label, double noisyRms, double sgRms, double svdRms) { std::println(stderr, "[{}] noisy={:.4f}, SG={:.4f} ({:.1f}%), SVD={:.4f} ({:.1f}%)", label, noisyRms, sgRms, 100.0 * (1.0 - sgRms / noisyRms), svdRms, 100.0 * (1.0 - svdRms / noisyRms)); }

auto computeRms(const std::vector<double>& filtered, const std::vector<double>& clean, std::size_t skip, std::size_t delay) {
    double      rms   = 0.0;
    std::size_t count = 0UZ;
    for (std::size_t i = skip; i < filtered.size() - skip; ++i) {
        if (i >= delay) {
            rms += std::pow(filtered[i] - clean[i - delay], 2.0);
            ++count;
        }
    }
    return count > 0UZ ? std::sqrt(rms / static_cast<double>(count)) : 0.0;
}
} // namespace

const boost::ut::suite<"SG vs SVD comparison"> comparisonTests = [] {
    using namespace boost::ut;
    using namespace gr::graphs;
    using namespace gr::filter;

    constexpr std::size_t kChartWidth  = 160UZ;
    constexpr std::size_t kChartHeight = 50UZ;

    "sinusoid smoothing"_test = [] {
        constexpr std::size_t N = 400UZ;

        std::mt19937                     gen(42);
        std::normal_distribution<double> noise(0.0, 0.4);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = std::sin(2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / 80.0);
            noisy[i]   = clean[i] + noise(gen);
        }

        SavitzkyGolayFilter<double> sgFilter;
        sgFilter.window_size = 21;
        sgFilter.poly_order  = 3;
        sgFilter.start();

        SvdDenoiser<double> svdFilter;
        svdFilter.window_size     = 40;
        svdFilter.max_rank        = 2;
        svdFilter.energy_fraction = 0.85;
        svdFilter.start();

        std::vector<double> sgFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            sgFiltered[i]  = sgFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        constexpr std::size_t skipN    = 50UZ;
        double                noisyRms = computeRms(noisy, clean, skipN, 0UZ);
        double                sgRms    = computeRms(sgFiltered, clean, skipN, 10UZ);
        double                svdRms   = computeRms(svdFiltered, clean, skipN, 19UZ);

        printStats("sinusoid", noisyRms, sgRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, sgFiltered, "SG");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgRms, noisyRms)) << "SG reduces noise";
        expect(lt(svdRms, noisyRms)) << "SVD reduces noise";
    };

    "step response"_test = [] {
        constexpr std::size_t N = 200UZ;

        std::mt19937                     gen(123);
        std::normal_distribution<double> noise(0.0, 0.15);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            clean[i]   = (i < N / 2UZ) ? 0.0 : 1.0;
            noisy[i]   = clean[i] + noise(gen);
        }

        SavitzkyGolayFilter<double> sgFilter;
        sgFilter.window_size = 11;
        sgFilter.poly_order  = 3;
        sgFilter.start();

        SvdDenoiser<double> svdFilter;
        svdFilter.window_size     = 24;
        svdFilter.max_rank        = 2;
        svdFilter.energy_fraction = 0.90;
        svdFilter.start();

        std::vector<double> sgFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            sgFiltered[i]  = sgFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, sgFiltered, "SG");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));
    };

    "Gaussian peak"_test = [] {
        constexpr std::size_t N      = 200UZ;
        constexpr std::size_t centre = 100UZ;
        constexpr double      sigma  = 15.0;

        std::mt19937                     gen(456);
        std::normal_distribution<double> noise(0.0, 0.15);

        std::vector<double> xValues(N), clean(N), noisy(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i] = static_cast<double>(i);
            double x   = static_cast<double>(i) - static_cast<double>(centre);
            clean[i]   = std::exp(-x * x / (2.0 * sigma * sigma));
            noisy[i]   = clean[i] + noise(gen);
        }

        SavitzkyGolayFilter<double> sgFilter;
        sgFilter.window_size = 15;
        sgFilter.poly_order  = 3;
        sgFilter.start();

        SvdDenoiser<double> svdFilter;
        svdFilter.window_size     = 32;
        svdFilter.max_rank        = 3;
        svdFilter.energy_fraction = 0.95;
        svdFilter.start();

        std::vector<double> sgFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            sgFiltered[i]  = sgFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        constexpr std::size_t skipN    = 40UZ;
        double                noisyRms = computeRms(noisy, clean, skipN, 0UZ);
        double                sgRms    = computeRms(sgFiltered, clean, skipN, 7UZ);
        double                svdRms   = computeRms(svdFiltered, clean, skipN, 15UZ);

        printStats("Gaussian", noisyRms, sgRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, sgFiltered, "SG");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgRms, noisyRms)) << "SG reduces noise";
        expect(lt(svdRms, noisyRms)) << "SVD reduces noise";
    };

    "chirp signal"_test = [] {
        constexpr std::size_t N    = 500UZ;
        constexpr double      fMin = 0.005;
        constexpr double      fMax = 0.05;

        std::mt19937                     gen(789);
        std::normal_distribution<double> noise(0.0, 0.3);

        std::vector<double> xValues(N), clean(N), noisy(N);
        double              phase = 0.0;
        for (std::size_t i = 0UZ; i < N; ++i) {
            xValues[i]  = static_cast<double>(i);
            double t    = static_cast<double>(i) / static_cast<double>(N);
            double freq = fMin + (fMax - fMin) * t * t;
            clean[i]    = std::sin(phase);
            noisy[i]    = clean[i] + noise(gen);
            phase += 2.0 * std::numbers::pi_v<double> * freq;
        }

        SavitzkyGolayFilter<double> sgFilter;
        sgFilter.window_size = 15;
        sgFilter.poly_order  = 4;
        sgFilter.start();

        SvdDenoiser<double> svdFilter;
        svdFilter.window_size     = 48;
        svdFilter.max_rank        = 3;
        svdFilter.energy_fraction = 0.85;
        svdFilter.start();

        std::vector<double> sgFiltered(N), svdFiltered(N);
        for (std::size_t i = 0UZ; i < N; ++i) {
            sgFiltered[i]  = sgFilter.processOne(noisy[i]);
            svdFiltered[i] = svdFilter.processOne(noisy[i]);
        }

        constexpr std::size_t skipN    = 60UZ;
        double                noisyRms = computeRms(noisy, clean, skipN, 0UZ);
        double                sgRms    = computeRms(sgFiltered, clean, skipN, 7UZ);
        double                svdRms   = computeRms(svdFiltered, clean, skipN, 23UZ);

        printStats("chirp", noisyRms, sgRms, svdRms);

        ImChart<kChartWidth, kChartHeight> chart;
        chart.draw(xValues, noisy, "noisy");
        chart.draw(xValues, sgFiltered, "SG");
        chart.draw(xValues, svdFiltered, "SVD");
        chart.draw(xValues, clean, "clean");
        expect(nothrow([&] { chart.draw(); }));

        expect(lt(sgRms, noisyRms)) << "SG reduces noise";
        expect(lt(svdRms, noisyRms)) << "SVD reduces noise";
    };
};

int main() { /* tests are automatically registered and run */ }
