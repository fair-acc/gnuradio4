#include <boost/ut.hpp>

#include <random>

#include <format>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>

const boost::ut::suite<"ImChart"> windowTests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;

    "linear axis transform"_test = [](const double valueCoordinate) {
        using namespace gr::graphs;
        constexpr std::size_t axisOffset = 5UZ;
        constexpr std::size_t axisWidth  = 65UZ;
        constexpr double      xMin       = 10.;
        constexpr double      xMax       = 100.;
        //
        auto identity = [](double value) {
            const auto screenCoordinate = LinearAxisTransform::toScreen(value, xMin, xMax, axisOffset, axisWidth);
            return LinearAxisTransform::fromScreen(screenCoordinate, xMin, xMax, axisOffset, axisWidth);
        };
        expect(approx(valueCoordinate, identity(valueCoordinate), 2.2)); // binning limited

        expect(eq(axisOffset, LinearAxisTransform::toScreen(xMin, xMin, xMax, axisOffset, axisWidth))) << "xMin does not correspond to min axis index";
        expect(eq(axisWidth - 1UZ, LinearAxisTransform::toScreen(xMax, xMin, xMax, axisOffset, axisWidth))) << "xMax does not correspond to max axis index";
    } | std::vector{10., 20., 42., 50., 100.};

    "log axis transform"_test = [](const double valueCoordinate) {
        using namespace gr::graphs;
        constexpr std::size_t axisOffset = 5UZ;
        constexpr std::size_t axisWidth  = 65UZ;
        constexpr double      xMin       = 10.;
        constexpr double      xMax       = 100.;
        //
        auto identity = [](double value) {
            const auto screenCoordinate = LogAxisTransform::toScreen(value, xMin, xMax, axisOffset, axisWidth);
            return LogAxisTransform::fromScreen(screenCoordinate, xMin, xMax, axisOffset, axisWidth);
        };
        expect(approx(valueCoordinate, identity(valueCoordinate), 2.2)); // binning limited

        expect(eq(axisOffset, LogAxisTransform::toScreen(xMin, xMin, xMax, axisOffset, axisWidth))) << "xMin does not correspond to min axis index";
        expect(eq(axisWidth - 1UZ, LogAxisTransform::toScreen(xMax, xMin, xMax, axisOffset, axisWidth))) << "xMax does not correspond to max axis index";

        expect(throws([] { std::ignore = LogAxisTransform::toScreen(0.0, 10., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::toScreen(1.0, 0., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::toScreen(1.0, 10., 0., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::fromScreen(40, 0., 100., 5UZ, 65UZ); }));
        expect(throws([] { std::ignore = LogAxisTransform::fromScreen(40, 10., 0., 5UZ, 65UZ); }));
    } | std::vector{10., 20., 42., 50., 100.};

    "optimal tick position"_test = [](std::size_t axisWidth) {
        for (std::size_t minGapSize : {1UZ, 2UZ, 3UZ}) {
            auto tickPositions = gr::graphs::detail::optimalTickScreenPositions(axisWidth, minGapSize);

            // first and last tick index fulfill [0, axisWidth - 1].
            expect(!tickPositions.empty());
            expect(ge(tickPositions.size(), 2UZ));
            expect(eq(tickPositions.front(), 0UZ));
            expect(eq(tickPositions.back(), axisWidth - 1UZ));

            // validate even spacing and minimum gap, if there are more than two ticks.
            if (tickPositions.size() == 2UZ) {
                continue;
            }
            auto expectedGap = tickPositions[1UZ] - tickPositions[0UZ];
            for (size_t i = 1; i < tickPositions.size() - 1; ++i) {
                auto actualGap = tickPositions[i + 1] - tickPositions[i];
                expect(eq(actualGap, expectedGap)) << std::format("uneven spacing {} exp. {} found at axis width {}", actualGap, expectedGap, axisWidth);
                expect(ge(actualGap, minGapSize)) << std::format("gap size {} less than minimum required {} at axis width {}", actualGap, minGapSize, axisWidth);
            }
        }
    } | std::vector{10UZ, 20UZ, 42UZ, 50UZ, 100UZ};

    [[maybe_unused]] constexpr std::size_t sizeX = 120;
    [[maybe_unused]] constexpr std::size_t sizeY = 16;
    [[maybe_unused]] constexpr double      xMin  = 0.0;
    [[maybe_unused]] constexpr double      xMax  = 100.0;
    [[maybe_unused]] constexpr double      yMin  = -5.0;
    [[maybe_unused]] constexpr double      yMax  = +5.0;

    "basic chart - lines"_test = [&](bool defaultConstructor) {
        using namespace gr::graphs;

        // drawing sine, and cosine curves
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());
        std::vector<double> cosineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]       = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i]   = 3.0 * std::sin(xValues[i] * 0.2);
            cosineYValues[i] = 3.0 * std::cos(xValues[i] * 0.2);
        }

        auto chart = defaultConstructor ? gr::graphs::ImChart<sizeX, sizeY>() : gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {yMin, yMax}});

        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));
        expect(nothrow([&] { chart.draw(xValues, cosineYValues, "cosine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    } | std::vector{true, false};

    "basic chart - no data"_test = [&]() {
        using namespace gr::graphs;
        auto chart = gr::graphs::ImChart<sizeX, sizeY>();
        expect(nothrow([&] { chart.draw(std::vector<double>(), std::vector<double>(), "sine-like"); }));
    };

    "basic chart - bars"_test = [&]() {
        using namespace gr::graphs;

        // drawing gauss curves
        std::vector<double> xValues(2000);
        std::vector<double> gauss1(xValues.size());
        std::vector<double> gauss2(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]              = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            constexpr double mu1    = 55.0;
            constexpr double sigma1 = 5.0;
            gauss1[i]               = 3.0 * std::exp(-std::pow(xValues[i] - mu1, 2.) / (2.0 * std::pow(sigma1, 2.))); // Gaussian function
            constexpr double mu2    = 70.0;
            constexpr double sigma2 = 10.0;
            gauss2[i]               = 2.0 * std::exp(-std::pow(xValues[i] - mu2, 2.) / (2.0 * std::pow(sigma2, 2.)));
        }

        auto chart        = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {0.0, 5.0}});
        chart.draw_border = true;
        expect(nothrow([&] { chart.draw<Style::Bars>(xValues, gauss1, "gauss-like1"); }));
        expect(nothrow([&] { chart.draw<Style::Bars>(xValues, gauss2, "gauss-like2"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - drawing markers"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {yMin, yMax}});
        expect(nothrow([&] { chart.draw<Style::Marker>(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - axis top"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = -5.0 + 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{xMin, xMax}, {-10, 0}});
        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - axis top/centre"_test = [&]() {
        using namespace gr::graphs;

        // drawing sine curve
        std::vector<double> xValues(2000);
        std::vector<double> sineYValues(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i]     = -50. + 100. * static_cast<double>(i) / static_cast<double>(xValues.size());
            sineYValues[i] = -5.0 + 3.0 * std::sin(xValues[i] * 0.2);
        }

        auto chart = gr::graphs::ImChart<sizeX, sizeY>({{-50, +50}, {-10, 0}});
        expect(nothrow([&] { chart.draw(xValues, sineYValues, "sine-like"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "basic chart - log axis"_test = []() {
        using namespace gr::graphs;
        constexpr std::size_t width = 120;

        auto sequence_range = [](auto start, auto end, auto step) {
            using namespace std::views; // for iota and transform
            return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
        };

        std::vector<double> xValues;
        for (auto subrange : {sequence_range(0.1, 0.9, 0.1), sequence_range(1.0, 9.0, 1.0), sequence_range(10.0, 90.0, 10.0), sequence_range(100.0, 1000.0, 100.0)}) {
            std::ranges::move(subrange, std::back_inserter(xValues));
        }
        std::vector<double> response1(xValues.size());
        std::vector<double> response2(xValues.size());

        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            response1[i] = -5. + 20. * std::log10(1.0 / (1.0 + std::pow(2.0, xValues[i] / 10)));
            response2[i] = -10. + 20. * std::log10(1.0 / (1.0 + std::pow(4.0, xValues[i] / 200.)));
            if (xValues[i] > 5000) {
                response1[i] = -10.;
                response2[i] = -10.;
            }
        }

        auto chart = gr::graphs::ImChart<width, 16, LogAxisTransform>({{0.1, 1'000.0}, {-100, 0}});
        expect(nothrow([&] { chart.draw(xValues, response1, "low-pass1"); }));
        expect(nothrow([&] { chart.draw(xValues, response2, "low-pass2"); }));

        expect(nothrow([&] { chart.draw(); }));
    };

    "DataSet chart"_test = []() {
        using namespace gr::dataset;
        constexpr std::size_t kLength       = 1024UZ;
        constexpr float       kSamplingRate = 1000.f;
        constexpr float       kFrequency    = 5.f;
        constexpr float       kAmplitude    = 1.f;
        constexpr float       kOffset       = 0.2f;

        auto sinDataSet    = generate::waveform<float>(generate::WaveType::Sine, kLength, kSamplingRate, kFrequency, kAmplitude, +kOffset);
        auto cosDataSet    = generate::waveform<float>(generate::WaveType::Cosine, kLength, kSamplingRate, kFrequency, kAmplitude, -kOffset);
        auto mergedDataSet = merge(sinDataSet, cosDataSet);

        expect(eq(sinDataSet.signal_values.size(), kLength));
        expect(eq(cosDataSet.signal_values.size(), kLength));
        expect(eq(mergedDataSet.signal_values.size(), 2UZ * kLength));

        draw(sinDataSet);
        draw(cosDataSet);
        draw(mergedDataSet);
    };
};

int main() { /* not needed for UT */ }
