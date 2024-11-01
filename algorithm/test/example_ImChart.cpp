#include <gnuradio-4.0/algorithm/ImChart.hpp>

#include <chrono>
#include <cmath>
#include <locale>
#include <ranges>
#include <thread>

int main() {
    setlocale(LC_ALL, "");
    constexpr std::size_t sizeX = 121;
    constexpr std::size_t sizeY = 41;
    constexpr double      xMin  = 0.0;
    constexpr double      xMax  = 100.0;
    constexpr double      yMin  = -5.0;
    constexpr double      yMax  = +5.0;

    using namespace gr::graphs;
    auto animator = ImChart<sizeX, sizeY>({{xMin, xMax}, {yMin, yMax}});
    // auto animator = ImChart<sizeX, sizeY>();
    double phase = 0.;

    while (true) {
        animator.clearScreen();

        // Drawing Linear, Sine, and Cosine curves
        std::vector<double> xValues(2000);
        std::vector<double> gauss1(xValues.size());
        std::vector<double> gauss2(xValues.size());
        std::vector<double> sineYValues(xValues.size());
        std::vector<double> cosineYValues(xValues.size());

        for (std::size_t i = 0; i < xValues.size(); ++i) {
            xValues[i]              = xMin + (xMax - xMin) * static_cast<double>(i) / static_cast<double>(xValues.size());
            constexpr double mu1    = 55.0;
            constexpr double sigma1 = 5.0;
            gauss1[i]               = 2.0 * std::exp(-std::pow(xValues[i] - mu1, 2.) / (2.0 * std::pow(sigma1, 2.))); // Gaussian function
            constexpr double mu2    = 70.0;
            constexpr double sigma2 = 10.0;
            gauss2[i]               = (0.5 + 0.5 * std::cos(0.5 - 5 * phase)) * std::exp(-std::pow(xValues[i] - mu2, 2.) / (2.0 * std::pow(sigma2, 2.)));
            sineYValues[i]          = 3.0 * std::sin(xValues[i] * 0.2 - phase);
            cosineYValues[i]        = 3.0 * std::cos(xValues[i] * 0.2 - phase);
        }

        animator.draw(xValues, sineYValues, "sine-like");
        animator.draw(xValues, cosineYValues, "cosine-like");
        animator.draw<Style::Bars>(xValues, gauss1, "gauss-like1");
        animator.draw<Style::Bars>(xValues, gauss2, "gauss-like2");

        animator.drawBorder();
        animator.drawAxes();
        animator.drawLegend();

        animator.printSourceLocation();
        animator.reset(); // reset to screen origin
        animator.printScreen();
        phase += 0.01;
        using namespace std::chrono_literals;

        std::this_thread::sleep_for(40ms);
    }

    return 0;
}
