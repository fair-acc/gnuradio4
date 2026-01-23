#include <boost/ut.hpp>
#include <cmath>
#include <complex>
#include <numbers>

#include <gnuradio-4.0/math/Rotator.hpp>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {

template<typename T>
std::vector<std::complex<T>> execRotator(const std::vector<std::complex<T>>& input, const gr::property_map& initSettings) {
    gr::blocks::math::Rotator<std::complex<T>> rot(initSettings);
    rot.settings().init();
    std::ignore = rot.settings().applyStagedParameters(); // needed for unit-test only when executed outside a Scheduler/Graph

    std::vector<std::complex<T>> output(input.size());
    for (std::size_t i = 0; i < input.size(); i++) {
        output[i] = rot.processOne(input[i]);
    }
    return output;
}

template<typename T>
void plotTimeDomain(const std::vector<std::complex<T>>& dataIn, const std::vector<std::complex<T>>& dataOut, float fs, const std::string& label) {
    std::vector<float> time(dataOut.size());
    for (std::size_t i = 0UZ; i < dataOut.size(); i++) {
        time[i] = static_cast<float>(i) / fs;
    }

    std::vector<float> inRe(dataOut.size());
    std::vector<float> inIm(dataOut.size());
    std::vector<float> outRe(dataOut.size());
    std::vector<float> outIm(dataOut.size());
    for (std::size_t i = 0UZ; i < dataOut.size(); i++) {
        inRe[i]  = static_cast<float>(dataIn[i].real());
        inIm[i]  = static_cast<float>(dataIn[i].imag());
        outRe[i] = static_cast<float>(dataOut[i].real());
        outIm[i] = static_cast<float>(dataOut[i].imag());
    }

    // quick chart
    gr::graphs::ImChart<100, 15> chart({{0.0f, time.back()}, {-1.5f, +1.5f}});
    chart.axis_name_x = "Time [s]";
    chart.axis_name_y = "Amplitude [a.u.]";

    chart.draw(time, inRe, "Re(in)");
    chart.draw(time, inIm, "Im(in)");
    chart.draw(time, outRe, std::format("out: Re({})", label));
    chart.draw(time, outIm, std::format("out: Im({})", label));
    chart.draw();
}

} // end anonymous namespace

const boost::ut::suite<"basic math tests"> basicMath = [] {
    using namespace boost::ut;
    using namespace gr::blocks::math;

    constexpr auto kArithmeticTypes = std::tuple<std::complex<float>, std::complex<double>>{};

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = {"visual", "benchmarks"}};
    }

    "Rotator - basic test"_test = []<typename T> {
        using value_t          = typename T::value_type;
        value_t    phase_shift = std::numbers::pi_v<value_t> / value_t(2);
        Rotator<T> rot({{"phase_increment", gr::pmt::Value(phase_shift)}, {"initial_phase", gr::pmt::Value(value_t(0))}, {"sample_rate", gr::pmt::Value(1.f)}});
        rot.settings().init();
        std::ignore = rot.settings().applyStagedParameters(); // needed for unit-test only when executed outside a Scheduler/Graph

        expect(approx(rot.frequency_shift, 0.25f, 1e-3f));
        expect(approx(rot.initial_phase, value_t(0), value_t(1e-3f)));

        std::vector<T> output(8UZ);
        for (std::size_t i = 0; i < 8; i++) {
            output[i] = rot.processOne(std::complex<value_t>(1, 0));
        }

        for (std::size_t i = 0; i < 8; i++) {
            value_t wantAngle = value_t(i + 1) * phase_shift;
            value_t wantCos   = std::cos(wantAngle);
            value_t wantSin   = std::sin(wantAngle);

            expect(approx(output[i].real(), wantCos, value_t(1e-5))) << "rotator real mismatch i=" << i;
            expect(approx(output[i].imag(), wantSin, value_t(1e-5))) << "rotator imag mismatch i=" << i;
        }
    } | kArithmeticTypes;

    constexpr static float fs    = 100.0; // sampling rate
    constexpr static float tMax  = 2.0;   // seconds
    constexpr static auto  nSamp = static_cast<std::size_t>(fs * tMax);

    tag("visual") / "RotatorTest - DC->2 Hz shift"_test = [] {
        std::vector<std::complex<double>> input(nSamp, std::complex<double>(std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0));
        auto                              output = execRotator(input, {{"frequency_shift", gr::pmt::Value(+2.f)}, {"sample_rate", gr::pmt::Value(fs)}});
        plotTimeDomain(input, output, fs, "DC->+2 Hz");
    };

    tag("visual") / "RotatorTest - 0.5 Hz => shift +1.5 => 2 Hz"_test = [] {
        std::vector<std::complex<double>> input(nSamp);
        for (std::size_t i = 0; i < nSamp; i++) { // 0.5 Hz complex sinusoid
            double t     = static_cast<double>(i) / static_cast<double>(fs);
            double angle = 2.0 * std::numbers::pi * 0.5 * t; // 0.5 Hz
            input[i]     = {std::cos(angle), std::sin(angle)};
        }
        auto output = execRotator(input, {{"frequency_shift", gr::pmt::Value(+1.5f)}, {"sample_rate", gr::pmt::Value(fs)}});
        plotTimeDomain(input, output, fs, ".5->2 Hz");
    };

    tag("visual") / "RotatorTest - 2 Hz => shift -1.5 => 0.5 Hz"_test = [] {
        std::vector<std::complex<double>> input(nSamp);
        for (std::size_t i = 0; i < nSamp; i++) { // 2 Hz complex sinusoid
            double t     = static_cast<double>(i) / static_cast<double>(fs);
            double angle = 2.0 * std::numbers::pi * 2.0 * t; // 2 Hz
            input[i]     = {std::cos(angle), std::sin(angle)};
        }
        auto output = execRotator(input, {{"frequency_shift", gr::pmt::Value(-1.5f)}, {"sample_rate", gr::pmt::Value(fs)}});
        plotTimeDomain(input, output, fs, "2->.5 Hz");
    };
};

int main() { /* not needed for UT */ }
