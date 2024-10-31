#include <gnuradio-4.0/basic/function_generator.hpp>

int main() {
    using namespace gr::basic;
    const double f_s = 1000.0; // 1 kHz
    Generator    generator(f_s);

    // For Constant Function
    generator.setMode(FunctionMode::Constant, 0.5, 0.5, 0.1);
    for (int i = 0; i < 100; i++) {
        std::cout << generator.getNextSample() << std::endl;
    }

    // For Linear Ramp
    generator.setMode(FunctionMode::LinearRamp, 0, 1, 0.1);
    for (int i = 0; i < 100; i++) {
        std::cout << generator.getNextSample() << std::endl;
    }

    // For Parabolic Ramp
    generator.setMode(FunctionMode::ParabolicRamp, 0, 1, 0.1, 0.02);
    for (int i = 0; i < 100; i++) {
        std::cout << generator.getNextSample() << std::endl;
    }

    // For Cubic Spline
    generator.setMode(FunctionMode::CubicSpline, 0, 1, 0.1);
    for (int i = 0; i < 100; i++) {
        std::cout << generator.getNextSample() << std::endl;
    }

    // For Impulse Response
    generator.setMode(FunctionMode::ImpulseResponse, 0, 1, 0.1, 0, 0.02, 0.06);
    for (int i = 0; i < 100; i++) {
        std::cout << generator.getNextSample() << std::endl;
    }

    return 0;
}
