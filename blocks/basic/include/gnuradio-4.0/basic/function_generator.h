#ifndef GRAPH_PROTOTYPE_DEMO_FUNCTION_GENERATOR_HPP
#define GRAPH_PROTOTYPE_DEMO_FUNCTION_GENERATOR_HPP

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// prototype: https://compiler-explorer.com/z/GnPsKcven

enum class FunctionMode { Constant, LinearRamp, ParabolicRamp, CubicSpline, ImpulseResponse };

struct FunctionGenerator {
    double         T_s{ 1 };
    mutable double time{ 0 };

    explicit FunctionGenerator(double T_s) : T_s(T_s) {}

    virtual ~FunctionGenerator() = default;

    virtual double
    getSample() const
            = 0;
};

struct ConstantFunction : FunctionGenerator {
    double value;

    explicit ConstantFunction(double value, double T_s) : FunctionGenerator(T_s), value(value) {}

    double
    getSample() const override {
        return value;
    }
};

struct LinearRamp : FunctionGenerator {
    double startValue, finalValue, duration;

    LinearRamp(double startValue, double finalValue, double duration, double T_s) : FunctionGenerator(T_s), startValue(startValue), finalValue(finalValue), duration(duration) {}

    double
    getSample() const override {
        time += T_s;
        double val = startValue + (finalValue - startValue) * (time / duration);
        return time > duration ? finalValue : val;
    }
};

struct ParabolicRamp : FunctionGenerator {
    double startValue;
    double finalValue;
    double duration;
    double roundOnTime;
    double roundOffTime;
    double slope;
    double a;
    double transitPoint1;
    double transitPoint2;

    ParabolicRamp(double startValue, double finalValue, double duration, double roundOffTime, double T_s)
        : FunctionGenerator(T_s), startValue(startValue), finalValue(finalValue), duration(duration), roundOnTime(roundOffTime), roundOffTime(roundOffTime) {
        init();
    }

    void
    init() noexcept {
        time                      = 0;
        const double linearLength = duration - (roundOnTime + roundOffTime);
        a                         = (finalValue - startValue) / (2 * roundOnTime * (linearLength + roundOffTime));

        slope                     = (finalValue - startValue - 2 * a * pow(roundOffTime, 2)) / linearLength;

        transitPoint1             = startValue + a * pow(roundOffTime, 2);
        transitPoint2             = finalValue - a * pow(roundOffTime, 2);
    }

    double
    getSample() const override {
        time += T_s;
        if (time > duration) {
            return finalValue;
        }

        if (time < roundOnTime) {
            // first parabolic section
            return startValue + a * pow(time, 2);
        }
        if (time < duration - roundOffTime) {
            // linear section
            return transitPoint1 + slope * (time - roundOffTime);
        }
        // second parabolic section
        double shiftedTime = time - (duration - roundOffTime);
        return transitPoint2 + slope * shiftedTime - a * pow(shiftedTime, 2);
    }
};

struct CubicSpline : FunctionGenerator {
    double startValue;
    double finalValue;
    double duration;

    CubicSpline(double startValue, double finalValue, double duration, double T_s) : FunctionGenerator(T_s), startValue(startValue), finalValue(finalValue), duration(duration) {}

    double
    getSample() const override {
        time += T_s;
        double normalizedTime = time / duration;
        double val            = (2 * pow(normalizedTime, 3) - 3 * pow(normalizedTime, 2) + 1) * startValue + (-2 * pow(normalizedTime, 3) + 3 * pow(normalizedTime, 2)) * finalValue;
        return time > duration ? finalValue : val;
    }
};

struct ImpulseResponse : FunctionGenerator {
    double startValue;
    double finalValue;
    double t0;
    double t1;

    ImpulseResponse(double startValue, double finalValue, double t0, double t1, double T_s) : FunctionGenerator(T_s), startValue(startValue), finalValue(finalValue), t0(t0), t1(t1) {}

    double
    getSample() const override {
        time += T_s;
        if (time < t0 || time > t0 + t1) {
            return startValue;
        }
        return finalValue;
    }
};

class Generator {
public:
    double                             T_s; // Sampling period in seconds
    std::unique_ptr<FunctionGenerator> functionGenerator;

    Generator(double f_s) : T_s(1.0 / f_s) {}

    void
    setMode(FunctionMode mode, double startValue, double finalValue, double duration, double roundOffTime = 0.0, double t0 = 0.0, double t1 = 0.0) {
        switch (mode) {
        case FunctionMode::Constant:
            if (startValue != finalValue) {
                throw std::runtime_error("Start and final values must be the same for the "
                                         "constant function.");
            }
            functionGenerator = std::make_unique<ConstantFunction>(startValue, T_s);
            break;
        case FunctionMode::LinearRamp: functionGenerator = std::make_unique<LinearRamp>(startValue, finalValue, duration, T_s); break;
        case FunctionMode::ParabolicRamp: functionGenerator = std::make_unique<ParabolicRamp>(startValue, finalValue, duration, roundOffTime, T_s); break;
        case FunctionMode::CubicSpline: functionGenerator = std::make_unique<CubicSpline>(startValue, finalValue, duration, T_s); break;
        case FunctionMode::ImpulseResponse: functionGenerator = std::make_unique<ImpulseResponse>(startValue, finalValue, t0, t1, T_s); break;
        default: throw std::runtime_error("Invalid function mode.");
        }
    }

    double
    getNextSample() {
        return functionGenerator->getSample();
    }
};

int
main() {
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

#endif // GRAPH_PROTOTYPE_DEMO_FUNCTION_GENERATOR_HPP
