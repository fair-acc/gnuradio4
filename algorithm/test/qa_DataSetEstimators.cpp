#include <boost/ut.hpp>
#include <fmt/format.h>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp> // for draw(...)
#include <gnuradio-4.0/meta/formatter.hpp>

#include <fmt/ranges.h>

#include <gnuradio-4.0/algorithm/dataset/DataSetEstimators.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetMath.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetTestFunctions.hpp>

namespace test::detail {

template<class TLhs, class TRhs, class TEpsilon>
[[nodiscard]] constexpr auto approx(const TLhs& lhs, const TRhs& rhs, const TEpsilon& epsilon) {
    if constexpr (gr::meta::complex_like<TLhs>) {
        return boost::ut::detail::and_{boost::ut::detail::approx_{lhs.real(), rhs.real(), epsilon}, boost::ut::detail::approx_{lhs.imag(), rhs.imag(), epsilon}};
    } else {
        return boost::ut::detail::approx_{lhs, rhs, epsilon};
    }
}
} // namespace test::detail

const boost::ut::suite<"DataSet<T> visual test functions"> _DataSetTestFcuntions = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    constexpr static std::size_t nSamples = 201UZ;

    "triangular DataSet"_test = []<typename T = double> {
        gr::DataSet<T> ds = generate::triangular<T>("triagonal", nSamples);
        gr::dataset::draw(ds);

        expect(gr::dataset::verify<true>(ds));

        gr::DataSet<T> ds1 = generate::triangular<double>("triagonal - odd", 11);
        fmt::println("\"{:20}\": {}", ds1.signalName(0UZ), ds1.signal_values);
        expect(eq(ds1.signalValues().front(), ds1.signalValues().back()));
        expect(eq(ds1.signalValues()[5UZ], 1.0));

        gr::DataSet<T> ds2 = generate::triangular<double>("triagonal - even", 10);
        fmt::println("\"{:20}\": {}", ds2.signalName(0UZ), ds2.signal_values);
        expect(eq(ds2.signalValues().front(), ds2.signalValues().back()));
        expect(eq(ds2.signalValues()[4UZ], ds2.signalValues()[5UZ]));
    };

    "ramp DataSet"_test = []<typename T = double> {
        gr::DataSet<T> ds = generate::ramp<T>("ramp", nSamples);
        expect(gr::dataset::verify<true>(ds));
        gr::dataset::draw(ds);
    };

    "gaussFunction DataSet"_test = []<typename T = double> {
        constexpr T mean          = T(nSamples) / T(2);
        constexpr T sigma         = T(nSamples) / T(10);
        const T     normalisation = sigma * gr::math::sqrt(T(2) * std::numbers::pi_v<T>);

        gr::DataSet<T> ds = generate::gaussFunction<T>("gaussFunction", nSamples, mean, sigma, T(0), normalisation);
        expect(gr::dataset::verify<true>(ds));
        gr::dataset::draw(ds);
    };

    "ramp + gauss DataSet"_test = []<typename T = double> {
        constexpr T mean          = T(nSamples) / T(2);
        constexpr T sigma         = T(nSamples) / T(10);
        const T     normalisation = sigma * gr::math::sqrt(T(2) * std::numbers::pi_v<T>);

        gr::DataSet<T> ds1 = generate::ramp<T>("ramp", nSamples, 0.0, 0.2);
        gr::DataSet<T> ds2 = generate::gaussFunction<T>("gaussFunction", nSamples, mean, sigma, T(0), normalisation);
        gr::DataSet<T> ds  = addFunction(ds1, ds2);

        expect(gr::dataset::verify<true>(ds));
        gr::dataset::draw(ds);
    };

    "randomStepFunction DataSet"_test = []<typename T = double> {
        gr::DataSet<T> ds = generate::randomStepFunction<T>("randomStepFunction", nSamples);
        expect(gr::dataset::verify<true>(ds));
        gr::dataset::draw(ds);
    };
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wimplicit-float-conversion" // GCC & Clang: Loss or promotion of floating-point precision, disabled only for unit-tests

const boost::ut::suite<"DataSet<T> element-wise accessor"> _dataSetAccessors = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    "basic element-wise access API "_test = []<typename T> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds       = generate::triangular<T>("triag", 3UZ); // data ~ [0, 1, 0]

        expect(approx(getIndexValue(ds, dim::Y, 0), value_t(0), value_t(1e-3f)));
        expect(approx(getIndexValue(ds, dim::Y, 1), value_t(1), value_t(1e-3f)));
        expect(approx(getIndexValue(ds, dim::Y, 0), value_t(0), value_t(1e-3f)));

        expect(!gr::math::isfinite(getIndexValue(ds, dim::X, 3))) << fmt::format("element is not NaN: {}", getIndexValue(ds, dim::X, 3));
        expect(!gr::math::isfinite(getIndexValue(ds, dim::Y, 3))) << fmt::format("element is not NaN: {}", getIndexValue(ds, dim::Y, 3));

        expect(throws([&] { std::ignore = getIndexValue(ds, dim::Z, 0); }));

        expect(approx(getDistance(ds, dim::X, 0UZ, 2UZ), value_t(2), value_t(1e-3f)));
        expect(approx(getDistance(ds, dim::X), value_t(2), value_t(1e-3f)));

        expect(approx(getDistance(ds, dim::Y, 0UZ, 1UZ), value_t(1), value_t(1e-3f))); // Y-distance of first to middle
        expect(approx(getDistance(ds, dim::Y), T(0), value_t(1e-3f)));                 // Y-distance of first to last

        expect(approx(getValue(ds, dim::X, value_t(0.123f)), static_cast<T>(0.123f), value_t(1e-3f))); // identity
        expect(approx(getValue(ds, dim::Y, value_t(0.5)), static_cast<T>(0.5f), value_t(1e-3f)));

        std::vector<T> copyX = getSubArrayCopy(ds, dim::X, 0UZ, 2UZ);
        for (std::size_t i = 0; i < copyX.size(); i++) {
            expect(eq(copyX[i], getIndexValue(ds, dim::X, i))) << fmt::format("X-index {} mismatch", i);
        }

        std::vector<T> copyY = getSubArrayCopy(ds, dim::X, 0UZ, 2UZ);
        for (std::size_t i = 0; i < copyY.size(); i++) {
            expect(eq(copyY[i], getIndexValue(ds, dim::Y, i))) << fmt::format("X-index {} mismatch", i);
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DSP helper"> _dspHelper = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    "common dsp helper functions"_test = []<typename T> {
        using value_t                     = T;
        const static std::string typeName = gr::meta::type_name<T>();

        "tenLog10"_test = [] {
            expect(approx(tenLog10(value_t(10)), value_t(10), value_t(1e-3f))) << "10 * log10(10) should be 10";
            expect(approx(tenLog10(value_t(1)), value_t(0), value_t(1e-3f))) << "10 * log10(1) should be 0";
            expect(approx(tenLog10(value_t(0.1)), value_t(-10), value_t(1e-3f))) << "10 * log10(0.1) should be -10";

            // edge cases
            expect(!gr::math::isfinite(tenLog10(value_t(0)))) << fmt::format("tenLog10<{}>(0) = {} should be -inf", typeName, tenLog10(value_t(0)));
            expect(!gr::math::isfinite(tenLog10(value_t(-1)))) << fmt::format("tenLog10<{}>(-1) = {} should be -inf", typeName, tenLog10(value_t(-1)));
        };

        "decibel"_test = [] {
            expect(approx(decibel(value_t(10)), value_t(20), value_t(1e-3f))) << "20 * log10(10) should be 20";
            expect(approx(decibel(value_t(1)), value_t(0), value_t(1e-3f))) << "20 * log10(1) should be 0";
            expect(approx(decibel(value_t(0.1)), value_t(-20), value_t(1e-3f))) << "20 * log10(0.1) should be -20";

            // edge cases
            expect(!gr::math::isfinite(decibel(value_t(0)))) << "decibel(0) should be -inf";
            expect(!gr::math::isfinite(decibel(value_t(-1)))) << "20 * log10(-1) should return -inf";
        };

        "inverseDecibel"_test = [] {
            expect(approx(inverseDecibel(value_t(20)), value_t(10), value_t(1e-3f))) << "10^(20 / 20) should be 10";
            expect(approx(inverseDecibel(value_t(0)), value_t(1), value_t(1e-3f))) << "10^(0 / 20) should be 1";
            expect(approx(inverseDecibel(value_t(-20)), value_t(0.1), value_t(1e-3f))) << "10^(-20 / 20) should be 0.1";

            // edge cases
            expect(approx(inverseDecibel(value_t(100)), gr::math::pow(value_t(10), value_t(5)), value_t(1e-3f))) << "10^(100 / 20) should be 10^5";
            expect(approx(inverseDecibel(value_t(-100)), gr::math::pow(value_t(10), value_t(-5)), value_t(1e-3f))) << "10^(-100 / 20) should be 10^-5";
        };
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DataSet<T> estimator"> _qaDataSetEstimators = [] {
    using namespace boost::ut;
    using namespace gr::dataset;
    using test::detail::approx;

    constexpr static size_t nSamples = 11;

    "basic estimators"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        // 10-sample triangle => ascending up to i=4..5 => data ~ [0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0]
        auto ds = generate::triangular<T>("triag", nSamples);

        expect(approx(estimators::computeCentreOfMass(ds), T(5), T(1e-3)));
        expect(gr::math::isfinite(estimators::computeCentreOfMass(ds, 0UZ, nSamples))) << "should be finite for partial range";

        std::vector<T> data{1, 2, 3, 2, 1};
        expect(approx(estimators::computeFWHM(data, 2), T(4), T(1e-5)));
        expect(approx(estimators::computeInterpolatedFWHM(data, 2), T(3), T(1e-5)));

        expect(eq(estimators::getLocationMaximum(ds, 0UZ, nSamples), int(5)));
        expect(eq(estimators::getLocationMinimum(ds, 0UZ, nSamples), int(0)));
        expect(eq(estimators::getLocationMaximum(ds), int(5)));
        expect(eq(estimators::getLocationMinimum(ds), int(0)));

        expect(eq(gr::value(estimators::getMaximum(ds, 0UZ, nSamples)), value_t(1)));
        expect(eq(gr::value(estimators::getMaximum(ds)), value_t(1)));
        expect(eq(gr::value(estimators::getMinimum(ds, 0UZ, nSamples)), value_t(0)));
        expect(eq(gr::value(estimators::getMinimum(ds)), value_t(0)));

        expect(approx(estimators::getMean(ds, 0UZ, nSamples), T(0.454545), T(1e-3f)));
        expect(approx(estimators::getMean(ds), T(0.454545), T(1e-3f)));

        expect(approx(estimators::getMedian(ds, 0UZ, nSamples), T(0.4), T(1e-3f)));
        expect(approx(estimators::getMedian(ds), T(0.4), T(1e-3f)));

        expect(eq(gr::value(estimators::getRange(ds, 0UZ, nSamples)), value_t(1)));
        expect(eq(gr::value(estimators::getRange(ds)), value_t(1)));

        expect(approx(estimators::getRms(ds, 0UZ, nSamples), T(0.320124), T(1e-3)));
        expect(approx(estimators::getRms(ds), T(0.320124), T(1e-3)));
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getRms(ds, 0UZ, nSamples)), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getRms(ds)), value_t(0)));
        }

        expect(approx(estimators::getIntegral(ds, 0UZ, nSamples), T(5.0), T(1e-3)));
        expect(approx(estimators::getIntegral(ds), T(5.0), T(1e-3)));
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getIntegral(ds, 0UZ, nSamples)), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getIntegral(ds)), value_t(0)));
        }

        expect(approx(estimators::getEdgeDetect(ds, 0UZ, nSamples), T(3), T(0.5))) << "50% is ~ 0.5 => crossing near i=3";
        expect(approx(estimators::getEdgeDetect(ds), T(3), T(0.5))) << "50% is ~ 0.5 => crossing near i=3";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
    ;

    "getDutyCycle"_test = []<typename T = double> {
        using value_t         = gr::meta::fundamental_base_value_type_t<T>;
        std::vector<T> localY = {0, 0, 0, 1, 1, 1}; // simple data set with 0,0,0,1,1,1 => 50% high
        gr::DataSet<T> ds;
        ds.axis_names = {"Time"};
        ds.axis_units = {"s"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(localY.size());
        ds.signal_names = {"simple step"};
        ds.signal_values.resize(localY.size());
        ds.extents = {1, static_cast<std::int32_t>(localY.size())};
        for (std::size_t i = 0; i < localY.size(); i++) {
            ds.axis_values[0][i] = T(i);
            ds.signal_values[i]  = localY[i];
        }
        ds.signal_ranges.push_back({T(0), T(1)});

        expect(eq(estimators::getDutyCycle(ds, 0UZ, localY.size()), value_t(0.5))) << "simple 3-high/3-low => duty=0.5";
        expect(eq(estimators::getDutyCycle(ds), value_t(0.5))) << "simple 3-high/3-low => duty=0.5";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "getFrequencyEstimate"_test = []<typename T = double> {
        using value_t         = gr::meta::fundamental_base_value_type_t<T>;
        std::vector<T> localY = {0, 1, 0, 1, 0, 1};
        gr::DataSet<T> ds;
        ds.axis_names = {"Time"};
        ds.axis_units = {"s"};
        ds.axis_values.resize(1);
        ds.axis_values[0].resize(localY.size());
        ds.signal_names = {"oscillator"};
        ds.signal_values.resize(localY.size());
        ds.extents = {1, int(localY.size())};
        for (std::size_t i = 0; i < localY.size(); i++) {
            ds.axis_values[0][i] = T(i);
            ds.signal_values[i]  = localY[i];
        }
        ds.signal_ranges.push_back({T(0), T(1)});

        expect(eq(estimators::getFrequencyEstimate(ds, 0UZ, localY.size()), value_t(0.5))) << "frequency ~ 0.5";
        expect(eq(estimators::getFrequencyEstimate(ds), value_t(0.5))) << "frequency ~ 0.5";
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "getLocationMaximumGaussInterpolated"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds1      = generate::triangular<T>("triag", 7UZ); // symmetric peak around index 3

        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds1, 0UZ, 7UZ)), value_t(3), T(1e-3f))) << "Gauss interpolation ~3";
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds1)), value_t(3), T(1e-3f))) << "Gauss interpolation ~3";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds1, 0UZ, 7UZ)), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds1)), value_t(0)));
        }

        auto ds2 = generate::triangular<T>("triag", 6UZ); // symmetric peak around index 2.5
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds2, 0UZ, 6UZ)), value_t(2.5), T(1e-3f))) << "Gauss interpolation ~2.5";
        expect(approx(gr::value(estimators::getLocationMaximumGaussInterpolated(ds2)), value_t(2.5), T(1e-3f))) << "Gauss interpolation ~2.5";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds2, 0UZ, 6UZ)), value_t(0)));
            expect(neq(gr::uncertainty(estimators::getLocationMaximumGaussInterpolated(ds2)), value_t(0)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "zeroCrossing test"_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds       = generate::triangular<T>("ZC", nSamples);

        expect(approx(gr::value(estimators::getZeroCrossing(ds, value_t(0.5))), value_t(2.5), value_t(1e-3))) << "zero crossing ~2.5";
        if constexpr (gr::UncertainValueLike<T>) {
            expect(neq(gr::uncertainty(estimators::getZeroCrossing(ds, value_t(0.5))), value_t(0)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};
};

const boost::ut::suite<"DataSet<T> math "> _dataSetMath = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using namespace gr::dataset;

    "basic math API "_test = []<typename T = double> {
        using value_t = gr::meta::fundamental_base_value_type_t<T>;
        auto ds1      = generate::triangular<T>("ds1", 5);
        auto ds2      = addFunction(ds1, value_t(2)); // add scalar '2'

        for (std::size_t i = 0; i < 5; i++) {
            T y1 = gr::dataset::getIndexValue(ds1, dim::Y, i);
            T y2 = gr::dataset::getIndexValue(ds2, dim::Y, i);
            expect(approx(y2, y1 + T(2), value_t(1e-3f)));
        }

        // add two dataSets
        auto ds3 = addFunction(ds1, ds1);
        for (std::size_t i = 0; i < 5; i++) {
            T y1 = gr::dataset::getIndexValue(ds1, 1, i);
            T y3 = gr::dataset::getIndexValue(ds3, 1, i);
            expect(approx(y3, value_t(2) * y1, value_t(1e-3f)));
        }
    } | std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>{};

    "mathFunction variants"_test = []<typename T = double> {
        auto ds1 = generate::triangular<T>("triag", 5);

        // 1) dataSet + dataSet
        auto dsAdd = mathFunction(ds1, ds1, MathOp::ADD);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsAdd.signal_values[i];
            expect(eq(newVal, T(2) * origVal)) << fmt::format("Add op failed at i={}, got {}", i, newVal);
        }

        // 2) dataSet + double
        auto dsAdd2 = mathFunction(ds1, T(2), MathOp::ADD);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsAdd2.signal_values[i];
            expect(eq(newVal, origVal + T(2)));
        }

        // 3) MULTIPLY
        auto dsMul = mathFunction(ds1, ds1, MathOp::MULTIPLY);
        for (std::size_t i = 0UZ; i < 5UZ; i++) {
            T origVal = ds1.signal_values[i];
            T newVal  = dsMul.signal_values[i];
            expect(approx(newVal, origVal * origVal, T(1e-5))) << "Multiply test";
        }

        // 4) SQR => (y1 + y2)^2 => ds1 + 2 => then squared
        auto dsSqr = mathFunction(ds1, T(2), MathOp::SQR);
        // check a single sample
        expect(eq(dsSqr.signal_values[0], T((ds1.signal_values[0] + T(2)) * (ds1.signal_values[0] + T(2)))));

        // 5) SQRT => sqrt(y1 + y2)
        auto dsSqrt = mathFunction(ds1, T(2.0), MathOp::SQRT);
        // if ds1.signal_values[0]=0 => sqrt(2) => ~1.4142
        expect(approx(dsSqrt.signal_values[0], T(1.4142f), T(1e-2)));

        // 6) LOG10 => 10*log10(y1 + y2)
        auto dsLog = mathFunction(ds1, T(2.0), MathOp::LOG10);
        // we can do a quick check for i=0 => 10 * log10(2.0) => 3.0103 dB
        expect(approx(dsLog.signal_values[0], T(3.01f), T(1e-2)));

        // 7) DB => 20*log10(y1 + y2)
        auto dsDb = mathFunction(ds1, T(2.0), MathOp::DB);
        // i=0 => 20*log10(2.0) => 6.0206 dB
        expect(approx(dsDb.signal_values[0], T(6.02f), T(1e-2)));

        // 8) INV_DB => 10^(y1/20), ignoring y2
        // if ds1.signal_values[0]=0 => => 10^(0/20)=>1
        auto dsInv = mathFunction(ds1, 2.0, MathOp::INV_DB);
        expect(eq(dsInv.signal_values[0], T(1))) << "inv_db test";
    };

    // WIP -- add more DataSet<T> math-related functionality here
};

#pragma GCC diagnostic pop

int main() { /* not needed for UT */ }
