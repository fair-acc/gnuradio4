#include <boost/ut.hpp>

#include <benchmark.hpp>

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>

template<gr::filter::ResponseType responseType, std::floating_point T>
[[nodiscard]] std::vector<T> calculateFrequencyResponse(const std::vector<T>& frequencies, const gr::filter::iir::PoleZeroLocations& value) {
    using namespace gr::filter;
    std::vector<double> response(frequencies.size());
    std::ranges::transform(frequencies, response.begin(), [&](const double& frequency) { return calculateResponse<Frequency::Hertz, responseType>(frequency, value); });
    return response;
}

template<gr::filter::ResponseType responseType, std::floating_point T, typename... TFilterCoefficients>
std::vector<double> calculateFrequencyResponse(const std::vector<double>& frequencies, T fs, TFilterCoefficients&&... filterCoefficients) {
    std::vector<gr::filter::FilterCoefficients<T>> collection{std::forward<TFilterCoefficients>(filterCoefficients)...};

    std::vector<double> response;
    response.reserve(frequencies.size());
    for (auto freq : frequencies) {
        auto transform = [normDigitalFrequency = static_cast<T>(freq) / fs](auto& filterSection) { return calculateResponse<gr::filter::Frequency::Normalised, responseType>(normDigitalFrequency, filterSection); };

        if constexpr (responseType == gr::filter::ResponseType::Magnitude) {
            // Magnitude combines multiplicative
            response.emplace_back(std::transform_reduce(collection.cbegin(), collection.cend(), static_cast<T>(1), std::multiplies{}, transform));
        } else {
            // MagnitudeDB, Phase, PhaseDegrees combine additive
            response.emplace_back(std::transform_reduce(collection.cbegin(), collection.cend(), static_cast<T>(0), std::plus{}, transform));
        }
    }
    return response;
}

template<typename Container>
void printFilter(std::string_view name, const Container& filters) {
    if constexpr (gr::filter::iir::HasPoleZeroLocations<Container>) {
        fmt::print("{}-section:{{ zero({}):{}, pole({}):{} }}\n", name, filters.zeros.size(), filters.zeros, filters.poles.size(), filters.poles);
    } else if constexpr (gr::filter::HasFilterCoefficients<Container>) {
        fmt::print("{}(1): a({}):{}, b({}):{}\n", name, filters.a.size(), filters.a, filters.b.size(), filters.b);
    } else if constexpr (std::ranges::range<Container> && gr::filter::HasFilterCoefficients<std::ranges::range_value_t<Container>>) {
        fmt::print("{}({}):", name, std::ranges::size(filters));
        for (const auto& filter : filters) {
            fmt::print("-section:{{ a({}):{}, b({}):{} }}\n", filter.a.size(), filter.a, filter.b.size(), filter.b);
        }
        fmt::print("\n");
    } else {
        static_assert(false, "container does not meet requirements for any known filter type.");
    }
}

template<std::size_t width = 51, std::size_t height = 21>
void poleZeroPlot(const gr::filter::iir::PoleZeroLocations& value, double radius = 2.0) {
    constexpr std::size_t samples = 360Z;
    std::vector<double>   xCircle(samples);
    std::vector<double>   yCircle(samples);

    auto angles = std::views::iota(0UZ, samples) | std::views::transform([](auto i) { return 2. * std::numbers::pi * static_cast<double>(i) / static_cast<double>(samples); });
    std::ranges::transform(angles, xCircle.begin(), [](double angle) { return std::cos(angle); });
    std::ranges::transform(angles, yCircle.begin(), [](double angle) { return std::sin(angle); });

    std::vector<double> xPoles(value.poles.size());
    std::vector<double> yPoles(value.poles.size());
    std::ranges::transform(value.poles, xPoles.begin(), [](auto pole) { return pole.real(); });
    std::ranges::transform(value.poles, yPoles.begin(), [](auto pole) { return pole.imag(); });

    std::vector<double> xZeros(value.zeros.size());
    std::vector<double> yZeros(value.zeros.size());
    std::ranges::transform(value.zeros, xZeros.begin(), [](auto pole) { return pole.real(); });
    std::ranges::transform(value.zeros, yZeros.begin(), [](auto pole) { return pole.imag(); });

    radius = std::max(radius, std::max(*std::max_element(xPoles.begin(), xPoles.end()), *std::max_element(yPoles.begin(), yPoles.end())));

    radius            = std::ceil(15 * radius) / 10.0;
    auto chart        = gr::graphs::ImChart<width, height>({{-radius, +radius}, {-radius, +radius}});
    chart.axis_name_x = "Re";
    chart.axis_name_y = "Im";
    chart.drawAxes();

    chart._lastColor = gr::graphs::Color::Type::White;
    chart.draw(xCircle, yCircle);
    chart._lastColor  = chart.kFirstColor;
    chart._n_datasets = 0;
    chart._datasets   = {};

    chart.template draw<gr::graphs::Style::Marker>(xPoles, yPoles, "poles");
    chart.template draw<gr::graphs::Style::Marker>(xZeros, yZeros, "zeros");

    chart.drawLegend();
    chart.printScreen();
}

const boost::ut::suite<"IIR FilterTool"> iirFilterToolTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using enum Frequency;
    using enum ResponseType;
    using enum iir::Design;
    using magic_enum::enum_name;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = {"visual", "benchmarks"}};
    }

    "IIR filter"_test = [](iir::Design filterDesign) {
        using namespace gr::filter::iir;

        "IIR prototype filter"_test = [&filterDesign] {
            for (std::size_t order = 0UZ; order <= 10UZ; ++order) {
                PoleZeroLocations analogPoleZeros;
                switch (filterDesign) { // only the Elliptic and inverse Chebyshev (Type II) filters have zeros.
                case BUTTERWORTH: analogPoleZeros = calculateFilterButterworth(order); break;
                case CHEBYSHEV1: analogPoleZeros = calculateFilterChebyshevType1(order); break;
                case CHEBYSHEV2: analogPoleZeros = calculateFilterChebyshevType2(order); break;
                case BESSEL: analogPoleZeros = calculateFilterBessel(order); break;
                default: throw std::invalid_argument("unknown filterType");
                }

                for (const auto& zero : analogPoleZeros.zeros) {
                    expect(le(zero.real(), 0.)) << fmt::format("{}::zero{} is not on the left-half plane", enum_name(filterDesign), zero);
                }

                for (const auto& pole : analogPoleZeros.poles) {
                    expect(le(pole.real(), 0.)) << fmt::format("{}::pole{} is not on the left-half plane", enum_name(filterDesign), pole);
                }

                expect(approx(1.0, iir::calculateResponse<Hertz, Magnitude>(0.0, analogPoleZeros), 0.001)) << fmt::format("{} - order {} has non-unity gain at DC: {:.4f}", enum_name(filterDesign), order, 1.0 / iir::calculateResponse<Hertz, Magnitude>(0.0, analogPoleZeros));
            }
        };

        "IIR analog filter missing parameters"_test = [&filterDesign] {
            expect(throws([&] { std::ignore = iir::designAnalogFilter(Type::LOWPASS, {}, filterDesign); })) << "undefined fLow should throw";
            expect(throws([&] { std::ignore = iir::designAnalogFilter(Type::HIGHPASS, {}, filterDesign); })) << "undefined fLow should throw";
            expect(throws([&] { std::ignore = iir::designAnalogFilter(Type::BANDPASS, {}, filterDesign); })) << "undefined fLow should throw";
            expect(throws([&] { std::ignore = iir::designFilter<double>(Type::LOWPASS, {.fLow = 1.0}, filterDesign); })) << "undefined fs should throw";
        };

        using enum Type;
        "IIR analog filter"_test = [&filterDesign](Type filterType) {
            for (std::size_t order = 2UZ; order <= 10UZ; ++order) {
                constexpr double kDC           = 0.0;
                constexpr double kFreqLow      = 1.0;
                constexpr double kFreqHigh     = 10.0;
                const double     kOmega0       = std::sqrt(kFreqLow * kFreqHigh);
                constexpr double kSampling     = 1000.0;
                constexpr double kNyquist      = kSampling / 2.0;
                const double     kTolerance    = ((filterDesign == CHEBYSHEV2) ? 10 : 1) * 0.01;
                const auto       kFilterParams = FilterParameters{.order = order, .fLow = kFreqLow, .fHigh = kFreqHigh, .attenuationDb = 30, .fs = kSampling};

                // compute analog filter
                expect(nothrow([&filterDesign, &filterType, &kFilterParams]() { std::ignore = iir::designAnalogFilter(filterType, kFilterParams, filterDesign); })) //
                    << fmt::format("{}({}) - order {} unexpectedly throws", enum_name(filterDesign), enum_name(filterType), order);
                PoleZeroLocations analogPoleZeros = iir::designAnalogFilter(filterType, kFilterParams, filterDesign);

                switch (filterType) {
                case BANDSTOP: expect(le(iir::calculateResponse<Hertz, Magnitude>(kOmega0, analogPoleZeros), kTolerance)) << fmt::format("{}({}) - order {} insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kOmega0, analogPoleZeros)); [[fallthrough]];
                case LOWPASS: expect(approx(1.0, iir::calculateResponse<Hertz, Magnitude>(kDC, analogPoleZeros), kTolerance)) << fmt::format("{}({}) - order {} has non-unity gain at DC: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kDC, analogPoleZeros)); break;
                case HIGHPASS:
                    expect(approx(1.0, iir::calculateResponse<Hertz, Magnitude>(kNyquist, analogPoleZeros), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at NQ: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kNyquist, analogPoleZeros));
                    expect(le(iir::calculateResponse<Hertz, Magnitude>(kDC, analogPoleZeros), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kOmega0, analogPoleZeros));
                    break;
                case BANDPASS:
                    expect(approx(1.0, iir::calculateResponse<Hertz, Magnitude>(kOmega0, analogPoleZeros), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at omega0: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kOmega0, analogPoleZeros));
                    expect(le(iir::calculateResponse<Hertz, Magnitude>(kDC, analogPoleZeros), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (DC): {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kDC, analogPoleZeros));
                    expect(le(iir::calculateResponse<Hertz, Magnitude>(kNyquist, analogPoleZeros), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (NQ): {:.4f}", enum_name(filterDesign), enum_name(filterType), order, iir::calculateResponse<Hertz, Magnitude>(kNyquist, analogPoleZeros));
                    break;
                }
            }
        } | std::vector{LOWPASS, HIGHPASS, BANDSTOP, BANDPASS};

        "IIR digital filter"_test = [&filterDesign](Type filterType) {
            const std::size_t startingOrder = filterDesign == CHEBYSHEV2 ? 3UZ : 1UZ; // Chebyshev with order <=2 has notorious poor performance for band-pass/stop and <=3 for high-pass at NQ
            for (std::size_t order = startingOrder; order <= 5UZ; ++order) {
                constexpr double kDC           = 0.0;
                constexpr double kFreqLow      = 1.0;
                constexpr double kFreqHigh     = 10.0;
                const double     kOmega0       = std::sqrt(kFreqLow * kFreqHigh);
                constexpr double kSampling     = 1000.0;
                constexpr double kNyquist      = kSampling / 2.0;
                constexpr double kTolerance    = 0.01;
                const auto       kFilterParams = FilterParameters{.order = order, .fLow = kFreqLow, .fHigh = kFreqHigh, .attenuationDb = 50, .fs = kSampling};
                // compute analog and digital filters
                const PoleZeroLocations analogPoleZeros = iir::designAnalogFilter(filterType, kFilterParams, filterDesign);
                expect(nothrow([&filterDesign, &filterType, &kFilterParams]() { std::ignore = iir::designFilter<double>(filterType, kFilterParams, filterDesign); })) //
                    << fmt::format("({}, {}, {}) creating digital unbound filter unexpectedly throws", enum_name(filterDesign), order, enum_name(filterType));
                const auto digitalPoleZerosFull = iir::designFilter<double>(filterType, kFilterParams, filterDesign);
                expect(nothrow([&filterDesign, &filterType, &kFilterParams]() { std::ignore = iir::designFilter<double, 2UZ>(filterType, kFilterParams, filterDesign); })) //
                    << fmt::format("({}, {}, {}) creating digital biquad filter unexpectedly throws", enum_name(filterDesign), enum_name(filterType), order);
                const auto digitalPoleZerosBiquads = iir::designFilter<double, 2UZ>(filterType, kFilterParams, filterDesign);

                // coarse response test
                switch (filterType) {
                case BANDSTOP: expect(le(calculateResponse<Normalised, Magnitude>(kOmega0 / kSampling, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kOmega0, digitalPoleZerosBiquads)); [[fallthrough]];
                case LOWPASS: expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kDC, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at DC: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kDC, digitalPoleZerosBiquads)); break;
                case HIGHPASS:
                    expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kNyquist / kSampling, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at NQ: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kNyquist / kSampling, digitalPoleZerosBiquads));
                    expect(le(calculateResponse<Normalised, Magnitude>(kDC, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kOmega0 / kSampling, digitalPoleZerosBiquads));
                    break;
                case BANDPASS:
                    expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kOmega0 / kSampling, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at omega0: {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kOmega0 / kSampling, digitalPoleZerosBiquads));
                    expect(le(calculateResponse<Normalised, Magnitude>(kDC, digitalPoleZerosBiquads), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (DC): {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kDC, digitalPoleZerosBiquads));
                    const double relaxFactor = filterDesign == CHEBYSHEV1 ? 20 : (order <= 2 ? 3.0 : 1.0);
                    expect(le(calculateResponse<Normalised, Magnitude>(kNyquist / kSampling, digitalPoleZerosBiquads), relaxFactor * kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (NQ): {:.4f}", enum_name(filterDesign), enum_name(filterType), order, calculateResponse<Normalised, Magnitude>(kNyquist / kSampling, digitalPoleZerosBiquads));
                    break;
                }

                // generate test frequency range
                auto sequence_range = [](auto start, auto end, auto step) {
                    using namespace std::views; // for iota and transform
                    return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
                };
                std::vector<double> frequencies;
                for (auto subrange : {sequence_range(0.1, 0.9, 0.01), sequence_range(1.0, 9.0, 0.1), sequence_range(10.0, 90.0, 1.0), sequence_range(100.0, 490.0, 10.0)}) {
                    std::ranges::move(subrange, std::back_inserter(frequencies));
                }
                std::vector<double> filterResponse   = calculateFrequencyResponse<Magnitude>(frequencies, analogPoleZeros);                    // provide in absolute values
                std::vector<double> responseDigital1 = calculateFrequencyResponse<Magnitude>(frequencies, kSampling, digitalPoleZerosFull);    // provide in absolute values
                std::vector<double> responseDigital2 = calculateFrequencyResponse<Magnitude>(frequencies, kSampling, digitalPoleZerosBiquads); // provide in absolute values

                //
                for (std::size_t i = 0UZ; i < frequencies.size(); ++i) {
                    const double reference = filterResponse[i]; // N.B in [1]
                    // non-biquad representation is known to be numerically unstable for order > ~4 (N.B. order: 2 -> band-[pass,stop] order: 4
                    if (order <= 2 && reference > 0.01) {
                        expect(le(std::abs(responseDigital1[i] - reference), (filterDesign == BESSEL || filterDesign == CHEBYSHEV2 || order <= 1UZ ? 10 : 1) * kTolerance)) //
                            << fmt::format("({}, {}, {})@f={}Hz analog-digital mismatch: {:.2f} vs {:.2f}",                                                                 //
                                   enum_name(filterDesign), enum_name(filterType), order, frequencies[i], reference, responseDigital1[i]);
                    }

                    // biquad style filter
                    if (reference > 0.01) { // BESSEL and CHEBYSHEV1 do not preserve the magnitude response perfectly (visually checked OK)
                        const double relaxFactor = (filterDesign == BESSEL || order <= 1UZ || (filterDesign == CHEBYSHEV1 && filterType == HIGHPASS) || (filterDesign == CHEBYSHEV2 && filterType == BANDSTOP)) ? 10 : 1;
                        expect(le(std::abs(responseDigital2[i] - reference), relaxFactor * kTolerance)) << fmt::format("({}, {}, {})@f={}Hz analog-digital biquad mismatch: {:.2f} vs {:.2f}", //
                            enum_name(filterDesign), enum_name(filterType), order, frequencies[i], reference, responseDigital2[i]);
                    }
                }
            }
        } | std::vector{LOWPASS, HIGHPASS, BANDPASS, BANDSTOP};
    } | std::vector{BUTTERWORTH, BESSEL, CHEBYSHEV1, CHEBYSHEV2};

    "IIR low-pass filter"_test = []<typename FilterType>(FilterType) {
        using namespace gr::filter::iir;
        constexpr double      fs           = 1000.;
        constexpr Design      filterDesign = Design::BUTTERWORTH;
        constexpr std::size_t order        = 4UZ;

        constexpr std::array frequencies{3.0, 3.5, 5., 6.5, 7.0};
        const auto           digitalBandPass = iir::designFilter<double>(Type::BANDPASS, {.order = order, .fLow = 4., .fHigh = 6., .fs = fs}, filterDesign);
        for (const auto& frequency : frequencies) {
            auto         filter       = FilterType(digitalBandPass);
            const double expectedGain = calculateResponse<Normalised, Magnitude>(frequency / fs, digitalBandPass);
            double       actualGain   = 0.0;
            for (std::size_t i = 0UZ; i < 10'000; ++i) {
                double value = filter.processOne(std::sin(2. * std::numbers::pi * frequency / fs * static_cast<double>(i)));
                if (i > 5'000UZ) { // skip initial transient
                    actualGain = std::max(actualGain, std::abs(value));
                }
            }
#if not defined(__EMSCRIPTEN__) // TODO: check why Emscripten fails to compute using float (old libc++) while gcc/clang do work
            expect(approx(expectedGain, actualGain, 0.01)) << fmt::format("frequency: {} Hz, expected {} vs. actual gain {} differs", frequency, expectedGain, actualGain);
#endif
        }
    } | std::tuple{Filter<double, 32UZ, Form::DF_I>(), Filter<double, 32UZ, Form::DF_II>(), Filter<double, 32UZ, Form::DF_I_TRANSPOSED>(), Filter<double, 32UZ, Form::DF_II_TRANSPOSED>()};
    ;

    tag("visual") / "basic analog low-/high-/band-pass filter - frequency"_test = []() {
        using namespace gr::graphs;
        using T                 = float;
        using PoleZeroLocations = gr::filter::iir::PoleZeroLocations;
        constexpr double fMin   = 0.1;
        constexpr double fMax   = 1'000.0;

        auto sequence_range = [](auto start, auto end, auto step) {
            using namespace std::views; // for iota and transform
            return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
        };

        std::vector<double> frequencies;
        for (auto subrange : {sequence_range(0.1, 0.9, 0.01), sequence_range(1.0, 9.0, 0.1), sequence_range(10.0, 90.0, 1.0), sequence_range(100.0, 1000.0, 10.0)}) {
            std::ranges::move(subrange, std::back_inserter(frequencies));
        }

        constexpr iir::Design   filterDesign     = BUTTERWORTH;
        constexpr std::size_t   order            = 5UZ;
        const PoleZeroLocations filter1          = iir::designAnalogFilter(Type::LOWPASS, {.order = order, .fLow = 1., .attenuationDb = 40}, filterDesign);
        const auto              digitalFilter1   = iir::designFilter<T>(Type::LOWPASS, {.order = order, .fLow = 1., .attenuationDb = 40, .fs = fMax}, filterDesign);
        std::vector<double>     lowPassResponse  = calculateFrequencyResponse<MagnitudeDB>(frequencies, filter1);
        std::vector<double>     lowPassDigital   = calculateFrequencyResponse<MagnitudeDB>(frequencies, static_cast<T>(fMax), digitalFilter1);
        const PoleZeroLocations filter2          = iir::designAnalogFilter(Type::HIGHPASS, {.order = order, .fHigh = 10., .attenuationDb = 40}, filterDesign);
        const auto              digitalFilter2   = iir::designFilter<T>(Type::HIGHPASS, {.order = order, .fHigh = 10., .attenuationDb = 40, .fs = fMax}, filterDesign);
        std::vector<double>     highPassResponse = calculateFrequencyResponse<MagnitudeDB>(frequencies, filter2);
        std::vector<double>     highPassDigital  = calculateFrequencyResponse<MagnitudeDB>(frequencies, static_cast<T>(fMax), digitalFilter2);
        const PoleZeroLocations filter3          = iir::designAnalogFilter(Type::BANDPASS, {.order = order, .fLow = 1., .fHigh = 10., .attenuationDb = 40}, filterDesign);
        const auto              digitalFilter3   = iir::designFilter<T>(Type::BANDPASS, {.order = order, .fLow = 1., .fHigh = 10., .attenuationDb = 40, .fs = fMax}, filterDesign);
        std::vector<double>     bandPassResponse = calculateFrequencyResponse<MagnitudeDB>(frequencies, filter3);
        std::vector<double>     bandPassDigital  = calculateFrequencyResponse<MagnitudeDB>(frequencies, static_cast<T>(fMax), digitalFilter3);

        expect(ge(frequencies.size(), 1UZ));
        expect(eq(frequencies.size(), lowPassResponse.size()));
        // pole-zero plots:
        poleZeroPlot(iir::calculateFilterButterworth(order), 1.2);
        poleZeroPlot(iir::calculateFilterChebyshevType2(order), 2.2);
        poleZeroPlot(iir::lowPassProtoToBandPass(iir::calculateFilterChebyshevType2(order, 40.), {.fLow = 10., .fHigh = 20.}), order);

        // plot
        auto chart        = gr::graphs::ImChart<119, 21, LogAxisTransform>({{fMin, fMax}, {-45., +5}});
        chart.axis_name_x = "[Hz]";
        chart.axis_name_y = "[dB]";
        chart.draw(frequencies, lowPassResponse, "low");
        chart.draw(frequencies, highPassResponse, "high");
        chart.draw(frequencies, bandPassResponse, "band-pass");

        chart.draw(frequencies, lowPassDigital, fmt::format("IIR low(N={})", countFilterCoefficients(digitalFilter1)));
        chart.draw(frequencies, highPassDigital, fmt::format("high({})", countFilterCoefficients(digitalFilter2)));
        chart.draw(frequencies, bandPassDigital, fmt::format("band-pass({})", countFilterCoefficients(digitalFilter3)));
        chart.draw();
    };

    tag("visual") / "basic IIR band-pass filter frequency"_test = []() {
        using namespace gr::filter::iir;
        using T                       = float;
        constexpr double fs           = 1000.;
        constexpr Design filterDesign = BUTTERWORTH;
        constexpr double xMin         = 0.0;
        constexpr double xMax         = 2.01;

        std::vector<double> xValues(static_cast<std::size_t>((xMax - xMin) * fs));
        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i] = xMin + static_cast<double>(i) / fs;
        }

        auto chart        = gr::graphs::ImChart<120, 16>({{xMin, xMax}, {-5.0, +5.0}});
        chart.axis_name_x = "time [s]";
        chart.axis_name_y = "IIR filter amplitude [a.u.]";

        const auto digitalBandPass = iir::designFilter<T>(Type::BANDPASS, {.order = 4UZ, .fLow = 4., .fHigh = 6., .fs = fs}, filterDesign);

        const auto secondOrderLowPass = FilterCoefficients<T>{{static_cast<T>(1. / 6474.5), static_cast<T>(2. / 6474.5), static_cast<T>(1. / 6474.5)}, {1, static_cast<T>(-1.96454), static_cast<T>(0.96515)}};
        printFilter("secondOrderLowPass", secondOrderLowPass);

        for (const auto& frequency : {3.0, 3.5, 5., 6.5, 7.0}) {
            auto filter = Filter<T, 32UZ>(digitalBandPass);
            // auto filter = Filter<double>(secondOrderLowPass);
            // auto filter = Filter<double>(simpleLowPass);
            std::vector<double> yValue(xValues.size());
            std::vector<double> yFiltered(xValues.size());
            std::transform(xValues.cbegin(), xValues.cend(), yValue.begin(), [frequency](double t) { return 3.0 * std::sin(2. * std::numbers::pi * frequency * t); });
            std::transform(yValue.cbegin(), yValue.cend(), yFiltered.begin(), [&filter](double y) { return filter.processOne(static_cast<T>(y)); });
            chart.draw(xValues, yFiltered, fmt::format("filtered@{}Hz", frequency));
        }

        chart.draw();
    };

    tag("visual") / "basic FIR/IIR step-response with UncertainValue Propagation"_test = [](const bool useFIR) {
        using namespace gr::filter::iir;
        using namespace gr::filter::fir;
        using enum gr::algorithm::window::Type;
        const double     noiseFigure   = useFIR ? 2.5 : 5.0;
        constexpr double fs            = 100.;
        constexpr double xMin          = 0.0;
        constexpr double xMax          = 2.01;
        constexpr double stepPosition  = 0.25;
        constexpr double stepAmplitude = 3.0;

        std::vector<double> xValues(static_cast<std::size_t>((xMax - xMin) * fs));
        std::vector<double> yValue(xValues.size());
        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i] = xMin + static_cast<double>(i) / fs;
        }
        std::transform(xValues.cbegin(), xValues.cend(), yValue.begin(), [](double t) { return t < stepPosition ? 0.0 : stepAmplitude; });

        fmt::println("");
        using Un = gr::UncertainValue<double>;
        for (const auto& frequency : {1.0, 2.0, 4.0}) {
            std::vector<FilterCoefficients<double>> digitalLowPass = useFIR ? std::vector{fir::designFilter<double>(Type::LOWPASS, {.order = 1UZ, .fLow = frequency, .fs = fs}, Hamming)} : iir::designFilter<double>(Type::LOWPASS, {.order = 2UZ, .fLow = frequency, .fs = fs}, BESSEL);
            auto                                    filter         = ErrorPropagatingFilter<Un>(digitalLowPass);
            filter.reset({0.0, noiseFigure});

            std::vector<Un>     yFiltered(xValues.size());
            std::vector<double> yMean(xValues.size());
            std::vector<double> yMin(xValues.size());
            std::vector<double> yMax(xValues.size());
            std::transform(yValue.cbegin(), yValue.cend(), yFiltered.begin(), [&filter, &noiseFigure](double y) { return filter.processOne({y, noiseFigure}); });
            std::transform(yFiltered.cbegin(), yFiltered.cend(), yMean.begin(), [](Un val) { return gr::value(val); });
            std::transform(yFiltered.cbegin(), yFiltered.cend(), yMin.begin(), [](Un val) { return gr::value(val) - gr::uncertainty(val); });
            std::transform(yFiltered.cbegin(), yFiltered.cend(), yMax.begin(), [](Un val) { return gr::value(val) + gr::uncertainty(val); });

            auto chart        = gr::graphs::ImChart<120, 18>({{xMin, xMax}, {-1.0, +4.0}});
            chart.axis_name_x = "time [s]";
            chart.axis_name_y = std::string(useFIR ? "FIR" : "IIR") + " filter amplitude [a.u.]";
            chart.draw(xValues, yValue, "reference");

            if (useFIR) {
                chart._lastColor = gr::graphs::Color::next(chart._lastColor);
            }
            const auto colourSave = chart._lastColor;
            chart.draw(xValues, yMin, "");
            chart._lastColor = colourSave;
            chart.draw(xValues, yMax, "");
            chart._lastColor = colourSave;
            chart.draw(xValues, yMean, fmt::format("LP@{}Hz - input {:.2} vs output {:.2} noise figure", frequency, noiseFigure, gr::uncertainty(yFiltered.back())));
            chart.draw();
            fmt::println("");
        }
    } | std::vector{true, false};

    "quantitative FIR/IIR step-response-pass with UncertainValue Propagation"_test = [](const bool useFIR) {
        using namespace gr::filter::iir;
        using namespace gr::filter::fir;
        using enum gr::algorithm::window::Type;
        const double     noiseFigure   = 5.0;
        constexpr double fs            = 100.;
        constexpr double flow          = 4.0;
        constexpr double xMin          = 0.0;
        constexpr double xMax          = 2.01;
        constexpr double stepPosition  = 1.0;
        constexpr double stepAmplitude = 3.0;

        std::vector<double> xValues(static_cast<std::size_t>((xMax - xMin) * fs));
        std::vector<double> yValue(xValues.size());
        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i] = xMin + static_cast<double>(i) / fs;
        }
        std::transform(xValues.cbegin(), xValues.cend(), yValue.begin(), [](double t) { return t < stepPosition ? 0.0 : stepAmplitude; });

        using Un = gr::UncertainValue<double>;

        std::vector<FilterCoefficients<double>> digitalLowPass = useFIR ? std::vector{fir::designFilter<double>(Type::LOWPASS, {.order = 1UZ, .fLow = flow, .fs = fs}, Hamming)} : iir::designFilter<double>(Type::LOWPASS, {.order = 2UZ, .fLow = flow, .fs = fs}, BESSEL);
        auto                                    filter         = ErrorPropagatingFilter<Un>(digitalLowPass);
        filter.reset({0.0, noiseFigure});

        std::vector<Un> yFiltered(xValues.size());
        std::transform(yValue.cbegin(), yValue.cend(), yFiltered.begin(), [&filter, &noiseFigure](double y) { return filter.processOne({y, noiseFigure}); });

        auto filterType = useFIR ? "FIR" : "IIR";
        expect(approx(0.0, gr::value(yFiltered.front()), 0.001)) << fmt::format("{} - initial value", filterType);
        expect(approx(stepAmplitude, gr::value(yFiltered.back()), 0.001)) << fmt::format("{} - last value", filterType);
        const double_t Neff = fs / flow;
        // FIR has a poorer noise rejection performance than IIR, thus the additional factor 1.5 relaxation.
        const double_t expectedNoiseFigure = noiseFigure / std::sqrt(Neff) * (useFIR ? 1.5 : 1.0);
        expect(le(gr::uncertainty(yFiltered[10]), expectedNoiseFigure)) << fmt::format("{} - initial value", filterType);
        expect(le(gr::uncertainty(yFiltered.back()), expectedNoiseFigure)) << fmt::format("{} - last value", filterType);
    } | std::vector{true, false};
};

const boost::ut::suite<"FIR FilterTool"> firFilterToolTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using enum Frequency;
    using enum ResponseType;
    using magic_enum::enum_name;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        // conditionally enable visual tests outside the CI
        boost::ext::ut::cfg<override> = {.tag = {"visual", "benchmarks"}};
    }

    using enum gr::algorithm::window::Type;
    "IIR digital filter"_test = [](gr::algorithm::window::Type filterDesign) {
        using enum gr::filter::Type;

        "IIR digital filter"_test = [&filterDesign](Type filterType) {
            constexpr auto kFilterParams = FilterParameters{.order = 4UZ, .fLow = 1.0, .fHigh = 10.0, .attenuationDb = 60, .fs = 1000.0};

            // compute analog and digital filters

            expect(nothrow([&filterDesign, &filterType, &kFilterParams]() { std::ignore = fir::designFilter<double>(filterType, kFilterParams, filterDesign); })) //
                << fmt::format("({}, {}, {}) creating digital unbound filter unexpectedly throws", enum_name(filterDesign), kFilterParams.order, enum_name(filterType));
            const FilterCoefficients<double> digitalFilter = fir::designFilter<double>(filterType, kFilterParams, filterDesign);

            // coarse response test
            constexpr double kTolerance = 0.01;
            constexpr double kDC        = 0.0;
            constexpr double kNyquist   = 0.48;
            const double     kOmega0    = std::sqrt(kFilterParams.fLow * kFilterParams.fHigh) / kFilterParams.fs;
            switch (filterType) {
            case BANDSTOP: expect(le(calculateResponse<Normalised, Magnitude>(kOmega0, digitalFilter), 5. * kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kOmega0, digitalFilter)); [[fallthrough]];
            case LOWPASS: expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kDC, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at DC: {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kDC, digitalFilter)); break;
            case HIGHPASS:
                expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kNyquist, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at NQ: {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kNyquist, digitalFilter));
                expect(le(calculateResponse<Normalised, Magnitude>(kDC, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band: {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kOmega0, digitalFilter));
                break;
            case BANDPASS:
                expect(approx(1.0, calculateResponse<Normalised, Magnitude>(kOmega0, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) has non-unity gain at omega0: {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kOmega0, digitalFilter));
                expect(le(calculateResponse<Normalised, Magnitude>(kDC, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (DC): {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kDC, digitalFilter));
                expect(le(calculateResponse<Normalised, Magnitude>(kNyquist, digitalFilter), kTolerance)) << fmt::format("({}, {}, {}) insufficient gain in stop-band (NQ): {:.4f}", enum_name(filterDesign), enum_name(filterType), kFilterParams.order, calculateResponse<Normalised, Magnitude>(kNyquist, digitalFilter));
                break;
            }

            // N.B. cannot test magnitude response shape since there are not analog-vs-digital equivalents to compare
        } | std::vector{LOWPASS, HIGHPASS, BANDPASS, BANDSTOP}; //
    } | std::vector{Kaiser, Hamming, Hann};

    tag("visual") / "basic fir tests"_test = []() {
        using namespace gr::graphs;
        constexpr auto kFilterParams = FilterParameters{.order = 4UZ, .fLow = 1.0, .fHigh = 10.0, .gain = 0.5, .attenuationDb = 50., .fs = 1000.0};

        using enum gr::algorithm::window::Type;
        const auto window = Kaiser;
        using enum gr::filter::Type;

        // generate test frequency range
        auto sequence_range = [](auto start, auto end, auto step) {
            using namespace std::views; // for iota and transform
            return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
        };
        std::vector<double> frequencies;
        for (auto subrange : {sequence_range(0.1, 0.9, 0.01), sequence_range(1.0, 9.0, 0.01), sequence_range(10.0, 90.0, 0.1), sequence_range(100.0, 490.0, 10.0)}) {
            std::ranges::move(subrange, std::back_inserter(frequencies));
        }
        const auto          lowPassFilter    = fir::designFilter<double>(LOWPASS, kFilterParams, window);
        const auto          highPassFilter   = fir::designFilter<double>(HIGHPASS, kFilterParams, window);
        const auto          bandPassFilter   = fir::designFilter<double>(BANDPASS, kFilterParams, window);
        std::vector<double> lowPassResponse  = calculateFrequencyResponse<MagnitudeDB>(frequencies, kFilterParams.fs, lowPassFilter);
        std::vector<double> highPassResponse = calculateFrequencyResponse<MagnitudeDB>(frequencies, kFilterParams.fs, highPassFilter);
        std::vector<double> bandPassResponse = calculateFrequencyResponse<MagnitudeDB>(frequencies, kFilterParams.fs, bandPassFilter);

        // plots
        auto chart        = gr::graphs::ImChart<119, 21, LogAxisTransform>({{0.1, kFilterParams.fs}, {-45., +5}});
        chart.axis_name_x = "[Hz]";
        chart.axis_name_y = "[dB]";
        chart.draw(frequencies, lowPassResponse, fmt::format("FIR low(N={})", countFilterCoefficients(lowPassFilter)));
        chart.draw(frequencies, highPassResponse, fmt::format("high({})", countFilterCoefficients(highPassFilter)));
        chart.draw(frequencies, bandPassResponse, fmt::format("band-pass({})", countFilterCoefficients(bandPassFilter)));
        chart.draw();

        auto highPassResponses        = gr::graphs::ImChart<119, 21, LogAxisTransform>({{0.1, kFilterParams.fs}, {-45., +5}});
        highPassResponses.axis_name_x = "[Hz]";
        highPassResponses.axis_name_y = "band-stop response [dB]";
        for (auto& windowType : {Kaiser, Hamming, Rectangular, Hann, BlackmanNuttall}) {
            auto                filter   = fir::designFilter<double>(BANDSTOP, kFilterParams, windowType);
            std::vector<double> response = calculateFrequencyResponse<MagnitudeDB>(frequencies, kFilterParams.fs, filter);
            highPassResponses.draw(frequencies, response, fmt::format("{}({})", enum_name(windowType), filter.b.size()));
        }
        highPassResponses.draw();
    };

    tag("visual") / "basic FIR band-pass filter frequency"_test = []() {
        constexpr auto   kFilterParams = FilterParameters{.order = 2UZ, .fLow = 4., .fHigh = 6., .attenuationDb = 50, .fs = 1000.0};
        constexpr double xMin          = 0.0;
        constexpr double xMax          = 2.01;

        std::vector<double> xValues(static_cast<std::size_t>((xMax - xMin) * kFilterParams.fs));
        for (std::size_t i = 0UZ; i < xValues.size(); ++i) {
            xValues[i] = xMin + static_cast<double>(i) / kFilterParams.fs;
        }

        auto chart        = gr::graphs::ImChart<120, 16>({{xMin, xMax}, {-5.0, +5.0}});
        chart.axis_name_x = "time [s]";
        chart.axis_name_y = "FIR filter amplitude [a.u.]";

        const auto window          = gr::algorithm::window::Type::Kaiser;
        const auto digitalBandPass = fir::designFilter<double>(Type::BANDPASS, kFilterParams, window);
        // printFilter("digitalBandPass", digitalBandPass);
        for (const auto& frequency : {3.0, 3.5, 5., 6.5, 7.0}) {
            auto                filter = Filter<double>(digitalBandPass);
            std::vector<double> yValue(xValues.size());
            std::vector<double> yFiltered(xValues.size());
            std::transform(xValues.cbegin(), xValues.cend(), yValue.begin(), [frequency](double t) { return 3.0 * std::sin(2. * std::numbers::pi * frequency * t); });
            std::transform(yValue.cbegin(), yValue.cend(), yFiltered.begin(), [&filter](double y) { return filter.processOne(y); });
            expect(nothrow([&] { chart.draw(xValues, yFiltered, fmt::format("filtered@{}Hz", frequency)); })) << fmt::format("filtered@{}Hz does not throw", frequency);
        }

        expect(nothrow([&] { chart.draw(); }));
    };
};

const boost::ut::suite<"IIR & FIR Benchmarks"> filterBenchmarks = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using enum Frequency;
    using enum ResponseType;
    using enum gr::filter::Type;
    using enum iir::Design;
    using magic_enum::enum_name;

    "IIR vs. FIR coefficients"_test = [](Type filterType) {
        constexpr FilterParameters kFilterParameter{.order = 4UZ, .fLow = 4., .fHigh = 6., .attenuationDb = 50., .fs = 1000.};
        const auto                 iirFilter = iir::designFilter<double>(filterType, kFilterParameter);
        const auto                 firFilter = fir::designFilter<double>(filterType, kFilterParameter);
        std::size_t                nIIR      = countFilterCoefficients(iirFilter);
        std::size_t                nFIR      = countFilterCoefficients(firFilter);
        expect(le(nIIR, nFIR)) << fmt::format("{} complexity: #IIR {} vs. #FIR {}", enum_name(filterType), nIIR, nFIR);
        tag("visual") / "filter benchmarks"_test = [&] { fmt::print("{} complexity: #IIR {} vs. #FIR {}\n", enum_name(filterType), nIIR, nFIR); };
    } | std::vector{LOWPASS, HIGHPASS, BANDPASS, BANDSTOP};

    "IIR vs. FIR magnitude and phase"_test = []() {
        using namespace gr::graphs;
        using enum gr::algorithm::window::Type;

        constexpr FilterParameters kFilterParameter{.order = 4UZ, .fLow = 10., .gain = 0.5, .fs = 1000.};
        const auto                 iirFilter1 = iir::designFilter<double>(LOWPASS, kFilterParameter, BUTTERWORTH);
        const auto                 iirFilter2 = iir::designFilter<double>(LOWPASS, kFilterParameter, BESSEL);
        const auto                 firFilter1 = fir::designFilter<double>(LOWPASS, kFilterParameter, Kaiser);
        const auto                 firFilter2 = fir::designFilter<double>(LOWPASS, kFilterParameter, Hamming);

        // generate test frequency range
        auto sequence_range = [](auto start, auto end, auto step) {
            using namespace std::views; // for iota and transform
            return iota(1, (end - start) / step + 2) | transform([=](auto i) { return start + step * (i - 1); });
        };
        std::vector<double> frequencies;
        for (auto subrange : {sequence_range(0.1, 0.9, 0.01), sequence_range(1.0, 9.0, 0.01), sequence_range(10.0, 90.0, 0.1), sequence_range(100.0, 490.0, 10.0)}) {
            std::ranges::move(subrange, std::back_inserter(frequencies));
        }

        //
        const auto magResponse = [&frequencies, &kFilterParameter](auto& filter) { return calculateFrequencyResponse<MagnitudeDB>(frequencies, kFilterParameter.fs, filter); };

        auto magnitudeChart        = gr::graphs::ImChart<119, 21, LogAxisTransform>({{0.1, kFilterParameter.fs}, {-45., +5}});
        magnitudeChart.axis_name_x = "[Hz]";
        magnitudeChart.axis_name_y = "IIR vs. FIR response [dB]";
        magnitudeChart.draw(frequencies, magResponse(iirFilter1), fmt::format("Butterworth(N={})", countFilterCoefficients(iirFilter1)));
        magnitudeChart.draw(frequencies, magResponse(iirFilter2), fmt::format("Bessel({})", countFilterCoefficients(iirFilter1)));
        magnitudeChart.draw(frequencies, magResponse(firFilter1), fmt::format("Kaiser({})", countFilterCoefficients(firFilter1)));
        magnitudeChart.draw(frequencies, magResponse(firFilter2), fmt::format("Hamming({})", countFilterCoefficients(firFilter2)));

        magnitudeChart.draw();

        const auto phaseResponse = [&frequencies, &kFilterParameter](auto& filter, double corr = 0UZ, double offset = 0.) {
            const auto groupDelay = (corr + static_cast<double>(countFilterCoefficients(filter)) - 1.0) / 2.0 / kFilterParameter.fs; // just a coarse estimate

            auto normalisedPhase = calculateFrequencyResponse<PhaseDegrees>(frequencies, kFilterParameter.fs, filter);
            std::transform(normalisedPhase.begin(), normalisedPhase.end(), frequencies.begin(), normalisedPhase.begin(),                                        //
                [groupDelay, offset](auto phase, auto frequency) { return std::fmod(offset + phase + groupDelay * frequency * 360.0 + 180.0, 360.) - 180.0; }); // [-180°, 180°]
            return normalisedPhase;
        };

        auto phaseChart        = gr::graphs::ImChart<130, 33, LogAxisTransform>({{0.1, kFilterParameter.fs}, {-.5, +.5}});
        phaseChart.axis_name_x = "[Hz]";
        phaseChart.axis_name_y = "IIR vs. FIR rel. non-linear phase response [°]";
        phaseChart.draw(frequencies, phaseResponse(iirFilter1, 72.5), fmt::format("Butterworth(N={})", countFilterCoefficients(iirFilter1)));
        phaseChart.draw(frequencies, phaseResponse(iirFilter2, 55.8), fmt::format("Bessel({})", countFilterCoefficients(iirFilter1)));
        phaseChart.draw(frequencies, phaseResponse(firFilter1, 0., -0.1), fmt::format("Kaiser({}) -0.1°", countFilterCoefficients(firFilter1)));
        phaseChart.draw(frequencies, phaseResponse(firFilter2, 0., -0.05), fmt::format("Hamming({}) -0.05°", countFilterCoefficients(firFilter2)));

        phaseChart.draw();
    };

    tag("benchmarks") / "filter benchmarks"_test = []<typename T>(T) {
        using namespace benchmark;
        using namespace gr::filter::iir;
        constexpr FilterParameters kFilterParameter{.order = 4UZ, .fLow = 4., .fHigh = 6., .attenuationDb = 50., .fs = 1000.};
        constexpr Design           filterDesign = BUTTERWORTH;

        constexpr std::size_t nSamples = 100'000;
        std::vector<T>        yValues(nSamples);
        const T               centreFrequency = std::sqrt(static_cast<T>(kFilterParameter.fLow * kFilterParameter.fHigh));
        for (std::size_t i = 0UZ; i < yValues.size(); ++i) {
            yValues[i] = std::sin(static_cast<T>(2) * std::numbers::pi_v<T> * centreFrequency / static_cast<T>(kFilterParameter.fs) * static_cast<T>(i));
        }

        auto processSignal = [](auto& filter, const std::vector<T>& signalValues) -> T {
            T           actualGain       = 0.0;
            std::size_t processedSamples = 0UZ;
            for (auto& signalValue : signalValues) {
                T filteredValue = filter.processOne(signalValue);
                processedSamples++;
                if (processedSamples > signalValues.size() / 2UZ) { // ignore initial transient
                    actualGain = std::max(actualGain, std::abs(filteredValue));
                }
            }
            return actualGain;
        };

        auto processSignalErrors = [](auto& filter, const std::vector<T>& signalValues) -> T {
            T           actualGain       = 0.0;
            std::size_t processedSamples = 0UZ;
            for (auto& signalValue : signalValues) {
                gr::UncertainValue<T> filteredValue = filter.processOne(signalValue);
                processedSamples++;
                if (processedSamples > signalValues.size() / 2UZ) { // ignore initial transient
                    actualGain = std::max(actualGain, std::abs(gr::value(filteredValue)));
                }
            }
            return actualGain;
        };

        {
            T                       actualGain = 0.0;
            gr::HistoryBuffer<T, 8> buffer;
            ::benchmark::benchmark<10>(fmt::format("HistoryBuffer<{}>", gr::meta::type_name<T>()), nSamples) = [&actualGain, &buffer, &yValues] {
                for (auto& yValue : yValues) {
                    buffer.push_back(yValue);
                    actualGain = std::max(actualGain, buffer[0]);
                }
            };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1)));
        }
        const auto digitalBandPass = iir::designFilter<T>(Type::BANDPASS, kFilterParameter, filterDesign);
        {
            T    actualGain                                       = 0.0;
            auto filter                                           = Filter<T, 32UZ, Form::DF_I, std::execution::unseq>(digitalBandPass);
            "IIR DF_I exec::unseq"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("IIR DF_I exec::unseq approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        {
            T    actualGain                                     = 0.0;
            auto filter                                         = Filter<T, 32UZ, Form::DF_I, std::execution::par>(digitalBandPass);
            "IIR DF_I exec::par"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("IIR DF_I exec::par approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        {
            T    actualGain                                     = 0.0;
            auto filter                                         = Filter<T, 32UZ, Form::DF_I>(digitalBandPass);
            "IIR DF_I exec::seq"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("IIR DF_I exec::sec approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        {
            T    actualGain                                      = 0.0;
            auto filter                                          = Filter<T, 32UZ, Form::DF_II>(digitalBandPass);
            "IIR DF_II exec::seq"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("DF_II exec::seq approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        {
            T    actualGain                                                 = 0.0;
            auto filter                                                     = Filter<T, 32UZ, Form::DF_I_TRANSPOSED>(digitalBandPass);
            "IIR DF_I_TRANSPOSED exec::seq "_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("DF_I_TRANSPOSED exec::seq approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
#if not defined(__EMSCRIPTEN__)
        {
            T    actualGain                                                 = 0.0;
            auto filter                                                     = Filter<T, 32UZ, Form::DF_II_TRANSPOSED>(digitalBandPass);
            "IIR DF_II_TRANSPOSED exec::seq"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << std::format("DF_II_TRANSPOSED exec::seq approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
#endif
        const auto digitalBandPassFir = fir::designFilter<T>(Type::BANDPASS, kFilterParameter);
        {
            T    actualGain                              = 0.0;
            auto filter                                  = Filter<T>(digitalBandPassFir);
            "FIR default"_benchmark.repeat<10>(nSamples) = [&processSignal, &actualGain, &filter, &yValues] { actualGain = processSignal(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("FIR approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        ::benchmark::results::add_separator();
        {
            T    actualGain                                     = 0.0;
            auto filter                                         = Filter<gr::UncertainValue<T>>(digitalBandPassFir);
            "FIR w/ uncertainty"_benchmark.repeat<10>(nSamples) = [&processSignalErrors, &actualGain, &filter, &yValues] { actualGain = processSignalErrors(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("FIR approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        {
            T    actualGain                                     = 0.0;
            auto filter                                         = Filter<gr::UncertainValue<T>>(digitalBandPass);
            "IIR w/ uncertainty"_benchmark.repeat<10>(nSamples) = [&processSignalErrors, &actualGain, &filter, &yValues] { actualGain = processSignalErrors(filter, yValues); };
            expect(approx(actualGain, static_cast<T>(1), static_cast<T>(0.1))) << fmt::format("IIR approx settling gain threshold for {}", gr::meta::type_name<T>());
        }
        ::benchmark::results::add_separator();
    } | std::tuple<double, float>{1.0, 1.0f};
};

int main() { /* not needed for UT */ }
