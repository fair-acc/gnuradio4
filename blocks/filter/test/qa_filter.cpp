#include <boost/ut.hpp>
#include <span>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/PolyphaseArbitraryResampler.hpp>
#include <gnuradio-4.0/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/filter/PolyphaseSynthesizer.hpp>
#include <gnuradio-4.0/filter/RationalResampler.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/test/DeviceExpectation.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

template<typename T, typename Range>
requires std::floating_point<T>
constexpr size_t estimate_settling_time(const Range& step_response, std::size_t offset = 0, T step_value = 1.0, T threshold = 0.001) {
    if (offset >= step_response.size()) {
        throw std::out_of_range("Offset is greater than the size of the step response.");
    }
    const T lower_bound = step_value - threshold;
    const T upper_bound = step_value + threshold;

    auto begin = step_response.begin() + static_cast<typename Range::difference_type>(offset);
    auto end   = step_response.end();

    auto it = std::find_if(begin, end, [lower_bound, upper_bound](T sample) { return sample >= lower_bound && sample <= upper_bound; });

    // If no such sample is found, return an error
    if (it == end) {
        throw gr::exception("No settling found within the given threshold.");
    }

    // Check if all subsequent samples stay within the acceptable range
    auto it_next = it;
    while (it_next != end) {
        it_next = std::find_if(it_next, end, [lower_bound, upper_bound](T sample) { return sample < lower_bound || sample > upper_bound; });

        if (it_next != end) {
            it = it_next++;
        }
    }

    // Return the settling time (or index)
    return static_cast<std::size_t>(std::distance(begin, it));
}

const boost::ut::suite SequenceTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;

    "FIR and IIR general tests"_test = [] {
        Tensor<double> fir_coeffs(10, 0.1); // box car filter
        Tensor<double> iir_coeffs_b(data_from, {0.55, 0.0});
        Tensor<double> iir_coeffs_a(data_from, {1.0, -0.45});

        // Create FIR and IIR filter instances
        fir_filter<double, ConvolutionDomain::Time> fir_filter;
        fir_filter.b = fir_coeffs;

        iir_filter<double, IIRForm::DF_I> iir_filter1;
        iir_filter1.b = iir_coeffs_b;
        iir_filter1.a = iir_coeffs_a;
        iir_filter<double, IIRForm::DF_II> iir_filter2;
        iir_filter2.b = iir_coeffs_b;
        iir_filter2.a = iir_coeffs_a;

        std::vector<double> step(20UZ);
        for (std::size_t i = 0UZ; i < step.size(); ++i) {
            step[i] = (i == 0) ? 0.0 : 1.0; // step function
        }

        // the FIR answers over a window primed with zeros, so it is sample-count preserving: as many outputs as
        // inputs, and y[0] belongs to x[0]
        std::vector<double>     fir_response(step.size());
        std::span<const double> stepSpan{step};
        std::span<double>       firSpan{fir_response};
        std::ignore = fir_filter.processBulk(stepSpan, firSpan);

        // the true step response of a unit-sum moving average: the transient really is 0.1, 0.2, ... not zeros
        for (std::size_t i = 1UZ; i < fir_filter.b.size(); ++i) {
            expect(approx(fir_response[i], 0.1 * static_cast<double>(i), 1e-9)) << "the start-up transient must be emitted, not padded away";
        }

        std::vector<double> iir_response1;
        std::vector<double> iir_response2;
        for (std::size_t i = 0UZ; i < step.size(); ++i) {
            iir_response1.push_back(iir_filter1.filterOne(step[i]));
            iir_response2.push_back(iir_filter1.filterOne(step[i]));
        }
        expect(eq(fir_response[0], 0.0));
        expect(eq(iir_response1[0], 0.0));
        expect(eq(iir_response2[0], 0.0));

        const std::size_t fir_settling_time  = estimate_settling_time<double>(fir_response);
        const std::size_t iir_settling_time1 = estimate_settling_time<double>(iir_response1);
        const std::size_t iir_settling_time2 = estimate_settling_time<double>(iir_response2);
        expect(eq(fir_settling_time, 10u)) << "FIR settling time";
        expect(eq(iir_settling_time1, 5u)) << "IIR (I) settling time";
        expect(eq(iir_settling_time2, 5u)) << "IIR (II) settling time";

        std::println("FIR      filter settling time: {} ms", fir_settling_time);
        std::println("IIR (I)  filter settling time: {} ms", iir_settling_time1);
        std::println("IIR (II) filter settling time: {} ms", iir_settling_time2);
    };

    "IIR equality tests"_test = [] {
        Tensor<double> iir_coeffs_b(data_from, {0.020083365564211, 0.040166731128423, 0.020083365564211});
        Tensor<double> iir_coeffs_a(data_from, {1.0, -1.561018075800718, 0.641351538057563});

        iir_filter<double, IIRForm::DF_I> iir_filter_I;
        iir_filter_I.b = iir_coeffs_b;
        iir_filter_I.a = iir_coeffs_a;
        iir_filter<double, IIRForm::DF_II> iir_filter_II;
        iir_filter_II.b = iir_coeffs_b;
        iir_filter_II.a = iir_coeffs_a;
        iir_filter<double, IIRForm::DF_I_TRANSPOSED> iir_filter_IT;
        iir_filter_IT.b = iir_coeffs_b;
        iir_filter_IT.a = iir_coeffs_a;
        iir_filter<double, IIRForm::DF_II_TRANSPOSED> iir_filter_IIT;
        iir_filter_IIT.b = iir_coeffs_b;
        iir_filter_IIT.a = iir_coeffs_a;

        constexpr double tolerance = 0.00001;
        for (std::size_t i = 0UL; i < 20; ++i) {
            const double input     = (i == 0) ? 0.0 : 1.0; // Step function
            const auto   form_I    = iir_filter_I.filterOne(input);
            const auto   form_II   = iir_filter_II.filterOne(input);
            const auto   form_I_T  = iir_filter_IT.filterOne(input);
            const auto   form_II_T = iir_filter_IIT.filterOne(input);
            expect(approx(form_II, form_I, tolerance)) << "direct form II";
            expect(approx(form_I_T, form_I, tolerance)) << "direct form I - transposed";
            expect(approx(form_II_T, form_I, tolerance)) << "direct form II - transposed";

#if defined(__GNUC__) && !defined(__OPTIMIZE__)
            std::print("input[{:2}]={}-> IIR= {:4.2f} (I) {:4.2f} (II) {:4.2f} (I-T) {:4.2f} (II-T)\n", //
                i, input, form_I, form_II, form_I_T, form_II_T);
#endif
        }
    };
};

template<typename T, gr::filter::FilterType type>
struct FilterTestParam {
    using value_type                  = T;
    static constexpr auto filter_type = type;
};

const boost::ut::suite<"Basic[Decimating]Filter"> BasicFilterTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using namespace std::string_literals;

    constexpr static auto maxOp = []<typename T>(const T a, const T b) -> bool { return std::abs(gr::value(a)) < std::abs(gr::value(b)); };

    constexpr static float       kSampleRate       = 1000.0;
    constexpr static float       kLowCutoff        = 100.0;
    constexpr static std::size_t kFilterOrder      = 4;
    constexpr static std::size_t kNumSamples       = 1000;
    constexpr static std::size_t kFilterDecimation = 5;

    "BasicFilter - Low-pass Filter Test"_test =
        []<typename TTestParameter>() {
            using T         = typename TTestParameter::value_type;
            using ValueType = meta::fundamental_base_value_type_t<T>;
            auto filterType = TTestParameter::filter_type;

            BasicFilter<T> filter;
            filter.filter_type       = filterType;
            filter.filter_response   = filter::Type::LOWPASS;
            filter.filter_order      = kFilterOrder;
            filter.f_low             = kLowCutoff;
            filter.sample_rate       = kSampleRate;
            filter.iir_design_method = filter::iir::Design::CHEBYSHEV1;
            filter.fir_design_method = algorithm::window::Type::Hamming;
            filter.designFilter(); // triggers filter re-computation and setting of internal enums

            "verify in-band signal passes through"_test = [&filter] {
                std::vector<T> outputSignal;
                outputSignal.reserve(kNumSamples);
                T phase = 0;
                for (std::size_t i = 0UZ; i < 2 * kNumSamples; i++) {
                    // generate a sine wave signal with a frequency below the cutoff
                    phase += T{2} * std::numbers::pi_v<ValueType> * static_cast<ValueType>(50) / static_cast<ValueType>(kSampleRate);
                    if (i < kNumSamples) { // ignore initial transient
                        std::ignore = filter.filterOne(gr::math::sin(phase));
                    } else {
                        outputSignal.push_back(filter.filterOne(gr::math::sin(phase)));
                    }
                }

                ValueType maxOutput = std::abs(gr::value(*std::ranges::max_element(outputSignal, maxOp)));
                expect(ge(maxOutput, static_cast<ValueType>(.9f))) << std::format("{} filter should pass in-band frequencies: max output {}", filter.filter_type, maxOutput);
            };

            "verify out-of-band signal is attenuated"_test = [&filter] {
                std::vector<T> outputSignal;
                outputSignal.reserve(kNumSamples);
                T phase = 0;
                for (std::size_t i = 0UZ; i < 2 * kNumSamples; i++) {
                    // generate a sine wave signal with a frequency below the cutoff
                    phase += T{2} * std::numbers::pi_v<ValueType> * static_cast<ValueType>(300) / static_cast<ValueType>(kSampleRate);
                    if (i < kNumSamples) { // ignore initial transient
                        std::ignore = filter.filterOne(gr::math::sin(phase));
                    } else {
                        outputSignal.push_back(filter.filterOne(gr::math::sin(phase)));
                    }
                }

                ValueType maxOutput = std::abs(gr::value(*std::ranges::max_element(outputSignal, maxOp)));
                expect(le(maxOutput, static_cast<ValueType>(.2f))) << std::format("{} filter should attenuate out-of-band frequencies: max output {}", filter.filter_type, maxOutput);
            };
        } |
        std::tuple<FilterTestParam<float, FilterType::FIR>,           //
            FilterTestParam<double, FilterType::FIR>,                 //
            FilterTestParam<UncertainValue<float>, FilterType::FIR>,  //
            FilterTestParam<UncertainValue<double>, FilterType::FIR>, //
            FilterTestParam<float, FilterType::IIR>,                  //
            FilterTestParam<double, FilterType::IIR>,                 //
            FilterTestParam<UncertainValue<float>, FilterType::IIR>,  //
            FilterTestParam<UncertainValue<double>, FilterType::IIR>>{};

    "BasicDecimatingFilter - Low-pass Filter Test"_test = [](const FilterType& filterType) {
        using T = double;

        // Instantiate the BasicDecimatingFilter with the desired decimation rate
        BasicDecimatingFilter<T> filter;
        filter.filter_type       = filterType;
        filter.filter_response   = filter::Type::LOWPASS;
        filter.filter_order      = kFilterOrder;
        filter.f_low             = kLowCutoff;
        filter.sample_rate       = kSampleRate;
        filter.iir_design_method = filter::iir::Design::CHEBYSHEV1;
        filter.fir_design_method = algorithm::window::Type::Hamming;
        filter.decimation        = kFilterDecimation;
        filter.designFilter(); // triggers filter re-computation and setting of internal enums

        expect(eq(filter.input_chunk_size, kFilterDecimation)) << "kFilterDecimation type mismatch";

        "verify in-band signal passes through"_test = [&filter] {
            std::vector<T> inputSignal(kNumSamples);
            std::vector<T> outputSignal(kNumSamples / kFilterDecimation);

            T    phase          = 0;
            auto generateSample = [&phase]() {
                // generate a sine wave signal with a frequency below the cutoff
                phase += 2 * std::numbers::pi_v<T> * static_cast<T>(50) / static_cast<T>(kSampleRate);
                return std::sin(phase);
            };
            const auto filterDecimated = [&filter](std::span<const T> samples, std::span<T> decimated) {
                std::size_t outIndex = 0UZ;
                for (std::size_t i = 0UZ; i < samples.size() && outIndex < decimated.size(); ++i) {
                    const T filtered = filter.filterOne(samples[i]);
                    if (i % kFilterDecimation == 0UZ) {
                        decimated[outIndex++] = filtered;
                    }
                }
            };
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);

            // 50 Hz at the decimated 200 Hz: consecutive outputs are 90 degrees apart, so each pair gives the amplitude whatever the filter's phase
            const T amplitude = std::hypot(outputSignal[outputSignal.size() - 2UZ], outputSignal.back());
            expect(ge(amplitude, T{0.9})) << std::format("{} filter should pass in-band frequencies: amplitude {}", filter.filter_type, amplitude);
        };

        "verify out-of-band signal is attenuated"_test = [&filter] {
            std::vector<T> inputSignal(kNumSamples);
            std::vector<T> outputSignal(kNumSamples / kFilterDecimation);

            T    phase          = 0;
            auto generateSample = [&phase]() {
                // generate a sine wave signal with a frequency above the cutoff
                phase += 2 * std::numbers::pi_v<T> * T(300) / T(kSampleRate);
                return std::sin(phase);
            };
            const auto filterDecimated = [&filter](std::span<const T> samples, std::span<T> decimated) {
                std::size_t outIndex = 0UZ;
                for (std::size_t i = 0UZ; i < samples.size() && outIndex < decimated.size(); ++i) {
                    const T filtered = filter.filterOne(samples[i]);
                    if (i % kFilterDecimation == 0UZ) {
                        decimated[outIndex++] = filtered;
                    }
                }
            };
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);

            double maxOutput = std::abs(*std::ranges::max_element(outputSignal, maxOp));
            expect(le(maxOutput, T{0.2})) << std::format("{} filter should attenuate out-of-band frequencies: max output {}", filter.filter_type, maxOutput);
        };
    } | std::vector<FilterType>({FilterType::FIR, FilterType::IIR});

    "Decimator - Low-pass Filter Test"_test = [] {
        using namespace gr::testing;
        using T = float;

        constexpr float      kInputRate        = 10'000.f;
        constexpr gr::Size_t kDecimationFactor = 10U;
        constexpr gr::Size_t kInputSamples     = 100U;

        gr::Graph flow;
        auto&     source    = flow.emplaceBlock<TagSource<T>>({{"sample_rate", kInputRate}, {"n_samples_max", kInputSamples}});
        auto&     decimator = flow.emplaceBlock<gr::filter::Decimator<T>>({{"decimation", kDecimationFactor}});
        auto&     sink      = flow.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", kInputSamples / kDecimationFactor}});
        expect(flow.connect<"out", "in">(source, decimator).has_value());
        expect(flow.connect<"out", "in">(decimator, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(decimator.decimation, kDecimationFactor));
        expect(eq(decimator.output_chunk_size, static_cast<gr::Size_t>(1)));
        expect(eq(decimator.input_chunk_size, kDecimationFactor));
        expect(eq(sink._nSamplesProduced, kInputSamples / kDecimationFactor));
        expect(eq(sink.sample_rate, kInputRate / static_cast<float>(kDecimationFactor))) << "rate seen downstream";
    };

    "BasicDecimatingFilter - keeps its input sample_rate"_test = [] {
        // the filter's own 'sample_rate' is its input rate; only the downstream tag carries the decimated rate.
        using namespace gr::testing;
        using T = float;

        constexpr float      kInputRate  = 32'000.f; // decimated rate must differ from the TagSink default, else the sink assert passes without a tag
        constexpr gr::Size_t kDecimation = 10U;
        constexpr gr::Size_t kNSamples   = 4'000U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<T>>({{"sample_rate", kInputRate}, {"n_samples_max", kNSamples}});
        auto&     filter = flow.emplaceBlock<BasicDecimatingFilter<T>>({{"sample_rate", kInputRate}, {"f_low", 400.f}, {"decimation", kDecimation}});
        auto&     sink   = flow.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", kNSamples / kDecimation}});

        expect(flow.connect<"out", "in">(source, filter).has_value());
        expect(flow.connect<"out", "in">(filter, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialise scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << "scheduler run";

        expect(eq(filter.sample_rate, kInputRate)) << "filter member holds its input rate";
        expect(eq(gr::test::get_value_or_fail<float>(*filter.settings().get("sample_rate")), kInputRate)) << "filter settings().get() must agree with its member";
        expect(eq(sink.sample_rate, kInputRate / static_cast<float>(kDecimation))) << "rate seen downstream";
    };
};

namespace basic_filter_test {
using namespace gr::testing;

/// the block owns its consume/publish now, so a case drives it through a graph rather than calling the body
template<typename TPrepare>
[[nodiscard]] inline std::vector<float> runBasic(std::string_view domain, gr::Size_t nSamples, TPrepare prepare) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::BasicFilter<float>>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    prepare(dut);
    dut.designFilter();

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());
    return std::vector<float>(sink._samples.begin(), sink._samples.end());
}

[[nodiscard]] inline std::vector<std::string_view> servedDomains() {
    std::vector<std::string_view> domains{"host"};
    for (std::string_view candidate : {"host:sycl", "gpu:sycl"}) {
        if (gr::device::DeviceContextRegistry::instance().isServedExactly(candidate)) {
            domains.push_back(candidate);
        }
    }
    return domains;
}
} // namespace basic_filter_test

const boost::ut::suite<"BasicFilter axes"> BasicFilterAxisTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using T = float;

    static const std::vector<T> taps{0.25f, 0.5f, 0.25f};

    "manual coefficients are the ones that filter"_test = [] {
        BasicFilter<T> filter;
        filter.filter_type        = FilterType::FIR;
        filter.coefficient_source = CoefficientSource::Manual;
        filter.b                  = gr::Tensor<T>(taps);
        filter.designFilter();

        std::vector<T> impulseResponse{filter.filterOne(T{1})};
        for (std::size_t i = 1UZ; i < taps.size(); ++i) {
            impulseResponse.push_back(filter.filterOne(T{0}));
        }
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            expect(approx(impulseResponse[i], taps[i], 1e-6f)) << std::format("tap {} reproduced by an impulse", i);
        }
    };

    "the same taps give the same samples in either domain"_test = [] {
        const std::vector<float> byTaps      = basic_filter_test::runBasic("host", 4096U, [](auto& dut) {
            dut.filter_type        = FilterType::FIR;
            dut.filter_domain      = ConvolutionDomain::Time;
            dut.coefficient_source = CoefficientSource::Manual;
            dut.b                  = gr::Tensor<float>(taps);
            dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only: state the denominator too
        });
        const std::vector<float> byTransform = basic_filter_test::runBasic("host", 4096U, [](auto& dut) {
            dut.filter_type        = FilterType::FIR;
            dut.filter_domain      = ConvolutionDomain::Frequency;
            dut.coefficient_source = CoefficientSource::Manual;
            dut.b                  = gr::Tensor<float>(taps);
            dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only: state the denominator too
            dut.outputs_per_frame  = 256U;
        });

        expect(gt(byTaps.size(), 1000UZ)) << "the tap arm must have produced output";
        expect(gt(byTransform.size(), 1000UZ)) << "the transform arm must have produced output";
        const std::size_t common = std::min(byTaps.size(), byTransform.size());
        // the transform arm drops the wrap-around of its first frame, so the two streams are compared where both
        // carry a settled window
        std::size_t agreeing = 0UZ;
        for (std::size_t i = taps.size(); i < common; ++i) {
            if (std::abs(byTaps[i] - byTransform[i - taps.size() + 1UZ]) < 1e-3f) {
                ++agreeing;
            }
        }
        expect(eq(agreeing, common - taps.size())) << "every sample past the ramp-up must agree, not merely most of them";
    };

    "a designed low-pass with gain 2 settles to twice a constant input"_test = [] {
        for (const FilterType type : {FilterType::FIR, FilterType::IIR}) {
            BasicFilter<T> filter;
            filter.filter_type     = type;
            filter.filter_response = gr::filter::Type::LOWPASS;
            filter.sample_rate     = 1000.f;
            filter.f_low           = 100.f;
            filter.gain            = 2.f;
            filter.designFilter();
            T settled{};
            for (std::size_t i = 0UZ; i < 2000UZ; ++i) {
                settled = filter.filterOne(T{1});
            }
            expect(approx(settled, 2.f, 1e-3f)) << magic_enum::enum_name(type);
        }
    };

    "a complex sample through real manual taps keeps both quadratures"_test = [] {
        using C = std::complex<float>;
        BasicFilter<C> filter;
        filter.filter_type        = FilterType::FIR;
        filter.coefficient_source = CoefficientSource::Manual;
        filter.b                  = gr::Tensor<C>(std::vector<C>{C{0.25f}, C{0.5f}, C{0.25f}});
        filter.a                  = gr::Tensor<C>(std::vector<C>{C{1.f}});
        filter.designFilter();
        const C              impulse{1.f, 2.f};
        const std::vector<C> response{filter.filterOne(impulse), filter.filterOne(C{}), filter.filterOne(C{})};
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            expect(le(std::abs(response[i] - taps[i] * impulse), 1e-6f)) << i;
        }
    };

    "an IIR is refused the transform rather than silently given the tap form"_test = [] {
        BasicFilter<T> filter;
        filter.filter_type   = FilterType::IIR;
        filter.filter_domain = ConvolutionDomain::Frequency;
        filter.designFilter();

        expect(eq(filter.input_chunk_size, gr::Size_t(1))) << "the frame geometry must not be adopted";
    };

    const auto drainMessages = [](gr::MsgPortIn& port) {
        std::vector<gr::Message> messages;
        if (const std::size_t available = port.streamReader().available(); available != 0UZ) {
            auto span = port.streamReader().get(available);
            messages.assign(span.begin(), span.end());
            std::ignore = span.consume(available);
        }
        return messages;
    };

    const auto sawErrorMentioning = [](const std::vector<gr::Message>& messages, std::string_view fragment) { return std::ranges::any_of(messages, [fragment](const gr::Message& message) { return !message.data.has_value() && message.data.error().message.find(fragment) != std::string::npos; }); };

    "designed taps take the transform route and adopt its frame geometry"_test = [] {
        BasicFilter<T> filter;
        filter.filter_type        = FilterType::FIR;
        filter.filter_domain      = ConvolutionDomain::Frequency;
        filter.coefficient_source = CoefficientSource::Designed;
        filter.sample_rate        = 1000.f;
        filter.f_low              = 100.f;
        filter.designFilter();

        expect(gt(filter.input_chunk_size.value, gr::Size_t(1))) << "the overlap-save frame is the input chunk";
        expect(gt(filter.output_chunk_size.value, gr::Size_t(0)));
        expect(eq(filter.stride.value, filter.output_chunk_size.value)) << "consecutive frames advance by the outputs they produce";
    };

    "the transform is refused on a device compute domain"_test = [&] {
        BasicFilter<T> filter;
        gr::MsgPortIn  fromBlock;
        expect(filter.msgOut.connect(fromBlock).has_value());

        filter.filter_type    = FilterType::FIR;
        filter.filter_domain  = ConvolutionDomain::Frequency;
        filter.compute_domain = std::string("gpu:sycl");
        filter.designFilter();

        expect(eq(filter.input_chunk_size, gr::Size_t(1))) << "the frame geometry must not be adopted";
        expect(sawErrorMentioning(drainMessages(fromBlock), "gpu:sycl")) << "the refusal names the compute domain";
    };

    "the transform is refused for a sample type that cannot ride it"_test = [&] {
        BasicFilter<gr::UncertainValue<float>> filter;
        gr::MsgPortIn                          fromBlock;
        expect(filter.msgOut.connect(fromBlock).has_value());

        filter.filter_type   = FilterType::FIR;
        filter.filter_domain = ConvolutionDomain::Frequency;
        filter.designFilter();

        expect(sawErrorMentioning(drainMessages(fromBlock), "plain floating-point sample type")) << "the refusal says why";
    };

    "a designed cascade runs its rows, and every served device returns what the host returns"_test = [] {
        std::ignore = gr::device::registerSyclRuntime();

        const auto designedIir = [](auto& dut) {
            dut.filter_type       = FilterType::IIR;
            dut.filter_response   = filter::Type::LOWPASS;
            dut.filter_order      = 4U;
            dut.f_low             = 0.1f;
            dut.sample_rate       = 1.0f;
            dut.iir_design_method = filter::iir::Design::BUTTERWORTH;
        };
        const std::vector<float> host = basic_filter_test::runBasic("host", 4096U, designedIir);
        expect(gt(host.size(), 1000UZ)) << "the host arm must have produced output";

        for (std::string_view domain : basic_filter_test::servedDomains()) {
            if (domain == "host") {
                continue;
            }
            const std::vector<float> onDevice = basic_filter_test::runBasic(domain, 4096U, designedIir);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of samples", domain);
            bool matches = onDevice.size() == host.size();
            for (std::size_t i = 0UZ; matches && i < host.size(); ++i) {
                // relative: a recursion is sensitive to whether a multiply-add is contracted, and the kernel is free to
                // fuse where the host build does not. A lost state or a wrong coefficient differs by order one.
                matches = std::abs(onDevice[i] - host[i]) <= 1e-4f * std::max(1.0f, std::abs(host[i]));
            }
            expect(matches) << std::format("'{}' must return what the same cascade returns on the host", domain);
        }
    };

    "a redesign does not leave the previous state behind"_test = [] {
        BasicFilter<float> filter;
        filter.filter_type        = FilterType::FIR;
        filter.coefficient_source = CoefficientSource::Manual;
        filter.b                  = gr::Tensor<float>(taps);
        filter.designFilter();

        for (std::size_t i = 0UZ; i < 64UZ; ++i) { // drive the state away from zero
            std::ignore = filter.filterOne(1.0f);
        }
        const gr::Size_t epochBefore = filter._design_epoch;
        filter.b                     = gr::Tensor<float>(std::vector<float>{1.0f});
        filter.designFilter();
        expect(gt(filter._design_epoch, epochBefore)) << "a redesign must bump the epoch the kernel compares against";

        // the state belongs to the coefficients that are gone, so the first sample of a pass-through must be itself
        const std::vector<float> fresh = basic_filter_test::runBasic("host", 64U, [](auto& dut) {
            dut.filter_type        = FilterType::FIR;
            dut.coefficient_source = CoefficientSource::Manual;
            dut.b                  = gr::Tensor<float>(std::vector<float>{1.0f});
            dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only: state the denominator too
        });
        expect(gt(fresh.size(), 8UZ));
        expect(approx(fresh[4], 4.0f, 1e-3f)) << "a unit tap must reproduce the ramp, so no stale state leaked in";
    };

    // the accumulators follow the coefficients, so there is no length left to refuse -- a section of any order is
    // carried rather than replaced by a pass-through, which is what the fixed state array used to force
    "a cascade of any length is carried, not refused"_test = [] {
        constexpr std::size_t kLong = 600UZ; // longer than the state array this block used to carry
        BasicFilter<float>    filter;
        filter.filter_type        = FilterType::FIR;
        filter.coefficient_source = CoefficientSource::Manual;
        filter.a                  = gr::Tensor<float>(std::vector<float>{1.0f});
        filter.b                  = gr::Tensor<float>(std::vector<float>(kLong, 1.0f / static_cast<float>(kLong)));
        filter.designFilter();

        expect(eq(filter.coefficientsPerSection(), kLong)) << "the section must be kept at its own length";
        for (std::size_t i = 0UZ; i < kLong; ++i) { // fill the history with ones, so the moving average settles at one
            std::ignore = filter.filterOne(1.0f);
        }
        expect(approx(filter.filterOne(1.0f), 1.0f, 1e-3f)) << "a unit-sum moving average of a constant is that constant";
    };

    "a decimating filter has no stride to hand the transform"_test = [] {
        expect(not BasicDecimatingFilter<T>::kCanTransform);
        expect(BasicFilter<T>::kCanTransform);
        expect(not BasicFilter<gr::UncertainValue<float>>::kCanTransform) << "an uncertainty-propagating value cannot ride a single-sample transform";
    };
};

namespace fir_window_test {
using namespace gr::testing;

struct RunResult {
    std::vector<float> samples;
    gr::Size_t         inputChunk  = 0U;
    gr::Size_t         outputChunk = 0U;
};

[[nodiscard]] inline RunResult runFir(std::string_view domain, std::vector<float> taps, gr::Size_t nSamples) {
    gr::Graph flow({{"auto_size_edges_to_chunks", true}});
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     dut    = flow.emplaceBlock<gr::filter::fir_filter<float, gr::filter::ConvolutionDomain::Time>>({{"gr:compute_domain", std::string(domain)}, {"b", gr::Tensor<float>(taps)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
    boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());

    gr::scheduler::Simple<> sched;
    boost::ut::expect(sched.exchange(std::move(flow)).has_value());
    boost::ut::expect(sched.runAndWait().has_value());

    return RunResult{.samples = std::vector<float>(sink._samples.begin(), sink._samples.end()), .inputChunk = dut.input_chunk_size, .outputChunk = dut.output_chunk_size};
}
} // namespace fir_window_test

/// what each filter computes where it runs: the window form lets the FIR leave the host, the span form lets the
/// IIR stay resident in a cascade. Device cases skip honestly where no backend is served.
const boost::ut::suite<"one FIR, two domains"> FirDomainTests = [] {
    using namespace boost::ut;
    using namespace gr::filter;
    using namespace gr::testing;

    static const auto rampTaps = [](std::size_t nTaps) {
        std::vector<float> taps(nTaps);
        for (std::size_t i = 0UZ; i < nTaps; ++i) {
            taps[i] = static_cast<float>(i + 1UZ) / static_cast<float>(nTaps);
        }
        return taps;
    };

    /// both forms declare their own geometry, so each is driven through a graph and compared on what came out
    static const auto runForm = []<typename TBlock>(std::string_view domain, const std::vector<float>& taps, gr::Size_t outputsPerFrame, gr::Size_t nSamples) {
        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
        auto&     dut    = flow.emplaceBlock<TBlock>({{"gr:compute_domain", std::string(domain)}});
        auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

        dut.b                 = gr::Tensor<float>(taps);
        dut.outputs_per_frame = outputsPerFrame;
        dut.settingsChanged({}, {});

        expect(flow.connect<"out", "in">(source, dut).has_value());
        expect(flow.connect<"out", "in">(dut, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());
        return std::vector<float>(sink._samples.begin(), sink._samples.end());
    };

    "a transform per frame returns what a tap per sample returns"_test = [] {
        for (const std::size_t nTaps : {1UZ, 33UZ, 65UZ}) { // 33 is not 2^k + 1, so the discard arithmetic is exercised
            const std::vector<float> taps  = rampTaps(nTaps);
            const gr::Size_t         frame = nTaps == 1UZ ? 256U : (nTaps == 33UZ ? 100U : 256U);

            const std::vector<float> byTaps      = runForm.template operator()<fir_filter<float, ConvolutionDomain::Time>>("host", taps, frame, 8192U);
            const std::vector<float> byTransform = runForm.template operator()<fir_filter<float, ConvolutionDomain::Frequency>>("host", taps, frame, 8192U);

            expect(gt(byTransform.size(), 0UZ)) << std::format("{} taps: the transform arm produced nothing", nTaps);
            expect(le(byTransform.size(), byTaps.size())) << std::format("{} taps: the transform arm emits whole frames only", nTaps);

            const float scale    = std::abs(*std::ranges::max_element(byTaps, [](float x, float y) { return std::abs(x) < std::abs(y); }));
            bool        agree    = true;
            std::size_t firstBad = byTransform.size();
            for (std::size_t i = 0UZ; i < byTransform.size(); ++i) { // index for index: the two forms are aligned
                if (std::abs(byTaps[i] - byTransform[i]) > 1e-5f * std::max(1.0f, scale)) {
                    agree    = false;
                    firstBad = std::min(firstBad, i);
                }
            }
            expect(agree) << std::format("{} taps: the two domains disagree from sample {}", nTaps, firstBad);
        }
    };

    "the AUTO form picks a domain and still returns the same samples"_test = [] {
        const std::vector<float> shortTaps = rampTaps(8UZ);
        const std::vector<float> longTaps  = rampTaps(512UZ); // past kFrequencyDomainFromTaps

        for (const std::vector<float>* taps : {&shortTaps, &longTaps}) {
            const std::vector<float> automatic = runForm.template operator()<fir_filter<float>>("host", *taps, 256U, 8192U);
            const std::vector<float> byTaps    = runForm.template operator()<fir_filter<float, ConvolutionDomain::Time>>("host", *taps, 256U, 8192U);
            expect(gt(automatic.size(), 0UZ)) << std::format("{} taps: AUTO produced nothing", taps->size());

            const float scale = std::abs(*std::ranges::max_element(byTaps, [](float x, float y) { return std::abs(x) < std::abs(y); }));
            bool        agree = automatic.size() <= byTaps.size();
            for (std::size_t i = 0UZ; agree && i < automatic.size(); ++i) {
                agree = std::abs(byTaps[i] - automatic[i]) <= 1e-5f * std::max(1.0f, scale);
            }
            expect(agree) << std::format("{} taps: whichever domain AUTO chose must compute the same filter", taps->size());
        }
    };

    "the transform returns what the host returns on every served device"_test = [] {
        std::ignore = gr::device::registerSyclRuntime();

        const std::vector<float> taps = rampTaps(65UZ);
        const std::vector<float> host = runForm.template operator()<fir_filter<float, ConvolutionDomain::Frequency>>("host", taps, 256U, 8192U);
        expect(gt(host.size(), 0UZ)) << "the host arm must have produced output";

        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
                expect(!gr::testing::deviceDomainRequired(domain)) << "GR4_REQUIRE_DEVICE names this domain, so the lane must exercise it rather than skip";
                continue;
            }
            const std::vector<float> onDevice = runForm.template operator()<fir_filter<float, ConvolutionDomain::Frequency>>(domain, taps, 256U, 8192U);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of samples", domain);
            // a device transform is a different butterfly order, so the comparison is relative to full scale
            const float scale = std::abs(*std::ranges::max_element(host, [](float x, float y) { return std::abs(x) < std::abs(y); }));
            bool        agree = onDevice.size() == host.size();
            for (std::size_t i = 0UZ; agree && i < host.size(); ++i) {
                agree = std::abs(onDevice[i] - host[i]) <= 1e-4f * std::max(1.0f, scale);
            }
            expect(agree) << std::format("'{}' must return what the transform returns on the host", domain);
        }
    };

    "the lean forms carry only their own state"_test = [] {
        expect(lt(sizeof(fir_filter<float, ConvolutionDomain::Time>), sizeof(fir_filter<float, ConvolutionDomain::Auto>))) << "a time-domain-only instantiation must not carry the transform's state";
        expect(std::is_same_v<fir_filter<float>, fir_filter<float, ConvolutionDomain::Auto>>) << "the unqualified name must keep meaning AUTO";
    };
};

const boost::ut::suite<"filters on every served device"> DeviceFilterTests = [] {
    using namespace boost::ut;
    using namespace fir_window_test;

    "the filter is 1:1 over the stream, whatever its tap count"_test = [] {
        const RunResult run = runFir("host", {0.25f, 0.25f, 0.25f, 0.25f}, 256U);
        expect(eq(run.inputChunk, gr::Size_t(1))) << "the lead-in makes the first window full, so no chunk is declared";
        expect(eq(run.outputChunk, gr::Size_t(1)));
        expect(eq(run.samples.size(), 256UZ)) << "as many outputs as inputs";
    };

    // the ramp source gives x[n] = n, and the window preceding the stream is zero, so a 4-tap moving average
    // answers (x[n-3] + x[n-2] + x[n-1] + x[n]) / 4 with the missing terms zero: n - 1.5 once the window is full
    "a moving average over a ramp is the ramp delayed by the group delay"_test = [] {
        const RunResult run = runFir("host", {0.25f, 0.25f, 0.25f, 0.25f}, 256U);
        expect(eq(run.samples.size(), 256UZ));

        bool transientMatches = true;
        for (std::size_t n = 0UZ; n < 3UZ; ++n) {
            float expected = 0.f;
            for (std::size_t j = 0UZ; j <= n; ++j) {
                expected += 0.25f * static_cast<float>(n - j);
            }
            transientMatches = transientMatches && std::abs(run.samples[n] - expected) < 1e-3f;
        }
        expect(transientMatches) << "the start-up transient is emitted, not skipped";

        bool matches = true;
        for (std::size_t n = 3UZ; n < run.samples.size(); ++n) {
            matches = matches && std::abs(run.samples[n] - (static_cast<float>(n) - 1.5f)) < 1e-3f;
        }
        expect(matches) << "every full window must be the average of the four inputs it covered";
    };

    "a single tap is a gain, and stays 1:1"_test = [] {
        const RunResult run = runFir("host", {2.f}, 64U);
        expect(eq(run.samples.size(), 64UZ)) << "one tap consumes and produces one sample";
        expect(std::abs(run.samples[10] - 20.f) < 1e-3f) << "x[10] = 10, doubled";
    };

    // an IIR is one work item over the whole span: it cannot be faster on a device, and it runs there so that a
    // cascade around it need not leave the device. Matching the host sample for sample also proves the recursion's
    // state survived the dispatch boundary, since a state reset per dispatch would show up as a discontinuity.
    "an IIR keeps its recursion across dispatches on every served device"_test = [] {
        const bool available = gr::device::registerSyclRuntime();
        expect(!available || gr::device::hostSyclIsServed()) << "a SYCL build must serve 'host:sycl'";

        const auto runIir = [](std::string_view domain) {
            gr::Graph flow({{"auto_size_edges_to_chunks", true}});
            auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", gr::Size_t(4096)}, {"mark_tag", false}});
            auto&     dut    = flow.emplaceBlock<gr::filter::iir_filter<float, gr::filter::IIRForm::DF_II>>({{"gr:compute_domain", std::string(domain)}});
            auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});
            dut.b            = gr::Tensor<float>{0.55f, 0.f};
            dut.a            = gr::Tensor<float>{1.f, -0.45f};
            boost::ut::expect(flow.connect<"out", "in">(source, dut).has_value());
            boost::ut::expect(flow.connect<"out", "in">(dut, sink).has_value());
            gr::scheduler::Simple<> sched;
            boost::ut::expect(sched.exchange(std::move(flow)).has_value());
            boost::ut::expect(sched.runAndWait().has_value());
            return std::vector<float>(sink._samples.begin(), sink._samples.end());
        };

        const std::vector<float> host = runIir("host");
        expect(gt(host.size(), 1000UZ)) << "the host arm must actually have produced output";
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
                boost::ut::expect(!gr::testing::deviceDomainRequired(domain)) << "GR4_REQUIRE_DEVICE names this domain, so the lane must exercise it rather than skip";
                continue;
            }
            const std::vector<float> onDevice = runIir(domain);
            expect(eq(onDevice.size(), host.size())) << std::format("'{}' produced a different number of samples", domain);
            bool matches = onDevice.size() == host.size();
            for (std::size_t i = 0UZ; matches && i < host.size(); ++i) {
                matches = std::abs(onDevice[i] - host[i]) < 1e-3f;
            }
            expect(matches) << std::format("'{}' must return what the same block returns on the host", domain);
        }
    };

    "every served device returns what the host returns"_test = [] {
        const bool available = gr::device::registerSyclRuntime();
        expect(!available || gr::device::hostSyclIsServed()) //
            << "a build with a SYCL backend must serve 'host:sycl'; without it this case skips and asserts nothing";

        const std::vector<float> taps{0.1f, 0.2f, 0.3f, 0.4f};
        const RunResult          host = runFir("host", taps, 1024U);
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
                boost::ut::expect(!gr::testing::deviceDomainRequired(domain)) << "GR4_REQUIRE_DEVICE names this domain, so the lane must exercise it rather than skip";
                continue;
            }
            const RunResult onDevice = runFir(domain, taps, 1024U);
            expect(eq(onDevice.samples.size(), host.samples.size())) << std::format("'{}' produced a different number of samples", domain);
            bool matches = onDevice.samples.size() == host.samples.size();
            for (std::size_t i = 0UZ; matches && i < host.samples.size(); ++i) {
                matches = std::abs(onDevice.samples[i] - host.samples[i]) < 1e-3f;
            }
            expect(matches) << std::format("'{}' must return what the same block returns on the host", domain);
        }
    };
};

const boost::ut::suite<"ComplexFilterTests"> complexFilterTests = [] {
    using namespace boost::ut;
    "complex direct FIR and IIR keep both quadratures and complex taps"_test = [] {
        using C = std::complex<double>;
        std::array<C, 3UZ>       input{C{1.0, 2.0}, C{3.0, 4.0}, C{5.0, 6.0}};
        const std::array<C, 2UZ> taps{C{1.0, 0.0}, C{0.0, 1.0}};
        std::array<C, 1UZ>       history{};
        std::array<C, 3UZ>       output{};
        gr::algorithm::filter::Fir<C>::convolveStreaming(input, taps, history, output);
        expect(eq(output[0], C{1.0, 2.0}));
        expect(eq(output[1], C{1.0, 5.0}));
        expect(eq(output[2], C{1.0, 9.0}));
        const std::array<C, 1UZ> numerator{C{1.0, 0.0}};
        const std::array<C, 2UZ> denominator{C{1.0, 0.0}, C{-0.5, 0.0}};
        std::array<C, 1UZ>       state{};
        gr::algorithm::filter::Iir<C>::filter(input, numerator, denominator, state, output);
        expect(eq(output[0], C{1.0, 2.0}));
        expect(eq(output[1], C{3.5, 5.0}));
        expect(eq(output[2], C{6.75, 8.5}));
    };
    "complex overlap save retains the imaginary part of its convolution"_test = [] {
        using C = std::complex<double>;
        gr::algorithm::filter::FastConvolution<C> convolution;
        const std::array<C, 2UZ>                  taps{C{1.0, 0.0}, C{0.0, 1.0}};
        convolution.prepareTaps(std::span<const C>(taps), 8UZ);
        const std::array<C, 8UZ> frame{C{}, C{1.0, 2.0}, C{3.0, 4.0}, C{5.0, 6.0}, C{}, C{}, C{}, C{}};
        std::array<C, 7UZ>       output{};
        convolution.convolveFrame(frame, output);
        const std::array<C, 4UZ> expected{C{1.0, 2.0}, C{1.0, 5.0}, C{1.0, 9.0}, C{-6.0, 5.0}};
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(le(std::abs(output[i] - expected[i]), 1.e-12));
        }
    };
    "a complex fir_filter block applies complex taps to both quadratures"_test = [] {
        using C = std::complex<double>;
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::TagSource<C>>({{"n_samples_max", gr::Size_t{3U}}, {"values", gr::Tensor<C>(gr::data_from, {C{1.0, 2.0}, C{3.0, 4.0}, C{5.0, 6.0}})}});
        auto&                     filter = fixture.emplace<gr::filter::fir_filter<C>>({{"b", gr::Tensor<C>(gr::data_from, {C{1.0, 0.0}, C{0.0, 1.0}})}});
        auto&                     sink   = fixture.emplace<gr::testing::TagSink<C, gr::testing::ProcessFunction::USE_PROCESS_BULK>>();
        expect(fixture.connect<"out", "in">(source, filter).has_value());
        expect(fixture.connect<"out", "in">(filter, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._samples.size(), 3UZ)));
        const std::array<C, 3UZ> expected{C{1.0, 2.0}, C{1.0, 5.0}, C{1.0, 9.0}}; // y[n] = x[n] + i x[n-1]
        for (std::size_t i = 0UZ; i < expected.size(); ++i) {
            expect(le(std::abs(sink._samples[i] - expected[i]), 1.e-12)) << i;
        }
    };
};

const boost::ut::suite<"RootRaisedCosineTests"> rootRaisedCosineTests = [] {
    using namespace boost::ut;
    "RRC taps follow the inverse transform of the root-raised-cosine spectrum"_test = [] {
        constexpr double      kAlpha   = 0.5;
        constexpr std::size_t kTaps    = 33UZ;
        constexpr double      kSamples = 8.0; // per symbol
        // h(t) = 2 * integral over 0..(1+alpha)/2 of H(f) cos(2 pi f t) df, symbol period 1; Simpson's rule per band
        const auto spectrum = [](double f) {
            const double passEdge = (1.0 - kAlpha) / 2.0;
            return f <= passEdge ? 1.0 : std::cos(std::numbers::pi / (2.0 * kAlpha) * (f - passEdge));
        };
        const auto pulseAt = [&](double t) {
            const auto simpson = [&](double from, double to) {
                constexpr std::size_t kIntervals = 4096UZ;
                const double          h          = (to - from) / static_cast<double>(kIntervals);
                double                sum        = 0.0;
                for (std::size_t k = 0UZ; k <= kIntervals; ++k) {
                    const double f      = from + h * static_cast<double>(k);
                    const double weight = k == 0UZ || k == kIntervals ? 1.0 : (k % 2UZ == 1UZ ? 4.0 : 2.0);
                    sum += weight * spectrum(f) * std::cos(2.0 * std::numbers::pi * f * t);
                }
                return sum * h / 3.0;
            };
            return 2.0 * (simpson(0.0, (1.0 - kAlpha) / 2.0) + simpson((1.0 - kAlpha) / 2.0, (1.0 + kAlpha) / 2.0));
        };
        const auto taps = gr::filter::fir::designRootRaisedCosine<double>(1.0, kSamples, 1.0, kAlpha, kTaps);
        expect(fatal(taps.has_value()));
        const double centreTap = taps->b[kTaps / 2UZ];
        const double centre    = pulseAt(0.0);
        for (std::size_t i = 0UZ; i < kTaps; ++i) {
            const double t = (static_cast<double>(i) - static_cast<double>(kTaps / 2UZ)) / kSamples;
            expect(approx(taps->b[i] / centreTap, pulseAt(t) / centre, 1.e-9)) << "tap" << i << "at t =" << t << "symbols, including the limits at t = 0 and 1/(4 alpha)";
        }
        expect(approx(std::accumulate(taps->b.begin(), taps->b.end(), 0.0), 1.0, 1.e-14));
    };
    "RRC with zero roll-off is a sinc, and gain scales the DC sum"_test = [] {
        const auto taps = gr::filter::fir::designRootRaisedCosine<double>(2.5, 4.0, 1.0, 0.0, 17UZ);
        expect(fatal(taps.has_value()));
        for (std::size_t i = 0UZ; i < 17UZ; ++i) {
            const double t    = (static_cast<double>(i) - 8.0) / 4.0;
            const double sinc = t == 0.0 ? 1.0 : std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
            expect(approx(taps->b[i] / taps->b[8UZ], sinc, 1.e-14));
        }
        expect(approx(std::accumulate(taps->b.begin(), taps->b.end(), 0.0), 2.5, 1.e-13));
    };
    "a matched pair of long RRC pulses has negligible symbol spaced interference"_test = [] {
        const auto pulse = gr::filter::fir::designRootRaisedCosine<double>(1.0, 8.0, 1.0, 0.5, 129UZ);
        expect(fatal(pulse.has_value()));
        double energy = 0.0, adjacent = 0.0;
        for (std::size_t i = 0UZ; i < pulse->b.size(); ++i) {
            energy += pulse->b[i] * pulse->b[i];
            if (i >= 8UZ) {
                adjacent += pulse->b[i] * pulse->b[i - 8UZ];
            }
        }
        expect(lt(std::abs(adjacent / energy), 1.e-3));
    };
    "RRC refuses a non-positive rate, a roll-off above one and an even tap count"_test = [] {
        expect(!gr::filter::fir::designRootRaisedCosine<double>(1.0, 0.0, 1.0, 0.5, 33UZ).has_value());
        expect(!gr::filter::fir::designRootRaisedCosine<double>(1.0, 8.0, 1.0, 1.1, 33UZ).has_value());
        expect(!gr::filter::fir::designRootRaisedCosine<double>(1.0, 8.0, 1.0, 0.5, 32UZ).has_value());
    };
    "a rational pulse shaper preserves supplied RRC taps and applies its gain once"_test = [] {
        const auto                            pulse = gr::filter::fir::designRootRaisedCosine<double>(1.0, 8.0, 1.0, 0.5, 33UZ).value_or(gr::filter::FilterCoefficients<double>{});
        gr::filter::RationalResampler<double> shaper;
        shaper.interpolation = 8U;
        shaper.decimation    = 1U;
        shaper.gain          = 2.0;
        shaper.taps          = gr::Tensor<double>(pulse.b);
        shaper.settingsChanged({}, {});
        expect(eq(shaper.phases.size(), 40UZ));
        for (std::size_t i = 0UZ; i < pulse.b.size(); ++i) {
            const std::size_t phaseIndex = (i % 8UZ) * 5UZ + i / 8UZ;
            expect(approx(shaper.phases[phaseIndex], 2.0 * pulse.b[i], 1.e-14));
        }
    };
    "custom taps split into phases as given and scaled by gain in every polyphase block"_test = [] {
        const std::vector<double> prototype{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
        const std::vector<double> kSplit{1.0, 4.0, 7.0, 2.0, 5.0, 0.0, 3.0, 6.0, 0.0}; // phase p holds taps p, p + 3, p + 6
        expect(eq(gr::algorithm::filter::PolyphaseBank<double>::splitPhases(prototype, 3UZ), kSplit));
        const auto expectScaledSplit = [&](auto& block, std::string_view name) {
            block.taps = gr::Tensor<double>(prototype);
            block.gain = 2.0;
            if constexpr (requires(gr::property_map forwarded) { block.settingsChanged({}, {}, forwarded); }) {
                gr::property_map forwarded;
                block.settingsChanged({}, {}, forwarded);
            } else {
                block.settingsChanged({}, {});
            }
            expect(fatal(eq(block.phases.size(), kSplit.size()))) << name;
            for (std::size_t i = 0UZ; i < kSplit.size(); ++i) {
                expect(eq(block.phases[i], 2.0 * kSplit[i])) << name << i;
            }
        };
        gr::filter::PolyphaseChannelizer<std::complex<double>> channelizer;
        channelizer.n_channels = 3U;
        expectScaledSplit(channelizer, "channelizer");
        gr::filter::PolyphaseSynthesizer<std::complex<double>> synthesizer;
        synthesizer.n_channels = 3U;
        expectScaledSplit(synthesizer, "synthesizer");
        gr::filter::PolyphaseArbitraryResampler<double> arbitrary;
        arbitrary.n_phases = 3U;
        expectScaledSplit(arbitrary, "arbitrary resampler");
    };
};

// the cases run here rather than from the UT runner's destructor: a test that reaches a device
// initialises the SYCL runtime, and at exit that runtime's own statics are already gone
int main() { return boost::ut::cfg<boost::ut::override>.run({.report_errors = true}); }
