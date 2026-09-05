#include <boost/ut.hpp>
#include <span>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/testing/DeviceExpectation.hpp>
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
        fir_filter<double, IRForm::TIME_DOMAIN> fir_filter;
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

        // the FIR answers over a window now, so it is driven in bulk and its response is padded back to the
        // step's own length: a window cannot answer until it is full, which costs the first b.size() - 1 samples
        const std::size_t       nTaps = fir_filter.b.size();
        std::vector<double>     windowed(step.size() - (nTaps - 1UZ));
        std::span<const double> stepSpan{step};
        std::span<double>       windowedSpan{windowed};
        std::ignore = fir_filter.processBulk(stepSpan, windowedSpan);
        std::vector<double> fir_response(nTaps - 1UZ, 0.0);
        fir_response.insert(fir_response.end(), windowed.begin(), windowed.end());

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

    constexpr static float       sampleRate     = 1000.0;
    constexpr static float       f_low          = 100.0;
    constexpr static std::size_t filterOrder    = 4;
    constexpr static std::size_t numSamples     = 1000;
    constexpr static std::size_t decimationRate = 5;

    "BasicFilter - Low-pass Filter Test"_test =
        []<typename TTestParameter>() {
            using T         = typename TTestParameter::value_type;
            using ValueType = meta::fundamental_base_value_type_t<T>;
            auto filterType = TTestParameter::filter_type;

            BasicFilter<T> filter;
            filter.filter_type       = filterType;
            filter.filter_response   = filter::Type::LOWPASS;
            filter.filter_order      = filterOrder;
            filter.f_low             = f_low;
            filter.sample_rate       = sampleRate;
            filter.iir_design_method = filter::iir::Design::CHEBYSHEV1;
            filter.fir_design_method = algorithm::window::Type::Hamming;
            filter.designFilter(); // triggers filter re-computation and setting of internal enums

            "verify in-band signal passes through"_test = [&filter] {
                std::vector<T> outputSignal;
                outputSignal.reserve(numSamples);
                T phase = 0;
                for (std::size_t i = 0UZ; i < 2 * numSamples; i++) {
                    // generate a sine wave signal with a frequency below the cutoff
                    phase += T{2} * std::numbers::pi_v<ValueType> * static_cast<ValueType>(50) / static_cast<ValueType>(sampleRate);
                    if (i < numSamples) { // ignore initial transient
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
                outputSignal.reserve(numSamples);
                T phase = 0;
                for (std::size_t i = 0UZ; i < 2 * numSamples; i++) {
                    // generate a sine wave signal with a frequency below the cutoff
                    phase += T{2} * std::numbers::pi_v<ValueType> * static_cast<ValueType>(300) / static_cast<ValueType>(sampleRate);
                    if (i < numSamples) { // ignore initial transient
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
        filter.filter_order      = filterOrder;
        filter.f_low             = f_low;
        filter.sample_rate       = sampleRate;
        filter.iir_design_method = filter::iir::Design::CHEBYSHEV1;
        filter.fir_design_method = algorithm::window::Type::Hamming;
        filter.decimate          = decimationRate;
        filter.designFilter(); // triggers filter re-computation and setting of internal enums

        expect(eq(filter.input_chunk_size, decimationRate)) << "decimationRate type mismatch";

        "verify in-band signal passes through"_test = [&filter] {
            std::vector<T> inputSignal(numSamples);
            std::vector<T> outputSignal(numSamples / decimationRate);

            T    phase          = 0;
            auto generateSample = [&phase]() {
                // generate a sine wave signal with a frequency below the cutoff
                phase += 2 * std::numbers::pi_v<T> * static_cast<T>(50) / static_cast<T>(sampleRate);
                return std::sin(phase);
            };
            const auto filterDecimated = [&filter](std::span<const T> samples, std::span<T> decimated) {
                std::size_t outIndex = 0UZ;
                for (std::size_t i = 0UZ; i < samples.size() && outIndex < decimated.size(); ++i) {
                    const T filtered = filter.filterOne(samples[i]);
                    if (i % decimationRate == 0UZ) {
                        decimated[outIndex++] = filtered;
                    }
                }
            };
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);
            std::ranges::generate(inputSignal, generateSample);
            filterDecimated(inputSignal, outputSignal);

            double maxOutput = std::abs(*std::ranges::max_element(outputSignal, maxOp));
            expect(ge(maxOutput, T{0.9})) << std::format("{} filter should pass in-band frequencies: max output {}", filter.filter_type, maxOutput);
        };

        "verify out-of-band signal is attenuated"_test = [&filter] {
            std::vector<T> inputSignal(numSamples);
            std::vector<T> outputSignal(numSamples / decimationRate);

            T    phase          = 0;
            auto generateSample = [&phase]() {
                // generate a sine wave signal with a frequency above the cutoff
                phase += 2 * std::numbers::pi_v<T> * T(300) / T(sampleRate);
                return std::sin(phase);
            };
            const auto filterDecimated = [&filter](std::span<const T> samples, std::span<T> decimated) {
                std::size_t outIndex = 0UZ;
                for (std::size_t i = 0UZ; i < samples.size() && outIndex < decimated.size(); ++i) {
                    const T filtered = filter.filterOne(samples[i]);
                    if (i % decimationRate == 0UZ) {
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
        auto&     decimator = flow.emplaceBlock<gr::filter::Decimator<T>>({{"decim", kDecimationFactor}});
        auto&     sink      = flow.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", kInputSamples / kDecimationFactor}});
        expect(flow.connect<"out", "in">(source, decimator).has_value());
        expect(flow.connect<"out", "in">(decimator, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(decimator.decim, kDecimationFactor));
        expect(eq(decimator.output_chunk_size, static_cast<gr::Size_t>(1)));
        expect(eq(decimator.input_chunk_size, kDecimationFactor));
        expect(eq(sink._nSamplesProduced, kInputSamples / kDecimationFactor));
        expect(eq(sink.sample_rate, kInputRate / static_cast<float>(kDecimationFactor))) << "rate seen downstream";
    };

    "BasicDecimatingFilter - keeps its input sample_rate"_test = [] {
        // the filter's own 'sample_rate' is its input rate; only the downstream tag carries the decimated rate.
        using namespace gr::testing;
        using T = float;

        constexpr float      kInputRate = 32'000.f; // decimated rate must differ from the TagSink default, else the sink assert passes without a tag
        constexpr gr::Size_t kDecimate  = 10U;
        constexpr gr::Size_t kNSamples  = 4'000U;

        gr::Graph flow;
        auto&     source = flow.emplaceBlock<TagSource<T>>({{"sample_rate", kInputRate}, {"n_samples_max", kNSamples}});
        auto&     filter = flow.emplaceBlock<BasicDecimatingFilter<T>>({{"sample_rate", kInputRate}, {"f_low", 400.f}, {"decimate", kDecimate}});
        auto&     sink   = flow.emplaceBlock<TagSink<T, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", kNSamples / kDecimate}});

        expect(flow.connect<"out", "in">(source, filter).has_value());
        expect(flow.connect<"out", "in">(filter, sink).has_value());

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialise scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << "scheduler run";

        expect(eq(filter.sample_rate, kInputRate)) << "filter member holds its input rate";
        expect(eq(gr::test::get_value_or_fail<float>(*filter.settings().get("sample_rate")), kInputRate)) << "filter settings().get() must agree with its member";
        expect(eq(sink.sample_rate, kInputRate / static_cast<float>(kDecimate))) << "rate seen downstream";
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
        if (gr::device::DeviceContextRegistry::instance().tryResolve(candidate) != nullptr) {
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
            dut.filter_domain      = FilterDomain::Time;
            dut.coefficient_source = CoefficientSource::Manual;
            dut.b                  = gr::Tensor<float>(taps);
            dut.a                  = gr::Tensor<float>(std::vector<float>{1.0f}); // feed-forward only: state the denominator too
        });
        const std::vector<float> byTransform = basic_filter_test::runBasic("host", 4096U, [](auto& dut) {
            dut.filter_type        = FilterType::FIR;
            dut.filter_domain      = FilterDomain::Frequency;
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

    "an IIR is refused the transform rather than silently given the tap form"_test = [] {
        BasicFilter<T> filter;
        filter.filter_type   = FilterType::IIR;
        filter.filter_domain = FilterDomain::Frequency;
        filter.designFilter();

        expect(eq(filter.input_chunk_size, gr::Size_t(1))) << "the frame geometry must not be adopted";
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

    "a cascade that does not fit is refused and passes the signal through"_test = [] {
        BasicFilter<float> filter;
        filter.filter_type        = FilterType::IIR;
        filter.coefficient_source = CoefficientSource::Manual;
        filter.a                  = gr::Tensor<float>(std::vector<float>(BasicFilter<float>::kMaxStates + 8UZ, 0.01f));
        filter.b                  = gr::Tensor<float>(std::vector<float>(BasicFilter<float>::kMaxStates + 8UZ, 0.01f));
        filter.designFilter();

        expect(eq(filter.coefficientsPerSection(), 1UZ)) << "an oversize set must be replaced, not kept and wrapped";
        expect(approx(filter.filterOne(2.0f), 2.0f, 1e-6f)) << "and the replacement must be a pass-through";
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
    auto&     dut    = flow.emplaceBlock<gr::filter::fir_filter<float, gr::filter::IRForm::TIME_DOMAIN>>({{"gr:compute_domain", std::string(domain)}, {"b", gr::Tensor<float>(taps)}});
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

            const std::vector<float> byTaps      = runForm.template operator()<fir_filter<float, IRForm::TIME_DOMAIN>>("host", taps, frame, 8192U);
            const std::vector<float> byTransform = runForm.template operator()<fir_filter<float, IRForm::FREQUENCY_DOMAIN>>("host", taps, frame, 8192U);

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
            const std::vector<float> byTaps    = runForm.template operator()<fir_filter<float, IRForm::TIME_DOMAIN>>("host", *taps, 256U, 8192U);
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
        const std::vector<float> host = runForm.template operator()<fir_filter<float, IRForm::FREQUENCY_DOMAIN>>("host", taps, 256U, 8192U);
        expect(gt(host.size(), 0UZ)) << "the host arm must have produced output";

        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                expect(!gr::testing::deviceDomainRequired(domain)) << "GR4_REQUIRE_DEVICE names this domain, so the lane must exercise it rather than skip";
                continue;
            }
            const std::vector<float> onDevice = runForm.template operator()<fir_filter<float, IRForm::FREQUENCY_DOMAIN>>(domain, taps, 256U, 8192U);
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
        expect(lt(sizeof(fir_filter<float, IRForm::TIME_DOMAIN>), sizeof(fir_filter<float, IRForm::AUTO>))) << "a time-domain-only instantiation must not carry the transform's state";
        expect(std::is_same_v<fir_filter<float>, fir_filter<float, IRForm::AUTO>>) << "the unqualified name must keep meaning AUTO";
    };
};

const boost::ut::suite<"filters on every served device"> DeviceFilterTests = [] {
    using namespace boost::ut;
    using namespace fir_window_test;

    "the declared window is one tap-length in, one sample out"_test = [] {
        const RunResult run = runFir("host", {0.25f, 0.25f, 0.25f, 0.25f}, 256U);
        expect(eq(run.inputChunk, gr::Size_t(4))) << "the window must span the taps";
        expect(eq(run.outputChunk, gr::Size_t(1))) << "and yield one sample per hop";
    };

    // the ramp source gives x[n] = n, so a 4-tap moving average answers (n-3 + n-2 + n-1 + n)/4 = n - 1.5
    "a moving average over a ramp is the ramp delayed by the group delay"_test = [] {
        const RunResult run = runFir("host", {0.25f, 0.25f, 0.25f, 0.25f}, 256U);
        expect(gt(run.samples.size(), 200UZ));

        bool matches = true;
        for (std::size_t n = 0UZ; n < run.samples.size(); ++n) {
            const float expected = static_cast<float>(n + 3UZ) - 1.5f; // the first window ends at input 3
            matches              = matches && std::abs(run.samples[n] - expected) < 1e-3f;
        }
        expect(matches) << "every output must be the average of the four inputs its window covered";
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
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
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
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
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

// the cases run here rather than from the UT runner's destructor: a test that reaches a device
// initialises the SYCL runtime, and at exit that runtime's own statics are already gone
int main() { return boost::ut::cfg<boost::ut::override>.run({.report_errors = true}); }
