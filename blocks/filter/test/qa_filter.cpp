#include <boost/ut.hpp>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

#include <gnuradio-4.0/filter/time_domain_filter.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

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
        std::vector<double> fir_coeffs(10, 0.1); // box car filter
        std::vector<double> iir_coeffs_b{0.55, 0};
        std::vector<double> iir_coeffs_a{1, -0.45};

        // Create FIR and IIR filter instances
        fir_filter<double> fir_filter;
        fir_filter.b = fir_coeffs;

        iir_filter<double, IIRForm::DF_I> iir_filter1;
        iir_filter1.b = iir_coeffs_b;
        iir_filter1.a = iir_coeffs_a;
        iir_filter<double, IIRForm::DF_II> iir_filter2;
        iir_filter2.b = iir_coeffs_b;
        iir_filter2.a = iir_coeffs_a;

        std::vector<double> fir_response;
        std::vector<double> iir_response1;
        std::vector<double> iir_response2;
        for (std::size_t i = 0UL; i < 20; ++i) {
            const double input = (i == 0) ? 0.0 : 1.0; // Step function

            fir_response.push_back(fir_filter.processOne(input));
            iir_response1.push_back(iir_filter1.processOne(input));
            iir_response2.push_back(iir_filter1.processOne(input));
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
        std::vector<double> iir_coeffs_b{0.020083365564211, 0.040166731128423, 0.020083365564211};
        std::vector<double> iir_coeffs_a{1.0, -1.561018075800718, 0.641351538057563};

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
            const auto   form_I    = iir_filter_I.processOne(input);
            const auto   form_II   = iir_filter_II.processOne(input);
            const auto   form_I_T  = iir_filter_IT.processOne(input);
            const auto   form_II_T = iir_filter_IIT.processOne(input);
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
                        std::ignore = filter.processOne(gr::math::sin(phase));
                    } else {
                        outputSignal.push_back(filter.processOne(gr::math::sin(phase)));
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
                        std::ignore = filter.processOne(gr::math::sin(phase));
                    } else {
                        outputSignal.push_back(filter.processOne(gr::math::sin(phase)));
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
            std::ranges::generate(inputSignal, generateSample);
            expect(filter.processBulk(inputSignal, outputSignal) == work::Status::OK) << "first processing failed";
            std::ranges::generate(inputSignal, generateSample);
            expect(filter.processBulk(inputSignal, outputSignal) == work::Status::OK) << "second processing failed";

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
            std::ranges::generate(inputSignal, generateSample);
            expect(filter.processBulk(inputSignal, outputSignal) == work::Status::OK) << "first processing failed";
            std::ranges::generate(inputSignal, generateSample);
            expect(filter.processBulk(inputSignal, outputSignal) == work::Status::OK) << "second processing failed";

            double maxOutput = std::abs(*std::ranges::max_element(outputSignal, maxOp));
            expect(le(maxOutput, T{0.2})) << std::format("{} filter should attenuate out-of-band frequencies: max output {}", filter.filter_type, maxOutput);
        };
    } | std::vector<FilterType>({FilterType::FIR, FilterType::IIR});

    "Decimator - Low-pass Filter Test"_test = [] {
        using namespace gr::testing;
        using T = int;

        constexpr gr::Size_t decimationFactor = 10;

        gr::Graph flow;
        auto&     source    = flow.emplaceBlock<CountingSource<T>>({{"n_samples_max", 10 * decimationFactor}});
        auto&     decimator = flow.emplaceBlock<gr::filter::Decimator<T>>({{"decim", decimationFactor}});
        auto&     sink      = flow.emplaceBlock<CountingSink<T>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(decimator)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect<"out">(decimator).to<"in">(sink)));

        gr::scheduler::Simple<> sched;
        ;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        expect(eq(decimator.decim, decimationFactor));
        expect(eq(decimator.output_chunk_size, static_cast<gr::Size_t>(1)));
        expect(eq(decimator.input_chunk_size, decimationFactor));

        expect(eq(sink.count, static_cast<gr::Size_t>(10)));
    };
};

int main() { /* not needed for UT */ }
