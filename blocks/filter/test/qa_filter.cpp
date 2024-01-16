#include <boost/ut.hpp>

#include <fmt/format.h>

#include <gnuradio-4.0/Block.hpp>

#include <gnuradio-4.0/filter/time_domain_filter.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

template<typename T, typename Range>
    requires std::floating_point<T>
constexpr size_t
estimate_settling_time(const Range &step_response, std::size_t offset = 0, T step_value = 1.0, T threshold = 0.001) {
    if (offset >= step_response.size()) {
        throw std::out_of_range("Offset is greater than the size of the step response.");
    }
    const T lower_bound = step_value - threshold;
    const T upper_bound = step_value + threshold;

    auto    begin       = step_response.begin() + static_cast<typename Range::difference_type>(offset);
    auto    end         = step_response.end();

    auto    it          = std::find_if(begin, end, [lower_bound, upper_bound](T sample) { return sample >= lower_bound && sample <= upper_bound; });

    // If no such sample is found, return an error
    if (it == end) {
        throw std::runtime_error("No settling found within the given threshold.");
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
        std::vector<double> iir_coeffs_b{ 0.55, 0 };
        std::vector<double> iir_coeffs_a{ 1, -0.45 };

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

        //        fmt::print("FIR      filter settling time: {} ms\n", fir_settling_time);
        //        fmt::print("IIR (I)  filter settling time: {} ms\n", iir_settling_time1);
        //        fmt::print("IIR (II) filter settling time: {} ms\n", iir_settling_time2);
    };

    "IIR equality tests"_test = [] {
        //        std::vector<double>              iir_coeffs_b{ 0.55, 0 };
        //        std::vector<double>              iir_coeffs_a{ 1, -0.45 };
        std::vector<double>               iir_coeffs_b{ 0.020083365564211, 0.040166731128423, 0.020083365564211 };
        std::vector<double>               iir_coeffs_a{ 1.0, -1.561018075800718, 0.641351538057563 };

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
        iir_filter_IIT.b           = iir_coeffs_b;
        iir_filter_IIT.a           = iir_coeffs_a;

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
            fmt::print("input[{:2}]={}-> IIR= {:4.2f} (I) {:4.2f} (II) {:4.2f} (I-T) {:4.2f} (II-T)\n", //
                       i, input, form_I, form_II, form_I_T, form_II_T);
#endif
        }
    };
};

int
main() { /* not needed for UT */
}