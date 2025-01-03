#include <boost/ut.hpp>

#include <gnuradio-4.0/DataSet.hpp>

const boost::ut::suite<"DataSet<T>"> _dataSetAPI = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using namespace std::string_view_literals;

    "DataSet axis + signal access"_test = [] {
        gr::DataSet<float> ds;

        ds.axis_names  = {"Time", "Frequency"};
        ds.axis_units  = {"s", "Hz"};
        ds.axis_values = {
            {0.f, 1.f, 2.f, 3.f},        // axis=0
            {100.f, 200.f, 300.f, 400.f} // axis=1
        };

        ds.signal_names      = {"sigA", "sigB"};
        ds.signal_quantities = {"quantityA", "quantityB"};
        ds.signal_units      = {"unitA", "unitB"};

        ds.signal_values = {
            0.f, 1.f, 2.f, 3.f,    // data for sigA
            10.f, 11.f, 12.f, 13.f // data for sigB
        };
        ds.signal_ranges = {
            {0.f, 3.f},  // for sigA
            {10.f, 13.f} // for sigB
        };

        expect(eq(ds.axisCount(), 2UZ));
        expect(eq(ds.size(), 2UZ));

        "non-const usage - axes"_test = [&] {
            ds.axisName(0) = "Time_modified";
            expect(eq(ds.axisName(0), "Time_modified"sv));

            ds.axisUnit(1) = "kHz";
            expect(eq(ds.axisUnit(1), "kHz"sv));

            auto timeVals = ds.axisValues(0);
            expect(eq(timeVals.size(), 4UZ));
            timeVals[2] = 42.f;
            expect(eq(ds.axisValues(0)[2], 42.f));
        };

        "non-const usage - signals"_test = [&] {
            ds.signalName(1) = "sigB_mod";
            expect(eq(ds.signalName(1), "sigB_mod"sv));

            auto sA = ds.signalValues(0);
            expect(eq(sA.size(), 4UZ));
            sA[3] = 99.f;
            expect(eq(ds.signalValues(0)[3], 99.f));

            ds.signalRange(1).min = 9.f;  // set min
            ds.signalRange(1).max = 15.f; // set max
            expect(eq(ds.signalRange(1).min, 9.f));
            expect(eq(ds.signalRange(1).max, 15.f));
        };

        "const usage - reading axes"_test = [&] {
            const auto& constDs = ds; // forces use of const methods

            expect(eq(constDs.axisCount(), 2UZ));
            expect(eq(constDs.axisName(0), "Time_modified"sv));
            expect(eq(constDs.axisUnit(1), "kHz"sv));
            expect(eq(constDs.axisValues(0)[2], 42.f));
        };

        "const usage - reading signals"_test = [&] {
            const auto& constDs = ds; // forces use of const methods
            expect(eq(constDs.signalName(1), "sigB_mod"sv));
            expect(eq(constDs.signalValues(0)[3], 99.f));
            expect(eq(constDs.signalRange(1).max, 15.f));
        };

        "out-of-range checks - axes"_test = [&] {
            expect(throws([&] { std::ignore = ds.axisName(99); }));
            expect(throws([&] { std::ignore = ds.axisValues(2); }));
        };

        "out-of-range checks - signals"_test = [&] {
            expect(throws([&] { std::ignore = ds.signalName(99); }));
            expect(throws([&] { std::ignore = ds.signalValues(99); }));
            expect(throws([&] { std::ignore = ds.signalRange(99); }));
        };
    };
};

int main() { /* tests are statically executed */ }
