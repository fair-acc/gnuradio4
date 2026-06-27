#include <cstddef>
#include <limits>
#include <string_view>

#include <boost/ut.hpp>

#include <gnuradio-4.0/WorkStatus.hpp>

#include <gnuradio-4.0/meta/reflection.hpp>

using namespace boost::ut;

const boost::ut::suite<"gr::work::Status"> _workStatus = [] {
    "sanitiseProcessStatus zeroes counts for ERROR"_test = [] {
        std::size_t processedIn = 3UZ, processedOut = 7UZ;
        gr::work::sanitiseProcessStatus(gr::work::Status::ERROR, processedIn, processedOut);
        expect(eq(processedIn, 0UZ));
        expect(eq(processedOut, 0UZ));
    };

    "sanitiseProcessStatus zeroes counts for INSUFFICIENT_INPUT_ITEMS"_test = [] {
        std::size_t processedIn = 3UZ, processedOut = 7UZ;
        gr::work::sanitiseProcessStatus(gr::work::Status::INSUFFICIENT_INPUT_ITEMS, processedIn, processedOut);
        expect(eq(processedIn, 0UZ));
        expect(eq(processedOut, 0UZ));
    };

    "sanitiseProcessStatus zeroes counts for INSUFFICIENT_OUTPUT_ITEMS"_test = [] {
        std::size_t processedIn = 3UZ, processedOut = 7UZ;
        gr::work::sanitiseProcessStatus(gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS, processedIn, processedOut);
        expect(eq(processedIn, 0UZ));
        expect(eq(processedOut, 0UZ));
    };

    "sanitiseProcessStatus leaves counts untouched for OK"_test = [] {
        std::size_t processedIn = 3UZ, processedOut = 7UZ;
        gr::work::sanitiseProcessStatus(gr::work::Status::OK, processedIn, processedOut);
        expect(eq(processedIn, 3UZ));
        expect(eq(processedOut, 7UZ));
    };

    "sanitiseProcessStatus leaves counts untouched for DONE"_test = [] {
        std::size_t processedIn = 3UZ, processedOut = 7UZ;
        gr::work::sanitiseProcessStatus(gr::work::Status::DONE, processedIn, processedOut);
        expect(eq(processedIn, 3UZ));
        expect(eq(processedOut, 7UZ));
    };

    "computePerformedWork returns zero for any non-OK status"_test = [] {
        for (auto status : {gr::work::Status::ERROR, gr::work::Status::INSUFFICIENT_INPUT_ITEMS, gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS, gr::work::Status::DONE}) {
            expect(eq(gr::work::computePerformedWork(status, 5UZ, 9UZ, /*isSource*/ false), 0UZ));
            expect(eq(gr::work::computePerformedWork(status, 5UZ, 9UZ, /*isSource*/ true), 0UZ));
        }
    };

    "computePerformedWork returns processedOut for sources on OK"_test = [] { expect(eq(gr::work::computePerformedWork(gr::work::Status::OK, 5UZ, 9UZ, /*isSource*/ true), 9UZ)); };

    "computePerformedWork returns processedIn for non-sources on OK"_test = [] { expect(eq(gr::work::computePerformedWork(gr::work::Status::OK, 5UZ, 9UZ, /*isSource*/ false), 5UZ)); };

    "Result default-constructs to unbounded pending work and OK status"_test = [] {
        constexpr gr::work::Result result{};
        expect(eq(result.requested_work, std::numeric_limits<std::size_t>::max()));
        expect(eq(result.performed_work, 0UZ));
        expect(result.status == gr::work::Status::OK);
    };

    "work::Status values format to their identifiers via the EnumTraits specialisation"_test = [] {
        expect(eq(gr::meta::enumName(gr::work::Status::ERROR).value_or(""), std::string_view{"ERROR"}));
        expect(eq(gr::meta::enumName(gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS).value_or(""), std::string_view{"INSUFFICIENT_OUTPUT_ITEMS"}));
        expect(eq(gr::meta::enumName(gr::work::Status::INSUFFICIENT_INPUT_ITEMS).value_or(""), std::string_view{"INSUFFICIENT_INPUT_ITEMS"}));
        expect(eq(gr::meta::enumName(gr::work::Status::DONE).value_or(""), std::string_view{"DONE"}));
        expect(eq(gr::meta::enumName(gr::work::Status::OK).value_or(""), std::string_view{"OK"}));
    };
};

int main() { /* statically registered */ }
