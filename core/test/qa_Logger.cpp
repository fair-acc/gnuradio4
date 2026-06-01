#include <source_location>
#include <string_view>
#include <tuple>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Logger.hpp>

using namespace boost::ut;
using namespace std::string_view_literals;

const boost::ut::suite<"gr::log"> _fatal = [] {
    "throws gr::exception"_test = [] { expect(throws<gr::exception>([] { gr::log::fatal("oops"); })); };

    "exception carries supplied message"_test = [] {
        try {
            gr::log::fatal("specific-error-string");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.message, "specific-error-string"sv));
        }
    };

    "captures source_location at call site by default"_test = [] {
        try {
            gr::log::fatal("loc test");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            const std::string_view file = e.sourceLocation.file_name();
            expect(file.contains("qa_Logger.cpp"sv));
        }
    };

    "accepts explicit source_location"_test = [] {
        const auto loc = std::source_location::current();
        try {
            gr::log::fatal("explicit loc", loc);
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.sourceLocation.line(), loc.line()));
            expect(std::string_view{e.sourceLocation.function_name()}.contains(std::string_view{loc.function_name()}));
        }
    };

    "what() includes message and location"_test = [] {
        try {
            gr::log::fatal("what-test");
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            const std::string_view rendered = e.what();
            expect(rendered.contains("what-test"sv));
            expect(rendered.contains("qa_Logger.cpp"sv));
        }
    };

    "empty message is permitted"_test = [] {
        try {
            gr::log::fatal({});
            expect(false) << "fatal should have thrown";
        } catch (const gr::exception& e) {
            expect(eq(e.message, ""sv));
        }
    };

    "warning returns gr::Error with message + source_location"_test = [] {
        const auto rec = gr::log::warning("warn-msg");
        expect(eq(rec.message, "warn-msg"sv));
        expect(std::string_view{rec.sourceLocation.file_name()}.contains("qa_Logger.cpp"sv));
    };

    "warning honours explicit source_location"_test = [] {
        const auto loc = std::source_location::current();
        const auto rec = gr::log::warning("with-loc", loc);
        expect(eq(rec.sourceLocation.line(), loc.line()));
    };

    "error returns gr::Error with message + source_location"_test = [] {
        const auto rec = gr::log::error("err-msg");
        expect(eq(rec.message, "err-msg"sv));
        expect(std::string_view{rec.sourceLocation.file_name()}.contains("qa_Logger.cpp"sv));
    };

    "warning/error are not [[noreturn]] — caller continues"_test = [] {
        int reached = 0;
        std::ignore = gr::log::warning("non-fatal warn");
        ++reached;
        std::ignore = gr::log::error("non-fatal err");
        ++reached;
        expect(eq(reached, 2));
    };
};

int main() { /* statically registered */ }
