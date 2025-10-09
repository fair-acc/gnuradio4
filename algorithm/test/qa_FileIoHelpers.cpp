#include <boost/ut.hpp>

#include <gnuradio-4.0/algorithm/fileio/FileIoHelpers.hpp>

const boost::ut::suite<"FileIO Helpers tests"> fileIoHelpersTests = [] {
    using namespace boost::ut;
    using namespace gr;
    namespace fio = gr::algorithm::fileio::detail;
    using namespace std::string_literals;

    "startsWithScheme"_test = [] {
        expect(fio::startsWithScheme("http://x") == true);
        expect(fio::startsWithScheme("https://x") == true);
        expect(fio::startsWithScheme("file:/x") == true);
        expect(fio::startsWithScheme("ABC:stuff") == true);
        expect(fio::startsWithScheme("abc") == false);
        expect(fio::startsWithScheme("1abc:") == false);
        expect(fio::startsWithScheme("ab+c:") == false);
        expect(fio::startsWithScheme(":x") == false);
    };

    "schemeOf"_test = [] {
        expect(eq(fio::schemeOf("http://x"), "http"s));
        expect(eq(fio::schemeOf("file:/x"), "file"s));
        expect(eq(fio::schemeOf("ABC:stuff"), "ABC"s));
        expect(eq(fio::schemeOf("abc"), ""s));
        expect(eq(fio::schemeOf("ab+c:"), ""s));
        expect(eq(fio::schemeOf(":x"), ""s));
    };

    "isHttpUri / isFileUri"_test = [] {
        expect(fio::isHttpUri("http://x") == true);
        expect(fio::isHttpUri("https://x") == true);
        expect(fio::isHttpUri("ftp://x") == false);
        expect(fio::isHttpUri("HTTP://x") == false);

        expect(fio::isFileUri("file:/x") == true);
        expect(fio::isFileUri("FILE:/x") == false);
    };

    "stripFileUriLocal - root dir"_test = [] {
        // All should normalize to "/"
        expect(eq(fio::stripFileUri("file:").value(), "/"s));
        expect(eq(fio::stripFileUri("file:/").value(), "/"s));
        expect(eq(fio::stripFileUri("file:///").value(), "/"s));
        expect(eq(fio::stripFileUri("file://localhost").value(), "/"s));
        expect(eq(fio::stripFileUri("file:////").value(), "/"s));
        expect(eq(fio::stripFileUri("file://localhost").value(), "/"s));
    };

    "stripFileUriLocal - absolute paths"_test = [] {
        expect(eq(fio::stripFileUri("file:/usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::stripFileUri("file:///usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::stripFileUri("file://localhost/usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::stripFileUri("file:/usr///local/bin").value(), "/usr///local/bin"s));
        expect(eq(fio::stripFileUri("file:usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::stripFileUri("file:////////usr/local/bin").value(), "/usr/local/bin"s));
    };

    "stripFileUriLocal - reject non-local authorities"_test = [] {
        auto r1 = fio::stripFileUri("file://host/etc");
        expect(!r1.has_value());

        auto r2 = fio::stripFileUri("file://example.com");
        expect(!r2.has_value());
    };
};

int main() {}
