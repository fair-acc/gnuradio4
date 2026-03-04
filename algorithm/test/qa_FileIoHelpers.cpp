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
        expect(fio::startsWithScheme("abc") == false);
        expect(fio::startsWithScheme("1abc:") == false);
        expect(fio::startsWithScheme("ab+c:") == false);
        expect(fio::startsWithScheme(":x") == false);
    };

    "schemeOf"_test = [] {
        expect(eq(fio::schemeOf("http://x"), "http"s));
        expect(eq(fio::schemeOf("file:/x"), "file"s));
        expect(eq(fio::schemeOf("abc"), ""s));
        expect(eq(fio::schemeOf("ab+c:"), ""s));
        expect(eq(fio::schemeOf(":x"), ""s));
    };

    "isHttpUri / isFileUri"_test = [] {
        expect(fio::isHttpUri("http://x") == true);
        expect(fio::isHttpUri("https://x") == true);
        expect(fio::isHttpUri("ftp://x") == false);
        expect(fio::isHttpUri("HTTP://x") == true);
        expect(fio::isFileUri("file:/x") == true);
        expect(fio::isFileUri("FILE:/x") == true);
    };

    "classifyUri"_test = [] {
        expect(fio::classifyUri("/tmp/x") == fio::UriKind::LocalPath);
        expect(fio::classifyUri("tmp/x") == fio::UriKind::LocalPath);
        expect(fio::classifyUri("file:/tmp/x") == fio::UriKind::FileUri);
        expect(fio::classifyUri("http://x") == fio::UriKind::HttpUri);
        expect(fio::classifyUri("download:/x") == fio::UriKind::DownloadUri);
        expect(fio::classifyUri("dialog:/open") == fio::UriKind::DialogUri);
        expect(fio::classifyUri("wrong_sch://wrong_uri") == fio::UriKind::UnsupportedUri);
        expect(fio::classifyUri("ftp://x") == fio::UriKind::UnsupportedUri);
    };

    "toLocalPath"_test = [] {
        expect(eq(fio::toLocalPath("/tmp/x").value(), "/tmp/x"s));
        expect(eq(fio::toLocalPath("tmp/x").value(), "tmp/x"s));
        expect(eq(fio::toLocalPath("file:/tmp/x").value(), "/tmp/x"s));
        expect(eq(fio::toLocalPath("file:/").value(), "/"s));
        expect(eq(fio::toLocalPath("file:///").value(), "/"s));
        expect(eq(fio::toLocalPath("file://localhost").value(), "/"s));
        expect(eq(fio::toLocalPath("file:////").value(), "/"s));
        expect(eq(fio::toLocalPath("file:/usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::toLocalPath("file:///usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::toLocalPath("file://localhost/usr/local/bin").value(), "/usr/local/bin"s));
        expect(eq(fio::toLocalPath("file:/usr///local/bin").value(), "/usr///local/bin"s));
        expect(eq(fio::toLocalPath("file:////////usr/local/bin").value(), "/usr/local/bin"s));
        expect(!fio::toLocalPath("http://x").has_value());
        auto r1 = fio::toLocalPath("file://host/etc");
        expect(!r1.has_value());

        auto r2 = fio::toLocalPath("file://example.com");
        expect(!r2.has_value());
    };
};

int main() {}
