#include <emscripten/emscripten.h>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

using namespace gr::algorithm;
using namespace gr::algorithm::fileio;

static fileio::Writer g_downloadWriter;

FILEIO_EXTERN_C {

    FILEIO_EXPORT
    void test_dialog_read() {
        std::println("test_dialog_read begin isMainThread:{}", isMainThread());

        auto readerExp = fileio::readAsync("dialog:/open", fileio::ReaderConfig{});
        if (!readerExp.has_value()) {
            std::println("Dialog readAsync error: {}", readerExp.error().message);
            return;
        }
        auto        reader = std::move(readerExp.value());
        std::thread readerThread{[reader] mutable {
            std::println("test_dialog_read dialog reader thread started, waiting for user to pick a file");
            auto allData = reader.get();
            if (allData.has_value()) {
                std::println("test_dialog_read finished: {} bytes total", allData.value().size());
            } else {
                std::println("test_dialog_read finished with error: {}", allData.error().message);
            }
        }};
        readerThread.detach();

        std::println("test_dialog_read end: Reader created, dialog triggered, thread detached.");
    }

    // Browser download
    FILEIO_EXPORT
    void test_save_file() {
        std::println("test_save_file begin isMainThread:{}", isMainThread());
        std::string               str = "0123456789";
        std::vector<std::uint8_t> bytes(str.begin(), str.end());

        auto writerExp = fileio::writeAsync("download:/path1/path2/test_file.txt", bytes, fileio::WriterConfig{});
        if (!writerExp.has_value()) {
            std::println("test_save_file error: {}", writerExp.error().message);
            return;
        }
        g_downloadWriter = std::move(writerExp.value());
        std::println("test_save_file: started download");
    }

    FILEIO_EXPORT
    void test_http_get() {
        std::println("test_http_get begin isMainThread: {}", isMainThread());

        auto readerExp = fileio::readAsync("http://localhost:8080/getNumbers", fileio::ReaderConfig{});
        if (!readerExp.has_value()) {
            std::println("test_http_get error: {}", readerExp.error().message);
            return;
        }

        auto        reader = std::move(readerExp.value());
        std::thread readerThread{[reader] mutable {
            auto allData = reader.get();
            if (allData.has_value()) {
                std::println("test_http_get finished: {} bytes total", allData.value().size());
            } else {
                std::println("test_http_get finished with error: {}", allData.error().message);
            }
        }};
        readerThread.detach();
        std::println("test_http_get end");
    }

    FILEIO_EXPORT
    void test_http_get_404() {
        std::println("test_http_get_404 begin isMainThread: {}", isMainThread());

        auto readerExp = fileio::readAsync("http://localhost:8080/return404", fileio::ReaderConfig{});
        if (!readerExp.has_value()) {
            std::println("test_http_get_404 error: {}", readerExp.error().message);
            return;
        }

        auto        reader = std::move(readerExp.value());
        std::thread readerThread{[reader] mutable {
            auto allData = reader.get();
            if (allData.has_value()) {
                std::println("test_http_get_404 finished: {} bytes total", allData.value().size());
            } else {
                std::println("test_http_get_404 finished with error: {}", allData.error().message);
            }
        }};
        readerThread.detach();
        std::println("test_http_get_404 end");
    }

    FILEIO_EXPORT
    void test_http_post() {
        std::println("test_http_post begin isMainThread:{}", isMainThread());
        std::string               str = "0123456789";
        std::vector<std::uint8_t> bytes(str.begin(), str.end());

        auto writerExp = fileio::writeAsync("http://localhost:8080/postNumbers", bytes, fileio::WriterConfig{});
        if (!writerExp.has_value()) {
            std::println("test_http_post error: {}", writerExp.error().message);
            return;
        }

        auto        writer = std::move(writerExp.value());
        std::thread writerThread{[writer] mutable {
            writer.wait();
            auto result = writer.result();
            if (result.has_value()) {
                std::println("test_http_post finished with status: {}, body: {} ", result.value().httpStatus, result.value().httpResponseBody);
            } else {
                std::println("test_http_post finished with error: {} ", result.error().message);
            }
        }};
        writerThread.detach();
    }
} // FILEIO_EXTERN_C

int main() {
#if __EMSCRIPTEN__
    gr::algorithm::fileio::setupEmscriptenDialogCallback();
#endif
    return 0;
}
