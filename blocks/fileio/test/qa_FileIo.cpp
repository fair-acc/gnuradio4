#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/fileio/FileIo.hpp>

#include <format>

#ifndef __EMSCRIPTEN__
#include <httplib.h>
#endif

template<typename T>
struct SimpleFileSource : gr::Block<SimpleFileSource<T>> {

    gr::PortOut<T> out;

    gr::Annotated<std::string, "file name"> file_name;

    gr::blocks::fileio::Subscription _subscription;

    GR_MAKE_REFLECTABLE(SimpleFileSource, out, file_name);

    void start() {
        auto subExp = gr::blocks::fileio::subscribe(file_name, {});
        if (!subExp.has_value()) {
            throw gr::exception(subExp.error().message);
        }
        _subscription = std::move(subExp.value());
    }

    void stop() { _subscription.cancel(); }

    [[nodiscard]] constexpr gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) noexcept {
        if (_subscription.finished()) {
            return gr::work::Status::DONE;
        }

        if (auto chunk = _subscription.poll(outSpan.size())) {
            if (chunk.size() > outSpan.size()) {
                std::println("Missing data, buffer is full: {}", chunk.size() - outSpan.size());
            }
            std::copy(chunk.begin(), std::next(chunk.begin(), static_cast<std::ptrdiff_t>(outSpan.size())), outSpan.begin());
        } else if (_subscription.error()) {
            throw gr::exception(_subscription.error().value().message);
        }

        return gr::work::Status::OK;
    }
};

[[nodiscard]] inline std::string createTestFile(std::string_view strFilePath, std::source_location srcLocation = std::source_location::current()) {
    using namespace boost::ut;
    namespace fs = std::filesystem;

    fs::path path{strFilePath};
    fs::create_directories(path.parent_path());
    std::string expectedString;
    for (int i = 0; i < 100; ++i) {
        std::format_to(std::back_inserter(expectedString), "{}", i);
    }
    std::ofstream out(path, std::ios::binary);
    expect(out.is_open()) << std::format("{}", srcLocation);
    out.write(expectedString.data(), static_cast<std::streamsize>(expectedString.size()));
    out.close();

    expect(fs::exists(path)) << std::format("{}", srcLocation);
    return expectedString;
}

const boost::ut::suite<"FileIO tests"> fileIOTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::blocks;
    namespace fs = std::filesystem;

    "FileIO - Native local"_test = [&] {
        const std::string path           = "/tmp/gr4_fileio_test/TestFileIo.bin";
        std::string       expectedString = createTestFile(path);

        fileio::RequestOptions opts;
        opts.chunkBytes    = 8;
        opts.bufferMinSize = 128;

        auto subExp = fileio::subscribe(std::format("file:/{}", path), opts);
        expect(subExp.has_value());
        auto                      sub = std::move(subExp.value());
        std::vector<std::uint8_t> allBytes;
        while (!sub.finished()) {
            if (auto chunk = sub.poll()) {
                allBytes.insert(allBytes.end(), chunk->begin(), chunk->end());
            } else if (sub.error()) {
                // handle *sub.error()
                break;
            }
        }
        expect(!sub.isRunning());

        std::string outString(reinterpret_cast<const char*>(allBytes.data()), allBytes.size());
        expect(eq(expectedString, outString));

        std::error_code ec;
        fs::remove(path, ec);
        expect(!ec);
    };

    "FileIO - Native http"_test = [&] {
        const std::string path           = "/tmp/gr4_fileio_test/TestFileIoHttp.bin";
        std::string       expectedString = createTestFile(path);
#ifndef __EMSCRIPTEN__
        httplib::Server server;
        server.Get("/getNumbers", [&](const httplib::Request, httplib::Response& res) { res.set_content(expectedString, "text/plain"); });

        auto thread = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
#endif

        auto subExp = fileio::subscribe("http://localhost:8080/getNumbers", {});
        expect(subExp.has_value());
        auto                      sub = std::move(subExp.value());
        std::vector<std::uint8_t> allBytes;
        while (!sub.finished()) {
            if (auto chunk = sub.poll()) {
                allBytes.insert(allBytes.end(), chunk->begin(), chunk->end());
            } else if (sub.error()) {
                // handle *sub.error()
                break;
            }
        }
        std::string outString(reinterpret_cast<const char*>(allBytes.data()), allBytes.size());
        expect(eq(expectedString, outString));
        expect(!sub.isRunning());

        std::error_code ec;
        fs::remove(path, ec);
        expect(!ec);

#ifndef __EMSCRIPTEN__
        server.stop();
        thread.join();
#endif
    };
};

int main() { /* not needed for UT */ }
