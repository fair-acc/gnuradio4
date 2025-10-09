#include <boost/ut.hpp>

#include <format>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

#ifndef __EMSCRIPTEN__
#include <httplib.h>
#endif

struct FileIoSource : gr::Block<FileIoSource> {
    gr::PortOut<std::uint8_t> out;

    gr::Annotated<std::string, "URI"> uri;

    gr::algorithm::fileio::Reader _reader;

    GR_MAKE_REFLECTABLE(FileIoSource, out, uri);

    void start() {
        auto readerExp = gr::algorithm::fileio::readAsync(uri, {});
        if (!readerExp.has_value()) {
            throw gr::exception(readerExp.error().message);
        }
        _reader = std::move(readerExp.value());
    }

    void stop() { _reader.cancel(); }

    [[nodiscard]] constexpr gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) noexcept {
        bool        finished          = false;
        std::size_t nSamplesToPublish = 0;
        _reader.poll(
            [&finished, &outSpan, &nSamplesToPublish](const auto& res) {
                finished = res.isFinal;
                if (res.data.has_value()) {
                    if (outSpan.size() >= res.data->size()) {
                        std::memcpy(outSpan.data(), res.data->data(), res.data->size());
                        nSamplesToPublish = res.data->size();
                    } else {
                        std::println("Error: This error should not happen, outSpan.size() < res.data->size() ({} < {})", outSpan.size(), res.data->size());
                    }
                } else {
                    std::println("Error: {}", res.data.error().message);
                }
            },
            outSpan.size(), false);

        outSpan.publish(nSamplesToPublish);
        return finished ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

struct FileIoSink : gr::Block<FileIoSink> {
    gr::PortIn<std::uint8_t> in;

    gr::Annotated<std::string, "URI">                       uri;
    gr::Annotated<gr::algorithm::fileio::WriteMode, "Mode"> mode{gr::algorithm::fileio::WriteMode::append};

    gr::algorithm::fileio::Writer _writer;

    GR_MAKE_REFLECTABLE(FileIoSink, in, uri, mode);

    [[nodiscard]] constexpr gr::work::Status processBulk(gr::InputSpanLike auto const& inSpan) noexcept {
        if (!inSpan.empty()) {
            auto spanBytes      = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(inSpan.data()), inSpan.size());
            auto writeResultExp = gr::algorithm::fileio::write(uri, spanBytes, gr::algorithm::fileio::WriterConfig{.mode = mode});
            if (!writeResultExp.has_value()) {
                throw gr::exception(writeResultExp.error().message, writeResultExp.error().sourceLocation);
            }
        }
        return gr::work::Status::OK;
    }
};

[[nodiscard]] inline std::string createTestString() {
    std::string expectedString;
    for (int i = 0; i < 100; ++i) {
        std::format_to(std::back_inserter(expectedString), "{}", i);
    }
    return expectedString;
}

[[nodiscard]] inline std::string createTestFile(std::string_view strFilePath, std::source_location srcLocation = std::source_location::current()) {
    using namespace boost::ut;
    namespace fs = std::filesystem;

    fs::path path{strFilePath};
    fs::create_directories(path.parent_path());
    std::string   expectedString = createTestString();
    std::ofstream out(path, std::ios::binary);
    expect(out.is_open()) << std::format("{}", srcLocation);
    out.write(expectedString.data(), static_cast<std::streamsize>(expectedString.size()));
    out.close();

    expect(fs::exists(path)) << std::format("{}", srcLocation);
    return expectedString;
}

#if __EMSCRIPTEN__
[[nodiscard]] inline std::expected<gr::algorithm::fileio::Reader, gr::Error> readAsyncEmscriptenHttpWorkerThread(std::string_view uri, gr::algorithm::fileio::ReaderConfig config = {}) {
    using namespace gr::algorithm::fileio;

    if (!detail::isHttpUri(uri)) {
        return std::unexpected(gr::Error{std::format("readAsyncEmscriptenHttpWorkerThread only http uri: {}", uri)});
    }
    auto state = std::make_shared<ReaderState>(std::string(uri), config);
    runHttpGetEmscripten<false>(state); // use worker thread, main has issues for unit test with `node`
    return Reader{state};
}

[[nodiscard]] inline std::expected<gr::algorithm::fileio::Writer, gr::Error> writeAsyncEmscriptenHttpWorkerThread(std::string_view uri, std::span<const std::uint8_t> data, gr::algorithm::fileio::WriterConfig config = {}) {
    using namespace gr::algorithm::fileio;

    if (!detail::isHttpUri(uri)) {
        return std::unexpected(gr::Error{std::format("writeAsyncEmscriptenHttpWorkerThread only http uri: {}", uri)});
    }
    auto state = std::make_shared<WriterState>(std::string(uri), config);
    state->data.assign(data.begin(), data.end());
    detail::runHttpPostEmscripten<false>(state); // use worker thread, main has issues for unit test with `node`
    return Writer{state};
}
#endif

struct ReadResult {
    std::vector<std::vector<std::uint8_t>> allData;
    std::size_t                            dataCounter  = 0;
    std::size_t                            errorCounter = 0;
};

template<typename TForEachCallback>
[[nodiscard]] inline ReadResult getReadResult(gr::algorithm::fileio::Reader& reader, std::size_t maxSize, bool doWait, TForEachCallback callback) {
    bool       finished = false;
    ReadResult readResult;
    while (!finished) {
        reader.poll(
            [&finished, &readResult, &callback](const auto& res) {
                finished = res.isFinal;
                if (res.data.has_value()) {
                    auto data = res.data.value();
                    if (!data.empty()) {
                        readResult.allData.emplace_back(std::vector<std::uint8_t>{data.begin(), data.end()});
                        readResult.dataCounter++;
                    }
                } else {
                    readResult.errorCounter++;
                    std::println("Error: {}", res.data.error().message);
                }
                callback();
            },
            maxSize, doWait);
    }
    return readResult;
};

[[nodiscard]] inline ReadResult getReadResult( //
    gr::algorithm::fileio::Reader& reader, std::size_t maxSize = std::numeric_limits<std::size_t>::max(), bool doWait = false) {
    return getReadResult(reader, maxSize, doWait, []() {});
}

[[nodiscard]] std::string joinBytesToString(const std::vector<std::vector<std::uint8_t>>& allData) {
    const auto  totalSize = std::accumulate(allData.begin(), allData.end(), std::size_t{0}, [](std::size_t acc, const auto& v) { return acc + v.size(); });
    std::string result;
    result.reserve(totalSize);
    for (const auto& v : allData) {
        result.append(reinterpret_cast<const char*>(v.data()), v.size());
    }
    return result;
}

[[nodiscard]] std::string bytesToString(const std::vector<std::uint8_t>& bytes) { return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }

const boost::ut::suite<"FileIO local - Native + Emscripten"> fileIoLocalTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::algorithm;

    struct LocalFileParams {
        // Note: Emscripten writes to MEMFS
        std::string uri            = "file:/tmp/gr4_fileio_test/TestFileIo.bin";
        std::string localPath      = fileio::detail::stripFileUri(uri).value();
        std::string expectedString = createTestFile(localPath);

        void cleanup() {
            std::error_code ec;
            std::filesystem::remove(localPath, ec);
            expect(!ec);
        }
    };

    "FileIO - Local"_test = [&] {
        std::println("FileIO - Local begin");

        LocalFileParams params;
        auto            readerExp = fileio::readAsync(params.uri, fileio::ReaderConfig{.chunkBytes = 11});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto reader     = std::move(readerExp.value());
            auto subResults = getReadResult(reader);
            expect(eq(subResults.dataCounter, 18uz)); // 190 bytes / 11 bytes per chunk
            expect(eq(subResults.errorCounter, 0uz));
            expect(eq(params.expectedString, joinBytesToString(subResults.allData)));
        }
        params.cleanup();
        std::println("FileIO - Local end");
    };

    "FileIO - Local with get()"_test = [&] {
        std::println("FileIO - Local  with get() begin");

        LocalFileParams params;
        auto            readerExp = fileio::readAsync(params.uri, fileio::ReaderConfig{.chunkBytes = 11});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        reader = std::move(readerExp.value());
            std::thread readerThread([&]() {
                auto expData = reader.get();
                expect(expData.has_value());
                if (expData.has_value()) {
                    expect(eq(params.expectedString, bytesToString(expData.value())));
                }
            });
            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        params.cleanup();
        std::println("FileIO - Local  with get() end");
    };

    "FileIO - Local with offset"_test = [&] {
        std::println("FileIO - Local with offset begin");

        LocalFileParams   params;
        const std::size_t offset = 10;
        expect(offset < params.expectedString.size());

        auto readerExp = fileio::readAsync(params.uri, fileio::ReaderConfig{.chunkBytes = 11, .offset = offset});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);

            expect(!sub.cancelRequested());
            expect(eq(subResults.errorCounter, 0uz));
            const std::string expectedTail = params.expectedString.substr(offset);
            expect(eq(expectedTail, joinBytesToString(subResults.allData)));
        }

        params.cleanup();
        std::println("FileIO - Local with offset end");
    };

    "FileIO - Local offset >= size"_test = [&] {
        std::println("FileIO - Local offset >= size begin");

        LocalFileParams params;
        auto            readerExp = fileio::readAsync(params.uri, fileio::ReaderConfig{.chunkBytes = 11, .offset = params.expectedString.size()});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);

            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 0uz));
            expect(eq(subResults.errorCounter, 0uz));
            expect(eq(std::string{}, joinBytesToString(subResults.allData)));
        }

        params.cleanup();
        std::println("FileIO - Local offset >= size end");
    };

    "FileIO - Local - file not found"_test = [&] {
        std::println("FileIO - Local - file not found begin");
        const std::string path      = "file:/path/does/not/exists.bin";
        auto              readerExp = fileio::readAsync(path, {});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);
            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 0uz));
            expect(eq(subResults.errorCounter, 1uz));
            expect(eq(std::string{}, joinBytesToString(subResults.allData)));
        }
        std::println("FileIO - Local - file not found end");
    };

    "FileIO - Writer local overwrite + append"_test = [&] {
        std::println("FileIO - Writer local append begin");

        const std::string uri     = "file:/tmp/gr4_fileio_test/TestFileIoWriterOverwriteAppend.bin";
        auto              pathExp = fileio::detail::stripFileUri(uri);
        expect(pathExp.has_value());
        const std::string     localPath = pathExp.value();
        std::filesystem::path path{localPath};

        const std::string part1            = "AAA_" + createTestString();
        const std::string part2            = "BBB_" + createTestString();
        const std::string expectedCombined = part1 + part2;

        std::thread writeThread([&]() {
            { // First write: overwrite
                std::vector<std::uint8_t> bytes(part1.begin(), part1.end());
                auto                      writeResultExp = fileio::write(uri, bytes, fileio::WriterConfig{.mode = fileio::WriteMode::overwrite});
                expect(writeResultExp.has_value());
            }

            { // Second write: append
                std::vector<std::uint8_t> bytes(part2.begin(), part2.end());
                auto                      writeResultExp = fileio::write(uri, bytes, fileio::WriterConfig{.mode = fileio::WriteMode::append});
                expect(writeResultExp.has_value());
            }
        });
        if (writeThread.joinable()) {
            writeThread.join();
        }

        auto readerExp = fileio::readAsync(uri, fileio::ReaderConfig{});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto reader      = std::move(readerExp.value());
            auto readResults = getReadResult(reader);
            expect(eq(expectedCombined, joinBytesToString(readResults.allData)));
        }

        std::error_code ec;
        std::filesystem::remove(path, ec);
        expect(!ec);

        std::println("FileIO - Writer local append end");
    };
};

#ifndef __EMSCRIPTEN__
const boost::ut::suite<"FileIO Native tests"> fileIoNativeTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::algorithm;
    namespace fs = std::filesystem;

    "FileIO - Native http"_test = [&] {
        std::println("FileIO - Native http begin");
        std::string expectedString = createTestString();

        httplib::Server server;
        server.Get("/getNumbers", [&](const httplib::Request&, httplib::Response& res) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            res.set_content(expectedString, "text/plain");
        });

        auto threadServer = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();
        auto readerExp = fileio::readAsync("http://localhost:8080/getNumbers", fileio::ReaderConfig{.chunkBytes = 11});
#if GR_FILEIO_CPR_AVAILABLE
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);
            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 18uz)); // 190 bytes / 11 bytes per chunk
            expect(eq(subResults.errorCounter, 0uz));
            expect(eq(expectedString, joinBytesToString(subResults.allData)));
        }
#else
        expect(!readerExp.has_value());
#endif
        server.stop();
        threadServer.join();
        std::println("FileIO - Native http end");
    };

    "FileIO - Native http - 404"_test = [&] {
        std::println("FileIO - Native http - 404 begin");

        httplib::Server server;
        server.Get("/return404", [](const httplib::Request, httplib::Response& res) { res.status = httplib::StatusCode::NotFound_404; });
        auto threadServer = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();

        auto readerExp = fileio::readAsync("http://localhost:8080/return404", {});
#if GR_FILEIO_CPR_AVAILABLE
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);
            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 0uz));
            expect(eq(subResults.errorCounter, 1uz));
            expect(eq(std::string{}, joinBytesToString(subResults.allData)));
        }
#else
        expect(!readerExp.has_value());
#endif
        server.stop();
        threadServer.join();
        std::println("FileIO - Native http - 404 end");
    };

    "FileIO - Native http - long polling with cancel"_test = [&] {
        std::println("FileIO - Native http - long polling with cancel begin");
        std::string expectedString = createTestString();

        httplib::Server server;
        server.Get("/getNumbers", [&](const httplib::Request&, httplib::Response& res) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            res.set_content(expectedString, "text/plain");
        });

        auto threadServer = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();

        fileio::ReaderConfig config;
        config.chunkBytes  = 1000; // guarantee one chunk per request, 190 bytes per request is expected
        config.longPolling = true;

        auto readerExp = fileio::readAsync("http://localhost:8080/getNumbers", config);
#if GR_FILEIO_CPR_AVAILABLE
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto                     sub = std::move(readerExp.value());
            std::atomic<std::size_t> cancelDataCounter{0};

            std::thread cancelThread{[&cancelDataCounter, &sub] {
                while (cancelDataCounter.load(std::memory_order_acquire) < 10) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                sub.cancel();
            }};

            auto subResults = getReadResult(sub, std::numeric_limits<std::size_t>::max(), true, [&cancelDataCounter]() { cancelDataCounter.fetch_add(1, std::memory_order_release); });
            expect(sub.cancelRequested());
            expect(ge(subResults.dataCounter, 9uz));
            expect(eq(subResults.errorCounter, 0uz));

            for (std::size_t i = 0; i < subResults.allData.size(); i++) {
                expect(eq(expectedString, bytesToString(subResults.allData[i])));
            }
            if (cancelThread.joinable()) {
                cancelThread.join();
            }
        }
#else
        expect(!readerExp.has_value());
#endif
        server.stop();
        if (threadServer.joinable()) {
            threadServer.join();
        }
        std::println("FileIO - Native http - long polling with cancel end");
    };

    "FileIO - Native http - long polling with error"_test = [&] {
        std::println("FileIO - Native http - long polling with error begin");
        std::string expectedString = createTestString();

        std::atomic<std::size_t> getNumbersAndErrorCounter{0};
        httplib::Server          server;
        server.Get("/getNumbersAndError", [&](const httplib::Request&, httplib::Response& res) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const std::size_t n = getNumbersAndErrorCounter.fetch_add(1, std::memory_order_release) + 1;
            if (n <= 2) {
                res.status = 200;
                res.set_content(expectedString, "text/plain");
            } else {
                res.status = 500;
            }
        });

        auto threadServer = std::thread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();

        fileio::ReaderConfig config;
        config.chunkBytes  = 1000; // 190 bytes per request is expected
        config.longPolling = true;

        auto readerExp = fileio::readAsync("http://localhost:8080/getNumbersAndError", config);
#if GR_FILEIO_CPR_AVAILABLE
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub);
            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 2uz));
            expect(eq(subResults.errorCounter, 1uz));

            for (std::size_t i = 0; i < subResults.allData.size(); i++) {
                expect(eq(expectedString, bytesToString(subResults.allData[i])));
            }
        }
#else
        expect(!readerExp.has_value());
#endif
        server.stop();
        if (threadServer.joinable()) {
            threadServer.join();
        }
        std::println("FileIO - Native http - long polling with error end");
    };

    "FileIO - Writer native http POST"_test = [&] {
        std::println("\n\n\nFileIO - Writer native http POST begin");

        const std::string expectedBody   = createTestString();
        const std::string serverResponse = "OK";

        httplib::Server server;
        server.Post("/postNumbers", [&](const httplib::Request& req, httplib::Response& res) {
            expect(eq(req.body, expectedBody));
            res.set_content(serverResponse, "text/plain");
        });
        std::thread serverThread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();

        std::vector<std::uint8_t> bytes(expectedBody.begin(), expectedBody.end());
        auto                      writeResultExp = fileio::write("http://localhost:8080/postNumbers", bytes, fileio::WriterConfig{});
#if GR_FILEIO_CPR_AVAILABLE
        expect(writeResultExp.has_value());
        if (writeResultExp.has_value()) {
            auto res = std::move(writeResultExp.value());
            expect(eq(res.httpStatus, 200l));
            expect(eq(res.httpResponseBody, serverResponse));
        }
#else
        expect(!writeResultExp.has_value());
#endif
        server.stop();
        if (serverThread.joinable()) {
            serverThread.join();
        }

        std::println("FileIO - Writer native http POST end");
    };

    "FileIO - Writer native http POST - error"_test = [&] {
        std::println("FileIO - Writer native http POST - error begin");

        httplib::Server server;
        server.Post("/postError", [&](const httplib::Request& /*req*/, httplib::Response& res) { res.status = httplib::StatusCode::InternalServerError_500; });

        std::thread serverThread{[&server] { server.listen("localhost", 8080); }};
        server.wait_until_ready();

        std::string               body = createTestString();
        std::vector<std::uint8_t> bytes(body.begin(), body.end());
        auto                      writeResultExp = fileio::write("http://localhost:8080/postError", std::span<const std::uint8_t>(bytes.data(), bytes.size()), fileio::WriterConfig{});
        expect(!writeResultExp.has_value()); // should be an error for HTTP 500 OR HTTP disabled
        server.stop();
        if (serverThread.joinable()) {
            serverThread.join();
        }

        std::println("FileIO - Writer native http POST - error end");
    };
};
#endif

#ifdef __EMSCRIPTEN__
const boost::ut::suite<"FileIO Emscripten tests"> fileIoEmscriptenTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::algorithm;
    namespace fs = std::filesystem;

    std::thread serverThread{[&]() {
        // see ./fileio_pre.js for the emscripten server implementation
        emscripten_run_script("startServer();");
    }};

    "FileIO - Emscripten reader http"_test = [&] {
        std::println("Emscripten http begin");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        fileio::ReaderConfig config;
        config.chunkBytes = 11;

        auto readerExp = readAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/getNumbers", config);
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        sub = std::move(readerExp.value());
            std::thread readerThread{[&]() {
                auto subResults = getReadResult(sub);
                expect(!sub.cancelRequested());
                expect(eq(subResults.dataCounter, 18uz)); // 190 bytes / 11 bytes per chunk
                expect(eq(subResults.errorCounter, 0uz));
                expect(eq(createTestString(), joinBytesToString(subResults.allData)));
            }};

            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        std::println("Emscripten http end");
    };

    "FileIO - Emscripten reader http - 404"_test = [&] {
        std::println("Emscripten http - 404 begin");

        auto readerExp = readAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/return404", {});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        sub = std::move(readerExp.value());
            std::thread readerThread{[&]() {
                auto subResults = getReadResult(sub);
                expect(!sub.cancelRequested());
                expect(eq(subResults.dataCounter, 0uz));
                expect(eq(subResults.errorCounter, 1uz));
                expect(eq(std::string{}, joinBytesToString(subResults.allData)));
            }};

            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        std::println("Emscripten http - 404 end");
    };

    "FileIO - Emscripten reader http long polling with cancel"_test = [&] {
        std::println("FileIO - Emscripten http long polling with cancel begin");
        std::string          expectedString = createTestString();
        fileio::ReaderConfig config;
        config.chunkBytes  = 1000; // expected data is 190 bytes per request
        config.longPolling = true;

        auto readerExp = readAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/getNumbers", config);
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub = std::move(readerExp.value());

            std::atomic<std::size_t> cancelDataCounter{0};

            std::thread cancelThread{[&cancelDataCounter, &sub] {
                while (cancelDataCounter.load(std::memory_order_acquire) < 10) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                sub.cancel();
            }};

            std::thread readerThread{[&]() {
                auto subResults = getReadResult(sub, std::numeric_limits<std::size_t>::max(), true, [&cancelDataCounter]() { cancelDataCounter.fetch_add(1, std::memory_order_release); });
                expect(sub.cancelRequested());
                expect(ge(subResults.dataCounter, 9uz));
                expect(eq(subResults.errorCounter, 0uz));

                for (std::size_t i = 0; i < subResults.allData.size(); i++) {
                    expect(eq(expectedString, bytesToString(subResults.allData[i])));
                }
            }};

            if (cancelThread.joinable()) {
                cancelThread.join();
            };

            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        std::println("FileIO - Emscripten http long polling with cancel end");
    };

    "FileIO - Emscripten reader http long polling with error"_test = [&] {
        std::println("FileIO - Emscripten http long polling with error begin");
        std::string          expectedString = createTestString();
        fileio::ReaderConfig config;
        config.chunkBytes  = 1000; // expected data is 190 bytes per request
        config.longPolling = true;

        auto readerExp = readAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/getNumbersAndError", config);
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        sub = std::move(readerExp.value());
            std::thread readerThread{[&]() {
                auto subResults = getReadResult(sub);
                expect(!sub.cancelRequested());
                expect(eq(subResults.dataCounter, 2uz));
                expect(eq(subResults.errorCounter, 1uz));

                for (std::size_t i = 0; i < subResults.allData.size(); i++) {
                    expect(eq(expectedString, bytesToString(subResults.allData[i])));
                }
            }};

            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        std::println("FileIO - Emscripten http long polling with error end");
    };

    "FileIO - Emscripten reader http with cancel"_test = [&] {
        std::println("Emscripten http with cancel");

        auto readerExp = readAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/getNumbersTimeout1s", fileio::ReaderConfig{});
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        sub = std::move(readerExp.value());
            std::thread readerThread{[&sub]() {
                auto subResults = getReadResult(sub);
                expect(sub.cancelRequested());
                expect(eq(subResults.dataCounter, 0uz));
                expect(eq(subResults.errorCounter, 0uz));
            }};
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            sub.cancel();

            if (readerThread.joinable()) {
                readerThread.join();
            }
        }
        std::println("Emscripten http with cancel end");
    };

    "FileIO - Emscripten writer http"_test = [&] {
        std::println("FileIO - Emscripten writer http POST begin");

        std::string               body = createTestString();
        std::vector<std::uint8_t> bytes(body.begin(), body.end());

        auto writerExp = writeAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/postNumbers", bytes, fileio::WriterConfig{});
        expect(writerExp.has_value());
        if (writerExp.has_value()) {
            auto        writer = std::move(writerExp.value());
            std::thread writerThread{[&writer]() {
                writer.wait();
                auto resExp = writer.result();
                expect(resExp.has_value());
                if (resExp.has_value()) {
                    auto res = std::move(resExp.value());
                    expect(eq(res.httpStatus, 200l));
                    expect(eq(res.httpResponseBody, std::string{"OK"}));
                }
            }};
            if (writerThread.joinable()) {
                writerThread.join();
            }
        }

        std::println("FileIO - Emscripten writer http POST end");
    };

    "FileIO - Emscripten writer http with error"_test = [&] {
        std::println("FileIO - Emscripten writer http POST - error begin");

        std::string               body = createTestString();
        std::vector<std::uint8_t> bytes(body.begin(), body.end());

        auto writerExp = writeAsyncEmscriptenHttpWorkerThread("http://127.0.0.1:8080/doesNotExists", bytes, fileio::WriterConfig{});
        expect(writerExp.has_value());
        if (writerExp.has_value()) {
            auto        writer = std::move(writerExp.value());
            std::thread writerThread{[&writer]() {
                writer.wait();
                auto resExp = writer.result();
                expect(!resExp.has_value());
            }};
            if (writerThread.joinable()) {
                writerThread.join();
            }
        }

        std::println("FileIO - Emscripten writer http POST - error end");
    };

    emscripten_run_script("stopServer();");
    if (serverThread.joinable()) {
        serverThread.join();
    }
};
#endif

const boost::ut::suite<"FileIO Memory Source tests"> fileIoMemorySourceTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::algorithm;

    "FileIO - Memory source"_test = [&] {
        std::println("FileIO - Memory source begin");
        std::string expectedString = createTestString();

        fileio::ReaderConfig config;
        config.chunkBytes = 11;

        std::vector<std::uint8_t> bytes(expectedString.begin(), expectedString.end());
        auto                      readerExp = fileio::readAsync(std::span<const std::uint8_t>(bytes.data(), bytes.size()), config, "<memory:test>");
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto sub        = std::move(readerExp.value());
            auto subResults = getReadResult(sub, std::numeric_limits<std::size_t>::max(), false);
            expect(!sub.cancelRequested());
            expect(eq(subResults.dataCounter, 18uz)); // 190 bytes / 11 bytes per chunk
            expect(eq(subResults.errorCounter, 0uz));
            expect(eq(expectedString, joinBytesToString(subResults.allData)));
        }
        std::println("FileIO - Memory source end");
    };
};

const boost::ut::suite<"FileIO error tests"> fileIoErrorTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::algorithm;

    "FileIO - Reader with wrong uri"_test = [&] {
        std::println("FileIO - Reader with wrong uri begin");

        auto readerExp = fileio::readAsync("wrong_sch://wrong_uri", fileio::ReaderConfig{});
        expect(!readerExp.has_value());
        std::println("FileIO - Reader with wrong uri end");
    };

    "FileIO - Writer with wrong uri"_test = [&] {
        std::println("FileIO - Writer with wrong begin");

        std::vector<std::uint8_t> data      = {0, 1, 2, 3, 4, 5};
        auto                      writerExp = fileio::writeAsync("wrong_sch://wrong_uri", data, fileio::WriterConfig{});
        expect(!writerExp.has_value());
        std::println("FileIO - Writer with wrong end");
    };

    "FileIO - poll maxSize too small"_test = [&] {
        using namespace boost::ut;
        using namespace gr;
        using namespace gr::algorithm;

        std::string expectedString = createTestString();

        fileio::ReaderConfig config;
        config.chunkBytes = 1000uz;

        std::vector<std::uint8_t> bytes(expectedString.begin(), expectedString.end());
        auto                      readerExp = fileio::readAsync(std::span<const std::uint8_t>(bytes.data(), bytes.size()), config, "<memory:maxSize>");
        expect(readerExp.has_value());
        if (readerExp.has_value()) {
            auto        sub          = std::move(readerExp.value());
            bool        finished     = false;
            std::size_t dataCounter  = 0;
            std::size_t errorCounter = 0;
            while (!finished) {
                const std::size_t maxSize = errorCounter < 5 ? 2uz : 1000uz;

                sub.poll(
                    [&finished, &dataCounter, &errorCounter](const auto& res) {
                        finished = res.isFinal;
                        if (res.data.has_value()) {
                            auto data = res.data.value();
                            if (!data.empty()) {
                                dataCounter++;
                            }
                        } else {
                            errorCounter++;
                            std::println("Error: {}", res.data.error().message);
                        }
                    },
                    maxSize, false);
            }
            expect(eq(dataCounter, 1uz));
            expect(ge(errorCounter, 5uz));
        }
    };
};

int main() {}
