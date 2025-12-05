## 1. Overview

**Header & namespace**

```cpp
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>
using namespace gr::algorithm::fileio;
```

**Main components**

- `Reader` – async read from file / HTTP / memory / dialog.
- `Writer` – async write to file / HTTP / browser download.
- `ReaderConfig` / `WriterConfig` – configuration structs.
- Convenience: `Reader::get()` and `write()` for synchronous use.

All heavy I/O runs on an internal thread pool (except in-memory readers).

---

## 2. URI schemes

| Scheme           | Meaning                               | Notes                                                                                          |
| ---------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `file:/path`     | Local file (native FS or MEMFS)       | Append mode only for `file:/`                                                                  |
| `http://...`     | HTTP GET / POST                       | Native: CPR; Emscripten: `emscripten_fetch`                                                    |
| `https://...`    | HTTPS GET / POST                      | `tlsVerifyPeer` controls SSL verify                                                            |
| `dialog:/open`   | File-open dialog -> memory or file    | Needs dialog callback registered                                                               |
| `<memory>`       | In-memory span (no URI, just a label) | No background work after `readAsync`                                                           |
| `download:/name` | Browser download (Writer only)        | Emscripten: browser “save file”; Native: writes to default Downloads (e.g. `~/Downloads/name`) |

If the URI scheme is unknown, `readAsync` / `writeAsync` return `std::unexpected<gr::Error>`.

---

## 3. Reader

### 3.1 `ReaderConfig`

```cpp
struct ReaderConfig {
    std::size_t                        chunkBytes       = std::numeric_limits<std::size_t>::max(); // max chunk size
    std::optional<std::size_t>         offset           = std::nullopt;                            // start at byte N
    std::uint64_t                      httpTimeoutNanos = 30'000'000'000;                          // 30 s HTTP timeout
    std::size_t                        bufferMinSize    = 1024;                                    // ring buffer slots
    bool                               longPolling      = false;                                   // HTTP long-poll loop
    std::map<std::string, std::string> httpHeaders      = {};                                      // extra headers
    bool                               tlsVerifyPeer    = true;                                    // native HTTPS
};
```

> **Important!** When you call `Reader::poll(maxSize = ...)`, make sure `maxSize >= chunkBytes` or you’ll get error results.

---

### 3.2 Creating Readers

```cpp
ReaderConfig cfg;
cfg.chunkBytes = 4096;

// 1) File
auto fileReaderExp = readAsync("file:/tmp/data.bin", cfg);

// 2) HTTP (native: requires GR_HTTP_ENABLED=1)
cfg.httpHeaders.emplace("Accept", "text/plain");
auto httpReaderExp = readAsync("http://localhost:8080/getNumbers", cfg);

// 3) In-memory (no background work after readAsync)
std::string payload = "Hello";
std::vector<std::uint8_t> bytes(payload.begin(), payload.end());
auto memReaderExp = readAsync(bytes, cfg, "<memory:hello>");

if (!fileReaderExp || !httpReaderExp || !memReaderExp) {
    // handle gr::Error via .error().message
}
```

---

### 3.3 Polling (`poll` + `wait`)

```cpp
Reader reader = std::move(fileReaderExp.value());
bool finished = false;

while (!finished) {
    reader.poll(
        [&](const Reader::PollResult& res) {
            finished = res.isFinal;
            if (res.data.has_value()) {
                auto chunk = res.data.value();   // span valid ONLY in this callback
                // process chunk here (copy if you need it later)
            } else {
                auto& err = res.data.error();    // HTTP 4xx/5xx, I/O error, etc.
                std::println("FileIO error: {}", err.message);
            }
        },
        /*maxSize=*/4096,   // must be >= cfg.chunkBytes
        /*doWait=*/true     // block until at least one message is available
    );
}
```

**Notes**

- `Reader::wait()` blocks until at least one message is available or the reader is canceled.
  Internally it uses `std::atomic::wait`, so it does not busy-loop and is light on CPU.
- Under Emscripten, `wait()` on the **main runtime thread** returns immediately with a warning (no blocking allowed there).

---

### 3.4 Synchronous read with `get()`

```cpp
Reader reader = std::move(httpReaderExp.value());

std::thread worker([&] {
    auto dataExp = reader.get();  // internally uses poll(doWait = true) until Final
    if (!dataExp) {
        std::println("HTTP read failed: {}", dataExp.error().message);
        return;
    }
    auto& data = dataExp.value(); // full concatenated payload
    std::string body(data.begin(), data.end());
    std::println("Body: {}", body);
});
worker.join();
```

> Run `get()` on a background thread.

---

### 3.5 Cancel

```cpp
Reader reader = std::move(readerExp.value()); // file:// or http(s)://

std::thread worker([&] {
    auto dataExp = reader.get();   // or loop with reader.poll(...)
    // handle dataExp (may fail or be partial if cancelled)
});

// later, from another thread:
reader.cancel();    // soft cancel; underlying I/O stops as soon as possible
worker.join();
```

Cancellation is best-effort for HTTP (especially under Emscripten).

---

## 4. Writer

### 4.1 `WriterConfig` + `WriteResult`

```cpp
enum class WriteMode { overwrite, append }; // append only for file:/ URIs

struct WriterConfig {
    WriteMode                          mode             = WriteMode::overwrite;
    std::uint64_t                      httpTimeoutNanos = 30'000'000'000; // 30 s HTTP timeout
    std::map<std::string, std::string> httpHeaders      = {};             // e.g. Content-Type
    bool                               tlsVerifyPeer    = true;           // native HTTPS
};

struct WriteResult {
    std::string httpResponseBody; // HTTP response body; empty for file writes
    long        httpStatus{0};    // HTTP status code; 0 for file writes
};
```

---

### 4.2 `writeAsync` + `Writer` usage

```cpp
std::string payload = "Hello";
std::vector<std::uint8_t> bytes(payload.begin(), payload.end());

// File overwrite
WriterConfig fileCfg{.mode = WriteMode::overwrite};
auto fileWriterExp = writeAsync("file:/tmp/out.bin", bytes, fileCfg);

// HTTP POST
WriterConfig httpCfg{};
httpCfg.httpHeaders.emplace("Content-Type", "text/plain");
auto httpWriterExp = writeAsync("http://localhost:8080/postNumbers", bytes, httpCfg);

if (!fileWriterExp || !httpWriterExp) {
    // e.g. HTTP disabled, bad URI, append to HTTP, etc.
}

Writer httpWriter = std::move(httpWriterExp.value());

std::thread t([&] {
    httpWriter.wait();                 // blocks until done (not allowed on WASM main thread)
    auto resExp = httpWriter.result(); // non-blocking; error if called before completion
    if (!resExp) {
        std::println("POST failed: {}", resExp.error().message);
        return;
    }
    auto res = resExp.value();
    std::println("Status: {}, body={}", res.httpStatus, res.httpResponseBody);
});
t.join();
```

---

### 4.3 Synchronous `write()`

```cpp
auto resExp = write("file:/tmp/out.bin", bytes, WriterConfig{}); // blocks until done

if (!resExp) {
    std::println("Write failed: {}", resExp.error().message);
} else {
    // For files: resExp->httpStatus == 0, no response body
}
```

> On Emscripten main runtime thread, `write()` returns an error instead of blocking.

---

### 4.4 Download (`download:/`)

On the web, browsers don’t let you write arbitrary paths on the user’s disk. You usually either

- use `file:/...` -> write into MEMFS (in-memory file system), or
- use `download:/name` -> hand the file to the user.

`download:/name` is a small cross-platform convenience:

- **Emscripten**: triggers a browser "Save file" download.
- **Native**: writes into the user’s default download folder (e.g. `~/Downloads/name`).

---

## 5. Emscripten caveats

- `file:/` uses MEMFS (in-memory file system).

- HTTP uses `emscripten_fetch`; `ReaderConfig::longPolling` re-issues requests in a loop.

- Blocking calls on **main runtime thread**:

  - `Reader::wait()`, `Writer::wait()`, `write()` -> refuse to block, return early / error.

- `download:/` is implemented via JS in the browser.

- Cancellation for HTTP is **best-effort**: the fetch cannot always be aborted immediately, but you will always get a Final message.
