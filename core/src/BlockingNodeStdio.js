// `--pre-js` of every Emscripten executable: under node, output -- a pthread's too, which Emscripten proxies there -- is
// written on the main thread with fs.writeSync, to stdio pipes node opened non-blocking, so a full pipe (a slow reader
// such as ctest) fails the write with EAGAIN, and the C++ that printed aborts
if (typeof process === "object" && typeof process.versions?.node === "string") {
    process.stdout._handle?.setBlocking?.(true);
    process.stderr._handle?.setBlocking?.(true);
}
