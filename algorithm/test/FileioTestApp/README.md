# FileioTestApp

Browser test app for `gr::algorithm::fileio`.

`gnuradio4` must be built with Emscripten first. See the main [README](../../../README.md).

After `FileioTestApp` is built, run this from the repository root:

```bash
python3 algorithm/test/FileioTestApp/serve.py --dir <wasm-build-dir>/algorithm/test/FileioTestApp
```

Then open in your browser: `http://localhost:8080/`
