# AudioSinkTestApp

Browser test app for `gr::audio::WavSource -> gr::audio::AudioSink`.

`gnuradio4` must be built with Emscripten first. See the main [README](../../../../README.md).

After `AudioSinkTestApp` is built, run this commands:

```bash
cd gnuradio4
python3 blocks/audio/test/AudioSinkTestApp/serve.py --dir <wasm-build-dir>/blocks/audio/test/AudioSinkTestApp
```

Then open in your browser: `http://localhost:8080/`
