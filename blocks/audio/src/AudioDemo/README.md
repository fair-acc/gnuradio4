# AudioDemo

Browser demo for audio playback (`WavSource → AudioSink`) and microphone loopback (`AudioSource → AudioSink`).

Build with Emscripten first (see the main [README](../../../../README.md)), then:

```bash
cd gnuradio4
python3 blocks/audio/src/AudioDemo/serve.py --dir <wasm-build-dir>/blocks/audio/src/AudioDemo
```

Open `http://localhost:8080/` in your browser.
