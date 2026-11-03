// Emscripten async dynamic-library open for gr::SharedLibrary.
//
// Compiled only into gnuradio-core-dynload (never into gnuradio-core alone).
// Provides a *strong* definition of gr::detail::s_emscriptenAsyncOpen that
// overrides the weak null definition in SharedLibrary.cpp. Because this object
// comes from an OBJECT library (always linked unconditionally into consumers of
// gnuradio-core-dynload), the strong definition is present when that target is
// used — no constructor or registration function required.

#include <dlfcn.h>
#include <emscripten/emscripten.h>

#include <gnuradio-4.0/SharedLibrary.hpp>

namespace gr::detail {

namespace {

void emscriptenAsyncOpenImpl(const char* path, void* userData, //
    void (*onSuccess)(void* userData, void* handle),           //
    void (*onError)(void* userData)) {
#ifdef __APPLE__
    constexpr int kFlags = RTLD_NOW | RTLD_GLOBAL;
#else
    constexpr int kFlags = RTLD_NOW | RTLD_LOCAL;
#endif
    emscripten_dlopen(path, kFlags, userData, onSuccess, onError);
}

} // namespace

// Strong definition: overrides the weak nullptr in SharedLibrary.cpp.
// Referencing emscriptenAsyncOpenImpl keeps it alive through wasm-ld GC.
EmscriptenAsyncOpenFn s_emscriptenAsyncOpen = &emscriptenAsyncOpenImpl;

} // namespace gr::detail
