#include <gnuradio-4.0/SharedLibrary.hpp>

#include <memory>
#include <string>
#include <utility>

#if defined(__wasi__)
// WASI does not support dynamic loading.
#elif defined(__EMSCRIPTEN__)
// emscripten_dlopen is NOT included here — that would force every Emscripten
// executable linking gnuradio-core to be built as a MAIN_MODULE.
// The async-open implementation lives in SharedLibraryEmscriptenDlopen.cpp,
// compiled only into gnuradio-core-dynload. A weak function pointer is the
// null fallback when that object file is not linked.
#include <dlfcn.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace gr {

namespace {

#if defined(__wasi__)

std::expected<void*, Error> platformOpen(const std::filesystem::path& /*filePath*/) { //
    return std::unexpected(Error{"Dynamic library loading is not supported on this platform."});
}

std::expected<void, Error> platformClose(void* /*handle*/) { //
    return std::unexpected(Error{"Dynamic library loading is not supported on this platform."});
}

std::expected<void*, Error> platformResolve(void* /*handle*/, const char* /*symbol*/) { //
    return std::unexpected(Error{"Dynamic library loading is not supported on this platform."});
}

#elif defined(__EMSCRIPTEN__)

std::expected<void, Error> platformClose(void* handle) {
    if (dlclose(handle) != 0) {
        const char* msg = dlerror();
        return std::unexpected(Error{msg ? msg : "Unknown dlclose error."});
    }
    return {};
}

std::expected<void*, Error> platformResolve(void* handle, const char* symbol) {
    dlerror();
    void*       addr = dlsym(handle, symbol);
    const char* msg  = dlerror();
    if (msg) {
        return std::unexpected(Error{msg});
    }
    return addr;
}

#elif defined(_WIN32)

std::string win32ErrorString(DWORD code) {
    LPSTR buf = nullptr;
    DWORD len = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, //
        nullptr, code, 0, reinterpret_cast<LPSTR>(&buf), 0, nullptr);
    if (buf) {
        while (len > 0 && (buf[len - 1] == '\r' || buf[len - 1] == '\n')) {
            --len;
        }
        std::string msg(buf, len);
        LocalFree(buf);
        return msg;
    }
    return "Unknown error (code " + std::to_string(code) + ")";
}

std::expected<void*, Error> platformOpen(const std::filesystem::path& filePath) {
    const std::wstring wpath  = filePath.wstring();
    HMODULE            module = LoadLibraryW(wpath.c_str());
    if (!module) {
        const DWORD code = GetLastError();
        return std::unexpected(Error{win32ErrorString(code)});
    }
    return static_cast<void*>(module);
}

std::expected<void, Error> platformClose(void* handle) {
    if (!FreeLibrary(static_cast<HMODULE>(handle))) {
        const DWORD code = GetLastError();
        return std::unexpected(Error{win32ErrorString(code)});
    }
    return {};
}

std::expected<void*, Error> platformResolve(void* handle, const char* symbol) {
    FARPROC addr = GetProcAddress(static_cast<HMODULE>(handle), symbol);
    if (!addr) {
        const DWORD code = GetLastError();
        return std::unexpected(Error{"Symbol '" + std::string(symbol) + "' not found: " + win32ErrorString(code)});
    }
    return reinterpret_cast<void*>(addr);
}

#else

std::expected<void*, Error> platformOpen(const std::filesystem::path& filePath) {
    // RTLD_LOCAL keeps plugin symbols isolated but breaks RTTI/dynamic_cast across dylib boundaries.
    // On macOS (Mach-O two-level namespace), RTLD_LOCAL also risks duplicating singletons such as
    // globalBlockRegistry(); use RTLD_GLOBAL there to match Linux ELF flat-namespace behaviour.
#ifdef __APPLE__
    constexpr int kFlags = RTLD_NOW | RTLD_GLOBAL;
#else
    constexpr int kFlags = RTLD_NOW | RTLD_LOCAL;
#endif
    void* handle = dlopen(filePath.c_str(), kFlags);
    if (!handle) {
        const char* msg = dlerror();
        return std::unexpected(Error{msg ? msg : "Unknown dlopen error."});
    }
    return handle;
}

std::expected<void, Error> platformClose(void* handle) {
    if (dlclose(handle) != 0) {
        const char* msg = dlerror();
        return std::unexpected(Error{msg ? msg : "Unknown dlclose error."});
    }
    return {};
}

std::expected<void*, Error> platformResolve(void* handle, const char* symbol) {
    dlerror();
    void*       addr = dlsym(handle, symbol);
    const char* msg  = dlerror();
    if (msg) {
        return std::unexpected(Error{msg});
    }
    return addr;
}

#endif

} // namespace

#if defined(__EMSCRIPTEN__)
namespace detail {
// Weak definition: null when SharedLibraryEmscriptenDlopen.cpp is not linked.
// That TU (compiled into gnuradio-core-dynload) overrides with a strong definition.
__attribute__((weak)) EmscriptenAsyncOpenFn s_emscriptenAsyncOpen = nullptr;
} // namespace detail
#endif

SharedLibrary::SharedLibrary() = default;

SharedLibrary::~SharedLibrary() {
    if (_handle) {
        std::ignore = unload();
    }
}

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept : _handle(std::exchange(other._handle, nullptr)), _fileName(std::move(other._fileName)), _lastError(std::move(other._lastError)) {}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (_handle) {
        std::ignore = unload();
    }
    _handle    = std::exchange(other._handle, nullptr);
    _fileName  = std::move(other._fileName);
    _lastError = std::move(other._lastError);
    return *this;
}

std::expected<void, Error> SharedLibrary::load(const std::filesystem::path& filePath) {
#if defined(__EMSCRIPTEN__)
    _fileName  = filePath;
    _lastError = Error{"SharedLibrary::load() is not available on Emscripten; use loadAsync() and link gnuradio4::gnuradio-core-dynload."};
    return std::unexpected(_lastError);
#else
    if (_handle) {
        if (auto r = unload(); !r) {
            return r;
        }
    }

    _fileName = filePath;
    auto open = platformOpen(filePath);
    if (!open) {
        _handle    = nullptr;
        _lastError = open.error();
        return std::unexpected(_lastError);
    }
    _handle = *open;
    return {};
#endif
}

void SharedLibrary::loadAsync(const std::filesystem::path& filePath, std::function<void(std::expected<void, Error>)> done) {
    if (!done) {
        return;
    }

#if defined(__EMSCRIPTEN__)
    if (_handle) {
        if (auto r = unload(); !r) {
            done(std::unexpected(r.error()));
            return;
        }
    }

    _fileName = filePath;

    if (!detail::s_emscriptenAsyncOpen) {
        _lastError = Error{"emscripten_dlopen is not available. Link gnuradio4::gnuradio-core-dynload to enable runtime plugin loading."};
        done(std::unexpected(_lastError));
        return;
    }

    struct Request {
        SharedLibrary*                                  self;
        std::function<void(std::expected<void, Error>)> done;
    };

    auto* req = new Request{this, std::move(done)};

    detail::s_emscriptenAsyncOpen(
        filePath.c_str(), req, //
        [](void* userData, void* handle) {
            std::unique_ptr<Request> r(static_cast<Request*>(userData));
            r->self->_handle = handle;
            r->done({});
        },
        [](void* userData) {
            std::unique_ptr<Request> r(static_cast<Request*>(userData));
            const char*              msg = dlerror();
            r->self->_handle             = nullptr;
            r->self->_lastError          = Error{msg ? msg : "Unknown emscripten_dlopen error."};
            r->done(std::unexpected(r->self->_lastError));
        });
#else
    done(load(filePath));
#endif
}

std::expected<void, Error> SharedLibrary::unload() {
    if (!_handle) {
        _lastError = Error{"No library is currently loaded."};
        return std::unexpected(_lastError);
    }

    auto close = platformClose(_handle);
    if (!close) {
        _lastError = close.error();
        return std::unexpected(_lastError);
    }

    _handle = nullptr;
    return {};
}

bool SharedLibrary::isLoaded() const noexcept { return _handle != nullptr; }

const std::filesystem::path& SharedLibrary::fileName() const noexcept { return _fileName; }

Error SharedLibrary::lastError() const { return _lastError; }

std::expected<void*, Error> SharedLibrary::resolveAddress(std::string_view symbol) {
    if (!_handle) {
        _lastError = Error{"No library is currently loaded."};
        return std::unexpected(_lastError);
    }
    if (symbol.empty()) {
        _lastError = Error{"Symbol name must not be empty."};
        return std::unexpected(_lastError);
    }

    const std::string localSymbol(symbol);
    auto              addr = platformResolve(_handle, localSymbol.c_str());
    if (!addr) {
        _lastError = addr.error();
        return std::unexpected(_lastError);
    }
    return *addr;
}

} // namespace gr
