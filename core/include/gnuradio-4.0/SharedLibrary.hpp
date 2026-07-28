#ifndef GNURADIO_SHARED_LIBRARY_HPP
#define GNURADIO_SHARED_LIBRARY_HPP

#include <bit>
#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/Logger.hpp>

namespace gr {

/**
 * @brief Cross-platform runtime shared-library loader (.so / .dylib / .dll / WASM side module).
 *
 * Loading a dynamic library executes native code from the given path with the privileges of the
 * process. Only load libraries from paths you fully trust; path policy is the caller's
 * responsibility.
 *
 * On Emscripten, dynamic open is asynchronous (`emscripten_dlopen`). Use loadAsync(). The
 * synchronous load() path is native-only and returns an error on Emscripten directing the caller
 * to loadAsync(). Enabling the Emscripten open implementation requires linking
 * gnuradio4::gnuradio-core-dynload (propagates MAIN_MODULE=2 and the async-open object file).
 */
class SharedLibrary {
public:
    SharedLibrary();
    ~SharedLibrary();

    SharedLibrary(const SharedLibrary&)            = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;
    SharedLibrary(SharedLibrary&&) noexcept;
    SharedLibrary& operator=(SharedLibrary&&) noexcept;

    /// Opens a dynamic library. Completes synchronously on native platforms.
    /// On Emscripten always fails — use loadAsync().
    [[nodiscard]] std::expected<void, Error> load(const std::filesystem::path& filePath);

    /// Opens a dynamic library and invokes done when finished.
    /// On native platforms done runs on the calling thread before loadAsync returns.
    /// On Emscripten done runs when the side module has been fetched and linked (or on error).
    void loadAsync(const std::filesystem::path& filePath, std::function<void(std::expected<void, Error>)> done);

    [[nodiscard]] std::expected<void, Error> unload();

    [[nodiscard]] bool isLoaded() const noexcept;

    [[nodiscard]] const std::filesystem::path& fileName() const noexcept;

    [[nodiscard]] Error lastError() const;

    [[nodiscard]] std::expected<void*, Error> resolveAddress(std::string_view symbol);

    template<typename Function>
    [[nodiscard]] std::expected<Function*, Error> resolve(std::string_view symbol) {
        static_assert(std::is_function_v<Function>, "SharedLibrary::resolve<Function>() requires a function type");
        static_assert(sizeof(void*) == sizeof(Function*), "function pointer size differs from void* on this platform");

        auto addr = resolveAddress(symbol);
        if (!addr) {
            return std::unexpected(addr.error());
        }
        return std::bit_cast<Function*>(*addr);
    }

private:
    void*                 _handle = nullptr;
    std::filesystem::path _fileName;
    Error                 _lastError;
};

namespace detail {

/// Platform plugin-file extension filter used by PluginLoader discovery.
[[nodiscard]] inline bool isPluginFileExtension(const std::filesystem::path& path) {
#if defined(_WIN32)
    return path.extension() == ".dll";
#elif defined(__EMSCRIPTEN__)
    const auto ext = path.extension();
    return ext == ".wasm" || ext == ".so";
#elif defined(__APPLE__)
    const auto ext = path.extension();
    return ext == ".dylib" || ext == ".so";
#else
    return path.extension() == ".so";
#endif
}

#if defined(__EMSCRIPTEN__)
/// Signature of the opt-in emscripten_dlopen wrapper (strong definition lives in
/// SharedLibraryEmscriptenDlopen.cpp, linked only via gnuradio-core-dynload).
using EmscriptenAsyncOpenFn = void (*)(const char* path, void* userData, //
    void (*onSuccess)(void* userData, void* handle),                     //
    void (*onError)(void* userData));

/// Weak null fallback in SharedLibrary.cpp; overridden when dynload object is linked.
extern EmscriptenAsyncOpenFn s_emscriptenAsyncOpen;
#endif

} // namespace detail

} // namespace gr

#endif // GNURADIO_SHARED_LIBRARY_HPP
