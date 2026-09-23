#include <boost/ut.hpp>

#include <gnuradio-4.0/SharedLibrary.hpp>

#include <atomic>
#include <dlfcn.h>
#include <expected>
#include <filesystem>
#include <string>

namespace {

[[nodiscard]] std::filesystem::path libcPath() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&::malloc), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(info.dli_fname);
}

} // namespace

const boost::ut::suite<"SharedLibrary"> sharedLibraryTests = [] {
    using namespace boost::ut;

    "isPluginFileExtension matches platform conventions"_test = [] {
#if defined(_WIN32)
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("x.dll")));
        expect(!gr::detail::isPluginFileExtension(std::filesystem::path("x.so")));
#elif defined(__APPLE__)
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("x.dylib")));
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("x.so")));
        expect(!gr::detail::isPluginFileExtension(std::filesystem::path("x.dll")));
#elif defined(__EMSCRIPTEN__)
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("x.wasm")));
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("x.so")));
#else
        expect(gr::detail::isPluginFileExtension(std::filesystem::path("plugin.so")));
        expect(!gr::detail::isPluginFileExtension(std::filesystem::path("plugin.dylib")));
        expect(!gr::detail::isPluginFileExtension(std::filesystem::path("plugin.dll")));
        expect(!gr::detail::isPluginFileExtension(std::filesystem::path("plugin.txt")));
#endif
    };

    "default constructed library is unloaded"_test = [] {
        gr::SharedLibrary lib;
        expect(!lib.isLoaded());
        expect(lib.fileName().empty());
        expect(!lib.unload());
        expect(!lib.resolveAddress("malloc"));
        expect(!lib.resolve<void*(std::size_t)>("malloc"));
    };

    "load missing path fails"_test = [] {
        gr::SharedLibrary lib;
        const auto        result = lib.load("/definitely/does/not/exist/gr4_shared_library_missing.so");
        expect(!result);
        expect(!lib.isLoaded());
        expect(!lib.lastError().message.empty());
    };

#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    "load resolve unload libc"_test = [] {
        const auto path = libcPath();
        expect(!path.empty()) << "dladdr(malloc) should yield a library path";

        gr::SharedLibrary lib;
        expect(static_cast<bool>(lib.load(path)));
        expect(lib.isLoaded());
        expect(eq(lib.fileName(), path));

        auto addr = lib.resolveAddress("malloc");
        expect(static_cast<bool>(addr));
        expect(addr.value() != nullptr);

        auto mallocFn = lib.resolve<void*(std::size_t)>("malloc");
        expect(static_cast<bool>(mallocFn));
        expect(mallocFn.value() != nullptr);

        expect(!lib.resolveAddress(""));
        expect(!lib.resolveAddress("gr4_definitely_missing_symbol_xyz"));

        expect(static_cast<bool>(lib.unload()));
        expect(!lib.isLoaded());
        expect(!lib.resolveAddress("malloc"));
    };

    "loadAsync invokes done synchronously on native"_test = [] {
        const auto path = libcPath();
        expect(!path.empty());

        gr::SharedLibrary              lib;
        std::atomic<bool>              called{false};
        std::expected<void, gr::Error> asyncResult = std::unexpected(gr::Error{"unset"});
        lib.loadAsync(path, [&](std::expected<void, gr::Error> result) {
            asyncResult = std::move(result);
            called.store(true);
        });
        expect(called.load());
        expect(static_cast<bool>(asyncResult));
        expect(lib.isLoaded());

        lib.loadAsync(path, {}); // null done is a no-op
        expect(lib.isLoaded());
    };

    "move transfers ownership"_test = [] {
        const auto path = libcPath();
        expect(!path.empty());

        gr::SharedLibrary first;
        expect(static_cast<bool>(first.load(path)));

        gr::SharedLibrary second(std::move(first));
        expect(!first.isLoaded());
        expect(second.isLoaded());

        gr::SharedLibrary third;
        third = std::move(second);
        expect(!second.isLoaded());
        expect(third.isLoaded());
        expect(static_cast<bool>(third.unload()));
    };

    "reload replaces previous handle"_test = [] {
        const auto path = libcPath();
        expect(!path.empty());

        gr::SharedLibrary lib;
        expect(static_cast<bool>(lib.load(path)));
        expect(static_cast<bool>(lib.load(path)));
        expect(lib.isLoaded());
        expect(static_cast<bool>(lib.unload()));
    };
#endif
};

int main() { return 0; }
