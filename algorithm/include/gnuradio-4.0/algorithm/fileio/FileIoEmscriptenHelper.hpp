#ifndef FILEIOEMSCRIPTENHELPER_HPP
#define FILEIOEMSCRIPTENHELPER_HPP

#include <gnuradio-4.0/algorithm/fileio/FileIoHelpers.hpp>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/emscripten.h>
#include <emscripten/fetch.h>
#include <emscripten/html5.h>
#include <emscripten/threading.h>
#endif

#if defined(__EMSCRIPTEN__)
#define FILEIO_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define FILEIO_EXPORT
#endif

#ifdef __cplusplus
#define FILEIO_EXTERN_C extern "C"
#else
#define FILEIO_EXTERN_C
#endif

namespace gr::algorithm::fileio {
constexpr bool isWebAssembly() noexcept {
#if defined(__EMSCRIPTEN__)
    return true;
#else
    return false;
#endif
}

inline bool isMainThread() {
#if defined(__EMSCRIPTEN__)
    return emscripten_is_main_runtime_thread();
#else
    return true; // Native: assume single-threaded or main thread
#endif
}

inline bool isTabVisible() {
#if defined(__EMSCRIPTEN__)
    EmscriptenVisibilityChangeEvent status;
    if (emscripten_get_visibility_status(&status) == EMSCRIPTEN_RESULT_SUCCESS) {
        return !status.hidden;
    }
#endif
    return true;
}

// clang-format off
inline void listPersistentFiles([[maybe_unused]] bool recursive = true) {
#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM_({
        function listDir(path, recursive, indent = "") {
            try {
                const entries = FS.readdir(path);
                for (let entry of entries) {
                    if (entry === '.' || entry === '..') continue;
                    const fullPath = path + (path.endsWith('/') ? "" : "/") + entry;
                    const stat = FS.stat(fullPath);
                    if (FS.isDir(stat.mode)) {
                        console.log(indent + '[Dir] ' + fullPath);
                        if (recursive) {
                            listDir(fullPath, recursive, indent + '  ');
                        }
                    } else {
                        console.log(indent + '[File] ' + fullPath);
                    }
                }
            } catch (e) {
                console.error('Error listing directory:', path, e);
            }
        }
        listDir('/', $0 !== 0);
    }, recursive ? 1 : 0);
#pragma GCC diagnostic pop
#else

#endif
}
// clang-format on

#if defined(__EMSCRIPTEN__)
namespace detail {
// clang-format off
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdollar-in-identifier-extension"
inline void startFileDialogEmscripten(DialogOpenHandle* handle) {
    EM_ASM(
        {
            const handlePtr = $0 >>> 0;

            const input = document.createElement('input');
            input.type  = 'file';
            input.style.display = 'none';
            document.body.appendChild(input);

            const cleanup = () => {
                if (input.parentNode) {
                    input.parentNode.removeChild(input);
                }
            };


            input.addEventListener('cancel', (event) => {
                Module.ccall('fileio_dialog_on_cancel',null,['number'],[handlePtr]);
            }, { once: true });

            input.addEventListener('change', async (e) => {
                const files = e.target.files;
                if (!files || files.length === 0) {
                    cleanup();
                    return;
                }

                try {
                    const f   = files[0];
                    const buf = await f.arrayBuffer();
                    const u8  = new Uint8Array(buf);
                    const ptr = Module._malloc(u8.byteLength);
                    Module.HEAPU8.set(u8, ptr);
                    Module.ccall('fileio_dialog_on_bytes',null,['number', 'number', 'number'],[handlePtr, ptr, u8.byteLength]);
                    Module._free(ptr);
                } catch (err) {
                    console.error('[fileio] dialog:/open – error reading file:', err);
                    Module.ccall('fileio_dialog_on_error',null,['number'],[handlePtr]);
                } finally {
                    cleanup();
                }
            }, { once: true });

            input.click();
        },
        handle);
}
#pragma GCC diagnostic pop
// clang-format on
} // namespace detail

inline void setupEmscriptenDialogCallback() {
    setDialogOpenCallback([](DialogOpenHandle& handle) {
        auto* handlePtr = &handle;
        if (isMainThread()) {
            detail::startFileDialogEmscripten(handlePtr);
        } else {
            emscripten_async_run_in_main_runtime_thread(
                EM_FUNC_SIG_VI,
                +[](void* p) {
                    auto* hPtr = static_cast<DialogOpenHandle*>(p);
                    detail::startFileDialogEmscripten(hPtr);
                },
                handlePtr);
        }
    });
}
#endif

#if defined(__EMSCRIPTEN__)
FILEIO_EXTERN_C {

    FILEIO_EXPORT
    void fileio_dialog_on_bytes(std::uintptr_t handleToken, const std::uint8_t* data, int len) {
        auto* handle = reinterpret_cast<DialogOpenHandle*>(handleToken);
        if (handle == nullptr) {
            return;
        }

        if (!data || len <= 0) {
            if (handle->fail) {
                handle->fail("Empty or invalid buffer");
            }
            return;
        }

        if (handle->completeWithMemory) {
            handle->completeWithMemory(std::span<const std::uint8_t>{data, static_cast<std::size_t>(len)});
        }
    }

    FILEIO_EXPORT
    void fileio_dialog_on_cancel(std::uintptr_t handleToken) {
        auto* handle = reinterpret_cast<DialogOpenHandle*>(handleToken);
        if (handle != nullptr && handle->fail) {
            handle->fail("User cancelled");
        }
    }

    FILEIO_EXPORT
    void fileio_dialog_on_error(std::uintptr_t handleToken) {
        auto* handle = reinterpret_cast<DialogOpenHandle*>(handleToken);
        if (handle != nullptr && handle->fail) {
            handle->fail("JS error while reading file");
        }
    }

} // FILEIO_EXTERN_C
#endif

} // namespace gr::algorithm::fileio

#endif // FILEIOEMSCRIPTENHELPER_HPP
