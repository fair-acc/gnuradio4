#ifndef GNURADIO_GL_COMPUTE_CONTEXT_HPP
#define GNURADIO_GL_COMPUTE_CONTEXT_HPP

#include <cstddef>
#include <cstring>
#include <expected>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if __has_include(<EGL/egl.h>) && __has_include(<GL/gl.h>)
#define GR_DEVICE_HAS_GL_COMPUTE 1
#include <EGL/egl.h>

#include <GL/gl.h>
#include <GL/glext.h>

#else
#define GR_DEVICE_HAS_GL_COMPUTE 0
#endif

namespace gr::device {

inline constexpr bool kHasGlCompute = GR_DEVICE_HAS_GL_COMPUTE;

struct ShaderDispatchInfo {
    std::string source;
    std::size_t workgroupSize = 256;
};

#if GR_DEVICE_HAS_GL_COMPUTE

/**
 * @brief Headless OpenGL 4.3 compute shader context via EGL.
 *
 * Creates an EGL context without a display server (works in CI, Docker, SSH).
 * Uses Mesa llvmpipe for software rendering when no GPU is available, or
 * delegates to hardware (NVIDIA, AMD, Intel) when present.
 *
 * Usage:
 * @code
 * gr::device::GlComputeContext gl;
 * auto id = gl.compileOrGetCached(glslSource);
 * gl.dispatch(id, inputData, outputData, N, 256);
 * @endcode
 */
struct GlComputeContext {
    EGLDisplay _display = EGL_NO_DISPLAY;
    EGLContext _context = EGL_NO_CONTEXT;

    // GL function pointers
    PFNGLCREATESHADERPROC       _glCreateShader       = nullptr;
    PFNGLSHADERSOURCEPROC       _glShaderSource       = nullptr;
    PFNGLCOMPILESHADERPROC      _glCompileShader      = nullptr;
    PFNGLGETSHADERIVPROC        _glGetShaderiv        = nullptr;
    PFNGLGETSHADERINFOLOGPROC   _glGetShaderInfoLog   = nullptr;
    PFNGLCREATEPROGRAMPROC      _glCreateProgram      = nullptr;
    PFNGLATTACHSHADERPROC       _glAttachShader       = nullptr;
    PFNGLLINKPROGRAMPROC        _glLinkProgram        = nullptr;
    PFNGLUSEPROGRAMPROC         _glUseProgram         = nullptr;
    PFNGLDELETESHADERPROC       _glDeleteShader       = nullptr;
    PFNGLDELETEPROGRAMPROC      _glDeleteProgram      = nullptr;
    PFNGLDISPATCHCOMPUTEPROC    _glDispatchCompute    = nullptr;
    PFNGLMEMORYBARRIERPROC      _glMemoryBarrier      = nullptr;
    PFNGLGENBUFFERSPROC         _glGenBuffers         = nullptr;
    PFNGLDELETEBUFFERSPROC      _glDeleteBuffers      = nullptr;
    PFNGLBINDBUFFERBASEPROC     _glBindBufferBase     = nullptr;
    PFNGLBUFFERDATAPROC         _glBufferData         = nullptr;
    PFNGLBINDBUFFERPROC         _glBindBuffer         = nullptr;
    PFNGLMAPBUFFERRANGEPROC     _glMapBufferRange     = nullptr;
    PFNGLUNMAPBUFFERPROC        _glUnmapBuffer        = nullptr;
    PFNGLGETPROGRAMIVPROC       _glGetProgramiv       = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC  _glGetProgramInfoLog  = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC _glGetUniformLocation = nullptr;
    PFNGLUNIFORM1UIPROC         _glUniform1ui         = nullptr;
    PFNGLCOPYBUFFERSUBDATAPROC  _glCopyBufferSubData  = nullptr;

    std::unordered_map<std::size_t, unsigned int> _shaderCache; // source hash → GL program
    std::mutex                                    _cacheMtx;
    bool                                          _initialised = false;

    GlComputeContext() { init(); }

    ~GlComputeContext() { shutdown(); }

    GlComputeContext(const GlComputeContext&)            = delete;
    GlComputeContext& operator=(const GlComputeContext&) = delete;

    [[nodiscard]] bool        isAvailable() const noexcept { return _initialised; }
    [[nodiscard]] std::string rendererName() const { return _initialised ? std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) : "n/a"; }
    [[nodiscard]] std::string vendorName() const { return _initialised ? std::string(reinterpret_cast<const char*>(glGetString(GL_VENDOR))) : "n/a"; }
    [[nodiscard]] std::string versionString() const { return _initialised ? std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) : "n/a"; }

    void init() {
        _display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (_display == EGL_NO_DISPLAY) {
            return;
        }

        EGLint major, minor;
        if (!eglInitialize(_display, &major, &minor)) {
            return;
        }

        // request OpenGL (not ES) with surfaceless config
        if (!eglBindAPI(EGL_OPENGL_API)) {
            return;
        }

        EGLint    configAttribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
        EGLConfig config;
        EGLint    numConfigs;
        if (!eglChooseConfig(_display, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
            return;
        }

        EGLint ctxAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
        _context            = eglCreateContext(_display, config, EGL_NO_CONTEXT, ctxAttribs);
        if (_context == EGL_NO_CONTEXT) {
            return;
        }

        if (!eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, _context)) {
            return;
        }

        loadFunctions();
        _initialised = (_glCreateShader != nullptr && _glDispatchCompute != nullptr);
    }

    void shutdown() {
        if (_display != EGL_NO_DISPLAY) {
            for (auto& [_, prog] : _shaderCache) {
                _glDeleteProgram(prog);
            }
            _shaderCache.clear();
            eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (_context != EGL_NO_CONTEXT) {
                eglDestroyContext(_display, _context);
            }
            eglTerminate(_display);
        }
        _display     = EGL_NO_DISPLAY;
        _context     = EGL_NO_CONTEXT;
        _initialised = false;
    }

    [[nodiscard]] std::expected<unsigned int, std::string> compileOrGetCached(std::string_view glslSource) {
        auto             hash = std::hash<std::string_view>{}(glslSource);
        std::scoped_lock lk(_cacheMtx);
        auto             it = _shaderCache.find(hash);
        if (it != _shaderCache.end()) {
            return it->second;
        }

        auto shader = _glCreateShader(GL_COMPUTE_SHADER);
        auto src    = glslSource.data();
        auto len    = static_cast<int>(glslSource.size());
        _glShaderSource(shader, 1, &src, &len);
        _glCompileShader(shader);

        int success = 0;
        _glGetShaderiv(shader, 0x8B81 /*GL_COMPILE_STATUS*/, &success);
        if (!success) {
            char log[512];
            _glGetShaderInfoLog(shader, 512, nullptr, log);
            _glDeleteShader(shader);
            return std::unexpected(std::string("GLSL compile error: ") + log);
        }

        auto prog = _glCreateProgram();
        _glAttachShader(prog, shader);
        _glLinkProgram(prog);
        _glDeleteShader(shader);

        _glGetProgramiv(prog, 0x8B82 /*GL_LINK_STATUS*/, &success);
        if (!success) {
            char log[512];
            _glGetProgramInfoLog(prog, 512, nullptr, log);
            _glDeleteProgram(prog);
            return std::unexpected(std::string("GLSL link error: ") + log);
        }

        _shaderCache[hash] = prog;
        return prog;
    }

    template<typename T>
    void dispatch(unsigned int program, const T* hostInput, T* hostOutput, std::size_t count, std::size_t workgroupSize) {
        unsigned int ssboIn = 0, ssboOut = 0;
        auto         byteSize = static_cast<long>(count * sizeof(T));

        _glGenBuffers(1, &ssboIn);
        _glGenBuffers(1, &ssboOut);

        // upload input
        _glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIn);
        _glBufferData(GL_SHADER_STORAGE_BUFFER, byteSize, hostInput, 0x88E8 /*GL_DYNAMIC_COPY*/);

        // allocate output
        _glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboOut);
        _glBufferData(GL_SHADER_STORAGE_BUFFER, byteSize, nullptr, 0x88E8 /*GL_DYNAMIC_COPY*/);

        // bind to shader storage binding points
        _glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboIn);
        _glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboOut);

        // dispatch
        _glUseProgram(program);
        auto numGroups = static_cast<unsigned int>((count + workgroupSize - 1) / workgroupSize);
        _glDispatchCompute(numGroups, 1, 1);
        _glMemoryBarrier(0x00000002 /*GL_SHADER_STORAGE_BARRIER_BIT*/);

        // read back
        _glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboOut);
        auto* mapped = static_cast<T*>(_glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, byteSize, 0x0001 /*GL_MAP_READ_BIT*/));
        if (mapped) {
            std::memcpy(hostOutput, mapped, count * sizeof(T));
            _glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }

        _glDeleteBuffers(1, &ssboIn);
        _glDeleteBuffers(1, &ssboOut);
    }

private:
    void loadFunctions() {
        auto load             = [](const char* name) { return reinterpret_cast<void*>(eglGetProcAddress(name)); };
        _glCreateShader       = reinterpret_cast<PFNGLCREATESHADERPROC>(load("glCreateShader"));
        _glShaderSource       = reinterpret_cast<PFNGLSHADERSOURCEPROC>(load("glShaderSource"));
        _glCompileShader      = reinterpret_cast<PFNGLCOMPILESHADERPROC>(load("glCompileShader"));
        _glGetShaderiv        = reinterpret_cast<PFNGLGETSHADERIVPROC>(load("glGetShaderiv"));
        _glGetShaderInfoLog   = reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(load("glGetShaderInfoLog"));
        _glCreateProgram      = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(load("glCreateProgram"));
        _glAttachShader       = reinterpret_cast<PFNGLATTACHSHADERPROC>(load("glAttachShader"));
        _glLinkProgram        = reinterpret_cast<PFNGLLINKPROGRAMPROC>(load("glLinkProgram"));
        _glUseProgram         = reinterpret_cast<PFNGLUSEPROGRAMPROC>(load("glUseProgram"));
        _glDeleteShader       = reinterpret_cast<PFNGLDELETESHADERPROC>(load("glDeleteShader"));
        _glDeleteProgram      = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(load("glDeleteProgram"));
        _glDispatchCompute    = reinterpret_cast<PFNGLDISPATCHCOMPUTEPROC>(load("glDispatchCompute"));
        _glMemoryBarrier      = reinterpret_cast<PFNGLMEMORYBARRIERPROC>(load("glMemoryBarrier"));
        _glGenBuffers         = reinterpret_cast<PFNGLGENBUFFERSPROC>(load("glGenBuffers"));
        _glDeleteBuffers      = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(load("glDeleteBuffers"));
        _glBindBufferBase     = reinterpret_cast<PFNGLBINDBUFFERBASEPROC>(load("glBindBufferBase"));
        _glBufferData         = reinterpret_cast<PFNGLBUFFERDATAPROC>(load("glBufferData"));
        _glBindBuffer         = reinterpret_cast<PFNGLBINDBUFFERPROC>(load("glBindBuffer"));
        _glMapBufferRange     = reinterpret_cast<PFNGLMAPBUFFERRANGEPROC>(load("glMapBufferRange"));
        _glUnmapBuffer        = reinterpret_cast<PFNGLUNMAPBUFFERPROC>(load("glUnmapBuffer"));
        _glGetProgramiv       = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(load("glGetProgramiv"));
        _glGetProgramInfoLog  = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(load("glGetProgramInfoLog"));
        _glGetUniformLocation = reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(load("glGetUniformLocation"));
        _glUniform1ui         = reinterpret_cast<PFNGLUNIFORM1UIPROC>(load("glUniform1ui"));
        _glCopyBufferSubData  = reinterpret_cast<PFNGLCOPYBUFFERSUBDATAPROC>(load("glCopyBufferSubData"));
    }
};

#else

struct GlComputeContext {
    [[nodiscard]] bool                                     isAvailable() const noexcept { return false; }
    [[nodiscard]] std::expected<unsigned int, std::string> compileOrGetCached(std::string_view) { return 0; }

    template<typename T>
    void dispatch(unsigned int, const T*, T*, std::size_t, std::size_t) {}
};

#endif // GR_DEVICE_HAS_GL_COMPUTE

} // namespace gr::device

#endif // GNURADIO_GL_COMPUTE_CONTEXT_HPP
