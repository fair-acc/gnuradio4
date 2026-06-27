#ifndef GNURADIO_DEVICE_CONTEXT_GLSL_HPP
#define GNURADIO_DEVICE_CONTEXT_GLSL_HPP

#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/GlComputeContext.hpp>

namespace gr::device {

#if GR_DEVICE_HAS_GL_COMPUTE

/**
 * @brief OpenGL 4.3 compute shader backend for DeviceContext.
 *
 * Device memory is backed by persistent SSBOs. The SSBO handle (GLuint) is stored as void*
 * token — not a real pointer. H2D/D2H use glBufferData/glMapBufferRange.
 * Shader dispatch via dispatch(program, inBuf, outBuf, count, workgroupSize).
 *
 * Requires a GlComputeContext (EGL headless or display-backed).
 *
 * @example
 * gr::device::GlComputeContext gl;
 * gr::device::DeviceContextGLSL ctx(gl);
 * auto* buf = ctx.allocateDevice<float>(1024);
 * ctx.copyHostToDevice(hostData, buf, 1024);
 * auto prog = gl.compileOrGetCached(glslSource);
 * ctx.dispatch(*prog, buf, outBuf, 1024, 256);
 * ctx.copyDeviceToHost(outBuf, hostData, 1024);
 */
struct DeviceContextGLSL final : DeviceContext {
    using DeviceContext::copyDeviceToHost;
    using DeviceContext::copyHostToDevice;

    GlComputeContext* _gl = nullptr;

    // track SSBO sizes for H2D/D2H (GL doesn't expose buffer size from handle)
    std::unordered_map<unsigned int, std::size_t> _ssboSizes;

    explicit DeviceContextGLSL(GlComputeContext& gl) : _gl(&gl) {}

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::GLSL; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::GPU; }
    [[nodiscard]] std::string   shortName() const override {
        auto renderer = _gl->rendererName();
        if (auto pos = renderer.find("/PCIe"); pos != std::string::npos) {
            renderer.resize(pos);
        }
        if (auto pos = renderer.find("/SSE"); pos != std::string::npos) {
            renderer.resize(pos);
        }
        return "GLSL:" + renderer;
    }
    [[nodiscard]] std::string name() const override { return _gl->rendererName() + " — " + _gl->vendorName(); }
    [[nodiscard]] std::string version() const override { return _gl->versionString(); }

    void copyHostToDevice(const void* host, void* device, std::size_t bytes) override {
        auto ssbo = toSsbo(device);
        _gl->_glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        _gl->_glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<long>(bytes), host, 0x88E8 /*GL_DYNAMIC_COPY*/);
    }

    void copyDeviceToHost(const void* device, void* host, std::size_t bytes) override {
        auto ssbo = toSsbo(device);
        _gl->_glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        auto* mapped = _gl->_glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<long>(bytes), 0x0001 /*GL_MAP_READ_BIT*/);
        if (mapped) {
            std::memcpy(host, mapped, bytes);
            _gl->_glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
    }

    void wait() override {
        // GL compute is synchronous after glMemoryBarrier — nothing to wait for
    }

    // GLSL-specific: dispatch a compiled compute shader
    void dispatch(unsigned int program, void* inBuf, void* outBuf, std::size_t count, std::size_t workgroupSize) {
        _gl->_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, toSsbo(inBuf));
        _gl->_glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, toSsbo(outBuf));
        _gl->_glUseProgram(program);
        auto numGroups = static_cast<unsigned int>((count + workgroupSize - 1) / workgroupSize);
        _gl->_glDispatchCompute(numGroups, 1, 1);
        _gl->_glMemoryBarrier(0x00000002 /*GL_SHADER_STORAGE_BARRIER_BIT*/);
    }

    [[nodiscard]] std::expected<unsigned int, std::string> compileOrGetCached(std::string_view glsl) { return _gl->compileOrGetCached(glsl); }

    [[nodiscard]] void* allocateDeviceRaw(std::size_t bytes, std::size_t /*alignment*/) override {
        unsigned int ssbo = 0;
        _gl->_glGenBuffers(1, &ssbo);
        _gl->_glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        _gl->_glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<long>(bytes), nullptr, 0x88E8 /*GL_DYNAMIC_COPY*/);
        _ssboSizes[ssbo] = bytes;
        return toPtr(ssbo);
    }

    [[nodiscard]] void* allocateHostRaw(std::size_t bytes, std::size_t) override { return ::operator new(bytes, std::align_val_t{alignof(std::max_align_t)}); }

    [[nodiscard]] void* allocateSharedRaw(std::size_t bytes, std::size_t /*alignment*/) override {
        return allocateDeviceRaw(bytes, 0); // GL has no shared memory concept — use SSBO
    }

    void deallocateRaw(void* ptr) override {
        auto ssbo = toSsbo(ptr);
        if (ssbo != 0) {
            _gl->_glDeleteBuffers(1, &ssbo);
            _ssboSizes.erase(ssbo);
        } else {
            ::operator delete(ptr, std::align_val_t{alignof(std::max_align_t)});
        }
    }

    static void*        toPtr(unsigned int ssbo) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ssbo)); }
    static unsigned int toSsbo(const void* ptr) { return static_cast<unsigned int>(reinterpret_cast<std::uintptr_t>(ptr)); }
};

#else

// stub when GL compute is not available
struct DeviceContextGLSL final : DeviceContext {
    using DeviceContext::copyDeviceToHost;
    using DeviceContext::copyHostToDevice;

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::GLSL; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::GPU; }
    [[nodiscard]] std::string   shortName() const override { return "GLSL:n/a"; }
    [[nodiscard]] std::string   name() const override { return "GLSL (not available)"; }
    [[nodiscard]] std::string   version() const override { return "n/a"; }

    void                copyHostToDevice(const void*, void*, std::size_t) override {}
    void                copyDeviceToHost(const void*, void*, std::size_t) override {}
    void                wait() override {}
    [[nodiscard]] void* allocateDeviceRaw(std::size_t, std::size_t) override { return nullptr; }
    [[nodiscard]] void* allocateHostRaw(std::size_t bytes, std::size_t) override { return ::operator new(bytes, std::align_val_t{alignof(std::max_align_t)}); }
    [[nodiscard]] void* allocateSharedRaw(std::size_t, std::size_t) override { return nullptr; }
    void                deallocateRaw(void*) override {}
};

#endif // GR_DEVICE_HAS_GL_COMPUTE

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_GLSL_HPP
