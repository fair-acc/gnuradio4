#ifndef GNURADIO_DEVICE_CONTEXT_CUDA_HPP
#define GNURADIO_DEVICE_CONTEXT_CUDA_HPP

#include <gnuradio-4.0/device/DeviceContext.hpp>

#if GR_DEVICE_HAS_CUDA

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

#include <cuda_runtime.h>

namespace gr::device {

/**
 * @brief `DeviceContext` over the CUDA runtime, to show the contract is backend-neutral.
 *
 * Allocation, transfer and synchronisation only -- all of which the CUDA RUNTIME API provides in plain C, so
 * this compiles with the ordinary C++ compiler and needs neither `nvcc` nor `enable_language(CUDA)`. Launching
 * a block-authored kernel does need both, and is deliberately not attempted here: what this proves is that
 * `DeviceContext`, `DeviceBuffer` and the residency model carry a second backend, not that existing device
 * blocks run on CUDA. The latter additionally needs the `GR_DEVICE_FN` annotations on every kernel body and
 * every device-block translation unit compiled by a CUDA compiler.
 *
 * Residency maps onto the three allocators the model already names: `host` is pinned host memory, `shared` is
 * managed memory, `devicePtr` is device-only. That is the same three-way split `DeviceContextSycl` makes.
 */
struct DeviceContextCuda final : DeviceContext {
    int         _device = 0;
    std::string _name;
    std::string _version;

    explicit DeviceContextCuda(int device = 0) : _device(device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, _device) == cudaSuccess) {
            _name    = properties.name;
            _version = std::format("{}.{}", properties.major, properties.minor);
        } else {
            _name    = "unknown CUDA device";
            _version = "0.0";
        }
    }

    /// whether a CUDA device is present at all; the registry asks before constructing one
    [[nodiscard]] static bool isServed() noexcept {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::CUDA; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::GPU; }
    [[nodiscard]] std::string   shortName() const override { return std::format("CUDA:{}", _name); }
    [[nodiscard]] std::string   name() const override { return _name; }
    [[nodiscard]] std::string   version() const override { return _version; }

    void wait() override {
        std::ignore = cudaSetDevice(_device);
        std::ignore = cudaDeviceSynchronize();
    }

    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        if (wanted == Residency::invalid || bytes == 0UZ) {
            return {};
        }
        // the same over-allocate-then-align dance the SYCL context does: CUDA's allocators promise 256-byte
        // alignment, but the caller may ask for more and the token must be aligned to what it asked for
        const std::size_t effective = std::max(align, alignof(std::max_align_t));
        const std::size_t request   = bytes + effective - 1UZ;
        std::ignore                 = cudaSetDevice(_device);

        void*       p      = nullptr;
        cudaError_t status = cudaErrorInvalidValue;
        switch (wanted) {
        case Residency::host: status = cudaMallocHost(&p, request); break;
        case Residency::shared: status = cudaMallocManaged(&p, request); break;
        case Residency::devicePtr: status = cudaMalloc(&p, request); break;
        case Residency::invalid: break;
        }
        if (status != cudaSuccess || p == nullptr) {
            return {}; // "cannot serve" -- never a lying token
        }
        const std::uintptr_t raw     = reinterpret_cast<std::uintptr_t>(p);
        const std::uintptr_t aligned = (raw + effective - 1UZ) & ~(static_cast<std::uintptr_t>(effective) - 1UZ);
        return DeviceBuffer{.token = aligned, .bytes = bytes, .residency = wanted, .align = effective, .allocation = raw};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf || buf.allocation == 0UZ) {
            return;
        }
        void* p     = reinterpret_cast<void*>(buf.allocation);
        std::ignore = cudaSetDevice(_device);
        if (buf.residency == Residency::host) {
            std::ignore = cudaFreeHost(p); // pinned memory has its own free
        } else {
            std::ignore = cudaFree(p);
        }
    }

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override {
        if (host == nullptr || !dst || bytes == 0UZ) {
            return;
        }
        std::ignore = cudaSetDevice(_device);
        std::ignore = cudaMemcpy(reinterpret_cast<void*>(dst.token), host, std::min(bytes, dst.bytes), cudaMemcpyHostToDevice);
    }

    void download(DeviceBuffer src, void* host, std::size_t bytes) override {
        if (host == nullptr || !src || bytes == 0UZ) {
            return;
        }
        std::ignore = cudaSetDevice(_device);
        std::ignore = cudaMemcpy(host, reinterpret_cast<const void*>(src.token), std::min(bytes, src.bytes), cudaMemcpyDeviceToHost);
    }

    [[nodiscard]] bool hasRemoteMemory() const noexcept override { return true; }

    [[nodiscard]] bool isDeviceAccessible(const void* ptr) const noexcept override {
        if (ptr == nullptr) {
            return false;
        }
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
            std::ignore = cudaGetLastError(); // an unregistered host pointer sets the sticky flag; clear it
            return false;
        }
        return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
    }

    [[nodiscard]] bool isDeviceOnly(const void* ptr) const noexcept override {
        if (ptr == nullptr) {
            return false;
        }
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
            std::ignore = cudaGetLastError();
            return false;
        }
        return attributes.type == cudaMemoryTypeDevice; // managed is reachable from the host, so it is not device-ONLY
    }

    [[nodiscard]] std::optional<std::string> peekDeviceError() noexcept override {
        const cudaError_t status = cudaPeekAtLastError();
        return status == cudaSuccess ? std::nullopt : std::optional<std::string>{cudaGetErrorString(status)};
    }

    [[nodiscard]] std::optional<std::string> pollDeviceError() noexcept override {
        const cudaError_t status = cudaGetLastError(); // clears, unlike peek
        return status == cudaSuccess ? std::nullopt : std::optional<std::string>{cudaGetErrorString(status)};
    }
};

} // namespace gr::device

#endif // GR_DEVICE_HAS_CUDA
#endif // GNURADIO_DEVICE_CONTEXT_CUDA_HPP
