#ifndef GNURADIO_DEVICE_CONTEXT_ROCM_HPP
#define GNURADIO_DEVICE_CONTEXT_ROCM_HPP

#include <gnuradio-4.0/device/DeviceContext.hpp>

#if GR_DEVICE_HAS_ROCM

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

#include <hip/hip_runtime.h>

namespace gr::device {

/**
 * @brief `DeviceContext` over the HIP runtime, the ROCm counterpart of `DeviceContextCuda`.
 *
 * HIP's runtime API mirrors CUDA's name for name, so this is the CUDA context with `cuda` replaced by `hip` and
 * `cudaMallocHost` by `hipHostMalloc`. Verified on ROCm 6.4 with an RX 7700S (gfx1102), compiled by GCC 15 and
 * Clang 20 without `hipcc`. Two details worth knowing:
 *
 *  * `hipHostFree` frees what `hipHostMalloc` returned -- NOT `hipFree`, exactly as on the CUDA side
 *  * `hipPointerGetAttributes` reports `hipMemoryTypeDevice`/`hipMemoryTypeManaged`; ROCm 6.4 also still
 *    defines the older `hipMemoryTypeUnified`, which is not what managed allocations report
 *
 * Like the CUDA one this carries memory and transfers only, has no kernel launch path, and therefore does not
 * join `kHasDeviceBackend`.
 */
struct DeviceContextRocm final : DeviceContext {
    int         _device = 0;
    std::string _name;
    std::string _version;

    explicit DeviceContextRocm(int device = 0) : _device(device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, _device) == hipSuccess) {
            _name    = properties.name;
            _version = std::format("{}.{}", properties.major, properties.minor);
        } else {
            _name    = "unknown ROCm device";
            _version = "0.0";
        }
    }

    [[nodiscard]] static bool isServed() noexcept {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    [[nodiscard]] DeviceBackend backend() const noexcept override { return DeviceBackend::ROCm; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return DeviceType::GPU; }
    [[nodiscard]] std::string   shortName() const override { return std::format("ROCm:{}", _name); }
    [[nodiscard]] std::string   name() const override { return _name; }
    [[nodiscard]] std::string   version() const override { return _version; }

    void wait() override {
        std::ignore = hipSetDevice(_device);
        std::ignore = hipDeviceSynchronize();
    }

    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        if (wanted == Residency::invalid || bytes == 0UZ) {
            return {};
        }
        const std::size_t effective = std::max(align, alignof(std::max_align_t));
        const std::size_t request   = bytes + effective - 1UZ;
        std::ignore                 = hipSetDevice(_device);

        void*      p      = nullptr;
        hipError_t status = hipErrorInvalidValue;
        switch (wanted) {
        case Residency::host: status = hipHostMalloc(&p, request, hipHostMallocDefault); break;
        case Residency::shared: status = hipMallocManaged(&p, request); break;
        case Residency::devicePtr: status = hipMalloc(&p, request); break;
        case Residency::invalid: break;
        }
        if (status != hipSuccess || p == nullptr) {
            return {};
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
        std::ignore = hipSetDevice(_device);
        if (buf.residency == Residency::host) {
            std::ignore = hipHostFree(p);
        } else {
            std::ignore = hipFree(p);
        }
    }

    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override {
        if (host == nullptr || !dst || bytes == 0UZ) {
            return;
        }
        std::ignore = hipSetDevice(_device);
        std::ignore = hipMemcpy(reinterpret_cast<void*>(dst.token), host, std::min(bytes, dst.bytes), hipMemcpyHostToDevice);
    }

    void download(DeviceBuffer src, void* host, std::size_t bytes) override {
        if (host == nullptr || !src || bytes == 0UZ) {
            return;
        }
        std::ignore = hipSetDevice(_device);
        std::ignore = hipMemcpy(host, reinterpret_cast<const void*>(src.token), std::min(bytes, src.bytes), hipMemcpyDeviceToHost);
    }

    [[nodiscard]] bool hasRemoteMemory() const noexcept override { return true; }

    [[nodiscard]] bool isDeviceAccessible(const void* ptr) const noexcept override {
        if (ptr == nullptr) {
            return false;
        }
        hipPointerAttribute_t attributes{};
        if (hipPointerGetAttributes(&attributes, ptr) != hipSuccess) {
            std::ignore = hipGetLastError();
            return false;
        }
        return attributes.type == hipMemoryTypeDevice || attributes.type == hipMemoryTypeManaged;
    }

    [[nodiscard]] bool isDeviceOnly(const void* ptr) const noexcept override {
        if (ptr == nullptr) {
            return false;
        }
        hipPointerAttribute_t attributes{};
        if (hipPointerGetAttributes(&attributes, ptr) != hipSuccess) {
            std::ignore = hipGetLastError();
            return false;
        }
        return attributes.type == hipMemoryTypeDevice;
    }

    [[nodiscard]] std::optional<std::string> peekDeviceError() noexcept override {
        const hipError_t status = hipPeekAtLastError();
        return status == hipSuccess ? std::nullopt : std::optional<std::string>{hipGetErrorString(status)};
    }

    [[nodiscard]] std::optional<std::string> pollDeviceError() noexcept override {
        const hipError_t status = hipGetLastError();
        return status == hipSuccess ? std::nullopt : std::optional<std::string>{hipGetErrorString(status)};
    }
};

} // namespace gr::device

#endif // GR_DEVICE_HAS_ROCM
#endif // GNURADIO_DEVICE_CONTEXT_ROCM_HPP
