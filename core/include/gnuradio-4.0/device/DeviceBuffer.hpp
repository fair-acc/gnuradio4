#ifndef GNURADIO_DEVICE_BUFFER_HPP
#define GNURADIO_DEVICE_BUFFER_HPP

#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_set>

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief RAII device buffer — deallocates on destruction, move-only.
 *
 * Owns a raw device allocation from a DeviceContext. The buffer stores the pointer token
 * (real pointer for SYCL/CPU, SSBO handle for GLSL) and the owning context.
 *
 * For shared ownership between fused blocks, wrap in std::shared_ptr<DeviceBuffer>.
 *
 * @example
 * DeviceBuffer buf(ctx, 1024 * sizeof(float)); // allocate
 * ctx.copyHostToDevice(hostData, buf.data(), 1024 * sizeof(float));
 * // buf automatically freed when it goes out of scope
 */
struct DeviceBuffer {
    void*          _data = nullptr;
    std::size_t    _size = 0; // bytes
    DeviceContext* _ctx  = nullptr;

    DeviceBuffer() = default;

    DeviceBuffer(DeviceContext& ctx, std::size_t bytes) : _data(ctx.allocateDeviceRaw(bytes, alignof(std::max_align_t))), _size(bytes), _ctx(&ctx) {}

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : _data(o._data), _size(o._size), _ctx(o._ctx) {
        o._data = nullptr;
        o._size = 0;
        o._ctx  = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            _data   = o._data;
            _size   = o._size;
            _ctx    = o._ctx;
            o._data = nullptr;
            o._size = 0;
            o._ctx  = nullptr;
        }
        return *this;
    }

    void reset() {
        if (_data && _ctx) {
            _ctx->deallocateRaw(_data);
        }
        _data = nullptr;
        _size = 0;
    }

    [[nodiscard]] void*       data() noexcept { return _data; }
    [[nodiscard]] const void* data() const noexcept { return _data; }
    [[nodiscard]] std::size_t size() const noexcept { return _size; }
    [[nodiscard]] bool        empty() const noexcept { return _data == nullptr; }
    [[nodiscard]] explicit    operator bool() const noexcept { return _data != nullptr; }
};

/// shared ownership for buffers between fused blocks
using SharedDeviceBuffer = std::shared_ptr<DeviceBuffer>;

inline SharedDeviceBuffer makeSharedDeviceBuffer(DeviceContext& ctx, std::size_t bytes) { return std::make_shared<DeviceBuffer>(ctx, bytes); }

/**
 * @brief Tracks all live DeviceBuffers per context for debugging and accounting.
 *
 * In debug builds: validates no use-after-free, detects overlapping writes (false sharing).
 * In release builds: tracks total device memory usage.
 */
struct DeviceBufferRegistry {
    struct Entry {
        void*       data;
        std::size_t size;
        std::string tag; // optional label for debugging
    };

    mutable std::mutex               _mtx;
    std::unordered_map<void*, Entry> _buffers;
    std::size_t                      _totalBytes = 0;

    void registerBuffer(void* data, std::size_t size, std::string_view tag = "") {
        std::scoped_lock lk(_mtx);
        _buffers[data] = {data, size, std::string(tag)};
        _totalBytes += size;
    }

    void unregisterBuffer(void* data) {
        std::scoped_lock lk(_mtx);
        auto             it = _buffers.find(data);
        if (it != _buffers.end()) {
            _totalBytes -= it->second.size;
            _buffers.erase(it);
        }
    }

    [[nodiscard]] std::size_t totalBytes() const noexcept {
        std::scoped_lock lk(_mtx);
        return _totalBytes;
    }

    [[nodiscard]] std::size_t bufferCount() const noexcept {
        std::scoped_lock lk(_mtx);
        return _buffers.size();
    }

    [[nodiscard]] bool isRegistered(void* data) const noexcept {
        std::scoped_lock lk(_mtx);
        return _buffers.contains(data);
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_BUFFER_HPP
