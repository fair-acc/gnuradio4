#ifndef GNURADIO_DEVICE_LOGGER_BACKEND_HPP
#define GNURADIO_DEVICE_LOGGER_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gnuradio-4.0/DeviceLog.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief owns the device log slab in the device-shared memory of a `DeviceContext`.
 *
 * Allocation goes through the backend-neutral `DeviceContext::allocateShared<T>()`, so one type
 * serves the CPU fallback (heap), SYCL (USM), and any future CUDA/ROCm context. Null-on-failure:
 * a failed allocation leaves `view()` invalid and device emitters then drop rather than write
 * through a null pointer.
 */
class DeviceLogSlabStorage {
    DeviceContext* _context{};
    std::byte*     _slots{};
    std::uint64_t* _counters{}; // [0]=cursor, [1]=dropped, [2]=truncated
    std::uint32_t  _slotCount{};
    std::uint32_t  _slotBytes{};

public:
    DeviceLogSlabStorage() = default;

    DeviceLogSlabStorage(DeviceContext& context, std::size_t slotCount, std::uint32_t slotBytes = gr::log::kDeviceLogSlotBytes) : _context(&context) {
        if (slotCount == 0UZ || slotBytes % gr::pmt::kBlobAlignment != 0U || gr::log::deviceLogPayloadCapacity(slotBytes) == 0U) {
            return;
        }
        _slots    = context.allocateShared<std::byte>(slotCount * slotBytes);
        _counters = context.allocateShared<std::uint64_t>(3UZ);
        if (_slots == nullptr || _counters == nullptr || (reinterpret_cast<std::uintptr_t>(_slots) & (gr::pmt::kBlobAlignment - 1UZ)) != 0UZ) {
            release();
            return;
        }

        _slotCount = static_cast<std::uint32_t>(slotCount);
        _slotBytes = slotBytes;
        view().reset();
    }

    ~DeviceLogSlabStorage() { release(); }

    DeviceLogSlabStorage(const DeviceLogSlabStorage&)            = delete;
    DeviceLogSlabStorage& operator=(const DeviceLogSlabStorage&) = delete;
    DeviceLogSlabStorage(DeviceLogSlabStorage&& other) noexcept { swap(other); }
    DeviceLogSlabStorage& operator=(DeviceLogSlabStorage&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    [[nodiscard]] bool valid() const noexcept { return _slotCount != 0U; }

    [[nodiscard]] gr::log::DeviceLogSlab view() const noexcept {
        if (_slotCount == 0U) {
            return {};
        }
        return gr::log::DeviceLogSlab{.slots = _slots, .cursor = &_counters[0], .dropped = &_counters[1], .truncated = &_counters[2], .slotCount = _slotCount, .slotBytes = _slotBytes};
    }

private:
    void release() noexcept {
        if (_context != nullptr) {
            if (_slots != nullptr) {
                _context->deallocate(_slots);
            }
            if (_counters != nullptr) {
                _context->deallocate(_counters);
            }
        }
        _slots     = nullptr;
        _counters  = nullptr;
        _slotCount = 0U;
        _slotBytes = 0U;
    }

    void swap(DeviceLogSlabStorage& other) noexcept {
        std::swap(_context, other._context);
        std::swap(_slots, other._slots);
        std::swap(_counters, other._counters);
        std::swap(_slotCount, other._slotCount);
        std::swap(_slotBytes, other._slotBytes);
    }
};

/**
 * @brief device-capable `gr::log::Backend`: host records go to the host backend, kernels write
 * `ValueMap` records into a device-shared slab that `flush()` decodes, renders and merges back.
 *
 * Install it before launching device code, hand `deviceLogger()` to the kernel, then synchronise
 * and flush:
 * @code
 * gr::device::DeviceLoggerBackend logBackend(context, 128UZ);
 * gr::log::setBackend(&logBackend);
 * context.parallelFor(n, Kernel{.log = logBackend.deviceLogger()});
 * context.wait();
 * std::ignore = gr::log::flush(); // decodes, renders with std::format, merges into the host backend
 * @endcode
 *
 * Host and device must never write the slab concurrently: cross-device atomics are not coherent, so
 * the device produces and the host drains only after a synchronisation barrier. Host-side
 * `gr::log::*` calls are unaffected — they go to the host backend's own ring.
 */
class DeviceLoggerBackend final : public gr::log::Backend {
    gr::log::Backend*    _host;
    DeviceLogSlabStorage _storage;

public:
    static constexpr std::size_t kDefaultSlotCount = 128UZ;

    explicit DeviceLoggerBackend(DeviceContext& context, std::size_t slotCount = kDefaultSlotCount, gr::log::Backend& hostBackend = gr::log::defaultBackend()) : _host(&hostBackend), _storage(context, slotCount) {
        if (!_storage.valid()) {
            gr::log::error("device log slab allocation failed - device records will be dropped");
        }
    }

    [[nodiscard]] bool                  valid() const noexcept { return _storage.valid(); }
    [[nodiscard]] gr::log::DeviceLogger deviceLogger() const noexcept { return gr::log::DeviceLogger{.slab = _storage.view()}; }

    [[nodiscard]] std::uint64_t droppedDeviceRecords() const noexcept { return _storage.view().droppedRecords(); }
    [[nodiscard]] std::uint64_t truncatedDeviceRecords() const noexcept { return _storage.view().truncatedRecords(); }

    bool publish(const gr::log::LogRecord& record) noexcept override { return _host->publish(record); }

    std::size_t drain(gr::log::RecordConsumer consumer, void* user) noexcept override { return _host->drain(consumer, user); }

    // call only after the device work has been synchronised (`DeviceContext::wait()`)
    std::size_t flush() noexcept override {
        const gr::log::DeviceLogSlab slab   = _storage.view();
        std::size_t                  merged = 0UZ;
#if __cpp_exceptions
        try { // host-side rendering goes through std::format, which may throw
            merged = gr::log::drainDeviceLog(slab, *_host);
        } catch (...) {
            merged = 0UZ;
        }
#else
        merged = gr::log::drainDeviceLog(slab, *_host);
#endif
        slab.recycle();
        return merged + _host->flush();
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_LOGGER_BACKEND_HPP
