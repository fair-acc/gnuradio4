#ifndef GNURADIO_DEVICE_CONTEXT_WEBGPU_HPP
#define GNURADIO_DEVICE_CONTEXT_WEBGPU_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <iterator>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/ComputeDomain.hpp>
#include <gnuradio-4.0/Export.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/BackendPolicy.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp> // AsyncErrorState

namespace gr::device {

/**
 * The one shape a WebGPU kernel has: a closure and a work-item index.
 *
 * Every instantiation is its own wasm function, compiled by the ordinary compiler with the host's data layout.
 * The post-link translator finds each one by this name in the linked binary and emits a WGSL compute pipeline
 * for it, so a block's unchanged `processOne`/`processBulk` reaches the GPU without a device compiler.
 * The function's address is the kernel's identity at run time, and calling it is the host fallback.
 */
template<typename TKernel>
[[gnu::noinline]] void webGpuKernel(const void* closure, std::uint32_t index) {
    (*static_cast<const TKernel*>(closure))(static_cast<std::size_t>(index));
}

using WebGpuKernelEntry = void (*)(const void*, std::uint32_t);

/**
 * The memory every WebGPU kernel can reach: ordinary wasm heap allocations, remembered while they are live and mirrored
 * at their own addresses into one GPU storage buffer.
 *
 * Kernels dereference pointers stored inside a block's mirror (a pmr field's `data()`), so a device address must equal
 * the host address of the same byte. A launch uploads the live allocations and writes back only the words the GPU
 * changed; static data goes up once. Nothing is reserved ahead; the GPU buffer reaches the highest live address, so the
 * adapter's `maxStorageBufferBindingSize` bounds it.
 */
struct WebGpuMemory final : MemoryResource {
    static constexpr std::size_t kAlign = 16UZ;

    /// device-only memory is the GPU copy's: launches leave it alone and the host reaches it through the context
    bool deviceOnly = false;

    explicit WebGpuMemory(bool deviceOnly_ = false) : deviceOnly(deviceOnly_) {}

    mutable std::mutex                    mutex;
    std::map<std::uintptr_t, std::size_t> live; // begin -> bytes

    /// nullptr when the heap has no room for the block, where `allocate` would throw; its bookkeeping can still throw
    /// `std::bad_alloc`, as any container insertion can
    [[nodiscard]] void* tryAllocate(std::size_t bytes, std::size_t align) {
        void* const p = ::operator new(bytes, std::align_val_t{std::max(align, kAlign)}, std::nothrow);
        return p == nullptr ? nullptr : remember(p, bytes);
    }

    void* do_allocate(std::size_t bytes, std::size_t align) override { return remember(::operator new(bytes, std::align_val_t{std::max(align, kAlign)}), bytes); }
    void* remember(void* p, std::size_t bytes) {
        std::scoped_lock lock(mutex);
        live[reinterpret_cast<std::uintptr_t>(p)] = bytes;
        return p;
    }
    void                                     do_deallocate(void* p, std::size_t bytes, std::size_t align) override;
    [[nodiscard]] bool                       do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    [[nodiscard]] MemoryResourceCapabilities capabilities() const noexcept override;

    /// one past the highest live byte, or 0
    [[nodiscard]] std::uint32_t highestEnd() const noexcept {
        std::scoped_lock lock(mutex);
        return live.empty() ? 0U : static_cast<std::uint32_t>(std::prev(live.end())->first + std::prev(live.end())->second);
    }

    [[nodiscard]] bool contains(const void* ptr) const noexcept {
        const auto       address = reinterpret_cast<std::uintptr_t>(ptr);
        std::scoped_lock lock(mutex);
        auto             after = live.upper_bound(address);
        return after != live.begin() && address < std::prev(after)->first + std::prev(after)->second;
    }

    /// word-aligned [begin, end) pairs covering every live allocation, adjacent ones merged
    [[nodiscard]] std::vector<std::uint32_t> liveRanges() const {
        std::vector<std::uint32_t> ranges;
        std::scoped_lock           lock(mutex);
        for (const auto& [address, size] : live) {
            const auto first = static_cast<std::uint32_t>(address & ~std::uintptr_t{3});
            const auto last  = static_cast<std::uint32_t>((address + size + 3U) & ~std::uintptr_t{3});
            if (!ranges.empty() && first <= ranges.back()) {
                ranges.back() = std::max(ranges.back(), last);
            } else {
                ranges.insert(ranges.end(), {first, last});
            }
        }
        return ranges;
    }
};

/// `ranges` with each gap of at most `maxGapBytes` closed, unless part of `keepOut` lies in it; both sorted [begin, end)
/// pairs as `liveRanges` returns them. The mirrored and device-only allocations share the wasm heap, so a gap may hold
/// device-only memory, which an upload of the gap would overwrite on the GPU.
[[nodiscard]] inline std::vector<std::uint32_t> bridgedRanges(std::span<const std::uint32_t> ranges, std::span<const std::uint32_t> keepOut, std::uint32_t maxGapBytes) {
    std::vector<std::uint32_t> bridged;
    std::size_t                nextKeepOut = 0UZ;
    for (std::size_t r = 0UZ; r < ranges.size(); r += 2UZ) {
        if (!bridged.empty()) {
            const std::uint32_t gapBegin = bridged.back();
            const std::uint32_t gapEnd   = ranges[r];
            while (nextKeepOut < keepOut.size() && keepOut[nextKeepOut + 1UZ] <= gapBegin) {
                nextKeepOut += 2UZ;
            }
            const bool keptOut = nextKeepOut < keepOut.size() && keepOut[nextKeepOut] < gapEnd;
            if (gapEnd - gapBegin <= maxGapBytes && !keptOut) {
                bridged.back() = ranges[r + 1UZ];
                continue;
            }
        }
        bridged.insert(bridged.end(), {ranges[r], ranges[r + 1UZ]});
    }
    return bridged;
}

/// process-lifetime, as the registry's contexts that hand it out: memory a kernel may still read is never released
[[nodiscard]] GNURADIO_EXPORT inline WebGpuMemory& webGpuMemory() {
    static auto& memory = *new WebGpuMemory();
    return memory;
}

/// edges between two blocks on the device: never moved by a launch, so their samples stay on the GPU
[[nodiscard]] GNURADIO_EXPORT inline WebGpuMemory& webGpuDeviceMemory() {
    static auto& memory = *new WebGpuMemory(true);
    return memory;
}

/// copies `bytes` between wasm addresses where either side may be device-only memory, on the browser main thread, in
/// order with launches: the queued launches run first, and a read-back to wasm memory rides on their flush; an error when
/// the WebGPU device cannot be reached
GNURADIO_EXPORT std::expected<void, std::string> webGpuTransfer(void* destination, const void* source, std::size_t bytes);

inline MemoryResourceCapabilities WebGpuMemory::capabilities() const noexcept {
    if constexpr (!kHasWebGpu) {
        return {}; // no device memory to copy, and `webGpuTransfer` exists only in a WebGPU build
    } else {
        if (!deviceOnly) {
            return {};
        }
        return {.deviceOnly = true, .copyWithin = +[](void* destination, const void* source, std::size_t bytes, void*) { std::ignore = webGpuTransfer(destination, source, bytes); }, .hostAddressable = &webGpuMemory()};
    }
}

/// runs `entry` for every index in [0, count), count > 0: nullopt when it ran, or is queued to run, on the GPU, else why it ran on
/// this thread (no pipeline, no room, the main thread); a failure in the user's words.
/// With `await`, or on a kernel's first launch, it returns once the results are in wasm memory. Without, a launch of a
/// kernel whose pipeline exists is queued, in order, until the next launch with `await`, transfer, `webGpuFlush()`, or
/// 64 queued launches, or a free of WebGPU memory: its memory may be read or written only after one of those.
GNURADIO_EXPORT std::expected<std::optional<std::string>, std::string> webGpuLaunch(WebGpuKernelEntry entry, const void* closure, std::size_t closureBytes, std::size_t count, bool await);

/// runs the queued launches, if any, in one submit: true when it did; their error, in the user's words
GNURADIO_EXPORT std::expected<bool, std::string> webGpuFlush();

/// runs the queued launches before memory they may write is freed, since both pools share the wasm heap and it hands
/// the address out again; not on the browser main thread, which cannot wait. A failure surfaces at the next `webGpuFlush()`.
GNURADIO_EXPORT void webGpuFlushBeforeFree();

inline void WebGpuMemory::do_deallocate(void* p, std::size_t /*bytes*/, std::size_t align) {
    if constexpr (kHasWebGpu) {
        webGpuFlushBeforeFree();
    }
    {
        std::scoped_lock lock(mutex);
        live.erase(reinterpret_cast<std::uintptr_t>(p));
    }
    ::operator delete(p, std::align_val_t{std::max(align, kAlign)});
}

/// what the browser offers: no adapter, a software one (SwiftShader, llvmpipe) or a hardware GPU
struct WebGpuAdapter {
    enum class Kind : int { none = 0, software = 1, hardware = 2 };
    Kind        kind = Kind::none;
    std::string name{};
};

/// asks the browser main thread for an adapter and a device, and loads the binary's translated kernels
GNURADIO_EXPORT WebGpuAdapter webGpuInitialise();

/**
 * WebGPU backend for DeviceContext: heap memory mirrored at host addresses, kernels translated from the linked wasm.
 *
 * Served as `gpu:webgpu` (also a bare `gpu`) for a hardware adapter and `gpu:swift` for SwiftShader, the browser's
 * software one, so `!` on `gpu:webgpu` means "a hardware GPU or stop".
 */
struct DeviceContextWebGpu final : DeviceContext {
    static constexpr DeviceBackend kBackend = DeviceBackend::WebGPU; // lets DeviceContext::as<> check without RTTI

    DeviceType                       type = DeviceType::CPU;
    std::string                      adapterName{"none"};
    std::shared_ptr<AsyncErrorState> errorState = std::make_shared<AsyncErrorState>();
    std::atomic<std::uint64_t>       gpuDispatches{0UZ};     // kernels that ran on the adapter, not on this thread: observable, as `syncCount()`
    std::atomic<std::uint64_t>       deviceToHostBytes{0UZ}; // device-only memory brought to the host: 0 for samples that stay on the GPU

    [[nodiscard]] DeviceBackend backend() const noexcept override { return kBackend; }
    [[nodiscard]] DeviceType    deviceType() const noexcept override { return type; }
    [[nodiscard]] std::string   shortName() const override { return "WebGPU:" + adapterName; }
    [[nodiscard]] std::string   name() const override { return adapterName; }
    [[nodiscard]] std::string   version() const override { return "webgpu"; }

    [[nodiscard]] std::pmr::memory_resource* resource(Access access) noexcept override { return access == Access::DeviceOnly ? &webGpuDeviceMemory() : &webGpuMemory(); }
    [[nodiscard]] bool                       isDeviceAccessible(const void* ptr) const noexcept override { return webGpuMemory().contains(ptr) || webGpuDeviceMemory().contains(ptr); }
    [[nodiscard]] bool                       isDeviceOnly(const void* ptr) const noexcept override { return webGpuDeviceMemory().contains(ptr); }

    void wait() override {
        _flushQueue();
        _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
    }

    [[nodiscard]] std::optional<std::string> peekDeviceError() noexcept override { return errorState->isPoisoned() ? std::optional<std::string>(errorState->messageCopy()) : std::nullopt; }
    [[nodiscard]] std::optional<std::string> pollDeviceError() noexcept override {
        _flushQueue(); // the queue wait a poll stands for
        return peekDeviceError();
    }
    void _flushQueue() {
        if (const std::expected<bool, std::string> flushed = webGpuFlush(); !flushed) {
            errorState->record(flushed.error());
        }
    }

    /// per thread: the launch and the block that asks run on the same scheduler thread
    static inline thread_local std::optional<std::string> pendingRefusal;
    [[nodiscard]] std::optional<std::string>              takeRefusal() noexcept override { return std::exchange(pendingRefusal, std::nullopt); }

    [[nodiscard]] DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) override {
        if (wanted == Residency::invalid) {
            return {};
        }
        const std::size_t effective = std::max(align, alignof(std::max_align_t));
        void* const       p         = (wanted == Residency::devicePtr ? webGpuDeviceMemory() : webGpuMemory()).tryAllocate(std::max(bytes, 1UZ), effective);
        if (p == nullptr) {
            return {};
        }
        return DeviceBuffer{.token = reinterpret_cast<std::uintptr_t>(p), .bytes = bytes, .residency = wanted, .align = effective, .allocation = reinterpret_cast<std::uintptr_t>(p)};
    }

    void deallocate(DeviceBuffer buf) override {
        if (!buf) {
            return;
        }
        (buf.residency == Residency::devicePtr ? webGpuDeviceMemory() : webGpuMemory()).deallocate(reinterpret_cast<void*>(buf.allocation), std::max(buf.bytes, 1UZ), buf.align);
    }

    void copy(void* destination, const void* source, std::size_t bytes, bool /*await*/ = true) override {
        if (isDeviceOnly(destination) || isDeviceOnly(source)) {
            if (isDeviceOnly(source) && !isDeviceOnly(destination)) {
                deviceToHostBytes.fetch_add(bytes, std::memory_order_relaxed);
            }
            if (const std::expected<void, std::string> moved = webGpuTransfer(destination, source, bytes); !moved) {
                errorState->record(moved.error());
            }
            return;
        }
        std::memcpy(destination, source, bytes);
    }
    void upload(const void* host, DeviceBuffer dst, std::size_t bytes) override { copy(reinterpret_cast<void*>(dst.token), host, bytes); }
    void download(DeviceBuffer src, void* host, std::size_t bytes) override { copy(host, reinterpret_cast<void*>(src.token), bytes); }

    /// with `await` the results are back in wasm memory on return; without, see `webGpuLaunch`
    template<typename TKernel>
    void parallelFor(std::size_t count, const TKernel& kernel, bool await) {
        if (errorState->isPoisoned()) {
            return;
        }
        if (count == 0UZ) {
            return;
        }
        const std::expected<std::optional<std::string>, std::string> ran = webGpuLaunch(&webGpuKernel<TKernel>, &kernel, sizeof(TKernel), count, await);
        if (!ran) {
            errorState->record(ran.error());
        } else if (!ran->has_value()) {
            gpuDispatches.fetch_add(1UZ, std::memory_order_relaxed);
        } else if (!pendingRefusal) {
            pendingRefusal = **ran; // the first of a dispatch's launches names it
        }
        if (await) {
            _syncCount.fetch_add(1UZ, std::memory_order_relaxed);
        }
    }
};

/// publishes `gpu:webgpu` for a hardware adapter or `gpu:swift` for SwiftShader; false when the build has no WebGPU backend
[[nodiscard]] inline bool registerWebGpuRuntime() {
    if constexpr (!kHasWebGpu) {
        return false;
    } else {
        static std::once_flag once;
        std::call_once(once, [] {
            const WebGpuAdapter adapter = webGpuInitialise();
            if (adapter.kind == WebGpuAdapter::Kind::none) {
                gr::log::info("webgpu: not served: {}", adapter.name);
                return;
            }
            const bool  hardware = adapter.kind == WebGpuAdapter::Kind::hardware;
            const char* backend  = hardware ? "webgpu" : "swift";
            ComputeRegistry::instance().registerProvider(backend, +[](const ComputeDomain& domain, void*) -> std::pmr::memory_resource* { return domain.access == Access::DeviceOnly ? &webGpuDeviceMemory() : &webGpuMemory(); });
            auto context                    = std::make_unique<DeviceContextWebGpu>();
            context->type                   = hardware ? DeviceType::GPU : DeviceType::CPU;
            context->adapterName            = adapter.name;
            DeviceContextRegistry& registry = DeviceContextRegistry::instance();
            registry.registerContext(std::format("gpu:{}:0", backend), std::move(context));
            registry.registerAlias(std::format("gpu:{}", backend), std::format("gpu:{}:0", backend));
            gr::log::info("webgpu: '{}' serves gpu:{}", adapter.name, backend);
        });
        return true;
    }
}

} // namespace gr::device

namespace gr {
/// WebGPU launches the closure's translated pipeline; see `gr::backend_policy`
template<>
struct backend_policy<device::DeviceContextWebGpu> {
    static constexpr bool has_parallel_for = device::kHasWebGpu;

    static constexpr device::DeviceBackend backend() noexcept { return device::DeviceContextWebGpu::kBackend; }

    template<typename TKernel>
    static void parallelFor(device::DeviceContextWebGpu& context, std::size_t count, TKernel kernel, bool await) {
        context.parallelFor(count, kernel, await);
    }
};
} // namespace gr

#endif // GNURADIO_DEVICE_CONTEXT_WEBGPU_HPP
