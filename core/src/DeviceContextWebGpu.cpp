#include <gnuradio-4.0/device/DeviceContextWebGpu.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <format>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include <emscripten/proxying.h>
#include <emscripten/threading.h>

#include <gnuradio-4.0/Logger.hpp>

extern "C" {
extern unsigned char __data_end; // wasm-ld: the end of static data
// DeviceContextWebGpu.js, on the browser main thread; the asynchronous ones finish `ctx` once their promise settles
void gr4WebGpuInit(em_proxying_ctx* ctx, int* status, char* name, int nameBytes, std::uint32_t staticEnd);
int  gr4WebGpuKernelStackBytes(std::uint32_t kernel);
void gr4WebGpuLaunch(em_proxying_ctx* ctx, void* batch, char* message, int messageBytes);
void gr4WebGpuTransfer(em_proxying_ctx* ctx, std::uint32_t destination, std::uint32_t source, std::uint32_t bytes, int deviceOnlySides, int* status);
}

namespace gr::device {
namespace {

/// one GPU operation at a time: the main thread's heap, staging and uniform buffers and its error scopes serve every
/// scheduler thread, and an interleaved launch would upload memory that predates another's results
std::mutex& gpuOperations() {
    static std::mutex mutex;
    return mutex;
}

/// runs `task(ctx)` on the browser main thread and blocks this one until the task finishes `ctx`
template<typename TTask>
void onMainThread(TTask& task) {
    emscripten_proxy_sync_with_ctx(emscripten_proxy_get_system_queue(), emscripten_main_runtime_thread_id(), [](em_proxying_ctx* ctx, void* arg) { (*static_cast<TTask*>(arg))(ctx); }, &task);
}

// which side of a transfer is device-only memory (the GPU's copy); the other side lives in wasm memory
constexpr int kToDevice = 1;
constexpr int kToHost   = 2;

constexpr std::int32_t kRefusedByAdapter = 2; // `gr4WebGpuLaunch`'s status: no pipeline on this adapter, the CPU runs the kernel

/// a launch enqueued with `await = false`, run with the next flush; its scratch (closure and stacks) lives until then
struct PendingLaunch {
    std::uint32_t kernel;
    std::uint32_t closure; // its scratch, [closure, stackTop), holds the closure and then each work item's stack
    std::uint32_t count;
    std::uint32_t stackTop;
    std::uint32_t stackBytes;
};
static_assert(sizeof(PendingLaunch) == 5UZ * sizeof(std::uint32_t), "gr4WebGpuLaunch reads the queue as is");

/// what `gr4WebGpuLaunch` reads: every pending launch in enqueue order, the live mirrored allocations, and an optional
/// copy of device-only memory to wasm memory appended to the same submit
struct FlushBatch {
    std::uint32_t launchCount;
    std::uint32_t launches;   // address of `launchCount` × `PendingLaunch`
    std::uint32_t heapEnd;    // end of the highest live allocation: how far the GPU buffer must reach
    std::uint32_t ranges;     // address of [begin, end) word pairs: the live mirrored allocations, small gaps bridged
    std::uint32_t rangeWords; // their count times two
    std::uint32_t live;       // address of the same allocations unbridged, which alone the GPU may change
    std::uint32_t liveWords;
    std::uint32_t downloadDestination;
    std::uint32_t downloadSource;
    std::uint32_t downloadBytes; // 0: no download
    std::int32_t  status = -1;
};

constexpr std::size_t   kMaxPendingLaunches = 64UZ;  // the uniforms of one flush: 64 slots of 256 bytes
constexpr std::uint32_t kMaxBridgedGapBytes = 4096U; // moving a page with a range costs less than another writeBuffer and copy

/// with `gpuOperations()` held
std::vector<PendingLaunch>& pendingLaunches() {
    static std::vector<PendingLaunch> pending;
    return pending;
}

/// whether `pendingLaunches()` holds any, read without `gpuOperations()`: a flush frees its launches' scratch holding it
std::atomic<bool>& launchesQueued() {
    static std::atomic<bool> queued{false};
    return queued;
}

/// with `gpuOperations()` held: the failure of a flush before a free, whose caller cannot be told; the next
/// `webGpuFlush()` reports it
std::string& unreportedFailure() {
    static std::string failure;
    return failure;
}

struct Download {
    std::uint32_t destination = 0U;
    std::uint32_t source      = 0U;
    std::uint32_t bytes       = 0U;
};

/// one round trip to the browser main thread, with `gpuOperations()` held: the pending launches in one submit, then
/// `download`; their scratch is released after. A status of `kRefusedByAdapter` only comes from a lone launch.
std::pair<std::int32_t, std::string> flushLocked(Download download = {}) {
    std::vector<PendingLaunch>& pending = pendingLaunches();
    if (pending.empty() && download.bytes == 0U) {
        return {0, {}};
    }
    const std::vector<std::uint32_t> live    = webGpuMemory().liveRanges(); // every pending launch's scratch included
    const std::vector<std::uint32_t> ranges  = bridgedRanges(live, webGpuDeviceMemory().liveRanges(), kMaxBridgedGapBytes);
    const std::uint32_t              readEnd = download.bytes == 0U ? 0U : download.source + download.bytes;
    FlushBatch                       batch{.launchCount = static_cast<std::uint32_t>(pending.size()), .launches = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pending.data())), .heapEnd = std::max({ranges.empty() ? 0U : ranges.back(), webGpuDeviceMemory().highestEnd(), readEnd}), .ranges = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(ranges.data())), .rangeWords = static_cast<std::uint32_t>(ranges.size()), .live = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(live.data())), .liveWords = static_cast<std::uint32_t>(live.size()), .downloadDestination = download.destination, .downloadSource = download.source, .downloadBytes = download.bytes};
    std::array<char, 512>            message{};
    auto                             task = [&](em_proxying_ctx* ctx) { gr4WebGpuLaunch(ctx, &batch, message.data(), static_cast<int>(message.size())); };
    onMainThread(task);
    const std::vector<PendingLaunch> ran = std::exchange(pending, {});
    launchesQueued().store(false, std::memory_order_release); // before the frees below, which would otherwise flush again
    for (const PendingLaunch& launch : ran) {
        webGpuMemory().deallocate(reinterpret_cast<void*>(static_cast<std::uintptr_t>(launch.closure)), launch.stackTop - launch.closure, WebGpuMemory::kAlign);
    }
    return {batch.status, std::string(message.data())};
}

/// a transfer with `gpuOperations()` already held, after the pending launches, which it would otherwise overtake
bool transferLocked(std::uint32_t destination, std::uint32_t source, std::uint32_t bytes, int sides) {
    if (const auto [status, message] = flushLocked(); status != 0) {
        gr::log::error("webgpu: the launches before a transfer failed: {}", message);
        return false;
    }
    int  status = -1;
    auto task   = [&](em_proxying_ctx* ctx) { gr4WebGpuTransfer(ctx, destination, source, bytes, sides, &status); };
    onMainThread(task);
    return status == 0;
}

/// per kernel asked for: its stack bytes per work item, -1 without a pipeline (none in the binary, or one the adapter
/// could not build), and whether it ran once on the adapter, so that its pipeline exists and a launch may be deferred
struct KnownKernels {
    struct Known {
        int  stackBytes = 0;
        bool built      = false;
    };
    std::mutex                               mutex;
    std::unordered_map<std::uint32_t, Known> kernels;
};

KnownKernels& knownKernels() {
    static KnownKernels known;
    return known;
}

int kernelStackBytes(std::uint32_t kernel) {
    KnownKernels&    known = knownKernels();
    std::scoped_lock lock(known.mutex);
    const auto [it, inserted] = known.kernels.try_emplace(kernel);
    if (inserted) {
        it->second.stackBytes = gr4WebGpuKernelStackBytes(kernel);
    }
    return it->second.stackBytes;
}

} // namespace

WebGpuAdapter webGpuInitialise() {
    if (emscripten_is_main_runtime_thread()) {
        gr::log::warning("webgpu: not served -- the browser main thread owns the GPU device and cannot wait for it; run main() on a pthread (-sPROXY_TO_PTHREAD)");
        return {};
    }
    int                   kind = 0;
    std::array<char, 256> name{};
    auto                  task = [&](em_proxying_ctx* ctx) { gr4WebGpuInit(ctx, &kind, name.data(), static_cast<int>(name.size()), static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&__data_end))); };
    onMainThread(task);
    return WebGpuAdapter{.kind = static_cast<WebGpuAdapter::Kind>(kind), .name = std::string(name.data())};
}

std::expected<std::optional<std::string>, std::string> webGpuLaunch(WebGpuKernelEntry entry, const void* closure, std::size_t closureBytes, std::size_t count, bool await) {
    assert(count > 0UZ);
    const auto kernel = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(entry));

    // a kernel the GPU cannot run runs here; device-only memory is the GPU's copy, so it comes to wasm memory first and
    // goes back after, under the lock, so that no launch on another thread changes it in between
    const auto runOnHost = [&](std::string reason) -> std::expected<std::optional<std::string>, std::string> {
        const std::vector<std::uint32_t> deviceOnly = webGpuDeviceMemory().liveRanges();
        if (!deviceOnly.empty() && emscripten_is_main_runtime_thread()) {
            return std::unexpected(std::format("kernel #{} {}, and device-only memory cannot be reached from the browser main thread", kernel, reason));
        }
        std::scoped_lock lock(gpuOperations());
        const auto       moveDeviceOnly = [&](int sides) {
            for (std::size_t r = 0UZ; r < deviceOnly.size(); r += 2UZ) {
                transferLocked(deviceOnly[r], deviceOnly[r], deviceOnly[r + 1UZ] - deviceOnly[r], sides);
            }
        };
        moveDeviceOnly(kToHost);
        for (std::uint32_t index = 0U; index < count; ++index) {
            entry(closure, index);
        }
        moveDeviceOnly(kToDevice);
        return std::optional<std::string>(std::move(reason));
    };

    if (emscripten_is_main_runtime_thread()) {
        return runOnHost("ran on the CPU: it was launched from the browser main thread, which cannot wait for the GPU");
    }
    const int stackBytes = kernelStackBytes(kernel);
    if (stackBytes < 0) {
        return runOnHost("ran on the CPU: it has no WebGPU pipeline (its .kernels.json names the reason)");
    }
    const std::size_t closureSlot = (closureBytes + 15UZ) & ~std::size_t{15}; // each work item's stack starts 16-byte aligned
    const auto        perItem     = static_cast<std::size_t>(stackBytes);
    if (perItem != 0UZ && count > (std::numeric_limits<std::size_t>::max() - closureSlot) / perItem) {
        return runOnHost(std::format("ran on the CPU: {} work items of {} stack bytes exceed wasm memory", count, perItem));
    }
    WebGpuMemory&     memory  = webGpuMemory();
    const std::size_t scratch = closureSlot + perItem * count;
    void* const       region  = [&]() -> void* {
        try {
            return memory.tryAllocate(scratch, WebGpuMemory::kAlign);
        } catch (const std::bad_alloc&) { // the bookkeeping's own allocation
            return nullptr;
        }
    }();
    if (region == nullptr) {
        return runOnHost(std::format("ran on the CPU: no room for its closure and {} bytes of stack", perItem * count));
    }
    std::memcpy(region, closure, closureBytes); // the closure lives on this thread's stack, which is not mirrored
    const auto base  = reinterpret_cast<std::uintptr_t>(region);
    const bool built = [&] {
        KnownKernels&    known = knownKernels();
        std::scoped_lock lock(known.mutex);
        return known.kernels[kernel].built;
    }();
    const PendingLaunch launch{.kernel = kernel, .closure = static_cast<std::uint32_t>(base), .count = static_cast<std::uint32_t>(count), .stackTop = static_cast<std::uint32_t>(base + scratch), .stackBytes = static_cast<std::uint32_t>(stackBytes)};
    std::int32_t        status = 0;
    std::string         message;
    {
        std::scoped_lock lock(gpuOperations());
        if (!built) { // a first launch builds its pipeline and may be refused, so it runs alone
            std::tie(status, message) = flushLocked();
        }
        if (status == 0) {
            pendingLaunches().push_back(launch);
            launchesQueued().store(true, std::memory_order_release);
            if (!await && built && pendingLaunches().size() < kMaxPendingLaunches) {
                return std::nullopt; // in order with every later launch and transfer, which flush it first
            }
            std::tie(status, message) = flushLocked();
        } else {
            memory.deallocate(region, scratch, WebGpuMemory::kAlign);
        }
    }
    if (status == 0 && !built) {
        KnownKernels&    known = knownKernels();
        std::scoped_lock lock(known.mutex);
        known.kernels[kernel].built = true;
    }
    if (status == kRefusedByAdapter) {
        {
            KnownKernels&    known = knownKernels();
            std::scoped_lock lock(known.mutex);
            known.kernels[kernel].stackBytes = -1; // later launches go to the CPU without a trip to the main thread
        }
        return runOnHost(std::format("ran on the CPU: {}", message));
    }
    if (status != 0) {
        return std::unexpected(std::format("kernel #{}: {}", kernel, message));
    }
    return std::nullopt;
}

std::expected<bool, std::string> webGpuFlush() {
    std::scoped_lock lock(gpuOperations());
    if (std::string failure = std::exchange(unreportedFailure(), {}); !failure.empty()) {
        return std::unexpected(std::move(failure));
    }
    if (pendingLaunches().empty()) {
        return false;
    }
    if (const auto [status, message] = flushLocked(); status != 0) {
        return std::unexpected(std::format("webgpu: deferred launches failed: {}", message));
    }
    return true;
}

void webGpuFlushBeforeFree() {
    if (!launchesQueued().load(std::memory_order_acquire) || emscripten_is_main_runtime_thread()) {
        return;
    }
    std::scoped_lock lock(gpuOperations());
    if (const auto [status, message] = flushLocked(); status != 0 && unreportedFailure().empty()) {
        unreportedFailure() = std::format("webgpu: deferred launches failed: {}", message);
    }
}

std::expected<void, std::string> webGpuTransfer(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0UZ) {
        return {};
    }
    if (emscripten_is_main_runtime_thread()) {
        return std::unexpected("a transfer to or from device-only memory was asked of the browser main thread, which cannot wait for the WebGPU device");
    }
    const int        sides = (webGpuDeviceMemory().contains(destination) ? kToDevice : 0) | (webGpuDeviceMemory().contains(source) ? kToHost : 0);
    const auto       to    = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(destination));
    const auto       from  = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(source));
    std::scoped_lock lock(gpuOperations());
    if (sides == kToHost) { // a read-back rides on the flush of the pending launches: one round trip for both
        if (const auto [status, message] = flushLocked({.destination = to, .source = from, .bytes = static_cast<std::uint32_t>(bytes)}); status != 0) {
            return std::unexpected(std::format("webgpu: a read-back of device-only memory failed: {}", message));
        }
        return {};
    }
    if (!transferLocked(to, from, static_cast<std::uint32_t>(bytes), sides)) {
        return std::unexpected("a transfer to or from device-only memory could not reach the WebGPU device");
    }
    return {};
}

} // namespace gr::device
