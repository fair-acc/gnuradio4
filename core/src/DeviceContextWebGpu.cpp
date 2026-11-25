#include <gnuradio-4.0/device/DeviceContextWebGpu.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
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
void gr4WebGpuInit(em_proxying_ctx* ctx, int* status, char* name, int nameBytes, std::uint32_t staticEnd, std::uint32_t* mirrorLimit);
int  gr4WebGpuKernelStackBytes(std::uint32_t kernel, char* refusal, int refusalBytes);
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

[[nodiscard]] std::uint32_t wasmAddress(const void* p) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(p)); }
[[nodiscard]] void*         atWasmAddress(std::uint32_t address) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(address)); }

constexpr int kDestinationDeviceOnly = 1; // a transfer's sides in device-only memory, the GPU's copy
constexpr int kSourceDeviceOnly      = 2;

constexpr std::int32_t kRefusedByAdapter = 2; // `gr4WebGpuLaunch`'s status: no pipeline on this adapter, the CPU runs the kernel

/// the `kernel` of a queued copy within device-only memory (wasm's table slot 0 is never a function): `count` bytes from
/// `closure` to `stackTop`, run by the flush in order with the launches around it, as an in-order SYCL queue runs one
constexpr std::uint32_t kQueuedCopy = 0U;

/// a launch enqueued with `await = false`, run with the next flush; its scratch (closure and stacks) lives until then
struct QueuedLaunch {
    std::uint32_t kernel;
    std::uint32_t closure; // a mirrored copy, so that the GPU sees it
    std::uint32_t count;
    std::uint32_t stackTop; // the work items' stacks lie below, in device-only memory: no flush moves them
    std::uint32_t stackBytes;
};
static_assert(sizeof(QueuedLaunch) == 5UZ * sizeof(std::uint32_t), "gr4WebGpuLaunch reads the queue as is");

/// what `gr4WebGpuLaunch` reads: every queued launch in enqueue order, the live mirrored allocations, and an optional
/// copy of device-only memory to wasm memory appended to the same submit
struct FlushBatch {
    std::uint32_t launchCount;
    std::uint32_t launchesAddress; // `launchCount` × `QueuedLaunch`
    std::uint32_t heapEnd;         // end of the highest live allocation: how far the GPU buffer must reach
    std::uint32_t rangesAddress;   // [begin, end) word pairs: the live mirrored allocations, small gaps bridged
    std::uint32_t rangeWords;      // their count times two
    std::uint32_t liveAddress;     // the same allocations unbridged, which alone the GPU may change
    std::uint32_t liveWords;
    std::uint32_t downloadDestination;
    std::uint32_t downloadSource;
    std::uint32_t downloadBytes; // 0: no download
    std::int32_t  status = -1;
};
static_assert(offsetof(FlushBatch, status) == 10UZ * sizeof(std::uint32_t), "gr4WebGpuLaunch writes the status at word 10");

constexpr std::size_t   kMaxQueuedLaunches  = 64UZ;  // the uniforms of one flush: 64 slots of 256 bytes
constexpr std::uint32_t kMaxBridgedGapBytes = 4096U; // moving a page with a range costs less than another writeBuffer and copy

/// with `gpuOperations()` held
std::vector<QueuedLaunch>& queuedLaunches() {
    static std::vector<QueuedLaunch> pending;
    return pending;
}

/// whether `queuedLaunches()` holds any, read without `gpuOperations()`: a flush frees its launches' scratch holding it
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

/// a launch's closure and stacks, once its flush ran or when it was never queued: no launch can still write them
void releaseScratch(const QueuedLaunch& launch) {
    if (launch.kernel == kQueuedCopy) {
        return; // its memory belongs to the caller
    }
    webGpuMemory().release(atWasmAddress(launch.closure), WebGpuMemory::kAlign);
    if (launch.stackBytes != 0U) {
        webGpuDeviceMemory().release(atWasmAddress(launch.stackTop - launch.stackBytes * launch.count), WebGpuMemory::kAlign);
    }
}

/// releases the browser main thread asked for while launches that may write them were queued, run after the next flush
struct HeldReleases {
    struct Held {
        WebGpuMemory* memory;
        void*         p;
        std::size_t   align;
    };
    std::mutex        mutex; // not `gpuOperations()`, which the main thread may not take
    std::vector<Held> releases;
};

HeldReleases& heldReleases() {
    static HeldReleases held;
    return held;
}

void releaseHeld() {
    std::vector<HeldReleases::Held> due;
    {
        std::scoped_lock lock(heldReleases().mutex);
        due.swap(heldReleases().releases);
    }
    for (const HeldReleases::Held& held : due) {
        held.memory->release(held.p, held.align);
    }
}

/// one round trip to the browser main thread, with `gpuOperations()` held: the queued launches in one submit, then
/// `download`; their scratch is released after. A status of `kRefusedByAdapter` only comes from a lone launch.
std::pair<std::int32_t, std::string> flushLocked(Download download = {}) {
    std::vector<QueuedLaunch>& pending = queuedLaunches();
    if (pending.empty() && download.bytes == 0U) {
        return {0, {}};
    }
    const std::vector<std::uint32_t> live    = webGpuMemory().liveRanges(); // every queued launch's closure included
    const std::vector<std::uint32_t> ranges  = bridgedRanges(live, webGpuDeviceMemory().liveRanges(), kMaxBridgedGapBytes);
    const std::uint32_t              readEnd = download.bytes == 0U ? 0U : download.source + download.bytes;
    const std::uint32_t              heapEnd = std::max({ranges.empty() ? 0U : ranges.back(), webGpuDeviceMemory().highestEnd(), readEnd});
    FlushBatch                       batch{.launchCount = static_cast<std::uint32_t>(pending.size()), .launchesAddress = wasmAddress(pending.data()), .heapEnd = heapEnd, .rangesAddress = wasmAddress(ranges.data()), .rangeWords = static_cast<std::uint32_t>(ranges.size()), .liveAddress = wasmAddress(live.data()), .liveWords = static_cast<std::uint32_t>(live.size()), .downloadDestination = download.destination, .downloadSource = download.source, .downloadBytes = download.bytes};
    std::array<char, 512>            message{};
    auto                             task = [&](em_proxying_ctx* ctx) { gr4WebGpuLaunch(ctx, &batch, message.data(), static_cast<int>(message.size())); };
    onMainThread(task);
    std::ranges::for_each(pending, releaseScratch);
    pending.clear();                                          // keeps its capacity for the next batch
    launchesQueued().store(false, std::memory_order_release); // before the held releases, so that none is held again
    releaseHeld();
    return {batch.status, std::string(message.data())};
}

/// a transfer with `gpuOperations()` already held, after the queued launches, which it would otherwise overtake
std::expected<void, std::string> transferLocked(std::uint32_t destination, std::uint32_t source, std::uint32_t bytes, int sides) {
    if (const auto [status, message] = flushLocked(); status != 0) {
        return std::unexpected(std::format("the launches before a transfer failed: {}", message));
    }
    int  status = -1;
    auto task   = [&](em_proxying_ctx* ctx) { gr4WebGpuTransfer(ctx, destination, source, bytes, sides, &status); };
    onMainThread(task);
    if (status != 0) {
        return std::unexpected("a transfer to or from device-only memory could not reach the WebGPU device");
    }
    return {};
}

/// the highest wasm address the GPU buffer may mirror: the adapter's storage binding and buffer limits, set once at
/// initialisation
std::uint32_t& mirrorLimit() {
    static std::uint32_t limit = 0U;
    return limit;
}

/// per kernel asked for: its stack bytes per work item, -1 without a pipeline (none in the binary, or one the adapter
/// could not build, `refusal` says why), and whether it ran once on the adapter, so that its pipeline exists and a launch
/// may be queued
struct KnownKernels {
    struct Known {
        int         stackBytes = 0;
        bool        built      = false;
        std::string refusal{};
    };
    std::mutex                               mutex;
    std::unordered_map<std::uint32_t, Known> byTableSlot;
};

KnownKernels& knownKernels() {
    static KnownKernels known;
    return known;
}

KnownKernels::Known knownKernel(std::uint32_t kernel) {
    KnownKernels&    known = knownKernels();
    std::scoped_lock lock(known.mutex);
    const auto [it, inserted] = known.byTableSlot.try_emplace(kernel);
    if (inserted) {
        std::array<char, 512> refusal{};
        it->second.stackBytes = gr4WebGpuKernelStackBytes(kernel, refusal.data(), static_cast<int>(refusal.size()));
        it->second.refusal    = refusal.data();
    }
    return it->second;
}

} // namespace

WebGpuAdapter webGpuInitialise() {
    if (emscripten_is_main_runtime_thread()) {
        gr::log::warning("webgpu: registerWebGpuRuntime() on the browser main thread, which owns the GPU device and cannot wait for it; call it from a pthread, or run main() on one (-sPROXY_TO_PTHREAD)");
        return {.kind = WebGpuAdapter::Kind::none, .description = "registerWebGpuRuntime() was called on the browser main thread"};
    }
    int                   kind = 0;
    std::array<char, 256> name{};
    auto                  task = [&](em_proxying_ctx* ctx) { gr4WebGpuInit(ctx, &kind, name.data(), static_cast<int>(name.size()), wasmAddress(&__data_end), &mirrorLimit()); };
    onMainThread(task);
    return WebGpuAdapter{.kind = static_cast<WebGpuAdapter::Kind>(kind), .description = std::string(name.data())};
}

std::expected<std::optional<std::string>, std::string> webGpuLaunch(WebGpuKernelEntry entry, const void* closure, std::size_t closureBytes, std::size_t count, bool await) {
    assert(count > 0UZ);
    const auto kernel = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(entry)); // its table slot

    // a kernel the GPU cannot run runs here, after the launches queued before it; device-only memory is the GPU's copy,
    // so it comes to wasm memory first and goes back after, under the lock, so that no launch on another thread changes it
    // in between. The browser main thread can neither wait for the GPU nor take the lock a waiting thread holds.
    const auto runOnHost = [&](std::string reason) -> std::expected<std::optional<std::string>, std::string> {
        const auto runHere = [&] {
            for (std::uint32_t index = 0U; index < count; ++index) {
                entry(closure, index);
            }
            return std::optional<std::string>(std::move(reason));
        };
        if (emscripten_is_main_runtime_thread()) {
            if (launchesQueued().load(std::memory_order_acquire) || webGpuDeviceMemory().highestEnd() != 0U) {
                return std::unexpected(std::format("{}, but the browser main thread can neither wait for queued launches nor reach device-only memory", reason));
            }
            return runHere();
        }
        std::scoped_lock lock(gpuOperations());
        if (const auto [status, message] = flushLocked(); status != 0) {
            return std::unexpected(std::format("the launches queued before a CPU fallback failed: {}", message));
        }
        const std::vector<std::uint32_t> deviceOnly     = webGpuDeviceMemory().liveRanges();
        const auto                       moveDeviceOnly = [&](int sides) -> std::expected<void, std::string> {
            for (std::size_t r = 0UZ; r < deviceOnly.size(); r += 2UZ) {
                if (auto moved = transferLocked(deviceOnly[r], deviceOnly[r], deviceOnly[r + 1UZ] - deviceOnly[r], sides); !moved) {
                    return moved;
                }
            }
            return {};
        };
        if (auto read = moveDeviceOnly(kSourceDeviceOnly); !read) {
            return std::unexpected(std::format("{}, but device-only memory could not be read back: {}", reason, read.error()));
        }
        auto ran = runHere();
        if (auto written = moveDeviceOnly(kDestinationDeviceOnly); !written) {
            return std::unexpected(std::format("{}, but device-only memory could not be written back: {}", *ran, written.error()));
        }
        return ran;
    };

    if (emscripten_is_main_runtime_thread()) {
        return runOnHost("ran on the CPU: it was launched from the browser main thread, which cannot wait for the GPU");
    }
    const KnownKernels::Known known = knownKernel(kernel);
    if (known.stackBytes < 0) {
        return runOnHost(std::format("ran on the CPU: {}", known.refusal));
    }
    const auto stackBytesPerItem = static_cast<std::size_t>(known.stackBytes);
    if (stackBytesPerItem != 0UZ && count > std::numeric_limits<std::uint32_t>::max() / stackBytesPerItem) {
        return runOnHost(std::format("ran on the CPU: {} work items of {} stack bytes exceed wasm memory", count, stackBytesPerItem));
    }
    void* const closureCopy = webGpuMemory().tryAllocate(closureBytes, WebGpuMemory::kAlign); // the caller's stack is not mirrored
    void* const stacks      = closureCopy == nullptr || stackBytesPerItem == 0UZ ? closureCopy : webGpuDeviceMemory().tryAllocate(stackBytesPerItem * count, WebGpuMemory::kAlign);
    if (closureCopy == nullptr || stacks == nullptr) {
        if (closureCopy != nullptr) {
            webGpuMemory().release(closureCopy, WebGpuMemory::kAlign);
        }
        return runOnHost(std::format("ran on the CPU: no room for its closure and {} bytes of stack", stackBytesPerItem * count));
    }
    std::memcpy(closureCopy, closure, closureBytes);
    const QueuedLaunch launch{.kernel = kernel, .closure = wasmAddress(closureCopy), .count = static_cast<std::uint32_t>(count), .stackTop = wasmAddress(stacks) + static_cast<std::uint32_t>(stackBytesPerItem * count), .stackBytes = static_cast<std::uint32_t>(stackBytesPerItem)};
    if (const std::uint32_t reach = std::max(webGpuMemory().highestEnd(), webGpuDeviceMemory().highestEnd()); reach > mirrorLimit()) {
        releaseScratch(launch);
        return runOnHost(std::format("ran on the CPU: wasm memory in use reaches {} MiB, beyond the {} MiB the adapter can bind", reach >> 20U, mirrorLimit() >> 20U));
    }
    const bool   built  = known.built;
    std::int32_t status = 0;
    std::string  message;
    {
        std::scoped_lock lock(gpuOperations());
        if (!built) { // a first launch builds its pipeline and may be refused, so it runs alone
            std::tie(status, message) = flushLocked();
        }
        if (status == 0) {
            queuedLaunches().push_back(launch);
            launchesQueued().store(true, std::memory_order_release);
            if (!await && built && queuedLaunches().size() < kMaxQueuedLaunches) {
                return std::nullopt; // in order with every later launch and transfer, which flush it first
            }
            std::tie(status, message) = flushLocked();
        } else {
            releaseScratch(launch);
        }
    }
    if (status == 0 && !built) {
        KnownKernels&    kernels = knownKernels();
        std::scoped_lock lock(kernels.mutex);
        kernels.byTableSlot[kernel].built = true;
    }
    if (status == kRefusedByAdapter) {
        {
            KnownKernels&    kernels = knownKernels();
            std::scoped_lock lock(kernels.mutex);
            kernels.byTableSlot[kernel].stackBytes = -1; // later launches go to the CPU without a trip to the main thread
            kernels.byTableSlot[kernel].refusal    = message;
        }
        return runOnHost(std::format("ran on the CPU: {}", message));
    }
    if (status != 0) {
        return std::unexpected(std::format("webgpu: {}", message));
    }
    return std::nullopt;
}

std::expected<bool, std::string> webGpuFlush() {
    if (emscripten_is_main_runtime_thread()) {
        if (launchesQueued().load(std::memory_order_acquire)) {
            return std::unexpected("webgpu: the browser main thread cannot wait for the launches queued on the GPU");
        }
        return false;
    }
    std::scoped_lock lock(gpuOperations());
    if (std::string failure = std::exchange(unreportedFailure(), {}); !failure.empty()) {
        return std::unexpected(std::move(failure));
    }
    if (queuedLaunches().empty()) {
        return false;
    }
    if (const auto [status, message] = flushLocked(); status != 0) {
        return std::unexpected(std::format("webgpu: queued launches failed: {}", message));
    }
    return true;
}

bool webGpuReleaseNow(WebGpuMemory& memory, void* p, std::size_t align) {
    if (!launchesQueued().load(std::memory_order_acquire)) {
        return true;
    }
    if (emscripten_is_main_runtime_thread()) {
        {
            std::scoped_lock lock(heldReleases().mutex);
            heldReleases().releases.push_back({.memory = &memory, .p = p, .align = align});
        }
        if (!launchesQueued().load(std::memory_order_acquire)) { // the flush that would release it ran meanwhile
            releaseHeld();
        }
        return false;
    }
    std::scoped_lock lock(gpuOperations());
    if (const auto [status, message] = flushLocked(); status != 0 && unreportedFailure().empty()) {
        unreportedFailure() = std::format("webgpu: queued launches failed: {}", message);
    }
    return true;
}

std::expected<void, std::string> webGpuTransfer(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0UZ) {
        return {};
    }
    if (emscripten_is_main_runtime_thread()) {
        return std::unexpected("a transfer to or from device-only memory was asked of the browser main thread, which cannot wait for the WebGPU device");
    }
    const int        sides = (webGpuDeviceMemory().contains(destination) ? kDestinationDeviceOnly : 0) | (webGpuDeviceMemory().contains(source) ? kSourceDeviceOnly : 0);
    const auto       to    = wasmAddress(destination);
    const auto       from  = wasmAddress(source);
    std::scoped_lock lock(gpuOperations());
    if (sides == (kDestinationDeviceOnly | kSourceDeviceOnly) && (to | from | static_cast<std::uint32_t>(bytes)) % 4U == 0U) { // a device-only ring's mirror copy, on every publish
        queuedLaunches().push_back(QueuedLaunch{.kernel = kQueuedCopy, .closure = from, .count = static_cast<std::uint32_t>(bytes), .stackTop = to, .stackBytes = 0U});
        launchesQueued().store(true, std::memory_order_release);
        if (queuedLaunches().size() < kMaxQueuedLaunches) {
            return {};
        }
        if (const auto [status, message] = flushLocked(); status != 0) {
            return std::unexpected(std::format("webgpu: queued launches failed: {}", message));
        }
        return {};
    }
    if (sides == kSourceDeviceOnly) { // a read-back rides on the flush of the queued launches: one round trip for both
        if (const auto [status, message] = flushLocked({.destination = to, .source = from, .bytes = static_cast<std::uint32_t>(bytes)}); status != 0) {
            return std::unexpected(std::format("webgpu: a read-back of device-only memory failed: {}", message));
        }
        return {};
    }
    return transferLocked(to, from, static_cast<std::uint32_t>(bytes), sides);
}

} // namespace gr::device
