#ifndef GNURADIO_DEVICE_FFT_HPP
#define GNURADIO_DEVICE_FFT_HPP

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/device/BackendPolicy.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::device {

/// GPU kernel tuning parameters (device-dependent defaults)
namespace config {

struct FftKernelParams {
    std::size_t workgroupSize = 1024; // 1024 (NVIDIA/Intel), 256 (AMD RDNA)
};

} // namespace config

/**
 * Unified FFT: SimdFFT on the host, Van Loan Stockham auto-sort on the GPU, radix-2 with pre-computed forward/inverse twiddle tables.
 * SYCL fuses the trailing stages in local memory; WebGPU runs each stage as one `parallelFor` on data in GPU-only memory.
 * The forward transform is unnormalised and the inverse divides by N, so inverse(forward(x)) = x.
 */
struct DeviceFFT {
    using C    = gr::complex<float>;
    using StdC = std::complex<float>;

    config::FftKernelParams _params;

    gr::algorithm::FFT<StdC, StdC> _simdFft;
    std::vector<StdC>              _simdFftOutput;

    std::vector<C> _twiddleHost;
    DeviceBuffer   _twiddleFwdDevice{}; // N/2 entries (base table, forward)
    DeviceBuffer   _twiddleInvDevice{}; // N/2 entries (base table, inverse)
    DeviceBuffer   _pingPongDevice{};
    std::size_t    _pingPongCapacity = 0;
    std::size_t    _fftSize          = 0;
    DeviceContext* _ctx              = nullptr;

    void init(DeviceContext& ctx, std::size_t fftSize, config::FftKernelParams params = {}) {
        if (_fftSize == fftSize && _ctx == &ctx && _params.workgroupSize == params.workgroupSize) {
            return;
        }
        cleanup();

        // the Stockham decomposition derives its stage count as log2(N) and its local stages from the work-group
        // size; anything else silently transforms a prefix and returns the rest unchanged, so it is refused here
        assert(fftSize >= 2UZ && std::has_single_bit(fftSize) && "DeviceFFT needs a power-of-two size of at least two");
        assert(params.workgroupSize >= 1UZ && std::has_single_bit(params.workgroupSize) && "DeviceFFT needs a power-of-two work-group size");

        _ctx     = &ctx;
        _fftSize = fftSize;
        _params  = params;

        _twiddleHost.resize(fftSize / 2);
        const float theta = -2.f * std::numbers::pi_v<float> / static_cast<float>(fftSize);
        for (std::size_t k = 0; k < fftSize / 2; ++k) {
            _twiddleHost[k] = {std::cos(theta * static_cast<float>(k)), std::sin(theta * static_cast<float>(k))};
        }

        std::vector<C> twiddleInvHost(fftSize / 2);
        for (std::size_t k = 0; k < fftSize / 2; ++k) {
            twiddleInvHost[k] = gr::conj(_twiddleHost[k]);
        }

        // the device twiddle tables are read by the SYCL kernel and the WebGPU stages, so other contexts skip them
#if GR_DEVICE_HAS_SYCL
        const bool deviceTwiddles = syclCtx(ctx) != nullptr || isWebGpu(ctx);
#else
        const bool deviceTwiddles = isWebGpu(ctx);
#endif
        if (deviceTwiddles) {
            _twiddleFwdDevice = ctx.allocateDevice<C>(fftSize / 2);
            _twiddleInvDevice = ctx.allocateDevice<C>(fftSize / 2);
            ctx.copyHostToDevice(_twiddleHost.data(), _twiddleFwdDevice, fftSize / 2);
            ctx.copyHostToDevice(twiddleInvHost.data(), _twiddleInvDevice, fftSize / 2);
        }

        _simdFftOutput.resize(fftSize);
        _pingPongCapacity = 0;
    }

    void cleanup() {
        if (_ctx) {
            auto dealloc = [this](DeviceBuffer& buf) {
                if (buf) {
                    _ctx->deallocate(buf);
                    buf = DeviceBuffer{};
                }
            };
            dealloc(_twiddleFwdDevice);
            dealloc(_twiddleInvDevice);
            dealloc(_pingPongDevice);
        }
        _fftSize = 0;
    }

    ~DeviceFFT() { cleanup(); }

    DeviceFFT()                            = default;
    DeviceFFT(const DeviceFFT&)            = delete;
    DeviceFFT& operator=(const DeviceFFT&) = delete;

    DeviceFFT(DeviceFFT&& o) noexcept : _params(o._params), _simdFft(std::move(o._simdFft)), _simdFftOutput(std::move(o._simdFftOutput)), _twiddleHost(std::move(o._twiddleHost)), _twiddleFwdDevice(std::exchange(o._twiddleFwdDevice, DeviceBuffer{})), _twiddleInvDevice(std::exchange(o._twiddleInvDevice, DeviceBuffer{})), _pingPongDevice(std::exchange(o._pingPongDevice, DeviceBuffer{})), _pingPongCapacity(o._pingPongCapacity), _fftSize(std::exchange(o._fftSize, 0)), _ctx(std::exchange(o._ctx, nullptr)) {}

    DeviceFFT& operator=(DeviceFFT&& o) noexcept {
        if (this != &o) {
            cleanup(); // frees the device buffers; the host members below are move-assigned, not leaked
            _params           = o._params;
            _simdFft          = std::move(o._simdFft);
            _simdFftOutput    = std::move(o._simdFftOutput);
            _twiddleHost      = std::move(o._twiddleHost);
            _twiddleFwdDevice = std::exchange(o._twiddleFwdDevice, DeviceBuffer{});
            _twiddleInvDevice = std::exchange(o._twiddleInvDevice, DeviceBuffer{});
            _pingPongDevice   = std::exchange(o._pingPongDevice, DeviceBuffer{});
            _pingPongCapacity = std::exchange(o._pingPongCapacity, 0UZ);
            _fftSize          = std::exchange(o._fftSize, 0UZ);
            _ctx              = std::exchange(o._ctx, nullptr);
        }
        return *this;
    }

    // Van Loan Stockham radix-2 on CPU — tests the same algorithm as the GPU kernels
    void forwardStockhamCpu(std::span<C> data) {
        const auto N = data.size();
        assert(N == _fftSize && "call init() first or fftSize mismatch");
        const std::size_t nStages = static_cast<std::size_t>(std::countr_zero(N));
        const std::size_t halfN   = N / 2;

        std::vector<C>      tmp(N);
        gr::complex<float>* src = data.data();
        gr::complex<float>* dst = tmp.data();

        for (std::size_t stage = 0; stage < nStages; ++stage) {
            for (std::size_t j = 0; j < halfN; ++j) {
                butterflies<2UZ>(src, dst, _twiddleHost.data(), nStages, stage, j);
            }
            std::swap(src, dst);
        }

        if (src != data.data()) {
            std::ranges::copy_n(src, static_cast<std::ptrdiff_t>(N), data.data());
        }
    }

    void forward(DeviceContext& ctx, std::span<C> data) { forwardBatch(ctx, data, data.size()); }
    void inverse(DeviceContext& ctx, std::span<C> data) { inverseBatch(ctx, data, data.size()); }

    void inverseBatch(DeviceContext& ctx, std::span<C> data, std::size_t fftSize) {
        assert(fftSize == _fftSize);
        const std::size_t nBatches = data.size() / fftSize;
#if GR_DEVICE_HAS_SYCL
        if (auto* sycl = syclCtx(ctx)) {
            ensurePingPongCapacity(ctx, data.size());
            stockhamGpu(*sycl->queue, data.data(), static_cast<uint32_t>(fftSize), static_cast<uint32_t>(nBatches), _twiddleInvDevice.devicePointer<C>());
            const float invN = 1.f / static_cast<float>(fftSize);
            sycl->parallelFor(data.size(), [d = data.data(), invN](std::size_t i) { d[i] = d[i] * invN; });
            return;
        }
#endif
        if (runsStages(ctx, data)) {
            stockhamStages(ctx, data, fftSize, _twiddleInvDevice.devicePointer<C>());
            scaleOnDevice(ctx, data.first(nBatches * fftSize), 1.f / static_cast<float>(fftSize));
            return;
        }
        ctx.wait(); // a launch queued before may still write `data`
        auto*       stdData = reinterpret_cast<StdC*>(data.data());
        const float invN    = 1.f / static_cast<float>(fftSize);
        for (std::size_t b = 0; b < nBatches; ++b) {
            auto* batch = stdData + b * fftSize;
            for (std::size_t i = 0; i < fftSize; ++i) {
                batch[i] = std::conj(batch[i]);
            }
            _simdFft.compute(std::span<const StdC>(batch, fftSize), _simdFftOutput);
            for (std::size_t i = 0; i < fftSize; ++i) {
                batch[i] = std::conj(_simdFftOutput[i]) * invN;
            }
        }
    }

    void forwardBatch(DeviceContext& ctx, std::span<C> data, std::size_t fftSize) {
        assert(fftSize == _fftSize);
        const std::size_t nBatches = data.size() / fftSize;
#if GR_DEVICE_HAS_SYCL
        if (auto* sycl = syclCtx(ctx)) {
            ensurePingPongCapacity(ctx, data.size());
            stockhamGpu(*sycl->queue, data.data(), static_cast<uint32_t>(fftSize), static_cast<uint32_t>(nBatches), _twiddleFwdDevice.devicePointer<C>());
            return;
        }
#endif
        if (runsStages(ctx, data)) {
            stockhamStages(ctx, data, fftSize, _twiddleFwdDevice.devicePointer<C>());
            return;
        }
        ctx.wait(); // a launch queued before may still write `data`
        auto* stdData = reinterpret_cast<StdC*>(data.data());
        for (std::size_t b = 0; b < nBatches; ++b) {
            auto* batch = stdData + b * fftSize;
            _simdFft.compute(std::span<const StdC>(batch, fftSize), _simdFftOutput);
            std::ranges::copy(_simdFftOutput, batch);
        }
    }

private:
#if GR_DEVICE_HAS_SYCL
    static DeviceContextSycl* syclCtx(DeviceContext& ctx) { return gr::backendCast<DeviceContextSycl>(ctx); }
#endif

    /// item `j` of log2(R) consecutive radix-2 Stockham stages, the first with half-span 2^log2m, from `source` to `target`
    /// (one batch of N = 2^log2N): the R values `source[j + p·N/R]` pass through those stages in registers, so the
    /// stages cost one pass over memory instead of log2(R); R = 2 is one radix-2 butterfly
    template<std::size_t R>
    static void butterflies(const C* source, C* target, const C* twiddle, std::size_t log2N, std::size_t log2m, std::size_t j) noexcept {
        constexpr std::size_t kLog2R = static_cast<std::size_t>(std::countr_zero(R));
        const std::size_t     k      = j & ((1UZ << log2m) - 1UZ);
        // unrolled at compile time and inlined, so every index into `v` is a constant and `v` stays in registers: on the
        // kernel's stack it would take mirrored memory per work item, which every flush moves both ways
        constexpr auto    kSlots = std::make_index_sequence<R>{};
        const std::size_t stride = 1UZ << (log2N - kLog2R);
        const auto        load   = [&]<std::size_t... p> [[gnu::always_inline]] (std::index_sequence<p...>) noexcept { return std::array<C, R>{source[j + p * stride]...}; };
        std::array<C, R>  v      = load(kSlots);
        const auto        stage  = [&]<std::size_t s> [[gnu::always_inline]] () noexcept {
            std::array<C, R> next;
            [&]<std::size_t... p> [[gnu::always_inline]] (std::index_sequence<p...>) noexcept {
                (
                    [&] [[gnu::always_inline]] () noexcept {
                        constexpr std::size_t kPlus = ((p >> s) << (s + 1UZ)) + (p & ((1UZ << s) - 1UZ)); // slots stay in the order of their positions
                        const C               w     = twiddle[(k + ((p & ((1UZ << s) - 1UZ)) << log2m)) << (log2N - log2m - s - 1UZ)];
                        const C               b     = v[p + R / 2UZ];
                        const C               wb    = C{w.re * b.re - w.im * b.im, w.re * b.im + w.im * b.re};
                        next[kPlus]                 = v[p] + wb;
                        next[kPlus + (1UZ << s)]    = v[p] - wb;
                    }(),
                    ...);
            }(std::make_index_sequence<R / 2UZ>{});
            v = next;
        };
        const auto stages = [&]<std::size_t... s> [[gnu::always_inline]] (std::index_sequence<s...>) noexcept { (stage.template operator()<s>(), ...); };
        stages(std::make_index_sequence<kLog2R>{});
        const std::size_t out   = ((j >> log2m) << (log2m + kLog2R)) + k;
        const auto        store = [&]<std::size_t... p> [[gnu::always_inline]] (std::index_sequence<p...>) noexcept { ((target[out + (p << log2m)] = v[p]), ...); };
        store(kSlots);
    }

    [[nodiscard]] static bool isWebGpu(const DeviceContext& ctx) noexcept { return ctx.backend() == DeviceBackend::WebGPU; }

    /// on WebGPU, data in GPU-only memory: out of the host transform's reach, so the stages run there, one `parallelFor` per
    /// pass of up to three stages; host-visible data stays with SimdFFT, as every dispatch moves each mirrored allocation both ways
    [[nodiscard]] static bool runsStages(const DeviceContext& ctx, std::span<const C> data) noexcept { return isWebGpu(ctx) && ctx.isDeviceOnly(data.data()); }

    /// `forwardStockhamCpu`'s stages over the whole batches of `data`, ping-ponging through device memory. The stages do not
    /// wait for each other: device-only memory reaches the host only through a copy, which runs after them.
    void stockhamStages(DeviceContext& ctx, std::span<C> data, std::size_t N, const C* twiddle) {
        const std::size_t whole = data.size() / N * N;
        ensurePingPongCapacity(ctx, whole);
        if (!_pingPongDevice || twiddle == nullptr) [[unlikely]] {
            gr::log::error("DeviceFFT: no device memory for the staged transform of {} values; they are left as they are", whole);
            return;
        }
        C*                source = data.data();
        C*                target = _pingPongDevice.devicePointer<C>();
        const std::size_t log2N  = static_cast<std::size_t>(std::countr_zero(N));
        for (std::size_t log2m = 0UZ; log2m < log2N;) { // radix 8 while three stages are left, then 4 or 2
            const std::size_t left = log2N - log2m;
            if (left >= 3UZ) {
                stockhamPass<8UZ>(ctx, source, target, twiddle, whole, log2N, log2m);
                log2m += 3UZ;
            } else if (left == 2UZ) {
                stockhamPass<4UZ>(ctx, source, target, twiddle, whole, log2N, log2m);
                log2m += 2UZ;
            } else {
                stockhamPass<2UZ>(ctx, source, target, twiddle, whole, log2N, log2m);
                log2m += 1UZ;
            }
            std::swap(source, target);
        }
        if (source != data.data()) {
            gr::device::parallelFor(ctx, whole, [source, destination = data.data()] GR_DEVICE_LAMBDA(std::size_t i) { destination[i] = source[i]; }, false);
        }
    }

    /// one `parallelFor` of `butterflies<R>` over the `whole` values, every batch of 2^log2N
    template<std::size_t R>
    static void stockhamPass(DeviceContext& ctx, const C* source, C* target, const C* twiddle, std::size_t whole, std::size_t log2N, std::size_t log2m) {
        gr::device::parallelFor(
            ctx, whole / R,
            [source, target, twiddle, log2N, log2m] GR_DEVICE_LAMBDA(std::size_t i) {
                constexpr std::size_t kLog2R = static_cast<std::size_t>(std::countr_zero(R));
                const std::size_t     batch  = (i >> (log2N - kLog2R)) << log2N;
                butterflies<R>(source + batch, target + batch, twiddle, log2N, log2m, i & ((1UZ << (log2N - kLog2R)) - 1UZ));
            },
            false);
    }

    static void scaleOnDevice(DeviceContext& ctx, std::span<C> data, float factor) {
        gr::device::parallelFor(ctx, data.size(), [d = data.data(), factor] GR_DEVICE_LAMBDA(std::size_t i) { d[i] = d[i] * factor; }, false);
    }

    void ensurePingPongCapacity(DeviceContext& ctx, std::size_t needed) {
        if (_pingPongCapacity >= needed) {
            return;
        }
        if (_pingPongDevice) {
            ctx.deallocate(_pingPongDevice);
        }
        _pingPongDevice = ctx.allocateDevice<C>(needed);
        // allocateDevice may answer with an invalid buffer meaning "cannot serve"; latching the capacity anyway
        // would short-circuit every later call and write through a null device pointer
        _pingPongCapacity = _pingPongDevice ? needed : 0UZ;
    }

    // GPU path: Van Loan Stockham auto-sort (radix-2, no bit-reversal)

#if GR_DEVICE_HAS_SYCL
    void stockhamGpu(sycl::queue& q, C* data, uint32_t N, uint32_t nBatches, const C* tw) {
        const uint32_t nStages   = static_cast<uint32_t>(std::countr_zero(N));
        const uint32_t wgSize    = static_cast<uint32_t>(std::min(_params.workgroupSize, static_cast<std::size_t>(N / 2)));
        const uint32_t blockSize = 2u * wgSize;
        // leading stages in global memory down to a block-sized tile, then the trailing log2(blockSize) fused in
        // local memory (sLocal == 0 keeps the whole transform in a single block)
        const uint32_t blockSizeLog = static_cast<uint32_t>(std::countr_zero(blockSize));
        const uint32_t sLocal       = nStages > blockSizeLog ? nStages - blockSizeLog : 0u;

        auto*       src = data;
        auto*       dst = _pingPongDevice.devicePointer<C>();
        sycl::event prev;

        // Phase 1: global memory stages (Van Loan radix-2)
        for (uint32_t stage = 0; stage < sLocal; ++stage) {
            prev = submitVanLoanStage(q, src, dst, N, nBatches, tw, stage, prev);
            std::swap(src, dst);
        }

        // Phase 2: fused local-memory stages
        if (sLocal < nStages) {
            prev = submitLocalStages(q, src, dst, N, nBatches, tw, sLocal, nStages, wgSize, prev);
            std::swap(src, dst);
        }

        prev.wait();
        if (src != data) {
            q.memcpy(data, src, static_cast<std::size_t>(nBatches) * N * sizeof(C)).wait();
        }
    }

    // Van Loan Stockham radix-2: srcLo = j (sequential), dst interleaved
    static sycl::event submitVanLoanStage(sycl::queue& q, const C* src, C* dst, uint32_t N, uint32_t nBatches, const C* tw, uint32_t stage, sycl::event dep) {
        const uint32_t halfN    = N >> 1;
        const uint32_t Ls2      = uint32_t(1) << stage; // half sub-FFT length
        const uint32_t Ls       = Ls2 << 1;             // full sub-FFT length
        const uint32_t Ls2Mask  = Ls2 - 1;
        const uint32_t twStride = N / Ls; // twiddle stride in base table
        const uint32_t nLog     = static_cast<uint32_t>(std::countr_zero(N));

        return q.submit([=](sycl::handler& h) {
            h.depends_on(dep);
            h.parallel_for(sycl::range<1>{static_cast<std::size_t>(nBatches) * halfN}, [=](sycl::id<1> gid) {
                const uint32_t idx   = static_cast<uint32_t>(gid[0]);
                const uint32_t batch = idx / halfN;
                const uint32_t j     = idx - batch * halfN;
                const uint32_t off   = batch << nLog;

                const uint32_t srcLo = off + j;
                const uint32_t srcHi = srcLo + halfN;

                const uint32_t group = j >> stage;
                const uint32_t k     = j & Ls2Mask;
                const uint32_t dstLo = off + group * Ls + k;
                const uint32_t dstHi = dstLo + Ls2;

                auto a  = src[srcLo];
                auto b  = src[srcHi];
                auto w  = tw[k * twStride];
                auto wb = C{w.re * b.re - w.im * b.im, w.re * b.im + w.im * b.re};

                dst[dstLo] = a + wb;
                dst[dstHi] = a - wb;
            });
        });
    }

    // Fused local-memory stages: each group runs one residue class r in [0, 2^g) (g = stageStart) as an
    // M-point (M = N/2^g) Stockham transform in shared memory; the cross-boundary twiddle indexes the base table.
    static sycl::event submitLocalStages(sycl::queue& q, const C* src, C* dst, uint32_t N, uint32_t nBatches, const C* tw, uint32_t stageStart, uint32_t nStages, uint32_t wgSize, sycl::event dep) {
        const uint32_t blockSizeLog   = static_cast<uint32_t>(std::countr_zero(2u * wgSize));
        const uint32_t blockSize      = uint32_t(1) << blockSizeLog;
        const uint32_t nLog           = nStages;
        const uint32_t blocksPerBatch = N >> blockSizeLog; // == 2^g residue classes per batch
        const uint32_t totalBlocks    = nBatches * blocksPerBatch;

        return q.submit([=](sycl::handler& h) {
            h.depends_on(dep);
            sycl::local_accessor<C, 1> localA(sycl::range<1>{blockSize}, h);
            sycl::local_accessor<C, 1> localB(sycl::range<1>{blockSize}, h);

            h.parallel_for(sycl::nd_range<1>{static_cast<std::size_t>(totalBlocks) * wgSize, static_cast<std::size_t>(wgSize)}, [=](sycl::nd_item<1> item) {
                const uint32_t lid            = static_cast<uint32_t>(item.get_local_id(0));
                const uint32_t globalBlockIdx = static_cast<uint32_t>(item.get_group_linear_id());
                const uint32_t batch          = globalBlockIdx / blocksPerBatch;
                const uint32_t r              = globalBlockIdx - batch * blocksPerBatch; // residue class 0..2^g-1
                const uint32_t g              = stageStart;
                const uint32_t nLocal         = nStages - stageStart;
                const uint32_t batchOff       = batch << nLog;

                // strided gather: global → local, stride 2^g within residue class r
                localA[lid]          = src[batchOff + (lid << g) + r];
                localA[lid + wgSize] = src[batchOff + ((lid + wgSize) << g) + r];
                sycl::group_barrier(item.get_group());

                auto* ping = &localA[0];
                auto* pong = &localB[0];

                for (uint32_t sigma = 0; sigma < nLocal; ++sigma) {
                    const uint32_t Ls2     = uint32_t(1) << sigma;
                    const uint32_t kHi     = lid & (Ls2 - 1);
                    const uint32_t group   = lid >> sigma;
                    const uint32_t kGlobal = (kHi << g) + r; // phase index straddling the block boundary
                    const uint32_t twIdx   = kGlobal << (nLocal - sigma - 1);
                    const uint32_t dLo     = group * (Ls2 << 1) + kHi;

                    auto a  = ping[lid];
                    auto b  = ping[lid + wgSize];
                    auto w  = tw[twIdx];
                    auto wb = C{w.re * b.re - w.im * b.im, w.re * b.im + w.im * b.re};

                    pong[dLo]       = a + wb;
                    pong[dLo + Ls2] = a - wb;

                    sycl::group_barrier(item.get_group());
                    auto* tmp = ping;
                    ping      = pong;
                    pong      = tmp;
                }

                // strided scatter: local → global, mirroring the gather
                dst[batchOff + (lid << g) + r]            = ping[lid];
                dst[batchOff + ((lid + wgSize) << g) + r] = ping[lid + wgSize];
            });
        });
    }
#endif
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_FFT_HPP
