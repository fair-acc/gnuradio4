#ifndef GNURADIO_ALGORITHM_SYCL_FFT_HPP
#define GNURADIO_ALGORITHM_SYCL_FFT_HPP

#include <cassert>
#include <cmath>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::device {

/// @brief GPU kernel tuning parameters (device-dependent defaults)
namespace config {

struct FftKernelParams {
    std::size_t workgroupSize = 1024; // 1024 (NVIDIA/Intel), 256 (AMD RDNA)
};

} // namespace config

/**
 * @brief Unified FFT: SimdFFT on CPU, Van Loan Stockham auto-sort on GPU (via SYCL).
 *
 * CPU: delegates to gr::algorithm::FFT (mixed-radix SIMD, {2,3,4,5}).
 * GPU: Van Loan Stockham radix-2 with fused local-memory stages, uint32_t bit-shift
 * index arithmetic, sycl::event chaining. Forward + inverse twiddle tables pre-computed.
 * Transforms are unnormalized: inverse(forward(x)) = N*x.
 *
 * @example
 * gr::device::DeviceContextCpu ctx;
 * gr::device::SyclFFT fft;
 * fft.init(ctx, 1024);
 * fft.forward(ctx, std::span{data, 1024});
 * fft.forwardBatch(ctx, std::span{data, 16*1024}, 1024);
 * fft.inverse(ctx, std::span{data, 1024});
 */
struct SyclFFT {
    using C    = gr::complex<float>;
    using StdC = std::complex<float>;

    config::FftKernelParams _params;

    gr::algorithm::FFT<StdC, StdC> _simdFft;
    std::vector<StdC>              _simdFftOutput;

    std::vector<C> _twiddleHost;
    C*             _twiddleFwdDevice = nullptr; // N/2 entries (base table, forward)
    C*             _twiddleInvDevice = nullptr; // N/2 entries (base table, inverse)
    C*             _pingPongDevice   = nullptr;
    std::size_t    _pingPongCapacity = 0;
    std::size_t    _fftSize          = 0;
    DeviceContext* _ctx              = nullptr;

    void init(DeviceContext& ctx, std::size_t fftSize, config::FftKernelParams params = {}) {
        if (_fftSize == fftSize && _ctx == &ctx && _params.workgroupSize == params.workgroupSize) {
            return;
        }
        cleanup();

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

        _twiddleFwdDevice = ctx.allocateDevice<C>(fftSize / 2);
        _twiddleInvDevice = ctx.allocateDevice<C>(fftSize / 2);
        ctx.copyHostToDevice(_twiddleHost.data(), _twiddleFwdDevice, fftSize / 2);
        ctx.copyHostToDevice(twiddleInvHost.data(), _twiddleInvDevice, fftSize / 2);

        _simdFftOutput.resize(fftSize);
        _pingPongCapacity = 0;
    }

    void cleanup() {
        if (_ctx) {
            auto dealloc = [this](C*& p) {
                if (p) {
                    _ctx->deallocate(p);
                    p = nullptr;
                }
            };
            dealloc(_twiddleFwdDevice);
            dealloc(_twiddleInvDevice);
            dealloc(_pingPongDevice);
        }
        _fftSize = 0;
    }

    ~SyclFFT() { cleanup(); }

    SyclFFT()                          = default;
    SyclFFT(const SyclFFT&)            = delete;
    SyclFFT& operator=(const SyclFFT&) = delete;

    SyclFFT(SyclFFT&& o) noexcept : _params(o._params), _simdFft(std::move(o._simdFft)), _simdFftOutput(std::move(o._simdFftOutput)), _twiddleHost(std::move(o._twiddleHost)), _twiddleFwdDevice(std::exchange(o._twiddleFwdDevice, nullptr)), _twiddleInvDevice(std::exchange(o._twiddleInvDevice, nullptr)), _pingPongDevice(std::exchange(o._pingPongDevice, nullptr)), _pingPongCapacity(o._pingPongCapacity), _fftSize(std::exchange(o._fftSize, 0)), _ctx(std::exchange(o._ctx, nullptr)) {}

    SyclFFT& operator=(SyclFFT&& o) noexcept {
        if (this != &o) {
            cleanup();
            new (this) SyclFFT(std::move(o));
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
            const std::size_t Ls2      = std::size_t(1) << stage;
            const std::size_t Ls       = Ls2 << 1;
            const std::size_t twStride = N / Ls;

            for (std::size_t j = 0; j < halfN; ++j) {
                const std::size_t group   = j / Ls2;
                const std::size_t k       = j % Ls2;
                const auto        a       = src[j];
                const auto        b       = src[j + halfN];
                const auto        w       = _twiddleHost[k * twStride];
                const auto        wb      = C{w.re * b.re - w.im * b.im, w.re * b.im + w.im * b.re};
                dst[group * Ls + k]       = a + wb;
                dst[group * Ls + k + Ls2] = a - wb;
            }
            std::swap(src, dst);
        }

        if (src != data.data()) {
            std::ranges::copy_n(src, static_cast<std::ptrdiff_t>(N), data.data());
        }
    }

    void forward([[maybe_unused]] DeviceContext& ctx, std::span<C> data) {
        const auto N = data.size();
        assert(N == _fftSize && "call init() first or fftSize mismatch");
#if GR_DEVICE_HAS_SYCL_IMPL
        if (auto* sycl = syclCtx(ctx)) {
            ensurePingPongCapacity(ctx, N);
            stockhamGpu(*sycl->queue, data.data(), static_cast<uint32_t>(N), 1, _twiddleFwdDevice);
            return;
        }
#endif
        auto* stdData = reinterpret_cast<StdC*>(data.data());
        _simdFft.compute(std::span<const StdC>(stdData, N), _simdFftOutput);
        std::ranges::copy(_simdFftOutput, stdData);
    }

    void inverse([[maybe_unused]] DeviceContext& ctx, std::span<C> data) {
        const auto N = data.size();
        assert(N == _fftSize);
#if GR_DEVICE_HAS_SYCL_IMPL
        if (auto* sycl = syclCtx(ctx)) {
            ensurePingPongCapacity(ctx, N);
            stockhamGpu(*sycl->queue, data.data(), static_cast<uint32_t>(N), 1, _twiddleInvDevice);
            float invN = 1.f / static_cast<float>(N);
            sycl->parallelFor(N, [d = data.data(), invN](std::size_t i) { d[i] = d[i] * invN; });
            return;
        }
#endif
        auto* stdData = reinterpret_cast<StdC*>(data.data());
        for (std::size_t i = 0; i < N; ++i) {
            stdData[i] = std::conj(stdData[i]);
        }
        _simdFft.compute(std::span<const StdC>(stdData, N), _simdFftOutput);
        float invN = 1.f / static_cast<float>(N);
        for (std::size_t i = 0; i < N; ++i) {
            stdData[i] = std::conj(_simdFftOutput[i]) * invN;
        }
    }

    void forwardBatch([[maybe_unused]] DeviceContext& ctx, std::span<C> data, std::size_t fftSize) {
        assert(fftSize == _fftSize);
        const std::size_t nBatches = data.size() / fftSize;
#if GR_DEVICE_HAS_SYCL_IMPL
        if (auto* sycl = syclCtx(ctx)) {
            ensurePingPongCapacity(ctx, data.size());
            stockhamGpu(*sycl->queue, data.data(), static_cast<uint32_t>(fftSize), static_cast<uint32_t>(nBatches), _twiddleFwdDevice);
            return;
        }
#endif
        auto* stdData = reinterpret_cast<StdC*>(data.data());
        for (std::size_t b = 0; b < nBatches; ++b) {
            auto* batch = stdData + b * fftSize;
            _simdFft.compute(std::span<const StdC>(batch, fftSize), _simdFftOutput);
            std::ranges::copy(_simdFftOutput, batch);
        }
    }

private:
#if GR_DEVICE_HAS_SYCL_IMPL
    static DeviceContextSycl* syclCtx(DeviceContext& ctx) { return ctx.backend() == DeviceBackend::SYCL ? static_cast<DeviceContextSycl*>(&ctx) : nullptr; }
#endif

    void ensurePingPongCapacity(DeviceContext& ctx, std::size_t needed) {
        if (_pingPongCapacity >= needed) {
            return;
        }
        if (_pingPongDevice) {
            ctx.deallocate(_pingPongDevice);
        }
        _pingPongDevice   = ctx.allocateDevice<C>(needed);
        _pingPongCapacity = needed;
    }

    // ── GPU path: Van Loan Stockham auto-sort (radix-2, no bit-reversal) ──

#if GR_DEVICE_HAS_SYCL_IMPL
    void stockhamGpu(sycl::queue& q, C* data, uint32_t N, uint32_t nBatches, const C* tw) {
        const uint32_t nStages   = static_cast<uint32_t>(std::countr_zero(N));
        const uint32_t wgSize    = static_cast<uint32_t>(std::min(_params.workgroupSize, static_cast<std::size_t>(N / 2)));
        const uint32_t localBits = static_cast<uint32_t>(std::countr_zero(2u * wgSize));
        const uint32_t sLocal    = (nStages > localBits) ? nStages - localBits : 0u;

        auto*       src = data;
        auto*       dst = _pingPongDevice;
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

    // Fused local-memory stages: Van Loan radix-2 in shared memory with ping-pong
    static sycl::event submitLocalStages(sycl::queue& q, const C* src, C* dst, uint32_t N, uint32_t nBatches, const C* tw, uint32_t stageStart, uint32_t nStages, uint32_t wgSize, sycl::event dep) {
        const uint32_t blockSizeLog   = static_cast<uint32_t>(std::countr_zero(2u * wgSize));
        const uint32_t blockSize      = uint32_t(1) << blockSizeLog;
        const uint32_t nLog           = nStages;
        const uint32_t blocksPerBatch = N >> blockSizeLog;
        const uint32_t totalBlocks    = nBatches * blocksPerBatch;

        return q.submit([=](sycl::handler& h) {
            h.depends_on(dep);
            sycl::local_accessor<C, 1> localA(sycl::range<1>{blockSize}, h);
            sycl::local_accessor<C, 1> localB(sycl::range<1>{blockSize}, h);

            h.parallel_for(sycl::nd_range<1>{static_cast<std::size_t>(totalBlocks) * wgSize, static_cast<std::size_t>(wgSize)}, [=](sycl::nd_item<1> item) {
                const uint32_t lid            = static_cast<uint32_t>(item.get_local_id(0));
                const uint32_t globalBlockIdx = static_cast<uint32_t>(item.get_group_linear_id());
                const uint32_t batch          = globalBlockIdx / blocksPerBatch;
                const uint32_t blockIdx       = globalBlockIdx - batch * blocksPerBatch;
                const uint32_t base           = (batch << nLog) + (blockIdx << blockSizeLog);

                // load global → local
                localA[lid]          = src[base + lid];
                localA[lid + wgSize] = src[base + lid + wgSize];
                sycl::group_barrier(item.get_group());

                auto* ping = &localA[0];
                auto* pong = &localB[0];

                // Van Loan butterfly stages in local memory
                for (uint32_t stage = stageStart; stage < nStages; ++stage) {
                    const uint32_t Ls2      = uint32_t(1) << stage;
                    const uint32_t Ls       = Ls2 << 1;
                    const uint32_t Ls2Mask  = Ls2 - 1;
                    const uint32_t twStride = N / Ls;

                    const uint32_t group = lid >> stage;
                    const uint32_t k     = lid & Ls2Mask;
                    const uint32_t sLo   = lid;
                    const uint32_t sHi   = sLo + wgSize;
                    const uint32_t dLo   = group * Ls + k;
                    const uint32_t dHi   = dLo + Ls2;

                    if (dHi < blockSize) {
                        auto a    = ping[sLo];
                        auto b    = ping[sHi];
                        auto w    = tw[k * twStride];
                        auto wb   = C{w.re * b.re - w.im * b.im, w.re * b.im + w.im * b.re};
                        pong[dLo] = a + wb;
                        pong[dHi] = a - wb;
                    }
                    sycl::group_barrier(item.get_group());
                    auto* tmp = ping;
                    ping      = pong;
                    pong      = tmp;
                }

                // write local → global
                dst[base + lid]          = ping[lid];
                dst[base + lid + wgSize] = ping[lid + wgSize];
            });
        });
    }
#endif
};

} // namespace gr::device

#endif // GNURADIO_ALGORITHM_SYCL_FFT_HPP
