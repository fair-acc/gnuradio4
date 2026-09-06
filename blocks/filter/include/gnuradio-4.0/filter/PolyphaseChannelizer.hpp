#ifndef GNURADIO_POLYPHASE_CHANNELIZER_HPP
#define GNURADIO_POLYPHASE_CHANNELIZER_HPP

#include <algorithm>
#include <complex>
#include <memory_resource>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::PolyphaseChannelizer, [T], [ std::complex<float>, std::complex<double> ])

/**
 * Splits one wideband stream into `n_channels` critically sampled channels.
 *
 * The prototype is filtered once at the OUTPUT rate and a transform across the polyphase arms separates the
 * channels, so the cost is one prototype rather than one mixer-and-filter chain per channel.
 *
 * The channel ports are a runtime collection, which is why the device path goes through `processBulk(ctx, ...)`:
 * a kernel cannot be handed a channel count that is only known when the graph is built.
 */
template<typename T>
requires gr::meta::complex_like<T>
struct PolyphaseChannelizer : Block<PolyphaseChannelizer<T>, Resampling<>, Stride<>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using Algorithm   = gr::algorithm::filter::PolyphaseChannelizer<Real>;
    using Description = Doc<R""(splits a wideband stream into `n_channels` critically sampled channels using one polyphase analysis filterbank.

The **flat** form: one bank, and one cross-channel DFT of size `n_channels` per output set, so that term grows as
`n_channels^2`. `HierarchicalPolyphaseChannelizer` splits it into two smaller stages; `HierarchicalIirChannelizer`
trades linear phase for a handful of coefficients per stage. `PolyphaseSynthesizer` is the inverse.

 * f. j. harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase filter banks for wireless
   communications", IEEE Trans. Microw. Theory Techn., vol. 51, no. 4, pp. 1395-1412, 2003.)"">;

    PortIn<T>               in;
    std::vector<PortOut<T>> out;

    Annotated<gr::Size_t, "n_channels", Doc<"channel count; each channel runs at the input rate divided by it">, Visible, Limits<2U, 1024U>> n_channels        = 4U;
    Annotated<gr::Size_t, "n_taps", Doc<"prototype length; longer gives sharper channel edges">, Limits<2U, 8192U>>                          n_taps            = 64U;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"output sets produced per work call">, Limits<1U, 65536U>>                                outputs_per_frame = 64U;

    std::pmr::vector<Real> phases{};       // phase-major prototype, re-seated onto device memory when the block runs there
    std::pmr::vector<T>    _arms{};        // one value per arm, sized in settingsChanged: a kernel cannot allocate
    std::pmr::vector<T>    _twiddles{};    // nChannels entries covering the whole transform, likewise re-seated
    std::pmr::vector<Real> _twiddleReal{}; // the same transform as a planar matrix, for the host's vector path
    std::pmr::vector<Real> _twiddleImag{};
    std::pmr::vector<Real> _armReal{}; // one set's arms and channels, planar, so the transform vectorises
    std::pmr::vector<Real> _armImag{};
    std::pmr::vector<Real> _outReal{};
    std::pmr::vector<Real> _outImag{};

    GR_MAKE_REFLECTABLE(PolyphaseChannelizer, in, out, n_channels, n_taps, outputs_per_frame, phases, _arms, _twiddles, _twiddleReal, _twiddleImag, _armReal, _armImag, _outReal, _outImag);

    /// taps per arm; derived rather than stored, so a retune cannot leave a stale cache behind on a device
    [[nodiscard]] constexpr std::size_t phaseLength() const noexcept { return phases.size() / std::max(std::size_t{1}, static_cast<std::size_t>(n_channels)); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        if (out.size() != nChannels) {
            out.resize(nChannels);
        }

        // at least two taps per arm, or `decompose` normalises each arm to a single unit tap and the bank
        // degenerates into a bare commutator -- every channel would then carry the whole band, aliased
        const std::size_t nTaps      = std::max(static_cast<std::size_t>(n_taps), 2UZ * nChannels);
        const auto        prototype  = Algorithm::designPrototype(nTaps, nChannels);
        const auto        decomposed = Algorithm::decompose(prototype, nChannels);
        phases.assign(decomposed.begin(), decomposed.end());

        const std::size_t phaseLen = Algorithm::phaseLength(prototype.size(), nChannels);
        const std::size_t nSets    = static_cast<std::size_t>(outputs_per_frame);

        this->output_chunk_size = static_cast<gr::Size_t>(nSets);
        this->input_chunk_size  = static_cast<gr::Size_t>(Algorithm::windowLength(nSets, nChannels, phaseLen));
        this->stride            = static_cast<gr::Size_t>(nSets * nChannels); // consecutive windows overlap by the history the arms reach back over
        _arms.assign(nChannels, T{});
        Algorithm::template seatTwiddles<T>(_twiddles, Algorithm::designTwiddles(nChannels));
        Algorithm::designPlanarTwiddles(nChannels, +1, _twiddleReal, _twiddleImag); // empty above kMaxPlanarChannels
        _armReal.assign(nChannels, Real{});
        _armImag.assign(nChannels, Real{});
        _outReal.assign(nChannels, Real{});
        _outImag.assign(nChannels, Real{});
    }

    template<typename TOutputSpan>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, std::span<TOutputSpan>& outputs) noexcept {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        const std::size_t phaseLen  = phaseLength();
        if (nChannels == 0UZ || phaseLen == 0UZ || outputs.empty()) {
            return gr::work::Status::OK;
        }

        const std::span<const T>    window{input.data(), input.size()};
        const std::span<const Real> taps{phases.data(), phases.size()};
        std::size_t                 nSets = outputs[0].size();
        for (const auto& channel : outputs) { // a collection's ports need not offer the same room
            nSets = std::min(nSets, channel.size());
        }

        if (_arms.size() < nChannels) {
            _arms.assign(nChannels, T{});
        }
        // the planar path is the whole reason the transform is not the bottleneck it used to be; it exists
        // only up to `kMaxPlanarChannels`, above which the matrix costs more cache than the vectors win
        const bool planar = _twiddleReal.size() == nChannels * nChannels && _armReal.size() >= nChannels;
        if (planar) {
            for (std::size_t m = 0UZ; m < nSets; ++m) {
                for (std::size_t p = 0UZ; p < nChannels; ++p) {
                    const T arm = Algorithm::template armOf<T>(window, taps, phaseLen, nChannels, m, p);
                    _armReal[p] = static_cast<Real>(arm.real());
                    _armImag[p] = static_cast<Real>(arm.imag());
                }
                Algorithm::transformSetPlanar(_armReal.data(), _armImag.data(), _twiddleReal.data(), _twiddleImag.data(), nChannels, _outReal.data(), _outImag.data());
                for (std::size_t k = 0UZ; k < nChannels; ++k) {
                    outputs[k][m] = T{_outReal[k], _outImag[k]};
                }
            }
            return gr::work::Status::OK;
        }

        for (std::size_t m = 0UZ; m < nSets; ++m) {
            for (std::size_t p = 0UZ; p < nChannels; ++p) {
                _arms[p] = Algorithm::template armOf<T>(window, taps, phaseLen, nChannels, m, p);
            }
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                outputs[k][m] = Algorithm::channelAt(_arms.data(), _twiddles.data(), nChannels, k);
            }
        }
        return gr::work::Status::OK;
    }

    /**
     * The same two passes, on a device.
     *
     * A dynamic port collection cannot be handed to a kernel — the channel count is only known once the graph
     * is built — so `Block.hpp` requires this hatch rather than relocating `processBulk`. What the kernel
     * needs instead is a device-visible ARRAY of the channels' output pointers, gathered here once per call.
     *
     * The two passes are the ones the algorithm documents: arms first, then the transform across them. Both
     * are flat index-independent loops, which is exactly what makes the FIR bank worth putting on a device.
     */
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, auto& outputs) noexcept {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        const std::size_t phaseLen  = phaseLength();
        if (nChannels == 0UZ || phaseLen == 0UZ || outputs.empty()) {
            return gr::work::Status::OK;
        }
        std::size_t nSets = outputs[0].size();
        for (const auto& channel : outputs) { // a collection's ports need not offer the same room
            nSets = std::min(nSets, channel.size());
        }

        // The port spans are the edge's ring, which at a device boundary is pinned HOST memory: a kernel can
        // address it, but every access crosses PCIe one transaction at a time and costs far more than the
        // arithmetic. So the frame is staged -- one bulk copy in, both passes on device memory, one bulk copy
        // out per channel. A CPU backend addresses the spans directly and skips all of it. `phases` and
        // `_twiddles` need no staging: being reflected members they were re-seated onto device memory already.
        const bool               stage         = ctx.hasRemoteMemory();
        gr::device::DeviceBuffer windowBuffer  = stage ? ctx.allocateDevice<T>(input.size()) : gr::device::DeviceBuffer{};
        gr::device::DeviceBuffer armBuffer     = ctx.allocateDevice<T>(nSets * nChannels);
        gr::device::DeviceBuffer channelBuffer = stage ? ctx.allocateDevice<T>(nSets * nChannels) : ctx.allocateShared<T*>(nChannels); // staged: channel-major [k * nSets + m]; direct: the ports' pointers
        T* const                 window        = stage ? windowBuffer.devicePointer<T>() : const_cast<T*>(input.data());
        T* const                 arms          = armBuffer.devicePointer<T>();
        T* const                 channels      = stage ? channelBuffer.devicePointer<T>() : nullptr;
        T** const                ports         = stage ? nullptr : channelBuffer.devicePointer<T*>();
        const auto               release       = [&] {
            ctx.deallocate(windowBuffer);
            ctx.deallocate(armBuffer);
            ctx.deallocate(channelBuffer);
        };
        if (window == nullptr || arms == nullptr || (stage ? channels == nullptr : ports == nullptr)) {
            release();
            return gr::work::Status::ERROR;
        }

        const std::size_t nWindow = input.size();
        if (stage) {
            ctx.copy(window, input.data(), nWindow * sizeof(T));
        } else {
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                ports[k] = outputs[k].data(); // gathered on the host; the kernel only indexes it
            }
        }

        const Real* const taps     = phases.data();
        const std::size_t nPhases  = phases.size();
        const T* const    twiddles = _twiddles.data();

        // pass 1: one arm per (output set, arm) pair
        gr::device::parallelFor(ctx, nSets * nChannels, [arms, window, nWindow, taps, nPhases, phaseLen, nChannels] GR_DEVICE_LAMBDA(std::size_t index) {
            const std::size_t m = index / nChannels;
            const std::size_t p = index % nChannels;
            arms[index]         = Algorithm::template armOf<T>(std::span<const T>{window, nWindow}, std::span<const Real>{taps, nPhases}, phaseLen, nChannels, m, p);
        });

        // pass 2: the transform across the arms of each set, written channel-major so that neighbouring work
        // items write neighbouring samples of one channel rather than one sample of each channel
        gr::device::parallelFor(ctx, nSets * nChannels, [arms, channels, ports, twiddles, nChannels, nSets, stage] GR_DEVICE_LAMBDA(std::size_t index) {
            const std::size_t k     = index / nSets;
            const std::size_t m     = index % nSets;
            const T           value = Algorithm::channelAt(arms + m * nChannels, twiddles, nChannels, k);
            if (stage) {
                channels[k * nSets + m] = value;
            } else {
                ports[k][m] = value;
            }
        });

        if (stage) {
            // queued rather than awaited one at a time: a channel's frame is only `nSets` samples, so a blocking
            // copy per channel pays the transfer's fixed cost `nChannels` times over and the payload never gets
            // large enough to amortise it. `wait()` below is what makes the batch safe to release.
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                ctx.copy(outputs[k].data(), channels + k * nSets, nSets * sizeof(T), false);
            }
            ctx.wait(); // the staging buffer is freed immediately after, so every copy must have landed
        }

        release();
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_POLYPHASE_CHANNELIZER_HPP
