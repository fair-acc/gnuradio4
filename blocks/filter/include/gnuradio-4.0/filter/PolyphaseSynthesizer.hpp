#ifndef GNURADIO_POLYPHASE_SYNTHESIZER_HPP
#define GNURADIO_POLYPHASE_SYNTHESIZER_HPP

#include <complex>
#include <memory_resource>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

#include <gnuradio-4.0/algorithm/filter/PolyphaseSynthesizer.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::PolyphaseSynthesizer, [T], [ std::complex<float>, std::complex<double> ])

/**
 * @brief Recombines `n_channels` critically sampled channels into one wideband stream.
 *
 * The mirror of `PolyphaseChannelizer`, over the same prototype: a transform across the channels feeds one
 * value per arm, each arm filters its own slow stream, and the commutator interleaves them back up to the
 * full rate. The channel ports are a runtime collection, so the device path needs `processBulk(ctx, ...)`.
 */
template<typename T>
requires gr::meta::complex_like<T>
struct PolyphaseSynthesizer : Block<PolyphaseSynthesizer<T>, Resampling<>, Stride<>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using Algorithm   = gr::algorithm::filter::PolyphaseSynthesizer<Real>;
    using Description = Doc<R""(recombines `n_channels` critically sampled channels into one wideband stream using a polyphase synthesis filterbank.

The inverse of `PolyphaseChannelizer`, built from the same bank, so a channelize/synthesize pair reconstructs the band.

 * f. j. harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase filter banks for wireless
   communications", IEEE Trans. Microw. Theory Techn., vol. 51, no. 4, pp. 1395-1412, 2003.)"">;

    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    Annotated<gr::Size_t, "n_channels", Doc<"channel count; the output rate is the channel rate times this">, Visible, Limits<2U, 1024U>> n_channels        = 4U;
    Annotated<gr::Size_t, "n_taps", Doc<"prototype length; longer gives sharper channel edges">, Limits<2U, 8192U>>                       n_taps            = 64U;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"output sets produced per work call">, Limits<1U, 65536U>>                             outputs_per_frame = 64U;

    std::pmr::vector<Real> phases{}; // phase-major prototype, re-seated onto device memory when the block runs there

    std::pmr::vector<T> _armInputs{}; // scratch, sized in settingsChanged: a kernel cannot allocate
    std::pmr::vector<T> _channelsOfSet{};
    std::pmr::vector<T> _twiddles{}; // nChannels entries covering the whole transform, likewise re-seated

    GR_MAKE_REFLECTABLE(PolyphaseSynthesizer, in, out, n_channels, n_taps, outputs_per_frame, phases, _armInputs, _channelsOfSet, _twiddles);

    [[nodiscard]] constexpr std::size_t phaseLength() const noexcept { return phases.size() / std::max(std::size_t{1}, static_cast<std::size_t>(n_channels)); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        if (in.size() != nChannels) {
            in.resize(nChannels);
        }

        const std::size_t nTaps      = std::max(static_cast<std::size_t>(n_taps), 2UZ * nChannels);
        const auto        prototype  = Algorithm::designPrototype(nTaps, nChannels);
        const auto        decomposed = Algorithm::decompose(prototype, nChannels);
        phases.assign(decomposed.begin(), decomposed.end());

        Algorithm::template seatTwiddles<T>(_twiddles, Algorithm::designTwiddles(nChannels));

        const std::size_t phaseLen = Algorithm::phaseLength(prototype.size(), nChannels);
        const std::size_t nSets    = static_cast<std::size_t>(outputs_per_frame);

        this->input_chunk_size  = static_cast<gr::Size_t>(Algorithm::windowLength(nSets, phaseLen)); // per channel port
        this->output_chunk_size = static_cast<gr::Size_t>(nSets * nChannels);
        this->stride            = static_cast<gr::Size_t>(nSets); // the channel streams advance one set per output set
        _armInputs.assign(Algorithm::windowLength(nSets, phaseLen) * nChannels, T{});
        _channelsOfSet.assign(nChannels, T{});
    }

    template<typename TInputSpan>
    [[nodiscard]] gr::work::Status processBulk(std::span<TInputSpan>& inputs, OutputSpanLike auto& output) noexcept {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        const std::size_t phaseLen  = phaseLength();
        if (nChannels == 0UZ || phaseLen == 0UZ || inputs.empty()) {
            return gr::work::Status::OK;
        }

        const std::size_t nSets    = output.size() / nChannels;
        const std::size_t nArmSets = Algorithm::windowLength(nSets, phaseLen);

        // pass 1: the transform across channels, flattened [set * nChannels + arm] so one arm's history is
        // contiguous at stride nChannels -- the same shape the analysis side reads
        if (_armInputs.size() < nArmSets * nChannels || _channelsOfSet.size() < nChannels) {
            _armInputs.assign(nArmSets * nChannels, T{}); // a span beyond the declared chunk: grow once
            _channelsOfSet.assign(nChannels, T{});
        }
        for (std::size_t m = 0UZ; m < nArmSets; ++m) {
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                _channelsOfSet[k] = m < inputs[k].size() ? inputs[k][m] : T{};
            }
            for (std::size_t p = 0UZ; p < nChannels; ++p) {
                _armInputs[m * nChannels + p] = Algorithm::armInputAt(_channelsOfSet.data(), _twiddles.data(), nChannels, p);
            }
        }

        // pass 2: each arm filters its own slow stream, the commutator interleaves the results
        const std::span<const T>    armSpan{_armInputs.data(), nArmSets * nChannels};
        const std::span<const Real> taps{phases.data(), phases.size()};
        for (std::size_t m = 0UZ; m < nSets; ++m) {
            for (std::size_t p = 0UZ; p < nChannels; ++p) {
                output[Algorithm::outputIndex(m, nChannels, p)] = Algorithm::outputAt(armSpan, taps, phaseLen, nChannels, m, p);
            }
        }
        return gr::work::Status::OK;
    }

    /**
     * @brief The same two passes, on a device.
     *
     * The mirror of `PolyphaseChannelizer::processBulk(ctx, ...)`: a runtime channel count cannot be handed to a
     * kernel, so the channels reach it as a device-visible array of the ports' pointers.
     *
     * Not optional. On an edge between two device blocks the residency is `Access::DeviceOnly`, so a block
     * that fell back to the host here would read device memory rather than merely run slower.
     */
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, auto& inputs, OutputSpanLike auto& output) noexcept {
        const std::size_t nChannels = static_cast<std::size_t>(n_channels);
        const std::size_t phaseLen  = phaseLength();
        if (nChannels == 0UZ || phaseLen == 0UZ || inputs.empty()) {
            return gr::work::Status::OK;
        }
        const std::size_t nSets    = output.size() / nChannels;
        const std::size_t nArmSets = Algorithm::windowLength(nSets, phaseLen);
        if (nSets == 0UZ) {
            return gr::work::Status::OK;
        }

        // the mirror of the analysis side's staging, one bulk copy per channel in and one for the result out
        const bool               stage         = ctx.hasRemoteMemory();
        gr::device::DeviceBuffer armBuffer     = ctx.allocateDevice<T>(nArmSets * nChannels);
        gr::device::DeviceBuffer channelBuffer = stage ? ctx.allocateDevice<T>(nChannels * nArmSets) : ctx.allocateShared<const T*>(nChannels);
        gr::device::DeviceBuffer outBuffer     = stage ? ctx.allocateDevice<T>(nSets * nChannels) : gr::device::DeviceBuffer{};
        T* const                 armInputs     = armBuffer.devicePointer<T>();
        T* const                 gathered      = stage ? channelBuffer.devicePointer<T>() : nullptr;
        const T** const          channels      = stage ? nullptr : channelBuffer.devicePointer<const T*>();
        T* const                 stagedOut     = stage ? outBuffer.devicePointer<T>() : nullptr;
        const auto               release       = [&] {
            ctx.deallocate(armBuffer);
            ctx.deallocate(channelBuffer);
            ctx.deallocate(outBuffer);
        };
        if (armInputs == nullptr || (stage ? (gathered == nullptr || stagedOut == nullptr) : channels == nullptr)) {
            release();
            return gr::work::Status::ERROR;
        }

        std::size_t available = nArmSets;
        for (std::size_t k = 0UZ; k < nChannels; ++k) {
            available = std::min(available, inputs[k].size());
        }
        if (stage) { // one bulk copy per channel into a channel-major block the kernel can stride through
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                ctx.copy(gathered + k * nArmSets, inputs[k].data(), available * sizeof(T));
            }
        } else {
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                channels[k] = inputs[k].data();
            }
        }

        T* const          published = stage ? stagedOut : output.data();
        const Real* const taps      = phases.data();
        const std::size_t nPhases   = phases.size();
        const T* const    twiddles  = _twiddles.data();

        // pass 1: one arm input per (set, arm) pair -- the transform across the channels of that set
        gr::device::parallelFor(ctx, nArmSets * nChannels, [armInputs, channels, gathered, twiddles, nChannels, nArmSets, available, stage] GR_DEVICE_LAMBDA(std::size_t index) {
            const std::size_t m = index / nChannels;
            const std::size_t p = index % nChannels;
            gr::complex<Real> acc{};
            std::size_t       turn = 0UZ;
            for (std::size_t k = 0UZ; k < nChannels; ++k) {
                const T sample = m < available ? (stage ? gathered[k * nArmSets + m] : channels[k][m]) : T{};
                acc += gr::complex<Real>{sample.real(), sample.imag()} * gr::complex<Real>{twiddles[turn].real(), twiddles[turn].imag()};
                turn += p;
                if (turn >= nChannels) {
                    turn -= nChannels;
                }
            }
            armInputs[index] = T{acc.real(), acc.imag()};
        });

        // pass 2: each arm filters its own slow stream, the commutator interleaves the results
        gr::device::parallelFor(ctx, nSets * nChannels, [published, armInputs, taps, nPhases, phaseLen, nChannels, nArmSets] GR_DEVICE_LAMBDA(std::size_t index) {
            const std::size_t m                                = index / nChannels;
            const std::size_t p                                = index % nChannels;
            published[Algorithm::outputIndex(m, nChannels, p)] = Algorithm::template outputAt<T>(std::span<const T>{armInputs, nArmSets * nChannels}, std::span<const Real>{taps, nPhases}, phaseLen, nChannels, m, p);
        });

        if (stage) {
            ctx.copy(output.data(), stagedOut, nSets * nChannels * sizeof(T));
        }

        release();
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_POLYPHASE_SYNTHESIZER_HPP
