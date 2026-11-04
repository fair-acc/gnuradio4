#ifndef GNURADIO_HIERARCHICAL_POLYPHASE_CHANNELIZER_HPP
#define GNURADIO_HIERARCHICAL_POLYPHASE_CHANNELIZER_HPP

#include <algorithm>
#include <complex>
#include <memory_resource>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>

#include <gnuradio-4.0/algorithm/filter/AllpassHalfBand.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::HierarchicalPolyphaseChannelizer, [T], [ std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK(gr::filter::HierarchicalIirChannelizer, [T], [ std::complex<float>, std::complex<double> ])

/**
 * Splits a wideband stream into `stage1_channels * stage2_channels` channels, in two stages.
 *
 * A flat bank needs a prototype sharp enough to separate channels 1/nChannels apart, so its tap count grows
 * with the channel count. Two stages need two SHORT prototypes instead: 64 channels as 8x8 costs roughly
 * 8 + 8 arms of filtering rather than 64. That saving is the only reason this structure exists — the output
 * is the same set of channels a flat `PolyphaseChannelizer` would give.
 *
 * Both stages are the same algorithm with different channel counts, so there is no second filterbank
 * implementation here; the block is the composition, and the DSP lives in `PolyphaseChannelizer`.
 *
 * IMPORTANT -- the channel grid is NOT the same as a flat `PolyphaseChannelizer` of the same total count.
 * Cascading two critically sampled stages puts the flat-bank centres at odd multiples of
 * `fs / (2 * stage1_channels)` exactly on stage-1 channel BOUNDARIES, where a tone splits evenly between two
 * ports rather than landing in one. Measured: a tone at fs/4 through a 2x2 bank divides 50/50. Matching a
 * flat grid needs an oversampled first stage or a half-channel offset in the second, which this does not do.
 *
 * So this block is for the case where the saving matters and the exact grid is the caller's to interpret.
 * Channel ordering is `stage1 * stage2_channels + stage2` as produced;
 * `qa_HierarchicalPolyphaseChannelizer` measures where a tone actually lands rather than assuming.
 */
template<typename T>
requires gr::meta::complex_like<T>
struct HierarchicalPolyphaseChannelizer : Block<HierarchicalPolyphaseChannelizer<T>, Resampling<>, Stride<>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using Algorithm   = gr::algorithm::filter::PolyphaseChannelizer<Real>;
    using Description = Doc<R""(splits a wideband stream into `stage1_channels * stage2_channels` channels using two cascaded polyphase banks.

Each stage transforms over its own channel count rather than the product, which is the decomposition that keeps the
cross-channel DFT from growing as the square of the total. `PolyphaseChannelizer` is the undecomposed form and the
simpler one to reason about.

 * f. j. harris, C. Dick and M. Rice, "Digital receivers and transmitters using polyphase filter banks for wireless
   communications", IEEE Trans. Microw. Theory Techn., vol. 51, no. 4, pp. 1395-1412, 2003.)"">;

    PortIn<T>               in;
    std::vector<PortOut<T>> out;

    Annotated<gr::Size_t, "stage1_channels", Doc<"coarse channels the first stage produces">, Visible, Limits<2U, 256U>>              stage1_channels   = 2U;
    Annotated<gr::Size_t, "stage2_channels", Doc<"channels the second stage splits each coarse one into">, Visible, Limits<2U, 256U>> stage2_channels   = 2U;
    Annotated<gr::Size_t, "n_taps", Doc<"prototype length per stage; each stage designs its own">, Limits<2U, 8192U>>                 n_taps            = 32U;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"output sets produced per work call">, Limits<1U, 16384U>>                         outputs_per_frame = 32U;

    std::pmr::vector<Real> phases_stage1{}; // both banks re-seat onto device memory when the block runs there
    std::pmr::vector<Real> phases_stage2{};

    std::pmr::vector<T> _coarse{};    // stage-1 output, flattened [set * stage1 + channel]
    std::pmr::vector<T> _arms{};      // one value per arm of whichever stage is running
    std::pmr::vector<T> _twiddles1{}; // one table per stage: the two transforms have different lengths
    std::pmr::vector<T> _twiddles2{};

    GR_MAKE_REFLECTABLE(HierarchicalPolyphaseChannelizer, in, out, stage1_channels, stage2_channels, n_taps, outputs_per_frame, phases_stage1, phases_stage2, _coarse, _arms, _twiddles1, _twiddles2);

    [[nodiscard]] constexpr std::size_t phaseLength1() const noexcept { return phases_stage1.size() / std::max(std::size_t{1}, static_cast<std::size_t>(stage1_channels)); }
    [[nodiscard]] constexpr std::size_t phaseLength2() const noexcept { return phases_stage2.size() / std::max(std::size_t{1}, static_cast<std::size_t>(stage2_channels)); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nStage1 = static_cast<std::size_t>(stage1_channels);
        const std::size_t nStage2 = static_cast<std::size_t>(stage2_channels);
        if (out.size() != nStage1 * nStage2) {
            out.resize(nStage1 * nStage2);
        }

        const std::size_t taps1      = std::max(static_cast<std::size_t>(n_taps), 2UZ * nStage1);
        const auto        prototype1 = Algorithm::designPrototype(taps1, nStage1);
        const auto        bank1      = Algorithm::decompose(prototype1, nStage1);
        phases_stage1.assign(bank1.begin(), bank1.end());

        const std::size_t taps2      = std::max(static_cast<std::size_t>(n_taps), 2UZ * nStage2);
        const auto        prototype2 = Algorithm::designPrototype(taps2, nStage2);
        const auto        bank2      = Algorithm::decompose(prototype2, nStage2);
        phases_stage2.assign(bank2.begin(), bank2.end());

        const std::size_t phaseLen1 = Algorithm::phaseLength(prototype1.size(), nStage1);
        const std::size_t phaseLen2 = Algorithm::phaseLength(prototype2.size(), nStage2);
        const std::size_t nSets     = static_cast<std::size_t>(outputs_per_frame);

        // stage 2 needs `windowLength` of ITS sets; each of those is one stage-1 set, which is nStage1 inputs
        const std::size_t stage1Sets = Algorithm::windowLength(nSets, nStage2, phaseLen2);
        this->output_chunk_size      = static_cast<gr::Size_t>(nSets);
        this->input_chunk_size       = static_cast<gr::Size_t>(Algorithm::windowLength(stage1Sets, nStage1, phaseLen1));
        this->stride                 = static_cast<gr::Size_t>(nSets * nStage1 * nStage2);

        _coarse.assign(stage1Sets * nStage1, T{}); // scratch sized here: a kernel cannot allocate
        _arms.assign(std::max(nStage1, nStage2), T{});
        Algorithm::template seatTwiddles<T>(_twiddles1, Algorithm::designTwiddles(nStage1));
        Algorithm::template seatTwiddles<T>(_twiddles2, Algorithm::designTwiddles(nStage2));
    }

    template<typename TOutputSpan>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, std::span<TOutputSpan>& outputs) noexcept {
        const std::size_t nStage1 = static_cast<std::size_t>(stage1_channels);
        const std::size_t nStage2 = static_cast<std::size_t>(stage2_channels);
        const std::size_t len1    = phaseLength1();
        const std::size_t len2    = phaseLength2();
        if (nStage1 == 0UZ || nStage2 == 0UZ || len1 == 0UZ || len2 == 0UZ || outputs.empty()) {
            return gr::work::Status::OK;
        }

        std::size_t nSets = outputs[0].size();
        for (const auto& channel : outputs) { // a collection's ports need not offer the same room
            nSets = std::min(nSets, channel.size());
        }
        const std::size_t stage1Sets = Algorithm::windowLength(nSets, nStage2, len2);

        // stage 1: the coarse split, held flattened as [set * nStage1 + channel] so each coarse channel's
        // history is at stride nStage1 -- the same shape the second stage reads it back at
        const std::span<const T>    window{input.data(), input.size()};
        const std::span<const Real> taps1{phases_stage1.data(), phases_stage1.size()};
        if (_coarse.size() < stage1Sets * nStage1 || _arms.size() < std::max(nStage1, nStage2)) {
            _coarse.assign(stage1Sets * nStage1, T{}); // a span beyond the declared chunk: grow once
            _arms.assign(std::max(nStage1, nStage2), T{});
        }
        for (std::size_t m = 0UZ; m < stage1Sets; ++m) {
            for (std::size_t p = 0UZ; p < nStage1; ++p) {
                _arms[p] = Algorithm::template armOf<T>(window, taps1, len1, nStage1, m, p);
            }
            for (std::size_t k = 0UZ; k < nStage1; ++k) {
                _coarse[m * nStage1 + k] = Algorithm::template channelAt<T>(_arms.data(), _twiddles1.data(), nStage1, k);
            }
        }

        // stage 2: each coarse channel is already in `_coarse` at stride nStage1, so its arms read the
        // flattened array at the COMBINED stride rather than being copied out to a contiguous buffer first
        const std::span<const Real> taps2{phases_stage2.data(), phases_stage2.size()};
        const std::span<const T>    coarseSpan{_coarse.data(), stage1Sets * nStage1};
        for (std::size_t c = 0UZ; c < nStage1; ++c) {
            for (std::size_t m = 0UZ; m < nSets; ++m) {
                for (std::size_t p = 0UZ; p < nStage2; ++p) {
                    const std::size_t newest = Algorithm::armNewest(m, nStage2, len2, p) * nStage1 + c;
                    _arms[p]                 = Algorithm::Bank::template armAt<T>(coarseSpan, taps2.data() + p * len2, len2, newest, nStage2 * nStage1);
                }
                for (std::size_t k = 0UZ; k < nStage2; ++k) {
                    outputs[c * nStage2 + k][m] = Algorithm::template channelAt<T>(_arms.data(), _twiddles2.data(), nStage2, k);
                }
            }
        }
        return gr::work::Status::OK;
    }

    /**
     * The same two stages, on a device — four flat passes, no de-interleaving.
     *
     * Required because the channel ports are a runtime collection, which `Block.hpp` will not hand to a
     * kernel. Each stage is an arm pass and a transform pass, and stage 2 reads the stage-1 output in place
     * at the COMBINED stride, so nothing is copied between them.
     */
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, auto& outputs) noexcept {
        const std::size_t nStage1 = static_cast<std::size_t>(stage1_channels);
        const std::size_t nStage2 = static_cast<std::size_t>(stage2_channels);
        const std::size_t len1    = phaseLength1();
        const std::size_t len2    = phaseLength2();
        if (nStage1 == 0UZ || nStage2 == 0UZ || len1 == 0UZ || len2 == 0UZ || outputs.empty()) {
            return gr::work::Status::OK;
        }
        std::size_t nSets = outputs[0].size();
        for (const auto& channel : outputs) { // a collection's ports need not offer the same room
            nSets = std::min(nSets, channel.size());
        }
        const std::size_t stage1Sets = Algorithm::windowLength(nSets, nStage2, len2);
        const std::size_t nChannels  = nStage1 * nStage2;

        // staged exactly as the flat bank stages, and for the same reason
        const bool               stage     = ctx.hasRemoteMemory();
        gr::device::DeviceBuffer windowBuf = stage ? ctx.allocateDevice<T>(input.size()) : gr::device::DeviceBuffer{};
        gr::device::DeviceBuffer armsOne   = ctx.allocateDevice<T>(stage1Sets * nStage1);
        gr::device::DeviceBuffer coarseBuf = ctx.allocateDevice<T>(stage1Sets * nStage1);
        gr::device::DeviceBuffer armsTwo   = ctx.allocateDevice<T>(nStage1 * nSets * nStage2);
        gr::device::DeviceBuffer outBuf    = stage ? ctx.allocateDevice<T>(nChannels * nSets) : ctx.allocateShared<T*>(nChannels);
        T* const                 arms1     = armsOne.devicePointer<T>();
        T* const                 coarse    = coarseBuf.devicePointer<T>();
        T* const                 arms2     = armsTwo.devicePointer<T>();
        T* const                 staged    = stage ? outBuf.devicePointer<T>() : nullptr;
        T** const                ports     = stage ? nullptr : outBuf.devicePointer<T*>();
        const auto               release   = [&] {
            ctx.deallocate(windowBuf);
            ctx.deallocate(armsOne);
            ctx.deallocate(coarseBuf);
            ctx.deallocate(armsTwo);
            ctx.deallocate(outBuf);
        };
        if (arms1 == nullptr || coarse == nullptr || arms2 == nullptr || (stage ? staged == nullptr : ports == nullptr)) {
            release();
            return gr::work::Status::ERROR;
        }

        const std::size_t nWindow = input.size();
        const T*          window  = input.data();
        if (stage) {
            T* const staged_in = windowBuf.devicePointer<T>();
            if (staged_in == nullptr) {
                release();
                return gr::work::Status::ERROR;
            }
            ctx.copy(staged_in, input.data(), nWindow * sizeof(T));
            window = staged_in;
        } else {
            for (std::size_t k = 0UZ; k < nChannels && k < outputs.size(); ++k) {
                ports[k] = outputs[k].data();
            }
        }

        const Real* const taps1     = phases_stage1.data();
        const std::size_t nTaps1    = phases_stage1.size();
        const Real* const taps2     = phases_stage2.data();
        const std::size_t nTaps2    = phases_stage2.size();
        const T* const    twiddles1 = _twiddles1.data();
        const T* const    twiddles2 = _twiddles2.data();

        gr::device::parallelFor(ctx, stage1Sets * nStage1, [arms1, window, nWindow, taps1, nTaps1, len1, nStage1] GR_DEVICE_LAMBDA(std::size_t i) { arms1[i] = Algorithm::template armOf<T>(std::span<const T>{window, nWindow}, std::span<const Real>{taps1, nTaps1}, len1, nStage1, i / nStage1, i % nStage1); });
        gr::device::parallelFor(ctx, stage1Sets * nStage1, [arms1, coarse, twiddles1, nStage1] GR_DEVICE_LAMBDA(std::size_t i) { //
            coarse[i] = Algorithm::template channelAt<T>(arms1 + (i / nStage1) * nStage1, twiddles1, nStage1, i % nStage1);
        });

        const std::size_t coarseCount = stage1Sets * nStage1;
        gr::device::parallelFor(ctx, nStage1 * nSets * nStage2, [arms2, coarse, coarseCount, taps2, nTaps2, len2, nStage1, nStage2, nSets] GR_DEVICE_LAMBDA(std::size_t i) {
            const std::size_t c      = i / (nSets * nStage2);
            const std::size_t m      = (i / nStage2) % nSets;
            const std::size_t p      = i % nStage2;
            const std::size_t newest = Algorithm::armNewest(m, nStage2, len2, p) * nStage1 + c;
            arms2[i]                 = Algorithm::Bank::template armAt<T>(std::span<const T>{coarse, coarseCount}, taps2 + p * len2, len2, newest, nStage2 * nStage1);
            (void)nTaps2;
        });
        // `m` fastest, so neighbouring work items write neighbouring samples of ONE channel
        gr::device::parallelFor(ctx, nChannels * nSets, [arms2, staged, ports, twiddles2, nStage2, nSets, stage] GR_DEVICE_LAMBDA(std::size_t i) {
            const std::size_t port  = i / nSets;
            const std::size_t m     = i % nSets;
            const std::size_t c     = port / nStage2;
            const std::size_t k     = port % nStage2;
            const T           value = Algorithm::template channelAt<T>(arms2 + (c * nSets + m) * nStage2, twiddles2, nStage2, k);
            if (stage) {
                staged[port * nSets + m] = value;
            } else {
                ports[port][m] = value;
            }
        });

        if (stage) {
            // queued, then awaited once -- see the note in `PolyphaseChannelizer`: a per-channel frame is too
            // small to amortise a blocking transfer's fixed cost, and there are `nChannels` of them
            for (std::size_t k = 0UZ; k < nChannels && k < outputs.size(); ++k) {
                ctx.copy(outputs[k].data(), staged + k * nSets, nSets * sizeof(T), false);
            }
            ctx.wait(); // `release()` frees the staging buffer, so the batch must have landed first
        }

        release();
        return gr::work::Status::OK;
    }
};

/**
 * The same channel split, as a binary tree of allpass IIR half-bands.
 *
 * The independent second implementation of `HierarchicalPolyphaseChannelizer`, kept beside it so the two can
 * be compared directly (see `bm_HierarchicalChannelizer`). Where the FIR bank filters once per output with a
 * long prototype and separates channels by a transform, this halves the band `log2(n_channels)` times with a
 * handful of allpass coefficients per stage — a steep transition for very few multiplies.
 *
 * The trade-offs are real and opposite:
 *
 * - **only powers of two.** The allpass two-branch form is a HALF-band decomposition, so the tree can only
 *   reach `n_channels = 2^k`. The FIR bank takes any factorisation.
 * - **recursive.** Each stage's output depends on its own previous output, so a device cannot parallelise
 *   across samples the way it can across the FIR bank's independent arms. Expect the FIR form to win on
 *   `gpu:sycl` and this one to win on the host for steep transitions.
 * - **not linear phase**, though `n_sections` controls how flat the group delay is; a Butterworth half-band
 *   is maximally flat by construction, which is why it is the design used here.
 *
 * Both bands of every stage are kept, so no information is discarded: the tree is critically sampled and the
 * `n_channels` leaves tile the input band.
 */
template<typename T>
requires gr::meta::complex_like<T>
struct HierarchicalIirChannelizer : Block<HierarchicalIirChannelizer<T>, Resampling<>, Stride<>> {
    using Real        = gr::meta::fundamental_base_value_type_t<T>;
    using HalfBand    = gr::algorithm::filter::AllpassHalfBand<Real>;
    using Description = Doc<R""(splits a wideband stream into `2^k` channels using a binary tree of allpass IIR half-bands.

A steep transition costs a handful of coefficients instead of a long tap set, but the phase is not linear and the
recursion is sequential: prefer `PolyphaseChannelizer` or `HierarchicalPolyphaseChannelizer` where phase matters, or
where the bank must run on a device.

 * R. Ansari and B. Liu, "Multirate signal processing", in Advanced Topics in Signal Processing, J. S. Lim and
   A. V. Oppenheim, Eds. Englewood Cliffs, NJ: Prentice Hall, 1988.)"">;

    PortIn<T>               in;
    std::vector<PortOut<T>> out;

    Annotated<gr::Size_t, "n_channels", Doc<"channel count; must be a power of two, since each stage halves the band">, Visible, Limits<2U, 256U>> n_channels        = 8U;
    Annotated<gr::Size_t, "n_sections", Doc<"allpass sections per stage; more sharpens the crossover">, Limits<1U, 8U>>                            n_sections        = 3U;
    Annotated<gr::Size_t, "outputs_per_frame", Doc<"samples produced per channel per work call">, Limits<1U, 16384U>>                              outputs_per_frame = 64U;

    std::pmr::vector<Real> alphas_even{}; // both branches' coefficients, shared by every stage of the tree
    std::pmr::vector<Real> alphas_odd{};
    std::pmr::vector<T>    _state{}; // one even+odd state block per tree node, laid out node-major

    std::pmr::vector<T> _current{}; // the level being read; the tree ping-pongs between these two
    std::pmr::vector<T> _next{};

    GR_MAKE_REFLECTABLE(HierarchicalIirChannelizer, in, out, n_channels, n_sections, outputs_per_frame, alphas_even, alphas_odd, _state, _current, _next);

    /// tree depth; `n_channels` is rounded DOWN to a power of two, because a half-band tree cannot express anything else
    [[nodiscard]] constexpr std::size_t depth() const noexcept {
        std::size_t levels = 0UZ;
        for (std::size_t m = static_cast<std::size_t>(n_channels); m > 1UZ; m /= 2UZ) {
            ++levels;
        }
        return levels;
    }

    [[nodiscard]] constexpr std::size_t channels() const noexcept { return 1UZ << depth(); }

    /// state values one node needs: an even branch cascade plus an odd one
    [[nodiscard]] constexpr std::size_t stateStride() const noexcept { return HalfBand::stateSize(alphas_even.size()) + HalfBand::stateSize(alphas_odd.size()); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        const std::size_t nChannels = channels();
        if (out.size() != nChannels) {
            out.resize(nChannels);
        }

        const auto        designed = HalfBand::designAllpass(static_cast<std::size_t>(n_sections));
        std::vector<Real> even, odd;
        HalfBand::splitCoefficients(designed, even, odd);
        alphas_even.assign(even.begin(), even.end());
        alphas_odd.assign(odd.begin(), odd.end());

        _state.assign((nChannels - 1UZ) * stateStride(), T{}); // one block per internal node of the tree

        const std::size_t nOut  = static_cast<std::size_t>(outputs_per_frame);
        this->output_chunk_size = static_cast<gr::Size_t>(nOut);
        // each stage halves the rate, and the odd branch reaches one sample further back at every level
        this->input_chunk_size = static_cast<gr::Size_t>(nOut * nChannels + nChannels);
        this->stride           = static_cast<gr::Size_t>(nOut * nChannels);

        // both tree buffers are sized for the widest level here: a kernel cannot allocate, and on the host
        // this removes one malloc per level per call
        _current.assign(static_cast<std::size_t>(this->input_chunk_size), T{});
        _next.assign(static_cast<std::size_t>(this->input_chunk_size), T{});
    }

    template<typename TOutputSpan>
    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, std::span<TOutputSpan>& outputs) noexcept {
        const std::size_t nChannels = channels();
        const std::size_t levels    = depth();
        if (levels == 0UZ || outputs.empty() || _state.empty()) {
            return gr::work::Status::OK;
        }

        const std::span<const Real> even{alphas_even.data(), alphas_even.size()};
        const std::span<const Real> odd{alphas_odd.data(), alphas_odd.size()};
        const std::size_t           nOut = outputs[0].size();

        // one buffer per level: level L holds 2^L bands, and the two buffers ping-pong between levels
        const std::size_t inputSize = input.size();
        if (_current.size() < inputSize || _next.size() < inputSize) {
            _current.assign(inputSize, T{}); // a span beyond the declared chunk: grow once, not per call
            _next.assign(inputSize, T{});
        }
        // level 0 reads the port's own span, so the frame is never copied in: only the levels after it need a
        // buffer of their own, and those ping-pong by swapping the two rather than moving samples
        const T*    source   = input.data();
        std::size_t node     = 0UZ;
        std::size_t bands    = 1UZ;
        std::size_t liveSize = inputSize;
        for (std::size_t level = 0UZ; level < levels; ++level) {
            const std::size_t inPerBand  = liveSize / bands;
            const std::size_t outPerBand = inPerBand < 1UZ ? 0UZ : (inPerBand - 1UZ) / 2UZ;
            for (std::size_t b = 0UZ; b < bands; ++b) {
                T* const state    = _state.data() + node * stateStride();
                T* const oddState = state + HalfBand::stateSize(even.size());
                HalfBand::template split<T>(std::span<const T>{source + b * inPerBand, inPerBand}, even, odd,                                                                                        //
                    std::span<T>{state, HalfBand::stateSize(even.size())}, std::span<T>{oddState, HalfBand::stateSize(odd.size())}, std::span<T>{_next.data() + (2UZ * b) * outPerBand, outPerBand}, // low half
                    std::span<T>{_next.data() + (2UZ * b + 1UZ) * outPerBand, outPerBand});                                                                                                          // high half
                ++node;
            }
            _current.swap(_next);
            source   = _current.data();
            liveSize = bands * 2UZ * outPerBand;
            bands *= 2UZ;
        }

        const std::size_t perBand = liveSize / nChannels;
        for (std::size_t k = 0UZ; k < nChannels && k < outputs.size(); ++k) {
            const std::size_t take = std::min(nOut, perBand);
            for (std::size_t i = 0UZ; i < take; ++i) {
                outputs[k][i] = _current[k * perBand + i];
            }
        }
        return gr::work::Status::OK;
    }

    /**
     * The same tree, on a device — one work item per band, per level.
     *
     * The parallelism here is across BANDS, not samples: each half-band is a recursion whose output depends
     * on its own previous output, so a band cannot be split across work items. Level 0 is therefore a single
     * work item and only the deeper levels widen (1, 2, 4, ... bands). That ceiling is a property of the
     * structure, and it is the price the IIR tree pays for its very low per-sample cost.
     */
    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, auto& outputs) noexcept {
        const std::size_t nChannels = channels();
        const std::size_t levels    = depth();
        if (levels == 0UZ || outputs.empty() || _state.empty()) {
            return gr::work::Status::OK;
        }
        std::size_t nOut = outputs[0].size();
        for (const auto& channel : outputs) { // a collection's ports need not offer the same room
            nOut = std::min(nOut, channel.size());
        }

        const std::size_t        inputSize = input.size();
        gr::device::DeviceBuffer bufA      = ctx.allocateShared<T>(inputSize);
        gr::device::DeviceBuffer bufB      = ctx.allocateShared<T>(inputSize);
        gr::device::DeviceBuffer portBuf   = ctx.allocateShared<T*>(nChannels);
        T*                       current   = bufA.devicePointer<T>();
        T*                       next      = bufB.devicePointer<T>();
        T** const                ports     = portBuf.devicePointer<T*>();
        const auto               release   = [&] {
            ctx.deallocate(bufA);
            ctx.deallocate(bufB);
            ctx.deallocate(portBuf);
        };
        if (current == nullptr || next == nullptr || ports == nullptr) {
            release();
            return gr::work::Status::ERROR;
        }
        std::copy_n(input.begin(), inputSize, current);
        for (std::size_t k = 0UZ; k < nChannels && k < outputs.size(); ++k) {
            ports[k] = outputs[k].data();
        }

        const Real* const evenTaps = alphas_even.data();
        const std::size_t nEven    = alphas_even.size();
        const Real* const oddTaps  = alphas_odd.data();
        const std::size_t nOdd     = alphas_odd.size();
        const std::size_t stride   = stateStride();
        T* const          state    = _state.data();

        std::size_t node     = 0UZ;
        std::size_t bands    = 1UZ;
        std::size_t liveSize = inputSize;
        for (std::size_t level = 0UZ; level < levels; ++level) {
            const std::size_t inPerBand  = liveSize / bands;
            const std::size_t outPerBand = inPerBand < 1UZ ? 0UZ : (inPerBand - 1UZ) / 2UZ;
            const std::size_t nodeBase   = node;
            gr::device::parallelFor(ctx, bands, [current, next, state, stride, nodeBase, inPerBand, outPerBand, evenTaps, nEven, oddTaps, nOdd] GR_DEVICE_LAMBDA(std::size_t b) {
                T* const bandState = state + (nodeBase + b) * stride;
                HalfBand::template split<T>(std::span<const T>{current + b * inPerBand, inPerBand},                                                       //
                    std::span<const Real>{evenTaps, nEven}, std::span<const Real>{oddTaps, nOdd},                                                         //
                    std::span<T>{bandState, HalfBand::stateSize(nEven)}, std::span<T>{bandState + HalfBand::stateSize(nEven), HalfBand::stateSize(nOdd)}, //
                    std::span<T>{next + (2UZ * b) * outPerBand, outPerBand}, std::span<T>{next + (2UZ * b + 1UZ) * outPerBand, outPerBand});
            });
            node += bands;
            std::swap(current, next);
            liveSize = bands * 2UZ * outPerBand;
            bands *= 2UZ;
        }

        const std::size_t perBand = nChannels == 0UZ ? 0UZ : liveSize / nChannels;
        const std::size_t take    = std::min(nOut, perBand);
        gr::device::parallelFor(ctx, nChannels * take, [ports, current, perBand, take] GR_DEVICE_LAMBDA(std::size_t i) { //
            ports[i / take][i % take] = current[(i / take) * perBand + (i % take)];
        });

        release();
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_HIERARCHICAL_POLYPHASE_CHANNELIZER_HPP
