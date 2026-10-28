#ifndef GNURADIO_FILTER_MOVING_AVERAGE_HPP
#define GNURADIO_FILTER_MOVING_AVERAGE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

enum class Statistic : std::uint8_t { mean, sum, meanSquare, rms, variance, unbiasedVariance, stdDev, unbiasedStdDev, minimum, maximum, logPower };

namespace detail {

template<Statistic statistic>
inline constexpr bool kTracksSpread = statistic == Statistic::variance || statistic == Statistic::unbiasedVariance || statistic == Statistic::stdDev || statistic == Statistic::unbiasedStdDev;
template<Statistic statistic>
inline constexpr bool kTracksExtreme = statistic == Statistic::minimum || statistic == Statistic::maximum;
template<Statistic statistic>
inline constexpr bool kFoldsPower = statistic == Statistic::meanSquare || statistic == Statistic::rms || statistic == Statistic::logPower;
template<Statistic statistic>
inline constexpr bool kUnbiased = statistic == Statistic::unbiasedVariance || statistic == Statistic::unbiasedStdDev;

template<typename T, Statistic statistic, bool exponential>
concept Averageable = (std::floating_point<T> || (meta::complex_like<T> && std::floating_point<typename T::value_type> && !kTracksExtreme<statistic>)) //
                      && !(exponential && (kUnbiased<statistic> || statistic == Statistic::sum));

/**
 * One lane -- a frame position, or one part of a complex one -- followed across frames; lanes interleave like the samples.
 * The window position is shared by all lanes and advanced by the host between calls; integrate-and-dump keeps no state.
 */
template<typename T, Statistic statistic, bool exponential>
struct PositionAverage {
    using R = meta::fundamental_base_value_type_t<T>;

    static constexpr bool        kComplex          = meta::complex_like<T>;
    static constexpr std::size_t kLanesPerPosition = kComplex && !kFoldsPower<statistic> ? 2UZ : 1UZ;
    static constexpr bool        kPairsLanes       = kLanesPerPosition == 2UZ && kTracksSpread<statistic>;
    static constexpr R           kPowerFloor       = R(1e-30);

    /** a dumped window sums up to 2^20 frames: in float that reaches per-cent errors */
    using Acc = std::conditional_t<std::same_as<R, float>, double, R>;
    template<typename V>
    using AccOf = typename std::conditional_t<std::same_as<V, R>, std::type_identity<Acc>, stdx::rebind_simd<Acc, V>>::type;

    const R*    input;
    R*          output;
    R*          ring;
    R*          centre;
    R*          spread;
    std::size_t frameSize;
    std::size_t window;
    std::size_t decimation;
    std::size_t nOut;
    std::size_t head;
    std::size_t count;
    R           alpha;

    [[nodiscard]] GR_DEVICE_FN std::size_t nLanes() const noexcept { return frameSize * kLanesPerPosition; }
    [[nodiscard]] GR_DEVICE_FN bool        dumps() const noexcept { return !exponential && decimation == window; }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN static V load(const R* at) noexcept {
        if constexpr (std::same_as<V, R>) {
            return *at;
        } else {
            return V(at, stdx::element_aligned);
        }
    }

    template<typename V>
    GR_DEVICE_FN static void store(const V& value, R* at) noexcept {
        if constexpr (std::same_as<V, R>) {
            *at = value;
        } else {
            value.copy_to(at, stdx::element_aligned);
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN static AccOf<V> widen(const V& value) noexcept {
        if constexpr (std::same_as<V, R>) {
            return static_cast<Acc>(value);
        } else {
            return stdx::static_simd_cast<AccOf<V>>(value);
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN static V narrow(const AccOf<V>& value) noexcept {
        if constexpr (std::same_as<V, R>) {
            return static_cast<R>(value);
        } else {
            return stdx::static_simd_cast<V>(value);
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN static V pickBetter(const V& held, const V& candidate) noexcept {
        V chosen = held;
        if constexpr (statistic == Statistic::maximum) {
            stdx::where(candidate > held, chosen) = candidate;
        } else {
            stdx::where(candidate < held, chosen) = candidate;
        }
        return chosen;
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN static V finishMoment(V moment) noexcept {
        if constexpr (statistic == Statistic::rms || statistic == Statistic::stdDev || statistic == Statistic::unbiasedStdDev) {
            using std::sqrt;
            stdx::where(moment < V(R{0}), moment) = V(R{0});
            return sqrt(moment);
        } else {
            return moment;
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V samplesAt(std::size_t frame, std::size_t lane) const noexcept {
        const R* frameSamples = input + frame * frameSize * (kComplex ? 2UZ : 1UZ);
        const V  folded       = [&] {
            if constexpr (kComplex && kLanesPerPosition == 1UZ) {
                const auto powerAt = [frameSamples](std::size_t at) {
                    const R re = frameSamples[2UZ * at];
                    const R im = frameSamples[2UZ * at + 1UZ];
                    return re * re + im * im;
                };
                if constexpr (std::same_as<V, R>) {
                    return powerAt(lane);
                } else {
                    return V([&](auto offset) { return powerAt(lane + offset); });
                }
            } else if constexpr (kFoldsPower<statistic>) {
                const V sample = load<V>(frameSamples + lane);
                return sample * sample;
            } else {
                return load<V>(frameSamples + lane);
            }
        }();
        if constexpr (statistic == Statistic::logPower) {
            const auto decibels = [](R power) { return R{10} * std::log10(std::max(power, kPowerFloor)); };
            if constexpr (std::same_as<V, R>) {
                return decibels(folded);
            } else {
                return V([&](auto offset) { return decibels(folded[offset]); });
            }
        } else {
            return folded;
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V scanRing(std::size_t lane) const noexcept {
        V extreme = load<V>(ring + lane);
        for (std::size_t slot = 1UZ; slot < window; ++slot) {
            extreme = pickBetter(extreme, load<V>(ring + slot * nLanes() + lane));
        }
        return extreme;
    }

    template<typename V>
    GR_DEVICE_FN void foldWindowed(const V& sample, std::size_t lane, std::size_t slot, std::size_t before, R weight) const noexcept {
        R* const   slotLane   = ring + slot * nLanes() + lane;
        const bool full       = before == window;
        const V    outgoing   = full ? load<V>(slotLane) : V(R{0});
        V          heldCentre = load<V>(centre + lane);
        store(sample, slotLane);

        if constexpr (kTracksExtreme<statistic>) {
            V          candidate = before == 0UZ ? sample : pickBetter(heldCentre, sample);
            const auto lost      = (outgoing == heldCentre) && (candidate != sample);
            if (full && stdx::any_of(lost)) {
                stdx::where(lost, candidate) = scanRing<V>(lane);
            }
            store(candidate, centre + lane);
            return;
        }

        if constexpr (!kTracksSpread<statistic>) {
            store(heldCentre + sample - outgoing, centre + lane);
            return;
        }

        if (!full) { // Welford [1]
            const V delta = sample - heldCentre;
            heldCentre    = heldCentre + delta * V(weight);
            store(load<V>(spread + lane) + delta * (sample - heldCentre), spread + lane);
        } else {
            const V nextCentre = heldCentre + (sample - outgoing) * V(weight);
            store(load<V>(spread + lane) + (sample - outgoing) * (sample - nextCentre + outgoing - heldCentre), spread + lane);
            heldCentre = nextCentre;
        }
        store(heldCentre, centre + lane);
    }

    template<typename V>
    GR_DEVICE_FN void foldExponential(const V& sample, std::size_t lane, std::size_t before) const noexcept {
        const R weight     = std::max(alpha, R{1} / static_cast<R>(before + 1UZ));
        const V heldCentre = load<V>(centre + lane);
        if constexpr (kTracksExtreme<statistic>) {
            V held                                                      = heldCentre + V(weight) * (sample - heldCentre);
            stdx::where(pickBetter(heldCentre, sample) == sample, held) = sample;
            store(before == 0UZ ? sample : held, centre + lane);
        } else {
            const V delta = sample - heldCentre;
            store(heldCentre + V(weight) * delta, centre + lane);
            if constexpr (kTracksSpread<statistic>) {
                store(V(R{1} - weight) * (load<V>(spread + lane) + V(weight) * delta * delta), spread + lane);
            }
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V heldMoment(std::size_t lane, std::size_t inWindow) const noexcept {
        if constexpr (kTracksSpread<statistic> && exponential) {
            return load<V>(spread + lane);
        } else if constexpr (kTracksSpread<statistic>) {
            const std::size_t degrees = kUnbiased<statistic> ? inWindow - 1UZ : inWindow;
            return degrees == 0UZ ? V(R{0}) : load<V>(spread + lane) * V(R{1} / static_cast<R>(degrees));
        } else if constexpr (exponential || kTracksExtreme<statistic> || statistic == Statistic::sum) {
            return load<V>(centre + lane);
        } else {
            return load<V>(centre + lane) * V(R{1} / static_cast<R>(inWindow));
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V dumpedMoment(std::size_t lane, std::size_t outFrame) const noexcept {
        const std::size_t first = outFrame * decimation;
        if constexpr (kTracksExtreme<statistic>) {
            V extreme = samplesAt<V>(first, lane);
            for (std::size_t frame = first + 1UZ; frame < first + decimation; ++frame) {
                extreme = pickBetter(extreme, samplesAt<V>(frame, lane));
            }
            return extreme;
        } else {
            AccOf<V> total = AccOf<V>(Acc{0});
            for (std::size_t frame = first; frame < first + decimation; ++frame) {
                total = total + widen(samplesAt<V>(frame, lane));
            }
            if constexpr (statistic == Statistic::sum) {
                return narrow<V>(total);
            }
            const AccOf<V> mean = total * AccOf<V>(Acc{1} / static_cast<Acc>(decimation));
            if constexpr (kTracksSpread<statistic>) {
                AccOf<V> squares = AccOf<V>(Acc{0});
                for (std::size_t frame = first; frame < first + decimation; ++frame) {
                    const AccOf<V> deviation = widen(samplesAt<V>(frame, lane)) - mean;
                    squares                  = squares + deviation * deviation;
                }
                const std::size_t degrees = kUnbiased<statistic> ? decimation - 1UZ : decimation;
                return degrees == 0UZ ? V(R{0}) : narrow<V>(squares * AccOf<V>(Acc{1} / static_cast<Acc>(degrees)));
            } else {
                return narrow<V>(mean);
            }
        }
    }

    /** a single lane's frames are contiguous, so they are the vector's lanes */
    [[nodiscard]] R dumpedAcrossFrames(std::size_t outFrame) const noexcept {
        if constexpr (kTracksExtreme<statistic>) {
            return dumpedMoment<R>(0UZ, outFrame);
        } else {
            using AccV                = stdx::native_simd<Acc>;
            const std::size_t first   = outFrame * decimation;
            const auto        sumOver = [&](auto&& termAt) {
                AccV        lanes{Acc{0}};
                std::size_t frame = 0UZ;
                for (; frame + AccV::size() <= decimation; frame += AccV::size()) {
                    lanes += AccV([&](auto offset) { return termAt(first + frame + offset); });
                }
                Acc total = stdx::reduce(lanes);
                for (; frame < decimation; ++frame) {
                    total += termAt(first + frame);
                }
                return total;
            };
            const Acc total = sumOver([&](std::size_t frame) { return static_cast<Acc>(samplesAt<R>(frame, 0UZ)); });
            if constexpr (statistic == Statistic::sum) {
                return static_cast<R>(total);
            }
            const Acc mean = total / static_cast<Acc>(decimation);
            if constexpr (kTracksSpread<statistic>) {
                const Acc         squares = sumOver([&](std::size_t frame) {
                    const Acc deviation = static_cast<Acc>(samplesAt<R>(frame, 0UZ)) - mean;
                    return deviation * deviation;
                });
                const std::size_t degrees = kUnbiased<statistic> ? decimation - 1UZ : decimation;
                return degrees == 0UZ ? R{0} : static_cast<R>(squares / static_cast<Acc>(degrees));
            } else {
                return static_cast<R>(mean);
            }
        }
    }

    template<typename V>
    [[nodiscard]] GR_DEVICE_FN V momentOf(std::size_t lane, std::size_t outFrame) const noexcept {
        return dumps() ? dumpedMoment<V>(lane, outFrame) : heldMoment<V>(lane, std::min(count + (outFrame + 1UZ) * decimation, window));
    }

    /** `forLanes(body)` calls `body.template operator()<V>(lane)` on the lanes its caller serves */
    GR_DEVICE_FN void foldInto(std::size_t outFrame, auto&& forLanes) const noexcept {
        if (dumps()) {
            return;
        }
        for (std::size_t frame = outFrame * decimation; frame < (outFrame + 1UZ) * decimation; ++frame) {
            const std::size_t before = std::min(count + frame, window);
            forLanes([&]<typename V>(std::size_t lane) {
                if constexpr (exponential) {
                    foldExponential(samplesAt<V>(frame, lane), lane, before);
                } else {
                    const R weight = R{1} / static_cast<R>(before == window ? window : before + 1UZ);
                    foldWindowed(samplesAt<V>(frame, lane), lane, (head + frame) % window, before, weight);
                }
            });
            const bool endsPass = !exponential && !kTracksExtreme<statistic> && count + frame >= window && (head + frame) % window + 1UZ == window;
            if (endsPass) {
                forLanes([&]<typename V>(std::size_t lane) {
                    V sum = V(R{0});
                    for (std::size_t slot = 0UZ; slot < window; ++slot) {
                        sum = sum + load<V>(ring + slot * nLanes() + lane);
                    }
                    if constexpr (!kTracksSpread<statistic>) {
                        store(sum, centre + lane);
                    } else {
                        const V mean    = sum * V(R{1} / static_cast<R>(window));
                        V       squares = V(R{0});
                        for (std::size_t slot = 0UZ; slot < window; ++slot) {
                            const V deviation = load<V>(ring + slot * nLanes() + lane) - mean;
                            squares           = squares + deviation * deviation;
                        }
                        store(mean, centre + lane);
                        store(squares, spread + lane);
                    }
                });
            }
        }
    }

    template<typename V>
    GR_DEVICE_FN void emit(std::size_t lane, std::size_t outFrame) const noexcept {
        store(finishMoment(momentOf<V>(lane, outFrame)), output + outFrame * nLanes() + lane);
    }

    GR_DEVICE_FN void emitPair(std::size_t position, std::size_t outFrame) const noexcept {
        const R realPart                        = momentOf<R>(2UZ * position, outFrame);
        const R imagPart                        = momentOf<R>(2UZ * position + 1UZ, outFrame);
        output[outFrame * frameSize + position] = finishMoment(realPart + imagPart);
    }

    /** the device's work item: one position across every frame of the call */
    GR_DEVICE_FN void operator()(std::size_t position) const noexcept {
        const auto positionLanes = [position](auto&& body) {
            for (std::size_t lane = position * kLanesPerPosition; lane < (position + 1UZ) * kLanesPerPosition; ++lane) {
                body.template operator()<R>(lane);
            }
        };
        for (std::size_t outFrame = 0UZ; outFrame < nOut; ++outFrame) {
            foldInto(outFrame, positionLanes);
            if constexpr (kPairsLanes) {
                emitPair(position, outFrame);
            } else {
                positionLanes([&]<typename V>(std::size_t lane) { emit<V>(lane, outFrame); });
            }
        }
    }
};

} // namespace detail

GR_REGISTER_BLOCK("gr::filter::MovingMean", gr::filter::Average, ([T], gr::filter::Statistic::mean, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingSum", gr::filter::Average, ([T], gr::filter::Statistic::sum, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingMeanSquare", gr::filter::Average, ([T], gr::filter::Statistic::meanSquare, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingRms", gr::filter::Average, ([T], gr::filter::Statistic::rms, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingVariance", gr::filter::Average, ([T], gr::filter::Statistic::variance, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingUnbiasedVariance", gr::filter::Average, ([T], gr::filter::Statistic::unbiasedVariance, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingStdDev", gr::filter::Average, ([T], gr::filter::Statistic::stdDev, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingUnbiasedStdDev", gr::filter::Average, ([T], gr::filter::Statistic::unbiasedStdDev, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::MovingMinimum", gr::filter::Average, ([T], gr::filter::Statistic::minimum, false), [ float, double ])
GR_REGISTER_BLOCK("gr::filter::MovingMaximum", gr::filter::Average, ([T], gr::filter::Statistic::maximum, false), [ float, double ])
GR_REGISTER_BLOCK("gr::filter::MovingLogPower", gr::filter::Average, ([T], gr::filter::Statistic::logPower, false), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialMean", gr::filter::Average, ([T], gr::filter::Statistic::mean, true), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialMeanSquare", gr::filter::Average, ([T], gr::filter::Statistic::meanSquare, true), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialRms", gr::filter::Average, ([T], gr::filter::Statistic::rms, true), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialVariance", gr::filter::Average, ([T], gr::filter::Statistic::variance, true), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialStdDev", gr::filter::Average, ([T], gr::filter::Statistic::stdDev, true), [ float, double, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::filter::ExponentialMinimum", gr::filter::Average, ([T], gr::filter::Statistic::minimum, true), [ float, double ])
GR_REGISTER_BLOCK("gr::filter::ExponentialMaximum", gr::filter::Average, ([T], gr::filter::Statistic::maximum, true), [ float, double ])
GR_REGISTER_BLOCK("gr::filter::ExponentialLogPower", gr::filter::Average, ([T], gr::filter::Statistic::logPower, true), [ float, double, std::complex<float>, std::complex<double> ])

template<typename T, Statistic statistic, bool exponential>
requires detail::Averageable<T, statistic, exponential>
struct Average : Block<Average<T, statistic, exponential>, Resampling<>> {
    using Description = Doc<R""(@brief a statistic of each frame position over the last `n_frames` frames: moving (equal weights) or exponential (`alpha = 2 / (n_frames + 1)`).

`frame_size = 1` follows the sample stream itself; `decimation` emits one frame per that many: 1 runs, `n_frames` integrates and dumps.
Until the window fills, the frames seen are the window.
A complex mean or sum keeps the phase (fed `x * conj(y)`: the averaged cross-spectrum); every other complex statistic is real.
`logPower` averages dB, which for noise reads 2.5 dB below the dB of the mean power.
Extremes: the exponential form attacks instantly and releases exponentially; the moving form rescans when the extreme leaves, O(n_frames) on a monotonic input.

[1] B. P. Welford, "Note on a method for calculating corrected sums of squares and products", Technometrics, vol. 4, no. 3, pp. 419-420, 1962.)"">;

    using Kernel = detail::PositionAverage<T, statistic, exponential>;
    using R      = Kernel::R;
    using TOut   = std::conditional_t<Kernel::kLanesPerPosition == 2UZ && !Kernel::kPairsLanes, T, R>;

    static constexpr std::uint64_t kMaxStateSamples = 1ULL << 28U;
    static constexpr gr::Size_t    kMaxLength       = 1U << 20U;

    PortIn<T>     in;
    PortOut<TOut> out;

    Annotated<gr::Size_t, "frame size", Doc<"positions per frame, each followed across frames; 1 = the sample stream itself">, Limits<1U, kMaxLength>> frame_size = 1U;
    Annotated<gr::Size_t, "window length", Doc<"frames averaged; the exponential form's equivalent length">, Limits<1U, kMaxLength>>                   n_frames   = 16U;
    Annotated<gr::Size_t, "decimation", Doc<"frames in per frame out: 1 runs, n_frames integrates and dumps">, Limits<1U, kMaxLength>>                 decimation = 1U;

    std::pmr::vector<R> _ring;
    std::pmr::vector<R> _centre;
    std::pmr::vector<R> _spread;

    GR_MAKE_REFLECTABLE(Average, in, out, frame_size, n_frames, decimation, _ring, _centre, _spread);

    std::size_t _head  = 0UZ;
    std::size_t _count = 0UZ;

    void start() {
        if (!holdsWindow()) {
            reset();
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<TOut> output) noexcept {
        if (!holdsWindow()) {
            return gr::work::Status::ERROR;
        }
        const Kernel kernel = kernelFor(reinterpret_cast<const R*>(input.data()), reinterpret_cast<R*>(output.data()), output.size() / frame_size.value);
        assert(input.size() == kernel.nOut * kernel.decimation * kernel.frameSize);

        using V             = stdx::native_simd<R>;
        const auto forLanes = [&kernel](auto&& body) {
            std::size_t lane = 0UZ;
            for (; lane + V::size() <= kernel.nLanes(); lane += V::size()) {
                body.template operator()<V>(lane);
            }
            for (; lane < kernel.nLanes(); ++lane) {
                body.template operator()<R>(lane);
            }
        };
        if (kernel.dumps() && kernel.nLanes() == 1UZ) {
            for (std::size_t outFrame = 0UZ; outFrame < kernel.nOut; ++outFrame) {
                output[outFrame] = static_cast<TOut>(Kernel::finishMoment(kernel.dumpedAcrossFrames(outFrame)));
            }
            return gr::work::Status::OK;
        }
        for (std::size_t outFrame = 0UZ; outFrame < kernel.nOut; ++outFrame) {
            kernel.foldInto(outFrame, forLanes);
            if constexpr (Kernel::kPairsLanes) {
                for (std::size_t position = 0UZ; position < kernel.frameSize; ++position) {
                    kernel.emitPair(position, outFrame);
                }
            } else {
                forLanes([&]<typename TV>(std::size_t lane) { kernel.template emit<TV>(lane, outFrame); });
            }
        }
        advance(kernel.nOut * kernel.decimation);
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t frame = frame_size.value;
        const std::size_t nOut  = std::min(output.size() / frame, input.size() / (frame * decimation.value));
        const bool        holds = holdsWindow();
        if (nOut == 0UZ || !holds) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return holds ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::ERROR;
        }
        const Kernel kernel = kernelFor(reinterpret_cast<const R*>(input.data()), reinterpret_cast<R*>(output.data()), nOut);
        gr::device::parallelFor(ctx, frame, [kernel] GR_DEVICE_LAMBDA(std::size_t position) { kernel(position); });
        advance(nOut * kernel.decimation);

        std::ignore = input.consume(nOut * kernel.decimation * frame);
        output.publish(nOut * frame);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (!newSettings.contains("frame_size") && !newSettings.contains("n_frames") && !newSettings.contains("decimation")) {
            return;
        }
        const std::uint64_t ringSamples = exponential || dumps() ? 0U : std::uint64_t{nLanes()} * n_frames.value;
        if (const std::uint64_t needed = std::max(chunk(), ringSamples); needed > kMaxStateSamples) [[unlikely]] {
            const std::string message = std::format("frame_size {}, n_frames {} and decimation {} need {} samples at once, more than {}: keeps the previous window", frame_size.value, n_frames.value, decimation.value, needed, kMaxStateSamples);
            this->emitErrorMessage("settingsChanged()", message);
            frame_size = oldSettings.value_or<gr::Size_t>("frame_size", 1U);
            n_frames   = oldSettings.value_or<gr::Size_t>("n_frames", 16U);
            decimation = oldSettings.value_or<gr::Size_t>("decimation", 1U);
            return;
        }
        reset();
    }

    [[nodiscard]] std::size_t nLanes() const noexcept { return Kernel::kLanesPerPosition * frame_size.value; }
    /** 64-bit: 2^20 x 2^20 wraps a 32-bit size_t (wasm32) */
    [[nodiscard]] std::uint64_t chunk() const noexcept { return std::uint64_t{decimation.value} * frame_size.value; }
    [[nodiscard]] bool          dumps() const noexcept { return !exponential && decimation.value == n_frames.value; }
    [[nodiscard]] bool          holdsWindow() const noexcept { return this->input_chunk_size.value == chunk() && _centre.size() == (dumps() ? 0UZ : nLanes()); }

    void reset() {
        this->input_chunk_size      = static_cast<gr::Size_t>(chunk());
        this->output_chunk_size     = frame_size.value;
        const std::size_t heldLanes = dumps() ? 0UZ : nLanes();
        _centre.assign(heldLanes, R{0});
        _spread.assign(detail::kTracksSpread<statistic> ? heldLanes : 0UZ, R{0});
        _ring.assign(exponential ? 0UZ : heldLanes * n_frames.value, R{0});
        _head  = 0UZ;
        _count = 0UZ;
    }

    [[nodiscard]] Kernel kernelFor(const R* input, R* output, std::size_t nOut) noexcept {
        return Kernel{.input = input, //
            .output          = output,
            .ring            = _ring.data(),
            .centre          = _centre.data(),
            .spread          = _spread.data(),
            .frameSize       = frame_size.value,
            .window          = n_frames.value,
            .decimation      = decimation.value,
            .nOut            = nOut,
            .head            = _head,
            .count           = _count,
            .alpha           = R{2} / static_cast<R>(n_frames.value + 1U)};
    }

    void advance(std::size_t nFrames) noexcept {
        _head  = (_head + nFrames) % n_frames.value;
        _count = std::min(_count + nFrames, static_cast<std::size_t>(n_frames.value));
    }
};

template<typename T, Statistic statistic = Statistic::mean>
using MovingAverage = Average<T, statistic, false>;

template<typename T, Statistic statistic = Statistic::mean>
using ExponentialAverage = Average<T, statistic, true>;

} // namespace gr::filter

#endif // GNURADIO_FILTER_MOVING_AVERAGE_HPP
