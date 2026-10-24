#ifndef GNURADIO_FILTER_PHASE_UNWRAP_HPP
#define GNURADIO_FILTER_PHASE_UNWRAP_HPP

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <span>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::filter {

enum class UnwrapAxis : std::uint8_t { frames, positions };

GR_REGISTER_BLOCK(gr::filter::PhaseUnwrap, [T], [ float, double ])

template<std::floating_point T>
struct PhaseUnwrap : Block<PhaseUnwrap<T>, Resampling<>> {
    using Description = Doc<R""(@brief removes the 2 pi jumps from a phase in radians: `frames` follows each frame position across frames (with `frame_size = 1` a phase stream), `positions` unwraps along each frame.

A step of exactly +-pi stays uncorrected; a non-finite sample passes without corrupting the count after it.)"">;

    struct Kernel {
        const T*    input;
        T*          output;
        T*          previous;
        T*          turns;
        std::size_t frameSize;
        std::size_t nFrames;

        GR_DEVICE_FN void unwrapAcrossFrames(std::size_t position) const noexcept {
            T previousRaw = previous[position];
            T count       = turns[position];
            for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
                const T raw = input[frame * frameSize + position];
                count += gr::algorithm::fft::unwrapTurns(raw - previousRaw);
                output[frame * frameSize + position] = raw + static_cast<T>(2) * std::numbers::pi_v<T> * count;
                previousRaw                          = raw;
            }
            previous[position] = previousRaw;
            turns[position]    = count;
        }

        template<typename V>
        void unwrapAcrossPositions(std::size_t frame, std::size_t position) const noexcept {
            const std::size_t at = frame * frameSize + position;
            const V           raw(input + at, stdx::element_aligned);
            const V           count = V(turns + position, stdx::element_aligned) + gr::algorithm::fft::unwrapTurns(raw - V(previous + position, stdx::element_aligned));
            (raw + static_cast<T>(2) * std::numbers::pi_v<T> * count).copy_to(output + at, stdx::element_aligned);
            raw.copy_to(previous + position, stdx::element_aligned);
            count.copy_to(turns + position, stdx::element_aligned);
        }

        GR_DEVICE_FN void unwrapAlongFrame(std::size_t frame) const noexcept {
            std::copy_n(input + frame * frameSize, frameSize, output + frame * frameSize);
            gr::algorithm::fft::unwrapPhase(std::span<T>(output + frame * frameSize, frameSize));
        }
    };

    static constexpr gr::Size_t kMaxFrameSize = 1U << 20U;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "frame size", Doc<"positions per frame; 1 = a plain phase stream">, Limits<1U, kMaxFrameSize>> frame_size = 1U;
    Annotated<UnwrapAxis, "axis", Doc<"frames: each position across successive frames; positions: along each frame">>    axis       = UnwrapAxis::frames;

    std::pmr::vector<T> _previous;
    std::pmr::vector<T> _turns;

    GR_MAKE_REFLECTABLE(PhaseUnwrap, in, out, frame_size, axis, _previous, _turns);

    void start() {
        if (_previous.size() != frame_size.value) {
            reset();
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept {
        const Kernel kernel = kernelFor(input.data(), output.data(), input.size());
        if (axis == UnwrapAxis::positions) {
            for (std::size_t frame = 0UZ; frame < kernel.nFrames; ++frame) {
                kernel.unwrapAlongFrame(frame);
            }
        } else if (kernel.frameSize == 1UZ) {
            kernel.unwrapAcrossFrames(0UZ);
        } else {
            using V = stdx::native_simd<T>;
            for (std::size_t frame = 0UZ; frame < kernel.nFrames; ++frame) {
                std::size_t position = 0UZ;
                for (; position + V::size() <= kernel.frameSize; position += V::size()) {
                    kernel.template unwrapAcrossPositions<V>(frame, position);
                }
                for (; position < kernel.frameSize; ++position) {
                    kernel.template unwrapAcrossPositions<stdx::simd<T, stdx::simd_abi::scalar>>(frame, position);
                }
            }
        }
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t frame = frame_size.value;
        const std::size_t n     = std::min(input.size(), output.size()) / frame * frame;
        if (n == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        const Kernel kernel = kernelFor(input.data(), output.data(), n);
        if (axis == UnwrapAxis::positions) {
            gr::device::parallelFor(ctx, kernel.nFrames, [kernel] GR_DEVICE_LAMBDA(std::size_t index) { kernel.unwrapAlongFrame(index); });
        } else {
            gr::device::parallelFor(ctx, frame, [kernel] GR_DEVICE_LAMBDA(std::size_t index) { kernel.unwrapAcrossFrames(index); });
        }
        std::ignore = input.consume(n);
        output.publish(n);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        this->input_chunk_size  = frame_size.value;
        this->output_chunk_size = frame_size.value;
        if (newSettings.contains("frame_size") || newSettings.contains("axis") || _previous.empty()) {
            reset();
        }
    }

    void reset() {
        _previous.assign(frame_size.value, std::numeric_limits<T>::quiet_NaN());
        _turns.assign(frame_size.value, T{0});
    }

    [[nodiscard]] Kernel kernelFor(const T* input, T* output, std::size_t nSamples) noexcept {
        return Kernel{.input = input, //
            .output          = output,
            .previous        = _previous.data(),
            .turns           = _turns.data(),
            .frameSize       = frame_size.value,
            .nFrames         = nSamples / frame_size.value};
    }
};

} // namespace gr::filter

#endif // GNURADIO_FILTER_PHASE_UNWRAP_HPP
