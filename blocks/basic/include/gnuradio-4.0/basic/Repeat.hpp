#ifndef GNURADIO_BASIC_REPEAT_HPP
#define GNURADIO_BASIC_REPEAT_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::Repeat, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct Repeat : Block<Repeat<T>, Resampling<1UZ, 1UZ, false>> {
    using Description = Doc<R""(@brief emits every sample `n` times (GR3: Repeat); the output rate is `n` times the input's.

A tag lands on the first copy of its sample; a new `n` takes effect at the sample it is set on and announces the new `sample_rate`.
An `n` whose copies the output buffer cannot hold is refused when the graph starts.)"">;

    static constexpr gr::Size_t kMaxCopies = 1U << 24U;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "n", Doc<"copies of each sample">, Limits<1U, kMaxCopies>> n           = 1U;
    gr::SampleRate                                                                   sample_rate = 0.f;

    GR_MAKE_REFLECTABLE(Repeat, in, out, n, sample_rate);

    bool _announceRate = false;

    void start() { this->output_chunk_size = n.value; }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t copies = n.value;
        const std::size_t nIn    = std::min(input.size(), output.size() / copies);
        if (nIn == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        if (copies == 1UZ) {
            std::ranges::copy_n(input.begin(), static_cast<std::ptrdiff_t>(nIn), output.begin());
        } else {
            for (std::size_t sample = 0UZ; sample < nIn; ++sample) {
                std::fill_n(output.begin() + static_cast<std::ptrdiff_t>(sample * copies), copies, input[sample]);
            }
        }
        announceRate(output);
        std::ignore = input.consume(nIn);
        output.publish(nIn * copies);
        return gr::work::Status::OK;
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t copies = n.value;
        const std::size_t nIn    = std::min(input.size(), output.size() / copies);
        if (nIn == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        gr::device::parallelFor(ctx, nIn * copies, [samples = input.data(), copied = output.data(), copies] GR_DEVICE_LAMBDA(std::size_t index) { copied[index] = samples[index / copies]; });
        announceRate(output);
        std::ignore = input.consume(nIn);
        output.publish(nIn * copies);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (!newSettings.contains("n")) {
            return;
        }
        if (!detail::chunkFitsBuffer(*this, out, n.value, std::format("n = {} copies", n.value))) {
            return;
        }
        this->output_chunk_size = n.value;
        _announceRate           = sample_rate.value > 0.f;
    }

    void announceRate(OutputSpanLike auto& output) {
        if (_announceRate) {
            output.publishTag(detail::sampleRateTag(sample_rate.value * static_cast<float>(n.value)), 0UZ);
            _announceRate = false;
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_REPEAT_HPP
