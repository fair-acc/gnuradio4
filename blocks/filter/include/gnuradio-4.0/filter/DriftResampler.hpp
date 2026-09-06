#ifndef GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP
#define GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/HermiteResampler.hpp>

namespace gr::filter {

GR_REGISTER_BLOCK(gr::filter::DriftResampler, [T], [ float, double ])

/**
 * @brief Resamples at an arbitrary, possibly drifting ratio.
 *
 * Where `RationalResampler` states a fixed window and lets the framework run those windows concurrently, this
 * one cannot: the number of outputs per input is not fixed, so it owns its own accounting and stays sequential.
 * That is the trade — an arbitrary ratio that may move while the stream runs, at the cost of the parallelism a
 * declared window buys.
 */
template<typename T>
requires std::floating_point<T>
struct DriftResampler : Block<DriftResampler<T>> {
    using Algorithm   = gr::algorithm::filter::HermiteResampler<T>;
    using Description = Doc<"resamples at an arbitrary ratio using four-point cubic Hermite interpolation">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<float, "ratio", Doc<"output samples per input sample; may be changed while running">> ratio = 1.f;

    GR_MAKE_REFLECTABLE(DriftResampler, in, out, ratio);

    static constexpr gr::Size_t kMinimumWindow = 4U; // the four points the interpolant is built from

    double _phase = 0.0;

    DriftResampler(gr::property_map init = {}) : Block<DriftResampler<T>>(std::move(init)) { in.min_samples = kMinimumWindow; }

    void reset() { _phase = 0.0; }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto progress = Algorithm::resample(std::span<const T>{input.data(), input.size()}, std::span<T>{output.data(), output.size()}, static_cast<double>(ratio), _phase);
        if (progress.produced == 0UZ) {
            // the window is shorter than the interpolator needs and will not grow again; draining it is what
            // ends the stream, where holding on to it would spin forever asking for samples nobody will send
            std::ignore = input.consume(input.size());
            output.publish(0UZ);
            return gr::work::Status::OK;
        }
        std::ignore = input.consume(progress.consumed);
        output.publish(progress.produced);
        return gr::work::Status::OK;
    }
};

} // namespace gr::filter

#endif // GNURADIO_BLOCKS_DRIFT_RESAMPLER_HPP
