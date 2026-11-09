#ifndef GNURADIO_MATH_ARGMAX_HPP
#define GNURADIO_MATH_ARGMAX_HPP

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK(gr::blocks::math::Argmax, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T>
requires std::is_arithmetic_v<T>
struct Argmax : Block<Argmax<T>, Resampling<1UZ, 1UZ, false>> {
    using Description = Doc<R""(@brief for each vector of `length` samples on every input, the position and the input of the largest sample (GR3: Argmax).

A tie goes to the smaller position, then to the lower input; NaN ranks below every number.)"">;

    std::vector<PortIn<T>> in{1UZ};
    PortOut<gr::Size_t>    index;
    PortOut<gr::Size_t>    stream;

    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"number of inputs">, Limits<1U, 32U>> n_inputs = 1U;
    Annotated<gr::Size_t, "length", Doc<"samples per vector">, Limits<1U, 1U << 20U>>    length   = 1U;

    GR_MAKE_REFLECTABLE(Argmax, in, index, stream, n_inputs, length);

    template<gr::InputSpanLike TInSpan>
    [[nodiscard]] gr::work::Status processBulk(std::span<TInSpan>& ins, OutputSpanLike auto& indices, OutputSpanLike auto& streams) {
        const std::size_t nInputs   = ins.size();
        const std::size_t perVector = length.value;
        const auto        shortest  = std::ranges::min(ins | std::views::transform([](const auto& span) { return span.size(); }));
        const std::size_t nVectors  = std::min({shortest / perVector, indices.size(), streams.size()});

        const auto ranksBelow = [](T lhs, T rhs) {
            if constexpr (std::floating_point<T>) {
                return std::isnan(lhs) ? !std::isnan(rhs) : lhs < rhs;
            } else {
                return lhs < rhs;
            }
        };
        for (std::size_t vector = 0UZ; vector < nVectors; ++vector) {
            const auto sampleAt = [&](std::size_t flat) { return ins[flat % nInputs][vector * perVector + flat / nInputs]; };
            const auto largest  = *std::ranges::max_element(std::views::iota(0UZ, perVector * nInputs), ranksBelow, sampleAt);
            indices[vector]     = static_cast<gr::Size_t>(largest / nInputs);
            streams[vector]     = static_cast<gr::Size_t>(largest % nInputs);
        }
        for (auto& span : ins) {
            std::ignore = span.consume(nVectors * perVector);
        }
        indices.publish(nVectors);
        streams.publish(nVectors);
        if (nVectors > 0UZ) {
            return gr::work::Status::OK;
        }
        return shortest < perVector ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("n_inputs")) {
            in.resize(n_inputs);
        }
        if (newSettings.contains("length")) {
            this->input_chunk_size = length.value;
        }
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_ARGMAX_HPP
