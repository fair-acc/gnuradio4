#ifndef GNURADIO_TESTING_TENSOR_SOURCE_HPP
#define GNURADIO_TESTING_TENSOR_SOURCE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>

namespace gr::testing {

/**
 * emits `_frames` -- each its extents and row-major elements -- then ends; the first carries a tag of `_firstTag` and,
 * unless `_sampleRate` is 0, `gr:sample_rate`; `_frameTags` adds a tag to the frame of the given index
 */
template<typename T>
struct TensorSource : gr::Block<TensorSource<T>> {
    gr::TensorPortOut<T> out;
    GR_MAKE_REFLECTABLE(TensorSource, out);

    struct Frame {
        std::vector<std::size_t> extents;
        std::vector<T>           elements;
    };
    std::vector<Frame>                                    _frames;
    float                                                 _sampleRate = 0.f;
    gr::property_map                                      _firstTag;
    std::vector<std::pair<std::size_t, gr::property_map>> _frameTags;
    std::size_t                                           _sent = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& output) {
        const std::size_t n = std::min(output.size(), _frames.size() - _sent);
        if (_sent == 0UZ && n > 0UZ) {
            gr::property_map first = _firstTag;
            if (_sampleRate > 0.f) {
                first.insert_or_assign(std::string(gr::tag::kSampleRate.key()), _sampleRate);
            }
            if (!first.empty()) {
                output.publishTag(first, 0UZ);
            }
        }
        for (std::size_t k = 0UZ; k < n; ++k) {
            const Frame& frame   = _frames[_sent + k];
            auto         emitted = gr::emit(output, k, std::span<const std::size_t>(frame.extents));
            if (!emitted) {
                return gr::work::Status::ERROR;
            }
            std::ranges::copy(frame.elements, emitted->begin());
            for (const auto& [taggedFrame, tag] : _frameTags) {
                if (taggedFrame == _sent + k) {
                    output.publishTag(tag, k);
                }
            }
        }
        output.publish(n);
        _sent += n;
        return _sent == _frames.size() ? gr::work::Status::DONE : gr::work::Status::OK;
    }
};

/** keeps a copy of every tensor's elements and extents, and of every tag with the index of the tensor it came with */
template<typename T, typename... Attributes>
struct TensorCollector : gr::Block<TensorCollector<T, Attributes...>> {
    gr::TensorPortIn<T, Attributes...> in;
    GR_MAKE_REFLECTABLE(TensorCollector, in);

    std::vector<std::vector<T>>                           _elements;
    std::vector<std::vector<std::size_t>>                 _shapes;
    std::vector<std::pair<std::size_t, gr::property_map>> _tags;

    gr::work::Status processBulk(gr::InputSpanLike auto& input) {
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            _tags.emplace_back(_elements.size() + static_cast<std::size_t>(relativeIndex), gr::property_map(tagMap.get()));
        }
        for (const auto& tensor : input) {
            _elements.emplace_back(tensor.data(), tensor.data() + tensor.size());
            _shapes.emplace_back(tensor.extents().begin(), tensor.extents().end());
        }
        std::ignore = input.consume(input.size());
        return gr::work::Status::OK;
    }
};

/** a deterministic, white-ish sequence in [-1, 1): a 32-bit linear congruential generator (Numerical Recipes constants) */
[[nodiscard]] inline std::vector<double> whiteNoise(std::size_t n, std::uint32_t seed) {
    std::linear_congruential_engine<std::uint32_t, 1664525U, 1013904223U, 0U> engine(seed);
    std::vector<double>                                                       values(n);
    std::ranges::generate(values, [&engine] { return static_cast<double>(engine()) / 2147483648.0 - 1.0; });
    return values;
}

} // namespace gr::testing

#endif // GNURADIO_TESTING_TENSOR_SOURCE_HPP
