#ifndef GNURADIO_BASIC_TENSOR_MONITOR_HPP
#define GNURADIO_BASIC_TENSOR_MONITOR_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::TensorMonitor, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct TensorMonitor : Block<TensorMonitor<T>, NoTagPropagation> {
    using Description = Doc<R""(@brief passes tensors through unchanged and reports each change of their shape:
on `shape` as a tensor of shape [1 + max rank] -- the rank, then the extents, zero-padded -- and as a log line.

It never holds the tensors back: a change that finds `shape` full waits as the latest report and goes out once there is space;
a report superseded meanwhile, or still waiting when the stream stops, is only logged.)"">;

    static constexpr std::size_t kShapeSize = 1UZ + detail::kMaxRank;
    using Shape                             = std::array<gr::Size_t, kShapeSize>;

    TensorPortIn<T>                            in;
    TensorPortOut<T>                           out;
    TensorPortOut<gr::Size_t, Async, Optional> shape;

    GR_MAKE_REFLECTABLE(TensorMonitor, in, out, shape);

    static constexpr Shape kNoShape{std::numeric_limits<gr::Size_t>::max()};

    Shape                _lastShape = kNoShape;
    std::optional<Shape> _pending;

    void start() {
        _lastShape = kNoShape;
        _pending.reset();
    }

    void stop() {
        if (_pending) {
            gr::log::info("{}: shape report {} not delivered: `shape` was still full when the stream stopped", this->name, *_pending);
        }
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output, OutputSpanLike auto& shapeOut) {
        const auto copy   = [](std::span<T> elements, const TensorView<T>& tensor) { std::ranges::copy(tensor, elements.begin()); };
        const auto report = [](std::span<gr::Size_t> stored, const Shape& reported) { std::ranges::copy(reported, stored.begin()); };
        return passThrough(input, output, shapeOut, copy, report);
    }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output, OutputSpanLike auto& shapeOut) {
        const auto             copy   = [&ctx](std::span<T> elements, const TensorView<T>& tensor) { ctx.copy(elements.data(), tensor.data(), tensor.size() * sizeof(T), false); };
        const auto             report = [&ctx](std::span<gr::Size_t> stored, const Shape& reported) { ctx.copy(stored.data(), reported.data(), sizeof(Shape), true); };
        const gr::work::Status status = passThrough(input, output, shapeOut, copy, report);
        ctx.wait();
        return status;
    }

    /** copies every tensor into `out` with `copy(elements, tensor)` and forwards its tags; a change of shape goes to `shape` with `report(stored, shape)` */
    [[nodiscard]] gr::work::Status passThrough(auto& input, auto& output, auto& shapeOut, auto&& copy, auto&& report) {
        const std::size_t n       = std::min(input.size(), output.size());
        std::size_t       nShapes = 0UZ;
        const auto        offer   = [&](const Shape& changed) {
            if (!shape.isConnected()) {
                return true;
            }
            if (nShapes == shapeOut.size()) {
                if (_pending) {
                    gr::log::info("{}: shape report {} superseded while `shape` was full", this->name, *_pending);
                }
                _pending = changed;
                return true;
            }
            auto stored = gr::emit(*this, shapeOut, nShapes++, std::array{kShapeSize});
            if (!stored) [[unlikely]] {
                return false;
            }
            report(*stored, changed);
            return true;
        };
        bool failed = false;
        if (_pending && shapeOut.size() > 0UZ) {
            const Shape waiting = *_pending;
            _pending.reset();
            failed = !offer(waiting);
        }
        std::size_t k = 0UZ;
        for (; k < n && !failed; ++k) {
            const TensorView<T>& tensor = input[k];
            if (const Shape current = shapeOf(tensor); current != _lastShape) {
                _lastShape = current;
                gr::log::info("{}: tensor {} has rank {} and extents {}", this->name, input.streamIndex + k, tensor.rank(), tensor.extents());
                failed = !offer(current);
            }
            auto elements = gr::emit(*this, output, k, tensor.extents());
            if (!elements) [[unlikely]] {
                failed = true;
                break;
            }
            copy(*elements, tensor);
        }
        auto outputs = std::tie(output);
        for (const auto& [relativeIndex, tagMap] : input.tags()) {
            if (relativeIndex >= 0 && static_cast<std::size_t>(relativeIndex) < k) {
                this->forwardedTag(tagMap.get(), [&](const auto& forwarded) { this->publishTagToSingleProducerOutputs(outputs, forwarded, static_cast<std::size_t>(relativeIndex)); });
            }
        }
        std::ignore = input.consume(k);
        output.publish(k);
        shapeOut.publish(nShapes);
        if (failed) [[unlikely]] {
            this->requestStop();
        }
        const bool blockedByOutput = k == 0UZ && n == 0UZ && input.size() > 0UZ;
        return blockedByOutput ? gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS : gr::work::Status::OK;
    }

    [[nodiscard]] static Shape shapeOf(const TensorView<T>& tensor) noexcept {
        Shape result{static_cast<gr::Size_t>(tensor.rank())};
        std::ranges::transform(tensor.extents(), result.begin() + 1, [](std::size_t extent) { return static_cast<gr::Size_t>(extent); });
        return result;
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_TENSOR_MONITOR_HPP
