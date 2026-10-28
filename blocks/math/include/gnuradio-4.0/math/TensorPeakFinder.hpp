#ifndef GNURADIO_MATH_TENSOR_PEAK_FINDER_HPP
#define GNURADIO_MATH_TENSOR_PEAK_FINDER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory_resource>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK(gr::blocks::math::TensorPeakFinder, [T], [ float, double ])

template<std::floating_point T>
struct TensorPeakFinder : Block<TensorPeakFinder<T>, NoTagPropagation> {
    using Description = Doc<R""(@brief the peaks of a score map of rank 1 to 3: [K, D+2] rows {D coordinates, score, confidence}, at most `max_peaks`, by score.

A peak is a cell strictly above all its 3^D - 1 neighbours, not at an end of a non-periodic dimension; per dimension a parabola through it
and its neighbours places it between cells and adds its rise to the score. The reference is the larger of the map's maximum and the highest
score; a peak scores at least `threshold` times it, and one within `min_separation` cells of a higher is suppressed (greedily, highest first).
Coordinates are in the units of `gr:axis_origin` and `gr:axis_step`, in cells without them; `gr:axis_periodic` makes a dimension's ends
neighbours, spanning extent * step. The confidence is the score over the reference. A map whose reference is not positive gives no peaks.
Tags go out with their frame's peaks, without the `gr:axis_*` keys: a list of peaks has no grid axes.
References: J. O. Smith III and X. Serra, "PARSHL: An analysis/synthesis program for non-harmonic sounds based on a sinusoidal
representation," in Proc. ICMC, 1987, pp. 290-297; A. Neubeck and L. Van Gool, "Efficient non-maximum suppression," in Proc. ICPR, 2006,
pp. 850-855.)"">;

    TensorPortIn<T>  in;
    TensorPortOut<T> out;

    Annotated<float, "threshold", Doc<"least score relative to the reference">, Limits<0.f, 1.f>>                                 threshold      = 0.5f;
    Annotated<float, "min_separation", Doc<"in cells: a lower peak this close to a higher one is suppressed">, Limits<0.f, 1e6f>> min_separation = 1.f;
    Annotated<gr::Size_t, "max_peaks", Doc<"most peaks per frame">, Limits<1U, 64U>>                                              max_peaks      = 1U;

    /** [cells] a peak's score, else -inf; reflected, as the two below, for a kernel */
    std::pmr::vector<T> _scores;
    /** [max_peaks, D+2] */
    std::pmr::vector<T> _rows;
    /** [1 + max_peaks]: how many were found, then their cells */
    std::pmr::vector<std::uint32_t> _accepted;

    GR_MAKE_REFLECTABLE(TensorPeakFinder, in, out, threshold, min_separation, max_peaks, _scores, _rows, _accepted);

    /** the map's axes as the last `gr:axis_*` tags described them: unit steps from 0 until one arrives */
    struct Axes {
        std::array<double, gr::algorithm::peak::kMaxGridRank> origin{};
        std::array<double, gr::algorithm::peak::kMaxGridRank> step{1.0, 1.0, 1.0};
        std::array<bool, gr::algorithm::peak::kMaxGridRank>   periodic{};
        std::size_t                                           rank = 0UZ;
    };
    Axes _axes;

    void start() { _axes = Axes{}; }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return findPeaks(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return findPeaks(ctx, input, output); }

    [[nodiscard]] gr::work::Status findPeaks(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        constexpr std::array<std::string_view, 5> kAxisKeys{gr::tag::kAxisOrigin.key(), gr::tag::kAxisStep.key(), gr::tag::kAxisPeriodic.key(), gr::tag::kAxisUnit.key(), gr::tag::kAxisName.key()};
        const std::size_t                         nOffered = std::min(input.size(), output.size());
        std::size_t                               k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& scores = input[k];
            const std::size_t    rank   = scores.rank();
            const std::size_t    cells  = scores.size();
            std::string          refusal;
            if (rank < 1UZ || rank > gr::algorithm::peak::kMaxGridRank || cells == 0UZ || cells >= std::numeric_limits<std::uint32_t>::max()) {
                refusal = std::format("frame {} has shape {}, expects rank 1 to 3, not empty", input.streamIndex + k, scores.extents());
            }
            for (const auto& [relativeIndex, tagMap] : input.tags()) {
                const auto& tag   = tagMap.get();
                const auto  latch = [&]<typename TElement>(std::string_view key, auto& target) {
                    if (!tag.contains(key)) {
                        return;
                    }
                    const auto values = tag.template value_or<gr::Tensor<TElement>>(key, gr::Tensor<TElement>{});
                    if (values.size() != rank) {
                        refusal = std::format("{} has {} entries for a frame of rank {}", key, values.size(), rank);
                        return;
                    }
                    for (std::size_t d = 0UZ; d < rank; ++d) {
                        target[d] = values[d];
                    }
                    _axes.rank = rank;
                };
                if (std::cmp_equal(relativeIndex, k)) {
                    latch.template operator()<double>(gr::tag::kAxisOrigin.key(), _axes.origin);
                    latch.template operator()<double>(gr::tag::kAxisStep.key(), _axes.step);
                    latch.template operator()<bool>(gr::tag::kAxisPeriodic.key(), _axes.periodic);
                }
            }
            if (std::ranges::any_of(_axes.step, [](double step) { return step == 0.0 || !std::isfinite(step); })) {
                refusal = std::format("{} must be finite and non-zero", gr::tag::kAxisStep.key());
            }
            if (!refusal.empty()) [[unlikely]] {
                this->emitErrorMessage("processBulk()", refusal);
                this->requestStop();
                break;
            }
            const bool                   axes = _axes.rank == rank;
            gr::algorithm::peak::Grid<T> grid;
            grid.rank = rank;
            for (std::size_t d = 0UZ; d < rank; ++d) {
                grid.extents[d]  = scores.extent(d);
                grid.origin[d]   = axes ? static_cast<T>(_axes.origin[d]) : T{0};
                grid.step[d]     = axes ? static_cast<T>(_axes.step[d]) : T{1};
                grid.periodic[d] = axes && _axes.periodic[d];
            }
            const std::size_t maxPeaks = static_cast<std::size_t>(max_peaks.value);
            const std::size_t width    = rank + 2UZ;
            _scores.resize(cells);
            _rows.resize(maxPeaks * width);
            _accepted.resize(1UZ + maxPeaks);

            const T* const       values   = scores.data();
            T* const             peaks    = _scores.data();
            T* const             rows     = _rows.data();
            std::uint32_t* const accepted = _accepted.data();
            gr::device::parallelFor(ctx, cells, [values, peaks, grid, cells] GR_DEVICE_LAMBDA(std::size_t cell) {
                T row[gr::algorithm::peak::kMaxGridRank + 2UZ]{};
                if (gr::algorithm::peak::isPeak(values, grid, cell, cells)) {
                    gr::algorithm::peak::describePeak(values, grid, cell, cells, T{1}, row);
                    peaks[cell] = row[grid.rank];
                } else {
                    peaks[cell] = -std::numeric_limits<T>::infinity();
                }
            });
            const T           relative   = static_cast<T>(threshold);
            const std::size_t separation = static_cast<std::size_t>(static_cast<double>(min_separation) * static_cast<double>(min_separation));
            gr::device::parallelFor(ctx, 1UZ, [values, peaks, rows, accepted, grid, cells, maxPeaks, width, relative, separation] GR_DEVICE_LAMBDA(std::size_t) { gr::algorithm::peak::selectPeaks(values, peaks, grid, cells, maxPeaks, relative, separation, accepted, rows, width); });

            const std::array<std::size_t, 2> extents{static_cast<std::size_t>(_accepted[0]), width};
            auto                             found = gr::emit(*this, output, k, extents);
            if (!found) [[unlikely]] {
                break;
            }
            T* const target = found->data();
            gr::device::parallelFor(ctx, extents[0] * width, [rows, target] GR_DEVICE_LAMBDA(std::size_t n) { target[n] = rows[n]; });
            for (const auto& [relativeIndex, tagMap] : input.tags()) {
                if (std::cmp_not_equal(relativeIndex, k)) {
                    continue;
                }
                this->forwardedTag(tagMap.get(), [&](const auto& forwarded) {
                    property_map withoutAxes(forwarded);
                    for (const std::string_view key : kAxisKeys) {
                        std::ignore = withoutAxes.erase(key);
                    }
                    if (!withoutAxes.empty()) {
                        output.publishTag(withoutAxes, k);
                    }
                });
            }
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_TENSOR_PEAK_FINDER_HPP
