#ifndef GNURADIO_LOCALISATION_SOURCE_LOCATION_HPP
#define GNURADIO_LOCALISATION_SOURCE_LOCATION_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/algorithm/SteeredPower.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/localisation/PairCrossSpectra.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::localisation {

/**
 * a scan grid of rank 1 to 3, row-major (the last dimension fastest): far [azimuth] or [elevation, azimuth] in rad, near [y, x] or [z, y, x]
 * in m; each dimension from its minimum to its maximum, both included, except an azimuth spanning a full turn, which excludes its end and
 * is periodic
 */
struct ScanGrid {
    gr::algorithm::peak::Grid<double> grid;
    gr::algorithm::Field              field = gr::algorithm::Field::far;

    [[nodiscard]] std::size_t cells() const noexcept { return grid.extents[0] * grid.extents[1] * grid.extents[2]; }

    /** far: the direction from the array towards cell n, near: its position */
    [[nodiscard]] std::array<double, 3> pointOf(std::size_t n) const noexcept {
        std::array<double, 3> coordinate{};
        for (std::size_t d = 0UZ; d < grid.rank; ++d) {
            coordinate[d] = grid.origin[d] + static_cast<double>(grid.indexOf(n, d)) * grid.step[d];
        }
        return gr::algorithm::pointOfCoordinates(coordinate.data(), grid.rank, field);
    }

    /** the tag naming the map's axes, one entry per dimension */
    [[nodiscard]] property_map axes() const {
        constexpr std::array<std::string_view, 2> kFar{"elevation", "azimuth"};
        constexpr std::array<std::string_view, 3> kNear{"z", "y", "x"};
        gr::Tensor<double>                        origins(gr::extents_from, {grid.rank});
        gr::Tensor<double>                        steps(gr::extents_from, {grid.rank});
        gr::Tensor<bool>                          wraps(gr::extents_from, {grid.rank});
        gr::Tensor<pmt::Value>                    units(gr::extents_from, {grid.rank});
        gr::Tensor<pmt::Value>                    names(gr::extents_from, {grid.rank});
        for (std::size_t d = 0UZ; d < grid.rank; ++d) {
            origins[d] = grid.origin[d];
            steps[d]   = grid.step[d];
            wraps[d]   = grid.periodic[d];
            units[d]   = pmt::Value{field == gr::algorithm::Field::far ? "rad" : "m"};
            names[d]   = pmt::Value{field == gr::algorithm::Field::far ? kFar[2UZ - grid.rank + d] : kNear[3UZ - grid.rank + d]};
        }
        return {{std::string(gr::tag::kAxisOrigin.key()), origins}, {std::string(gr::tag::kAxisStep.key()), steps}, {std::string(gr::tag::kAxisPeriodic.key()), wraps}, //
            {std::string(gr::tag::kAxisUnit.key()), units}, {std::string(gr::tag::kAxisName.key()), names}};
    }
};

/** the grid `minima`..`maxima` in `steps` points per dimension; an error names what is wrong */
[[nodiscard]] inline std::expected<ScanGrid, std::string> scanGridOf(gr::algorithm::Field field, const Tensor<double>& minima, const Tensor<double>& maxima, const Tensor<gr::Size_t>& steps) {
    const std::size_t rank       = steps.size();
    const bool        rankFits   = field == gr::algorithm::Field::far ? rank >= 1UZ && rank <= 2UZ : rank >= 2UZ && rank <= 3UZ;
    const bool        lengthsFit = minima.size() == rank && maxima.size() == rank;
    const bool        stepsFit   = std::ranges::all_of(steps, [](gr::Size_t n) { return n >= 1U; });
    if (!rankFits || !lengthsFit || !stepsFit) {
        return std::unexpected(std::format("grid_min, grid_max and grid_steps need one entry per dimension, 1 or 2 far, 2 or 3 near, at least one point each; got {}, {}, {}", minima.size(), maxima.size(), rank));
    }
    ScanGrid scan{.grid = {}, .field = field};
    scan.grid.rank = rank;
    for (std::size_t d = 0UZ; d < rank; ++d) {
        const double span          = maxima[d] - minima[d];
        const bool   fullTurn      = field == gr::algorithm::Field::far && d + 1UZ == rank && std::abs(span - 2.0 * std::numbers::pi) < 1e-9;
        scan.grid.extents[d]       = static_cast<std::size_t>(steps[d]);
        scan.grid.origin[d]        = minima[d];
        scan.grid.periodic[d]      = fullTurn;
        const double inclusiveStep = steps[d] > 1U ? span / static_cast<double>(steps[d] - 1U) : 1.0;
        scan.grid.step[d]          = fullTurn ? span / static_cast<double>(steps[d]) : inclusiveStep;
    }
    return scan;
}

GR_REGISTER_BLOCK(gr::localisation::SteeredPowerMap, [T], [ float, double ])

template<std::floating_point T>
struct SteeredPowerMap : Block<SteeredPowerMap<T>, Reshaping> {
    using Description = Doc<R""(@brief the steered response power over a scan grid: per grid point the sum over the element pairs of their lag curves [P,L]
(as from `PairCrossCorrelation`: lag 0 in the middle, a positive lag element i hearing later) at the pair's predicted delay, linearly interpolated.

The map is [A] or [El,Az] in the far field (rad), [Y,X] or [Z,Y,X] in the near field (m), as `grid_min`, `grid_max` and `grid_steps`
give it per dimension, slowest first (a near [Y,X] map lies in the plane z = 0); its axes go out as `gr:axis_*` tags with the first map and after each change. A delay in samples is
the delay in s times `gr:sample_rate`; a delay beyond the curve adds nothing. The elements sit at `element_positions` [M,3]: a planar grid
cannot tell its two sides apart (scan its front hemisphere, elevation 0 to pi/2), a tetrahedron resolves the full sphere at low resolution.
References: J. H. DiBiase, "A high-accuracy, low-latency technique for talker localization in reverberant environments using microphone
arrays," Ph.D. dissertation, Brown Univ., 2000; M. Cobos, A. Marti, and J. J. Lopez, "A modified SRP-PHAT functional for robust real-time
sound source localization with scalable spatial sampling," IEEE Signal Process. Lett., vol. 18, no. 1, pp. 71-74, 2011.)"">;

    TensorPortIn<T>  in;
    TensorPortOut<T> out;

    gr::algorithm::ElementPositions                                                                                                                                                     element_positions;
    gr::algorithm::PropagationSpeed                                                                                                                                                     propagation_speed = 0.0;
    gr::algorithm::FieldSetting                                                                                                                                                         field             = gr::algorithm::Field::far;
    Annotated<Tensor<double>, "grid_min", Doc<"per map dimension, slowest first: the first point (rad far, m near)">>                                                                   grid_min          = Tensor<double>(gr::data_from, {-std::numbers::pi});
    Annotated<Tensor<double>, "grid_max", Doc<"per map dimension: the last point, or the end of a full azimuth turn">>                                                                  grid_max          = Tensor<double>(gr::data_from, {std::numbers::pi});
    Annotated<Tensor<gr::Size_t>, "grid_steps", Doc<"per map dimension: the number of points">>                                                                                         grid_steps        = Tensor<gr::Size_t>(gr::data_from, {gr::Size_t(360)});
    Annotated<float, "sample_rate", Unit<"Hz">, Doc<"lags per second; forwarded from upstream">, Limits<0.f, std::numeric_limits<float>::max(), [](float rate) { return 0.f < rate; }>> sample_rate       = 1.f;

    /** [cells, P]: each pair's predicted lag in samples; reflected, so a kernel reads it in place */
    std::pmr::vector<T> _lags;

    GR_MAKE_REFLECTABLE(SteeredPowerMap, in, out, element_positions, propagation_speed, field, grid_min, grid_max, grid_steps, sample_rate, _lags);

    ScanGrid _grid;
    /** the pairs the lags were computed for; 0: recompute */
    std::size_t _pairs         = 0UZ;
    std::size_t _inputElements = 0UZ;
    bool        _axesPublished = false;

    void start() {
        _pairs         = 0UZ;
        _inputElements = 0UZ;
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return scan(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return scan(ctx, input, output); }

    [[nodiscard]] gr::work::Status scan(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& curves    = input[k];
            const std::size_t    P         = curves.rank() == 2UZ ? curves.extent(0UZ) : 0UZ;
            const std::size_t    L         = curves.rank() == 2UZ ? curves.extent(1UZ) : 0UZ;
            const bool           oddCurves = L >= 3UZ && L % 2UZ == 1UZ;
            if (!oddCurves) [[unlikely]] {
                this->emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects [P,L] with L odd, at least 3", input.streamIndex + k, curves.extents()));
                this->requestStop();
                break;
            }
            if (const auto prepared = prepare(P); !prepared) [[unlikely]] {
                this->emitErrorMessage("processBulk()", prepared.error());
                this->requestStop();
                break;
            }
            const std::size_t cells = _grid.cells();
            auto              map   = gr::emit(*this, output, k, std::span<const std::size_t>(_grid.grid.extents.data(), _grid.grid.rank));
            if (!map) [[unlikely]] {
                break;
            }
            const T* const lagCurves = curves.data();
            const T* const lags      = _lags.data();
            T* const       scores    = map->data();
            gr::device::parallelFor(ctx, cells, [lagCurves, lags, scores, P, L] GR_DEVICE_LAMBDA(std::size_t cell) { scores[cell] = gr::algorithm::computeSteeredPower(lagCurves, lags + cell * P, P, L); });
            if (!_axesPublished) {
                output.publishTag(_grid.axes(), k);
                _axesPublished = true;
            }
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        gr::algorithm::refuseGeometry(*this, oldSettings, newSettings, _inputElements);
        _pairs = 0UZ;
    }

    /** the grid and each cell's pair lags for P pairs; an error names what the settings cannot give */
    [[nodiscard]] std::expected<void, std::string> prepare(std::size_t P) {
        if (P == _pairs) {
            return {};
        }
        const auto& positions = element_positions.value;
        const auto  elements  = gr::algorithm::elementsOf(positions, propagation_speed.value);
        const auto  grid      = scanGridOf(field.value, grid_min.value, grid_max.value, grid_steps.value);
        if (!elements) {
            return std::unexpected(elements.error());
        }
        if (!grid) {
            return std::unexpected(grid.error());
        }
        if (detail::pairsOf(*elements) != P) {
            return std::unexpected(std::format("{} pairs of curves for {} elements", P, *elements));
        }
        _grid                 = *grid;
        const double* const p = positions.data();
        _lags.resize(_grid.cells() * P);
        for (std::size_t cell = 0UZ; cell < _grid.cells(); ++cell) {
            const std::array<double, 3> point = _grid.pointOf(cell);
            for (std::size_t pair = 0UZ; pair < P; ++pair) {
                const double seconds   = gr::algorithm::pairDelayOf(p, detail::pairOf(pair, *elements), point, _grid.field, propagation_speed.value);
                _lags[cell * P + pair] = static_cast<T>(seconds * static_cast<double>(sample_rate));
            }
        }
        _pairs         = P;
        _inputElements = *elements;
        _axesPublished = false;
        return {};
    }
};

struct Located {
    std::array<double, 3> point{};
    double                rmsResidual = 0.0;
    std::size_t           iterations  = 0UZ;
    std::size_t           rank        = 0UZ;
};

/**
 * the sources whose pair delays tau [P] (tau_i - tau_j in s, pairs in `pairOf` order) best match, for elements at `positions` [M,3], nearest first:
 * far, one: the least-squares direction u from (p_j - p_i)^T u = c tau, completed to unit length outside the baselines' span and normalised;
 * near and planar, the closed form (Smith and Abel): per-element delays s make |q - p_m|^2 = (c s_m + rho)^2 linear in (q, rho, |q|^2 - rho^2); one
 * direction left free (four elements in 3-D, three for planar) meets that constraint in up to two exact starts; each is refined by Gauss-Newton.
 * Outside the span (flat arrays, also far): a plane's normal, a line's z x axis (x x axis for a vertical line), towards +z, else +x, else +y
 */
[[nodiscard]] inline std::expected<std::vector<Located>, std::string> locate(const Tensor<double>& positions, std::span<const double> tau, double speed, gr::algorithm::Field field, std::size_t maxIterations, double tolerance) {
    const std::size_t M        = positions.extent(0UZ);
    const std::size_t P        = tau.size();
    const auto        position = [&](std::size_t m) { return std::array{positions[m, 0UZ], positions[m, 1UZ], positions[m, 2UZ]}; };
    const auto        residual = [&](const std::array<double, 3>& point, std::size_t p) { return gr::algorithm::pairDelayOf(positions.data(), detail::pairOf(p, M), point, field, speed) - tau[p]; };
    const auto        rmsOf    = [&](const std::array<double, 3>& point) {
        double sum = 0.0;
        for (std::size_t p = 0UZ; p < P; ++p) {
            sum += residual(point, p) * residual(point, p);
        }
        return std::sqrt(sum / static_cast<double>(P));
    };
    const auto norm  = [](const std::array<double, 3>& v) { return std::hypot(v[0], v[1], v[2]); };
    const auto cross = [](const std::array<double, 3>& a, const std::array<double, 3>& b) { return std::array{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; };
    if (!std::ranges::all_of(tau, [](double delay) { return std::isfinite(delay); })) {
        return std::unexpected("a pair delay is not finite");
    }

    Tensor<double> A({P, 3UZ});
    for (std::size_t p = 0UZ; p < P; ++p) {
        const auto ij = detail::pairOf(p, M);
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            A[p, c] = positions[ij.j, c] - positions[ij.i, c];
        }
    }
    const auto baselines = gr::math::pseudo_inverse(A);
    if (!baselines) {
        return std::unexpected(std::format("no least-squares solution: {}", gr::meta::enumName(baselines.error()).value_or("SVD failed")));
    }
    const bool flat   = baselines->rank == 1UZ || baselines->rank == 2UZ;
    const bool planar = field == gr::algorithm::Field::planar;
    if (planar && baselines->rank != 2UZ) {
        return std::unexpected(std::format("field planar needs elements spanning one plane; their baselines have rank {}", baselines->rank));
    }
    const auto outside = [&]() {
        std::array<double, 3> longest{};
        for (std::size_t a = 0UZ; a < P; ++a) {
            const std::array<double, 3> row{A[a, 0UZ], A[a, 1UZ], A[a, 2UZ]};
            for (std::size_t b = a + 1UZ; baselines->rank == 2UZ && b < P; ++b) {
                const auto normal = cross(row, std::array{A[b, 0UZ], A[b, 1UZ], A[b, 2UZ]});
                longest           = norm(normal) > norm(longest) ? normal : longest;
            }
            const auto acrossZ = cross({0.0, 0.0, 1.0}, row);
            const auto across  = norm(acrossZ) > 1e-9 * norm(row) ? acrossZ : cross({1.0, 0.0, 0.0}, row);
            longest            = baselines->rank == 1UZ && norm(across) > norm(longest) ? across : longest;
        }
        const std::array<double, 3> unit{longest[0] / norm(longest), longest[1] / norm(longest), longest[2] / norm(longest)};
        const std::array<double, 3> order{unit[2], unit[0], unit[1]};
        const auto                  side = std::ranges::find_if(order, [](double c) { return std::abs(c) > 1e-9; });
        const double                sign = side != order.end() && *side < 0.0 ? -1.0 : 1.0;
        return std::array{sign * unit[0], sign * unit[1], sign * unit[2]};
    };

    if (field == gr::algorithm::Field::far) {
        std::array<double, 3> u{};
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            for (std::size_t p = 0UZ; p < P; ++p) {
                u[c] += baselines->inverse[c, p] * speed * tau[p];
            }
        }
        if (flat) {
            const auto   direction = outside();
            const double rest      = std::sqrt(std::max(0.0, 1.0 - norm(u) * norm(u)));
            for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                u[c] += direction[c] * rest;
            }
        }
        const double length = norm(u);
        if (length > 0.0) {
            u = {u[0] / length, u[1] / length, u[2] / length};
        }
        return std::vector{Located{.point = u, .rmsResidual = rmsOf(u), .iterations = 0UZ, .rank = baselines->rank}};
    }

    std::array<double, 3> centroid{};
    for (std::size_t m = 0UZ; m < M; ++m) {
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            centroid[c] += positions[m, c] / static_cast<double>(M);
        }
    }
    double aperture = 0.0;
    for (std::size_t p = 0UZ; p < P; ++p) {
        aperture = std::max(aperture, norm({A[p, 0UZ], A[p, 1UZ], A[p, 2UZ]}));
    }
    const auto distance = [&](const std::array<double, 3>& a, const std::array<double, 3>& b) { return norm({a[0] - b[0], a[1] - b[1], a[2] - b[2]}); };
    const auto rangeOf  = [&](const std::array<double, 3>& point) { return distance(point, centroid); };

    constexpr std::size_t       kMaxStepHalvings = 40UZ;
    const std::array<double, 3> normal           = planar ? outside() : std::array<double, 3>{};
    const auto                  refine           = [&](std::array<double, 3> q) {
        Located located{};
        for (; located.iterations < maxIterations; ++located.iterations) {
            Tensor<double> J({P, 3UZ});
            for (std::size_t p = 0UZ; p < P; ++p) {
                const auto   ij = detail::pairOf(p, M);
                const auto   pi = position(ij.i);
                const auto   pj = position(ij.j);
                const double di = std::max(norm({q[0] - pi[0], q[1] - pi[1], q[2] - pi[2]}), 1e-12);
                const double dj = std::max(norm({q[0] - pj[0], q[1] - pj[1], q[2] - pj[2]}), 1e-12);
                for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                    J[p, c] = ((q[c] - pi[c]) / di - (q[c] - pj[c]) / dj) / speed;
                }
            }
            const auto inverse = gr::math::pseudo_inverse(J);
            if (!inverse) {
                break;
            }
            std::array<double, 3> stepTaken{};
            for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                for (std::size_t p = 0UZ; p < P; ++p) {
                    stepTaken[c] -= inverse->inverse[c, p] * residual(q, p);
                }
            }
            const double outOfPlane = stepTaken[0] * normal[0] + stepTaken[1] * normal[1] + stepTaken[2] * normal[2];
            for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                stepTaken[c] -= outOfPlane * normal[c];
            }
            located.rank = inverse->rank;
            // backtracking: a step is halved until it lowers the residual (J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed., Springer, 2006, ch. 3)
            const double cost     = rmsOf(q);
            std::size_t  halvings = 0UZ;
            while (halvings < kMaxStepHalvings && !(rmsOf({q[0] + stepTaken[0], q[1] + stepTaken[1], q[2] + stepTaken[2]}) < cost)) {
                stepTaken = {stepTaken[0] / 2.0, stepTaken[1] / 2.0, stepTaken[2] / 2.0};
                ++halvings;
            }
            if (halvings == kMaxStepHalvings) {
                ++located.iterations;
                break;
            }
            q = {q[0] + stepTaken[0], q[1] + stepTaken[1], q[2] + stepTaken[2]};
            if (norm(stepTaken) < tolerance) {
                ++located.iterations;
                break;
            }
        }
        // a range is resolved only while the wavefront's sag across the aperture, D^2 / 8R, exceeds the path residual c rms
        const double fitted   = rmsOf(q);
        const double resolved = fitted > 0.0 ? aperture * aperture / (8.0 * speed * fitted) : std::numeric_limits<double>::infinity();
        const double scale    = std::min(1.0, resolved / rangeOf(q));
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            located.point[c] = centroid[c] + scale * (q[c] - centroid[c]);
        }
        located.rmsResidual = rmsOf(located.point);
        return located;
    };

    Tensor<double> incidence({P, M});
    incidence.fill(0.0);
    for (std::size_t p = 0UZ; p < P; ++p) {
        const auto ij      = detail::pairOf(p, M);
        incidence[p, ij.i] = 1.0;
        incidence[p, ij.j] = -1.0;
    }
    const auto elementDelays = gr::math::pseudo_inverse(incidence);
    if (!elementDelays) {
        return std::unexpected(std::format("no per-element delays: {}", gr::meta::enumName(elementDelays.error()).value_or("SVD failed")));
    }
    std::vector<double> s(M, 0.0);
    for (std::size_t m = 0UZ; m < M; ++m) {
        for (std::size_t p = 0UZ; p < P; ++p) {
            s[m] += elementDelays->inverse[m, p] * tau[p];
        }
    }
    Tensor<double>      G({M, 5UZ});
    std::vector<double> h(M);
    for (std::size_t m = 0UZ; m < M; ++m) {
        const auto                  element = position(m);
        const std::array<double, 3> fromCentroid{element[0] - centroid[0], element[1] - centroid[1], element[2] - centroid[2]};
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            G[m, c] = -2.0 * fromCentroid[c];
        }
        G[m, 3UZ] = -2.0 * speed * s[m];
        G[m, 4UZ] = 1.0;
        h[m]      = speed * speed * s[m] * s[m] - norm(fromCentroid) * norm(fromCentroid);
    }
    const auto spherical = gr::math::pseudo_inverse(G);
    if (!spherical) {
        return std::unexpected(std::format("no closed-form start: {}", gr::meta::enumName(spherical.error()).value_or("SVD failed")));
    }
    std::array<double, 5> x{};
    for (std::size_t k = 0UZ; k < 5UZ; ++k) {
        for (std::size_t m = 0UZ; m < M; ++m) {
            x[k] += spherical->inverse[k, m] * h[m];
        }
    }
    const bool        outOfSpanFixed = planar || baselines->rank == 3UZ;
    const std::size_t freeDirections = 2UZ + baselines->rank - spherical->rank;
    const auto        starts         = [&]() -> std::vector<std::array<double, 5>> {
        if (freeDirections != 1UZ || !outOfSpanFixed) {
            return {x};
        }
        // the free direction: the longest column of the null-space projector I - G^+ G, less its out-of-span part I - A^+ A
        std::array<double, 5> n{};
        for (std::size_t column = 0UZ; column < 5UZ; ++column) {
            std::array<double, 5> nullColumn{};
            for (std::size_t row = 0UZ; row < 5UZ; ++row) {
                nullColumn[row] = row == column ? 1.0 : 0.0;
                for (std::size_t m = 0UZ; m < M; ++m) {
                    nullColumn[row] -= spherical->inverse[row, m] * G[m, column];
                }
                if (row < 3UZ && column < 3UZ) {
                    nullColumn[row] -= row == column ? 1.0 : 0.0;
                    for (std::size_t p = 0UZ; p < P; ++p) {
                        nullColumn[row] += baselines->inverse[row, p] * A[p, column];
                    }
                }
            }
            const auto length = [](const std::array<double, 5>& v) { return std::hypot(std::hypot(v[0], v[1], v[2]), std::hypot(v[3], v[4])); };
            n                 = length(nullColumn) > length(n) ? nullColumn : n;
        }
        const double a2           = n[3] * n[3] - n[0] * n[0] - n[1] * n[1] - n[2] * n[2];
        const double a1           = n[4] - 2.0 * (x[0] * n[0] + x[1] * n[1] + x[2] * n[2]) + 2.0 * x[3] * n[3];
        const double a0           = x[4] - x[0] * x[0] - x[1] * x[1] - x[2] * x[2] + x[3] * x[3];
        const double discriminant = a1 * a1 - 4.0 * a2 * a0;
        const auto   along        = [&](double alpha) {
            std::array<double, 5> shifted = x;
            for (std::size_t k = 0UZ; k < 5UZ; ++k) {
                shifted[k] += alpha * n[k];
            }
            return shifted;
        };
        if (discriminant < 0.0) {
            return {along(-a1 / (2.0 * a2))};
        }
        const double earliestDelay   = *std::ranges::min_element(s);
        const auto   noRangeNegative = [&](const std::array<double, 5>& candidate) { return candidate[3] + speed * earliestDelay >= -tolerance; };
        // roots a0 / pivot and pivot / a2 without cancellation (W. H. Press et al., Numerical Recipes, 3rd ed., Cambridge University Press, 2007, sec. 5.6)
        const double                       pivot = -0.5 * (a1 + std::copysign(std::sqrt(discriminant), a1));
        std::vector<std::array<double, 5>> roots;
        for (const double alpha : {pivot / a2, a0 / pivot}) {
            if (std::isfinite(alpha) && noRangeNegative(along(alpha))) {
                roots.push_back(along(alpha));
            }
        }
        const auto vertex = along(-a1 / (2.0 * a2));
        return !roots.empty() ? roots : std::vector{std::isfinite(vertex[0]) ? vertex : x};
    }();

    constexpr double     kOffSpanFraction = 0.05;
    constexpr double     kSamePosition    = 1e-6;
    std::vector<Located> located;
    for (const auto& start : starts) {
        std::array<double, 3> q{start[0], start[1], start[2]};
        const double          inPlane   = q[0] * normal[0] + q[1] * normal[1] + q[2] * normal[2];
        const double          outOfSpan = outOfSpanFixed ? -inPlane : std::sqrt(std::max(0.0, start[4] - norm(q) * norm(q) + start[3] * start[3]));
        const auto            direction = outOfSpanFixed ? normal : outside();
        for (std::size_t c = 0UZ; c < 3UZ; ++c) {
            q[c] += centroid[c] + outOfSpan * direction[c];
        }
        const Located refined = [&] {
            const Located fromStart = refine(q);
            if (outOfSpanFixed || outOfSpan >= kOffSpanFraction * aperture) {
                return fromStart;
            }
            // a start (nearly) in a flat array's span has no out-of-span gradient; half an aperture beside it may reach a lower residual
            const auto    side   = outside();
            const double  offset = aperture / 2.0;
            const Located beside = refine({q[0] + side[0] * offset, q[1] + side[1] * offset, q[2] + side[2] * offset});
            return beside.rmsResidual < fromStart.rmsResidual ? beside : fromStart;
        }();
        const auto same = [&](const Located& other) { return distance(other.point, refined.point) < kSamePosition * aperture; };
        if (std::ranges::none_of(located, same)) {
            located.push_back(refined);
        }
    }
    std::ranges::sort(located, {}, [&](const Located& candidate) { return rangeOf(candidate.point); });
    return located;
}

GR_REGISTER_BLOCK(gr::localisation::SourceFromDelays, [T], [ float, double ])

template<std::floating_point T>
struct SourceFromDelays : Block<SourceFromDelays<T>, Reshaping> {
    using Description = Doc<R""(@brief the sources that best explain the element pairs' delays: per frame [P,C] (column 0 each pair's delay in s, as from
`PairDelay`) rows [K, D+3] = {coordinates, RMS delay residual in s, iterations, rank}, nearest first: K = 2 where two positions fit exactly, else 1.

Coordinates in map order, slowest first, as `TensorPeakFinder` gives a peak of a map: far, elevation and azimuth in rad (D = 2), the
least-squares direction; a planar array cannot tell the two sides of its plane and takes the side its normal points to with positive z
(rank 2); a linear one knows only the cone angle about its axis and takes the direction on the cone along z x axis (x x axis for a
vertical line), oriented towards +z, else +x, else +y (rank 1). Near, z, y and x in m (D = 3): the closed-form least-squares position,
refined by Gauss-Newton with backtracking until a step moves less than `tolerance` or after `max_iterations`, outwards no farther than the
residual resolves the wavefront's curvature across the aperture D (range D^2 / 8 c rms); a planar or linear array takes the side the far
field takes. Four elements spanning 3-D fit two positions exactly for many sources outside them; five or more fit one. Planar, as near
with the position in the elements' plane, refused unless they span one. Azimuth in [-pi, pi).
Runs on the host: the solve uses `gr::math::pseudo_inverse`.
References: W. H. Foy, "Position-location solutions by Taylor-series estimation," IEEE Trans. Aerosp. Electron. Syst., vol. AES-12, no. 2,
pp. 187-194, 1976; J. O. Smith and J. S. Abel, "Closed-form least-squares source location estimation from range-difference measurements,"
IEEE Trans. Acoust., Speech, Signal Process., vol. 35, no. 12, pp. 1661-1669, 1987.)"">;

    TensorPortIn<T>  in;
    TensorPortOut<T> out;

    gr::algorithm::ElementPositions                                                                                       element_positions;
    gr::algorithm::PropagationSpeed                                                                                       propagation_speed = 0.0;
    gr::algorithm::FieldSetting                                                                                           field             = gr::algorithm::Field::far;
    Annotated<gr::Size_t, "max_iterations", Doc<"near, planar: most Gauss-Newton steps">, Limits<1U, 1000U>>              max_iterations    = 20U;
    Annotated<double, "tolerance", Unit<"m">, Doc<"near, planar: any shorter step ends the iteration">, Limits<0.0, 1e6>> tolerance         = 1e-6;

    GR_MAKE_REFLECTABLE(SourceFromDelays, in, out, element_positions, propagation_speed, field, max_iterations, tolerance);

    std::size_t _inputElements = 0UZ;

    void start() { _inputElements = 0UZ; }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<T>& delays         = input[k];
            const auto&          positions      = element_positions.value;
            const auto           elements       = gr::algorithm::elementsOf(positions, propagation_speed.value);
            const std::size_t    P              = delays.rank() == 2UZ && delays.extent(1UZ) >= 1UZ ? delays.extent(0UZ) : 0UZ;
            const auto           firstLagOfPair = [&](std::size_t p) { return static_cast<double>(delays.data()[p * delays.extent(1UZ)]); };
            const auto           tau            = std::views::iota(0UZ, P) | std::views::transform(firstLagOfPair) | std::ranges::to<std::vector<double>>();
            const auto           located        = [&]() -> std::expected<std::vector<Located>, std::string> {
                if (!elements) {
                    return std::unexpected(elements.error());
                }
                if (P == 0UZ || detail::pairsOf(*elements) != P) {
                    return std::unexpected(std::format("frame {} has shape {}, expects [P,C], P the pairs of the {} elements", input.streamIndex + k, delays.extents(), *elements));
                }
                return locate(positions, tau, propagation_speed.value, field.value, static_cast<std::size_t>(max_iterations.value), tolerance.value);
            }();
            if (!located) [[unlikely]] {
                this->emitErrorMessage("processBulk()", located.error());
                this->requestStop();
                break;
            }
            const std::size_t                D = field.value == gr::algorithm::Field::far ? 2UZ : 3UZ;
            const std::array<std::size_t, 2> extents{located->size(), D + 3UZ};
            auto                             rows = gr::emit(*this, output, k, extents);
            if (!rows) [[unlikely]] {
                break;
            }
            for (std::size_t r = 0UZ; r < located->size(); ++r) {
                const Located&     candidate = (*located)[r];
                const std::span<T> row       = rows->subspan(r * (D + 3UZ), D + 3UZ);
                const auto&        u         = candidate.point;
                if (field.value == gr::algorithm::Field::far) {
                    row[0]                      = static_cast<T>(std::asin(std::clamp(u[2], -1.0, 1.0)));
                    const double azimuth        = std::atan2(u[1], u[0]);
                    const double wrappedAzimuth = azimuth >= std::numbers::pi ? -std::numbers::pi : azimuth;
                    row[1]                      = static_cast<T>(wrappedAzimuth);
                } else {
                    std::ranges::transform(u | std::views::reverse, row.begin(), [](double c) { return static_cast<T>(c); });
                }
                row[D]       = static_cast<T>(candidate.rmsResidual);
                row[D + 1UZ] = static_cast<T>(candidate.iterations);
                row[D + 2UZ] = static_cast<T>(candidate.rank);
            }
            _inputElements = *elements;
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) { gr::algorithm::refuseGeometry(*this, oldSettings, newSettings, _inputElements); }
};

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_SOURCE_LOCATION_HPP
