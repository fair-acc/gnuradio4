#ifndef GNURADIO_ALGORITHM_ARRAY_GEOMETRY_HPP
#define GNURADIO_ALGORITHM_ARRAY_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::algorithm {

/**
 * far: a source by its direction (azimuth from +x towards +y, elevation from the x-y plane towards +z, in rad), its wavefront plane;
 * near: a source by its position (x, y, z in m), its wavefront spherical; free field, a non-dispersive medium
 */
enum class Field : std::uint8_t { far, near };

inline constexpr std::size_t kMaxElements = 4096UZ;

using ElementPositions = Annotated<Tensor<double>, "element_positions", Doc<"[M,3], row m the element's x, y, z; gr::algorithm::geometry places a line, grid, tetrahedron or circle centred at the origin">, Unit<"m">>;
using PropagationSpeed = Annotated<double, "propagation_speed", Doc<"of the wave in the medium, e.g. 343 in air or 299792458 in vacuum; 0 until set">, Unit<"m/s">>;
using FieldSetting     = Annotated<Field, "field", Doc<"far: directions, near: positions">>;

[[nodiscard]] constexpr double degreesOf(double radians) noexcept { return radians * 180.0 / std::numbers::pi; }
[[nodiscard]] constexpr double radiansOf(double degrees) noexcept { return degrees * std::numbers::pi / 180.0; }

struct ElementPair {
    std::size_t i;
    std::size_t j;
};

/** pair p of M elements in lexicographic order: (0,1), (0,2), ..., (0,M-1), (1,2), ...; p < M(M-1)/2 */
[[nodiscard]] GR_DEVICE_FN constexpr ElementPair pairOf(std::size_t p, std::size_t M) noexcept {
    std::size_t i = 0UZ;
    while (p >= M - 1UZ - i) {
        p -= M - 1UZ - i;
        ++i;
    }
    return {.i = i, .j = i + 1UZ + p};
}

[[nodiscard]] constexpr std::size_t pairsOf(std::size_t M) noexcept { return M * (M - 1UZ) / 2UZ; }

template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN std::array<T, 3> directionOf(T azimuth, T elevation) noexcept {
    return {std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation)};
}

/** the source named by the leading `rank` of `coordinates`, in map order (slowest first): far (azimuth) or (elevation, azimuth) in rad, near (y, x) or (z, y, x) in m */
template<std::floating_point T>
[[nodiscard]] std::array<double, 3> pointOfCoordinates(const T* coordinates, std::size_t rank, Field field) noexcept {
    const auto at = [coordinates](std::size_t c) { return static_cast<double>(coordinates[c]); };
    if (field == Field::far) {
        return rank == 1UZ ? directionOf(at(0UZ), 0.0) : directionOf(at(1UZ), at(0UZ));
    }
    return rank == 2UZ ? std::array{at(1UZ), at(0UZ), 0.0} : std::array{at(2UZ), at(1UZ), at(0UZ)};
}

/** delay in s of element p for a source at `point`: far -p^T u / c, near |q - p| / c; a pair's delay tau_i - tau_j > 0 means element i hears the source later (Chen, Benesty and Huang 2006) */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN T delayOf(const T* p, const std::array<T, 3>& point, Field field, T speed) noexcept {
    if (field == Field::far) {
        return -(p[0] * point[0] + p[1] * point[1] + p[2] * point[2]) / speed;
    }
    const T dx = point[0] - p[0];
    const T dy = point[1] - p[1];
    const T dz = point[2] - p[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) / speed;
}

/** tau_i - tau_j in s of `pair` of `positions` [M,3] (row-major) for a source at `point`: positive when element i hears it later */
template<std::floating_point T>
[[nodiscard]] GR_DEVICE_FN T pairDelayOf(const T* positions, ElementPair pair, const std::array<T, 3>& point, Field field, T speed) noexcept {
    return delayOf(positions + 3UZ * pair.i, point, field, speed) - delayOf(positions + 3UZ * pair.j, point, field, speed);
}

/** the M elements `positions` [M,3] place, 2 to kMaxElements with finite coordinates; else an error says why */
[[nodiscard]] inline std::expected<std::size_t, std::string> placedElements(const Tensor<double>& positions) {
    if (positions.rank() != 2UZ || positions.extent(1UZ) != 3UZ || positions.extent(0UZ) < 2UZ || positions.extent(0UZ) > kMaxElements) {
        return std::unexpected(std::format("element_positions have shape {}, expects [M,3] with M from 2 to {}", positions.extents(), kMaxElements));
    }
    if (!std::ranges::all_of(positions, [](double coordinate) { return std::isfinite(coordinate); })) {
        return std::unexpected("element_positions hold a non-finite coordinate");
    }
    return positions.extent(0UZ);
}

[[nodiscard]] inline std::expected<std::size_t, std::string> elementsOf(const Tensor<double>& positions, double speed) {
    const auto elements = placedElements(positions);
    if (!elements) {
        return elements;
    }
    if (!(speed > 0.0) || !std::isfinite(speed)) {
        return std::unexpected(std::format("propagation_speed {} m/s: set the wave's speed in the medium", speed));
    }
    return elements;
}

/** the largest distance between two of `positions` [M,3]; 0 for fewer than two */
[[nodiscard]] inline double apertureOf(const Tensor<double>& positions) noexcept {
    const std::size_t M        = positions.rank() == 2UZ && positions.extent(1UZ) == 3UZ ? positions.extent(0UZ) : 0UZ;
    double            aperture = 0.0;
    for (std::size_t i = 0UZ; i < M; ++i) {
        for (std::size_t j = i + 1UZ; j < M; ++j) {
            aperture = std::max(aperture, std::hypot(positions[i, 0UZ] - positions[j, 0UZ], positions[i, 1UZ] - positions[j, 1UZ], positions[i, 2UZ] - positions[j, 2UZ]));
        }
    }
    return aperture;
}

/**
 * a change of `block.element_positions` that places no array, or another element count than the frames already seen
 * (`inputElements`, 0 before the first), is refused: an error message, and the previous positions are kept
 */
template<typename TBlock>
void refuseGeometry(TBlock& block, const property_map& oldSettings, const property_map& newSettings, std::size_t inputElements) {
    constexpr std::string_view kKey = std::remove_cvref_t<decltype(block.element_positions)>::description();
    if (!newSettings.contains(kKey)) {
        return;
    }
    const auto elements = placedElements(block.element_positions.value);
    if (elements && (inputElements == 0UZ || *elements == inputElements)) {
        return;
    }
    const std::string reason = elements ? std::format("it places {} elements, the input has {}", *elements, inputElements) : elements.error();
    block.emitErrorMessage("settingsChanged()", std::format("array geometry refused, the previous one kept: {}", reason));
    block.element_positions = oldSettings.value_or<Tensor<double>>(kKey, Tensor<double>{});
}

/** element layouts as `element_positions` [M,3], centred at the origin; the line along x, planar arrays in the x-y plane facing +z */
namespace geometry {

inline constexpr double kLineSpacing       = 1.0;
inline constexpr double kStereoLineSpacing = 0.3;
inline constexpr double kQuadLineSpacing   = 2.0;
inline constexpr double kGridSpacing       = 0.042;
inline constexpr double kTetrahedronEdge   = 0.3;

[[nodiscard]] constexpr double lineSpacingOf(std::size_t M) noexcept {
    if (M <= 3UZ) {
        return kStereoLineSpacing;
    }
    return M == 4UZ ? kQuadLineSpacing : kLineSpacing;
}

[[nodiscard]] inline Tensor<double> positionsOf(std::span<const std::array<double, 3>> rows) {
    Tensor<double> positions({rows.size(), 3UZ});
    std::ranges::copy(rows | std::views::join, positions.begin());
    return positions;
}

[[nodiscard]] constexpr double centred(std::size_t n, std::size_t N) noexcept { return static_cast<double>(n) - static_cast<double>(N - 1UZ) / 2.0; }

/** M elements along x, `pitch` m apart (0: `lineSpacingOf(M)`) */
[[nodiscard]] inline Tensor<double> line(std::size_t M, double pitch = 0.0) {
    const double                       d = pitch > 0.0 ? pitch : lineSpacingOf(M);
    std::vector<std::array<double, 3>> rows(M);
    for (std::size_t m = 0UZ; m < M; ++m) {
        rows[m] = {centred(m, M) * d, 0.0, 0.0};
    }
    return positionsOf(rows);
}

/** rows x columns, row-major: element r C + c at ((c - (C-1)/2) d, (r - (R-1)/2) d, 0), `pitch` d in m (0: kGridSpacing) */
[[nodiscard]] inline Tensor<double> grid(std::size_t rows, std::size_t columns, double pitch = 0.0) {
    const double                       d = pitch > 0.0 ? pitch : kGridSpacing;
    std::vector<std::array<double, 3>> placed(rows * columns);
    for (std::size_t m = 0UZ; m < placed.size(); ++m) {
        placed[m] = {centred(m % columns, columns) * d, centred(m / columns, rows) * d, 0.0};
    }
    return positionsOf(placed);
}

/** four vertices (+-a, +-a, +-a) of even parity, every edge `edge` m (0: kTetrahedronEdge) */
[[nodiscard]] inline Tensor<double> tetrahedron(double edge = 0.0) {
    constexpr std::array<std::array<double, 3>, 4> kEvenParity{{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    const double                                   a = (edge > 0.0 ? edge : kTetrahedronEdge) / (2.0 * std::numbers::sqrt2);
    std::array<std::array<double, 3>, 4>           rows{};
    for (std::size_t m = 0UZ; m < rows.size(); ++m) {
        rows[m] = {a * kEvenParity[m][0], a * kEvenParity[m][1], a * kEvenParity[m][2]};
    }
    return positionsOf(rows);
}

/** M elements on a circle of `radius` m (> 0), element k at angle 2 pi k / M from +x towards +y; with `withCentre` one more at the origin, last */
[[nodiscard]] inline Tensor<double> circle(std::size_t M, double radius, bool withCentre = false) {
    assert(radius > 0.0);
    std::vector<std::array<double, 3>> rows(M + (withCentre ? 1UZ : 0UZ));
    for (std::size_t k = 0UZ; k < M; ++k) {
        const double angle = 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(M);
        rows[k]            = {radius * std::cos(angle), radius * std::sin(angle), 0.0};
    }
    return positionsOf(rows);
}

/** the layout a flow graph or command line names: line (M, pitch), grid (rows x columns, pitch), tetrahedron (edge pitch), circle (M on a ring of radius pitch), circle-centre (M - 1 on the ring and one at the centre) */
[[nodiscard]] inline std::expected<Tensor<double>, std::string> layoutOf(std::string_view layout, std::size_t M, std::size_t rows, std::size_t columns, double pitch) {
    if (layout == "line") {
        return line(M, pitch);
    }
    if (layout == "grid") {
        return grid(rows, columns, pitch);
    }
    if (layout == "tetrahedron") {
        return tetrahedron(pitch);
    }
    if (layout == "circle" || layout == "circle-centre") {
        if (!(pitch > 0.0)) {
            return std::unexpected(std::format("layout {} needs the ring's radius as its spacing", layout));
        }
        const bool withCentre = layout == "circle-centre";
        return circle(withCentre ? M - 1UZ : M, pitch, withCentre);
    }
    return std::unexpected(std::format("layout {}: use line, grid, tetrahedron, circle or circle-centre", layout));
}

/** "x,y,z;x,y,z;..." in m, blanks ignored, a trailing ';' allowed; an error names the first malformed triple */
[[nodiscard]] inline std::expected<Tensor<double>, std::string> parsePositions(std::string_view text) {
    const auto                         isBlank = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    std::vector<std::array<double, 3>> rows;
    for (const auto triple : text | std::views::split(';')) {
        const std::string_view tripleText(triple.begin(), triple.end());
        if (std::ranges::all_of(tripleText, isBlank)) {
            continue;
        }
        std::array<double, 3> row{};
        std::size_t           coordinates = 0UZ;
        for (const auto field : tripleText | std::views::split(',')) {
            const std::string_view padded(field.begin(), field.end());
            const auto             first = std::ranges::find_if_not(padded, isBlank);
            const auto             last  = std::ranges::find_if_not(padded | std::views::reverse, isBlank).base();
            const std::string_view fieldText(first, first < last ? last : first); // an all-blank field is empty, and then not a number
            double                 value  = 0.0;
            const auto             parsed = std::from_chars(fieldText.data(), fieldText.data() + fieldText.size(), value);
            const bool             whole  = parsed.ec == std::errc{} && parsed.ptr == fieldText.data() + fieldText.size() && std::isfinite(value);
            if (!whole || coordinates == 3UZ) {
                return std::unexpected(std::format("element {}: '{}' is not three finite coordinates x,y,z", rows.size(), tripleText));
            }
            row[coordinates++] = value;
        }
        if (coordinates != 3UZ) {
            return std::unexpected(std::format("element {}: '{}' is not three finite coordinates x,y,z", rows.size(), tripleText));
        }
        rows.push_back(row);
    }
    return positionsOf(rows);
}

} // namespace geometry

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_ARRAY_GEOMETRY_HPP
