#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <ranges>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>

namespace {
using namespace boost::ut;
using gr::algorithm::apertureOf;
using gr::algorithm::delayOf;
using gr::algorithm::directionOf;
using gr::algorithm::elementsOf;
using gr::algorithm::Field;
using gr::algorithm::placedElements;
namespace geometry = gr::algorithm::geometry;

constexpr double kPi = std::numbers::pi;

[[nodiscard]] double distance(const gr::Tensor<double>& p, std::size_t a, std::size_t b) { return std::hypot(p[a, 0UZ] - p[b, 0UZ], p[a, 1UZ] - p[b, 1UZ], p[a, 2UZ] - p[b, 2UZ]); }

void expectCentred(const gr::Tensor<double>& p, std::string_view layout) {
    for (std::size_t c = 0UZ; c < 3UZ; ++c) {
        double sum = 0.0;
        for (std::size_t m = 0UZ; m < p.extent(0UZ); ++m) {
            sum += p[m, c];
        }
        expect(le(std::abs(sum), 1e-12)) << layout << "centroid component" << c;
    }
}

void expectRow(const gr::Tensor<double>& p, std::size_t m, const std::array<double, 3>& xyz, std::string_view layout) {
    for (std::size_t c = 0UZ; c < 3UZ; ++c) {
        expect(le(std::abs(p[m, c] - xyz[c]), 1e-12)) << layout << "element" << m << "coordinate" << c;
    }
}

const auto geometryTests = [] {
    "azimuth turns from +x towards +y, elevation from the x-y plane towards +z"_test = [] {
        const std::array<std::array<double, 3>, 4> expected{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-0.5, 0, std::sqrt(3.0) / 2.0}}};
        const std::array<std::array<double, 2>, 4> angles{{{0, 0}, {kPi / 2, 0}, {0, kPi / 2}, {kPi, kPi / 3}}};
        for (std::size_t n = 0UZ; n < angles.size(); ++n) {
            const auto u = directionOf(angles[n][0], angles[n][1]);
            for (std::size_t c = 0UZ; c < 3UZ; ++c) {
                expect(le(std::abs(u[c] - expected[n][c]), 1e-12)) << "case" << n << "component" << c;
            }
        }
    };

    "an element towards a far source hears it earlier, by its projection over the speed"_test = [] {
        const std::array<double, 3> p{3.0, 0.0, 0.0};
        expect(le(std::abs(delayOf(p.data(), directionOf(0.0, 0.0), Field::far, 1000.0) + 3e-3), 1e-15)) << "3 m towards the source at 1000 m/s: 3 ms early";
        expect(le(std::abs(delayOf(p.data(), directionOf(kPi / 2, 0.0), Field::far, 1000.0)), 1e-15)) << "broadside: no delay";
    };

    "a near source's delay is the distance over the speed"_test = [] {
        const std::array<double, 3> origin{0.0, 0.0, 0.0};
        expect(le(std::abs(delayOf(origin.data(), {3.0, 4.0, 0.0}, Field::near, 343.0) - 5.0 / 343.0), 1e-15));
    };

    "a line of M elements lies along x, `pitch` apart and centred; by default 0.3 m for two or three, 2 m for four, 1 m beyond"_test = [] {
        struct DefaultLine {
            std::size_t         elements;
            std::vector<double> x;
        };
        const std::vector<DefaultLine> kDefaults{{2UZ, {-0.15, 0.15}}, {3UZ, {-0.3, 0.0, 0.3}}, {4UZ, {-3.0, -1.0, 1.0, 3.0}}, {5UZ, {-2.0, -1.0, 0.0, 1.0, 2.0}}};
        for (const auto& line : kDefaults) {
            const auto placed = geometry::line(line.elements);
            expect(eq(placed.extent(0UZ), line.elements) && eq(placed.extent(1UZ), 3UZ));
            for (std::size_t m = 0UZ; m < line.x.size(); ++m) {
                expectRow(placed, m, {line.x[m], 0.0, 0.0}, "default line");
            }
            expect(eq(geometry::lineSpacingOf(line.elements), line.x[1UZ] - line.x[0UZ])) << line.elements << "elements";
        }
        const auto narrow = geometry::line(4UZ, 0.05);
        expect(le(std::abs(distance(narrow, 0UZ, 1UZ) - 0.05), 1e-15) && le(std::abs(distance(narrow, 2UZ, 3UZ) - 0.05), 1e-15));
        expectCentred(narrow, "line");
    };

    "a grid is row-major in the x-y plane: element r C + c at ((c - (C-1)/2) d, (r - (R-1)/2) d, 0); 0.042 m by default"_test = [] {
        const auto grid = geometry::grid(4UZ, 4UZ);
        expect(eq(grid.extent(0UZ), 16UZ));
        for (std::size_t m = 0UZ; m < 16UZ; ++m) {
            expectRow(grid, m, {(static_cast<double>(m % 4UZ) - 1.5) * 0.042, (static_cast<double>(m / 4UZ) - 1.5) * 0.042, 0.0}, "4x4 grid");
        }
        expect(le(std::abs(distance(grid, 0UZ, 1UZ) - 0.042), 1e-15)) << "along a row: the pitch";
        expect(le(std::abs(distance(grid, 0UZ, 4UZ) - 0.042), 1e-15)) << "to the next row: the pitch";
        expectCentred(grid, "4x4 grid");

        const auto wide = geometry::grid(2UZ, 3UZ, 0.5);
        expect(eq(wide.extent(0UZ), 6UZ));
        expectRow(wide, 5UZ, {0.5, 0.25, 0.0}, "2x3 grid, row 1 column 2");
        expectCentred(wide, "2x3 grid");
    };

    "a tetrahedron's six edges equal `edge` (0.3 m by default), its centroid at the origin; 0 left, 1 right and 2 top face the source, 3 sits behind"_test = [] {
        for (const double edge : {0.0, 2.0}) {
            const auto   p            = geometry::tetrahedron(edge);
            const double expectedEdge = edge > 0.0 ? edge : 0.3;
            expect(eq(p.extent(0UZ), 4UZ));
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                for (std::size_t j = i + 1UZ; j < 4UZ; ++j) {
                    expect(le(std::abs(distance(p, i, j) - expectedEdge), 1e-12)) << "edge" << i << j << "of" << expectedEdge;
                }
            }
            // a regular tetrahedron of edge a: height a sqrt(2/3), each face a quarter of it from the centroid
            const double height = expectedEdge * std::sqrt(2.0 / 3.0);
            expectRow(p, 0UZ, {-expectedEdge / 2.0, -expectedEdge / (2.0 * std::sqrt(3.0)), height / 4.0}, "left");
            expectRow(p, 1UZ, {expectedEdge / 2.0, -expectedEdge / (2.0 * std::sqrt(3.0)), height / 4.0}, "right");
            expectRow(p, 2UZ, {0.0, expectedEdge / std::sqrt(3.0), height / 4.0}, "top");
            expectRow(p, 3UZ, {0.0, 0.0, -3.0 * height / 4.0}, "back");
            expectCentred(p, "tetrahedron");
        }
    };

    "the UMA-16 places MIC1..MIC16 on the 4 x 4 grid of 42 mm as its manual draws them: MIC1 second from the left in the bottom row, MIC16 second from the right"_test = [] {
        const auto board = geometry::uma16();
        expect(eq(board.extent(0UZ), 16UZ));
        expectRow(board, 0UZ, {-0.021, -0.063, 0.0}, "MIC1");
        expectRow(board, 1UZ, {-0.063, -0.063, 0.0}, "MIC2");
        expectRow(board, 8UZ, {0.063, 0.063, 0.0}, "MIC9");
        expectRow(board, 15UZ, {0.021, -0.063, 0.0}, "MIC16");
        const auto onGridLine = [](double v) { return std::ranges::any_of(std::array{-0.063, -0.021, 0.021, 0.063}, [v](double line) { return std::abs(v - line) < 1e-12; }); };
        const auto onGrid     = [&](std::size_t m) { return onGridLine(board[m, 0UZ]) && onGridLine(board[m, 1UZ]) && board[m, 2UZ] == 0.0; };
        const auto distinct   = [&](std::size_t m) { return std::ranges::none_of(std::views::iota(0UZ, m), [&](std::size_t k) { return distance(board, m, k) < 1e-12; }); };
        const bool sameShape  = std::ranges::all_of(std::views::iota(0UZ, 16UZ), [&](std::size_t m) { return onGrid(m) && distinct(m); });
        expect(sameShape) << "the sixteen grid positions, each once";
        expectCentred(board, "uma16");
    };

    "a circle of M elements starts at +x and turns towards +y; the optional centre element comes last"_test = [] {
        const auto ring = geometry::circle(4UZ, 2.0);
        expect(eq(ring.extent(0UZ), 4UZ));
        expectRow(ring, 0UZ, {2.0, 0.0, 0.0}, "ring");
        expectRow(ring, 1UZ, {0.0, 2.0, 0.0}, "ring");
        expectRow(ring, 2UZ, {-2.0, 0.0, 0.0}, "ring");
        expectRow(ring, 3UZ, {0.0, -2.0, 0.0}, "ring");
        expectCentred(ring, "ring");

        const auto centredRing = geometry::circle(5UZ, 0.25, true);
        expect(eq(centredRing.extent(0UZ), 6UZ));
        expectRow(centredRing, 5UZ, {0.0, 0.0, 0.0}, "centre");
        for (std::size_t k = 0UZ; k < 5UZ; ++k) {
            const double angle = 2.0 * kPi * static_cast<double>(k) / 5.0;
            expectRow(centredRing, k, {0.25 * std::cos(angle), 0.25 * std::sin(angle), 0.0}, "ring of five");
            expect(le(std::abs(distance(centredRing, k, 5UZ) - 0.25), 1e-12)) << "radius of element" << k;
        }
    };

    "positions parse from x,y,z triples separated by semicolons; a malformed triple is named"_test = [] {
        const auto parsed = geometry::parsePositions(" 0, 0, 0 ; 1,0,0; -0.5 , 2e-1 , 3;");
        expect(fatal(parsed.has_value()));
        expect(eq(parsed->extent(0UZ), 3UZ) && eq(parsed->extent(1UZ), 3UZ));
        expectRow(*parsed, 1UZ, {1.0, 0.0, 0.0}, "parsed");
        expectRow(*parsed, 2UZ, {-0.5, 0.2, 3.0}, "parsed");
        for (const std::string_view bad : {"0,0", "0,0,0;1,0", "0,0,0,0", "a,b,c", "0,0,0;1,nan,0", "0, ,0", "0,0,  "}) {
            const auto refused = geometry::parsePositions(bad);
            expect(!refused.has_value()) << bad;
        }
        expect(eq(geometry::parsePositions("0,0,0;1,x,0").error().find("element 1"), 0UZ));
        expect(eq(geometry::parsePositions("").value().extent(0UZ), 0UZ)) << "no triples";
    };

    "a layout by name: line, grid, tetrahedron, circle, circle-centre and uma16; a circle needs its radius, an unknown name is refused"_test = [] {
        expect(std::ranges::equal(geometry::layoutOf("line", 4UZ, 1UZ, 1UZ, 0.0).value(), geometry::line(4UZ)));
        expect(std::ranges::equal(geometry::layoutOf("grid", 6UZ, 2UZ, 3UZ, 0.5).value(), geometry::grid(2UZ, 3UZ, 0.5)));
        expect(std::ranges::equal(geometry::layoutOf("tetrahedron", 4UZ, 1UZ, 1UZ, 0.0).value(), geometry::tetrahedron()));
        expect(std::ranges::equal(geometry::layoutOf("uma16", 16UZ, 4UZ, 4UZ, 0.0).value(), geometry::uma16()));
        expect(std::ranges::equal(geometry::layoutOf("circle", 5UZ, 1UZ, 1UZ, 0.25).value(), geometry::circle(5UZ, 0.25)));
        expect(std::ranges::equal(geometry::layoutOf("circle-centre", 5UZ, 1UZ, 1UZ, 0.25).value(), geometry::circle(4UZ, 0.25, true)));
        expect(!geometry::layoutOf("circle", 5UZ, 1UZ, 1UZ, 0.0).has_value()) << "no radius";
        expect(!geometry::layoutOf("ring", 5UZ, 1UZ, 1UZ, 0.1).has_value()) << "unknown";
    };

    "the aperture is the largest pairwise distance"_test = [] {
        expect(le(std::abs(apertureOf(geometry::line(2UZ, 0.3)) - 0.3), 1e-15));
        expect(le(std::abs(apertureOf(geometry::tetrahedron(0.3)) - 0.3), 1e-12));
        expect(le(std::abs(apertureOf(geometry::grid(2UZ, 2UZ, 1.0)) - std::sqrt(2.0)), 1e-12));
        expect(eq(apertureOf(gr::Tensor<double>({1UZ, 3UZ})), 0.0));
    };

    "positions [M,3] with 2 to 4096 finite rows and a positive speed are an array; else an error says why"_test = [] {
        expect(eq(elementsOf(gr::Tensor<double>({2UZ, 3UZ}), 343.0).value_or(0UZ), 2UZ));
        expect(!placedElements(gr::Tensor<double>({1UZ, 3UZ})).has_value()) << "one element";
        expect(!placedElements(gr::Tensor<double>({2UZ, 2UZ})).has_value()) << "two coordinates";
        expect(!placedElements(gr::Tensor<double>({4097UZ, 3UZ})).has_value()) << "above kMaxElements";
        expect(!placedElements(gr::Tensor<double>{}).has_value()) << "unset";
        gr::Tensor<double> infinite({2UZ, 3UZ});
        infinite[1UZ, 2UZ] = std::numeric_limits<double>::infinity();
        expect(!placedElements(infinite).has_value()) << "a non-finite coordinate";
        expect(!elementsOf(gr::Tensor<double>({2UZ, 3UZ}), 0.0).has_value()) << "no speed";
    };
};
} // namespace

int main() {
    "ArrayGeometry"_test = geometryTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
