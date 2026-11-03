#include <boost/ut.hpp>

#include <algorithm>
#include <complex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/beamforming/CrossSpectralMatrix.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::beamforming::CrossSpectralMatrix;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

using C = std::complex<float>;

/** two frames [2,2] (M = 2 elements, F = 2 bins) and, written out, R per bin after each at alpha 0.5 */
const std::vector<C> kFirst{C(1, 0), C(0, 2), C(0, 1), C(1, 1)};  // bin 0: x = (1, i), bin 1: x = (2i, 1+i)
const std::vector<C> kSecond{C(2, 0), C(1, 0), C(0, 0), C(0, 1)}; // bin 0: x = (2, 0), bin 1: x = (1, i)
// R_f[a,b] = x_a conj(x_b), [F,M,M]; frame 1 alone: bin 0 {1, -i; i, 1}, bin 1 {4, 2+2i; 2-2i, 2}
const std::vector<C> kAfterFirst{C(1, 0), C(0, -1), C(0, 1), C(1, 0), C(4, 0), C(2, 2), C(2, -2), C(2, 0)};
// frame 2 alone: bin 0 {4, 0; 0, 0}, bin 1 {1, -i; i, 1}; the mean of both
const std::vector<C> kAfterSecond{C(2.5f, 0), C(0, -0.5f), C(0, 0.5f), C(0.5f, 0), C(2.5f, 0), C(1, 0.5f), C(1, -0.5f), C(1.5f, 0)};

[[nodiscard]] float largestDifference(std::span<const C> a, std::span<const C> b) {
    float largest = 0.f;
    for (std::size_t n = 0UZ; n < std::min(a.size(), b.size()); ++n) {
        largest = std::max(largest, std::abs(a[n] - b[n]));
    }
    return largest;
}

void averagesTwoFrames(auto& fixture, std::string_view domain = {}) {
    auto& source   = fixture.template emplace<TensorSource<C>>();
    source._frames = {{.extents = {2UZ, 2UZ}, .elements = kFirst}, {.extents = {2UZ, 2UZ}, .elements = kSecond}};
    gr::property_map settings{{"alpha", 0.5f}};
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& covariance = fixture.template emplace<CrossSpectralMatrix<float>>(settings);
    auto& sink       = fixture.template emplace<TensorCollector<C>>();
    expect(fixture.template connect<"out", "in">(source, covariance).has_value());
    expect(fixture.template connect<"out", "in">(covariance, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(eq(sink._elements.size(), 2UZ)) << domain;
    if (sink._elements.size() == 2UZ) {
        expect(sink._shapes[0] == std::vector<std::size_t>{2UZ, 2UZ, 2UZ}) << domain;
        expect(le(largestDifference(sink._elements[0], kAfterFirst), 1e-6f)) << domain << "the first frame sets R";
        expect(le(largestDifference(sink._elements[1], kAfterSecond), 1e-6f)) << domain << "alpha 0.5 averages";
    }
}

const auto averaging = [] {
    "the first frame sets R = x x^H per bin, the next averages in at alpha"_test = [] {
        gr::testing::GraphFixture<> fixture;
        averagesTwoFrames(fixture);
    };

    "a snapshot [M] is one bin, and a change of shape restarts the average"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<C>>();
        source._frames                     = {{.extents = {2UZ, 2UZ}, .elements = kFirst}, {.extents = {2UZ}, .elements = {C(2, 0), C(0, 1)}}};
        auto& covariance                   = fixture.emplace<CrossSpectralMatrix<float>>({{"alpha", 0.5f}});
        auto& sink                         = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "in">(source, covariance).has_value());
        expect(fixture.connect<"out", "in">(covariance, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 2UZ));
        expect(sink._shapes[1] == std::vector<std::size_t>{1UZ, 2UZ, 2UZ});
        expect(le(largestDifference(sink._elements[1], std::vector<C>{C(4, 0), C(0, -2), C(0, 2), C(1, 0)}), 1e-6f)) << "x = (2, i) alone";
    };

    "a change of M at an equal number of covariance entries restarts the average"_test = [] {
        std::vector<C> second(8UZ, C(0, 0));
        second[0] = C(1, 0); // element 0, bin 0: x_0 = (1, 0, 0, 0)
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<C>>();
        source._frames                     = {{.extents = {2UZ, 8UZ}, .elements = std::vector<C>(16UZ, C(1, 1))}, {.extents = {4UZ, 2UZ}, .elements = second}};
        auto& covariance                   = fixture.emplace<CrossSpectralMatrix<std::complex<float>>>({{"alpha", 0.5f}});
        auto& sink                         = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "in">(source, covariance).has_value());
        expect(fixture.connect<"out", "in">(covariance, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink._elements.size(), 2UZ));
        if (sink._elements.size() == 2UZ) {
            std::vector<C> alone(32UZ, C(0, 0));
            alone[0] = C(1, 0); // R_0[0,0] = |x_0|^2 = 1, all else 0
            expect(sink._shapes[1] == std::vector<std::size_t>{2UZ, 4UZ, 4UZ});
            expect(le(largestDifference(sink._elements[1], alone), 1e-6f)) << "the new frame alone, not mixed with the old R";
        }
    };

    "a rank-3 frame stops the covariance"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<C>>();
        source._frames                     = {{.extents = {2UZ, 2UZ, 2UZ}, .elements = std::vector<C>(8UZ, C(1, 0))}};
        auto& covariance                   = fixture.emplace<CrossSpectralMatrix<float>>();
        auto& sink                         = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "in">(source, covariance).has_value());
        expect(fixture.connect<"out", "in">(covariance, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;
    "the covariance averages on each served domain"_domain_test = [](auto& fixture, std::string_view domain) { averagesTwoFrames(fixture, domain); } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
} // namespace

int main() {
    "CrossSpectralMatrix"_test                 = averaging;
    "CrossSpectralMatrix on every domain"_test = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
