#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/beamforming/Beamformer.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/TensorSource.hpp>

namespace {
using namespace boost::ut;
using gr::algorithm::Field;
using gr::algorithm::pointOfCoordinates;
using gr::beamforming::binFrequency;
using gr::beamforming::ConventionalBeamformer;
using gr::beamforming::MinimumVarianceBeamformer;
using gr::beamforming::NullingBeamformer;
using gr::beamforming::SteeringVectors;
using gr::testing::TensorCollector;
using gr::testing::TensorSource;

constexpr double kPi = std::numbers::pi;
using C              = std::complex<float>;

const auto functions = [] {
    "bin frequencies: one-sided for real samples, signed plus the carrier for complex ones, the carrier for a snapshot"_test = [] {
        expect(eq(binFrequency<float>(2UZ, 5UZ, false, 8.0, 100.0), 2.0)) << "F 5, N 8 at 8 Hz: bin 2 at 2 Hz, no carrier";
        expect(eq(binFrequency<std::complex<float>>(3UZ, 4UZ, false, 4.0, 100.0), 99.0)) << "N 4 at 4 Hz: bin 3 at -1 Hz, plus 100";
        expect(eq(binFrequency<std::complex<float>>(0UZ, 1UZ, true, 4.0, 100.0), 100.0)) << "snapshot";
        expect(eq(binFrequency<float>(0UZ, 1UZ, false, 8.0, 100.0), 0.0)) << "a real spectrum of one bin has no samples to place it by";
    };

    "a peak row's coordinates in map order give the source"_test = [] {
        const std::array<double, 3> azimuth{kPi / 2, 9, 1};
        const std::array<double, 3> elevationAzimuth{kPi / 2, 0, 9};
        const std::array<double, 3> yx{1, 2, 9};
        const std::array<double, 3> zyx{3, 1, 2};
        const auto                  far1 = pointOfCoordinates(azimuth.data(), 1UZ, Field::far);
        const auto                  far2 = pointOfCoordinates(elevationAzimuth.data(), 2UZ, Field::far);
        expect(le(std::abs(far1[0]) + std::abs(far1[1] - 1.0) + std::abs(far1[2]), 1e-12)) << "azimuth pi/2: +y";
        expect(le(std::abs(far2[0]) + std::abs(far2[1]) + std::abs(far2[2] - 1.0), 1e-12)) << "elevation pi/2: +z";
        expect(pointOfCoordinates(yx.data(), 2UZ, Field::near) == std::array{2.0, 1.0, 0.0}) << "(y, x) = (1, 2)";
        expect(pointOfCoordinates(zyx.data(), 3UZ, Field::near) == std::array{2.0, 1.0, 3.0}) << "(z, y, x) = (3, 1, 2)";
    };
};

[[nodiscard]] gr::property_map pairOfElements() { return {{"element_positions", gr::Tensor<double>(std::array<std::size_t, 2>{2UZ, 3UZ}, std::vector<double>{0, 0, 0, 1, 0, 0})}}; }

/**
 * spectra [2,3] of real samples at 4 Hz (bins 0, 1 and the Nyquist bin 2 Hz), elements 1 m apart along x, waves at 8 m/s, a peak at azimuth 0
 * (element 1 hears it 1/8 s early): element 1's vector exp(+j 2 pi f / 8) is 1, (1 + j)/sqrt(2), j; element 0's all 1; the peak emitted first
 */
void steeringOn(auto& fixture, std::string_view domain = {}) {
    auto& peaks               = fixture.template emplace<TensorSource<float>>();
    peaks._frames             = {{.extents = {1UZ, 3UZ}, .elements = {0.f, 1.f, 1.f}}};
    auto&      spectra        = fixture.template emplace<TensorSource<C>>();
    const auto frame          = typename TensorSource<C>::Frame{.extents = {2UZ, 3UZ}, .elements = std::vector<C>(6UZ, C{1.f, 0.f})};
    spectra._frames           = {frame, frame, frame};
    gr::property_map settings = pairOfElements();
    settings.merge(gr::property_map{{"propagation_speed", 8.0}, {"sample_rate", 4.f}});
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& steering = fixture.template emplace<SteeringVectors<float>>(settings);
    auto& sink     = fixture.template emplace<TensorCollector<C>>();
    expect(fixture.template connect<"out", "peaks">(peaks, steering).has_value());
    expect(fixture.template connect<"out", "in">(spectra, steering).has_value());
    expect(fixture.template connect<"out", "in">(steering, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    expect(fatal(eq(sink._elements.size(), 3UZ))) << domain;
    expect(sink._shapes.back() == std::vector<std::size_t>{1UZ, 3UZ, 2UZ}) << domain << "the last frame is steered: [K,F,M]";
    const float            r = std::sqrt(0.5f);
    const std::array<C, 6> expected{C{1, 0}, C{1, 0}, C{1, 0}, C{r, r}, C{1, 0}, C{0, 1}};
    for (std::size_t n = 0UZ; n < expected.size(); ++n) {
        expect(le(std::abs(sink._elements.back()[n] - expected[n]), 1e-6f)) << domain << "bin" << n / 2UZ << "element" << n % 2UZ;
    }
}

template<typename TSample>
[[nodiscard]] std::vector<C> steeredOnce(typename TensorSource<C>::Frame spectra, gr::property_map extra) {
    gr::testing::GraphFixture<> fixture;
    auto&                       peaks = fixture.emplace<TensorSource<float>>();
    peaks._frames                     = {{.extents = {1UZ, 3UZ}, .elements = {0.f, 1.f, 1.f}}};
    auto& source                      = fixture.emplace<TensorSource<C>>();
    source._frames                    = {spectra};
    gr::property_map settings         = pairOfElements();
    settings.merge(gr::property_map{{"propagation_speed", 4.0}, {"sample_rate", 4.f}});
    for (const auto& [key, value] : extra) {
        settings.insert_or_assign(key, value);
    }
    auto& steering = fixture.emplace<SteeringVectors<TSample>>(settings);
    auto& sink     = fixture.emplace<TensorCollector<C>>();
    expect(fixture.connect<"out", "peaks">(peaks, steering).has_value());
    expect(fixture.connect<"out", "in">(source, steering).has_value());
    expect(fixture.connect<"out", "in">(steering, sink).has_value());
    expect(fixture.run().has_value());
    return sink._elements.empty() ? std::vector<C>{} : sink._elements.back();
}

void expectVectors(const std::vector<C>& found, const std::vector<C>& expected, std::string_view context) {
    expect(fatal(eq(found.size(), expected.size()))) << context;
    for (std::size_t n = 0UZ; n < expected.size(); ++n) {
        expect(le(std::abs(found[n] - expected[n]), 1e-4f)) << context << "entry" << n;
    }
}

template<typename TFixture>
[[nodiscard]] std::string messagesOfStoppedRun(TFixture& fixture) {
    gr::MsgPortIn fromBlock;
    expect(fatal(fixture.scheduler().msgOut.connect(fromBlock).has_value()));
    std::ignore = fixture.run();
    std::string       errors;
    const std::size_t available = fromBlock.streamReader().available();
    if (available != 0UZ) {
        auto span = fromBlock.streamReader().get(available);
        for (const gr::Message& message : span) {
            if (!message.data.has_value()) {
                errors += message.data.error().message + "\n";
            }
        }
        std::ignore = span.consume(available);
    }
    return errors;
}

const auto block = [] {
    "the selected peak's vectors per bin, from the frame the selection reaches on"_test = [] {
        gr::testing::GraphFixture<> fixture;
        steeringOn(fixture);
    };

    "without a selection every frame gets [0,F,M]"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       source = fixture.emplace<TensorSource<C>>();
        source._frames                     = {{.extents = {2UZ, 3UZ}, .elements = std::vector<C>(6UZ)}};
        gr::property_map settings          = pairOfElements();
        settings.insert_or_assign("propagation_speed", 4.0);
        auto& steering = fixture.emplace<SteeringVectors<float>>(settings);
        auto& sink     = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "in">(source, steering).has_value());
        expect(fixture.connect<"out", "in">(steering, sink).has_value());
        expect(fixture.run().has_value());
        expect(fatal(eq(sink._shapes.size(), 1UZ)));
        expect(sink._shapes[0] == std::vector<std::size_t>{0UZ, 3UZ, 2UZ});
    };

    "complex samples steer the carrier plus each bin's signed offset; a snapshot the carrier"_test = [] {
        // element 1: exp(j 2 pi f / 4); bins at 100 + 0, 1, -2, -1 Hz: 1, j, -1, -j
        const auto full = steeredOnce<std::complex<float>>({.extents = {2UZ, 4UZ}, .elements = std::vector<C>(8UZ)}, {{"frequency", 100.0}});
        expectVectors(full, {C{1, 0}, C{1, 0}, C{1, 0}, C{0, 1}, C{1, 0}, C{-1, 0}, C{1, 0}, C{0, -1}}, "complex, carrier 100 Hz");
        const auto snapshot = steeredOnce<std::complex<float>>({.extents = {2UZ}, .elements = std::vector<C>(2UZ)}, {{"frequency", 101.0}});
        expectVectors(snapshot, {C{1, 0}, C{0, 1}}, "snapshot at 101 Hz");
    };

    "a change of geometry steers the latest selection again"_test = [] {
        SteeringVectors<float> steering;
        steering.element_positions = gr::Tensor<double>(std::array<std::size_t, 2>{2UZ, 3UZ}, std::vector<double>{0, 0, 0, 1, 0, 0});
        steering.propagation_speed = 4.0;
        const std::vector<float> peak{0.f, 1.f, 1.f};
        expect(fatal(steering.placeSources(peak, 1UZ, 3UZ).has_value()));
        steering._selection        = peak;
        steering._selectionColumns = 3UZ;
        expect(le(std::abs(steering._delays[1] + 0.25f), 1e-7f)) << "1 m at 4 m/s";
        steering.propagation_speed = 2.0;
        steering.settingsChanged({}, {});
        expect(le(std::abs(steering._delays[1] + 0.5f), 1e-7f)) << "1 m at 2 m/s";
        steering.propagation_speed = 0.0;
        steering.settingsChanged({}, {});
        expect(eq(steering._sources, 0UZ)) << "no speed: the selection places no source";
    };

    "a refused geometry keeps the previous positions: another element count than the input's, or a shape that is not [M,3]"_test = [] {
        gr::property_map settings = pairOfElements();
        settings.merge(gr::property_map{{"propagation_speed", 4.0}});
        SteeringVectors<float> steering(settings);
        steering.init(steering.progress);
        const auto expectKept = [&steering](std::string_view context) { expect(std::ranges::equal(steering.element_positions.value, gr::Tensor<double>(std::array<std::size_t, 2>{2UZ, 3UZ}, std::vector<double>{0, 0, 0, 1, 0, 0}))) << context; };
        expectKept("as constructed");
        steering._inputElements = 2UZ;
        std::ignore             = steering.settings().setStaged({{"element_positions", gr::algorithm::geometry::grid(3UZ, 5UZ, 0.1)}});
        std::ignore             = steering.settings().applyStagedParameters();
        expectKept("a 3x5 grid for an input of 2");
        std::ignore = steering.settings().setStaged({{"element_positions", gr::Tensor<double>({2UZ, 2UZ})}});
        std::ignore = steering.settings().applyStagedParameters();
        expectKept("two coordinates per element");
        std::ignore = steering.settings().setStaged({{"element_positions", gr::algorithm::geometry::line(2UZ, 0.1)}});
        std::ignore = steering.settings().applyStagedParameters();
        expect(std::ranges::equal(steering.element_positions.value, gr::algorithm::geometry::line(2UZ, 0.1))) << "a line of two for an input of 2: accepted";
    };

    "a selection of the wrong column count stops the block with a message"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       peaks   = fixture.emplace<TensorSource<float>>();
        auto&                       spectra = fixture.emplace<TensorSource<C>>();
        peaks._frames                       = {{.extents = {1UZ, 1UZ}, .elements = {0.f}}};
        spectra._frames                     = {{.extents = {2UZ, 3UZ}, .elements = std::vector<C>(6UZ)}};
        gr::property_map settings           = pairOfElements();
        settings.merge(gr::property_map{{"propagation_speed", 4.0}, {"coordinates", gr::Size_t(2)}});
        auto& steering = fixture.emplace<SteeringVectors<float>>(settings);
        auto& sink     = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "peaks">(peaks, steering).has_value());
        expect(fixture.connect<"out", "in">(spectra, steering).has_value());
        expect(fixture.connect<"out", "in">(steering, sink).has_value());
        const std::string errors = messagesOfStoppedRun(fixture);
        expect(errors.contains("peaks [1, 1] for 2 coordinates")) << errors;
        expect(sink._elements.empty());
    };

    "a spectra frame whose M differs from the elements stops the block with a message"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       peaks   = fixture.emplace<TensorSource<float>>();
        auto&                       spectra = fixture.emplace<TensorSource<C>>();
        peaks._frames                       = {{.extents = {1UZ, 3UZ}, .elements = {0.f, 1.f, 1.f}}};
        spectra._frames                     = {{.extents = {3UZ, 3UZ}, .elements = std::vector<C>(9UZ)}};
        gr::property_map settings           = pairOfElements();
        settings.merge(gr::property_map{{"propagation_speed", 4.0}});
        auto& steering = fixture.emplace<SteeringVectors<float>>(settings);
        auto& sink     = fixture.emplace<TensorCollector<C>>();
        expect(fixture.connect<"out", "peaks">(peaks, steering).has_value());
        expect(fixture.connect<"out", "in">(spectra, steering).has_value());
        expect(fixture.connect<"out", "in">(steering, sink).has_value());
        const std::string errors = messagesOfStoppedRun(fixture);
        expect(errors.contains("for the 2 elements")) << errors;
        expect(sink._elements.empty());
    };
};

const auto steeringVectorsDomains = [] {
    using gr::testing::operator""_domain_test;
    "steering vectors on each served domain"_domain_test = [](auto& fixture, std::string_view domain) { steeringOn(fixture, domain); } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

using Cd    = std::complex<double>;
using Frame = TensorSource<Cd>::Frame;

// three elements 1 m apart along x, waves at 4 m/s, at 1 Hz: towards azimuth 0 element m hears the wave m/4 s early, a_t = (1, j, -1);
// broadside, a_i = (1, 1, 1)
const std::array<Cd, 3> kTarget{Cd{1, 0}, Cd{0, 1}, Cd{-1, 0}};
const std::array<Cd, 3> kInterferer{Cd{1, 0}, Cd{1, 0}, Cd{1, 0}};

[[nodiscard]] std::vector<Cd> concat(std::initializer_list<std::array<Cd, 3>> parts) {
    std::vector<Cd> all;
    for (const auto& part : parts) {
        all.insert(all.end(), part.begin(), part.end());
    }
    return all;
}

/** R = 10 a_i a_i^H + I, the interferer 10 dB above the noise, as [1,3,3], `bins` times */
[[nodiscard]] std::vector<Cd> interferenceCovariance(std::size_t bins = 1UZ) {
    std::vector<Cd> R(9UZ);
    for (std::size_t a = 0UZ; a < 3UZ; ++a) {
        for (std::size_t b = 0UZ; b < 3UZ; ++b) {
            R[a * 3UZ + b] = 10.0 * kInterferer[a] * std::conj(kInterferer[b]) + (a == b ? Cd{1, 0} : Cd{});
        }
    }
    std::vector<Cd> all;
    for (std::size_t f = 0UZ; f < bins; ++f) {
        all.insert(all.end(), R.begin(), R.end());
    }
    return all;
}

template<typename TBeamformer>
[[nodiscard]] std::vector<std::vector<Cd>> beamsOn(auto& fixture, std::vector<std::array<Cd, 3>> snapshots, std::vector<Cd> steering, std::size_t K, gr::property_map settings, std::string_view domain) {
    if (!domain.empty()) {
        settings.insert_or_assign("gr:compute_domain", std::string(domain) + "!");
    }
    auto& spectra  = fixture.template emplace<TensorSource<Cd>>();
    auto& vectors  = fixture.template emplace<TensorSource<Cd>>();
    auto& beamform = fixture.template emplace<TBeamformer>(settings);
    auto& sink     = fixture.template emplace<TensorCollector<Cd>>();
    for (const auto& snapshot : snapshots) {
        spectra._frames.push_back(Frame{.extents = {3UZ}, .elements = {snapshot.begin(), snapshot.end()}});
        vectors._frames.push_back(Frame{.extents = {K, 1UZ, 3UZ}, .elements = steering});
    }
    expect(fixture.template connect<"out", "in">(spectra, beamform).has_value());
    expect(fixture.template connect<"out", "steering">(vectors, beamform).has_value());
    if constexpr (std::remove_cvref_t<decltype(beamform)>::kUsesCovariance) {
        auto& covariance = fixture.template emplace<TensorSource<Cd>>();
        for (std::size_t n = 0UZ; n < snapshots.size(); ++n) {
            covariance._frames.push_back(Frame{.extents = {1UZ, 3UZ, 3UZ}, .elements = interferenceCovariance()});
        }
        expect(fixture.template connect<"out", "covariance">(covariance, beamform).has_value());
    }
    expect(fixture.template connect<"out", "in">(beamform, sink).has_value());
    const auto ran = fixture.run();
    expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);
    return sink._elements;
}

template<typename TBeamformer>
[[nodiscard]] std::vector<std::vector<Cd>> beamsOf(std::vector<std::array<Cd, 3>> snapshots, std::vector<Cd> steering, std::size_t K, gr::property_map settings = {}) {
    gr::testing::GraphFixture<> fixture;
    return beamsOn<TBeamformer>(fixture, std::move(snapshots), std::move(steering), K, std::move(settings), {});
}

void expectBeams(const std::vector<std::vector<Cd>>& beams, const std::vector<std::vector<Cd>>& expected, std::string_view context) {
    expect(fatal(eq(beams.size(), expected.size()))) << context;
    for (std::size_t frame = 0UZ; frame < expected.size(); ++frame) {
        expect(fatal(eq(beams[frame].size(), expected[frame].size()))) << context;
        for (std::size_t k = 0UZ; k < expected[frame].size(); ++k) {
            expect(le(std::abs(beams[frame][k] - expected[frame][k]), 1e-9)) << context << "frame" << frame << "beam" << k << "got" << beams[frame][k].real() << beams[frame][k].imag();
        }
    }
}

[[nodiscard]] std::vector<Cd> rankOneCovariance() {
    std::vector<Cd> R(9UZ);
    R[0UZ] = Cd{1, 0};
    return R;
}

const auto beamforming = [] {
    "delay-and-sum passes the steered wave unchanged and averages a broadside one"_test = [] {
        // a_t^H a_i / 3 = (1 - j - 1) / 3 = -j/3
        const auto beams = beamsOf<ConventionalBeamformer<double>>({kTarget, kInterferer}, concat({kTarget}), 1UZ);
        expectBeams(beams, {{Cd{1, 0}}, {Cd{0, -1.0 / 3.0}}}, "DSB");
    };

    "MVDR keeps the target at unit gain and suppresses the interferer 10 dB above the noise to -j/83"_test = [] {
        // Sherman-Morrison: R^-1 = I - 10 a_i a_i^H / 31; a_t^H R^-1 a_t = 83/31; w^H a_i = (a_t^H a_i + 30j/31) / (83/31) = -j/83
        const auto beams = beamsOf<MinimumVarianceBeamformer<double>>({kTarget, kInterferer}, concat({kTarget}), 1UZ, {{"diagonal_loading", 0.f}});
        expectBeams(beams, {{Cd{1, 0}}, {Cd{0, -1.0 / 83.0}}}, "MVDR");
    };

    "LCMV gives each source unit gain in its beam and none in the other's"_test = [] {
        const auto beams = beamsOf<NullingBeamformer<double>>({kTarget, kInterferer}, concat({kTarget, kInterferer}), 2UZ, {{"diagonal_loading", 0.f}});
        expectBeams(beams, {{Cd{1, 0}, Cd{0, 0}}, {Cd{0, 0}, Cd{1, 0}}}, "LCMV");
    };

    "MVDR loaded by its mean diagonal: R' = 10 a_i a_i^H + 12 I gives the interferer -3j/29"_test = [] {
        // trace 33, M 3, loading 1: + 11 I; R'^-1 = (I - 10 a_i a_i^H / 42) / 12; a_t^H R'^-1 a_i = -j/42, a_t^H R'^-1 a_t = 116/504
        const auto beams = beamsOf<MinimumVarianceBeamformer<double>>({kTarget, kInterferer}, concat({kTarget}), 1UZ, {{"diagonal_loading", 1.f}});
        expectBeams(beams, {{Cd{1, 0}}, {Cd{0, -3.0 / 29.0}}}, "loaded MVDR");
    };

    "two bins, each its own spectrum, steering vector and covariance"_test = [] {
        using TBeamformer = MinimumVarianceBeamformer<double>;
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                     = {Frame{.extents = {3UZ, 2UZ}, .elements = {kTarget[0], kInterferer[0], kTarget[1], kInterferer[1], kTarget[2], kInterferer[2]}}};
        vectors._frames                     = {Frame{.extents = {1UZ, 2UZ, 3UZ}, .elements = concat({kTarget, kTarget})}};
        auto& beamform                      = fixture.emplace<TBeamformer>({{"diagonal_loading", 0.f}});
        auto& sink                          = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        auto& covariance   = fixture.emplace<TensorSource<Cd>>();
        covariance._frames = {Frame{.extents = {2UZ, 3UZ, 3UZ}, .elements = interferenceCovariance(2UZ)}};
        expect(fixture.connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        expect(fixture.run().has_value());
        expectBeams(sink._elements, {{Cd{1, 0}, Cd{0, -1.0 / 83.0}}}, "[1,2]");
    };

    "a bin without power is beamformed as by delay-and-sum: w = a / M"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra    = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors    = fixture.emplace<TensorSource<Cd>>();
        auto&                       covariance = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                        = {Frame{.extents = {3UZ}, .elements = {kTarget.begin(), kTarget.end()}}, Frame{.extents = {3UZ}, .elements = {kInterferer.begin(), kInterferer.end()}}};
        vectors._frames                        = {Frame{.extents = {1UZ, 1UZ, 3UZ}, .elements = concat({kTarget})}, Frame{.extents = {1UZ, 1UZ, 3UZ}, .elements = concat({kTarget})}};
        covariance._frames                     = {Frame{.extents = {1UZ, 3UZ, 3UZ}, .elements = std::vector<Cd>(9UZ)}, Frame{.extents = {1UZ, 3UZ, 3UZ}, .elements = std::vector<Cd>(9UZ)}};
        auto& beamform                         = fixture.emplace<MinimumVarianceBeamformer<double>>();
        auto& sink                             = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        expect(fixture.run().has_value());
        expectBeams(sink._elements, {{Cd{1, 0}}, {Cd{0, -1.0 / 3.0}}}, "silence");
    };

    "an adaptive beamformer stops with a message naming the bin when the covariance cannot be solved"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra    = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors    = fixture.emplace<TensorSource<Cd>>();
        auto&                       covariance = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                        = {Frame{.extents = {3UZ, 2UZ}, .elements = {kTarget[0], kTarget[0], kTarget[1], kTarget[1], kTarget[2], kTarget[2]}}};
        vectors._frames                        = {Frame{.extents = {1UZ, 2UZ, 3UZ}, .elements = concat({kTarget, kTarget})}};
        std::vector<Cd> twoBins                = interferenceCovariance();
        const auto      deficient              = rankOneCovariance();
        twoBins.insert(twoBins.end(), deficient.begin(), deficient.end());
        covariance._frames = {Frame{.extents = {2UZ, 3UZ, 3UZ}, .elements = twoBins}};
        auto& beamform     = fixture.emplace<MinimumVarianceBeamformer<double>>({{"diagonal_loading", 0.f}});
        auto& sink         = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        const std::string errors = messagesOfStoppedRun(fixture);
        expect(errors.contains("bin 1: the loaded covariance cannot be solved")) << errors;
        expect(!errors.contains("bin 0")) << "the well-conditioned bin is not blamed";
        expect(sink._elements.empty()) << "no beams of a frame with an unsolvable bin";
    };

    "an adaptive beamformer stops with a message naming the source when a steering vector is zero"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra    = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors    = fixture.emplace<TensorSource<Cd>>();
        auto&                       covariance = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                        = {Frame{.extents = {3UZ}, .elements = {kTarget.begin(), kTarget.end()}}};
        vectors._frames                        = {Frame{.extents = {2UZ, 1UZ, 3UZ}, .elements = {kTarget[0], kTarget[1], kTarget[2], Cd{}, Cd{}, Cd{}}}};
        covariance._frames                     = {Frame{.extents = {1UZ, 3UZ, 3UZ}, .elements = interferenceCovariance()}};
        auto& beamform                         = fixture.emplace<MinimumVarianceBeamformer<double>>();
        auto& sink                             = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        const std::string errors = messagesOfStoppedRun(fixture);
        expect(errors.contains("source 1:")) << errors;
        expect(!errors.contains("source 0")) << "the usable source is not blamed";
        expect(sink._elements.empty());
    };

    "MVDR without a covariance connection stops"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                     = {Frame{.extents = {3UZ}, .elements = {kTarget.begin(), kTarget.end()}}};
        vectors._frames                     = {Frame{.extents = {1UZ, 1UZ, 3UZ}, .elements = concat({kTarget})}};
        auto& beamform                      = fixture.emplace<MinimumVarianceBeamformer<double>>();
        auto& sink                          = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        const std::string errors = messagesOfStoppedRun(fixture);
        expect(errors.contains("needs the covariance")) << errors;
        expect(sink._elements.empty());
    };

    "LCMV refuses more sources than elements"_test = [] {
        const std::vector<Cd>       four = concat({kTarget, kInterferer, kTarget, kInterferer});
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra    = fixture.emplace<TensorSource<Cd>>();
        auto&                       vectors    = fixture.emplace<TensorSource<Cd>>();
        auto&                       covariance = fixture.emplace<TensorSource<Cd>>();
        spectra._frames                        = {Frame{.extents = {3UZ}, .elements = {kTarget.begin(), kTarget.end()}}};
        vectors._frames                        = {Frame{.extents = {4UZ, 1UZ, 3UZ}, .elements = four}};
        covariance._frames                     = {Frame{.extents = {1UZ, 3UZ, 3UZ}, .elements = interferenceCovariance()}};
        auto& beamform                         = fixture.emplace<NullingBeamformer<double>>();
        auto& sink                             = fixture.emplace<TensorCollector<Cd>>();
        expect(fixture.connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.connect<"out", "in">(beamform, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };

    "steering vectors for another number of elements stop the beams"_test = [] {
        gr::testing::GraphFixture<> fixture;
        auto&                       spectra = fixture.template emplace<TensorSource<Cd>>();
        auto&                       vectors = fixture.template emplace<TensorSource<Cd>>();
        spectra._frames                     = {Frame{.extents = {3UZ}, .elements = {kTarget.begin(), kTarget.end()}}};
        vectors._frames                     = {Frame{.extents = {1UZ, 1UZ, 2UZ}, .elements = {Cd{1, 0}, Cd{1, 0}}}};
        auto& beamform                      = fixture.template emplace<ConventionalBeamformer<double>>();
        auto& sink                          = fixture.template emplace<TensorCollector<Cd>>();
        expect(fixture.template connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.template connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.template connect<"out", "in">(beamform, sink).has_value());
        expect(!fixture.run().has_value());
        expect(sink._elements.empty());
    };
};

const auto beamformerDomains = [] {
    using gr::testing::operator""_domain_test;
    "delay-and-sum on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto beams = beamsOn<ConventionalBeamformer<double>>(fixture, {kTarget, kInterferer}, concat({kTarget}), 1UZ, {}, domain);
        expectBeams(beams, {{Cd{1, 0}}, {Cd{0, -1.0 / 3.0}}}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "MVDR on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto beams = beamsOn<MinimumVarianceBeamformer<double>>(fixture, {kTarget, kInterferer}, concat({kTarget}), 1UZ, {{"diagonal_loading", 0.f}}, domain);
        expectBeams(beams, {{Cd{1, 0}}, {Cd{0, -1.0 / 83.0}}}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "LCMV drops a singular bin and preserves neighbouring independent beams"_domain_test = [](auto& fixture, std::string_view domain) {
        auto& spectra      = fixture.template emplace<TensorSource<Cd>>();
        auto& vectors      = fixture.template emplace<TensorSource<Cd>>();
        auto& covariance   = fixture.template emplace<TensorSource<Cd>>();
        spectra._frames    = {Frame{.extents = {2UZ, 2UZ}, .elements = {Cd{3}, Cd{3}, Cd{3}, Cd{1}}}};
        vectors._frames    = {Frame{.extents = {2UZ, 2UZ, 2UZ}, .elements = {Cd{1}, Cd{1}, Cd{1}, Cd{1}, Cd{1}, Cd{1}, Cd{1}, Cd{-1}}}};
        covariance._frames = {Frame{.extents = {2UZ, 2UZ, 2UZ}, .elements = {Cd{1}, Cd{}, Cd{}, Cd{1}, Cd{1}, Cd{}, Cd{}, Cd{1}}}};
        auto& beamform     = fixture.template emplace<NullingBeamformer<double>>();
        auto& sink         = fixture.template emplace<TensorCollector<Cd>>();
        expect(fixture.template connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.template connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.template connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.template connect<"out", "in">(beamform, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran ? std::string{} : ran.error().message);
        expectBeams(sink._elements, {{Cd{}, Cd{2}, Cd{}, Cd{1}}}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "float LCMV rejects unresolved pivots and preserves independent constraints at different scales"_domain_test = [](auto& fixture, std::string_view domain) {
        using Cf                     = std::complex<float>;
        using FloatFrame             = typename TensorSource<Cf>::Frame;
        const float       unresolved = std::sqrt(std::numeric_limits<float>::epsilon());
        constexpr float   kResolved  = 1.f / 1024.f;
        constexpr float   kScaled    = 1.f / 65536.f;
        std::array<Cf, 4> gram{Cf{1}, Cf{1}, Cf{1}, Cf{1.f + unresolved * unresolved}};
        expect(gr::math::cholesky_factor(gram.data(), 2UZ));
        auto& spectra      = fixture.template emplace<TensorSource<Cf>>();
        auto& vectors      = fixture.template emplace<TensorSource<Cf>>();
        auto& covariance   = fixture.template emplace<TensorSource<Cf>>();
        spectra._frames    = {FloatFrame{.extents = {2UZ, 3UZ}, .elements = {Cf{3}, Cf{3}, Cf{2}, Cf{unresolved}, Cf{kResolved}, Cf{3.f * kScaled}}}};
        vectors._frames    = {FloatFrame{.extents = {2UZ, 3UZ, 2UZ}, .elements = {Cf{1}, Cf{}, Cf{1}, Cf{}, Cf{1}, Cf{}, Cf{1}, Cf{unresolved}, Cf{1}, Cf{kResolved}, Cf{}, Cf{kScaled}}}};
        covariance._frames = {FloatFrame{.extents = {3UZ, 2UZ, 2UZ}, .elements = {Cf{1}, Cf{}, Cf{}, Cf{1}, Cf{1}, Cf{}, Cf{}, Cf{1}, Cf{1}, Cf{}, Cf{}, Cf{1}}}};
        auto& beamform     = fixture.template emplace<NullingBeamformer<float>>(gr::property_map{{"diagonal_loading", 0.f}});
        auto& sink         = fixture.template emplace<TensorCollector<Cf>>();
        expect(fixture.template connect<"out", "in">(spectra, beamform).has_value());
        expect(fixture.template connect<"out", "steering">(vectors, beamform).has_value());
        expect(fixture.template connect<"out", "covariance">(covariance, beamform).has_value());
        expect(fixture.template connect<"out", "in">(beamform, sink).has_value());
        const auto ran = fixture.run();
        expect(ran.has_value()) << domain << (ran ? std::string{} : ran.error().message);
        expect(fatal(eq(sink._elements.size(), 1UZ)));
        const std::array expected{Cf{}, Cf{2}, Cf{2}, Cf{}, Cf{1}, Cf{3}};
        expect(fatal(eq(sink._elements[0].size(), expected.size())));
        for (std::size_t k = 0UZ; k < expected.size(); ++k) {
            expect(le(std::abs(sink._elements[0][k] - expected[k]), 1e-5f)) << domain << k;
        }
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "LCMV zeros dependent constraints on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto beams = beamsOn<NullingBeamformer<double>>(fixture, {kTarget, kInterferer}, concat({kTarget, kTarget}), 2UZ, {{"diagonal_loading", 0.05f}}, domain);
        expectBeams(beams, {{Cd{}, Cd{}}, {Cd{}, Cd{}}}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);

    "LCMV on each served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        const auto beams = beamsOn<NullingBeamformer<double>>(fixture, {kTarget, kInterferer}, concat({kTarget, kInterferer}), 2UZ, {{"diagonal_loading", 0.f}}, domain);
        expectBeams(beams, {{Cd{1, 0}, Cd{0, 0}}, {Cd{0, 0}, Cd{1, 0}}}, domain);
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};
} // namespace

int main() {
    "functions"_test                       = functions;
    "SteeringVectors"_test                 = block;
    "SteeringVectors on every domain"_test = steeringVectorsDomains;
    "beamformers"_test                     = beamforming;
    "beamformers on every domain"_test     = beamformerDomains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
