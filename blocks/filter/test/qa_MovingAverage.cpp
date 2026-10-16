#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/filter/MovingAverage.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace {
using namespace boost::ut;
using gr::filter::Statistic;

constexpr std::size_t kStatistics = 11UZ;

// spelled out apart from gr::filter::detail::kFoldsPower and its siblings: the oracle must not share predicates with the code under test
template<Statistic statistic>
constexpr bool kOfPower = statistic == Statistic::meanSquare || statistic == Statistic::rms || statistic == Statistic::logPower;
template<Statistic statistic>
constexpr bool kOfSpread = statistic == Statistic::variance || statistic == Statistic::unbiasedVariance || statistic == Statistic::stdDev || statistic == Statistic::unbiasedStdDev;
template<Statistic statistic>
constexpr bool kOfExtreme = statistic == Statistic::minimum || statistic == Statistic::maximum;
template<Statistic statistic>
constexpr bool kBesselCorrected = statistic == Statistic::unbiasedVariance || statistic == Statistic::unbiasedStdDev;

template<typename T>
[[nodiscard]] std::vector<T> noisyFrames(std::size_t nSamples, double offset, unsigned seed = 42U) {
    std::mt19937                     generator{seed};
    std::normal_distribution<double> noise{0., 1.};
    std::vector<T>                   samples(nSamples);
    for (auto& sample : samples) {
        if constexpr (gr::meta::complex_like<T>) {
            sample = T{static_cast<typename T::value_type>(offset + noise(generator)), static_cast<typename T::value_type>(-offset + noise(generator))};
        } else {
            sample = static_cast<T>(offset + noise(generator));
        }
    }
    return samples;
}

template<typename T, Statistic statistic, bool exponential>
[[nodiscard]] std::vector<double> expectedFigures(const std::vector<T>& samples, std::size_t frameSize, std::size_t window, std::size_t decimation) {
    constexpr bool        kComplex    = gr::meta::complex_like<T>;
    constexpr std::size_t kComponents = kComplex && !kOfPower<statistic> ? 2UZ : 1UZ;
    const std::size_t     nFrames     = samples.size() / frameSize;
    const double          alpha       = 2. / static_cast<double>(window + 1UZ);
    const auto            componentAt = [&](std::size_t frame, std::size_t position, std::size_t component) {
        const auto sample = samples[frame * frameSize + position];
        double     folded = 0.;
        if constexpr (kComplex) {
            const double re = static_cast<double>(sample.real());
            const double im = static_cast<double>(sample.imag());
            folded          = kComponents == 1UZ ? re * re + im * im : (component == 0UZ ? re : im);
        } else {
            folded = static_cast<double>(sample);
            if constexpr (kOfPower<statistic>) {
                folded *= folded;
            }
        }
        if constexpr (statistic == Statistic::logPower) {
            folded = 10. * std::log10(std::max(folded, 1e-30));
        }
        return folded;
    };

    std::vector<double> figures;
    std::vector<double> centre(frameSize * kComponents, 0.);
    std::vector<double> spread(frameSize * kComponents, 0.);
    for (std::size_t frame = 0UZ; frame < nFrames; ++frame) {
        const std::size_t   first    = frame + 1UZ >= window ? frame + 1UZ - window : 0UZ;
        const std::size_t   inWindow = frame - first + 1UZ;
        std::vector<double> perComponent(frameSize * kComponents);
        for (std::size_t position = 0UZ; position < frameSize; ++position) {
            for (std::size_t component = 0UZ; component < kComponents; ++component) {
                const std::size_t lane = component * frameSize + position;
                if constexpr (exponential) {
                    const double sample = componentAt(frame, position, component);
                    const double weight = std::max(alpha, 1. / static_cast<double>(std::min(frame, window) + 1UZ));
                    if constexpr (kOfExtreme<statistic>) {
                        const bool takes = frame == 0UZ || (statistic == Statistic::maximum ? sample >= centre[lane] : sample <= centre[lane]);
                        centre[lane]     = takes ? sample : centre[lane] + weight * (sample - centre[lane]);
                    } else {
                        const double delta = sample - centre[lane];
                        centre[lane] += weight * delta;
                        spread[lane] = (1. - weight) * (spread[lane] + weight * delta * delta);
                    }
                    perComponent[lane] = kOfSpread<statistic> ? spread[lane] : (statistic == Statistic::sum ? centre[lane] * static_cast<double>(inWindow) : centre[lane]);
                } else {
                    double sum     = 0.;
                    double extreme = componentAt(first, position, component);
                    for (std::size_t f = first; f <= frame; ++f) {
                        sum += componentAt(f, position, component);
                        extreme = statistic == Statistic::maximum ? std::max(extreme, componentAt(f, position, component)) : std::min(extreme, componentAt(f, position, component));
                    }
                    const double mean    = sum / static_cast<double>(inWindow);
                    double       squares = 0.;
                    for (std::size_t f = first; f <= frame; ++f) {
                        squares += (componentAt(f, position, component) - mean) * (componentAt(f, position, component) - mean);
                    }
                    const std::size_t degrees = kBesselCorrected<statistic> ? inWindow - 1UZ : inWindow;
                    if constexpr (kOfSpread<statistic>) {
                        perComponent[lane] = degrees == 0UZ ? 0. : squares / static_cast<double>(degrees);
                    } else if constexpr (kOfExtreme<statistic>) {
                        perComponent[lane] = extreme;
                    } else {
                        perComponent[lane] = statistic == Statistic::sum ? sum : mean;
                    }
                }
            }
        }
        if ((frame + 1UZ) % decimation != 0UZ) {
            continue;
        }
        for (std::size_t position = 0UZ; position < frameSize; ++position) {
            if constexpr (kComplex && (statistic == Statistic::mean || statistic == Statistic::sum)) {
                figures.push_back(perComponent[position]);
                figures.push_back(perComponent[frameSize + position]);
            } else {
                double figure = perComponent[position] + (kComponents == 2UZ ? perComponent[frameSize + position] : 0.);
                if constexpr (statistic == Statistic::rms || statistic == Statistic::stdDev || statistic == Statistic::unbiasedStdDev) {
                    figure = std::sqrt(std::max(figure, 0.));
                }
                figures.push_back(figure);
            }
        }
    }
    return figures;
}

[[nodiscard]] std::vector<double> flattened(const auto& outputs) {
    std::vector<double> figures;
    for (const auto& output : outputs) {
        if constexpr (gr::meta::complex_like<std::remove_cvref_t<decltype(output)>>) {
            figures.push_back(static_cast<double>(output.real()));
            figures.push_back(static_cast<double>(output.imag()));
        } else {
            figures.push_back(static_cast<double>(output));
        }
    }
    return figures;
}

template<typename T, Statistic statistic, bool exponential>
[[nodiscard]] std::vector<double> computedFigures(const std::vector<T>& samples, std::size_t frameSize, std::size_t window, std::size_t decimation, std::size_t outFramesPerCall) {
    using Block = gr::filter::Average<T, statistic, exponential>;
    Block block({{"frame_size", static_cast<gr::Size_t>(frameSize)}, {"n_frames", static_cast<gr::Size_t>(window)}, {"decim", static_cast<gr::Size_t>(decimation)}});
    block.init(block.progress);

    const std::size_t                 nOut = samples.size() / (frameSize * decimation);
    std::vector<typename Block::TOut> outputs(nOut * frameSize);
    for (std::size_t done = 0UZ; done < nOut; done += outFramesPerCall) {
        const std::size_t now = std::min(outFramesPerCall, nOut - done);
        expect(gr::work::Status::OK == block.processBulk(std::span<const T>(samples).subspan(done * decimation * frameSize, now * decimation * frameSize), std::span(outputs).subspan(done * frameSize, now * frameSize)));
    }
    return flattened(outputs);
}

[[nodiscard]] bool closeTo(const std::vector<double>& computed, const std::vector<double>& expected, double tolerance, std::string& firstMiss) {
    if (computed.size() != expected.size()) {
        firstMiss = std::format("{} figures, expected {}", computed.size(), expected.size());
        return false;
    }
    for (std::size_t i = 0UZ; i < computed.size(); ++i) {
        if (std::abs(computed[i] - expected[i]) > tolerance * std::max(1., std::abs(expected[i]))) {
            firstMiss = std::format("figure {}: {} vs {}", i, computed[i], expected[i]);
            return false;
        }
    }
    return true;
}

template<typename T, bool exponential, Statistic statistic>
void expectDefinition(std::size_t frameSize, std::size_t window, std::size_t decimation) {
    if constexpr (gr::filter::detail::Averageable<T, statistic, exponential>) {
        const auto        samples   = noisyFrames<T>(frameSize * 60UZ, 3.);
        const auto        expected  = expectedFigures<T, statistic, exponential>(samples, frameSize, window, decimation);
        const double      tolerance = std::same_as<gr::meta::fundamental_base_value_type_t<T>, float> ? 2e-4 : 1e-9;
        const std::string what      = std::format("{} {} of {}, frame {}, window {}, decimation {}", exponential ? "exponential" : "moving", gr::meta::enumName(statistic).value_or("?"), gr::meta::type_name<T>(), frameSize, window, decimation);
        for (const std::size_t perCall : {1UZ, 1000UZ}) {
            std::string firstMiss;
            expect(closeTo(computedFigures<T, statistic, exponential>(samples, frameSize, window, decimation, perCall), expected, tolerance, firstMiss)) << what << std::format(", {} out frames per call: ", perCall) << firstMiss;
        }
    }
}

template<typename T, bool exponential>
void expectEveryStatistic(std::size_t frameSize, std::size_t window, std::size_t decimation) {
    [&]<std::size_t... kIndex>(std::index_sequence<kIndex...>) { (expectDefinition<T, exponential, static_cast<Statistic>(kIndex)>(frameSize, window, decimation), ...); }(std::make_index_sequence<kStatistics>{});
}

const auto definitions = [] {
    "every statistic is its definition, window by window, however the stream is cut"_test = [] {
        expectEveryStatistic<float, false>(19UZ, 4UZ, 1UZ);
        expectEveryStatistic<float, false>(19UZ, 4UZ, 3UZ);
        expectEveryStatistic<float, false>(19UZ, 4UZ, 4UZ);
        expectEveryStatistic<std::complex<float>, false>(7UZ, 3UZ, 3UZ);
        expectEveryStatistic<double, false>(5UZ, 7UZ, 1UZ);
        expectEveryStatistic<std::complex<float>, false>(19UZ, 4UZ, 2UZ);
        expectEveryStatistic<std::complex<double>, false>(3UZ, 1UZ, 1UZ);
    };

    "every exponential statistic is its recursion, however the stream is cut"_test = [] {
        expectEveryStatistic<float, true>(19UZ, 4UZ, 1UZ);
        expectEveryStatistic<float, true>(19UZ, 9UZ, 4UZ);
        expectEveryStatistic<double, true>(5UZ, 7UZ, 1UZ);
        expectEveryStatistic<std::complex<float>, true>(19UZ, 4UZ, 2UZ);
    };

    "a sliding extreme falls back once the extreme leaves the window"_test = [] {
        const std::vector<float> ramp{1.f, 5.f, 2.f, 3.f, 1.f, 0.f, 4.f};
        const auto               maxima = computedFigures<float, Statistic::maximum, false>(ramp, 1UZ, 3UZ, 1UZ, 1UZ);
        expect(maxima == std::vector<double>{1., 5., 5., 5., 3., 3., 4.});
        const auto minima = computedFigures<float, Statistic::minimum, false>(ramp, 1UZ, 3UZ, 1UZ, 1UZ);
        expect(minima == std::vector<double>{1., 1., 1., 2., 1., 0., 0.});
    };

    "before the window fills, the frames seen so far are the window"_test = [] {
        const std::vector<float> steps{2.f, 4.f, 6.f, 8.f, 10.f};
        expect(computedFigures<float, Statistic::mean, false>(steps, 1UZ, 4UZ, 1UZ, 1UZ) == std::vector<double>{2., 3., 4., 5., 7.});
        expect(computedFigures<float, Statistic::mean, true>(steps, 1UZ, 4UZ, 1UZ, 1UZ).front() == 2.) << "no start-up bias towards zero";
        expect(computedFigures<float, Statistic::unbiasedVariance, false>(steps, 1UZ, 4UZ, 1UZ, 1UZ).front() == 0.) << "one frame has no spread to correct";
    };

    "a complex mean keeps the phase: the averaged cross-spectrum of two shifted tones"_test = [] {
        std::vector<std::complex<float>> crossSpectrum(4UZ * 16UZ);
        for (std::size_t frame = 0UZ; frame < 16UZ; ++frame) {
            for (std::size_t bin = 0UZ; bin < 4UZ; ++bin) {
                crossSpectrum[frame * 4UZ + bin] = std::polar(1.f, 0.5f * static_cast<float>(bin)); // x conj(y) of a fixed lag
            }
        }
        const auto averaged = computedFigures<std::complex<float>, Statistic::mean, false>(crossSpectrum, 4UZ, 8UZ, 8UZ, 2UZ);
        expect(eq(averaged.size(), 2UZ * 2UZ * 4UZ));
        for (std::size_t bin = 0UZ; bin < 4UZ; ++bin) {
            expect(std::abs(std::atan2(averaged[2UZ * bin + 1UZ], averaged[2UZ * bin]) - 0.5 * static_cast<double>(bin)) < 1e-5) << "bin " << bin;
        }
    };

    "an exponential mean of a unit step closes in as 1 - (1 - alpha)^k once its window has filled"_test = [] {
        constexpr std::size_t kWindow = 9UZ; // alpha = 2 / (9 + 1) = 0.2
        std::vector<float>    step(40UZ, 0.f);
        std::fill(step.begin() + 20, step.end(), 1.f);
        const auto means = computedFigures<float, Statistic::mean, true>(step, 1UZ, kWindow, 1UZ, 1UZ);
        for (std::size_t k = 1UZ; k <= 20UZ; ++k) {
            expect(std::abs(means[19UZ + k] - (1. - std::pow(0.8, static_cast<double>(k)))) < 1e-6) << "k = " << k;
        }
        expect(computedFigures<float, Statistic::mean, true>(std::vector<float>(5UZ, 3.f), 1UZ, kWindow, 1UZ, 1UZ) == std::vector<double>(5UZ, 3.)) << "a constant is its own mean from the first frame";
    };

    "a long dumped window keeps float input exact to its own precision: it is summed in double"_test = [] {
        constexpr std::size_t kWindow = 1UZ << 16UZ;
        std::vector<float>    ramp(kWindow);
        double                squares = 0.;
        for (std::size_t i = 0UZ; i < kWindow; ++i) {
            ramp[i] = static_cast<float>(i);
            squares += static_cast<double>(i) * static_cast<double>(i);
        }
        const auto rms = computedFigures<float, Statistic::rms, false>(ramp, 1UZ, kWindow, kWindow, 1UZ);
        expect(fatal(eq(rms.size(), 1UZ)));
        expect(std::abs(rms[0] / std::sqrt(squares / static_cast<double>(kWindow)) - 1.) < 1e-6) << std::format("RMS of a 2^16-sample ramp off by {}", rms[0] / std::sqrt(squares / static_cast<double>(kWindow)) - 1.);
    };

    "an exponential variance of white noise settles at 2 (1 - alpha) / (2 - alpha) sigma^2"_test = [] {
        constexpr std::size_t kPositions = 4096UZ;
        constexpr std::size_t kWindow    = 9UZ; // alpha = 0.2
        const auto            noise      = noisyFrames<float>(kPositions * 200UZ, 0.);
        const auto            variances  = computedFigures<float, Statistic::variance, true>(noise, kPositions, kWindow, 1UZ, 200UZ);
        const double          alpha      = 2. / static_cast<double>(kWindow + 1UZ);
        double                mean       = 0.;
        for (std::size_t p = 0UZ; p < kPositions; ++p) {
            mean += variances[199UZ * kPositions + p] / static_cast<double>(kPositions);
        }
        expect(std::abs(mean - 2. * (1. - alpha) / (2. - alpha)) < 0.03) << std::format("mean over {} positions of the settled variance: {}", kPositions, mean);
    };

    "the power of silence averages to -300 dB, not to minus infinity"_test = [] {
        expect(computedFigures<float, Statistic::logPower, false>(std::vector<float>(4UZ, 0.f), 1UZ, 2UZ, 1UZ, 1UZ) == std::vector<double>(4UZ, -300.));
        expect(computedFigures<double, Statistic::logPower, true>(std::vector<double>(4UZ, 0.), 1UZ, 2UZ, 1UZ, 1UZ) == std::vector<double>(4UZ, -300.)) << "the same floor for both precisions";
    };

    "reset empties the window"_test = [] {
        gr::filter::MovingAverage<float> block({{"n_frames", gr::Size_t(4)}});
        block.init(block.progress);
        std::vector<float>       averaged(2UZ);
        const std::vector<float> high{10.f, 20.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(high), std::span(averaged)));
        block.reset();
        const std::vector<float> low{1.f, 2.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(low), std::span(averaged)));
        expect(averaged == std::vector<float>{1.f, 1.5f});
    };

    "a window too large to hold is refused, not allocated, and the previous one kept"_test = [] {
        gr::filter::MovingAverage<float> block({{"frame_size", 2U}, {"n_frames", 4U}});
        block.init(block.progress);
        expect(block.settings().set({{"frame_size", 1U << 20U}, {"n_frames", 1U << 20U}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();

        expect(eq(block.frame_size.value, 2U));
        expect(eq(block.n_frames.value, 4U));
        std::vector<float>       averaged(2UZ);
        const std::vector<float> frame{1.f, 3.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(frame), std::span(averaged))) << "the block runs on with the window it holds";
        expect(averaged == frame);

        gr::filter::MovingAverage<float> tooLarge({{"frame_size", 1U << 20U}, {"n_frames", 1U << 20U}});
        tooLarge.init(tooLarge.progress);
        expect(eq(tooLarge.frame_size.value, 1U)) << "a refused initial window keeps the defaults";
        expect(eq(tooLarge.n_frames.value, 16U));
    };

    "a long run does not drift: the window is re-summed once per pass"_test = [] {
        const auto  samples  = noisyFrames<float>(200'000UZ, 1000.);
        const auto  computed = computedFigures<float, Statistic::variance, false>(samples, 1UZ, 64UZ, 1UZ, 4096UZ);
        const auto  expected = expectedFigures<float, Statistic::variance, false>(samples, 1UZ, 64UZ, 1UZ);
        std::string firstMiss;
        expect(closeTo(std::vector(computed.end() - 1000, computed.end()), std::vector(expected.end() - 1000, expected.end()), 1e-2, firstMiss)) << "unit variance on an offset of 1000, after 200k samples: " << firstMiss;
    };

    "a new window length restarts the average"_test = [] {
        gr::filter::MovingAverage<float> block({{"n_frames", gr::Size_t(2)}});
        block.init(block.progress);
        std::vector<float> first{10.f, 20.f};
        std::vector<float> averaged(2UZ);
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(first), std::span<float>(averaged)));
        std::ignore = block.settings().setStaged({{"n_frames", gr::Size_t(3)}});
        std::ignore = block.settings().applyStagedParameters();
        std::vector<float> second{1.f, 2.f};
        expect(gr::work::Status::OK == block.processBulk(std::span<const float>(second), std::span<float>(averaged)));
        expect(averaged == std::vector<float>{1.f, 1.5f}) << "nothing of the earlier window survives" << std::format(" got {} with n_frames {}", averaged, block.n_frames.value);
    };

    "integrate and dump: one figure per window, the sample rate follows, a tag survives"_test = [] {
        constexpr float      kInputRate = 48000.f;
        constexpr gr::Size_t kWindow    = 64U;

        gr::Graph flow({{"auto_size_edges_to_chunks", true}});
        auto&     source = flow.emplaceBlock<gr::testing::ProbeSource<float>>({{"n_samples_max", gr::Size_t(4096)}, {"sample_rate", kInputRate}, {"mark_tag", false}});
        source._tags     = {{512UZ, {{"key", "mid_stream"}}}};
        auto& rms        = flow.emplaceBlock<gr::filter::MovingAverage<float, Statistic::rms>>({{"n_frames", kWindow}, {"decim", kWindow}});
        auto& sink       = flow.emplaceBlock<gr::testing::ProbeSink<float>>({{"log_samples", true}});
        expect(flow.connect<"out", "in">(source, rms).has_value());
        expect(flow.connect<"out", "in">(rms, sink).has_value());
        gr::scheduler::Simple<> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.runAndWait().has_value());

        expect(eq(sink._samples.size(), 4096UZ / kWindow));
        const auto matchesClosedForm = [&](std::size_t window) {
            double squares = 0.;
            for (std::size_t i = 0UZ; i < kWindow; ++i) {
                squares += static_cast<double>(window * kWindow + i) * static_cast<double>(window * kWindow + i);
            }
            return std::abs(static_cast<double>(sink._samples[window]) - std::sqrt(squares / kWindow)) < 1e-2 * std::sqrt(squares / kWindow);
        };
        expect(std::ranges::all_of(std::views::iota(0UZ, sink._samples.size()), matchesClosedForm)) << "each figure is the RMS of its own window";
        expect(std::ranges::any_of(sink._tags, [](const auto& tag) { return tag.map.contains("key"); }));
        expect(std::abs(sink.sample_rate - kInputRate / static_cast<float>(kWindow)) < 1.f) << std::format("sink saw {} Hz", sink.sample_rate);
    };
};

const auto domains = [] {
    using gr::testing::operator""_domain_test;

    "spectra average the same on every served domain"_domain_test = [](auto& fixture, std::string_view domain) {
        constexpr std::size_t kFrame  = 256UZ;
        constexpr std::size_t kWindow = 8UZ;
        gr::property_map      settings{{"frame_size", static_cast<gr::Size_t>(kFrame)}, {"n_frames", static_cast<gr::Size_t>(kWindow)}, {"decim", gr::Size_t(2)}, {"gr:compute_domain", std::string(domain) + "!"}};
        gr::property_map      dumpSettings = settings;
        dumpSettings["decim"]              = static_cast<gr::Size_t>(kWindow);

        const auto spectra       = noisyFrames<std::complex<float>>(kFrame * 24UZ, 2.);
        const auto power         = noisyFrames<float>(kFrame * 24UZ, 5.);
        auto&      spectraSource = fixture.template emplace<gr::testing::ProbeSource<std::complex<float>>>({{"values", spectra}, {"n_samples_max", static_cast<gr::Size_t>(spectra.size())}, {"mark_tag", false}});
        auto&      powerSource   = fixture.template emplace<gr::testing::ProbeSource<float>>({{"values", power}, {"n_samples_max", static_cast<gr::Size_t>(power.size())}, {"mark_tag", false}});
        const auto averaged      = [&fixture]<typename TBlock>(auto& source, const gr::property_map& blockSettings) -> auto& {
            auto& average = fixture.template emplace<TBlock>(blockSettings);
            auto& sink    = fixture.template emplace<gr::testing::ProbeSink<typename TBlock::TOut>>({{"log_samples", true}});
            expect(fixture.template connect<"out", "in">(source, average).has_value());
            expect(fixture.template connect<"out", "in">(average, sink).has_value());
            return sink;
        };
        auto& mean                           = averaged.template operator()<gr::filter::MovingAverage<std::complex<float>>>(spectraSource, settings);
        auto& spread                         = averaged.template operator()<gr::filter::MovingAverage<std::complex<float>, Statistic::unbiasedStdDev>>(spectraSource, settings);
        auto& hold                           = averaged.template operator()<gr::filter::MovingAverage<float, Statistic::maximum>>(powerSource, settings);
        auto& decay                          = averaged.template operator()<gr::filter::ExponentialAverage<float, Statistic::logPower>>(powerSource, settings);
        auto& dumped                         = averaged.template operator()<gr::filter::MovingAverage<float, Statistic::variance>>(powerSource, dumpSettings);
        const auto                       ran = fixture.run();
        expect(ran.has_value()) << domain << (ran.has_value() ? std::string{} : ran.error().message);

        const auto check = [&domain](const auto& sink, const std::vector<double>& expected, std::string_view what) {
            std::string firstMiss;
            expect(closeTo(flattened(sink._samples), expected, 2e-4, firstMiss)) << domain << " " << what << ": " << firstMiss;
        };
        check(mean, expectedFigures<std::complex<float>, Statistic::mean, false>(spectra, kFrame, kWindow, 2UZ), "coherent mean");
        check(spread, expectedFigures<std::complex<float>, Statistic::unbiasedStdDev, false>(spectra, kFrame, kWindow, 2UZ), "complex unbiased std dev");
        check(hold, expectedFigures<float, Statistic::maximum, false>(power, kFrame, kWindow, 2UZ), "max hold");
        check(decay, expectedFigures<float, Statistic::logPower, true>(power, kFrame, kWindow, 2UZ), "exponential log power");
        check(dumped, expectedFigures<float, Statistic::variance, false>(power, kFrame, kWindow, kWindow), "integrate-and-dump variance");
    } | gr::testing::overGraphs(gr::testing::kAllDomains);
};

static_assert(!gr::filter::detail::Averageable<float, Statistic::unbiasedVariance, true>, "an exponential window has no Bessel correction");
static_assert(!gr::filter::detail::Averageable<std::complex<float>, Statistic::maximum, false>, "complex samples have no order");
static_assert(!gr::filter::detail::Averageable<float, Statistic::sum, true>, "an exponential window has no unambiguous sum");
static_assert(std::same_as<gr::filter::MovingAverage<std::complex<float>, Statistic::rms>::TOut, float>);
static_assert(std::same_as<gr::filter::MovingAverage<std::complex<float>, Statistic::sum>::TOut, std::complex<float>>);

} // namespace

int main() {
    "MovingAverage and ExponentialAverage"_test = definitions;
    "MovingAverage on every domain"_test        = domains;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
