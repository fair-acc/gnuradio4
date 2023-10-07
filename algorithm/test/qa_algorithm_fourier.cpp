#include <array>
#include <numbers>
#include <vector>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>

template<gr::meta::array_or_vector_type T, gr::meta::array_or_vector_type U = T>
bool
equalVectors(const T &v1, const U &v2, double tolerance = std::is_same_v<typename T::value_type, double> ? 1.e-5 : 1e-4) {
    if constexpr (gr::algorithm::ComplexType<T>) {
        return std::ranges::equal(v1, v2, [&tolerance](const auto &l, const auto &r) {
            return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type::value_type>(tolerance);
        });
    } else {
        return std::ranges::equal(v1, v2, [&tolerance](const auto &l, const auto &r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

const boost::ut::suite<"window functions"> windowTests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using gr::algorithm::window::create;

    "window pre-computed array tests"_test = []<typename T>() { // this tests regression w.r.t. changed implementations
        // Expected value for size 8
        std::array RectangularRef{ 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        std::array HammingRef{ 0.07672f, 0.2119312255f, 0.53836f, 0.8647887745f, 1.0f, 0.8647887745f, 0.53836f, 0.2119312255f };
        std::array HannRef{ 0.f, 0.1882550991f, 0.611260467f, 0.950484434f, 0.950484434f, 0.611260467f, 0.1882550991f, 0.f };
        std::array BlackmanRef{ 0.f, 0.09045342435f, 0.4591829575f, 0.9203636181f, 0.9203636181f, 0.4591829575f, 0.09045342435f, 0.f };
        std::array BlackmanHarrisRef{ 0.00006f, 0.03339172348f, 0.3328335043f, 0.8893697722f, 0.8893697722f, 0.3328335043f, 0.03339172348f, 0.00006f };
        std::array BlackmanNuttallRef{ 0.0003628f, 0.03777576895f, 0.34272762f, 0.8918518611f, 0.8918518611f, 0.34272762f, 0.03777576895f, 0.0003628f };
        std::array ExponentialRef{ 1.f, 1.042546905f, 1.08690405f, 1.133148453f, 1.181360413f, 1.231623642f, 1.284025417f, 1.338656724f };
        std::array FlatTopRef{ 0.004f, -0.1696424054f, 0.04525319348f, 3.622389212f, 3.622389212f, 0.04525319348f, -0.1696424054f, 0.004f };
        std::array HannExpRef{ 0.f, 0.611260467f, 0.950484434f, 0.1882550991f, 0.1882550991f, 0.950484434f, 0.611260467f, 0.f };
        std::array NuttallRef{ 0.f, 0.0311427368f, 0.3264168059f, 0.8876284573f, 0.8876284573f, 0.3264168059f, 0.0311427368f, 0.f };
        std::array KaiserRef{ 0.5714348848f, 0.7650986027f, 0.9113132365f, 0.9899091685f, 0.9899091685f, 0.9113132365f, 0.7650986027f, 0.5714348848f };

        // check all windows for unwanted changes
        using enum gr::algorithm::window::Type;
        expect(equalVectors(create<T>(None, 8), RectangularRef)) << fmt::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(None, 8), RectangularRef);
        expect(equalVectors(create<T>(Rectangular, 8), RectangularRef)) << fmt::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(Rectangular, 8), RectangularRef);
        expect(equalVectors(create<T>(Hamming, 8), HammingRef)) << fmt::format("<{}> equal Hamming vector {} vs. ref: {}", type_name<T>(), create<T>(Hamming, 8), HammingRef);
        expect(equalVectors(create<T>(Hann, 8), HannRef)) << fmt::format("<{}> equal Hann vector {} vs. ref: {}", type_name<T>(), create<T>(Hann, 8), HannRef);
        expect(equalVectors(create<T>(Blackman, 8), BlackmanRef)) << fmt::format("<{}> equal Blackman vvector {} vs. ref: {}", type_name<T>(), create<T>(Blackman, 8), BlackmanRef);
        expect(equalVectors(create<T>(BlackmanHarris, 8), BlackmanHarrisRef))
                << fmt::format("<{}> equal BlackmanHarris vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanHarris, 8), BlackmanHarrisRef);
        expect(equalVectors(create<T>(BlackmanNuttall, 8), BlackmanNuttallRef))
                << fmt::format("<{}> equal BlackmanNuttall vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanNuttall, 8), BlackmanNuttallRef);
        expect(equalVectors(create<T>(Exponential, 8), ExponentialRef)) << fmt::format("<{}> equal Exponential vector {} vs. ref: {}", type_name<T>(), create<T>(Exponential, 8), ExponentialRef);
        expect(equalVectors(create<T>(FlatTop, 8), FlatTopRef)) << fmt::format("<{}> equal FlatTop vector {} vs. ref: {}", type_name<T>(), create<T>(FlatTop, 8), FlatTopRef);
        expect(equalVectors(create<T>(HannExp, 8), HannExpRef)) << fmt::format("<{}> equal HannExp vector {} vs. ref: {}", type_name<T>(), create<T>(HannExp, 8), HannExpRef);
        expect(equalVectors(create<T>(Nuttall, 8), NuttallRef)) << fmt::format("<{}> equal Nuttall vector {} vs. ref: {}", type_name<T>(), create<T>(Nuttall, 8), NuttallRef);
        expect(equalVectors(create<T>(Kaiser, 8), KaiserRef)) << fmt::format("<{}> equal Kaiser vector {} vs. ref: {}", type_name<T>(), create<T>(Kaiser, 8), KaiserRef);

        // test zero length
        expect(eq(create<T>(None, 0).size(), 0u)) << fmt::format("<{}> zero size None vectors", type_name<T>());
        expect(eq(create<T>(Rectangular, 0).size(), 0u)) << fmt::format("<{}> zero size Rectangular vectors", type_name<T>());
        expect(eq(create<T>(Hamming, 0).size(), 0u)) << fmt::format("<{}> zero size Hamming vectors", type_name<T>());
        expect(eq(create<T>(Hann, 0).size(), 0u)) << fmt::format("<{}> zero size Hann vectors", type_name<T>());
        expect(eq(create<T>(Blackman, 0).size(), 0u)) << fmt::format("<{}> zero size Blackman vectors", type_name<T>());
        expect(eq(create<T>(BlackmanHarris, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanHarris vectors", type_name<T>());
        expect(eq(create<T>(BlackmanNuttall, 0).size(), 0u)) << fmt::format("<{}> zero size BlackmanNuttall vectors", type_name<T>());
        expect(eq(create<T>(Exponential, 0).size(), 0u)) << fmt::format("<{}> zero size Exponential vectors", type_name<T>());
        expect(eq(create<T>(FlatTop, 0).size(), 0u)) << fmt::format("<{}> zero size FlatTop vectors", type_name<T>());
        expect(eq(create<T>(HannExp, 0).size(), 0u)) << fmt::format("<{}> zero size HannExp vectors", type_name<T>());
        expect(eq(create<T>(Nuttall, 0).size(), 0u)) << fmt::format("<{}> zero size Nuttall vectors", type_name<T>());
        expect(eq(create<T>(Kaiser, 0).size(), 0u)) << fmt::format("<{}> zero size Kaiser vectors", type_name<T>());
    } | std::tuple<float, double>();

    "basic window tests"_test = [](gr::algorithm::window::Type window) {
        using enum gr::algorithm::window::Type;
        expect(gr::algorithm::window::parse(gr::algorithm::window::to_string(window)) == window) << fmt::format("window {} parse(to_string) identity\n", gr::algorithm::window::to_string(window));

        const auto w = create(window, 1024U);
        expect(eq(w.size(), 1024U));

        if (window == Exponential || window == FlatTop || window == Blackman || window == Nuttall) {
            return; // min max out of [0, 1] by design and/or numerical corner cases
        }
        const auto [min, max] = std::ranges::minmax_element(w);
        expect(ge(*min, 0.f)) << fmt::format("window {} min value\n", gr::algorithm::window::to_string(window));
        expect(le(*max, 1.f)) << fmt::format("window {} max value\n", gr::algorithm::window::to_string(window));
    } | gr::algorithm::window::TypeList;

    "window corner cases"_test = []<typename T>() {
        expect(throws<std::invalid_argument>([] { std::ignore = gr::algorithm::window::parse("UnknownWindow"); })) << "invalid window name";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 1); })) << "invalid Kaiser window size";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 2, -1.f); })) << "invalid Kaiser window beta";
    } | std::tuple<float, double>();
};

int
main() { /* not needed for UT */
}
