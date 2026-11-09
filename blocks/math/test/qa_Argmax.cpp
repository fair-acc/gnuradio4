#include <boost/ut.hpp>

#include <gnuradio-4.0/math/Argmax.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>

namespace {
struct Position {
    std::vector<gr::Size_t> index;
    std::vector<gr::Size_t> stream;
    float                   rate = 0.f;
};

template<typename T>
Position argmaxOf(const std::vector<std::vector<T>>& inputs, gr::property_map settings, decltype(gr::testing::ProbeSource<T>::_tags) tagsOnFirstInput = {}) {
    settings.insert_or_assign("n_inputs", static_cast<gr::Size_t>(inputs.size()));
    gr::testing::ProbedPorts<gr::blocks::math::Argmax<T>, T, gr::Size_t> probed(inputs, std::move(settings), gr::testing::indexedPorts("in", inputs.size()), {"index", "stream"});
    probed.sources[0]->_tags = std::move(tagsOnFirstInput);
    const auto outputs       = probed.run();
    return {outputs[0], outputs[1], probed.sinks[0]->sample_rate};
}
} // namespace

const auto argmaxTests = [] {
    using namespace boost::ut;
    using Indices = std::vector<gr::Size_t>;

    "a default Argmax has one input"_test = [] {
        const gr::blocks::math::Argmax<float> argmax;
        expect(eq(argmax.in.size(), 1UZ) and eq(argmax.n_inputs.value, gr::Size_t{1U}));
    };

    "one input of one vector reports where its maximum is"_test = []<typename T>(const T&) {
        const Position found = argmaxOf<T>({{4, 7, 17}}, {{"length", gr::Size_t{3U}}});
        expect(found.index == Indices{2} and found.stream == Indices{0});
    } | std::tuple<std::uint8_t, std::int32_t, float, double>{};

    "the maximum over two inputs names the input that holds it"_test = [] {
        const Position found = argmaxOf<float>({{4, 7, 17}, {18, 18, 18}}, {{"length", gr::Size_t{3U}}});
        expect(found.index == Indices{0} and found.stream == Indices{1});
    };

    "a tie goes to the smaller position, then to the lower input"_test = [] {
        const Position found = argmaxOf<std::int16_t>({{1, 9, 9}, {9, 9, 1}}, {{"length", gr::Size_t{3U}}});
        expect(found.index == Indices{0} and found.stream == Indices{1});
        const Position samePosition = argmaxOf<std::int16_t>({{1, 9}, {1, 9}}, {{"length", gr::Size_t{2U}}});
        expect(samePosition.index == Indices{1} and samePosition.stream == Indices{0});
    };

    "every vector yields one result and an incomplete vector none"_test = [] {
        const Position found = argmaxOf<std::uint8_t>({{5, 1, 0, 2, 3, 0, 9}}, {{"length", gr::Size_t{3U}}});
        expect(found.index == Indices{0, 1} and found.stream == Indices{0, 0});
    };

    "the output rate is the input rate divided by length"_test = [] { expect(eq(argmaxOf<float>({{1, 2, 3, 4, 5, 6}}, {{"length", gr::Size_t{3U}}}).rate, gr::testing::kProbeRate / 3.f)); };

    "NaN ranks below every number, and a vector of NaN reports its first position"_test = [] {
        constexpr float kNaN  = std::numeric_limits<float>::quiet_NaN();
        const Position  found = argmaxOf<float>({{kNaN, 1.f, kNaN, kNaN, kNaN, kNaN}}, {{"length", gr::Size_t{3U}}});
        expect(found.index == Indices{1, 0});
    };

    "a length set by a tag takes effect from the tagged sample on"_test = [] {
        const Position found = argmaxOf<float>({{5.f, 3.f, 1.f, 9.f, 8.f, 2.f}}, {}, {{2UZ, {{"length", gr::Size_t{2U}}}}});
        expect(found.index == Indices{0, 0, 1, 0});
    };

    "a position beyond the range of the sample type is reported in full"_test = [] {
        std::vector<std::uint8_t> ramp(300UZ, std::uint8_t{0});
        ramp[299UZ] = 1;
        expect(argmaxOf<std::uint8_t>({ramp}, {{"length", gr::Size_t{300U}}}).index == Indices{299});
    };
};

int main() {
    using namespace boost::ut;
    "Argmax"_test = argmaxTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
