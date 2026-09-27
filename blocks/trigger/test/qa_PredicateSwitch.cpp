#include <boost/ut.hpp>

#include <string>
#include <vector>

#include "Marble.hpp"
#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/trigger/PredicateSwitch.hpp>

using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] const gr::property_map& kValues() {
    static const gr::property_map map{{"a", 1.f}, {"b", 5.f}, {"c", -1.f}};
    return map;
}
/** L carries a level the predicates may read, H a higher one */
[[nodiscard]] const gr::property_map& kTags() {
    static const gr::property_map map{{"L", gr::property_map{{"level", 0.5f}}}, {"H", gr::property_map{{"level", 2.f}}}};
    return map;
}

struct Routed {
    std::string first;
    std::string second;
    gr::Size_t  unmatched;
};

[[nodiscard]] Routed route(std::string script, std::vector<std::string> predicates, std::size_t chunk = 0UZ) {
    gr::testing::GraphFixture fixture;
    auto&                     source = fixture.emplace<MarbleSource<float>>({{"script", std::move(script)}, {"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     split  = fixture.emplace<PredicateSwitch<float>>({{"predicates", std::move(predicates)}});
    auto&                     first  = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    auto&                     second = fixture.emplace<MarbleSink<float>>({{"sample_values", kValues()}, {"sample_tags", kTags()}});
    if (chunk > 0UZ) {
        split.in.max_samples = chunk;
    }
    boost::ut::expect(fixture.connect<"out", "in">(source, split).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#0")}, first, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.graph.connect(split, gr::PortDefinition{std::string("out#1")}, second, gr::PortDefinition{std::string("in")}).has_value());
    boost::ut::expect(fixture.run().has_value());
    return {first.script(), second.script(), split.n_unmatched.value};
}
} // namespace

const boost::ut::suite<"PredicateSwitch"> _predicateSwitch = [] {
    using namespace boost::ut;

    "each sample goes to the first predicate that holds for it"_test = [] {
        const auto found = route("a b c b a |", {"x > 4", "x > 0"});
        expect(eq(found.first, std::string("b b |")));
        expect(eq(found.second, std::string("a a |")));
        expect(eq(found.unmatched, 1U)) << "the negative sample matches neither";
    };

    "a predicate reads a field of the tags seen so far"_test = [] {
        const auto found = route("L:a a H:a a |", {"x > level", "x <= level"});
        expect(eq(found.first, std::string("L:a a |"))) << "the tag travels with the sample it marks";
        expect(eq(found.second, std::string("H:a a |")));
        expect(eq(found.unmatched, 0U));
    };

    "a field no tag has carried yet matches nothing"_test = [] {
        const auto found = route("a a L:a |", {"x > level", "x > 10"});
        expect(eq(found.first, std::string("L:a |")));
        expect(eq(found.second, std::string("|")));
        expect(eq(found.unmatched, 2U));
    };

    "a predicate that does not compile never matches, and the others still route"_test = [] {
        const auto found = route("a b |", {"x >", "x > 0"});
        expect(eq(found.first, std::string("|")));
        expect(eq(found.second, std::string("a b |")));
    };

    "the routing does not depend on where the stream is cut"_test = [] {
        const auto whole = route("L:a b c H:a b a |", {"x > level", "x > 0"});
        for (const std::size_t chunk : {1UZ, 2UZ, 3UZ}) {
            const auto cut = route("L:a b c H:a b a |", {"x > level", "x > 0"}, chunk);
            expect(eq(cut.first, whole.first)) << chunk;
            expect(eq(cut.second, whole.second)) << chunk;
            expect(eq(cut.unmatched, whole.unmatched)) << chunk;
        }
    };
};

int main() { /* not needed for UT */ }
