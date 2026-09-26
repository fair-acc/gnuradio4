#include <boost/ut.hpp>

#include <gnuradio-4.0/test/GraphFixture.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite<"GraphFixture"> _fixture = [] {
    using namespace boost::ut;

    "a block stays alive for the assertions made on it"_test = [] {
        gr::testing::GraphFixture fixture;
        auto&                     source = fixture.emplace<gr::testing::ConstantSource<float>>({{"n_samples_max", gr::Size_t(32)}});
        auto&                     sink   = fixture.emplace<gr::testing::CountingSink<float>>();
        expect(fixture.connect<"out", "in">(source, sink).has_value());
        expect(fixture.run().has_value());
        expect(eq(sink.count, 32UZ)) << "a scheduler scoped to a helper would have freed the sink by now";
    };
};

int main() { /* tests are statically executed */ }
