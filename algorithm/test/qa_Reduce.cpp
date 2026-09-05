#include <boost/ut.hpp>

#include <numeric>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/Reduce.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

const boost::ut::suite<"Reduce"> _reduce = [] {
    using namespace boost::ut;
    using gr::algorithm::Reduce;

    "arg-max finds the one largest element"_test = [] {
        for (const std::size_t count : {1UZ, 2UZ, 255UZ, 256UZ, 257UZ, 4096UZ}) {
            std::vector<float> values(count);
            std::iota(values.begin(), values.end(), 1.f);
            const std::size_t planted = count / 3UZ;
            values[planted]           = 1e6f;

            const auto best = Reduce::argMaxHost<float>(values);
            expect(eq(best.index, planted)) << std::format("count {}: found index {}", count, best.index);
            expect(eq(best.value, 1e6f));
        }
    };

    "arg-max on an empty span reports nothing rather than reading it"_test = [] {
        const auto best = Reduce::argMaxHost<float>({});
        expect(eq(best.index, 0UZ));
        expect(lt(best.value, 0.f)) << "the initial value has to lose to every real sample";
    };

    "the sum is the sum"_test = [] {
        std::vector<double> values(1000);
        std::iota(values.begin(), values.end(), 1.0);
        expect(eq(Reduce::sumHost<double>(values), 500500.0)) << "1..1000 sums to n(n+1)/2";
    };

    "ties go to the first, so the answer does not depend on the split"_test = [] {
        std::vector<float> values(1024, 0.f);
        values[100]     = 5.f;
        values[900]     = 5.f;
        const auto best = Reduce::argMaxHost<float>(values);
        expect(eq(best.index, 100UZ)) << "a later equal element must not displace an earlier one";
    };
};

int main() { /* tests run from the suite */ }
