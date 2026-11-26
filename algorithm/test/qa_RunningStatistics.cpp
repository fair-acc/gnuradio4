#include <boost/ut.hpp>

#include <cmath>
#include <initializer_list>

#include <gnuradio-4.0/algorithm/RunningStatistics.hpp>

namespace {
using gr::algorithm::RunningStatistics;

[[nodiscard]] constexpr RunningStatistics statisticsOf(std::initializer_list<double> values) {
    RunningStatistics statistics;
    for (const double value : values) {
        statistics.add(value);
    }
    return statistics;
}

static_assert(statisticsOf({1., 3.}).mean == 2. && statisticsOf({1., 3.}).m2 == 2., "a kernel or a constant expression accumulates as the host does");

const boost::ut::suite<"RunningStatistics"> tests = [] {
    using namespace boost::ut;

    "values far from zero keep their mean and deviations exactly"_test = [] {
        const RunningStatistics statistics = statisticsOf({1e9 + 1., 1e9 + 2., 1e9 + 3., 1e9 + 4.});
        expect(eq(statistics.count, 4U));
        expect(eq(statistics.mean, 1e9 + 2.5));
        expect(eq(statistics.m2, 5.)) << "1.5^2 + 0.5^2 + 0.5^2 + 1.5^2; sum(x*x) at 4e18 resolves only 512";
        expect(eq(statistics.populationStdDev(), std::sqrt(1.25)));
        expect(eq(statistics.sampleStdDev(), std::sqrt(5. / 3.)));
    };

    "the extremes are the smallest and largest values in any order, negatives included"_test = [] {
        const RunningStatistics statistics = statisticsOf({3., -2., 5., 0.});
        expect(eq(statistics.smallest, -2.));
        expect(eq(statistics.largest, 5.));
        expect(eq(statistics.mean, 1.5));
    };

    "without a value every figure is 0, and a single value spreads by none"_test = [] {
        const RunningStatistics none;
        expect(eq(none.count, 0U) && eq(none.mean, 0.) && eq(none.populationStdDev(), 0.) && eq(none.sampleStdDev(), 0.));

        const RunningStatistics single = statisticsOf({7.});
        expect(eq(single.mean, 7.) && eq(single.smallest, 7.) && eq(single.largest, 7.));
        expect(eq(single.populationStdDev(), 0.));
        expect(eq(single.sampleStdDev(), 0.)) << "the divisor n - 1 is 0: no sample deviation";
    };
};
} // namespace

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
