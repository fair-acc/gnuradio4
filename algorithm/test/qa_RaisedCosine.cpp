#include <boost/ut.hpp>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/RaisedCosine.hpp>

const boost::ut::suite<"RaisedCosine"> raisedCosineTests = [] {
    using namespace boost::ut;
    using gr::algorithm::filter::raisedCosine;
    using gr::algorithm::filter::rootRaisedCosine;
    using gr::algorithm::filter::rootRaisedCosineTaps;

    "removable singularities"_test = [] {
        constexpr double kNeighbourDistance = 1e-6;
        for (const double excessBandwidth : {0.25, 0.5, 1.}) {
            for (const double singularTime : {0., 1. / (4. * excessBandwidth), -1. / (4. * excessBandwidth)}) {
                const double neighbourMean = (rootRaisedCosine(singularTime - kNeighbourDistance, excessBandwidth) + rootRaisedCosine(singularTime + kNeighbourDistance, excessBandwidth)) / 2.;
                expect(approx(rootRaisedCosine(singularTime, excessBandwidth), neighbourMean, 1e-8)) << "excess_bandwidth=" << excessBandwidth << " t=" << singularTime;
            }
        }
    };

    "raised cosine singularities"_test = [] {
        constexpr double kNeighbourDistance = 1e-6;
        for (const double excessBandwidth : {0.25, 0.5, 1.}) {
            for (const double singularTime : {0., 1. / (2. * excessBandwidth), -1. / (2. * excessBandwidth)}) {
                const double neighbourMean = (raisedCosine(singularTime - kNeighbourDistance, excessBandwidth) + raisedCosine(singularTime + kNeighbourDistance, excessBandwidth)) / 2.;
                expect(approx(raisedCosine(singularTime, excessBandwidth), neighbourMean, 1e-8)) << "excess_bandwidth=" << excessBandwidth << " t=" << singularTime;
            }
        }
    };

    "raised cosine is the matched root-raised-cosine pair"_test = [] {
        constexpr std::size_t     kSamplesPerSymbol = 8UZ;
        const std::vector<double> taps              = rootRaisedCosineTaps<double>(kSamplesPerSymbol, 0.35, 20UZ);
        for (std::size_t lag = 0UZ; lag < taps.size(); ++lag) {
            double matchedPair = 0.;
            for (std::size_t k = lag; k < taps.size(); ++k) {
                matchedPair += taps[k] * taps[k - lag];
            }
            expect(approx(matchedPair, raisedCosine(static_cast<double>(lag) / static_cast<double>(kSamplesPerSymbol), 0.35), 0.01)) << "lag=" << lag;
        }
    };

    "unit energy and symmetry"_test = [] {
        const std::vector<double> taps = rootRaisedCosineTaps<double>(4UZ, 0.35, 10UZ);
        expect(eq(taps.size(), 41UZ));
        expect(approx(std::inner_product(taps.begin(), taps.end(), taps.begin(), 0.), 1., 1e-12)) << "unit energy";
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            expect(approx(taps[i], taps[taps.size() - 1UZ - i], 1e-15)) << "i=" << i;
        }
    };

    "matched pair"_test = [] {
        for (const std::size_t samplesPerSymbol : {4UZ, 8UZ}) {
            constexpr std::size_t     kSpan       = 10UZ;
            const std::vector<double> taps        = rootRaisedCosineTaps<double>(samplesPerSymbol, 0.35, kSpan);
            const auto                matchedPair = [&taps](std::size_t lag) {
                double sum = 0.;
                for (std::size_t k = lag; k < taps.size(); ++k) {
                    sum += taps[k] * taps[k - lag];
                }
                return sum;
            };
            expect(approx(matchedPair(0UZ), 1., 1e-12)) << "unit peak";
            for (std::size_t symbol = 1UZ; symbol <= kSpan; ++symbol) {
                expect(lt(std::abs(matchedPair(symbol * samplesPerSymbol)), 0.01)) << "samples_per_symbol=" << samplesPerSymbol << " symbol=" << symbol;
            }
        }
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
