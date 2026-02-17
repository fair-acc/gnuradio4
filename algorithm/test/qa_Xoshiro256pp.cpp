#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <vector>

#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;

const boost::ut::suite xoshiro256ppTests = [] {
    using gr::rng::Xoshiro256pp;

    "URBG concept satisfied"_test = [] {
        static_assert(std::uniform_random_bit_generator<Xoshiro256pp>);
        expect(eq(Xoshiro256pp::min(), std::uint64_t{0}));
        expect(eq(Xoshiro256pp::max(), std::numeric_limits<std::uint64_t>::max()));
    };

    "deterministic output for same seed"_test = [] {
        Xoshiro256pp rng1(42);
        Xoshiro256pp rng2(42);
        for (int i = 0; i < 1000; ++i) {
            expect(eq(rng1(), rng2())) << std::format("mismatch at draw {}", i);
        }
    };

    "different seeds produce different sequences"_test = [] {
        Xoshiro256pp rng1(0);
        Xoshiro256pp rng2(1);
        bool         allSame = true;
        for (int i = 0; i < 100; ++i) {
            if (rng1() != rng2()) {
                allSame = false;
                break;
            }
        }
        expect(!allSame) << "different seeds produced identical output";
    };

    "re-seeding resets state"_test = [] {
        Xoshiro256pp               rng(123);
        std::vector<std::uint64_t> first(10);
        for (auto& v : first) {
            v = rng();
        }
        rng.seed(123);
        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(rng(), first[i])) << std::format("mismatch after re-seed at {}", i);
        }
    };

    "known-answer test seed=0"_test = [] {
        Xoshiro256pp rng(0);
        // first 5 draws from seed=0, locked for regression
        constexpr std::array<std::uint64_t, 5> expected{
            0x53175d61490b23dfULL,
            0x61da6f3dc380d507ULL,
            0x5c0fdf91ec9a7bfcULL,
            0x02eebf8c3bbe5e1aULL,
            0x7eca04ebaf4a5eeaULL,
        };
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto val = rng();
            expect(eq(val, expected[i])) << std::format("draw {} mismatch: got {:#018x}, expected {:#018x}", i, val, expected[i]);
        }
    };

    "uniform01<double> range [0, 1)"_test = [] {
        Xoshiro256pp  rng(7);
        constexpr int N      = 1'000'000;
        double        minVal = 1.0;
        double        maxVal = 0.0;
        double        sum    = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = rng.uniform01<double>();
            minVal         = std::min(minVal, v);
            maxVal         = std::max(maxVal, v);
            sum += v;
        }
        expect(ge(minVal, 0.0)) << "uniform01 produced negative value";
        expect(lt(maxVal, 1.0)) << "uniform01 produced value >= 1.0";
        const double mean = sum / N;
        expect(approx(mean, 0.5, 0.005)) << std::format("mean {:.6f} not near 0.5", mean);
    };

    "uniform01<float> range [0, 1)"_test = [] {
        Xoshiro256pp  rng(7);
        constexpr int N      = 1'000'000;
        float         minVal = 1.0f;
        float         maxVal = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float v = rng.uniform01<float>();
            minVal        = std::min(minVal, v);
            maxVal        = std::max(maxVal, v);
        }
        expect(ge(minVal, 0.0f)) << "uniform01<float> produced negative value";
        expect(lt(maxVal, 1.0f)) << "uniform01<float> produced value >= 1.0";
    };

    "uniformM11<double> range [-1, +1)"_test = [] {
        Xoshiro256pp  rng(13);
        constexpr int N      = 1'000'000;
        double        minVal = 0.0;
        double        maxVal = 0.0;
        double        sum    = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = rng.uniformM11<double>();
            minVal         = std::min(minVal, v);
            maxVal         = std::max(maxVal, v);
            sum += v;
        }
        expect(ge(minVal, -1.0)) << "uniformM11 produced value < -1";
        expect(lt(maxVal, 1.0)) << "uniformM11 produced value >= 1.0";
        const double mean = sum / N;
        expect(approx(mean, 0.0, 0.005)) << std::format("mean {:.6f} not near 0", mean);
    };

    "triangularM11<double> range and mean"_test = [] {
        Xoshiro256pp  rng(17);
        constexpr int N      = 1'000'000;
        double        minVal = 0.0;
        double        maxVal = 0.0;
        double        sum    = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = rng.triangularM11<double>();
            minVal         = std::min(minVal, v);
            maxVal         = std::max(maxVal, v);
            sum += v;
        }
        expect(ge(minVal, -1.0)) << "triangularM11 below -1";
        expect(lt(maxVal, 1.0)) << "triangularM11 above +1";
        const double mean = sum / N;
        expect(approx(mean, 0.0, 0.005)) << std::format("mean {:.6f} not near 0", mean);
    };

    "usable with std::uniform_int_distribution"_test = [] {
        Xoshiro256pp                       rng(99);
        std::uniform_int_distribution<int> dist(0, 9);
        std::array<int, 10>                counts{};
        constexpr int                      N = 100'000;
        for (int i = 0; i < N; ++i) {
            ++counts[static_cast<std::size_t>(dist(rng))];
        }
        for (int bucket = 0; bucket < 10; ++bucket) {
            const double fraction = static_cast<double>(counts[static_cast<std::size_t>(bucket)]) / N;
            expect(approx(fraction, 0.1, 0.02)) << std::format("bucket {} fraction {:.4f} too far from 0.1", bucket, fraction);
        }
    };
};

int main() { /* not needed for UT */ }
