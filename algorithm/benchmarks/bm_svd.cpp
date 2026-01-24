#include <benchmark.hpp>

#include "gnuradio-4.0/meta/UnitTestHelper.hpp"

#include <format>
#include <random>

#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tensor.hpp>

template<typename T>
void randomizeMatrix(gr::Tensor<T>& matrix, unsigned seed = 42) {
    std::mt19937 gen(seed);
    if constexpr (gr::meta::complex_like<T>) {
        using RealT = typename T::value_type;
        std::uniform_real_distribution<RealT> dis(RealT{-1}, RealT{1});
        for (auto& x : matrix) {
            x = T{dis(gen), dis(gen)};
        }
    } else {
        std::uniform_real_distribution<T> dis(T{-1}, T{1});
        for (auto& x : matrix) {
            x = dis(gen);
        }
    }
}

template<typename T>
struct SVDPrecision {
    using type = T;
};

template<gr::meta::complex_like T>
struct SVDPrecision<T> {
    using type = typename T::value_type;
};

template<typename T, bool FullSVD, std::size_t NRepetitions = 100UZ>
void testSVD(std::size_t m, std::size_t n) {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using namespace gr;
    using namespace gr::math;

    using RealT = typename SVDPrecision<T>::type;

    constexpr std::size_t nRepetitions = NRepetitions;

    // SVD complexity: O(min(m,n) * max(m,n)^2)
    const std::size_t minDim  = std::min(m, n);
    const std::size_t maxDim  = std::max(m, n);
    const std::size_t scaling = minDim * maxDim * maxDim;

    const std::string modeName = FullSVD ? "full" : "singular values only";
    const std::string bmName   = std::format("SVD {:6} - {:22} {}x{}", modeName, type_name<T>(), m, n);

    try {
        Tensor<T> A({m, n});
        randomizeMatrix(A);

        if constexpr (FullSVD) {
            Tensor<T>     U, V;
            Tensor<RealT> S;

            // warm-up
            auto status = gesvd(U, S, V, A);
            expect(status == svd::Status::Success || status == svd::Status::EarlyReturn) << bmName << fatal;

            ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&] { std::ignore = gesvd(U, S, V, A); };

            // sanity check: singular values should be non-negative
            expect(S.size() > 0UZ) << bmName << fatal;
            expect(ge(S[0], RealT{0})) << bmName << fatal;
        } else {
            Tensor<RealT> S;

            // warm-up
            auto status = gesvd(S, A);
            expect(status == svd::Status::Success || status == svd::Status::EarlyReturn) << bmName << fatal;

            ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&] { std::ignore = gesvd(S, A); };

            // sanity check
            expect(S.size() > 0UZ) << bmName << fatal;
            expect(ge(S[0], RealT{0})) << bmName << fatal;
        }
    } catch (const std::exception& e) {
        std::println(stderr, "{} : {}", bmName, e.what());
    } catch (...) {
        std::println(stderr, "{} : unknown exception", bmName);
    }
}

void benchmarkFullSVD() {
    using boost::ut::operator""_test;

    "full SVD: float"_test = [] {
        testSVD<float, true, 5000UZ>(32, 32);
        testSVD<float, true, 200UZ>(128, 128);
        testSVD<float, true, 3UZ>(512, 512);
        testSVD<float, true, 1UZ>(1024, 1024);
        testSVD<float, true, 30UZ>(1024, 64);
        testSVD<float, true, 30UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };

    "full SVD: double"_test = [] {
        testSVD<double, true, 5000UZ>(32, 32);
        testSVD<double, true, 200UZ>(128, 128);
        testSVD<double, true, 2UZ>(512, 512);
        testSVD<double, true, 1UZ>(1024, 1024);
        testSVD<double, true, 30UZ>(1024, 64);
        testSVD<double, true, 30UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };

    "full SVD: std::complex<float>"_test = [] {
        testSVD<std::complex<float>, true, 5000UZ>(32, 32);
        testSVD<std::complex<float>, true, 100UZ>(128, 128);
        testSVD<std::complex<float>, true, 1UZ>(512, 512);
        testSVD<std::complex<float>, true, 1UZ>(1024, 1024);
        testSVD<std::complex<float>, true, 15UZ>(1024, 64);
        testSVD<std::complex<float>, true, 15UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };
}

void benchmarkSingularValuesOnly() {
    using boost::ut::operator""_test;

    "singular values only: float"_test = [] {
        testSVD<float, false, 5000UZ>(32, 32);
        testSVD<float, false, 200UZ>(128, 128);
        testSVD<float, false, 3UZ>(512, 512);
        testSVD<float, false, 1UZ>(1024, 1024);
        testSVD<float, false, 30UZ>(1024, 64);
        testSVD<float, false, 30UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };

    "singular values only: double"_test = [] {
        testSVD<double, false, 5000UZ>(32, 32);
        testSVD<double, false, 200UZ>(128, 128);
        testSVD<double, false, 2UZ>(512, 512);
        testSVD<double, false, 1UZ>(1024, 1024);
        testSVD<double, false, 30UZ>(1024, 64);
        testSVD<double, false, 30UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };

    "singular values only: std::complex<float>"_test = [] {
        testSVD<std::complex<float>, false, 5000UZ>(32, 32);
        testSVD<std::complex<float>, false, 100UZ>(128, 128);
        testSVD<std::complex<float>, false, 1UZ>(512, 512);
        testSVD<std::complex<float>, false, 1UZ>(1024, 1024);
        testSVD<std::complex<float>, false, 15UZ>(1024, 64);
        testSVD<std::complex<float>, false, 15UZ>(64, 1024);
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"SVD benchmark"> _svd_bm_tests = [] {
    benchmarkFullSVD();
    benchmarkSingularValuesOnly();

    std::println("N.B. ops/s values are scaled with min(m,n)*max(m,n)^2 (SVD complexity).");
};

int main() { /* not needed by the UT framework */ }
