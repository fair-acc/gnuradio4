#include <algorithm>
#include <execution>
#include <format>
#include <numbers>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeindex>

#include <benchmark.hpp>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TensorMath.hpp>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

using namespace std::string_view_literals;

template<gr::TensorLike Tensor, typename T = typename Tensor::value_type>
void randomise(Tensor& tensor, T min = T{-1}, T max = T{1}) {
    std::random_device rd;
    std::mt19937       gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dis(min, max);
        for (auto& x : tensor) {
            x = static_cast<T>(dis(gen));
        }
    } else {
        std::uniform_int_distribution<T> dis(min, max);
        for (auto& x : tensor) {
            x = dis(gen);
        }
    }
}

template<gr::TensorLike A, gr::TensorLike B>
auto max_abs_diff_2d(const A& a, const B& b) -> typename A::value_type {
    using T      = typename A::value_type;
    const auto m = a.extents()[0];
    const auto n = a.extents()[1];

    T max_diff{};
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const auto d = std::abs(a[i, j] - b[i, j]);
            if (d > max_diff) {
                max_diff = d;
            }
        }
    }
    return max_diff;
}

template<gr::TensorLike A, gr::TensorLike B>
auto max_abs_diff_1d(const A& a, const B& b) -> typename A::value_type {
    using T      = typename A::value_type;
    const auto n = a.extents()[0];

    T max_diff{};
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = std::abs(a[i] - b[i]);
        if (d > max_diff) {
            max_diff = d;
        }
    }
    return max_diff;
}
namespace test::reference {
// naive O(M*K*N) GEMM: C = A * B
template<typename T, gr::TensorOf<T> C, gr::TensorOf<T> A, gr::TensorOf<T> B>
constexpr void gemm(C& C_out, const A& A_in, const B& B_in) {
    const std::size_t m = A_in.extents()[0UZ];
    const std::size_t k = A_in.extents()[1UZ];
    const std::size_t n = B_in.extents()[1UZ];

#ifndef NDEBUG
    const std::size_t k_b = B_in.extents()[0UZ];
    if (k != k_b || C_out.extents()[0UZ] != m || C_out.extents()[1UZ] != n) {
        throw std::runtime_error("reference::gemm: incompatible extents");
    }
#endif

    std::ranges::fill(C_out, T{}); // required -> see accumulate below

    for (std::size_t i = 0UZ; i < m; ++i) {
        T* crow = &C_out[i, 0UZ]; // row i of C

        for (std::size_t p = 0UZ; p < k; ++p) {
            const T  a_ip  = A_in[i, p];    // scalar A[i,p]
            const T* b_row = &B_in[p, 0UZ]; // row p of B, contiguous

            // crow[j] += a_ip * brow[j]  for j in [0, n)
#if defined(__GLIBCXX__)
            std::transform(std::execution::unseq, b_row, b_row + n, crow, crow, [a_ip](const T& b, const T& c) noexcept { return c + a_ip * b; });
#else
            std::transform(b_row, b_row + n, crow, crow, [a_ip](const T& b, const T& c) noexcept { return c + a_ip * b; });
#endif
        }
    }
}
} // namespace test::reference

template<std::size_t m, std::size_t k, std::size_t n>
struct TestParameter {
    constexpr static std::size_t M = m;
    constexpr static std::size_t K = k;
    constexpr static std::size_t N = n;
};

constexpr auto testCases = std::tuple{TestParameter<6UZ, 6UZ, 6UZ>{}, TestParameter<64UZ, 64UZ, 64UZ>{}, TestParameter<256UZ, 256UZ, 256UZ>{}, TestParameter<1024UZ, 1024UZ, 1024UZ>{}};

template<bool runReference, typename T, typename Parameter>
void testGEMM() {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using gr::Tensor;

    const std::string typeName = gr::meta::type_name<T>();

    constexpr std::size_t nRepetitions{10UZ};

    Tensor<T> A({Parameter::M, Parameter::K});
    Tensor<T> B({Parameter::K, Parameter::N});
    Tensor<T> C_prod({Parameter::M, Parameter::N});
    Tensor<T> C_ref({Parameter::M, Parameter::N});

    randomise(A, T{-1}, T{1});
    randomise(B, T{-1}, T{1});

    gr::math::gemm(C_ref, A, B);
    const std::size_t scaling = 2UZ * Parameter::M * Parameter::K * Parameter::N;

    if constexpr (runReference) {
        const std::string bmName = std::format("GEMM(naive)  {:12} M={},K={},N={}", typeName, Parameter::M, Parameter::K, Parameter::N);

        ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&]() { test::reference::gemm<T>(C_prod, A, B); };
        expect(le(max_abs_diff_2d(C_prod, C_ref), static_cast<T>(1e-4f))) << bmName << fatal;
    } else {
        const std::string bmName = std::format("GEMM         {:12} M={},K={},N={}", typeName, Parameter::M, Parameter::K, Parameter::N);

        ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&]() { gr::math::gemm(C_prod, A, B); };
        expect(le(max_abs_diff_2d(C_prod, C_ref), static_cast<T>(1e-4f))) << bmName << fatal;
    }
}

namespace test::reference {
// naive O(M*K) GEMV: y = A * x
template<typename T, gr::TensorOf<T> Y, gr::TensorOf<T> A, gr::TensorOf<T> X>
void gemv(Y& y, const A& a, const X& x) {
    const std::size_t m = a.extents()[0UZ];
    const std::size_t k = a.extents()[1UZ];

#ifndef NDEBUG
    if (x.extents()[0UZ] != k || y.extents()[0UZ] != m) {
        throw std::runtime_error("reference::gemv: incompatible extents");
    }
#endif
    for (std::size_t i = 0UZ; i < m; ++i) {
        const T* rowBegin = &a[i, 0UZ];
        const T* rowEnd   = &a[i, k];

        // dot product of A[i,*] and x[*]
#if defined(__GLIBCXX__)
        y[i] = std::transform_reduce(std::execution::unseq, rowBegin, rowEnd, x.cbegin(), T{}, std::plus<>{}, std::multiplies<>{});
#else
        y[i] = std::transform_reduce(rowBegin, rowEnd, x.cbegin(), T{}, std::plus<>{}, std::multiplies<>{});
#endif
    }
}
} // namespace test::reference

template<bool runReference, typename T, typename Parameter>
void testGEMV() {
    using namespace benchmark;
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using gr::Tensor;

    const std::string typeName = gr::meta::type_name<T>();

    constexpr int nRepetitions{10000UZ};

    Tensor<T> A({Parameter::M, Parameter::K});
    Tensor<T> x({Parameter::K});
    Tensor<T> y_prod({Parameter::M});
    Tensor<T> y_ref({Parameter::M});

    randomise(A, T{-1}, T{1});
    randomise(x, T{-1}, T{1});

    gr::math::gemv(y_ref, A, x);
    const std::size_t scaling = 2UZ * Parameter::M * Parameter::K;

    if constexpr (runReference) {
        const std::string bmName = std::format("GEMV(naive)  {:12} M={},K={}", typeName, Parameter::M, Parameter::K);

        ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&]() { test::reference::gemv<T>(y_prod, A, x); };
        expect(le(max_abs_diff_1d(y_prod, y_ref), static_cast<T>(1e-4f))) << bmName << fatal;
    } else {
        const std::string bmName = std::format("GEMV         {:12} M={},K={}", typeName, Parameter::M, Parameter::K);

        ::benchmark::benchmark<nRepetitions>(std::string_view(bmName), scaling) = [&]() { gr::math::gemv(y_prod, A, x); };
        expect(le(max_abs_diff_1d(y_prod, y_ref), static_cast<T>(1e-4f))) << bmName << fatal;
    }
}

inline const boost::ut::suite<"Tensor GEMM/GEMV benchmarks"> _tensor_bm_tests = [] {
    using namespace boost::ut;

    "gemm"_test = []<typename T>(T) {
        std::apply([]<typename... Params>(Params&&...) { (testGEMM<true, T, std::remove_cvref_t<Params>>(), ...); }, testCases);
        ::benchmark::results::add_separator();
        std::apply([]<typename... Params>(Params&&...) { (testGEMM<false, T, std::remove_cvref_t<Params>>(), ...); }, testCases);
        ::benchmark::results::add_separator();
    } | std::tuple{float{}, double{}};

    "bench gemv"_test = []<typename T>(T) {
        std::apply([]<typename... Params>(Params&&...) { (testGEMV<true, T, std::remove_cvref_t<Params>>(), ...); }, testCases);
        ::benchmark::results::add_separator();
        std::apply([]<typename... Params>(Params&&...) { (testGEMV<false, T, std::remove_cvref_t<Params>>(), ...); }, testCases);
        ::benchmark::results::add_separator();
    } | std::tuple{float{}, double{}};
};

int main() { /* not needed by the UT framework */ }
