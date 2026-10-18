#include <boost/ut.hpp>

#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TensorMath.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <random>
#include <tuple>

using namespace gr::math;

const boost::ut::suite<"SVD gr::complex detail"> _svd_gr_complex = [] {
    using namespace boost::ut;
    "detail conj/innerProduct/sign accept gr::complex (bucket-B via ADL)"_test = [] {
        const auto c = gr::math::detail::conj(gr::complex<float>{1.f, 2.f});
        expect(eq(c.re, 1.f) and eq(c.im, -2.f));
        const auto ip = gr::math::detail::innerProduct(gr::complex<float>{1.f, 1.f}, gr::complex<float>{2.f, 0.f});
        expect(eq(ip.re, 2.f) and eq(ip.im, -2.f)); // conj(1+1i)*(2) = 2-2i
        const auto s = gr::math::detail::sign(gr::complex<float>{3.f, 4.f});
        expect(approx(s.re, 0.6f, 1e-6f) and approx(s.im, 0.8f, 1e-6f)); // (3+4i)/|5|
    };
};

template<typename T, svd::Algorithm Algo>
struct SvdTestConfig {
    using value_type                          = T;
    static constexpr svd::Algorithm algorithm = Algo;

    static std::string name() {
        std::string typeName = gr::meta::complex_like<T> ? "complex" : (std::is_same_v<T, float> ? "float" : "double");
        std::string algoName = (Algo == svd::Algorithm::Auto) ? "Auto" : (Algo == svd::Algorithm::Jacobi ? "Jacobi" : "GolubReinsch");
        return typeName + "/" + algoName;
    }
};

// Real-valued type/algorithm combinations
using RealSvdConfigs = std::tuple<                       //
    SvdTestConfig<double, svd::Algorithm::Auto>,         //
    SvdTestConfig<double, svd::Algorithm::Jacobi>,       //
    SvdTestConfig<double, svd::Algorithm::GolubReinsch>, //
    SvdTestConfig<float, svd::Algorithm::Auto>,          //
    SvdTestConfig<float, svd::Algorithm::Jacobi>,        //
    SvdTestConfig<float, svd::Algorithm::GolubReinsch>>; //

// Complex-valued type/algorithm combinations (GolubReinsch falls back to Jacobi)
using ComplexSvdConfigs = std::tuple<                                  //
    SvdTestConfig<std::complex<double>, svd::Algorithm::Auto>,         //
    SvdTestConfig<std::complex<double>, svd::Algorithm::Jacobi>,       //
    SvdTestConfig<std::complex<double>, svd::Algorithm::GolubReinsch>, //
    SvdTestConfig<std::complex<float>, svd::Algorithm::Auto>,          //
    SvdTestConfig<std::complex<float>, svd::Algorithm::Jacobi>,        //
    SvdTestConfig<std::complex<float>, svd::Algorithm::GolubReinsch>>; //

// All configurations
using AllSvdConfigs = std::tuple<                                      //
    SvdTestConfig<double, svd::Algorithm::Auto>,                       //
    SvdTestConfig<double, svd::Algorithm::Jacobi>,                     //
    SvdTestConfig<double, svd::Algorithm::GolubReinsch>,               //
    SvdTestConfig<float, svd::Algorithm::Auto>,                        //
    SvdTestConfig<float, svd::Algorithm::Jacobi>,                      //
    SvdTestConfig<float, svd::Algorithm::GolubReinsch>,                //
    SvdTestConfig<std::complex<double>, svd::Algorithm::Auto>,         //
    SvdTestConfig<std::complex<double>, svd::Algorithm::Jacobi>,       //
    SvdTestConfig<std::complex<double>, svd::Algorithm::GolubReinsch>, //
    SvdTestConfig<std::complex<float>, svd::Algorithm::Auto>,          //
    SvdTestConfig<std::complex<float>, svd::Algorithm::Jacobi>,        //
    SvdTestConfig<std::complex<float>, svd::Algorithm::GolubReinsch>>; //

// ============================================================================
// Test Helpers
// ============================================================================

template<gr::TensorLike Tensor>
void randomize(Tensor& tensor, typename Tensor::value_type min = typename Tensor::value_type{-1}, typename Tensor::value_type max = typename Tensor::value_type{1}, unsigned seed = 42) {
    using T = typename Tensor::value_type;
    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dis(min, max);
        for (auto& x : tensor) {
            x = dis(gen);
        }
    } else if constexpr (gr::meta::complex_like<T>) {
        using RealT = typename T::value_type;
        std::uniform_real_distribution<RealT> dis(std::real(min), std::real(max));
        for (auto& x : tensor) {
            x = T{dis(gen), dis(gen)};
        }
    } else {
        std::uniform_int_distribution<T> dis(min, max);
        for (auto& x : tensor) {
            x = dis(gen);
        }
    }
}

template<typename T>
[[nodiscard]] bool approx_equal(T a, T b, gr::meta::fundamental_base_value_type_t<T> epsilon = static_cast<gr::meta::fundamental_base_value_type_t<T>>(1e-6)) {
    if constexpr (gr::meta::complex_like<T>) {
        return std::abs(a - b) <= epsilon;
    } else {
        return std::abs(a - b) <= epsilon * std::max(T{1}, std::max(std::abs(a), std::abs(b)));
    }
}

template<gr::TensorLike TensorA, typename T = typename TensorA::value_type>
[[nodiscard]] bool tensors_approximately_equal(const TensorA& a, gr::TensorOf<T> auto const& b, gr::meta::fundamental_base_value_type_t<T> epsilon = static_cast<gr::meta::fundamental_base_value_type_t<T>>(1e-6)) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a.data()[i], b.data()[i], epsilon)) {
            return false;
        }
    }
    return true;
}

template<typename T>
[[nodiscard]] constexpr gr::meta::fundamental_base_value_type_t<T> defaultTolerance() {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    return std::is_same_v<RealT, float> ? static_cast<RealT>(1e-4f) : static_cast<RealT>(1e-9f);
}

// Reconstruct A from U, S, V and check against original
template<typename T, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] bool verifyReconstruction(const gr::Tensor<T>& A, const gr::Tensor<T>& U, const gr::Tensor<RealT>& S, const gr::Tensor<T>& V, RealT tol) {
    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);
    const std::size_t k = S.size();

    if (k == 0) {
        return true; // empty SVD is trivially correct
    }

    // Build diagonal matrix from S
    gr::Tensor<T> S_diag({k, k});
    S_diag.fill(T{0});
    for (std::size_t i = 0; i < k; ++i) {
        S_diag[i, i] = static_cast<T>(S[i]);
    }

    // US = U * S_diag
    gr::Tensor<T> US({m, k});
    US.fill(T{0});
    gemm(US, U, S_diag);

    // A_recon = US * V^H (use transpose for real, conjTranspose for complex)
    gr::Tensor<T> Vh;
    if constexpr (gr::meta::complex_like<T>) {
        Vh = conjTranspose(V);
    } else {
        Vh = transpose(V);
    }
    gr::Tensor<T> A_recon({m, n});
    A_recon.fill(T{0});
    gemm(A_recon, US, Vh);

    // Compute relative Frobenius error for debugging
    RealT normDiff{0}, normA{0};
    for (std::size_t i = 0; i < A.size(); ++i) {
        auto diff = A.data()[i] - A_recon.data()[i];
        if constexpr (gr::meta::complex_like<T>) {
            normDiff += std::norm(diff);
            normA += std::norm(A.data()[i]);
        } else {
            normDiff += diff * diff;
            normA += A.data()[i] * A.data()[i];
        }
    }
    RealT relError = std::sqrt(normDiff) / std::sqrt(normA);
    bool  pass     = tensors_approximately_equal(A, A_recon, tol);
    if (!pass) {
        std::cerr << "  verifyReconstruction FAILED: relError=" << relError << ", tol=" << tol << "\n";
    }
    return pass;
}

// Verify matrix is orthogonal/unitary: Q^H * Q = I
template<typename T, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
[[nodiscard]] bool verifyOrthogonal(const gr::Tensor<T>& Q, RealT tol) {
    const std::size_t cols = Q.extent(1);

    gr::Tensor<T> Qh = conjTranspose(Q);
    gr::Tensor<T> QhQ({cols, cols});
    QhQ.fill(T{0});
    gemm(QhQ, Qh, Q);

    for (std::size_t i = 0; i < cols; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            T expected = (i == j) ? T{1} : T{0};
            if (!approx_equal(QhQ[i, j], expected, tol)) {
                return false;
            }
        }
    }
    return true;
}

const boost::ut::suite<"SVD Basic Properties"> _svd_basic = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "singular values are non-negative"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({4, 4});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        for (const auto& s : S) {
            expect(ge(s, RealT{0})) << Config::name();
        }
    } | AllSvdConfigs{};

    "singular values are sorted descending"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({5, 5});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        for (std::size_t i = 0; i + 1 < S.size(); ++i) {
            expect(ge(S[i], S[i + 1])) << Config::name();
        }
    } | AllSvdConfigs{};

    "correct output dimensions for square matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({4, 4});
        randomize(A, T{-5}, T{5});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(U.extent(0), 4UZ)) << Config::name();
        expect(eq(U.extent(1), 4UZ)) << Config::name();
        expect(eq(S.size(), 4UZ)) << Config::name();
        expect(eq(V.extent(0), 4UZ)) << Config::name();
        expect(eq(V.extent(1), 4UZ)) << Config::name();
    } | AllSvdConfigs{};

    "correct output dimensions for tall matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({6, 3});
        randomize(A, T{-5}, T{5});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(U.extent(0), 6UZ)) << Config::name();
        expect(eq(U.extent(1), 3UZ)) << Config::name();
        expect(eq(S.size(), 3UZ)) << Config::name();
        expect(eq(V.extent(0), 3UZ)) << Config::name();
        expect(eq(V.extent(1), 3UZ)) << Config::name();
    } | AllSvdConfigs{};

    "correct output dimensions for wide matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({3, 6});
        randomize(A, T{-5}, T{5});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(U.extent(0), 3UZ)) << Config::name();
        expect(eq(U.extent(1), 3UZ)) << Config::name();
        expect(eq(S.size(), 3UZ)) << Config::name();
        expect(eq(V.extent(0), 6UZ)) << Config::name();
        expect(eq(V.extent(1), 3UZ)) << Config::name();
    } | AllSvdConfigs{};
};

const boost::ut::suite<"SVD Reconstruction"> _svd_reconstruction = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "reconstruction A = U*S*V^H (square)"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        // Use a simple known matrix instead of random
        Tensor<T> A({3, 3});
        A = {T{4}, T{1}, T{2}, T{1}, T{5}, T{3}, T{2}, T{3}, T{6}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};

    "reconstruction A = U*S*V^H (tall)"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({6, 3});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};

    "reconstruction A = U*S*V^H (wide)"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({3, 7});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};
};

const boost::ut::suite<"SVD Orthogonality"> _svd_orthogonality = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "U is orthogonal/unitary"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({5, 4});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(verifyOrthogonal(U, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};

    "V is orthogonal/unitary"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({4, 5});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(verifyOrthogonal(V, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};
};

const boost::ut::suite<"SVD Special Matrices"> _svd_special = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "identity matrix has all singular values = 1"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> I({4, 4});
        I.fill(T{0});
        for (std::size_t i = 0; i < 4; ++i) {
            I[i, i] = T{1};
        }

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, I, config);

        for (const auto& s : S) {
            expect(approx_equal(s, RealT{1}, defaultTolerance<T>())) << Config::name();
        }
    } | AllSvdConfigs{};

    "diagonal matrix singular values equal diagonal entries (sorted)"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> D({4, 4});
        D.fill(T{0});
        D[0, 0] = T{3};
        D[1, 1] = T{7};
        D[2, 2] = T{1};
        D[3, 3] = T{5};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, D, config);

        Tensor<RealT> expected({4});
        expected = {7, 5, 3, 1};
        expect(tensors_approximately_equal(S, expected, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "zero matrix has all singular values = 0"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> Z({3, 4});
        Z.fill(T{0});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, Z, config);

        for (const auto& s : S) {
            expect(approx_equal(s, RealT{0}, defaultTolerance<T>())) << Config::name();
        }
    } | AllSvdConfigs{};

    "single element matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({1, 1});
        A[0] = T{5};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(S.size(), 1UZ)) << Config::name();
        expect(approx_equal(S[0], RealT{5}, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};
};

const boost::ut::suite<"SVD Algorithm Comparison"> _svd_algo_compare = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Jacobi and GolubReinsch produce same singular values"_test = [] {
        "small square matrix"_test = [] {
            Tensor<double> A({4, 4});
            A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, A, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, A, grConfig);

            // verify singular values match
            expect(tensors_approximately_equal(S_j, S_gr, 1e-9));

            // verify reconstruction works for both algorithms
            expect(verifyReconstruction(A, U_j, S_j, V_j, 1e-9));
            expect(verifyReconstruction(A, U_gr, S_gr, V_gr, 1e-9));
        };

        "random 5x5"_test = [] {
            Tensor<double> A({5, 5});
            randomize(A, -10.0, 10.0);

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, A, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, A, grConfig);

            expect(tensors_approximately_equal(S_j, S_gr, 1e-9));
        };

        "tall matrix 8x3"_test = [] {
            Tensor<double> A({8, 3});
            randomize(A, -5.0, 5.0);

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, A, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, A, grConfig);

            expect(tensors_approximately_equal(S_j, S_gr, 1e-9));
        };

        "wide matrix 3x8"_test = [] {
            Tensor<double> A({3, 8});
            randomize(A, -5.0, 5.0);

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, A, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, A, grConfig);

            expect(tensors_approximately_equal(S_j, S_gr, 1e-9));
        };

        "rank-deficient matrix"_test = [] {
            Tensor<double> A({4, 4});
            // rank-1 matrix: A = u * v^T
            A = {1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16};

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, A, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, A, grConfig);

            expect(tensors_approximately_equal(S_j, S_gr, 1e-6));
        };

        "Hilbert 4x4 (ill-conditioned)"_test = [] {
            Tensor<double> H({4, 4});
            for (std::size_t i = 0; i < 4; ++i) {
                for (std::size_t j = 0; j < 4; ++j) {
                    H[i, j] = 1.0 / static_cast<double>(i + j + 1);
                }
            }

            svd::Config<double> jacobiConfig;
            jacobiConfig.algorithm = svd::Algorithm::Jacobi;

            svd::Config<double> grConfig;
            grConfig.algorithm = svd::Algorithm::GolubReinsch;

            Tensor<double> U_j, V_j, S_j;
            Tensor<double> U_gr, V_gr, S_gr;

            gesvd(U_j, S_j, V_j, H, jacobiConfig);
            gesvd(U_gr, S_gr, V_gr, H, grConfig);

            expect(tensors_approximately_equal(S_j, S_gr, 1e-9));
        };
    };
};

const boost::ut::suite<"SVD Edge Cases"> _svd_edge_cases = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "empty matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({0, 0});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        expect(eq(U.size(), 0UZ)) << Config::name();
        expect(eq(S.size(), 0UZ)) << Config::name();
        expect(eq(V.size(), 0UZ)) << Config::name();
    } | RealSvdConfigs{};

    "single row vector"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({1, 5});
        A = {T{1}, T{2}, T{3}, T{4}, T{5}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(U.extent(0), 1UZ)) << Config::name();
        expect(eq(S.size(), 1UZ)) << Config::name();
        expect(eq(V.extent(0), 5UZ)) << Config::name();

        RealT expected_s = std::sqrt(RealT{1 + 4 + 9 + 16 + 25});
        expect(approx_equal(S[0], expected_s, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "single column vector"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({5, 1});
        A = {T{1}, T{2}, T{3}, T{4}, T{5}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        expect(eq(U.extent(0), 5UZ)) << Config::name();
        expect(eq(S.size(), 1UZ)) << Config::name();
        expect(eq(V.extent(0), 1UZ)) << Config::name();

        RealT expected_s = std::sqrt(RealT{1 + 4 + 9 + 16 + 25});
        expect(approx_equal(S[0], expected_s, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "very tall matrix 100x2"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({100, 2});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        expect(eq(U.extent(0), 100UZ)) << Config::name();
        expect(eq(U.extent(1), 2UZ)) << Config::name();
        expect(eq(S.size(), 2UZ)) << Config::name();
        expect(eq(V.extent(0), 2UZ)) << Config::name();
        expect(eq(V.extent(1), 2UZ)) << Config::name();
        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "very wide matrix 2x100"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({2, 100});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        expect(eq(U.extent(0), 2UZ)) << Config::name();
        expect(eq(U.extent(1), 2UZ)) << Config::name();
        expect(eq(S.size(), 2UZ)) << Config::name();
        expect(eq(V.extent(0), 100UZ)) << Config::name();
        expect(eq(V.extent(1), 2UZ)) << Config::name();
        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "rank-1 outer product"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        // A = u * v^T where u = [1,2,3,4]^T, v = [1,1,1]^T
        Tensor<T> A({4, 3});
        A = {T{1}, T{1}, T{1}, T{2}, T{2}, T{2}, T{3}, T{3}, T{3}, T{4}, T{4}, T{4}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        gesvd(U, S, V, A, config);

        // Only one non-zero singular value
        expect(gt(S[0], RealT{1})) << Config::name();
        expect(approx_equal(S[1], RealT{0}, static_cast<RealT>(1e-6f))) << Config::name();
        expect(approx_equal(S[2], RealT{0}, static_cast<RealT>(1e-6f))) << Config::name();
    } | RealSvdConfigs{};

    "clustered singular values"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        // Diagonal with very close values
        Tensor<T> D({4, 4});
        D.fill(T{0});
        D[0, 0] = T{1.0};
        D[1, 1] = static_cast<T>(1.0f + 1e-10f);
        D[2, 2] = static_cast<T>(1.0f + 2e-10f);
        D[3, 3] = T{0.5};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, D, config);

        expect(status == svd::Status::Success) << Config::name();
        // Check reconstruction even with clustered values
        expect(verifyReconstruction(D, U, S, V, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "matrix with embedded zero row"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({4, 3});
        A = {T{1}, T{2}, T{3}, T{0}, T{0}, T{0}, T{4}, T{5}, T{6}, T{7}, T{8}, T{9}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "matrix with embedded zero column"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({3, 4});
        A = {T{1}, T{0}, T{2}, T{3}, T{4}, T{0}, T{5}, T{6}, T{7}, T{0}, T{8}, T{9}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | RealSvdConfigs{};

    "Vandermonde-like matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        // V[i,j] = x[i]^j for x = [1, 2, 3, 4]
        Tensor<T> V_mat({4, 4});
        for (std::size_t i = 0; i < 4; ++i) {
            T x = T(i + 1);
            for (std::size_t j = 0; j < 4; ++j) {
                V_mat[i, j] = std::pow(x, T(j));
            }
        }

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, V_mat, config);

        expect(status == svd::Status::Success) << Config::name();
        // Vandermonde matrices are ill-conditioned, need looser tolerance
        auto tol = std::is_same_v<RealT, float> ? static_cast<RealT>(1e-3f) : static_cast<RealT>(1e-8);
        expect(verifyReconstruction(V_mat, U, S, V, tol)) << Config::name();
    } | RealSvdConfigs{};
};

const boost::ut::suite<"SVD Numerical Stability"> _svd_stability = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Hilbert matrix (ill-conditioned)"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        if constexpr (!gr::meta::complex_like<T>) {
            Tensor<T> H({5, 5});
            for (std::size_t i = 0; i < 5; ++i) {
                for (std::size_t j = 0; j < 5; ++j) {
                    H[i, j] = T{1} / T(i + j + 1);
                }
            }

            svd::Config<RealT> config;
            config.algorithm = Config::algorithm;

            Tensor<T>     U, V;
            Tensor<RealT> S;
            auto          status = gesvd(U, S, V, H, config);

            expect(status == svd::Status::Success) << Config::name();
            // Hilbert matrices are highly ill-conditioned (cond ~10^5 for 5x5), need looser tolerance
            auto tol = std::is_same_v<RealT, float> ? static_cast<RealT>(1e-2f) : static_cast<RealT>(1e-8);
            expect(verifyReconstruction(H, U, S, V, tol)) << Config::name();
        }
    } | RealSvdConfigs{};

    "matrix with widely varying singular values"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        if constexpr (!gr::meta::complex_like<T>) {
            Tensor<T> A({3, 3});
            A.fill(T{0});
            A[0, 0] = T{1e6};
            A[1, 1] = T{1};
            A[2, 2] = static_cast<T>(1e-6f);

            svd::Config<RealT> config;
            config.algorithm = Config::algorithm;

            Tensor<T>     U, V;
            Tensor<RealT> S;
            gesvd(U, S, V, A, config);

            expect(approx_equal(S[0], RealT{1e6}, RealT{1})) << Config::name();
            expect(approx_equal(S[1], RealT{1}, static_cast<RealT>(1e-6f))) << Config::name();
            expect(approx_equal(S[2], static_cast<RealT>(1e-6f), static_cast<RealT>(1e-12f))) << Config::name();
        }
    } | RealSvdConfigs{};

    "nearly singular matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        // Matrix with condition number ~ 10^6
        Tensor<T> A({3, 3});
        A = {T{1}, T{1}, T{1}, T{1}, T{1}, static_cast<T>(1.000001f), T{1}, static_cast<T>(1.000001f), T{1}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        // Smallest singular value should be very small
        expect(lt(S[2], static_cast<RealT>(1e-4f))) << Config::name();
    } | RealSvdConfigs{};

    "matrix with negative elements"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        Tensor<T> A({3, 3});
        A = {T{-5}, T{2}, T{-1}, T{3}, T{-4}, T{2}, T{-1}, T{1}, T{-3}};

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success) << Config::name();
        // All singular values must be non-negative
        for (const auto& s : S) {
            expect(ge(s, RealT{0})) << Config::name();
        }
        expect(verifyReconstruction(A, U, S, V, defaultTolerance<T>())) << Config::name();
    } | AllSvdConfigs{};
};

const boost::ut::suite<"SVD Large Matrices"> _svd_large = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "50x50 random matrix"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        if constexpr (!gr::meta::complex_like<T>) {
            Tensor<T> A({50, 50});
            randomize(A, T{-10}, T{10});

            svd::Config<RealT> config;
            config.algorithm = Config::algorithm;

            Tensor<T>     U, V;
            Tensor<RealT> S;
            auto          status = gesvd(U, S, V, A, config);

            expect(status == svd::Status::Success) << Config::name();
            expect(eq(S.size(), 50UZ)) << Config::name();

            for (std::size_t i = 0; i + 1 < S.size(); ++i) {
                expect(ge(S[i], S[i + 1])) << Config::name();
            }
        }
    } | RealSvdConfigs{};

    "100x100 random double"_test = [] {
        using Config = SvdTestConfig<double, svd::Algorithm::Auto>;
        using T      = double;
        using RealT  = double;

        Tensor<T> A({100, 100});
        randomize(A, T{-10}, T{10});

        svd::Config<RealT> config;
        config.algorithm = Config::algorithm;

        Tensor<T>     U, V;
        Tensor<RealT> S;
        auto          status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success);
        expect(eq(S.size(), 100UZ));
    };

    "30x100 rectangular"_test = []<typename Config> {
        using T     = Config::value_type;
        using RealT = gr::meta::fundamental_base_value_type_t<T>;

        if constexpr (!gr::meta::complex_like<T>) {
            Tensor<T> A({30, 100});
            randomize(A, T{-5}, T{5});

            svd::Config<RealT> config;
            config.algorithm = Config::algorithm;

            Tensor<T>     U, V;
            Tensor<RealT> S;
            auto          status = gesvd(U, S, V, A, config);

            expect(status == svd::Status::Success) << Config::name();
            expect(eq(U.extent(0), 30UZ)) << Config::name();
            expect(eq(U.extent(1), 30UZ)) << Config::name();
            expect(eq(S.size(), 30UZ)) << Config::name();
            expect(eq(V.extent(0), 100UZ)) << Config::name();
            expect(eq(V.extent(1), 30UZ)) << Config::name();
        }
    } | RealSvdConfigs{};
};

const boost::ut::suite<"SVD Configuration"> _svd_config = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "compute only k singular values"_test = [] {
        Tensor<double> A({6, 6});
        randomize(A, -10.0, 10.0);

        svd::Config<double> config;
        config.k = 3;

        Tensor<double> U, V, S;
        auto           status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::EarlyReturn);
        expect(eq(S.size(), 3UZ));
        expect(eq(U.extent(1), 3UZ));
        expect(eq(V.extent(1), 3UZ));

        for (std::size_t i = 0; i + 1 < S.size(); ++i) {
            expect(ge(S[i], S[i + 1]));
        }
    };

    "custom tolerance"_test = [] {
        Tensor<double> A({4, 4});
        randomize(A, -10.0, 10.0);

        svd::Config<double> config;
        config.tolerance = 1e-14;

        Tensor<double> U, V, S;
        auto           status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success);
    };

    "custom max iterations"_test = [] {
        Tensor<double> A({4, 4});
        randomize(A, -10.0, 10.0);

        svd::Config<double> config;
        config.maxIterations = 1000;

        Tensor<double> U, V, S;
        auto           status = gesvd(U, S, V, A, config);

        expect(status == svd::Status::Success);
    };

    "singular values only (no U, V)"_test = [] {
        Tensor<double> A({5, 5});
        randomize(A, -10.0, 10.0);

        Tensor<double> S;
        auto           status = gesvd(S, A);

        expect(status == svd::Status::Success);
        expect(eq(S.size(), 5UZ));

        // Compare with full SVD
        Tensor<double> U, V, S_full;
        gesvd(U, S_full, V, A);

        expect(tensors_approximately_equal(S, S_full, 1e-9));
    };
};

const boost::ut::suite<"SVD Complex Support"> _svd_complex = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;
    using Complex  = std::complex<double>;
    using ComplexF = std::complex<float>;

    "complex reconstruction with conjugate transpose"_test = [] {
        Tensor<Complex> A({4, 4});
        randomize(A, Complex{-5, -5}, Complex{5, 5});

        Tensor<Complex> U, V;
        Tensor<double>  S;
        gesvd(U, S, V, A);

        expect(verifyReconstruction(A, U, S, V, 1e-9));
    };

    "complex U^H * U = I"_test = [] {
        Tensor<Complex> A({5, 3});
        randomize(A, Complex{-10, -10}, Complex{10, 10});

        Tensor<Complex> U, V;
        Tensor<double>  S;
        gesvd(U, S, V, A);

        expect(verifyOrthogonal(U, 1e-9));
    };

    "complex V^H * V = I"_test = [] {
        Tensor<Complex> A({3, 5});
        randomize(A, Complex{-10, -10}, Complex{10, 10});

        Tensor<Complex> U, V;
        Tensor<double>  S;
        gesvd(U, S, V, A);

        expect(verifyOrthogonal(V, 1e-9));
    };

    "complex float precision"_test = [] {
        Tensor<ComplexF> A({3, 3});
        A = {ComplexF{1, 2}, ComplexF{3, 4}, ComplexF{5, 6}, ComplexF{7, 8}, ComplexF{9, 10}, ComplexF{11, 12}, ComplexF{13, 14}, ComplexF{15, 16}, ComplexF{17, 18}};

        Tensor<ComplexF> U, V;
        Tensor<float>    S;
        auto             status = gesvd(U, S, V, A);

        expect(status == svd::Status::Success);
        expect(eq(S.size(), 3UZ));

        for (const auto& s : S) {
            expect(ge(s, 0.0f));
        }
    };

    "Hermitian matrix"_test = [] {
        // A = A^H
        Tensor<Complex> A({3, 3});
        A[0, 0] = Complex{3, 0};
        A[0, 1] = Complex{1, 2};
        A[0, 2] = Complex{0, -1};
        A[1, 0] = Complex{1, -2};
        A[1, 1] = Complex{4, 0};
        A[1, 2] = Complex{2, 1};
        A[2, 0] = Complex{0, 1};
        A[2, 1] = Complex{2, -1};
        A[2, 2] = Complex{5, 0};

        Tensor<Complex> U, V;
        Tensor<double>  S;
        auto            status = gesvd(U, S, V, A);

        expect(status == svd::Status::Success);
        expect(verifyReconstruction(A, U, S, V, 1e-9));
    };
};

const boost::ut::suite<"GolubReinsch Internals -- Regression Tests"> _gr_internals = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;
    using gr::math::detail::householderVector;
    using gr::math::detail::applyHouseholderLeft;
    using gr::math::detail::bidiagonalise;
    using gr::math::detail::accumulateU;
    using gr::math::detail::accumulateV;

    "Householder vector computation"_test = [] {
        "basic 3-element vector"_test = [] {
            std::vector<double> x = {3.0, 4.0, 0.0};
            double              tau;

            double beta = householderVector(x.data(), 3, tau);

            expect(approx_equal(beta, -5.0, 1e-10)) << "beta = " << beta;

            // Verify H*[3,4,0] = [beta, 0, 0]
            std::vector<double> orig = {3.0, 4.0, 0.0};
            double              v0 = 1.0, v1 = x[1], v2 = x[2];
            double              dot   = v0 * orig[0] + v1 * orig[1] + v2 * orig[2];
            double              scale = tau * dot;

            expect(approx_equal(orig[0] - scale * v0, beta, 1e-10));
            expect(approx_equal(orig[1] - scale * v1, 0.0, 1e-10));
            expect(approx_equal(orig[2] - scale * v2, 0.0, 1e-10));
        };

        "single element (no reflection needed)"_test = [] {
            std::vector<double> x = {5.0};
            double              tau;
            double              beta = householderVector(x.data(), 1, tau);

            expect(approx_equal(beta, 5.0, 1e-10));
            expect(approx_equal(tau, 0.0, 1e-10));
        };

        "negative first element"_test = [] {
            std::vector<double> x = {-3.0, 4.0};
            double              tau;
            double              beta = householderVector(x.data(), 2, tau);

            expect(approx_equal(beta, 5.0, 1e-10)) << "beta = " << beta;
        };
    };

    "Bidiagonalisation"_test = [] {
        "produces correct structure"_test = [] {
            Tensor<double> A({4, 3});
            A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

            Tensor<double> A_orig(A);

            std::vector<double> tauU, tauV, diag, superdiag;
            bidiagonalise(A, tauU, tauV, diag, superdiag);

            expect(eq(diag.size(), 3UZ));
            expect(eq(superdiag.size(), 2UZ));
            expect(eq(tauU.size(), 3UZ));
            expect(eq(tauV.size(), 2UZ));

            // Accumulate U and V
            Tensor<double> U, V;
            accumulateU(A, tauU, U);
            accumulateV(A, tauV, V);

            // Build bidiagonal B
            Tensor<double> B({3, 3});
            B.fill(0.0);
            for (std::size_t i = 0; i < 3; ++i) {
                B[i, i] = diag[i];
            }
            for (std::size_t i = 0; i < 2; ++i) {
                B[i, i + 1] = superdiag[i];
            }

            // Verify U^T * A_orig * V ≈ B
            Tensor<double> UT = transpose(U);
            Tensor<double> UtA({3, 3});
            UtA.fill(0.0);
            gemm(UtA, UT, A_orig);

            Tensor<double> UtAV({3, 3});
            UtAV.fill(0.0);
            gemm(UtAV, UtA, V);

            expect(tensors_approximately_equal(UtAV, B, 1e-9));
        };

        "U and V are orthogonal"_test = [] {
            Tensor<double> A({4, 3});
            randomize(A, -10.0, 10.0);

            std::vector<double> tauU, tauV, diag, superdiag;
            bidiagonalise(A, tauU, tauV, diag, superdiag);

            Tensor<double> U, V;
            accumulateU(A, tauU, U);
            accumulateV(A, tauV, V);

            expect(verifyOrthogonal(U, 1e-9));
            expect(verifyOrthogonal(V, 1e-9));
        };

        "square 4x4 matrix produces correct structure"_test = [] {
            Tensor<double> A({4, 4});
            A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

            Tensor<double> A_orig(A);

            std::vector<double> tauU, tauV, diag, superdiag;
            bidiagonalise(A, tauU, tauV, diag, superdiag);

            expect(eq(diag.size(), 4UZ));
            expect(eq(superdiag.size(), 3UZ));

            // Accumulate U and V
            Tensor<double> U, V;
            accumulateU(A, tauU, U);
            accumulateV(A, tauV, V);

            // Build bidiagonal B
            Tensor<double> B({4, 4});
            B.fill(0.0);
            for (std::size_t i = 0; i < 4; ++i) {
                B[i, i] = diag[i];
            }
            for (std::size_t i = 0; i < 3; ++i) {
                B[i, i + 1] = superdiag[i];
            }

            // Verify U^T * A_orig * V ≈ B
            Tensor<double> UT = transpose(U);
            Tensor<double> UtA({4, 4});
            UtA.fill(0.0);
            gemm(UtA, UT, A_orig);

            Tensor<double> UtAV({4, 4});
            UtAV.fill(0.0);
            gemm(UtAV, UtA, V);

            // Verify U^T * A * V = B (bidiagonal)
            expect(tensors_approximately_equal(UtAV, B, 1e-9));

            // Verify A = U * B * V^T (reconstruction)
            Tensor<double> UB({4, 4});
            UB.fill(0.0);
            gemm(UB, U, B);
            Tensor<double> UBVt({4, 4});
            UBVt.fill(0.0);
            Tensor<double> Vt = transpose(V);
            gemm(UBVt, UB, Vt);

            expect(tensors_approximately_equal(A_orig, UBVt, 1e-9));
        };
    };
};

template<typename TDifference>
[[nodiscard]] double maxAbsDiff(std::size_t rows, std::size_t cols, TDifference&& difference) {
    double largest = 0.0;
    for (std::size_t i = 0UZ; i < rows; ++i) {
        for (std::size_t j = 0UZ; j < cols; ++j) {
            largest = std::max(largest, static_cast<double>(std::abs(difference(i, j))));
        }
    }
    return largest;
}

/** the third column holds (1, 2.5, 1.5), the other two are zero: rank one with singular value sqrt(9.5) */
[[nodiscard]] gr::Tensor<double> zeroColumnsMatrix() {
    gr::Tensor<double> A({3UZ, 3UZ});
    A.fill(0.0);
    A[0UZ, 2UZ] = 1.0;
    A[1UZ, 2UZ] = 2.5;
    A[2UZ, 2UZ] = 1.5;
    return A;
}

template<typename TensorA>
struct Decomposition {
    using Real = gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>;

    TensorA          u; /**< U [m, k] */
    gr::Tensor<Real> s; /**< the singular values */
    TensorA          v; /**< V [n, k] */
    svd::Status      status;
};

template<typename TensorA>
[[nodiscard]] Decomposition<TensorA> decompose(const TensorA& A, svd::Algorithm algorithm) {
    Decomposition<TensorA>                       result;
    svd::Config<typename decltype(result)::Real> config;
    config.algorithm = algorithm;
    result.status    = gesvd(result.u, result.s, result.v, A, config);
    return result;
}

const boost::ut::suite<"SVD pseudo-inverse"> _svd_pseudo_inverse = [] {
    using namespace boost::ut;
    using gr::Tensor;

    "Tikhonov filter factors s / (s^2 + lambda^2) of diag(1, 1e-3) with lambda = 1e-2"_test = [] {
        Tensor<double> A({2UZ, 2UZ});
        A.fill(0.0);
        A[0UZ, 0UZ]   = 1.0;
        A[1UZ, 1UZ]   = 1e-3;
        const auto pi = pseudo_inverse(A, pseudo_inverse_config<double>{.tikhonov = 1e-2});
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 2UZ));
        expect(le(std::abs(pi->inverse[0UZ, 0UZ] - 1.0 / (1.0 + 1e-4)), 1e-12));
        expect(le(std::abs(pi->inverse[1UZ, 1UZ] - 1e-3 / (1e-6 + 1e-4)), 1e-12));
        expect(le(std::abs(pi->inverse[0UZ, 1UZ]) + std::abs(pi->inverse[1UZ, 0UZ]), 1e-12));
    };
    "a cutoff of 1e-2 drops the singular value 1e-3 of diag(1, 1e-3): rank 1"_test = [] {
        Tensor<double> A({2UZ, 2UZ});
        A.fill(0.0);
        A[0UZ, 0UZ]   = 1.0;
        A[1UZ, 1UZ]   = 1e-3;
        const auto pi = pseudo_inverse(A, pseudo_inverse_config<double>{.cutoff = 1e-2});
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 1UZ));
        expect(le(std::abs(pi->inverse[0UZ, 0UZ] - 1.0), 1e-12));
        expect(le(std::abs(pi->inverse[1UZ, 1UZ]), 1e-12));
    };
    "the pseudo-inverse of a tall diagonal matrix, by hand"_test = [] {
        Tensor<double> A({3UZ, 2UZ});
        A.fill(0.0);
        A[0UZ, 0UZ]   = 1.0;
        A[1UZ, 1UZ]   = 2.0;
        const auto pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 2UZ));
        expect(pi->inverse.extent(0UZ) == 2UZ && pi->inverse.extent(1UZ) == 3UZ);
        const std::array<std::array<double, 3>, 2> expected{{{1.0, 0.0, 0.0}, {0.0, 0.5, 0.0}}};
        expect(le(maxAbsDiff(2UZ, 3UZ, [&](std::size_t i, std::size_t j) { return pi->inverse[i, j] - expected[i][j]; }), 1e-12));
    };

    "a rank-one matrix [[1, 2], [2, 4]]: rank 1, pseudo-inverse A^T / 25"_test = [] {
        Tensor<double> A({2UZ, 2UZ});
        A[0UZ, 0UZ]   = 1.0;
        A[0UZ, 1UZ]   = 2.0;
        A[1UZ, 0UZ]   = 2.0;
        A[1UZ, 1UZ]   = 4.0;
        const auto pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 1UZ));
        expect(le(maxAbsDiff(2UZ, 2UZ, [&](std::size_t i, std::size_t j) { return pi->inverse[i, j] - A[j, i] / 25.0; }), 1e-12));
    };

    "a rank-one matrix with zero columns: its pseudo-inverse b^T / |b|^2 in the third row"_test = [] {
        const auto A  = zeroColumnsMatrix();
        const auto pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 1UZ));
        const std::array<double, 3> b{1.0, 2.5, 1.5};
        for (std::size_t j = 0UZ; j < 3UZ; ++j) {
            expect(le(std::abs(pi->inverse[2UZ, j] - b[j] / 9.5), 1e-12)) << "column" << j;
            expect(le(std::abs(pi->inverse[0UZ, j]) + std::abs(pi->inverse[1UZ, j]), 1e-12)) << "column" << j;
        }
    };

    "a complex diagonal matrix: its inverse, conjugates included"_test = [] {
        using C = std::complex<double>;
        Tensor<C> A({2UZ, 2UZ});
        A.fill(C{});
        A[0UZ, 0UZ]   = C(0.0, 1.0);
        A[1UZ, 1UZ]   = C(2.0, 0.0);
        const auto pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 2UZ));
        expect(le(std::abs(pi->inverse[0UZ, 0UZ] - C(0.0, -1.0)), 1e-12)) << "1/i = -i";
        expect(le(std::abs(pi->inverse[1UZ, 1UZ] - C(0.5, 0.0)), 1e-12));
        expect(le(std::abs(pi->inverse[0UZ, 1UZ]) + std::abs(pi->inverse[1UZ, 0UZ]), 1e-12));
    };

    "all ones plus 1 % on the diagonal, three equal singular values: its inverse (I - J/4.01) / 0.01"_test = [] {
        using C = std::complex<double>;
        Tensor<C> A({4UZ, 4UZ});
        for (std::size_t a = 0UZ; a < 4UZ; ++a) {
            for (std::size_t b = 0UZ; b < 4UZ; ++b) {
                A[a, b] = C{1.0 + (a == b ? 0.01 : 0.0), 0.0};
            }
        }
        const auto pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        expect(eq(pi->rank, 4UZ));
        /** (J + lambda I)^-1 = (I - J / (lambda + 4)) / lambda */
        const auto expected = [](std::size_t a, std::size_t b) { return ((a == b ? 1.0 : 0.0) - 1.0 / 4.01) / 0.01; };
        expect(le(maxAbsDiff(4UZ, 4UZ, [&](std::size_t a, std::size_t b) { return pi->inverse[a, b] - C{expected(a, b), 0.0}; }), 1e-9));
    };

    "the pseudo-inverse solves an overdetermined consistent system exactly"_test = [] {
        Tensor<double>              A({4UZ, 2UZ});
        const std::array<double, 8> rows{1, 0, 0, 1, 1, 1, 1, -1};
        for (std::size_t n = 0UZ; n < rows.size(); ++n) {
            A[n / 2UZ, n % 2UZ] = rows[n];
        }
        const std::array<double, 2> x{3.0, -2.0};
        const auto                  pi = pseudo_inverse(A);
        expect(fatal(pi.has_value()));
        for (std::size_t i = 0UZ; i < 2UZ; ++i) {
            double solution = 0.0;
            for (std::size_t j = 0UZ; j < 4UZ; ++j) {
                solution += pi->inverse[i, j] * (A[j, 0UZ] * x[0] + A[j, 1UZ] * x[1]);
            }
            expect(le(std::abs(solution - x[i]), 1e-12)) << "x" << i;
        }
    };

    "a loaded rank-one complex matrix of entries 1e10: the inverse of its 1e-10 scaled copy, times 1e-10, without scaling"_test = [] {
        using C = std::complex<double>;
        Tensor<C> A({4UZ, 4UZ});
        Tensor<C> B({4UZ, 4UZ});
        for (std::size_t a = 0UZ; a < 4UZ; ++a) {
            for (std::size_t b = 0UZ; b < 4UZ; ++b) {
                B[a, b] = C{1.0 + (a == b ? 0.01 : 0.0), 0.0};
                A[a, b] = B[a, b] * 1e10;
            }
        }
        const auto large = pseudo_inverse(A);
        const auto unit  = pseudo_inverse(B);
        expect(fatal(large.has_value() && unit.has_value()));
        expect(eq(large->rank, 4UZ));
        expect(le(maxAbsDiff(4UZ, 4UZ, [&](std::size_t a, std::size_t b) { return large->inverse[a, b] * 1e10 - unit->inverse[a, b]; }), 1e-9 * std::abs(unit->inverse[0UZ, 0UZ])));
    };
};

const boost::ut::suite<"SVD fixes"> _svd_fixes = [] {
    using namespace boost::ut;
    using gr::Tensor;

    const auto orthonormalError = [](const auto& M) {
        using T      = typename std::remove_cvref_t<decltype(M)>::value_type;
        double error = 0.0;
        for (std::size_t a = 0UZ; a < M.extent(1); ++a) {
            for (std::size_t b = 0UZ; b < M.extent(1); ++b) {
                T dot{};
                for (std::size_t k = 0UZ; k < M.extent(0); ++k) {
                    dot += gr::math::detail::conj(M[k, a]) * M[k, b];
                }
                error = std::max(error, static_cast<double>(std::abs(dot - static_cast<T>(a == b ? 1 : 0))));
            }
        }
        return error;
    };

    const auto loadedOnes = []<typename T>(T scale) {
        Tensor<T> A({3UZ, 3UZ});
        for (std::size_t a = 0UZ; a < 3UZ; ++a) {
            for (std::size_t b = 0UZ; b < 3UZ; ++b) {
                A[a, b] = scale * (a == b ? static_cast<T>(1.01) : T{1});
            }
        }
        return A;
    };

    const auto checkLoadedOnes = [&]<typename T>(T scale) {
        const auto A                 = loadedOnes(scale);
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::Jacobi);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success))) << scale;
        const T eps = std::numeric_limits<T>::epsilon();
        expect(le(std::abs(S[0UZ] / scale - static_cast<T>(3.01)), 8 * eps * static_cast<T>(3.01))) << scale;
        expect(le(std::abs(S[1UZ] / scale - static_cast<T>(0.01)), 8 * eps * static_cast<T>(3.01))) << scale;
        expect(le(std::abs(S[2UZ] / scale - static_cast<T>(0.01)), 8 * eps * static_cast<T>(3.01))) << scale;
        expect(le(orthonormalError(U), 16.0 * static_cast<double>(eps)));
        expect(le(orthonormalError(V), 16.0 * static_cast<double>(eps)));
    };

    "Jacobi float: all ones plus 1 % on the diagonal, scaled by 1e-12, 1 and 1e10"_test = [&] {
        checkLoadedOnes(1e-12F);
        checkLoadedOnes(1.0F);
        checkLoadedOnes(1e10F);
    };

    "Jacobi double: all ones plus 1 % on the diagonal, scaled by 1e-12, 1 and 1e10"_test = [&] {
        checkLoadedOnes(1e-12);
        checkLoadedOnes(1.0);
        checkLoadedOnes(1e10);
    };

    const auto checkLoadedRankOne = [&]<typename R>(R) {
        using C = std::complex<R>;
        Tensor<C> A({4UZ, 4UZ});
        for (std::size_t r = 0UZ; r < 4UZ; ++r) {
            for (std::size_t c = 0UZ; c < 4UZ; ++c) {
                const C ar = std::polar(R{1}, static_cast<R>(0.3 * static_cast<double>(r)));
                const C ac = std::polar(R{1}, static_cast<R>(0.3 * static_cast<double>(c)));
                A[r, c]    = ar * std::conj(ac) + (r == c ? C{static_cast<R>(0.01)} : C{});
            }
        }
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::Jacobi);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success)));
        const R eps = std::numeric_limits<R>::epsilon();
        expect(le(std::abs(S[0UZ] - static_cast<R>(4.01)), 16 * eps * static_cast<R>(4.01)));
        for (std::size_t k = 1UZ; k < 4UZ; ++k) {
            expect(le(std::abs(S[k] - static_cast<R>(0.01)), 16 * eps * static_cast<R>(4.01))) << k;
        }
        expect(le(orthonormalError(U), 32.0 * static_cast<double>(eps)));
        expect(le(orthonormalError(V), 32.0 * static_cast<double>(eps)));
    };

    "Jacobi complex: loaded rank-one 4x4 a a^H + 0.01 I reaches Success, singular values 4.01 and 0.01 x3"_test = [&] {
        checkLoadedRankOne(1.0F);
        checkLoadedRankOne(1.0);
    };

    "Golub-Reinsch: a matrix with two zero columns reconstructs"_test = [&] {
        const auto A                 = zeroColumnsMatrix();
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::GolubReinsch);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success)));
        expect(le(std::abs(S[0UZ] - std::sqrt(9.5)), 1e-14));
        double error = 0.0;
        for (std::size_t i = 0UZ; i < 3UZ; ++i) {
            for (std::size_t j = 0UZ; j < 3UZ; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0UZ; k < 3UZ; ++k) {
                    sum += U[i, k] * S[k] * V[j, k];
                }
                error = std::max(error, std::abs(sum - A[i, j]));
            }
        }
        expect(le(error, 1e-14));
        expect(le(orthonormalError(U), 1e-14));
        expect(le(orthonormalError(V), 1e-14));
    };

    "Golub-Reinsch: [[1, 1], [1, 1]] has singular values (2, 0) and Success"_test = [&] {
        Tensor<double> A({2UZ, 2UZ});
        A.fill(1.0);
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::GolubReinsch);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success)));
        expect(le(std::abs(S[0UZ] - 2.0), 1e-14));
        expect(le(std::abs(S[1UZ]), 1e-14));
        expect(le(orthonormalError(U), 1e-14));
        expect(le(orthonormalError(V), 1e-14));
    };

    "Jacobi rank-deficient: zero columns of U are completed to an orthonormal basis"_test = [&] {
        const auto A                 = zeroColumnsMatrix();
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::Jacobi);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success)));
        expect(le(orthonormalError(U), 1e-14));
        expect(le(orthonormalError(V), 1e-14));
    };

    "Jacobi rank-one dense matrix a b^T: singular value |a||b| and an orthonormal U"_test = [&] {
        Tensor<double>              A({3UZ, 3UZ});
        const std::array<double, 3> a{1.0, 2.0, 3.0};
        const std::array<double, 3> b{1.0, -1.0, 2.0};
        for (std::size_t i = 0UZ; i < 3UZ; ++i) {
            for (std::size_t j = 0UZ; j < 3UZ; ++j) {
                A[i, j] = a[i] * b[j];
            }
        }
        const auto [U, S, V, status] = decompose(A, svd::Algorithm::Jacobi);
        expect(eq(static_cast<int>(status), static_cast<int>(svd::Status::Success)));
        expect(le(std::abs(S[0UZ] - std::sqrt(14.0 * 6.0)), 1e-14));
        expect(le(S[1UZ], 1e-14));
        expect(le(orthonormalError(U), 1e-14));
    };
};

const boost::ut::suite<"cholesky_factor and cholesky_solve_factored"> _cholesky_building_blocks = [] {
    using namespace boost::ut;

    "a 2x2 real system [[4, 2], [2, 3]] x = (2, 5): factor L = [[2, .], [1, sqrt 2]], x = (-1/2, 2) by hand"_test = [] {
        /** upper triangle is not read */
        std::array<double, 4UZ> A{4.0, 99.0, 2.0, 3.0};
        expect(cholesky_factor(A.data(), 2UZ));
        expect(le(std::abs(A[0UZ] - 2.0), 1e-14));
        expect(le(std::abs(A[2UZ] - 1.0), 1e-14));
        expect(le(std::abs(A[3UZ] - std::sqrt(2.0)), 1e-14));
        expect(eq(A[1UZ], 99.0));
        std::array<double, 2UZ> b{2.0, 5.0};
        cholesky_solve_factored(A.data(), 2UZ, b.data());
        expect(le(std::abs(b[0UZ] + 0.5), 1e-14));
        expect(le(std::abs(b[1UZ] - 2.0), 1e-14));
    };

    "a 2x2 gr::complex<double> Hermitian system [[2, i], [-i, 2]] x = (1, 0): x = (2, 1i) / 3"_test = [] {
        using C = gr::complex<double>;
        std::array<C, 4UZ> A{C{2.0, 0.0}, C{0.0, 1.0}, C{0.0, -1.0}, C{2.0, 0.0}};
        expect(cholesky_factor(A.data(), 2UZ));
        std::array<C, 2UZ> b{C{1.0, 0.0}, C{0.0, 0.0}};
        cholesky_solve_factored(A.data(), 2UZ, b.data());
        expect(le(std::abs(b[0UZ].real() - 2.0 / 3.0) + std::abs(b[0UZ].imag()), 1e-14));
        expect(le(std::abs(b[1UZ].real()) + std::abs(b[1UZ].imag() - 1.0 / 3.0), 1e-14));
    };

    "std::complex<float> and gr::complex<float> solve the same system to float precision"_test = [] {
        const auto solves = []<typename C>() {
            using std::abs;
            std::array<C, 4UZ> A{C{2.f, 0.f}, C{0.f, 1.f}, C{0.f, -1.f}, C{2.f, 0.f}};
            expect(cholesky_factor(A.data(), 2UZ));
            std::array<C, 2UZ> b{C{1.f, 0.f}, C{0.f, 0.f}};
            cholesky_solve_factored(A.data(), 2UZ, b.data());
            expect(le(abs(b[0UZ] - C{2.f / 3.f, 0.f}), 1e-6f));
            expect(le(abs(b[1UZ] - C{0.f, 1.f / 3.f}), 1e-6f));
        };
        solves.template operator()<std::complex<float>>();
        solves.template operator()<gr::complex<float>>();
    };

    "a 3x3 Hermitian positive definite matrix factors as L L^H == A"_test = [] {
        using C = gr::complex<double>;
        /** A = B B^H + I with B = [[1, i, 0], [0, 1, 2], [1, 0, 1i]] */
        const std::array<C, 9UZ> B{C{1, 0}, C{0, 1}, C{0, 0}, C{0, 0}, C{1, 0}, C{2, 0}, C{1, 0}, C{0, 0}, C{0, 1}};
        std::array<C, 9UZ>       A{};
        for (std::size_t i = 0UZ; i < 3UZ; ++i) {
            for (std::size_t j = 0UZ; j < 3UZ; ++j) {
                C sum = i == j ? C{1, 0} : C{0, 0};
                for (std::size_t k = 0UZ; k < 3UZ; ++k) {
                    sum += B[i * 3UZ + k] * conj(B[j * 3UZ + k]);
                }
                A[i * 3UZ + j] = sum;
            }
        }
        std::array<C, 9UZ> L = A;
        expect(cholesky_factor(L.data(), 3UZ));
        for (std::size_t i = 0UZ; i < 3UZ; ++i) {
            for (std::size_t j = 0UZ; j <= i; ++j) {
                C sum{};
                for (std::size_t k = 0UZ; k <= j; ++k) {
                    sum += L[i * 3UZ + k] * conj(L[j * 3UZ + k]);
                }
                expect(le(abs(sum - A[i * 3UZ + j]), 1e-13)) << "entry" << i << j;
            }
        }
    };

    "a matrix that is not positive definite is refused"_test = [] {
        /** eigenvalues 3 and -1 */
        std::array<double, 4UZ> A{1.0, 0.0, 2.0, 1.0};
        expect(!cholesky_factor(A.data(), 2UZ));
        std::array<double, 1UZ> zero{0.0};
        expect(!cholesky_factor(zero.data(), 1UZ));
        std::array<float, 1UZ> notANumber{std::numeric_limits<float>::quiet_NaN()};
        expect(!cholesky_factor(notANumber.data(), 1UZ));
    };
};

int main() { /* not needed for boost-ut */ return 0; }
