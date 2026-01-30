#ifndef GNURADIO_GEMM_SIMD_HPP
#define GNURADIO_GEMM_SIMD_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstring>
#include <execution>
#include <memory>
#include <numeric>
#include <span>
#include <vir/simd.h>

namespace gr::math {

namespace detail {

namespace stdx = vir::stdx;

namespace helper::gemm {
template<typename V, typename S, typename U>
constexpr V fma(const V& a, const S& b, const U& c) noexcept {
    if constexpr (stdx::is_simd_v<V>) {
        using T = typename V::value_type;

        if constexpr (std::is_floating_point_v<T>) { // SIMD FMA for float/double simd<T>
            return stdx::fma<T>(a, b, c);
        } else { // integer SIMD: just do mul+add, compiler may still fuse
            return a * b + c;
        }
    } else { // scalar path
        using CT = std::common_type_t<V, S, V>;

        if constexpr (std::is_floating_point_v<CT>) {
            return static_cast<V>(std::fma(static_cast<CT>(a), static_cast<CT>(b), static_cast<CT>(c)));
        } else {
            return static_cast<V>(static_cast<CT>(a) * static_cast<CT>(b) + static_cast<CT>(c));
        }
    }
}
} // namespace helper::gemm

namespace config {

template<typename T>
struct GemmParams {
    static constexpr std::size_t SIMD_WIDTH = stdx::native_simd<T>::size();

    static constexpr std::size_t KC = 256;  // Inner K dimension - fits in L1
    static constexpr std::size_t MC = 128;  // M blocking for L2
    static constexpr std::size_t NC = 1024; // N can be larger since we stream through it
};

} // namespace config

template<typename T>
struct CacheOptimizedGemm {
    template<typename ATensor, typename BTensor, typename CTensor>
    static void compute(const ATensor& A, const BTensor& B, CTensor& C, T alpha, T beta) {
        using params = config::GemmParams<T>;
        using simd_t = stdx::native_simd<T>;

        const auto C_ext = C.extents();
        const auto A_ext = A.extents();

        const std::size_t M = C_ext[0];
        const std::size_t N = C_ext[1];
        const std::size_t K = A_ext[1];

        if (beta == T{0}) {
            C.fill(T{0});
        } else if (beta != T{1}) {
            for (std::size_t i = 0; i < M; ++i) {
                T* c_row = &C[i, 0];
#if defined(__GLIBCXX__) && !defined(__ACPP__)
                std::transform(std::execution::unseq, c_row, c_row + N, c_row, [beta](T x) { return beta * x; });
#else
                std::transform(c_row, c_row + N, c_row, [beta](T x) { return beta * x; });
#endif
            }
        }

        if (alpha == T{0}) {
            return;
        }

        constexpr std::size_t KC       = params::KC;
        constexpr std::size_t MC       = params::MC;
        constexpr std::size_t VEC_SIZE = stdx::native_simd<T>::size();

        // block over K dimension first (most important for cache reuse)
        for (std::size_t kc = 0UZ; kc < K; kc += KC) {
            const std::size_t k_end = std::min(kc + KC, K);

            // block over M dimension
            for (std::size_t ic = 0UZ; ic < M; ic += MC) {
                const std::size_t m_end = std::min(ic + MC, M);

                // process this M x K block against full N
                for (std::size_t i = ic; i < m_end; ++i) {
                    T* __restrict__ c_row = &C[i, 0];

                    // rank-1 updates for this row
                    for (std::size_t k = kc; k < k_end; ++k) {
                        const T a_ik                = alpha * A[i, k];
                        const T* __restrict__ b_row = &B[k, 0];

                        if (N >= 64) { // large N -> use explicit SIMD with unrolling
                            simd_t      a_vec(a_ik);
                            std::size_t j = 0;

                            for (; j + 4 * VEC_SIZE <= N; j += 4 * VEC_SIZE) {
                                simd_t b0, b1, b2, b3;
                                simd_t c0, c1, c2, c3;

                                b0.copy_from(&b_row[j], stdx::element_aligned);
                                b1.copy_from(&b_row[j + VEC_SIZE], stdx::element_aligned);
                                b2.copy_from(&b_row[j + 2 * VEC_SIZE], stdx::element_aligned);
                                b3.copy_from(&b_row[j + 3 * VEC_SIZE], stdx::element_aligned);

                                c0.copy_from(&c_row[j], stdx::element_aligned);
                                c1.copy_from(&c_row[j + VEC_SIZE], stdx::element_aligned);
                                c2.copy_from(&c_row[j + 2 * VEC_SIZE], stdx::element_aligned);
                                c3.copy_from(&c_row[j + 3 * VEC_SIZE], stdx::element_aligned);

                                c0 = helper::gemm::fma(a_vec, b0, c0);
                                c1 = helper::gemm::fma(a_vec, b1, c1);
                                c2 = helper::gemm::fma(a_vec, b2, c2);
                                c3 = helper::gemm::fma(a_vec, b3, c3);

                                c0.copy_to(&c_row[j], stdx::element_aligned);
                                c1.copy_to(&c_row[j + VEC_SIZE], stdx::element_aligned);
                                c2.copy_to(&c_row[j + 2 * VEC_SIZE], stdx::element_aligned);
                                c3.copy_to(&c_row[j + 3 * VEC_SIZE], stdx::element_aligned);
                            }

                            // handle remainder with single vectors
                            for (; j + VEC_SIZE <= N; j += VEC_SIZE) {
                                simd_t b_vec, c_vec;
                                b_vec.copy_from(&b_row[j], stdx::element_aligned);
                                c_vec.copy_from(&c_row[j], stdx::element_aligned);
                                c_vec = helper::gemm::fma(a_vec, b_vec, c_vec);
                                c_vec.copy_to(&c_row[j], stdx::element_aligned);
                            }

                            // epilog
                            for (; j < N; ++j) {
                                c_row[j] = helper::gemm::fma(a_ik, b_row[j], c_row[j]);
                            }
                        } else { // small N -> auto-vectorise
#if defined(__GLIBCXX__) && !defined(__ACPP__)
                            std::transform(std::execution::unseq, b_row, b_row + N, c_row, c_row, [a_ik](T b, T c) { return helper::gemm::fma(a_ik, b, c); });
#else
                            std::transform(b_row, b_row + N, c_row, c_row, [a_ik](T b, T c) { return helper::gemm::fma(a_ik, b, c); });
#endif
                        }
                    }
                }
            }
        }
    }
};

template<typename T>
struct SmallGemm {
    template<typename ATensor, typename BTensor, typename CTensor>
    static constexpr void compute(const ATensor& A, const BTensor& B, CTensor& C, T alpha, T beta) noexcept {
        const auto C_ext = C.extents();
        const auto A_ext = A.extents();

        const std::size_t M = C_ext[0];
        const std::size_t N = C_ext[1];
        const std::size_t K = A_ext[1];

        if (beta == T{0}) {
            C.fill(T{0});
        } else if (beta != T{1}) {
            for (std::size_t i = 0; i < M; ++i) {
                T* c_row = &C[i, 0];
#if defined(__GLIBCXX__) && !defined(__ACPP__)
                std::transform(std::execution::unseq, c_row, c_row + N, c_row, [beta](T x) { return beta * x; });
#else
                std::transform(c_row, c_row + N, c_row, [beta](T x) { return beta * x; });
#endif
            }
        }

        if (alpha == T{0}) {
            return;
        }

        // naive algorithm
        for (std::size_t i = 0UZ; i < M; ++i) {
            T* c_row = &C[i, 0UZ];

            for (std::size_t k = 0UZ; k < K; ++k) {
                const T  a_ik  = alpha * A[i, k];
                const T* b_row = &B[k, 0UZ];
#if defined(__GLIBCXX__) && !defined(__ACPP__)
                std::transform(std::execution::unseq, b_row, b_row + N, c_row, c_row, [a_ik](T b, T c) { return helper::gemm::fma(a_ik, b, c); });
#else
                std::transform(b_row, b_row + N, c_row, c_row, [a_ik](T b, T c) { return helper::gemm::fma(a_ik, b, c); });
#endif
            }
        }
    }
};

/**
 * @brief General Matrix-Matrix Multiplication: C = alpha * op(A) * op(B) + beta * C
 */
template<TransposeOp TransA = TransposeOp::NoTrans, TransposeOp TransB = TransposeOp::NoTrans, typename ExecutionPolicy, TensorLike TensorC, TensorLike TensorA, TensorLike TensorB, typename T = typename TensorC::value_type>
requires TensorOf<TensorC, T> && TensorOf<TensorA, T> && TensorOf<TensorB, T>
void gemm(ExecutionPolicy&& /*policy*/, TensorC& C, const TensorA& A, const TensorB& B, T alpha = T{1}, T beta = T{0}) {
    if (C.rank() != 2 || A.rank() != 2 || B.rank() != 2) {
        throw std::runtime_error("gemm requires 2D tensors");
    }

    if (!C.is_contiguous() || !A.is_contiguous() || !B.is_contiguous()) {
        throw std::runtime_error("gemm requires contiguous tensors");
    }

    constexpr auto apply_transpose = []<TransposeOp trans, TensorOf<T> Tensor>(const Tensor& tensor) {
        using ConstVal = std::add_const_t<typename tensor_traits<Tensor>::value_type>;

        if constexpr (trans == TransposeOp::NoTrans) {
            return TensorView<ConstVal>(tensor);
        } else {
            auto tmp = tensor.transpose();
            return TensorView<ConstVal>(tmp);
        }
    };

    auto A_op = apply_transpose.template operator()<TransA>(A);
    auto B_op = apply_transpose.template operator()<TransB>(B);

    const auto A_ext = A_op.extents();
    const auto B_ext = B_op.extents();
    const auto C_ext = C.extents();

    if (A_ext[0] != C_ext[0] || B_ext[1] != C_ext[1] || A_ext[1] != B_ext[0]) {
        throw std::runtime_error("gemm: incompatible matrix dimensions");
    }

    const std::size_t M = C_ext[0];
    const std::size_t N = C_ext[1];
    const std::size_t K = A_ext[1];

    // Decision threshold
    constexpr std::size_t CACHE_BLOCK_THRESHOLD = 128UZ;

    // Use SmallGemm for complex types (SIMD doesn't support std::complex) or small matrices
    if constexpr (gr::meta::complex_like<T>) {
        SmallGemm<T>::compute(A_op, B_op, C, alpha, beta); // complex types -> use naive algorithm
    } else if (M <= CACHE_BLOCK_THRESHOLD && N <= CACHE_BLOCK_THRESHOLD && K <= CACHE_BLOCK_THRESHOLD) {
        SmallGemm<T>::compute(A_op, B_op, C, alpha, beta); // small matrices -> use simple algorithm
    } else {
        CacheOptimizedGemm<T>::compute(A_op, B_op, C, alpha, beta); // large matrices -> use cache-blocked algorithm with explicit SIMD
    }
}

} // namespace detail

} // namespace gr::math

#endif // GNURADIO_GEMM_SIMD_HPP
