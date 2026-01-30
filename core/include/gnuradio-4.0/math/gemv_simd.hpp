#ifndef GNURADIO_GEMV_SIMD_HPP
#define GNURADIO_GEMV_SIMD_HPP

#include "gnuradio-4.0/TensorMath.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <vir/simd.h>

namespace gr::math::detail {

namespace stdx = vir::stdx;

namespace helper::gemv {
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
            return static_cast<V>(helper::gemv::fma(static_cast<CT>(a), static_cast<CT>(b), static_cast<CT>(c)));
        } else {
            return static_cast<V>(static_cast<CT>(a) * static_cast<CT>(b) + static_cast<CT>(c));
        }
    }
}
} // namespace helper::gemv

template<typename T>
struct GemvNoTrans { // GEMV for non-transposed A: y = alpha * A * x + beta * y
    template<typename ATensor, typename XTensor, typename YTensor>
    static void compute(const ATensor& A, const XTensor& x, YTensor& y, T alpha, T beta) {
        using simd_t                   = stdx::native_simd<T>;
        constexpr std::size_t VEC_SIZE = simd_t::size();

        const auto        A_ext = A.extents();
        const std::size_t M     = A_ext[0];
        const std::size_t N     = A_ext[1];

        if (beta == T{0}) {
            std::fill_n(&y[0], M, T{0});
        } else if (beta != T{1}) {
            for (std::size_t i = 0; i < M; ++i) {
                y[i] *= beta;
            }
        }

        if (alpha == T{0}) {
            return;
        }

        // process each row of A i.e. dot product with x
        for (std::size_t i = 0; i < M; ++i) {
            const T* a_row = &A[i, 0];

            if (N >= 32) { // Use SIMD for larger vectors
                simd_t acc0(T{0});
                simd_t acc1(T{0});
                simd_t acc2(T{0});
                simd_t acc3(T{0});

                std::size_t j = 0;

                // manually unrolled SIMD loop
                for (; j + 4 * VEC_SIZE <= N; j += 4 * VEC_SIZE) {
                    simd_t a0, a1, a2, a3;
                    simd_t x0, x1, x2, x3;

                    a0.copy_from(&a_row[j], stdx::element_aligned);
                    a1.copy_from(&a_row[j + VEC_SIZE], stdx::element_aligned);
                    a2.copy_from(&a_row[j + 2 * VEC_SIZE], stdx::element_aligned);
                    a3.copy_from(&a_row[j + 3 * VEC_SIZE], stdx::element_aligned);

                    x0.copy_from(&x[j], stdx::element_aligned);
                    x1.copy_from(&x[j + VEC_SIZE], stdx::element_aligned);
                    x2.copy_from(&x[j + 2 * VEC_SIZE], stdx::element_aligned);
                    x3.copy_from(&x[j + 3 * VEC_SIZE], stdx::element_aligned);

                    if constexpr (std::is_floating_point_v<T>) {
                        acc0 = helper::gemv::fma(a0, x0, acc0);
                        acc1 = helper::gemv::fma(a1, x1, acc1);
                        acc2 = helper::gemv::fma(a2, x2, acc2);
                        acc3 = helper::gemv::fma(a3, x3, acc3);
                    } else {
                        acc0 = acc0 + a0 * x0;
                        acc1 = acc1 + a1 * x1;
                        acc2 = acc2 + a2 * x2;
                        acc3 = acc3 + a3 * x3;
                    }
                }

                for (; j + VEC_SIZE <= N; j += VEC_SIZE) {
                    simd_t a_vec, x_vec;
                    a_vec.copy_from(&a_row[j], stdx::element_aligned);
                    x_vec.copy_from(&x[j], stdx::element_aligned);

                    if constexpr (std::is_floating_point_v<T>) {
                        acc0 = helper::gemv::fma(a_vec, x_vec, acc0);
                    } else {
                        acc0 = acc0 + a_vec * x_vec;
                    }
                }

                simd_t total = acc0 + acc1 + acc2 + acc3;
                T      dot   = stdx::reduce(total);

                for (; j < N; ++j) {
                    dot += a_row[j] * x[j];
                }

                y[i] += alpha * dot;
            } else { // small N -> use simple loop
                T dot = T{0};
                for (std::size_t j = 0; j < N; ++j) {
                    dot += a_row[j] * x[j];
                }
                y[i] += alpha * dot;
            }
        }
    }
};

template<typename T>
struct GemvTrans { // GEMV for transposed A: y = alpha * A^T * x + beta * y
    template<typename ATensor, typename XTensor, typename YTensor>
    static void compute(const ATensor& A, const XTensor& x, YTensor& y, T alpha, T beta) {
        using simd_t                   = stdx::native_simd<T>;
        constexpr std::size_t VEC_SIZE = simd_t::size();

        // N.B. For transposed view, A still has original dimensions but we're computing y = A^T * x, so output size is A.extents()[1]
        const auto        A_ext = A.extents();
        const std::size_t N     = A_ext[0]; // Input size (rows of original A)
        const std::size_t M     = A_ext[1]; // Output size (cols of original A)

        // Handle beta scaling
        if (beta == T{0}) {
            y.fill(T{0});
        } else if (beta != T{1}) {
            TensorOps<T>::multiply_scalar_inplace(y, beta);
        }

        if (alpha == T{0}) {
            return;
        }

        // compute y += alpha * A^T * x
        // for each column j of A: add x[i] * A[i,j] to y[j]
        for (std::size_t i = 0; i < N; ++i) {
            const T x_i = alpha * x[i];
            if (x_i == T{0}) {
                continue;
            }

            const T* a_row = &A[i, 0]; // row i of original A = column i of A^T

            if (M >= 32) { // use SIMD for larger output vectors
                const simd_t x_broadcast(x_i);
                std::size_t  j = 0;

                // 4-way unrolled SIMD loop
                for (; j + 4 * VEC_SIZE <= M; j += 4 * VEC_SIZE) {
                    simd_t a0, a1, a2, a3;
                    simd_t y0, y1, y2, y3;

                    // load A row values
                    a0.copy_from(&a_row[j], stdx::element_aligned);
                    a1.copy_from(&a_row[j + VEC_SIZE], stdx::element_aligned);
                    a2.copy_from(&a_row[j + 2 * VEC_SIZE], stdx::element_aligned);
                    a3.copy_from(&a_row[j + 3 * VEC_SIZE], stdx::element_aligned);

                    // load current y values
                    y0.copy_from(&y[j], stdx::element_aligned);
                    y1.copy_from(&y[j + VEC_SIZE], stdx::element_aligned);
                    y2.copy_from(&y[j + 2 * VEC_SIZE], stdx::element_aligned);
                    y3.copy_from(&y[j + 3 * VEC_SIZE], stdx::element_aligned);

                    // FMA operations: y += x[i] * A[i,:]
                    if constexpr (std::is_floating_point_v<T>) {
                        y0 = helper::gemv::fma(x_broadcast, a0, y0);
                        y1 = helper::gemv::fma(x_broadcast, a1, y1);
                        y2 = helper::gemv::fma(x_broadcast, a2, y2);
                        y3 = helper::gemv::fma(x_broadcast, a3, y3);
                    } else {
                        y0 = y0 + x_broadcast * a0;
                        y1 = y1 + x_broadcast * a1;
                        y2 = y2 + x_broadcast * a2;
                        y3 = y3 + x_broadcast * a3;
                    }

                    // Store back
                    y0.copy_to(&y[j], stdx::element_aligned);
                    y1.copy_to(&y[j + VEC_SIZE], stdx::element_aligned);
                    y2.copy_to(&y[j + 2 * VEC_SIZE], stdx::element_aligned);
                    y3.copy_to(&y[j + 3 * VEC_SIZE], stdx::element_aligned);
                }

                // Handle remaining full vectors
                for (; j + VEC_SIZE <= M; j += VEC_SIZE) {
                    simd_t a_vec, y_vec;
                    a_vec.copy_from(&a_row[j], stdx::element_aligned);
                    y_vec.copy_from(&y[j], stdx::element_aligned);

                    if constexpr (std::is_floating_point_v<T>) {
                        y_vec = helper::gemv::fma(x_broadcast, a_vec, y_vec);
                    } else {
                        y_vec = y_vec + x_broadcast * a_vec;
                    }

                    y_vec.copy_to(&y[j], stdx::element_aligned);
                }

                // epilogue
                for (; j < M; ++j) {
                    y[j] += x_i * a_row[j];
                }
            } else { // small M -> use simple loop
                for (std::size_t j = 0; j < M; ++j) {
                    y[j] += x_i * a_row[j];
                }
            }
        }
    }
};

// Simple GEMV for small vectors using auto-vectorization
template<typename T>
struct SimpleGemv {
    template<typename ATensor, typename XTensor, typename YTensor>
    static void compute(const ATensor& A, const XTensor& x, YTensor& y, T alpha, T beta) {
        const auto        A_ext = A.extents();
        const std::size_t M     = A_ext[0];
        const std::size_t N     = A_ext[1];

        // Handle beta
        if (beta == T{0}) {
            std::fill_n(&y[0], M, T{0});
        } else if (beta != T{1}) {
            for (std::size_t i = 0; i < M; ++i) {
                y[i] *= beta;
            }
        }

        if (alpha == T{0}) {
            return;
        }

        // simple dot product approach - let compiler auto-vectorize
        for (std::size_t i = 0; i < M; ++i) {
            const T* a_row = &A[i, 0];

#if defined(__GLIBCXX__) && !defined(__ACPP__)
            T dot = std::transform_reduce(std::execution::unseq, a_row, a_row + N, &x[0], T{0}, std::plus<>{}, std::multiplies<>{});
#else
            T dot = std::transform_reduce(a_row, a_row + N, &x[0], T{0}, std::plus<>{}, std::multiplies<>{});
#endif

            y[i] += alpha * dot;
        }
    }
};

template<TransposeOp TransA = TransposeOp::NoTrans, typename ExecutionPolicy, TensorLike TensorY, TensorLike TensorA, TensorLike TensorX, typename T = typename TensorY::value_type>
requires TensorOf<TensorY, T> && TensorOf<TensorA, T> && TensorOf<TensorX, T>
void gemv(ExecutionPolicy&& /*policy*/, TensorY& y, const TensorA& A, const TensorX& x, T alpha = T{1}, T beta = T{0}) {
    if (A.rank() != 2) {
        throw std::runtime_error("gemv: A must be 2D");
    }
    if (x.rank() != 1 || y.rank() != 1) {
        throw std::runtime_error("gemv: x and y must be 1D");
    }

    // check dimensions and dispatch based on transpose flag
    if constexpr (TransA == TransposeOp::NoTrans) {
        const auto A_ext = A.extents();
        const auto x_ext = x.extents();
        const auto y_ext = y.extents();

        if (A_ext[0] != y_ext[0] || A_ext[1] != x_ext[0]) {
            throw std::runtime_error("gemv: incompatible dimensions for y = A*x");
        }

        const std::size_t N = A_ext[1];

        if (N <= 64) {
            SimpleGemv<T>::compute(A, x, y, alpha, beta);
        } else {
            GemvNoTrans<T>::compute(A, x, y, alpha, beta);
        }
    } else { // transposed case: A is a transposed view
        // The view still reports original dimensions, but accesses are transposed
        const auto A_ext = A.extents();
        const auto x_ext = x.extents();
        const auto y_ext = y.extents();

        // for A^T * x: input size is A_ext[0], output size is A_ext[1]
        if (A_ext[1] != y_ext[0] || A_ext[0] != x_ext[0]) {
            throw std::runtime_error("gemv: incompatible dimensions for y = A^T*x");
        }

        GemvTrans<T>::compute(A, x, y, alpha, beta);
    }
}

} // namespace gr::math::detail

#endif // GNURADIO_GEMV_SIMD_HPP
