#ifndef GNURADIO_GEMV_SIMD_HPP
#define GNURADIO_GEMV_SIMD_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>

namespace gr::math::detail {

namespace config {
template<typename T>
struct GemvParams {
    static constexpr std::size_t SIMD_WIDTH    = simd_width<T>;
    static constexpr std::size_t UNROLL_FACTOR = 4;
};
} // namespace config

struct GemvKernel {
    template<typename T, TensorLike ATensor, TensorLike XTensor, TensorLike YTensor>
    static void compute(const ATensor& A, const XTensor& x, YTensor& y, T alpha, T beta) {
        using simd_t = native_simd<T>;
        using params = config::GemvParams<T>;

        const auto        A_ext = A.extents();
        const std::size_t M     = A_ext[0];
        const std::size_t N     = A_ext[1];

        constexpr std::size_t UNROLL = params::UNROLL_FACTOR;

        // beta scaling
        if (beta == T{0}) {
            for (std::size_t i = 0; i < M; ++i) {
                y[i] = T{0};
            }
        } else if (beta != T{1}) {
            for (std::size_t i = 0; i < M; ++i) {
                y[i] *= beta;
            }
        }

        if (alpha == T{0}) {
            return;
        }

        // process each row
        for (std::size_t i = 0; i < M; ++i) {
            // unrolled accumulators
            std::array<simd_t, UNROLL> accum;
            for (auto& acc : accum) {
                acc = simd_t(T{0});
            }

            std::size_t j = 0;

            // unrolled SIMD loop
            const std::size_t simd_unroll_width = UNROLL * simd_width<T>;
            for (; j + simd_unroll_width <= N; j += simd_unroll_width) {
                for (std::size_t u = 0; u < UNROLL; ++u) {
                    std::size_t j_offset = j + u * simd_width<T>;

                    alignas(stdx::memory_alignment_v<simd_t>) T a_buf[simd_width<T>];
                    alignas(stdx::memory_alignment_v<simd_t>) T x_buf[simd_width<T>];

                    for (std::size_t k = 0; k < simd_width<T>; ++k) {
                        a_buf[k] = A[i, j_offset + k];
                        x_buf[k] = x[j_offset + k];
                    }

                    simd_t a_vec(a_buf, stdx::vector_aligned);
                    simd_t x_vec(x_buf, stdx::vector_aligned);

                    accum[u] = stdx::fma(a_vec, x_vec, accum[u]);
                }
            }

            // single SIMD width processing
            for (; j + simd_width<T> <= N; j += simd_width<T>) {
                alignas(stdx::memory_alignment_v<simd_t>) T a_buf[simd_width<T>];
                alignas(stdx::memory_alignment_v<simd_t>) T x_buf[simd_width<T>];

                for (std::size_t k = 0; k < simd_width<T>; ++k) {
                    a_buf[k] = A[i, j + k];
                    x_buf[k] = x[j + k];
                }

                simd_t a_vec(a_buf, stdx::vector_aligned);
                simd_t x_vec(x_buf, stdx::vector_aligned);

                accum[0] = stdx::fma(a_vec, x_vec, accum[0]);
            }

            // reduce all accumulators
            simd_t total_sum = accum[0];
            for (std::size_t u = 1; u < UNROLL; ++u) {
                total_sum = total_sum + accum[u];
            }

            T dot = stdx::reduce(total_sum);

            // epilogue
            for (; j < N; ++j) {
                dot += A[i, j] * x[j];
            }

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

    constexpr auto apply_transpose = []<TransposeOp trans, TensorOf<T> Tensor>(const Tensor& tensor) {
        using ConstVal = std::add_const_t<typename tensor_traits<Tensor>::value_type>;

        if constexpr (trans == TransposeOp::NoTrans) {
            return TensorView<ConstVal>(tensor);
        } else {
            auto tmp = tensor.transpose();
            return TensorView<ConstVal>(tmp);
        }
    };

    auto A_op                                  = apply_transpose.template operator()<TransA>(A);
    const auto                           A_ext = A_op.extents();
    const auto                           x_ext = x.extents();
    const auto                           y_ext = y.extents();

    if (A_ext[0] != y_ext[0] || A_ext[1] != x_ext[0]) {
        throw std::runtime_error("gemv: incompatible dimensions");
    }

    GemvKernel::compute(A_op, x, y, alpha, beta);
}

} // namespace gr::math::detail

#endif // GNURADIO_GEMV_SIMD_HPP
