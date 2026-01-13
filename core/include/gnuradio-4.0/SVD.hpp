#ifndef GNURADIO_SVD_HPP
#define GNURADIO_SVD_HPP

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TensorMath.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>

namespace gr::math {

namespace svd {
enum class Status : std::uint8_t {
    Success,
    EarlyReturn,   /// truncated to k singular values
    MaxIterations, /// did not converge within iteration limit
    InvalidInput,
    NumericalError
};

enum class Algorithm : std::uint8_t {
    Jacobi,       /// one-sided Jacobi iteration (robust, parallelisable, best for tall matrices)
    GolubReinsch, /// Householder bidiagonalisation + implicit QR (faster for large matrices)
    Auto          /// automatically select based on matrix characteristics
};

template<typename T>
struct Config {
    std::size_t maxIterations = 0;               /// 0 = auto (30 * max(m,n))
    T           tolerance     = T{-1};           /// <0 = auto (eps * max(m,n))
    std::size_t k             = 0;               /// 0 = all singular values, >0 = first k only
    Algorithm   algorithm     = Algorithm::Auto; /// defaults to numerically safe and then efficient choice
    bool        fullMatrices  = false;           /// false = thin SVD, true = full U and V
};
} // namespace svd

namespace detail {

template<TensorLike TensorA>
auto frobeniusNorm(const TensorA& A) {
    using T             = typename TensorA::value_type;
    using BaseValueType = gr::meta::fundamental_base_value_type_t<T>;
    return std::sqrt(std::transform_reduce(A.begin(), A.end(), BaseValueType{0}, std::plus<>{}, [](const T& val) { return squaredMagnitude(val); }));
}

template<TensorLike TensorX>
auto norm2(const TensorX& x) {
    using T             = typename TensorX::value_type;
    using BaseValueType = gr::meta::fundamental_base_value_type_t<T>;
    if (x.rank() != 1) {
        throw std::runtime_error("norm2: requires 1D tensor");
    }
    return std::sqrt(std::transform_reduce(x.begin(), x.end(), BaseValueType{0}, std::plus<>{}, [](const T& val) { return squaredMagnitude(val); }));
}

template<typename T>
constexpr auto sign(T val) {
    if constexpr (gr::meta::complex_like<T>) {
        using BaseValueType = gr::meta::fundamental_base_value_type_t<T>;
        auto mag            = std::abs(val);
        return (mag > BaseValueType{0}) ? (val / mag) : T{1};
    } else {
        return (val > T{0}) ? T{1} : ((val < T{0}) ? T{-1} : T{0});
    }
}

template<typename T>
constexpr T innerProduct(const T& a, const T& b) {
    if constexpr (gr::meta::complex_like<T>) {
        return std::conj(a) * b;
    } else {
        return a * b;
    }
}

template<typename T>
constexpr T conj(const T& val) {
    if constexpr (gr::meta::complex_like<T>) {
        return std::conj(val);
    } else {
        return val;
    }
}

// one-sided Jacobi SVD: works best when m >= n (tall matrices)
// A is modified in-place to become U, V and singularValues are outputs
template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
svd::Status svdJacobi(Tensor<T>& A, Tensor<T>* V, Tensor<BaseValueType>& singularValues, std::size_t maxIter, BaseValueType tol, bool computeV) {
    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);

    if (n == 0 || m == 0) {
        if (computeV && V) {
            V->resize({n, 0});
        }
        singularValues.resize({0});
        return svd::Status::Success;
    }

    // initialise V as identity if needed
    if (computeV && V) {
        V->resize({n, n});
        V->fill(T{0});
        for (std::size_t i = 0; i < n; ++i) {
            (*V)[i, i] = T{1};
        }
    }

    std::size_t iter      = 0;
    bool        converged = false;

    // main Jacobi iteration
    while (iter < maxIter && !converged) {
        converged = true;

        for (std::size_t i = 0; i < n - 1; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                // compute a_i^H * a_j and ||a_i||^2, ||a_j||^2
                T             dot_ij{0};
                BaseValueType dot_ii{0};
                BaseValueType dot_jj{0};

                for (std::size_t k = 0; k < m; ++k) {
                    dot_ij += innerProduct(A[k, i], A[k, j]);
                    dot_ii += squaredMagnitude(A[k, i]);
                    dot_jj += squaredMagnitude(A[k, j]);
                }

                // check if columns are orthogonal enough
                constexpr BaseValueType epsilon   = std::numeric_limits<BaseValueType>::epsilon() * BaseValueType{1000};
                BaseValueType           threshold = tol * std::sqrt(std::max(dot_ii, epsilon) * std::max(dot_jj, epsilon));

                if (std::abs(dot_ij) > threshold) {
                    converged = false;

                    // compute Jacobi rotation to orthogonalise columns i and j
                    BaseValueType c;
                    T             s;

                    if constexpr (gr::meta::complex_like<T>) {
                        // complex case: use complex Givens rotation
                        BaseValueType absOff = std::abs(dot_ij);
                        if (std::abs(dot_jj - dot_ii) < epsilon * (dot_ii + dot_jj)) {
                            c = BaseValueType{1} / std::sqrt(BaseValueType{2});
                            s = (dot_ij / absOff) * c;
                        } else {
                            BaseValueType zeta = (dot_jj - dot_ii) / (BaseValueType{2} * absOff);
                            BaseValueType t    = BaseValueType{1} / (std::abs(zeta) + std::sqrt(BaseValueType{1} + zeta * zeta));
                            if (zeta < BaseValueType{0}) {
                                t = -t;
                            }
                            c = BaseValueType{1} / std::sqrt(BaseValueType{1} + t * t);
                            s = (dot_ij / absOff) * c * t;
                        }
                    } else {
                        // real case
                        if (std::abs(dot_jj - dot_ii) < epsilon * (dot_ii + dot_jj)) {
                            BaseValueType sqrt2 = std::sqrt(BaseValueType{2});
                            c                   = BaseValueType{1} / sqrt2;
                            s                   = (dot_ij > BaseValueType{0}) ? (BaseValueType{1} / sqrt2) : (BaseValueType{-1} / sqrt2);
                        } else {
                            BaseValueType zeta = (dot_jj - dot_ii) / (BaseValueType{2} * dot_ij);
                            BaseValueType t    = sign(zeta) / (std::abs(zeta) + std::sqrt(BaseValueType{1} + zeta * zeta));
                            c                  = BaseValueType{1} / std::sqrt(BaseValueType{1} + t * t);
                            s                  = c * t;
                        }
                    }

                    // apply rotation to A: columns i and j
                    for (std::size_t k = 0; k < m; ++k) {
                        T temp_i = c * A[k, i] - conj(s) * A[k, j];
                        T temp_j = s * A[k, i] + c * A[k, j];
                        A[k, i]  = temp_i;
                        A[k, j]  = temp_j;
                    }

                    // apply rotation to V if computing it
                    if (computeV && V) {
                        for (std::size_t k = 0; k < n; ++k) {
                            T temp_i   = c * (*V)[k, i] - conj(s) * (*V)[k, j];
                            T temp_j   = s * (*V)[k, i] + c * (*V)[k, j];
                            (*V)[k, i] = temp_i;
                            (*V)[k, j] = temp_j;
                        }
                    }
                }
            }
        }

        ++iter;
    }

    // extract singular values and normalise columns of A to get U
    singularValues.resize({n});
    for (std::size_t j = 0; j < n; ++j) {
        BaseValueType norm{0};
        for (std::size_t i = 0; i < m; ++i) {
            norm += squaredMagnitude(A[i, j]);
        }
        singularValues[j] = std::sqrt(norm);
    }

    // normalise columns
    BaseValueType epsilon = std::numeric_limits<BaseValueType>::epsilon();
    for (std::size_t j = 0; j < n; ++j) {
        if (singularValues[j] > epsilon) {
            for (std::size_t i = 0; i < m; ++i) {
                A[i, j] /= singularValues[j];
            }
        } else {
            singularValues[j] = BaseValueType{0};
        }
    }

    return converged ? svd::Status::Success : svd::Status::MaxIterations;
}

// ============================================================================
// Golub-Reinsch SVD: Householder bidiagonalisation + implicit QR iteration
// ============================================================================

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
BaseValueType householderVector(T* x, std::size_t n, BaseValueType& tau) {
    // compute Householder vector v and scalar tau such that:
    // H = I - tau * v * v^H applies H * x = [beta; 0; 0; ...]
    // x is overwritten with v (v[0] = 1 is implicit)
    // returns beta (the resulting first element after reflection)

    if (n == 0) {
        tau = BaseValueType{0};
        return BaseValueType{0};
    }
    if (n == 1) {
        tau = BaseValueType{0};
        if constexpr (gr::meta::complex_like<T>) {
            return std::abs(x[0]);
        } else {
            return static_cast<BaseValueType>(x[0]);
        }
    }

    // compute ||x[1:n]||^2
    BaseValueType xnorm_sq{0};
    for (std::size_t i = 1; i < n; ++i) {
        xnorm_sq += squaredMagnitude(x[i]);
    }

    if (xnorm_sq == BaseValueType{0}) {
        // x is already [alpha, 0, 0, ...], no reflection needed
        tau = BaseValueType{0};
        if constexpr (gr::meta::complex_like<T>) {
            return std::abs(x[0]);
        } else {
            return static_cast<BaseValueType>(x[0]);
        }
    }

    // compute norm of entire vector: ||x|| = sqrt(|x[0]|^2 + xnorm_sq)
    BaseValueType x0_mag_sq = squaredMagnitude(x[0]);
    BaseValueType xnorm     = std::sqrt(x0_mag_sq + xnorm_sq);

    // choose sign to avoid cancellation: beta = -sign(x[0]) * ||x||
    T beta;
    if constexpr (gr::meta::complex_like<T>) {
        // for complex: use phase of x[0]
        if (std::abs(x[0]) > BaseValueType{0}) {
            T phase = x[0] / std::abs(x[0]);
            beta    = -phase * xnorm;
        } else {
            beta = T{-xnorm, 0};
        }
    } else {
        beta = (x[0] >= BaseValueType{0}) ? -xnorm : xnorm;
    }

    // v[0] = x[0] - beta, v[1:n] = x[1:n]
    T             v0        = x[0] - beta;
    BaseValueType v0_mag_sq = squaredMagnitude(v0);
    BaseValueType vnorm_sq  = v0_mag_sq + xnorm_sq;

    // for normalized vector v' = [1, x[1]/v0, ...]:
    // H = I - tau * v * v^H = I - tau' * v' * v'^H where tau' = tau * |v0|^2
    // tau = 2 / ||v||^2, so tau' = 2 * |v0|^2 / ||v||^2
    tau = BaseValueType{2} * v0_mag_sq / vnorm_sq;

    // store normalized tail: x[1:n] /= v0
    // x[0] stores v0 for back-accumulation (but we use implicit v[0]=1 in transformations)
    x[0] = v0;
    if (std::abs(v0) > std::numeric_limits<BaseValueType>::epsilon()) {
        T scale = T{1} / v0;
        for (std::size_t i = 1; i < n; ++i) {
            x[i] *= scale;
        }
    }

    // for complex: return |beta| (the bidiagonal will be real)
    // for real: return beta with sign (handle signs during SVD)
    if constexpr (gr::meta::complex_like<T>) {
        return std::abs(beta);
    } else {
        return beta;
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void applyHouseholderLeft(Tensor<T>& A, const T* vTail, BaseValueType tau, std::size_t rowStart, std::size_t colStart, std::size_t tailLen) {
    // performs: A[rowStart:, colStart:] = H * A[rowStart:, colStart:]
    // where H = I - tau * v * v^H
    // v has implicit first element v[0] = 1, followed by tailLen elements in vTail

    if (tau == BaseValueType{0}) {
        return;
    }

    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);

    // for each column j >= colStart:
    // A[:, j] -= tau * v * (v^H * A[:, j])
    for (std::size_t j = colStart; j < n; ++j) {
        // compute dot = v^H * A[rowStart:, j]
        T dot = A[rowStart, j]; // v[0] = 1 (implicit)
        for (std::size_t i = 0; i < tailLen && (rowStart + 1 + i) < m; ++i) {
            dot += conj(vTail[i]) * A[rowStart + 1 + i, j];
        }

        // A[rowStart:, j] -= tau * dot * v
        T scale = tau * dot;
        A[rowStart, j] -= scale; // v[0] = 1
        for (std::size_t i = 0; i < tailLen && (rowStart + 1 + i) < m; ++i) {
            A[rowStart + 1 + i, j] -= scale * vTail[i];
        }
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void applyHouseholderRight(Tensor<T>& A, const T* vTail, BaseValueType tau, std::size_t rowStart, std::size_t rowEnd, std::size_t colStart, std::size_t tailLen) {
    // performs: A[rowStart:rowEnd, colStart:] = A[rowStart:rowEnd, colStart:] * H
    // where H = I - tau * v * v^H
    // v has implicit first element v[0] = 1, followed by tailLen elements in vTail

    if (tau == BaseValueType{0}) {
        return;
    }

    const std::size_t n = A.extent(1);

    // for each row i in [rowStart, rowEnd):
    // A[i, :] -= tau * (A[i, :] * v) * v^H
    for (std::size_t i = rowStart; i < rowEnd; ++i) {
        // compute dot = A[i, colStart:] * v
        T dot = A[i, colStart]; // v[0] = 1 (implicit)
        for (std::size_t j = 0; j < tailLen && (colStart + 1 + j) < n; ++j) {
            dot += A[i, colStart + 1 + j] * vTail[j];
        }

        // A[i, colStart:] -= tau * dot * v^H
        T scale = tau * dot;
        A[i, colStart] -= scale; // v[0] = 1
        for (std::size_t j = 0; j < tailLen && (colStart + 1 + j) < n; ++j) {
            A[i, colStart + 1 + j] -= scale * conj(vTail[j]);
        }
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void bidiagonalise(Tensor<T>& A, std::vector<BaseValueType>& tauU, std::vector<BaseValueType>& tauV, std::vector<BaseValueType>& diag, std::vector<BaseValueType>& superdiag) {
    // bidiagonalise A in-place: A -> U^H * A * V = B (upper bidiagonal)
    // stores Householder vectors in-place, tauU and tauV store the tau values
    // diag and superdiag store the real bidiagonal elements

    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    tauU.resize(minDim);
    tauV.resize(minDim > 1 ? minDim - 1 : 0);
    diag.resize(minDim);
    superdiag.resize(minDim > 1 ? minDim - 1 : 0);

    std::vector<T> work(std::max(m, n));

    for (std::size_t k = 0; k < minDim; ++k) {
        // left Householder: eliminate A[k+1:m, k]
        std::size_t colLen = m - k;
        for (std::size_t i = 0; i < colLen; ++i) {
            work[i] = A[k + i, k];
        }

        BaseValueType tauL;
        BaseValueType beta = householderVector(work.data(), colLen, tauL);
        tauU[k]            = tauL;
        diag[k]            = beta;

        // store Householder vector back (skip first element, it's implicit 1)
        A[k, k] = work[0]; // store v0 for back-accumulation
        for (std::size_t i = 1; i < colLen; ++i) {
            A[k + i, k] = work[i];
        }

        // apply H_L to remaining columns A[k:m, k+1:n]
        if (k + 1 < n && colLen > 1) {
            applyHouseholderLeft(A, work.data() + 1, tauL, k, k + 1, colLen - 1);
        }

        // right Householder: eliminate A[k, k+2:n]
        if (k + 1 < n) {
            std::size_t rowLen = n - k - 1;
            for (std::size_t j = 0; j < rowLen; ++j) {
                work[j] = A[k, k + 1 + j];
            }

            BaseValueType tauR;
            BaseValueType gamma = householderVector(work.data(), rowLen, tauR);
            if (k < tauV.size()) {
                tauV[k]      = tauR;
                superdiag[k] = gamma;
            }

            // store Householder vector back
            A[k, k + 1] = work[0]; // store v0
            for (std::size_t j = 1; j < rowLen; ++j) {
                A[k, k + 1 + j] = work[j];
            }

            // apply H_R to remaining rows A[k+1:m, k+1:n]
            if (k + 1 < m && rowLen > 1) {
                applyHouseholderRight(A, work.data() + 1, tauR, k + 1, m, k + 1, rowLen - 1);
            }
        }
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void givensRotation(BaseValueType a, BaseValueType b, BaseValueType& c, BaseValueType& s, BaseValueType& r) {
    // compute Givens rotation: [c, s] such that [c, -conj(s); s, c] * [a; b] = [r; 0]

    if (b == BaseValueType{0}) {
        c = (a >= BaseValueType{0}) ? BaseValueType{1} : BaseValueType{-1};
        s = BaseValueType{0};
        r = std::abs(a);
    } else if (a == BaseValueType{0}) {
        c = BaseValueType{0};
        s = (b >= BaseValueType{0}) ? BaseValueType{1} : BaseValueType{-1};
        r = std::abs(b);
    } else if (std::abs(b) > std::abs(a)) {
        BaseValueType t = a / b;
        BaseValueType u = std::copysign(std::sqrt(BaseValueType{1} + t * t), b);
        s               = BaseValueType{1} / u;
        c               = s * t;
        r               = b * u;
    } else {
        BaseValueType t = b / a;
        BaseValueType u = std::copysign(std::sqrt(BaseValueType{1} + t * t), a);
        c               = BaseValueType{1} / u;
        s               = c * t;
        r               = a * u;
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void applyGivensLeft(Tensor<T>& A, std::size_t i, std::size_t k, BaseValueType c, BaseValueType s, std::size_t colStart, std::size_t colEnd) {
    // apply Givens rotation to rows i and k of columns [colStart, colEnd)

    for (std::size_t j = colStart; j < colEnd; ++j) {
        T temp  = c * A[i, j] + s * A[k, j];
        A[k, j] = -s * A[i, j] + c * A[k, j];
        A[i, j] = temp;
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void applyGivensRight(Tensor<T>& A, std::size_t i, std::size_t k, BaseValueType c, BaseValueType s, std::size_t rowStart, std::size_t rowEnd) {
    // apply Givens rotation to columns i and k of rows [rowStart, rowEnd)

    for (std::size_t j = rowStart; j < rowEnd; ++j) {
        T temp  = c * A[j, i] + s * A[j, k];
        A[j, k] = -s * A[j, i] + c * A[j, k];
        A[j, i] = temp;
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
bool bidiagonalQRStep(std::vector<BaseValueType>& diag, std::vector<BaseValueType>& superdiag, Tensor<T>* U, Tensor<T>* V, std::size_t start, std::size_t end, BaseValueType tol) {
    // one implicit QR step on bidiagonal matrix B with Wilkinson shift
    // B is stored as diag (length n) and superdiag (length n-1)
    // returns true if converged (superdiag[end-1] became negligible)

    if (end <= start + 1) {
        return true;
    }

    // Wilkinson shift: eigenvalue of trailing 2x2 of B^T * B closest to last diagonal
    BaseValueType d_nm1 = diag[end - 1];
    BaseValueType d_nm2 = (end >= 2) ? diag[end - 2] : BaseValueType{0};
    BaseValueType e_nm2 = (end >= 2 && start < end - 1) ? superdiag[end - 2] : BaseValueType{0};

    // elements of B^T * B trailing 2x2
    BaseValueType a11 = d_nm2 * d_nm2 + ((end >= 3 && start < end - 2) ? superdiag[end - 3] * superdiag[end - 3] : BaseValueType{0});
    BaseValueType a12 = d_nm2 * e_nm2;
    BaseValueType a22 = d_nm1 * d_nm1 + e_nm2 * e_nm2;

    // Wilkinson shift: eigenvalue of [[a11, a12], [a12, a22]] closest to a22
    BaseValueType delta = (a11 - a22) / BaseValueType{2};
    BaseValueType shift;
    if (delta == BaseValueType{0}) {
        shift = a22 - std::abs(a12);
    } else {
        BaseValueType sign_delta = (delta > BaseValueType{0}) ? BaseValueType{1} : BaseValueType{-1};
        shift                    = a22 - a12 * a12 / (delta + sign_delta * std::sqrt(delta * delta + a12 * a12));
    }

    // initial Givens to introduce bulge
    BaseValueType x = diag[start] * diag[start] - shift;
    BaseValueType z = diag[start] * superdiag[start];

    // chase bulge down the bidiagonal
    for (std::size_t k = start; k < end - 1; ++k) {
        // Right Givens: zero out z using rotation on columns k and k+1
        BaseValueType c, s, r;
        givensRotation<T>(x, z, c, s, r);

        // For k > start, the bulge from previous iteration is at (k-1, k)
        // The right Givens zeros it: superdiag[k-1] = r
        if (k > start) {
            superdiag[k - 1] = r;
        }

        // Apply right Givens to bidiagonal: B := B * G^T
        // G^T = [[c, -s], [s, c]] applied to columns k, k+1
        BaseValueType d_k  = diag[k];
        BaseValueType e_k  = superdiag[k];
        BaseValueType d_k1 = diag[k + 1];

        // New values after B * G^T:
        // B[k,k] = c * d_k + s * e_k
        // B[k,k+1] = -s * d_k + c * e_k
        // B[k+1,k] = s * d_k1  (bulge created!)
        // B[k+1,k+1] = c * d_k1
        diag[k]             = c * d_k + s * e_k;
        superdiag[k]        = -s * d_k + c * e_k;
        BaseValueType bulge = s * d_k1; // the bulge at position (k+1, k)
        diag[k + 1]         = c * d_k1;

        // accumulate V: V := V * G^T
        if (V) {
            applyGivensRight(*V, k, k + 1, c, s, 0, V->extent(0));
        }

        // Left Givens: zero out bulge at (k+1, k)
        // Rotate rows k and k+1 to zero bulge
        givensRotation<T>(diag[k], bulge, c, s, r);

        // Apply left Givens to bidiagonal: B := G * B
        // G = [[c, s], [-s, c]] applied to rows k, k+1
        diag[k] = r;

        // Row k: [r, superdiag[k], 0, ...]
        // Row k+1: [0, diag[k+1], superdiag[k+1], ...]
        // After G * B:
        // New B[k, k+1] = c * superdiag[k] + s * diag[k+1]
        // New B[k+1, k+1] = -s * superdiag[k] + c * diag[k+1]
        // New B[k, k+2] = s * superdiag[k+1]  (new bulge if k+2 < end!)
        // New B[k+1, k+2] = c * superdiag[k+1]
        BaseValueType old_e  = superdiag[k];
        BaseValueType old_d1 = diag[k + 1];

        superdiag[k] = c * old_e + s * old_d1;
        diag[k + 1]  = -s * old_e + c * old_d1;

        // Set up for next iteration
        if (k + 2 < end) {
            // New bulge appears at (k, k+2) from left Givens
            z                = s * superdiag[k + 1];
            superdiag[k + 1] = c * superdiag[k + 1];
            x                = superdiag[k]; // for next right Givens
        }

        // accumulate U: U := U * G
        if (U) {
            applyGivensRight(*U, k, k + 1, c, s, 0, U->extent(0));
        }
    }

    // check convergence: is superdiag[end-2] negligible?
    BaseValueType off    = std::abs(superdiag[end - 2]);
    BaseValueType thresh = tol * (std::abs(diag[end - 2]) + std::abs(diag[end - 1]));
    return off <= thresh;
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void accumulateU(const Tensor<T>& A, const std::vector<BaseValueType>& tauU, Tensor<T>& U) {
    // accumulate U from stored Householder vectors (left transformations)

    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    // initialize U as identity
    U.resize({m, minDim});
    U.fill(T{0});
    for (std::size_t i = 0; i < minDim; ++i) {
        U[i, i] = T{1};
    }

    // apply Householder transformations in reverse order
    std::vector<T> v(m);
    for (std::size_t k = minDim; k-- > 0;) {
        std::size_t vLen = m - k;

        // reconstruct Householder vector
        v[0] = T{1}; // implicit
        for (std::size_t i = 1; i < vLen; ++i) {
            v[i] = A[k + i, k];
        }

        // U = H_k * U = (I - tau * v * v^H) * U
        // for each column j of U
        for (std::size_t j = k; j < minDim; ++j) {
            T dot{0};
            for (std::size_t i = 0; i < vLen; ++i) {
                dot += conj(v[i]) * U[k + i, j];
            }
            T scale = tauU[k] * dot;
            for (std::size_t i = 0; i < vLen; ++i) {
                U[k + i, j] -= scale * v[i];
            }
        }
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
void accumulateV(const Tensor<T>& A, const std::vector<BaseValueType>& tauV, Tensor<T>& V) {
    // accumulate V from stored Householder vectors (right transformations)

    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    // initialize V as identity
    V.resize({n, minDim});
    V.fill(T{0});
    for (std::size_t i = 0; i < minDim; ++i) {
        V[i, i] = T{1};
    }

    if (minDim <= 1) {
        return;
    }

    // apply Householder transformations in forward order (H_0, H_1, ..., H_{n-2})
    // bidiagonalise applies: A := A * H_0 * H_1 * ... * H_{n-2}, so V = H_0 * H_1 * ... * H_{n-2}
    std::vector<T> v(n);
    for (std::size_t k = 0; k < tauV.size(); ++k) {
        std::size_t vLen = n - k - 1;
        if (vLen == 0) {
            continue;
        }

        // reconstruct Householder vector
        v[0] = T{1}; // implicit
        for (std::size_t j = 1; j < vLen; ++j) {
            v[j] = A[k, k + 1 + j];
        }

        // V = V * H_k = V * (I - tau * v * v^H)
        // for each row i of V
        for (std::size_t i = 0; i < n; ++i) {
            T dot{0};
            for (std::size_t j = 0; j < vLen; ++j) {
                dot += V[i, k + 1 + j] * v[j];
            }
            T scale = tauV[k] * dot;
            for (std::size_t j = 0; j < vLen; ++j) {
                V[i, k + 1 + j] -= scale * conj(v[j]);
            }
        }
    }
}

template<typename T, typename BaseValueType = gr::meta::fundamental_base_value_type_t<T>>
svd::Status svdGolubReinsch(Tensor<T>& A, Tensor<T>* V, Tensor<BaseValueType>& singularValues, std::size_t maxIter, BaseValueType tol, bool computeV) {
    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    if (minDim == 0) {
        singularValues.resize({0});
        if (computeV && V) {
            V->resize({n, 0});
        }
        return svd::Status::Success;
    }

    // step 1: bidiagonalise A
    std::vector<BaseValueType> tauU, tauV;
    std::vector<BaseValueType> diag, superdiag;
    bidiagonalise(A, tauU, tauV, diag, superdiag);

    // step 2: accumulate U and V if needed
    Tensor<T> Umat, Vmat;
    accumulateU(A, tauU, Umat);
    if (computeV) {
        accumulateV(A, tauV, Vmat);
    }

    // step 3: implicit QR iteration on bidiagonal
    std::size_t iter = 0;
    std::size_t end  = minDim;

    while (end > 1 && iter < maxIter) {
        // find largest unreduced submatrix
        std::size_t start = 0;

        // check for negligible superdiagonal elements (deflation)
        for (std::size_t i = end - 1; i > 0; --i) {
            BaseValueType off    = std::abs(superdiag[i - 1]);
            BaseValueType thresh = tol * (std::abs(diag[i - 1]) + std::abs(diag[i]));
            if (off <= thresh) {
                superdiag[i - 1] = BaseValueType{0};
                if (i == end - 1) {
                    --end; // deflate from bottom
                } else {
                    start = i; // found unreduced block
                    break;
                }
            }
        }

        if (end <= 1) {
            break;
        }

        // check for zero diagonal (special handling)
        bool hasZeroDiag = false;
        for (std::size_t i = start; i < end; ++i) {
            if (std::abs(diag[i]) < tol * std::numeric_limits<BaseValueType>::epsilon()) {
                hasZeroDiag = true;
                // zero out row by Givens rotations
                if (i < end - 1 && std::abs(superdiag[i]) > tol * std::numeric_limits<BaseValueType>::epsilon()) {
                    for (std::size_t j = i + 1; j < end; ++j) {
                        BaseValueType c, s, r;
                        givensRotation<T>(diag[j], superdiag[i], c, s, r);
                        diag[j] = r;
                        if (j < end - 1) {
                            BaseValueType temp = superdiag[j];
                            superdiag[j]       = c * temp;
                            superdiag[i]       = -s * temp;
                        } else {
                            superdiag[i] = BaseValueType{0};
                        }
                        applyGivensRight(Umat, j, i, c, -s, 0, m);
                    }
                }
                break;
            }
        }

        if (hasZeroDiag) {
            ++iter;
            continue;
        }

        // perform one QR step
        bidiagonalQRStep<T>(diag, superdiag, &Umat, computeV ? &Vmat : nullptr, start, end, tol);
        ++iter;
    }

    // step 4: ensure singular values are positive and sort
    for (std::size_t i = 0; i < minDim; ++i) {
        if (diag[i] < BaseValueType{0}) {
            diag[i] = -diag[i];
            // flip sign of corresponding column in Umat
            for (std::size_t j = 0; j < m; ++j) {
                Umat[j, i] = -Umat[j, i];
            }
        }
    }

    // copy results
    singularValues.resize({minDim});
    for (std::size_t i = 0; i < minDim; ++i) {
        singularValues[i] = diag[i];
    }

    A = std::move(Umat);
    if (computeV && V) {
        *V = std::move(Vmat);
    }

    return (iter < maxIter) ? svd::Status::Success : svd::Status::MaxIterations;
}

template<typename BaseValueType, typename T>
void sortSingularValues(Tensor<BaseValueType>& S, Tensor<T>* U, Tensor<T>* V) {
    const std::size_t n = S.size();
    if (n == 0) {
        return;
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::ranges::sort(indices, [&S](std::size_t i, std::size_t j) { return S[i] > S[j]; });

    Tensor<BaseValueType> sortedS({n});
    for (std::size_t i = 0; i < n; ++i) {
        sortedS[i] = S[indices[i]];
    }
    S = std::move(sortedS);

    if (U && U->size() > 0 && U->extent(1) >= n) {
        Tensor<T> sortedU({U->extent(0), U->extent(1)});
        for (std::size_t row = 0; row < U->extent(0); ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                sortedU[row, col] = (*U)[row, indices[col]];
            }
            for (std::size_t col = n; col < U->extent(1); ++col) {
                sortedU[row, col] = (*U)[row, col];
            }
        }
        *U = std::move(sortedU);
    }

    if (V && V->size() > 0 && V->extent(1) >= n) {
        Tensor<T> sortedV({V->extent(0), V->extent(1)});
        for (std::size_t row = 0; row < V->extent(0); ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                sortedV[row, col] = (*V)[row, indices[col]];
            }
            for (std::size_t col = n; col < V->extent(1); ++col) {
                sortedV[row, col] = (*V)[row, col];
            }
        }
        *V = std::move(sortedV);
    }
}

} // namespace detail

/**
 * @brief singular value decomposition: A = U * diag(S) * V^H
 *
 * computes the thin SVD by default (U is m×min(m,n), V is n×min(m,n))
 * set config.fullMatrices = true for full SVD
 *
 * @param policy execution policy (cpu_policy, cpu_parallel_policy)
 * @param U output left singular vectors [m × k] where k = min(m,n) or m if fullMatrices
 * @param S output singular values [k], always real and non-negative, descending order
 * @param V output right singular vectors [n × k] where k = min(m,n) or n if fullMatrices
 * @param A input matrix [m × n]
 * @param config SVD configuration
 * @return status indicating success or type of failure
 */
template<ExecutionPolicy Policy, TensorLike TensorU, TensorLike TensorS, TensorLike TensorV, TensorLike TensorA>
requires std::same_as<typename TensorU::value_type, typename TensorA::value_type> && std::same_as<typename TensorV::value_type, typename TensorA::value_type> && std::same_as<typename TensorS::value_type, gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>
svd::Status gesvd(Policy&& /*policy*/, TensorU& U, TensorS& S, TensorV& V, const TensorA& A, const svd::Config<gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>& config = {}) {
    using T             = typename TensorA::value_type;
    using BaseValueType = gr::meta::fundamental_base_value_type_t<T>;

    if (A.rank() != 2) {
        return svd::Status::InvalidInput;
    }

    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    if (m == 0 || n == 0) {
        U.resize({m, 0});
        S.resize({0});
        V.resize({n, 0});
        return svd::Status::Success;
    }

    std::size_t actualK = (config.k > 0 && config.k <= minDim) ? config.k : minDim;

    BaseValueType tol = config.tolerance;
    if (tol < BaseValueType{0}) {
        tol = std::numeric_limits<BaseValueType>::epsilon() * static_cast<BaseValueType>(std::max(m, n));
    }

    std::size_t maxIter = config.maxIterations;
    if (maxIter == 0) {
        maxIter = 30 * std::max(m, n);
    }

    // select algorithm
    svd::Algorithm algo = config.algorithm;
    if (algo == svd::Algorithm::Auto) {
        // GolubReinsch for real types, Jacobi for complex (GR complex support is WIP)
        if constexpr (gr::meta::complex_like<T>) {
            algo = svd::Algorithm::Jacobi;
        } else {
            algo = svd::Algorithm::GolubReinsch;
        }
    } else if (algo == svd::Algorithm::GolubReinsch && gr::meta::complex_like<T>) {
        // GolubReinsch doesn't fully support complex yet, fall back to Jacobi
        algo = svd::Algorithm::Jacobi;
    }

    // both algorithms work best on tall matrices (m >= n)
    // for wide matrices (m < n), transpose and swap U/V
    bool      transposed = false;
    Tensor<T> Awork;

    if (m < n) {
        transposed = true;
        Awork      = conjTranspose(A);
    } else {
        Awork = Tensor<T>(A);
    }

    // perform SVD
    svd::Status           status;
    Tensor<T>             Vtmp;
    Tensor<BaseValueType> Stmp;

    if (transposed) {
        if (algo == svd::Algorithm::Jacobi) {
            status = detail::svdJacobi(Awork, &Vtmp, Stmp, maxIter, tol, true);
        } else {
            status = detail::svdGolubReinsch(Awork, &Vtmp, Stmp, maxIter, tol, true);
        }
        V = std::move(Awork);
        U = std::move(Vtmp);
        S = std::move(Stmp);
    } else {
        if (algo == svd::Algorithm::Jacobi) {
            status = detail::svdJacobi(Awork, &V, S, maxIter, tol, true);
        } else {
            status = detail::svdGolubReinsch(Awork, &V, S, maxIter, tol, true);
        }
        U = std::move(Awork);
    }

    // ensure correct dimensions for thin SVD
    if (U.extent(0) != m || U.extent(1) != minDim) {
        Tensor<T> U_correct({m, minDim});
        U_correct.fill(T{0});
        for (std::size_t i = 0; i < std::min(m, U.extent(0)); ++i) {
            for (std::size_t j = 0; j < std::min(minDim, U.extent(1)); ++j) {
                U_correct[i, j] = U[i, j];
            }
        }
        U = std::move(U_correct);
    }

    if (V.extent(0) != n || V.extent(1) != minDim) {
        Tensor<T> V_correct({n, minDim});
        V_correct.fill(T{0});
        for (std::size_t i = 0; i < std::min(n, V.extent(0)); ++i) {
            for (std::size_t j = 0; j < std::min(minDim, V.extent(1)); ++j) {
                V_correct[i, j] = V[i, j];
            }
        }
        V = std::move(V_correct);
    }

    if (S.size() != minDim) {
        Tensor<BaseValueType> S_correct({minDim});
        S_correct.fill(BaseValueType{0});
        for (std::size_t i = 0; i < std::min(minDim, S.size()); ++i) {
            S_correct[i] = S[i];
        }
        S = std::move(S_correct);
    }

    // sort singular values in descending order
    detail::sortSingularValues(S, &U, &V);

    // truncate if user requested fewer singular values
    if (actualK < minDim) {
        Tensor<T>             U_final({m, actualK});
        Tensor<BaseValueType> S_final({actualK});
        Tensor<T>             V_final({n, actualK});

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < actualK; ++j) {
                U_final[i, j] = U[i, j];
            }
        }

        for (std::size_t i = 0; i < actualK; ++i) {
            S_final[i] = S[i];
        }

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < actualK; ++j) {
                V_final[i, j] = V[i, j];
            }
        }

        U = std::move(U_final);
        S = std::move(S_final);
        V = std::move(V_final);

        if (status == svd::Status::Success) {
            status = svd::Status::EarlyReturn;
        }
    }

    return status;
}

/**
 * @brief singular value decomposition: A = U * diag(S) * V^H  (default to cpu_policy)
 *
 * computes the thin SVD by default (U is m×min(m,n), V is n×min(m,n))
 * set config.fullMatrices = true for full SVD
 *
 * @param U output left singular vectors [m × k] where k = min(m,n) or m if fullMatrices
 * @param S output singular values [k], always real and non-negative, descending order
 * @param V output right singular vectors [n × k] where k = min(m,n) or n if fullMatrices
 * @param A input matrix [m × n]
 * @param config SVD configuration
 * @return status indicating success or type of failure
 */
template<TensorLike TensorU, TensorLike TensorS, TensorLike TensorV, TensorLike TensorA>
requires std::same_as<typename TensorU::value_type, typename TensorA::value_type> && std::same_as<typename TensorV::value_type, typename TensorA::value_type> && std::same_as<typename TensorS::value_type, gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>
svd::Status gesvd(TensorU& U, TensorS& S, TensorV& V, const TensorA& A, const svd::Config<gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>& config = {}) {
    return gesvd(cpu_policy{}, U, S, V, A, config);
}

/**
 * @brief singular values only decomposition
 *
 * computes only the singular values S without U and V matrices
 * more efficient when only singular values are needed
 *
 * @param policy execution policy
 * @param S output singular values [min(m,n)], real and non-negative, descending order
 * @param A input matrix [m × n]
 * @param config SVD configuration
 * @return status indicating success or type of failure
 */
template<ExecutionPolicy Policy, TensorLike TensorS, TensorLike TensorA>
requires std::same_as<typename TensorS::value_type, gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>
svd::Status gesvd(Policy&& /*policy*/, TensorS& S, const TensorA& A, const svd::Config<gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>& config = {}) {
    using T             = typename TensorA::value_type;
    using BaseValueType = gr::meta::fundamental_base_value_type_t<T>;

    if (A.rank() != 2) {
        return svd::Status::InvalidInput;
    }

    const std::size_t m      = A.extent(0);
    const std::size_t n      = A.extent(1);
    const std::size_t minDim = std::min(m, n);

    if (m == 0 || n == 0) {
        S.resize({0});
        return svd::Status::Success;
    }

    std::size_t actualK = (config.k > 0 && config.k <= minDim) ? config.k : minDim;

    BaseValueType tol = config.tolerance;
    if (tol < BaseValueType{0}) {
        tol = std::numeric_limits<BaseValueType>::epsilon() * static_cast<BaseValueType>(std::max(m, n));
    }

    std::size_t maxIter = config.maxIterations;
    if (maxIter == 0) {
        maxIter = 30 * std::max(m, n);
    }

    svd::Algorithm algo = config.algorithm;
    if (algo == svd::Algorithm::Auto) {
        // Golub-Reinsch for real types, Jacobi for complex (GR complex support is WIP)
        if constexpr (gr::meta::complex_like<T>) {
            algo = svd::Algorithm::Jacobi;
        } else {
            algo = svd::Algorithm::GolubReinsch;
        }
    } else if (algo == svd::Algorithm::GolubReinsch && gr::meta::complex_like<T>) {
        // Golub-Reinsch doesn't fully support complex yet, fall back to Jacobi
        algo = svd::Algorithm::Jacobi;
    }

    bool      transposed = (m < n);
    Tensor<T> Awork      = transposed ? conjTranspose(A) : Tensor<T>(A);

    svd::Status status;
    Tensor<T>*  noV = nullptr; // explicit type for template deduction
    if (algo == svd::Algorithm::Jacobi) {
        status = detail::svdJacobi(Awork, noV, S, maxIter, tol, false);
    } else {
        status = detail::svdGolubReinsch(Awork, noV, S, maxIter, tol, false);
    }

    // ensure correct size
    if (S.size() != minDim) {
        Tensor<BaseValueType> S_correct({minDim});
        S_correct.fill(BaseValueType{0});
        for (std::size_t i = 0; i < std::min(minDim, S.size()); ++i) {
            S_correct[i] = S[i];
        }
        S = std::move(S_correct);
    }

    // sort descending
    std::ranges::sort(S, std::greater<>{});

    // truncate if requested
    if (actualK < minDim) {
        Tensor<BaseValueType> S_final({actualK});
        for (std::size_t i = 0; i < actualK; ++i) {
            S_final[i] = S[i];
        }
        S = std::move(S_final);

        if (status == svd::Status::Success) {
            status = svd::Status::EarlyReturn;
        }
    }

    return status;
}

/**
 * @brief singular values only decomposition (default to cpu_policy)
 *
 * computes only the singular values S without U and V matrices
 * more efficient when only singular values are needed
 *
 * @param S output singular values [min(m,n)], real and non-negative, descending order
 * @param A input matrix [m × n]
 * @param config SVD configuration
 * @return status indicating success or type of failure
 */
template<TensorLike TensorS, TensorLike TensorA>
requires std::same_as<typename TensorS::value_type, gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>
svd::Status gesvd(TensorS& S, const TensorA& A, const svd::Config<gr::meta::fundamental_base_value_type_t<typename TensorA::value_type>>& config = {}) {
    return gesvd(cpu_policy{}, S, A, config);
}

} // namespace gr::math

#endif // GNURADIO_SVD_HPP
