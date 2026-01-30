#ifndef GNURADIO_TENSOR_OPS_HPP
#define GNURADIO_TENSOR_OPS_HPP

#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <execution>
#include <expected>
#include <functional>
#include <numeric>
#include <span>
#include <type_traits>

#include <format>
#include <gnuradio-4.0/Message.hpp>

namespace gr::math {
namespace config {

// Size thresholds (can override with -DGR_MATH_TINY_THRESHOLD=value)
#ifndef GR_MATH_TINY_THRESHOLD
constexpr std::size_t TINY_THRESHOLD = 32;
#else
constexpr std::size_t TINY_THRESHOLD = GR_MATH_TINY_THRESHOLD;
#endif

#ifndef GR_MATH_MEDIUM_THRESHOLD
constexpr std::size_t MEDIUM_THRESHOLD = 1024;
#else
constexpr std::size_t MEDIUM_THRESHOLD = GR_MATH_MEDIUM_THRESHOLD;
#endif

// SIMD Kernel Parameters - Architecture specific
template<typename T>
struct KernelParams;

template<>
struct KernelParams<float> {
    static constexpr std::size_t MR         = 16;   // Rows in micro-kernel
    static constexpr std::size_t NR         = 6;    // Cols in micro-kernel
    static constexpr std::size_t SIMD_WIDTH = 8;    // AVX2: 8 floats
    static constexpr std::size_t NC         = 4096; // L3 cache blocking
    static constexpr std::size_t KC         = 256;  // Inner dim blocking
    static constexpr std::size_t MC         = 128;  // L2 cache blocking
};

template<>
struct KernelParams<double> {
    static constexpr std::size_t MR         = 8;
    static constexpr std::size_t NR         = 6;
    static constexpr std::size_t SIMD_WIDTH = 4; // AVX2: 4 doubles
    static constexpr std::size_t NC         = 4096;
    static constexpr std::size_t KC         = 256;
    static constexpr std::size_t MC         = 128;
};

template<typename T>
requires std::integral<T>
struct KernelParams<T> {
    static constexpr std::size_t MR         = 4;
    static constexpr std::size_t NR         = 4;
    static constexpr std::size_t SIMD_WIDTH = 1; // integer types: smaller kernels (no SIMD for now)
    static constexpr std::size_t NC         = 2048;
    static constexpr std::size_t KC         = 256;
    static constexpr std::size_t MC         = 64;
};

template<typename T>
struct GemvParams {
    static constexpr std::size_t SIMD_WIDTH  = KernelParams<T>::SIMD_WIDTH;
    static constexpr std::size_t UNROLL      = 4;
    static constexpr std::size_t CACHE_BLOCK = 2048;
};

} // namespace config

enum class TransposeOp : std::int8_t {
    NoTrans   = 0, /// no transpose
    Trans     = 1, /// transpose
    ConjTrans = 2, /// conjugated transpose
    Dynamic   = -1 /// runtime decision (not yet implemented)
};

enum class MatrixSize : std::int8_t {
    Tiny,   /// typ. 32x32
    Medium, /// typ 4096x4096
    Large   /// > Medium
};

struct cpu_policy {
    static constexpr bool is_gpu       = false;
    static constexpr bool use_parallel = false;
};

struct cpu_parallel_policy {
    static constexpr bool is_gpu       = false;
    static constexpr bool use_parallel = true;
    std::size_t           num_threads  = 0UZ; // 0 = auto-detect
};

template<typename QueueType>
struct gpu_policy { /// GPU policy (for future SYCL/CUDA support)
    static constexpr bool is_gpu = true;
    QueueType&            queue;
    explicit gpu_policy(QueueType& q) : queue(q) {}
};

template<typename T>
concept ExecutionPolicy = requires {
    { T::is_gpu } -> std::convertible_to<bool>;
};

template<typename T>
concept CpuExecutionPolicy = ExecutionPolicy<T> && (!T::is_gpu);

template<typename T>
concept GpuExecutionPolicy = ExecutionPolicy<T> && T::is_gpu;

struct dimension_mismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct invalid_transpose : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace gr::math

namespace gr {
template<typename T, std::size_t... Ex>
struct Tensor;

template<typename T, std::size_t... Ex>
struct TensorView;
} // namespace gr

namespace gr::math {

template<typename T, std::size_t... Ex>
struct TensorOps {
    using TensorType = Tensor<T, Ex...>;

    [[maybe_unused]] static constexpr TensorType& add_inplace(TensorType& self, const TensorType& other) {
        auto same_shape = [](std::span<const std::size_t> a1, std::span<const std::size_t> b1) { return a1.size() == b1.size() && std::equal(a1.begin(), a1.end(), b1.begin()); };
        if (!same_shape(self.extents(), other.extents())) {
            throw std::runtime_error("Tensor dimensions must match for element-wise operations");
        }
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), other.begin(), self.begin(), std::plus<>{});
#else
        std::transform(self.begin(), self.end(), other.begin(), self.begin(), std::plus<>{});
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType add(const TensorType& a, const TensorType& b) {
        TensorType result(a);
        return add_inplace(result, b);
    }

    [[maybe_unused]] static constexpr TensorType& subtract_inplace(TensorType& self, const TensorType& other) {
        auto same_shape = [](std::span<const std::size_t> a1, std::span<const std::size_t> b1) { return a1.size() == b1.size() && std::equal(a1.begin(), a1.end(), b1.begin()); };
        if (!same_shape(self.extents(), other.extents())) {
            throw std::runtime_error("Tensor dimensions must match for element-wise operations");
        }
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), other.begin(), self.begin(), std::minus<>{});
#else
        std::transform(self.begin(), self.end(), other.begin(), self.begin(), std::minus<>{});
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType subtract(const TensorType& a, const TensorType& b) {
        TensorType result(a);
        return subtract_inplace(result, b);
    }

    [[maybe_unused]] static constexpr TensorType& multiply_scalar_inplace(TensorType& self, const T& scalar) {
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), self.begin(), [scalar](const T& x) { return x * scalar; });
#else
        std::transform(self.begin(), self.end(), self.begin(), [scalar](const T& x) { return x * scalar; });
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType multiply_scalar(const TensorType& tensor, const T& scalar) {
        TensorType result(tensor);
        return multiply_scalar_inplace(result, scalar);
    }

    [[maybe_unused]] static constexpr TensorType& divide_scalar_inplace(TensorType& self, const T& scalar) {
        if (scalar == T{0}) {
            throw std::runtime_error("Division by zero");
        }
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), self.begin(), [scalar](const T& x) { return x / scalar; });
#else
        std::ranges::transform(self.begin(), self.end(), self.begin(), [scalar](const T& x) { return x / scalar; });
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType divide_scalar(const TensorType& tensor, const T& scalar) {
        TensorType result(tensor);
        return divide_scalar_inplace(result, scalar);
    }

    // element-wise multiplication (Hadamard product)
    [[maybe_unused]] static constexpr TensorType& multiply_elementwise_inplace(TensorType& self, const TensorType& other) {
        auto same_shape = [](std::span<const std::size_t> a1, std::span<const std::size_t> b1) { return a1.size() == b1.size() && std::equal(a1.begin(), a1.end(), b1.begin()); };
        if (!same_shape(self.extents(), other.extents())) {
            throw std::runtime_error("Tensor dimensions must match for element-wise operations");
        }

#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::ranges::transform(std::execution::unseq, self, other.begin(), self.begin(), std::multiplies<>{});
#else
        std::ranges::transform(self, other.begin(), self.begin(), std::multiplies<>{});
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType multiply_elementwise(const TensorType& a, const TensorType& b) {
        TensorType result(a);
        return multiply_elementwise_inplace(result, b);
    }

    [[maybe_unused]] static constexpr TensorType& divide_elementwise_inplace(TensorType& self, const TensorType& other) {
        auto same_shape = [](std::span<const std::size_t> a1, std::span<const std::size_t> b1) { return a1.size() == b1.size() && std::equal(a1.begin(), a1.end(), b1.begin()); };
        if (!same_shape(self.extents(), other.extents())) {
            throw std::runtime_error("Tensor dimensions must match for element-wise operations");
        }
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::ranges::transform(std::execution::unseq, self, other.begin(), self.begin(), std::divides<>{});
#else
        std::ranges::transform(self, other.begin(), self.begin(), std::divides<>{});
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType divide_elementwise(const TensorType& a, const TensorType& b) {
        TensorType result(a);
        return divide_elementwise_inplace(result, b);
    }

    [[nodiscard]] static constexpr T sum(const TensorType& self) { return std::accumulate(self.begin(), self.end(), T{0}); }

    [[nodiscard]] static constexpr T product(const TensorType& self) { return std::reduce(self.begin(), self.end(), T{1}, std::multiplies<>{}); }

    [[nodiscard]] static constexpr auto mean(const TensorType& self) {
        if (self.empty()) {
            throw std::runtime_error("Cannot compute mean of empty tensor");
        }
        if constexpr (std::is_integral_v<T>) {
            return static_cast<double>(sum(self)) / static_cast<double>(self.size());
        } else {
            return sum(self) / static_cast<T>(self.size());
        }
    }

    [[nodiscard]] static constexpr auto variance(const TensorType& self) {
        auto m      = mean(self);
        T    sum_sq = 0;
        for (const auto& x : self) {
            auto diff = x - m;
            sum_sq += diff * diff;
        }
        return sum_sq / static_cast<T>(self.size());
    }

    [[nodiscard]] static constexpr auto std_dev(const TensorType& self) { return std::sqrt(variance(self)); }

    [[nodiscard]] static constexpr T min(const TensorType& self) {
        if (self.empty()) {
            throw std::runtime_error("Cannot find min of empty tensor");
        }
        return *std::min_element(self.begin(), self.end());
    }

    [[nodiscard]] static constexpr T max(const TensorType& self) {
        if (self.empty()) {
            throw std::runtime_error("Cannot find max of empty tensor");
        }
        return *std::max_element(self.begin(), self.end());
    }

    [[nodiscard]] static constexpr std::size_t argmin(const TensorType& self) {
        if (self.empty()) {
            throw std::runtime_error("Cannot find argmin of empty tensor");
        }
        auto it = std::min_element(self.begin(), self.end());
        return static_cast<std::size_t>(std::distance(self.begin(), it));
    }

    [[nodiscard]] static constexpr std::size_t argmax(const TensorType& self) {
        if (self.empty()) {
            throw std::runtime_error("Cannot find argmax of empty tensor");
        }
        auto it = std::max_element(self.begin(), self.end());
        return static_cast<std::size_t>(std::distance(self.begin(), it));
    }

    [[nodiscard]] static constexpr TensorType sum_axis(const TensorType& self, std::size_t axis) {
        if (axis >= self.rank()) {
            throw std::out_of_range("Axis out of range");
        }

        auto                     extents = self.extents();
        std::vector<std::size_t> new_extents;
        for (std::size_t i = 0; i < self.rank(); ++i) {
            if (i != axis) {
                new_extents.push_back(extents[i]);
            }
        }
        if (new_extents.empty()) {
            new_extents.push_back(1); // scalar
        }

        TensorType result(extents_from, new_extents);
        result.fill(T{0});

        auto result_strides = result.strides();

        std::vector<std::size_t> indices(self.rank());
        for (std::size_t i = 0; i < self.size(); ++i) {
            // Decode flat index properly
            std::size_t tmp = i;
            for (std::size_t dim = self.rank(); dim-- > 0;) {
                indices[dim] = tmp % extents[dim];
                tmp /= extents[dim];
            }

            // Map to reduced index
            std::size_t result_idx = 0, rd = 0;
            for (std::size_t dim = 0; dim < self.rank(); ++dim) {
                if (dim != axis) {
                    result_idx += indices[dim] * result_strides[rd];
                    rd++;
                }
            }
            result.data()[result_idx] += self.data()[i];
        }

        return result;
    }

    [[nodiscard]] static constexpr TensorType mean_axis(const TensorType& self, std::size_t axis) {
        auto sum_result = sum_axis(self, axis);
        auto count      = self.extent(axis);
        return divide_scalar_inplace(sum_result, static_cast<T>(count));
    }

    [[nodiscard]] static constexpr bool contains_nan(const TensorType& self)
    requires std::floating_point<T>
    {
        return std::any_of(self.begin(), self.end(), [](const T& x) { return std::isnan(x); });
    }

    [[nodiscard]] static constexpr bool contains_inf(const TensorType& self)
    requires std::floating_point<T>
    {
        return std::any_of(self.begin(), self.end(), [](const T& x) { return std::isinf(x); });
    }

    [[maybe_unused]] static constexpr TensorType& replace_nan(TensorType& self, const T& value)
    requires std::floating_point<T>
    {
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), self.begin(), [value](const T& x) { return std::isnan(x) ? value : x; });
#else
        std::transform(self.begin(), self.end(), self.begin(), [value](const T& x) { return std::isnan(x) ? value : x; });
#endif
        return self;
    }

    [[maybe_unused]] static constexpr TensorType& clip_inplace(TensorType& self, const T& min_val, const T& max_val) {
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::ranges::transform(std::execution::unseq, self, self.begin(), [min_val, max_val](const T& x) { return std::clamp(x, min_val, max_val); });
#else
        std::ranges::transform(self, self.begin(), [min_val, max_val](const T& x) { return std::clamp(x, min_val, max_val); });
#endif
        return self;
    }

    [[nodiscard]] static constexpr TensorType clip(const TensorType& self, const T& min_val, const T& max_val) {
        TensorType result(self);
        return clip_inplace(result, min_val, max_val);
    }

    [[nodiscard]] static constexpr TensorType abs(const TensorType& self) {
        TensorType     result(extents_from, self.extents());
        constexpr auto abs = [](const T& x) {
            if constexpr (std::is_unsigned_v<T>) {
                return x;
            } else {
                return std::abs(x);
            }
        };
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::ranges::transform(std::execution::unseq, self, result.begin(), abs);
#else
        std::ranges::transform(self, result.begin(), abs);
#endif
    }

    [[nodiscard]] static constexpr TensorType sign(const TensorType& self) {
        TensorType     result(extents_from, self.extents());
        constexpr auto sign = [](const T& x) {
            if (x > T{0}) {
                return T{1};
            }
            if (x < T{0}) {
                return T{-1};
            }
            return T{0};
        };
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::ranges::transform(std::execution::unseq, self, result.begin(), sign);
#else
        std::ranges::transform(self, result.begin(), sign);
#endif
        return result;
    }

    [[nodiscard]] static constexpr TensorType transpose(const TensorType& self) {
        if (self.rank() != 2) {
            throw std::runtime_error("Transpose currently only supports 2D tensors");
        }

        auto       extents = self.extents();
        TensorType result({extents[1], extents[0]});

        for (std::size_t i = 0; i < extents[0]; ++i) {
            for (std::size_t j = 0; j < extents[1]; ++j) {
                result[j, i] = self[i, j];
            }
        }

        return result;
    }

    [[nodiscard]] static constexpr TensorType diag(const TensorType& self) {
        if (self.rank() != 2) {
            throw std::runtime_error("Diagonal extraction requires 2D tensor");
        }

        auto        extents = self.extents();
        std::size_t n       = std::min(extents[0], extents[1]);

        TensorType result({n});
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = self[i, i];
        }

        return result;
    }

    [[nodiscard]] static constexpr TensorType diag_matrix(const TensorType& self) {
        if (self.rank() != 1) {
            throw std::runtime_error("Creating diagonal matrix requires 1D tensor");
        }

        std::size_t n = self.size();
        TensorType  result({n, n});
        result.fill(T{0});

        for (std::size_t i = 0; i < n; ++i) {
            result[i, i] = self[i];
        }

        return result;
    }

    template<typename... Tensors>
    [[nodiscard]] static constexpr TensorType concatenate(std::size_t axis, const TensorType& first, const Tensors&... rest) {
        std::vector<const TensorType*> tensors{&first, &rest...};
        if (axis >= first.rank()) {
            throw std::out_of_range("concatenation axis out of range");
        }

        auto ref_extents = first.extents();
        for (auto* t : tensors) {
            if (t->rank() != first.rank()) {
                throw std::runtime_error("all tensors must have same rank for concatenation");
            }
            auto te = t->extents();
            for (std::size_t i = 0; i < first.rank(); ++i) {
                if (i != axis && te[i] != ref_extents[i]) {
                    throw std::runtime_error("all dimensions except concatenation axis must match");
                }
            }
        }

        std::vector<std::size_t> result_extents(ref_extents.begin(), ref_extents.end());
        result_extents[axis] = 0;
        for (auto* t : tensors) {
            result_extents[axis] += t->extent(axis);
        }

        TensorType result(extents_from, result_extents);
        auto       result_strides = result.strides();

        // Copy data from each tensor
        std::size_t axis_offset = 0;
        for (auto* tensor_ptr : tensors) {
            auto                     tensor_extents = tensor_ptr->extents();
            std::vector<std::size_t> indices(first.rank());

            for (std::size_t i = 0; i < tensor_ptr->size(); ++i) {
                // Proper flat -> multi index using extents
                std::size_t tmp = i;
                for (std::size_t dim = tensor_ptr->rank(); dim-- > 0;) {
                    indices[dim] = tmp % tensor_extents[dim];
                    tmp /= tensor_extents[dim];
                }

                // Adjust index along concatenation axis
                indices[axis] += axis_offset;

                // Calculate flat index in result
                std::size_t result_idx = 0;
                for (std::size_t dim = 0; dim < result.rank(); ++dim) {
                    result_idx += indices[dim] * result_strides[dim];
                }

                result.data()[result_idx] = tensor_ptr->data()[i];
            }

            axis_offset += tensor_ptr->extent(axis);
        }

        return result;
    }

    [[nodiscard]] static constexpr std::vector<TensorType> split(const TensorType& self, std::size_t axis, std::size_t n_splits) {
        if (axis >= self.rank()) {
            throw std::out_of_range("Split axis out of range");
        }

        std::size_t axis_size = self.extent(axis);
        if (axis_size % n_splits != 0) {
            throw std::runtime_error("Tensor size along axis must be divisible by number of splits");
        }

        std::size_t             split_size = axis_size / n_splits;
        std::vector<TensorType> results;

        auto self_extents = self.extents();
        auto self_strides = self.strides();

        for (std::size_t split_idx = 0; split_idx < n_splits; ++split_idx) {
            // Create result tensor for this split
            std::vector<std::size_t> split_extents(self_extents.begin(), self_extents.end());
            split_extents[axis] = split_size;
            TensorType split_tensor(extents_from, split_extents);

            // copy data for this split
            for (std::size_t i = 0; i < split_tensor.size(); ++i) {
                // Convert flat index to multi-dimensional indices for split tensor
                std::size_t              temp = i;
                std::vector<std::size_t> indices(split_tensor.rank());
                for (std::size_t d = split_tensor.rank(); d-- > 0;) {
                    indices[d] = temp % split_extents[d];
                    temp /= split_extents[d];
                }

                // adjust index along split axis to reference original tensor
                indices[axis] += split_idx * split_size;

                // calculate flat index in original tensor
                std::size_t orig_idx = 0;
                for (std::size_t dim = 0; dim < self.rank(); ++dim) {
                    orig_idx += indices[dim] * self_strides[dim];
                }

                split_tensor.data()[i] = self.data()[orig_idx];
            }

            results.push_back(std::move(split_tensor));
        }

        return results;
    }

    template<typename U = T>
    [[nodiscard]] static constexpr auto real(const TensorType& self) -> std::enable_if_t<std::is_same_v<U, std::complex<typename U::value_type>>, Tensor<typename U::value_type, Ex...>> {
        using RealType = U::value_type;
        Tensor<RealType, Ex...> result(extents_from, self.extents());
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), result.begin(), [](const U& x) { return x.real(); });
#else
        std::ranges::transform(self, result.begin(), [](const U& x) { return x.real(); });
#endif
        return result;
    }

    template<typename U = T>
    [[nodiscard]] static constexpr auto imag(const TensorType& self) -> std::enable_if_t<std::is_same_v<U, std::complex<typename U::value_type>>, Tensor<typename U::value_type, Ex...>> {
        using RealType = U::value_type;
        Tensor<RealType, Ex...> result(extents_from, self.extents());
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), result.begin(), [](const U& x) { return x.imag(); });
#else
        std::ranges::transform(self, result.begin(), [](const U& x) { return x.imag(); });
#endif
        return result;
    }

    template<typename U = T>
    [[nodiscard]] static constexpr auto conj(const TensorType& self) -> std::enable_if_t<std::is_same_v<U, std::complex<typename U::value_type>>, TensorType> {
        TensorType result(extents_from, self.extents());
#if defined(__GLIBCXX__) && !defined(__ACPP__)
        std::transform(std::execution::unseq, self.begin(), self.end(), result.begin(), [](const U& x) { return std::conj(x); });
#else
        std::ranges::transform(self, result.begin(), [](const U& x) { return std::conj(x); });
#endif
        return result;
    }
};

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator+(const Tensor<T, Ex...>& a, const Tensor<T, Ex...>& b) {
    return TensorOps<T, Ex...>::add(a, b);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...>& operator+=(Tensor<T, Ex...>& a, const Tensor<T, Ex...>& b) {
    return TensorOps<T, Ex...>::add_inplace(a, b);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator-(const Tensor<T, Ex...>& a, const Tensor<T, Ex...>& b) {
    return TensorOps<T, Ex...>::subtract(a, b);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...>& operator-=(Tensor<T, Ex...>& a, const Tensor<T, Ex...>& b) {
    return TensorOps<T, Ex...>::subtract_inplace(a, b);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator*(const Tensor<T, Ex...>& tensor, const T& scalar) {
    return TensorOps<T, Ex...>::multiply_scalar(tensor, scalar);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator*(const T& scalar, const Tensor<T, Ex...>& tensor) {
    return TensorOps<T, Ex...>::multiply_scalar(tensor, scalar);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...>& operator*=(Tensor<T, Ex...>& tensor, const T& scalar) {
    return TensorOps<T, Ex...>::multiply_scalar_inplace(tensor, scalar);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator/(const Tensor<T, Ex...>& tensor, const T& scalar) {
    return TensorOps<T, Ex...>::divide_scalar(tensor, scalar);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...>& operator/=(Tensor<T, Ex...>& tensor, const T& scalar) {
    return TensorOps<T, Ex...>::divide_scalar_inplace(tensor, scalar);
}

template<class T, std::size_t... Ex>
Tensor<T, Ex...> operator-(const Tensor<T, Ex...>& tensor) {
    return TensorOps<T, Ex...>::multiply_scalar(tensor, T{-1});
}

template<class T, std::size_t... Ex1, std::size_t... Ex2>
auto gemm(const Tensor<T, Ex1...>& a, const Tensor<T, Ex2...>& b) {
    return TensorOps<T>::gemm(a, b);
}

template<class T, std::size_t... Ex>
T dot(const Tensor<T, Ex...>& a, const Tensor<T, Ex...>& b) {
    if (a.rank() != 1 || b.rank() != 1) {
        throw std::runtime_error("Dot product requires 1D tensors");
    }
    if (a.size() != b.size()) {
        throw std::runtime_error("Tensors must have same size for dot product");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), T{0});
}

template<class T, std::size_t... Ex>
auto transpose(const Tensor<T, Ex...>& tensor) {
    return TensorOps<T, Ex...>::transpose(tensor);
}

template<class T, std::size_t... Ex, typename... Tensors>
auto concatenate(std::size_t axis, const Tensor<T, Ex...>& first, const Tensors&... rest) {
    return TensorOps<T, Ex...>::concatenate(axis, first, rest...);
}

template<class T>
std::pair<Tensor<T>, Tensor<T>> broadcast(const Tensor<T>& a, const Tensor<T>& b) {
    auto a_ext = a.extents();
    auto b_ext = b.extents();

    bool same_shape = (a_ext.size() == b_ext.size()) && std::equal(a_ext.begin(), a_ext.end(), b_ext.begin());

    if (same_shape) {
        return {a, b};
    }

    // handle scalar broadcasting
    if (a.size() == 1) {
        Tensor<T> broadcast_a(b.extents());
        broadcast_a.fill(a.data()[0]);
        return {broadcast_a, b};
    }

    if (b.size() == 1) {
        Tensor<T> broadcast_b(a.extents());
        broadcast_b.fill(b.data()[0]);
        return {a, broadcast_b};
    }

    throw std::runtime_error("Broadcasting not supported for these tensor shapes");
}

template<typename T>
[[nodiscard]] constexpr auto squaredMagnitude(const T& val) noexcept { // complex-safe squared magnitude: |x|^2 for both real and complex types
    if constexpr (gr::meta::complex_like<T>) {
        return std::norm(val); // |z|^2 = real^2 + imag^2
    } else {
        return val * val;
    }
}

template<typename T>
[[nodiscard]] constexpr T hypot(T a, T b) noexcept {
    static_assert(std::is_floating_point_v<T>, "hypot requires floating-point type");
    a = std::abs(a);
    b = std::abs(b);
    if (a == T{0}) {
        return b;
    }
    if (b == T{0}) {
        return a;
    }
    if (a < b) {
        std::swap(a, b);
    }
    const T ratio = b / a;
    return a * std::sqrt(T{1} + ratio * ratio);
}

template<TensorLike Tensor>
[[nodiscard]] bool isFinite(const Tensor& t) {
    using T = typename tensor_traits<std::remove_cvref_t<Tensor>>::value_type;
    if constexpr (gr::meta::complex_like<T>) {
        return std::all_of(t.begin(), t.end(), [](const T& x) { return std::isfinite(x.real()) && std::isfinite(x.imag()); });
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::all_of(t.begin(), t.end(), [](const T& x) { return std::isfinite(x); });
    } else {
        return true; // integral types are always finite
    }
}

template<TensorLike Tensor>
[[nodiscard]] auto conjTranspose(const Tensor& A) {
    using T = typename tensor_traits<std::remove_cvref_t<Tensor>>::value_type;
    if (A.rank() != 2) {
        throw std::runtime_error("conjTranspose requires 2D tensor");
    }

    auto          extents = A.extents();
    gr::Tensor<T> result({extents[1], extents[0]});

    if constexpr (gr::meta::complex_like<T>) {
        for (std::size_t i = 0; i < extents[0]; ++i) {
            for (std::size_t j = 0; j < extents[1]; ++j) {
                result[j, i] = std::conj(A[i, j]);
            }
        }
    } else {
        for (std::size_t i = 0; i < extents[0]; ++i) {
            for (std::size_t j = 0; j < extents[1]; ++j) {
                result[j, i] = A[i, j];
            }
        }
    }

    return result;
}

/**
 * @brief Euclidean (L2) norm of a vector: ||x||_2 = sqrt(Σ|x_i|²)
 */
template<TensorLike TensorX>
[[nodiscard]] auto norm2(const TensorX& x) noexcept {
    using T     = typename tensor_traits<std::remove_cvref_t<TensorX>>::value_type;
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    if (x.empty()) {
        return RealT{0};
    }
    RealT sumSq{0};
    for (const auto& val : x) {
        sumSq += squaredMagnitude(val);
    }
    return std::sqrt(sumSq);
}

/**
 * @brief Frobenius norm of a matrix: ||A||_F = sqrt(Σ|a_ij|²)
 */
template<TensorLike TensorA>
[[nodiscard]] auto normFrobenius(const TensorA& A) noexcept {
    using T     = typename tensor_traits<std::remove_cvref_t<TensorA>>::value_type;
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    if (A.empty()) {
        return RealT{0};
    }
    RealT sumSq{0};
    for (const auto& val : A) {
        sumSq += squaredMagnitude(val);
    }
    return std::sqrt(sumSq);
}

/**
 * @brief Build Hankel matrix from signal: H[i,j] = signal[i + j]
 *
 * For signal length N and nRows rows, produces nRows × (N - nRows + 1) matrix.
 *
 * @param signal Input signal of length N
 * @param nRows Number of rows in the Hankel matrix (1 ≤ nRows ≤ N)
 * @param loc Source location for error reporting
 * @return Hankel matrix or error if dimensions are invalid
 */
template<typename T>
[[nodiscard]] std::expected<gr::Tensor<T>, gr::Error> hankel(std::span<const T> signal, std::size_t nRows, std::source_location loc = std::source_location::current()) {
    const std::size_t N = signal.size();
    if (nRows == 0UZ || nRows > N) {
        return std::unexpected(gr::Error(std::format("hankel: nRows ({}) must be in range [1, {}]", nRows, N), loc));
    }

    const std::size_t nCols = N - nRows + 1UZ;
    gr::Tensor<T>     H({nRows, nCols});

    for (std::size_t i = 0UZ; i < nRows; ++i) {
        for (std::size_t j = 0UZ; j < nCols; ++j) {
            H[i, j] = signal[i + j];
        }
    }
    return H;
}

/**
 * @brief Reconstruct signal from Hankel matrix by averaging anti-diagonals.
 *
 * This is the inverse operation of hankel() for exact Hankel matrices.
 * For approximate Hankel matrices, returns the best rank-preserving reconstruction.
 *
 * @param H Input Hankel matrix of shape [L × K]
 * @param loc Source location for error reporting
 * @return Reconstructed signal of length L + K - 1, or error if input is invalid
 */
template<typename T>
[[nodiscard]] std::expected<std::vector<T>, gr::Error> hankelAverage(const gr::Tensor<T>& H, std::source_location loc = std::source_location::current()) {
    if (H.rank() != 2) {
        return std::unexpected(gr::Error(std::format("hankelAverage: input must be 2D tensor, got rank {}", H.rank()), loc));
    }

    const std::size_t L = H.extent(0);
    const std::size_t K = H.extent(1);
    const std::size_t N = L + K - 1UZ;

    std::vector<T>           signal(N, T{0});
    std::vector<std::size_t> counts(N, 0UZ);

    for (std::size_t i = 0UZ; i < L; ++i) {
        for (std::size_t j = 0UZ; j < K; ++j) {
            signal[i + j] += H[i, j];
            ++counts[i + j];
        }
    }

    for (std::size_t d = 0UZ; d < N; ++d) {
        if (counts[d] > 0UZ) {
            using RealT = gr::meta::fundamental_base_value_type_t<T>;
            signal[d] /= static_cast<RealT>(counts[d]);
        }
    }
    return signal;
}

/**
 * @brief Compute Givens rotation parameters to zero out element b.
 *
 * Computes c, s, r such that:
 *   [c  s] [a]   [r]
 *   [-s c] [b] = [0]
 *
 * where c² + s² = 1 and r = sqrt(a² + b²).
 *
 * @param a First element
 * @param b Element to be zeroed
 * @param[out] c Cosine of rotation angle
 * @param[out] s Sine of rotation angle
 * @param[out] r Resulting non-zero element (hypotenuse)
 */
template<typename T>
void givens(T a, T b, T& c, T& s, T& r) noexcept {
    static_assert(std::is_floating_point_v<T>, "givens requires floating-point type");
    if (b == T{0}) {
        c = (a >= T{0}) ? T{1} : T{-1};
        s = T{0};
        r = std::abs(a);
    } else if (a == T{0}) {
        c = T{0};
        s = (b >= T{0}) ? T{1} : T{-1};
        r = std::abs(b);
    } else if (std::abs(b) > std::abs(a)) {
        const T t = a / b;
        const T u = std::copysign(std::sqrt(T{1} + t * t), b);
        s         = T{1} / u;
        c         = s * t;
        r         = b * u;
    } else {
        const T t = b / a;
        const T u = std::copysign(std::sqrt(T{1} + t * t), a);
        c         = T{1} / u;
        s         = c * t;
        r         = a * u;
    }
}

/**
 * @brief Apply Givens rotation from the left to rows i and k of matrix A.
 *
 * For each column j in [colStart, colEnd):
 *   [A[i,j]]   [c  s] [A[i,j]]
 *   [A[k,j]] = [-s c] [A[k,j]]
 *
 * @param A Matrix to modify in-place
 * @param i First row index
 * @param k Second row index
 * @param c Cosine of rotation
 * @param s Sine of rotation
 * @param colStart Starting column (inclusive)
 * @param colEnd Ending column (exclusive)
 */
template<typename T, TensorLike TensorA>
void applyGivensLeft(TensorA& A, std::size_t i, std::size_t k, T c, T s, std::size_t colStart, std::size_t colEnd) noexcept {
    using ElemT = typename tensor_traits<std::remove_cvref_t<TensorA>>::value_type;
    for (std::size_t j = colStart; j < colEnd; ++j) {
        const ElemT ai = A[i, j];
        const ElemT ak = A[k, j];
        A[i, j]        = static_cast<ElemT>(c) * ai + static_cast<ElemT>(s) * ak;
        A[k, j]        = static_cast<ElemT>(c) * ak - static_cast<ElemT>(s) * ai;
    }
}

/**
 * @brief Apply Givens rotation from the right to columns i and k of matrix A.
 *
 * For each row r in [rowStart, rowEnd):
 *   [A[r,i] A[r,k]] = [A[r,i] A[r,k]] [c -s]
 *                                      [s  c]
 *
 * @param A Matrix to modify in-place
 * @param i First column index
 * @param k Second column index
 * @param c Cosine of rotation
 * @param s Sine of rotation
 * @param rowStart Starting row (inclusive)
 * @param rowEnd Ending row (exclusive)
 */
template<typename T, TensorLike TensorA>
void applyGivensRight(TensorA& A, std::size_t i, std::size_t k, T c, T s, std::size_t rowStart, std::size_t rowEnd) noexcept {
    using ElemT = typename tensor_traits<std::remove_cvref_t<TensorA>>::value_type;
    for (std::size_t r = rowStart; r < rowEnd; ++r) {
        const ElemT ai = A[r, i];
        const ElemT ak = A[r, k];
        A[r, i]        = static_cast<ElemT>(c) * ai + static_cast<ElemT>(s) * ak;
        A[r, k]        = static_cast<ElemT>(c) * ak - static_cast<ElemT>(s) * ai;
    }
}

} // namespace gr::math

#include "math/gemm_simd.hpp"
#include "math/gemv_simd.hpp"

namespace gr::math {
/**
 * @brief General Matrix-Matrix Multiplication: C = alpha * op(A) * op(B) + beta * C
 *
 * @param C Output matrix [M x N]
 * @param A Input matrix A [M x K] or [K x M] if transposed
 * @param B Input matrix B [K x N] or [N x K] if transposed
 * @param alpha Scalar multiplier for A*B (default: 1)
 * @param beta Scalar multiplier for C (default: 0)
 *
 * @throws dimension_mismatch If dimensions are incompatible
 * @throws std::runtime_error If tensors are not contiguous
 */
template<TransposeOp TransA = TransposeOp::NoTrans, TransposeOp TransB = TransposeOp::NoTrans, ExecutionPolicy Policy, typename T, TensorOf<T> TensorC, TensorOf<T> TensorA, TensorOf<T> TensorB>
void gemm(Policy&& policy, TensorC& C, const TensorA& A, const TensorB& B, T alpha = T{1}, T beta = T{0}) {
    if (C.size() == 0 || A.size() == 0 || B.size() == 0) {
        return; // nothing to compute for empty tensors
    }
    if constexpr (CpuExecutionPolicy<Policy>) {
        detail::gemm<TransA, TransB>(policy, C, A, B, alpha, beta);
    } else if constexpr (GpuExecutionPolicy<Policy>) {
        static_assert(gr::meta::always_false<T>, "GPU GEMM not yet implemented");
    }
}

template<TransposeOp TransA = TransposeOp::NoTrans, TransposeOp TransB = TransposeOp::NoTrans, TensorLike TensorC, TensorLike TensorA, TensorLike TensorB, typename T = TensorC::value_type>
void gemm(TensorC& C, const TensorA& A, const TensorB& B, T alpha = T{1}, T beta = T{0}) { // simplified interface: C = A * B & auto-detect policy version
    gemm<TransA, TransB>(cpu_policy{}, C, A, B, alpha, beta);
}

/**
 * @brief General Matrix-Vector Multiplication: y = alpha * op(A) * x + beta * y
 *
 * @tparam TransA Transpose operation for A (compile-time)
 * @tparam Policy Execution policy
 * @tparam TensorY Output vector type
 * @tparam TensorA Input matrix type
 * @tparam TensorX Input vector type
 * @tparam T Scalar type
 *
 * @param policy Execution policy object
 * @param y Output vector [M] or [N] if A transposed
 * @param A Input matrix [M x N]
 * @param x Input vector [N] or [M] if A transposed
 * @param alpha Scalar multiplier for A*x (default: 1)
 * @param beta Scalar multiplier for y (default: 0)
 */
template<TransposeOp TransA = TransposeOp::NoTrans, ExecutionPolicy Policy, typename T, TensorOf<T> TensorY, TensorOf<T> TensorA, TensorOf<T> TensorX>
void gemv(Policy&& policy, TensorY& y, const TensorA& A, const TensorX& x, T alpha = T{1}, T beta = T{0}) {
    if (y.size() == 0 || A.size() == 0 || x.size() == 0) {
        return; // nothing to compute for empty tensors
    }
    if constexpr (CpuExecutionPolicy<Policy>) {
        detail::gemv<TransA>(policy, y, A, x, alpha, beta);
    } else if constexpr (GpuExecutionPolicy<Policy>) {
        static_assert(gr::meta::always_false<T>, "GPU GEMV not yet implemented");
    }
}

template<TransposeOp TransA = TransposeOp::NoTrans, TensorLike TensorY, TensorLike TensorA, TensorLike TensorX, typename T = TensorY::value_type>
void gemv(TensorY& y, const TensorA& A, const TensorX& x, T alpha = T{1}, T beta = T{0}) { // simplified: y = A * x && auto-detect policy version
    gemv<TransA>(cpu_policy{}, y, A, x, alpha, beta);
}

} // namespace gr::math

#endif // GNURADIO_TENSOR_OPS_HPP
