#ifndef GNURADIO_SAVITZKY_GOLAY_HPP
#define GNURADIO_SAVITZKY_GOLAY_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TensorMath.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm::savitzky_golay {

enum class Alignment : std::uint8_t {
    Centred, // symmetric window, linear-phase, group delay D=(W-1)/2
    Causal   // past-only window, zero-latency, non-linear phase
};

enum class BoundaryPolicy : std::uint8_t {
    Default,       // pre-fill history with config.defaultValue
    ZeroOrderHold, // fill history with first sample (lazy)
    Reflect,       // batch: reflect at boundaries; streaming: same as ZeroOrderHold
    Replicate      // batch: replicate edges; streaming: same as ZeroOrderHold
};

template<typename T>
struct Config {
    static_assert(std::floating_point<T>, "Config requires a floating-point type");
    std::size_t    derivOrder     = 0UZ;                     /// derivative order (0 = smoothing)
    T              delta          = T{1};                    /// sample spacing (1/sample_rate)
    Alignment      alignment      = Alignment::Centred;      /// Centred or Causal
    BoundaryPolicy boundaryPolicy = BoundaryPolicy::Default; /// initialization / boundary handling
    T              defaultValue   = T{0};                    /// fill value for BoundaryPolicy::Default
};

namespace detail {

[[nodiscard]] constexpr std::size_t factorial(std::size_t n) noexcept {
    std::size_t result = 1UZ;
    for (std::size_t i = 2UZ; i <= n; ++i) {
        result *= i;
    }
    return result;
}

[[nodiscard]] constexpr std::size_t reflectIndex(std::ptrdiff_t i, std::size_t N) noexcept {
    if (N == 0UZ) {
        return 0UZ;
    }
    if (i < 0) {
        i = -i - 1;
    }
    const auto period = static_cast<std::ptrdiff_t>(2UZ * N);
    i                 = i % period;
    if (i >= static_cast<std::ptrdiff_t>(N)) {
        i = period - i - 1;
    }
    return static_cast<std::size_t>(i);
}

[[nodiscard]] constexpr std::size_t replicateIndex(std::ptrdiff_t i, std::size_t N) noexcept {
    if (N == 0UZ) {
        return 0UZ;
    }
    if (i < 0) {
        return 0UZ;
    }
    if (i >= static_cast<std::ptrdiff_t>(N)) {
        return N - 1UZ;
    }
    return static_cast<std::size_t>(i);
}

template<typename T>
[[nodiscard]] constexpr std::size_t boundaryIndex(std::ptrdiff_t i, std::size_t N, const Config<T>& config) noexcept {
    switch (config.boundaryPolicy) {
    case BoundaryPolicy::Reflect: return reflectIndex(i, N);
    case BoundaryPolicy::Replicate: return replicateIndex(i, N);
    default: return replicateIndex(i, N);
    }
}

} // namespace detail

/**
 * @brief Compute Savitzky-Golay filter coefficients using SVD pseudoinverse.
 */
template<std::floating_point T>
[[nodiscard]] std::vector<T> computeCoefficients(std::size_t windowSize, std::size_t polyOrder, const Config<T>& config) {
    windowSize      = std::max(windowSize, 1UZ);
    polyOrder       = std::min(polyOrder, windowSize - 1UZ);
    auto derivOrder = std::min(config.derivOrder, polyOrder);
    auto delta      = std::max(config.delta, std::numeric_limits<T>::epsilon());

    const std::size_t W = windowSize;
    const std::size_t p = polyOrder;
    const std::size_t d = derivOrder;

    // build Vandermonde matrix A[i,j] = t_i^j
    const auto D = (config.alignment == Alignment::Centred) ? static_cast<std::ptrdiff_t>((W - 1UZ) / 2UZ) : static_cast<std::ptrdiff_t>(W - 1UZ);

    Tensor<T> A({W, p + 1UZ});
    for (std::size_t i = 0UZ; i < W; ++i) {
        const T t     = static_cast<T>(static_cast<std::ptrdiff_t>(i) - D);
        T       power = T{1};
        for (std::size_t j = 0UZ; j <= p; ++j) {
            A[i, j] = power;
            power *= t;
        }
    }

    // compute pseudoinverse via SVD: A^+ = V * S^{-1} * U^T
    Tensor<T> U, V, S;
    auto      status = gr::math::gesvd(U, S, V, A);
    if (status != gr::math::svd::Status::Success && status != gr::math::svd::Status::EarlyReturn) {
        return std::vector<T>(W, T{1} / static_cast<T>(W)); // fallback: uniform
    }

    const std::size_t rank = std::min({U.extent(1), S.size(), V.extent(1)});
    Tensor<T>         Apinv({p + 1UZ, W});
    Apinv.fill(T{0});

    const T tolerance = std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(W, p + 1UZ)) * (S.size() > 0UZ ? std::abs(S[0]) : T{1});

    for (std::size_t k = 0UZ; k < rank; ++k) {
        if (std::abs(S[k]) < tolerance) {
            continue;
        }
        const T sinv = T{1} / S[k];
        for (std::size_t i = 0UZ; i < p + 1UZ; ++i) {
            for (std::size_t j = 0UZ; j < W; ++j) {
                Apinv[i, j] += V[i, k] * sinv * U[j, k];
            }
        }
    }

    std::vector<T> coeffs(W);
    for (std::size_t j = 0UZ; j < W; ++j) {
        coeffs[j] = Apinv[d, j];
    }

    const T scale = static_cast<T>(detail::factorial(d)) / std::pow(delta, static_cast<T>(d));
    std::ranges::for_each(coeffs, [scale](T& c) { c *= scale; });

    return coeffs;
}

/**
 * @brief Apply Savitzky-Golay filter to a signal (batch processing).
 */
template<typename T>
void apply(std::span<const T> in, std::span<T> out, std::span<const T> coeffs, const Config<T>& config) {
    const std::size_t N = in.size();
    const std::size_t W = coeffs.size();
    if (N == 0UZ || W == 0UZ || out.size() < N) {
        return;
    }

    const auto D = (config.alignment == Alignment::Centred) ? static_cast<std::ptrdiff_t>((W - 1UZ) / 2UZ) : static_cast<std::ptrdiff_t>(W - 1UZ);

    for (std::size_t n = 0UZ; n < N; ++n) {
        out[n] = std::transform_reduce(coeffs.begin(), coeffs.end(), std::views::iota(0UZ, W).begin(), T{0}, std::plus<>{}, [&](T c, std::size_t k) {
            const auto idx = static_cast<std::ptrdiff_t>(n) + static_cast<std::ptrdiff_t>(k) - D;
            return c * in[detail::boundaryIndex(idx, N, config)];
        });
    }
}

/**
 * @brief Apply Savitzky-Golay filter with zero-phase (forward-backward) filtering.
 */
template<typename T>
void applyZeroPhase(std::span<const T> in, std::span<T> out, std::span<const T> coeffs, const Config<T>& config) {
    const std::size_t N = in.size();
    if (N == 0UZ || coeffs.empty() || out.size() < N) {
        return;
    }

    std::vector<T> temp(N), temp2(N);
    Config<T>      centredConfig = config;
    centredConfig.alignment      = Alignment::Centred;

    apply(in, std::span<T>(temp), coeffs, centredConfig);
    std::ranges::reverse(temp);
    apply(std::span<const T>(temp), std::span<T>(temp2), coeffs, centredConfig);
    std::ranges::reverse(temp2);
    std::ranges::copy(temp2, out.begin());
}

/**
 * @brief Streaming Savitzky-Golay filter.
 */
template<typename T>
class SavitzkyGolayFilter {
public:
    using value_type = T;

private:
    std::size_t          _windowSize;
    std::size_t          _polyOrder;
    Config<T>            _config;
    std::vector<T>       _coeffs;
    gr::HistoryBuffer<T> _history;
    bool                 _initialized{false};

    [[nodiscard]] bool needsLazyInit() const noexcept { return _config.boundaryPolicy != BoundaryPolicy::Default; }

    void fillHistory(T value) {
        for (std::size_t i = 0UZ; i < _windowSize; ++i) {
            _history.push_back(value);
        }
    }

public:
    explicit SavitzkyGolayFilter(std::size_t windowSize = 11UZ, std::size_t polyOrder = 4UZ, const Config<T>& config = {}) : _windowSize(std::max(windowSize, 1UZ)), _polyOrder(std::min(polyOrder, _windowSize - 1UZ)), _config(config), _coeffs(computeCoefficients<T>(_windowSize, _polyOrder, _config)), _history(_windowSize), _initialized(!needsLazyInit()) {
        if (_config.boundaryPolicy == BoundaryPolicy::Default) {
            fillHistory(_config.defaultValue);
        }
    }

    [[nodiscard]] T processOne(T input) {
        if (!_initialized) {
            fillHistory(input);
            _initialized = true;
        }
        _history.push_back(input);
        return std::transform_reduce(_history.begin(), _history.end(), _coeffs.begin(), T{0});
    }

    void reset() {
        _history.reset();
        _initialized = !needsLazyInit();
        if (_config.boundaryPolicy == BoundaryPolicy::Default) {
            fillHistory(_config.defaultValue);
        }
    }

    [[nodiscard]] std::size_t        windowSize() const noexcept { return _windowSize; }
    [[nodiscard]] std::size_t        polyOrder() const noexcept { return _polyOrder; }
    [[nodiscard]] std::size_t        derivOrder() const noexcept { return _config.derivOrder; }
    [[nodiscard]] T                  delta() const noexcept { return _config.delta; }
    [[nodiscard]] Alignment          alignment() const noexcept { return _config.alignment; }
    [[nodiscard]] BoundaryPolicy     boundaryPolicy() const noexcept { return _config.boundaryPolicy; }
    [[nodiscard]] const Config<T>&   config() const noexcept { return _config; }
    [[nodiscard]] std::span<const T> coefficients() const noexcept { return _coeffs; }
    [[nodiscard]] std::size_t        delay() const noexcept { return (_config.alignment == Alignment::Centred) ? (_windowSize - 1UZ) / 2UZ : 0UZ; }

    void setParameters(std::size_t windowSize, std::size_t polyOrder, const Config<T>& config = {}) {
        _windowSize = std::max(windowSize, 1UZ);
        _polyOrder  = std::min(polyOrder, _windowSize - 1UZ);
        _config     = config;
        _coeffs     = computeCoefficients<T>(_windowSize, _polyOrder, _config);
        _history.resize(_windowSize);
        reset();
    }

    void setConfig(const Config<T>& config) {
        _config = config;
        _coeffs = computeCoefficients<T>(_windowSize, _polyOrder, _config);
        reset();
    }
};

} // namespace gr::algorithm::savitzky_golay

#endif // GNURADIO_SAVITZKY_GOLAY_HPP
