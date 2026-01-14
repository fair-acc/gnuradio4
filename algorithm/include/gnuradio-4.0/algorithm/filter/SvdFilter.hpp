#ifndef GNURADIO_SVD_FILTER_HPP
#define GNURADIO_SVD_FILTER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/SVD.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/TensorMath.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm::svd_filter {

enum class BoundaryPolicy : std::uint8_t {
    Default,       /// pre-fill history with config.defaultValue
    ZeroOrderHold, /// fill history with first sample (lazy)
    Reflect,       /// streaming: same as ZeroOrderHold
    Replicate      /// streaming: same as ZeroOrderHold
};

template<typename T>
struct Config {
    static_assert(std::floating_point<T>, "Config requires a floating-point type");

    std::size_t    maxRank           = std::numeric_limits<std::size_t>::max(); /// maximum singular values to keep
    T              relativeThreshold = std::numeric_limits<T>::epsilon();       /// minimum ratio σ_i/σ_0 to keep
    T              absoluteThreshold = std::numeric_limits<T>::epsilon();       /// minimum absolute σ_i to keep
    T              energyFraction    = T{1};                                    /// fraction of total energy to retain (e.g. 0.95)
    T              hopFraction       = T{0.25};                                 /// SVD recomputation interval as fraction of windowSize
    BoundaryPolicy boundaryPolicy    = BoundaryPolicy::Default;                 /// initialisation / boundary handling
    T              defaultValue      = T{0};                                    /// fill value for BoundaryPolicy::Default
};

namespace detail {

template<typename T>
[[nodiscard]] std::size_t computeEffectiveRank(std::span<const T> singularValues, const Config<T>& config) {
    if (singularValues.empty()) {
        return 0UZ;
    }

    const T     sigma0       = singularValues[0];
    const T     totalEnergy  = std::transform_reduce(singularValues.begin(), singularValues.end(), T{0}, std::plus<>{}, [](T s) { return s * s; });
    const T     energyCutoff = config.energyFraction * totalEnergy;
    T           cumEnergy    = T{0};
    std::size_t rank         = 0UZ;

    for (const T sigma : singularValues) {
        if (rank >= config.maxRank || sigma / sigma0 < config.relativeThreshold || sigma < config.absoluteThreshold) {
            break;
        }
        cumEnergy += sigma * sigma;
        ++rank;
        if (cumEnergy >= energyCutoff) {
            break;
        }
    }
    return std::max(rank, 1UZ);
}

} // namespace detail

/**
 * @brief Low-rank approximation via truncated SVD: A ≈ U_k × diag(S_k) × V_k^H
 */
template<typename T>
[[nodiscard]] Tensor<T> lowRankApproximation(const Tensor<T>& A, const Config<gr::meta::fundamental_base_value_type_t<T>>& config = {}) {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;

    if (A.rank() != 2) {
        throw std::invalid_argument("Input must be a 2D tensor");
    }

    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);

    Tensor<T>     U, V;
    Tensor<RealT> S;
    if (auto status = gr::math::gesvd(U, S, V, A); status != gr::math::svd::Status::Success && status != gr::math::svd::Status::EarlyReturn) {
        throw std::runtime_error("SVD computation failed");
    }

    const std::size_t k = std::min(detail::computeEffectiveRank(std::span<const RealT>(S.data(), S.size()), config), std::min(m, n));

    // US = U[:,:k] × diag(S[:k])
    Tensor<T> US({m, k});
    for (std::size_t i = 0UZ; i < m; ++i) {
        for (std::size_t j = 0UZ; j < k; ++j) {
            US[i, j] = U[i, j] * static_cast<T>(S[j]);
        }
    }

    // VkH = V[:,:k]^H
    Tensor<T> VkH({k, n});
    for (std::size_t i = 0UZ; i < k; ++i) {
        for (std::size_t j = 0UZ; j < n; ++j) {
            if constexpr (gr::meta::complex_like<T>) {
                VkH[i, j] = std::conj(V[j, i]);
            } else {
                VkH[i, j] = V[j, i];
            }
        }
    }

    Tensor<T> result({m, n});
    result.fill(T{0});
    gr::math::gemm(result, US, VkH);
    return result;
}

/**
 * @brief Denoise signal window using Hankel-SVD method.
 */
template<typename T>
[[nodiscard]] std::vector<T> denoiseWindow(std::span<const T> signal, std::size_t hankelRows = 0UZ, const Config<gr::meta::fundamental_base_value_type_t<T>>& config = {}) {
    const std::size_t N = signal.size();
    if (N < 2UZ) {
        return std::vector<T>(signal.begin(), signal.end());
    }

    hankelRows = (hankelRows == 0UZ) ? N / 2UZ : std::clamp(hankelRows, 1UZ, N);

    auto H        = gr::math::hankel(signal, hankelRows).value();
    auto H_approx = lowRankApproximation(H, config);
    return gr::math::hankelAverage(H_approx).value();
}

/**
 * @brief Streaming SVD denoiser with overlap-save caching.
 *
 * Maintains a sliding window using HistoryBuffer and outputs denoised samples
 * one at a time. Group delay = (windowSize-1)/2 samples.
 *
 * SVD is recomputed every hopSize samples (default: windowSize/4) and results
 * are cached for efficiency.
 */
template<typename T>
class SvdDenoiser {
public:
    using value_type = T;
    using RealT      = gr::meta::fundamental_base_value_type_t<T>;

private:
    std::size_t          _windowSize;
    std::size_t          _hankelRows;
    std::size_t          _hopSize;
    Config<RealT>        _config;
    gr::HistoryBuffer<T> _history;
    std::vector<T>       _outputCache;
    std::size_t          _cacheIndex{0UZ};
    bool                 _initialized{false};

    [[nodiscard]] bool needsLazyInit() const noexcept { return _config.boundaryPolicy != BoundaryPolicy::Default; }

    void fillHistory(T value) {
        for (std::size_t i = 0UZ; i < _windowSize; ++i) {
            (void)i;
            _history.push_back(value);
        }
    }

    void computeSvdAndCache() {
        auto historySpan = _history.get_span(0UZ, _windowSize);
        auto H           = gr::math::hankel(historySpan, _hankelRows).value();
        auto H_approx    = lowRankApproximation(H, _config);
        auto denoised    = gr::math::hankelAverage(H_approx).value();

        const std::size_t delay    = (_windowSize - 1UZ) / 2UZ;
        const std::size_t startIdx = (_windowSize > delay) ? (_windowSize - 1UZ - delay) : 0UZ;
        const std::size_t safeIdx  = std::min(startIdx, _windowSize > _hopSize ? _windowSize - _hopSize : 0UZ);

        _outputCache.assign(denoised.begin() + static_cast<std::ptrdiff_t>(safeIdx), denoised.begin() + static_cast<std::ptrdiff_t>(safeIdx + _hopSize));
        _cacheIndex = 0UZ;
    }

public:
    explicit SvdDenoiser(std::size_t windowSize = 64UZ, std::size_t hankelRows = 0UZ, const Config<RealT>& config = {}) : _windowSize(std::max(windowSize, 2UZ)), _hankelRows(hankelRows == 0UZ ? _windowSize / 2UZ : hankelRows), _hopSize(std::max(1UZ, static_cast<std::size_t>(static_cast<RealT>(_windowSize) * config.hopFraction))), _config(config), _history(_windowSize), _initialized(!needsLazyInit()) {
        _outputCache.reserve(_hopSize);
        if (_config.boundaryPolicy == BoundaryPolicy::Default) {
            fillHistory(static_cast<T>(_config.defaultValue));
        }
    }

    [[nodiscard]] T processOne(T input) {
        if (!_initialized) {
            fillHistory(input);
            _initialized = true;
        }

        _history.push_back(input);

        if (_cacheIndex >= _outputCache.size()) {
            computeSvdAndCache();
        }
        return _outputCache[_cacheIndex++];
    }

    void reset() {
        _history.reset();
        _outputCache.clear();
        _cacheIndex  = 0UZ;
        _initialized = !needsLazyInit();
        if (_config.boundaryPolicy == BoundaryPolicy::Default) {
            fillHistory(static_cast<T>(_config.defaultValue));
        }
    }

    [[nodiscard]] std::size_t          windowSize() const noexcept { return _windowSize; }
    [[nodiscard]] std::size_t          hankelRows() const noexcept { return _hankelRows; }
    [[nodiscard]] std::size_t          hopSize() const noexcept { return _hopSize; }
    [[nodiscard]] std::size_t          delay() const noexcept { return (_windowSize - 1UZ) / 2UZ; }
    [[nodiscard]] const Config<RealT>& config() const noexcept { return _config; }
    [[nodiscard]] BoundaryPolicy       boundaryPolicy() const noexcept { return _config.boundaryPolicy; }

    void setParameters(std::size_t windowSize, std::size_t hankelRows = 0UZ, const Config<RealT>& config = {}) {
        _windowSize = std::max(windowSize, 2UZ);
        _hankelRows = hankelRows == 0UZ ? _windowSize / 2UZ : hankelRows;
        _hopSize    = std::max(1UZ, static_cast<std::size_t>(static_cast<RealT>(_windowSize) * config.hopFraction));
        _config     = config;
        _history.resize(_windowSize);
        _outputCache.reserve(_hopSize);
        reset();
    }

    void setConfig(const Config<RealT>& config) {
        _config  = config;
        _hopSize = std::max(1UZ, static_cast<std::size_t>(static_cast<RealT>(_windowSize) * _config.hopFraction));
        reset();
    }
};

} // namespace gr::algorithm::svd_filter

#endif // GNURADIO_SVD_FILTER_HPP
