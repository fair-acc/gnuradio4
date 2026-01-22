#ifndef GNURADIO_SAVITZKY_GOLAY_FILTER_HPP
#define GNURADIO_SAVITZKY_GOLAY_FILTER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/filter/SavitzkyGolay.hpp>

namespace gr::filter {

using namespace gr;

// ============================================================================
// Streaming Savitzky-Golay Filter (scalar T samples)
// ============================================================================

GR_REGISTER_BLOCK(gr::filter::SavitzkyGolayFilter, [T], [ float, double ])

template<typename T>
struct SavitzkyGolayFilter : Block<SavitzkyGolayFilter<T>> {
    using Block<SavitzkyGolayFilter<T>>::Block;
    static_assert(std::floating_point<T>, "T must be floating_point");

    using Description = Doc<R""(@brief Savitzky-Golay streaming filter

Applies local polynomial smoothing/differentiation to streaming scalar data.
Filter coefficients are computed using SVD-based least-squares fitting.

Alignment modes:
- Centred: symmetric window, linear-phase, group delay = (window_size-1)/2 samples
- Causal: past-only window, minimal latency, non-linear phase
)"">; // clang-format off

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "window size", Doc<"filter window size (samples, must be >= poly_order+1)">>  window_size  = 11U;
    Annotated<gr::Size_t, "polynomial order", Doc<"order of fitting polynomial">>                       poly_order   = 4U;
    Annotated<gr::Size_t, "derivative order", Doc<"derivative order (0=smooth, 1=1st deriv, ...)">>     deriv_order  = 0U;
    Annotated<float, "sample rate", Doc<"input sample rate for derivative scaling">, Unit<"Hz">>        sample_rate  = 1.0f;
    Annotated<std::string, "alignment", Doc<"Centred or Causal">>                                       alignment    = std::string("Centred");

    GR_MAKE_REFLECTABLE(SavitzkyGolayFilter, in, out, window_size, poly_order, deriv_order, sample_rate, alignment);

private:
    algorithm::savitzky_golay::SavitzkyGolayFilter<T> _state;

    [[nodiscard]] algorithm::savitzky_golay::Config<T> buildConfig() const {
        const T sampleRateT = static_cast<T>(sample_rate.value);
        return {
            .derivOrder = static_cast<std::size_t>(deriv_order),
            .delta      = (sampleRateT > T{0}) ? T{1} / sampleRateT : T{1},
            .alignment  = (alignment.value == "Causal")
                ? algorithm::savitzky_golay::Alignment::Causal
                : algorithm::savitzky_golay::Alignment::Centred
        };
    }

public:
    void start() {
        _state = algorithm::savitzky_golay::SavitzkyGolayFilter<T>(
            static_cast<std::size_t>(window_size),
            static_cast<std::size_t>(poly_order),
            buildConfig());
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window_size") || newSettings.contains("poly_order") ||
            newSettings.contains("deriv_order") || newSettings.contains("sample_rate") ||
            newSettings.contains("alignment")) {
            _state.setParameters(
                static_cast<std::size_t>(window_size),
                static_cast<std::size_t>(poly_order),
                buildConfig());
        }
    }

    void reset() { _state.reset(); }

    [[nodiscard]] constexpr T processOne(T input) noexcept { return _state.processOne(input); }
};

// ============================================================================
// DataSet Savitzky-Golay Filter (zero-phase batch processing)
// ============================================================================

GR_REGISTER_BLOCK(gr::filter::SavitzkyGolayDataSetFilter, [T], [ float, double ])

template<typename T>
struct SavitzkyGolayDataSetFilter : Block<SavitzkyGolayDataSetFilter<T>> {
    using Block<SavitzkyGolayDataSetFilter<T>>::Block;
    static_assert(std::floating_point<T>, "T must be floating_point");

    using Description = Doc<R""(@brief Savitzky-Golay filter for DataSet (zero-phase)

Applies zero-phase Savitzky-Golay filtering to DataSet signal_values using
forward-backward filtering. This eliminates phase distortion and preserves
peak positions - essential for spectroscopy applications.

Note: zero-phase filtering squares the magnitude response, providing
additional smoothing compared to single-pass filtering.

Boundary policy:
- Reflect: mirror indices at boundaries (default, continuous derivatives)
- Replicate: extend edge values
)"">; // clang-format off

    PortIn<DataSet<T>>  in;
    PortOut<DataSet<T>> out;

    Annotated<gr::Size_t, "window size", Doc<"filter window size (samples, must be >= poly_order+1)">>  window_size     = 11U;
    Annotated<gr::Size_t, "polynomial order", Doc<"order of fitting polynomial">>                       poly_order      = 4U;
    Annotated<gr::Size_t, "derivative order", Doc<"derivative order (0=smooth, 1=1st deriv, ...)>">>    deriv_order     = 0U;
    Annotated<std::string, "boundary policy", Doc<"Reflect or Replicate">>                              boundary_policy = std::string("Reflect");

    GR_MAKE_REFLECTABLE(SavitzkyGolayDataSetFilter, in, out, window_size, poly_order, deriv_order, boundary_policy);

private:
    std::vector<T> _coeffs;

    [[nodiscard]] algorithm::savitzky_golay::Config<T> buildConfig() const {
        return {
            .derivOrder     = static_cast<std::size_t>(deriv_order),
            .delta          = T{1},
            .alignment      = algorithm::savitzky_golay::Alignment::Centred,
            .boundaryPolicy = (boundary_policy.value == "Replicate")
                ? algorithm::savitzky_golay::BoundaryPolicy::Replicate
                : algorithm::savitzky_golay::BoundaryPolicy::Reflect
        };
    }

    void updateCoefficients() {
        _coeffs = algorithm::savitzky_golay::computeCoefficients<T>(
            static_cast<std::size_t>(window_size),
            static_cast<std::size_t>(poly_order),
            buildConfig());
    }

public:
    void start() { updateCoefficients(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window_size") || newSettings.contains("poly_order") ||
            newSettings.contains("deriv_order") || newSettings.contains("boundary_policy")) {
            updateCoefficients();
        }
    }

    [[nodiscard]] DataSet<T> processOne(DataSet<T> input) {
        if (input.signal_values.empty() || _coeffs.empty()) {
            return input;
        }

        std::vector<T> filtered(input.signal_values.size());
        algorithm::savitzky_golay::applyZeroPhase<T>(
            std::span<const T>(input.signal_values),
            std::span<T>(filtered),
            std::span<const T>(_coeffs),
            buildConfig());

        input.signal_values = std::move(filtered);
        return input;
    }
};

} // namespace gr::filter

#endif // GNURADIO_SAVITZKY_GOLAY_FILTER_HPP
