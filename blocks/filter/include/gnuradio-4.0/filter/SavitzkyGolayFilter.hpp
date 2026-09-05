#ifndef GNURADIO_SAVITZKY_GOLAY_FILTER_HPP
#define GNURADIO_SAVITZKY_GOLAY_FILTER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/algorithm/filter/SavitzkyGolay.hpp>

namespace gr::filter {

using namespace gr;

GR_REGISTER_BLOCK(gr::filter::SavitzkyGolayFilter, [T], [ float, double ])

/// applies local polynomial smoothing/differentiation to streaming scalar data via SVD-based least-squares
/// fitted coefficients. `alignment` trades group delay for phase: `Centred` is symmetric with a delay of
/// (window_size-1)/2 samples, `Causal` is past-only with minimal latency but non-linear phase.
template<typename T>
struct SavitzkyGolayFilter : Block<SavitzkyGolayFilter<T>> {
    using Block<SavitzkyGolayFilter<T>>::Block;
    static_assert(std::floating_point<T>, "T must be floating_point");

    using Description = Doc<"local polynomial smoothing/differentiation filter for streaming scalar data">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "window size", Doc<"filter window size (samples, must be >= poly_order+1)">>                                                          window_size = 11U;
    Annotated<gr::Size_t, "polynomial order", Doc<"order of fitting polynomial">>                                                                               poly_order  = 4U;
    Annotated<gr::Size_t, "derivative order", Doc<"derivative order (0=smooth, 1=1st deriv, ...)">>                                                             deriv_order = 0U;
    Annotated<float, "sample rate", Doc<"input sample rate for derivative scaling">, Unit<"Hz">>                                                                sample_rate = 1.0f;
    Annotated<algorithm::savitzky_golay::Alignment, "alignment", Doc<"Centred: symmetric, linear-phase, group delay (W-1)/2; Causal: past-only, zero-latency">> alignment   = algorithm::savitzky_golay::Alignment::Centred;

    GR_MAKE_REFLECTABLE(SavitzkyGolayFilter, in, out, window_size, poly_order, deriv_order, sample_rate, alignment);

private:
    algorithm::savitzky_golay::SavitzkyGolayFilter<T> _state;

    [[nodiscard]] algorithm::savitzky_golay::Config<T> buildConfig() const {
        const T sampleRateT = static_cast<T>(sample_rate.value);
        return {.derivOrder = static_cast<std::size_t>(deriv_order), .delta = (sampleRateT > T{0}) ? T{1} / sampleRateT : T{1}, .alignment = alignment};
    }

public:
    void start() { _state = algorithm::savitzky_golay::SavitzkyGolayFilter<T>(static_cast<std::size_t>(window_size), static_cast<std::size_t>(poly_order), buildConfig()); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window_size") || newSettings.contains("poly_order") || newSettings.contains("deriv_order") || newSettings.contains("sample_rate") || newSettings.contains("alignment")) {
            _state.setParameters(static_cast<std::size_t>(window_size), static_cast<std::size_t>(poly_order), buildConfig());
        }
    }

    void reset() { _state.reset(); }

    [[nodiscard]] constexpr T processOne(T input) noexcept { return _state.processOne(input); }
};

GR_REGISTER_BLOCK(gr::filter::SavitzkyGolayDataSetFilter, [T], [ float, double ])

/// applies zero-phase Savitzky-Golay filtering to a DataSet's `signal_values` by forward-backward filtering,
/// eliminating phase distortion and preserving peak positions. `boundary_policy` fills the history a window
/// needs at a record's edges: `Reflect` mirrors indices there (default), `Replicate` extends the edge value.
template<typename T>
struct SavitzkyGolayDataSetFilter : Block<SavitzkyGolayDataSetFilter<T>> {
    using Block<SavitzkyGolayDataSetFilter<T>>::Block;
    static_assert(std::floating_point<T>, "T must be floating_point");

    using Description = Doc<"zero-phase Savitzky-Golay filter for a DataSet's signal_values">;

    PortIn<DataSet<T>>  in;
    PortOut<DataSet<T>> out;

    Annotated<gr::Size_t, "window size", Doc<"filter window size (samples, must be >= poly_order+1)">>                            window_size     = 11U;
    Annotated<gr::Size_t, "polynomial order", Doc<"order of fitting polynomial">>                                                 poly_order      = 4U;
    Annotated<gr::Size_t, "derivative order", Doc<"derivative order (0=smooth, 1=1st deriv, ...)>">>                              deriv_order     = 0U;
    Annotated<algorithm::savitzky_golay::BoundaryPolicy, "boundary policy", Doc<"how the history is filled at a record's edges">> boundary_policy = algorithm::savitzky_golay::BoundaryPolicy::Reflect;

    GR_MAKE_REFLECTABLE(SavitzkyGolayDataSetFilter, in, out, window_size, poly_order, deriv_order, boundary_policy);

private:
    std::vector<T> _coeffs;

    [[nodiscard]] algorithm::savitzky_golay::Config<T> buildConfig() const { return {.derivOrder = static_cast<std::size_t>(deriv_order), .delta = T{1}, .alignment = algorithm::savitzky_golay::Alignment::Centred, .boundaryPolicy = boundary_policy}; }

    void updateCoefficients() { _coeffs = algorithm::savitzky_golay::computeCoefficients<T>(static_cast<std::size_t>(window_size), static_cast<std::size_t>(poly_order), buildConfig()); }

public:
    void start() { updateCoefficients(); }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window_size") || newSettings.contains("poly_order") || newSettings.contains("deriv_order") || newSettings.contains("boundary_policy")) {
            updateCoefficients();
        }
    }

    [[nodiscard]] DataSet<T> processOne(DataSet<T> input) {
        if (input.signal_values.empty() || _coeffs.empty()) {
            return input;
        }

        std::vector<T> filtered(input.signal_values.size());
        algorithm::savitzky_golay::applyZeroPhase<T>(std::span<const T>(input.signal_values), std::span<T>(filtered), std::span<const T>(_coeffs), buildConfig());

        input.signal_values = std::move(filtered);
        return input;
    }
};

} // namespace gr::filter

#endif // GNURADIO_SAVITZKY_GOLAY_FILTER_HPP
