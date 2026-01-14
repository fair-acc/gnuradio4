#ifndef GNURADIO_SVD_DENOISER_HPP
#define GNURADIO_SVD_DENOISER_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/SvdFilter.hpp>

namespace gr::filter {

using namespace gr;

GR_REGISTER_BLOCK(gr::filter::SvdDenoiser, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct SvdDenoiser : Block<SvdDenoiser<T>> {
    using Block<SvdDenoiser<T>>::Block;
    static_assert(std::floating_point<T> || gr::meta::complex_like<T>, "T must be floating_point or complex_like");

    using Description = Doc<R""(@brief SVD-based signal denoiser

Denoises signals using Singular Value Decomposition applied to a Hankel matrix
representation. Effective for removing broadband noise from periodic or
quasi-periodic signals while preserving signal structure.

Singular values are kept if ALL criteria are satisfied:
- count ≤ max_rank
- σ_i / σ_0 ≥ relative_threshold
- σ_i ≥ absolute_threshold
- cumulative energy ≤ energy_fraction × total energy
)"">; // clang-format off

    using RealT = gr::meta::fundamental_base_value_type_t<T>;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "window size", Doc<"analysis window size (samples)">>                   window_size = 64U;
    Annotated<gr::Size_t, "Hankel rows", Doc<"number of Hankel matrix rows (0 = window_size/2)">> hankel_rows = 0U;
    Annotated<gr::Size_t, "max rank", Doc<"maximum singular values to keep (max = no limit)">>    max_rank    = std::numeric_limits<gr::Size_t>::max();

    Annotated<RealT, "relative threshold", Doc<"minimum ratio σ_i/σ_0 to keep">, Unit<"ratio">>
        relative_threshold = std::numeric_limits<RealT>::epsilon();

    Annotated<RealT, "absolute threshold", Doc<"minimum absolute value of σ_i to keep">>
        absolute_threshold = std::numeric_limits<RealT>::epsilon();

    Annotated<RealT, "energy fraction", Doc<"fraction of total energy to retain (1.0 = all)">, Unit<"ratio">>
        energy_fraction = RealT{1};

    Annotated<RealT, "hop fraction", Doc<"SVD recomputation interval as fraction of window_size">, Unit<"ratio">>
        hop_fraction = RealT{0.25};

    GR_MAKE_REFLECTABLE(SvdDenoiser, in, out, window_size, hankel_rows, max_rank, relative_threshold, absolute_threshold, energy_fraction, hop_fraction);

private:
    algorithm::svd_filter::SvdDenoiser<T> _state;

    [[nodiscard]] algorithm::svd_filter::Config<RealT> buildConfig() const {
        return {
            .maxRank           = static_cast<std::size_t>(max_rank),
            .relativeThreshold = relative_threshold,
            .absoluteThreshold = absolute_threshold,
            .energyFraction    = energy_fraction,
            .hopFraction       = hop_fraction
        };
    }

public:
    void start() {
        _state = algorithm::svd_filter::SvdDenoiser<T>(
            static_cast<std::size_t>(window_size),
            static_cast<std::size_t>(hankel_rows),
            buildConfig());
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("window_size") || newSettings.contains("hankel_rows") ||
            newSettings.contains("max_rank") || newSettings.contains("relative_threshold") ||
            newSettings.contains("absolute_threshold") || newSettings.contains("energy_fraction") ||
            newSettings.contains("hop_fraction")) {
            _state.setParameters(
                static_cast<std::size_t>(window_size),
                static_cast<std::size_t>(hankel_rows),
                buildConfig());
        }
    }

    void reset() { _state.reset(); }

    [[nodiscard]] constexpr T processOne(T input) noexcept { return _state.processOne(input); }
};

} // namespace gr::filter

#endif // GNURADIO_SVD_DENOISER_HPP
