#ifndef GNURADIO_DIGITAL_ROOTRAISEDCOSINEFILTER_HPP
#define GNURADIO_DIGITAL_ROOTRAISEDCOSINEFILTER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/filter/RaisedCosine.hpp>

namespace gr::digital {

GR_REGISTER_BLOCK(gr::digital::RootRaisedCosineFilter, [T], [ float, double ])

template<std::floating_point T>
struct RootRaisedCosineFilter : Block<RootRaisedCosineFilter<T>> {
    using Description = Doc<R""(@brief Root-raised-cosine filter: the matched filter for ConstellationModulator, placed before SymbolSync.

With the same settings it has the same pulse as ConstellationModulator, and the two pulses together form a raised-cosine pulse.
At the symbol instants, the output then gives back the sent points with almost no interference from neighbouring symbols.
It also removes the noise outside (1 + excess_bandwidth) times the symbol rate and delays the signal by filter_span / 2 symbols.

 * J. G. Proakis and M. Salehi, "Digital Communications", 5th ed. New York: McGraw-Hill, 2008, ch. 9.)"">;

    static constexpr gr::Size_t kMaxSamplesPerSymbol = 1024U;
    static constexpr gr::Size_t kMaxFilterSpan       = 1024U;

    PortIn<std::complex<T>>  in;
    PortOut<std::complex<T>> out;

    Annotated<gr::Size_t, "samples per symbol", Visible, Doc<"input samples per symbol">>          samples_per_symbol = 4U;
    Annotated<float, "excess bandwidth", Visible, Doc<"roll-off of the root-raised-cosine pulse">> excess_bandwidth   = 0.35f;
    Annotated<gr::Size_t, "filter span", Doc<"pulse length in symbols">>                           filter_span        = 10U;

    GR_MAKE_REFLECTABLE(RootRaisedCosineFilter, in, out, samples_per_symbol, excess_bandwidth, filter_span);

    std::vector<T>               _taps;
    std::vector<std::complex<T>> _window;

    void start() {
        updateFromSettings();
        reset();
    }

    void reset() { std::ranges::fill(_window, std::complex<T>{}); }

    [[nodiscard]] gr::work::Status processBulk(InputViewLike auto& input, OutputViewLike auto& output) noexcept {
        const std::size_t nLead = _taps.size() - 1UZ;
        _window.resize(nLead + input.size());
        std::ranges::copy(input, _window.begin() + static_cast<std::ptrdiff_t>(nLead));
        std::ranges::fill(output, std::complex<T>{});
        for (std::size_t age = 0UZ; age < _taps.size(); ++age) {
            const std::span<const std::complex<T>> delayed{_window.data() + nLead - age, input.size()};
            std::ranges::transform(output, delayed, output.begin(), [tap = _taps[age]](std::complex<T> sum, std::complex<T> sample) { return sum + tap * sample; });
        }
        std::shift_left(_window.begin(), _window.end(), static_cast<std::ptrdiff_t>(input.size()));
        _window.resize(nLead);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const bool validSamplesPerSymbol = samples_per_symbol.value >= 2U && samples_per_symbol.value <= kMaxSamplesPerSymbol;
        const bool validFilterSpan       = filter_span.value >= 1U && filter_span.value <= kMaxFilterSpan;
        const bool validExcessBandwidth  = std::isfinite(excess_bandwidth.value) && excess_bandwidth.value > 0.f && excess_bandwidth.value <= 1.f;
        if (!validSamplesPerSymbol || !validFilterSpan || !validExcessBandwidth) {
            this->emitErrorMessage("settingsChanged()", std::format("samples_per_symbol {} must be within [2, {}], filter_span {} within [1, {}] and excess_bandwidth {} within (0, 1], keeping the previous settings", samples_per_symbol.value, kMaxSamplesPerSymbol, filter_span.value, kMaxFilterSpan, excess_bandwidth.value));
            samples_per_symbol = oldSettings.value_or<gr::Size_t>("samples_per_symbol", samples_per_symbol.value);
            filter_span        = oldSettings.value_or<gr::Size_t>("filter_span", filter_span.value);
            excess_bandwidth   = oldSettings.value_or<float>("excess_bandwidth", excess_bandwidth.value);
            return;
        }
        updateFromSettings();
    }

    void updateFromSettings() {
        _taps = gr::algorithm::filter::rootRaisedCosineTaps<T>(samples_per_symbol.value, static_cast<double>(excess_bandwidth.value), filter_span.value);
        std::vector<std::complex<T>> window(_taps.size() - 1UZ);
        const std::size_t            nKept = std::min(window.size(), _window.size());
        std::ranges::copy(std::span(_window).last(nKept), window.end() - static_cast<std::ptrdiff_t>(nKept));
        _window = std::move(window);
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_ROOTRAISEDCOSINEFILTER_HPP
