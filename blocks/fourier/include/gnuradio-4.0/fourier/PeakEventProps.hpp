#ifndef GNURADIO_PEAK_EVENT_PROPS_HPP
#define GNURADIO_PEAK_EVENT_PROPS_HPP

#include <memory_resource>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/algorithm/peak/Peak.hpp>

namespace gr::blocks::fourier {

/**
 * The event keys both detectors publish. ex06 scores them against each other, so they must agree
 * key for key; keeping one definition is what stops them drifting apart. `positionScale` maps model
 * bins onto input bins for a resampling detector and is 1 for one that works in input bins.
 */
[[nodiscard]] inline gr::property_map peakEventProps(const gr::algorithm::peak::PeakResult& peak, const gr::algorithm::peak::NoiseEstimate& noise, float positionScale = 1.f) {
    return {
        {std::pmr::string("confidence"), gr::pmt::Value(peak.confidence)},
        /**
         * the timing_events index is an integer bin, so the fractional centre must be carried
         * separately or sub-bin accuracy is silently truncated to +-0.5 bin
         */
        {std::pmr::string("centre"), gr::pmt::Value(peak.centre * positionScale)},
        {std::pmr::string("fwhm"), gr::pmt::Value(peak.fwhm() * positionScale)},
        {std::pmr::string("hwhm_l"), gr::pmt::Value(peak.hwhmLeft * positionScale)},
        {std::pmr::string("hwhm_r"), gr::pmt::Value(peak.hwhmRight * positionScale)},
        /**
         * info only, so a consumer can guess the shape; evaluation compares the half-maximum widths.
         * Distinct from the generator's "shape_scale", which is that shape's own parameter (sigma for
         * a Gaussian, gamma for a Lorentzian) rather than a Gaussian-equivalent one.
         */
        {std::pmr::string("gaussian_equivalent_sigma"), gr::pmt::Value(peak.gaussianEquivalentSigma() * positionScale)},
        {std::pmr::string("asymmetry"), gr::pmt::Value(gr::algorithm::peak::asymmetryOf(peak))},
        {std::pmr::string("amplitude"), gr::pmt::Value(peak.amplitude)},
        {std::pmr::string("amplitude_measured"), gr::pmt::Value(peak.amplitudeMeasured)},
        {std::pmr::string("prominence"), gr::pmt::Value(peak.prominence)},
        /**
         * provenance: 0 for a plain single-stage detection, otherwise model-defined (the cascade
         * writes the 1-based index of the stage that promoted the peak)
         */
        {std::pmr::string("type_tag"), gr::pmt::Value(peak.typeTag)},
        {std::pmr::string("isolation"), gr::pmt::Value(peak.isolation * positionScale)},
        {std::pmr::string("w68"), gr::pmt::Value(peak.w68 * positionScale)},
        {std::pmr::string("w96"), gr::pmt::Value(peak.w96 * positionScale)},
        {std::pmr::string("w99"), gr::pmt::Value(peak.w99 * positionScale)},
        {std::pmr::string("kurtosis"), gr::pmt::Value(peak.kurtosis)},
        {std::pmr::string("noise_sigma"), gr::pmt::Value(noise.robustSigma)},
        {std::pmr::string("noise_floor"), gr::pmt::Value(noise.median)},
        {std::pmr::string("position_uncertainty"), gr::pmt::Value(peak.positionUncertainty * positionScale)},
        {std::pmr::string("width_uncertainty"), gr::pmt::Value(peak.widthUncertainty * positionScale)},
        {std::pmr::string("amplitude_uncertainty"), gr::pmt::Value(peak.amplitudeUncertainty)},
    };
}

} // namespace gr::blocks::fourier

#endif // GNURADIO_PEAK_EVENT_PROPS_HPP
