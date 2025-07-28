#ifndef INCLUDED_ANALOG_PHASE_MODULATOR_HPP
#define INCLUDED_ANALOG_PHASE_MODULATOR_HPP

#include <cmath>
#include <complex>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::blocks::analog {

struct PhaseModulator
  : gr::Block<PhaseModulator, gr::Resampling<1UZ, 1UZ, false>>   // 1 : 1
{
    using Description = Doc<R""(@brief Phase modulator (float→complex))"">;

    PortIn<float>               in;
    PortOut<std::complex<float>> out;

    Annotated<float, "sensitivity",
              Doc<"phase change per input unit [rad]">> sensitivity{1.0f};

    GR_MAKE_REFLECTABLE(PhaseModulator, in, out, sensitivity);

    explicit PhaseModulator(gr::property_map) {}

    void set_sensitivity(float s) { sensitivity = s; }

    void settingsChanged(const property_map&, const property_map&)
    { /* nothing else to recompute */ }

    template<InputSpanLike  InSpan,
             OutputSpanLike OutSpan>
    [[nodiscard]] work::Status
    processBulk(const InSpan& xs, OutSpan& ys)
    {
        const std::size_t n = std::min(xs.size(), ys.size());
        for (std::size_t i = 0; i < n; ++i) {
            const float phi = sensitivity * xs[i];
            ys[i] = { std::cos(phi), std::sin(phi) };
        }
        ys.publish(n);
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::analog::PhaseModulator",
                  gr::blocks::analog::PhaseModulator)

} // namespace gr::blocks::analog
#endif /* INCLUDED_ANALOG_PHASE_MODULATOR_HPP */
