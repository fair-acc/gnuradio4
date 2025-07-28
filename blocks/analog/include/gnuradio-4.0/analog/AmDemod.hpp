#ifndef INCLUDED_ANALOG_AM_DEMOD_HPP
#define INCLUDED_ANALOG_AM_DEMOD_HPP

#include <cmath>
#include <complex>
#include <numbers>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::blocks::analog
{
struct AmDemod : gr::Block<AmDemod>          // fixed‑rate block (1→1)
{
    using Description = Doc<
        R""(@brief Envelope AM demodulator (complex → float))"">;

    PortIn<std::complex<float>> in;
    PortOut<float>              out;

    Annotated<float, "chan_rate",  Doc<"complex sample‑rate [Hz]">>
        chan_rate{48'000.f};
    Annotated<int,   "audio_decim",Doc<"legacy decimation factor ≥ 1">>
        audio_decim{8};
    Annotated<float, "audio_pass", Doc<"audio LPF corner [Hz]">>
        audio_pass{4'000.f};
    Annotated<float, "audio_stop", Doc<"stop‑band edge (kept for API parity)">>
        audio_stop{5'500.f};

    GR_MAKE_REFLECTABLE(
        AmDemod, in, out, chan_rate, audio_decim, audio_pass, audio_stop);

    void set_chan_rate (float fs) { chan_rate   = fs; _recalc(); }
    void set_audio_decim(int d )  { audio_decim = std::max(1, d); _recalc(); }
    void set_audio_pass (float fp){ audio_pass  = fp; _recalc(); }

    explicit AmDemod(property_map) { _recalc(); }
    AmDemod(float fs, int d, float fp, float fsb = 0.f)
        : chan_rate(fs), audio_decim(std::max(1, d)),
          audio_pass(fp), audio_stop(fsb)
    { _recalc(); }

    void settingsChanged(const property_map&, const property_map&)
    { _recalc(); }

    template<InputSpanLike  InSpan,
             OutputSpanLike OutSpan>
    [[nodiscard]] work::Status
    processBulk(const InSpan& xs, OutSpan& ys)
    {
        std::size_t produced = 0;

        for (auto x : xs) {
            const float env = std::abs(x);
            _y = env + _alpha * (_y - env);

            if (produced == ys.size()) {
                ys.publish(produced);
                return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
            }

            ys[produced++] = _y;                // 1 : 1 output
        }

        ys.publish(produced);
        return work::Status::OK;
    }

private:
    float _alpha{1.f};      // IIR coefficient
    float _y{0.f};          // filter state

    void _recalc()
    {
        /* one‑pole IIR coefficient (designed at the INPUT rate) */
        const float dt = 1.0f / chan_rate;
        _alpha = std::exp(-2.f * std::numbers::pi_v<float>
                          * audio_pass * dt);
    }
};

GR_REGISTER_BLOCK("gr::blocks::analog::AmDemod",
                  gr::blocks::analog::AmDemod)

} // namespace gr::blocks::analog
#endif /* INCLUDED_ANALOG_AM_DEMOD_HPP */
