#ifndef POWERESTIMATORS_HPP
#define POWERESTIMATORS_HPP

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::electrical {

template<typename T, std::size_t nPhases>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
struct PowerMetrics : Block<PowerMetrics<T, nPhases>> {
    using Description = Doc<R""(@brief PowerMetrics

Computes per-phase active power (P), reactive power (Q), apparent power (S), RMS voltage (U_rms), and RMS current (I_rms)
for single-phase or multi-phase electrical systems. It processes voltage and current inputs,
applies low-pass filters to calculate average values, and outputs decimated results at a specified rate.

[0] "IEEE Standard Definitions for the Measurement of Electric Power Quantities Under Sinusoidal, Nonsinusoidal,
    Balanced, or Unbalanced Conditions," in IEEE Std 1459-2010 (Revision of IEEE Std 1459-2000), vol., no., pp.1-50,
    19 March 2010, doi: 10.1109/IEEESTD.2010.5439063, https://ieeexplore.ieee.org/document/5439063
)"">;
    // ports
    std::vector<PortIn<T>> U{nPhases};
    std::vector<PortIn<T>> I{nPhases};

    std::vector<PortOut<T>> P{nPhases}; // active power (in-phase)
    std::vector<PortOut<T>> Q{nPhases}; // reactive power (i.e. 90-degree out of phase component)
    std::vector<PortOut<T>> S{nPhases}; // apparent power (i.e. U*I)
    std::vector<PortOut<T>> U_rms{nPhases};
    std::vector<PortOut<T>> I_rms{nPhases};

    // settings
    Annotated<float, "sample_rate", gr::Unit<"Hz">>                               sample_rate{10'000.f};
    Annotated<gr::Size_t, "decimation factor", Doc<"decimation_factor">, Visible> decim{100U};

    GR_MAKE_REFLECTABLE(PowerMetrics, U, I, P, Q, S, U_rms, I_rms, sample_rate, decim);

    // private state for exponential moving average (EMA)
    using FilterImpl = std::conditional_t<UncertainValueLike<T>, filter::ErrorPropagatingFilter<T>, filter::Filter<meta::fundamental_base_value_type_t<T>>>;

    std::array<FilterImpl, nPhases> _lpVoltageSquared;
    std::array<FilterImpl, nPhases> _lpCurrentSquared;
    std::array<FilterImpl, nPhases> _lpActivePower;

    void initFilters() {
        using namespace gr::filter;
        using ValueType = meta::fundamental_base_value_type_t<T>;

        const double cutoff_frequency = 0.5 * static_cast<double>(sample_rate) / static_cast<double>(decim);
        const auto   filter_init      = [&, cutoff_frequency](auto) {                                                    //
            return FilterImpl(iir::designFilter<ValueType>(Type::LOWPASS,                                         //
                       FilterParameters{.order = 2UZ, .fLow = cutoff_frequency, .fs = static_cast<double>(sample_rate)}, //
                       iir::Design::BUTTERWORTH));
        };

        constexpr auto indices = std::views::iota(0UZ, nPhases);
        std::ranges::transform(indices, _lpVoltageSquared.begin(), filter_init);
        std::ranges::transform(indices, _lpCurrentSquared.begin(), filter_init);
        std::ranges::transform(indices, _lpActivePower.begin(), filter_init);
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) { initFilters(); }

    template<typename TInputSpanType, typename TOutputSpanType>
    constexpr work::Status processBulk(std::span<TInputSpanType>& voltage, std::span<TInputSpanType>& current,                         // inputs
        std::span<TOutputSpanType>& activePower, std::span<TOutputSpanType>& reactivePower, std::span<TOutputSpanType>& apparentPower, // power outputs
        std::span<TOutputSpanType>& rmsVoltage, std::span<TOutputSpanType>& rmsCurrent) {
        for (std::size_t phaseIdx = 0UZ; phaseIdx < nPhases; ++phaseIdx) { // process each phase
            for (std::size_t i = 0UZ; i < voltage[phaseIdx].size(); ++i) { // iterate over samples
                const T u_i = voltage[phaseIdx][i];
                const T i_i = current[phaseIdx][i];

                const T p_i    = u_i * i_i;                                         // instantaneous power
                const T ema_p  = _lpActivePower[phaseIdx].processOne(p_i);          // update exponential moving average for power
                const T ema_u2 = _lpVoltageSquared[phaseIdx].processOne(u_i * u_i); // update exponential moving average for voltage squared
                const T ema_i2 = _lpCurrentSquared[phaseIdx].processOne(i_i * i_i); // update exponential moving average for current squared

                if (i % static_cast<std::size_t>(decim) == 0UZ) {
                    const std::size_t outIdx = i / static_cast<std::size_t>(decim);
                    const T           u_rms  = math::sqrt(ema_u2);
                    const T           i_rms  = math::sqrt(ema_i2);

                    const T S_i = u_rms * i_rms;                                         // apparent power
                    T       Q_i = math::sqrt(std::max(S_i * S_i - ema_p * ema_p, T(0))); // reactive power

                    activePower[phaseIdx][outIdx]   = ema_p;
                    reactivePower[phaseIdx][outIdx] = Q_i;
                    apparentPower[phaseIdx][outIdx] = S_i;

                    rmsVoltage[phaseIdx][outIdx] = u_rms;
                    rmsCurrent[phaseIdx][outIdx] = i_rms;
                }
            }
        }

        return work::Status::OK;
    }
};

template<typename T>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
using ThreePhasePowerMetrics = PowerMetrics<T, 3UZ>;

static_assert(BlockLike<ThreePhasePowerMetrics<float>>, "block constraints not satisfied");
static_assert(BlockLike<ThreePhasePowerMetrics<UncertainValue<float>>>, "block constraints not satisfied");
static_assert(gr::HasProcessBulkFunction<ThreePhasePowerMetrics<float>>);
static_assert(gr::HasProcessBulkFunction<ThreePhasePowerMetrics<UncertainValue<float>>>);

template<typename T>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
using SinglePhasePowerMetrics = PowerMetrics<T, 1UZ>;
static_assert(BlockLike<SinglePhasePowerMetrics<float>>, "block constraints not satisfied");

template<typename T, std::size_t nPhases>
requires(std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>)
struct PowerFactor : gr::Block<PowerFactor<T, nPhases>> {
    using Description = Doc<R""(@brief PowerFactor

Calculates the power factor and phase angle for each phase using active power (P) and apparent power (S) inputs.
It computes the power factor as cos(𝜙)=P/S, and the phase angle 𝜙=arccos(P/S), ensuring values are within valid ranges.
)"">;
    // ports
    std::vector<PortIn<T>>  P{nPhases}; // active power
    std::vector<PortIn<T>>  S{nPhases}; // apparent power
    std::vector<PortOut<T>> power_factor{nPhases};
    std::vector<PortOut<T>> phase{nPhases};

    GR_MAKE_REFLECTABLE(PowerFactor, P, S, power_factor, phase);

    template<typename TInputSpanType, typename TOutputSpanType>
    constexpr work::Status processBulk(std::span<TInputSpanType> pIn, std::span<TInputSpanType> sIn, //
        std::span<TOutputSpanType> powerFactorOut, std::span<TOutputSpanType> phaseAngle) {
        for (std::size_t phaseIdx = 0; phaseIdx < nPhases; ++phaseIdx) {
            const std::size_t n_samples = pIn[phaseIdx].size();

            for (std::size_t n = 0; n < n_samples; ++n) {
                T P_i = pIn[phaseIdx][n];
                T S_i = sIn[phaseIdx][n];

                T cos_phi = S_i != T(0) ? P_i / S_i : T(0); // avoid division by zero
                cos_phi   = std::clamp(cos_phi, T(-1), T(1));

                T phi = std::acos(gr::value(cos_phi));

                powerFactorOut[phaseIdx][n] = cos_phi;
                phaseAngle[phaseIdx][n]     = phi;
            }
        }

        return work::Status::OK;
    }
};

template<typename T>
using SinglePhasePowerFactorCalculator = PowerFactor<T, 1>;

template<typename T>
using ThreePhasePowerFactorCalculator = PowerFactor<T, 3>;

template<typename T, std::size_t nPhases>
requires((std::floating_point<T> or std::is_arithmetic_v<meta::fundamental_base_value_type_t<T>>) && (nPhases > 1)) // unbalance calculation requires at least two phases
struct SystemUnbalance : Block<SystemUnbalance<T, nPhases>> {
    using Description = Doc<R""(@brief SystemUnbalance

Computes the total active power, voltage unbalance, and current unbalance in multi-phase systems. Unbalance percentages
are calculated based on the maximum deviation from average RMS values across all phases.

[0] IEC 61000-4-30 Electromagnetic compatibility (EMC) - Part 4-30:
    Testing and measurement techniques - Power quality measurement methods, https://webstore.iec.ch/en/publication/21844
)"">;

    // ports
    std::vector<PortIn<T>> voltage_rms_inputs{nPhases};  // U_rms_in
    std::vector<PortIn<T>> current_rms_inputs{nPhases};  // I_rms_in
    std::vector<PortIn<T>> active_power_inputs{nPhases}; // P_in

    PortOut<T> total_active_power_output; // P_total_out
    PortOut<T> voltage_unbalance_output;  // U_unbalance_out
    PortOut<T> current_unbalance_output;  // I_unbalance_out

    GR_MAKE_REFLECTABLE(SystemUnbalance, voltage_rms_inputs, current_rms_inputs, active_power_inputs, total_active_power_output, voltage_unbalance_output, current_unbalance_output);

    template<typename TInputSpanType>
    constexpr work::Status processBulk(std::span<TInputSpanType>& rmsUIn, std::span<TInputSpanType>& rmsIIn, std::span<TInputSpanType>& PIn, //
        std::span<T>& totalP, std::span<T>& unbalancedU, std::span<T>& unbalancedI) {
        const std::size_t n_samples = rmsUIn[0].size();

        for (std::size_t n = 0; n < n_samples; ++n) {
            T                      P_total = T(0);
            std::array<T, nPhases> U_rms_values;
            std::array<T, nPhases> I_rms_values;

            for (std::size_t phaseIdx = 0; phaseIdx < nPhases; ++phaseIdx) {
                U_rms_values[phaseIdx] = rmsUIn[phaseIdx][n];
                I_rms_values[phaseIdx] = rmsIIn[phaseIdx][n];
                P_total += PIn[phaseIdx][n];
            }

            // voltage unbalance
            T U_avg      = std::accumulate(U_rms_values.begin(), U_rms_values.end(), T(0)) / static_cast<T>(nPhases);
            T deltaU_max = std::transform_reduce(
                U_rms_values.begin(), U_rms_values.end(), T(0), //
                [](T a, T b) { return std::max(a, b); },        // reduction operator
                [U_avg](T U_i) { return std::abs(gr::value(U_i - U_avg)); });

            T VoltageUnbalance = gr::value(U_avg) > 0 ? (deltaU_max / U_avg) * T(100) : T(0);

            // current Unbalance
            T I_avg       = std::accumulate(I_rms_values.begin(), I_rms_values.end(), T(0)) / static_cast<T>(nPhases);
            T Delta_I_max = *std::max_element(I_rms_values.begin(), I_rms_values.end(), //
                [I_avg](T a, T b) { return std::abs(gr::value(a - I_avg)) < std::abs(gr::value(b - I_avg)); });
            Delta_I_max   = std::abs(gr::value(Delta_I_max - I_avg));

            T CurrentUnbalance = gr::value(I_avg) > 0 ? (Delta_I_max / I_avg) * T(100) : T(0);

            // output values
            totalP[n]      = P_total;
            unbalancedU[n] = VoltageUnbalance;
            unbalancedI[n] = CurrentUnbalance;
        }

        return work::Status::OK;
    }
};

template<typename T>
using TwoPhaseSystemUnbalanceCalculator = SystemUnbalance<T, 2>;

template<typename T>
using ThreePhaseSystemUnbalanceCalculator = SystemUnbalance<T, 3>;

} // namespace gr::electrical

inline static auto registerPowerMetrics = gr::registerBlock<gr::electrical::ThreePhasePowerMetrics, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry())              //
                                          + gr::registerBlock<gr::electrical::SinglePhasePowerMetrics, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry())           //
                                          + gr::registerBlock<gr::electrical::SinglePhasePowerFactorCalculator, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry())  //
                                          + gr::registerBlock<gr::electrical::ThreePhasePowerFactorCalculator, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry())   //
                                          + gr::registerBlock<gr::electrical::TwoPhaseSystemUnbalanceCalculator, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry()) //
                                          + gr::registerBlock<gr::electrical::ThreePhaseSystemUnbalanceCalculator, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>(gr::globalBlockRegistry());

#endif // POWERESTIMATORS_HPP
