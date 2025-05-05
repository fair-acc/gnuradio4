#include <boost/ut.hpp>

#include <cmath>
#include <random>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/electrical/PowerEstimators.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::electrical {
static_assert(BlockLike<ThreePhasePowerMetrics<float>>, "block constraints not satisfied");
static_assert(BlockLike<ThreePhasePowerMetrics<UncertainValue<float>>>, "block constraints not satisfied");
static_assert(gr::HasProcessBulkFunction<ThreePhasePowerMetrics<float>>);
static_assert(gr::HasProcessBulkFunction<ThreePhasePowerMetrics<UncertainValue<float>>>);

static_assert(BlockLike<SinglePhasePowerMetrics<float>>, "block constraints not satisfied");
} // namespace gr::electrical

const boost::ut::suite<"Power Metrics Estimators"> powerEstimatorTests = [] {
    using namespace boost::ut;
    using namespace gr::electrical;

    constexpr static float                freq           = 50.0f;    // Hz
    constexpr static float                duration       = 1.0f;     // seconds
    constexpr static float                sample_rate    = 10000.0f; // Hz
    constexpr static float                output_rate    = 50.0f;    // Hz
    constexpr static float                noiseAmplitude = 0.01f;    // rel. noise amplitude
    constexpr static float                V_rms          = 230.0f;
    constexpr static float                I_rms          = 10.0f;
    constexpr static float                V_peak         = V_rms * std::numbers::sqrt2_v<float>;
    constexpr static float                I_peak         = I_rms * std::numbers::sqrt2_v<float>;
    constexpr static float                V_offset       = +1.0f; // should be suppressed by input AC-coupling
    constexpr static float                I_offset       = -1.0f; // should be suppressed by input AC-coupling
    constexpr static std::array<float, 3> phaseShift     = {+0.0f, -2.0f * std::numbers::pi_v<float> / 3.0f, +2.0f * std::numbers::pi_v<float> / 3.0f};
    constexpr static std::array<float, 3> phaseOffset{0.1f, 0.2f, 0.3f};

    "[Single, ThreePhase]PowerMetrics"_test =
        []<typename TestParam>() {
            using T                        = typename TestParam::first_type;
            constexpr std::size_t kNPhases = TestParam::second_type::value;

            // small phase delay for current in each phase
            std::array<float, kNPhases> current_phase_delays = {};
            current_phase_delays.fill(0.0f); // Initialize to zero
            current_phase_delays[0] = 0.1f;  // phase delay in radians
            if constexpr (kNPhases > 1) {
                current_phase_delays[1] = 0.2f;
                current_phase_delays[2] = 0.3f;
            }

            // create voltage and current signals
            const auto                      nSamplesIn = static_cast<std::size_t>(duration * sample_rate);
            std::vector<std::vector<T>>     voltageIn(kNPhases, std::vector<T>(nSamplesIn));
            std::vector<std::vector<T>>     currentIn(kNPhases, std::vector<T>(nSamplesIn));
            std::mt19937                    rng(42);                     // fixed seed for reproducibility
            std::normal_distribution<float> noise(0.0f, noiseAmplitude); // 1% measurement noise

            for (std::size_t phase = 0; phase < kNPhases; ++phase) {
                float voltage_phase = phaseShift[phase];
                float current_phase = phaseShift[phase] + current_phase_delays[phase];

                for (std::size_t n = 0; n < nSamplesIn; ++n) {
                    const float t       = static_cast<float>(n) / sample_rate;
                    float       voltage = V_offset + V_peak * (std::sin(2.0f * std::numbers::pi_v<float> * freq * t + voltage_phase) + noise(rng));
                    float       current = I_offset + I_peak * (std::sin(2.0f * std::numbers::pi_v<float> * freq * t + current_phase) + noise(rng));

                    if constexpr (std::is_same_v<T, float>) {
                        voltageIn[phase][n] = voltage;
                        currentIn[phase][n] = current;
                    } else { // T is gr::UncertainValue<float>
                        voltageIn[phase][n] = gr::UncertainValue<float>(voltage, std::abs(V_peak) * noiseAmplitude);
                        currentIn[phase][n] = gr::UncertainValue<float>(current, std::abs(V_peak) * noiseAmplitude);
                    }
                }
            }

            // prepare actual unit-test
            const std::string         testName = std::format("{}", gr::meta::type_name<PowerMetrics<T, kNPhases>>());
            PowerMetrics<T, kNPhases> block;
            block.decimate = sample_rate / output_rate;
            block.initFilters();

            // prepare input spans
            std::vector<std::span<const T>> voltageSpans;
            std::vector<std::span<const T>> currentSpans;

            for (std::size_t i = 0; i < kNPhases; ++i) {
                voltageSpans.emplace_back(voltageIn[i]);
                currentSpans.emplace_back(currentIn[i]);
            }

            std::span<std::span<const T>> spanInVoltage(voltageSpans);
            std::span<std::span<const T>> spanInCurrent(currentSpans);

            // prepare output vectors
            std::size_t                 nSamplesOut = nSamplesIn / block.decimate;
            std::vector<std::vector<T>> activePowerVec(kNPhases, std::vector<T>(nSamplesOut));
            std::vector<std::vector<T>> reactivePowerVec(kNPhases, std::vector<T>(nSamplesOut));
            std::vector<std::vector<T>> apparentPowerVec(kNPhases, std::vector<T>(nSamplesOut));
            std::vector<std::vector<T>> rmsVoltageVec(kNPhases, std::vector<T>(nSamplesOut));
            std::vector<std::vector<T>> rmsCurrentVec(kNPhases, std::vector<T>(nSamplesOut));

            // prepare output spans
            std::vector<std::span<T>> activePowerSpans;
            std::vector<std::span<T>> reactivePowerSpans;
            std::vector<std::span<T>> apparentPowerSpans;
            std::vector<std::span<T>> rmsVoltageSpans;
            std::vector<std::span<T>> rmsCurrentSpans;

            for (std::size_t i = 0; i < kNPhases; ++i) {
                activePowerSpans.emplace_back(activePowerVec[i]);
                reactivePowerSpans.emplace_back(reactivePowerVec[i]);
                apparentPowerSpans.emplace_back(apparentPowerVec[i]);
                rmsVoltageSpans.emplace_back(rmsVoltageVec[i]);
                rmsCurrentSpans.emplace_back(rmsCurrentVec[i]);
            }

            std::span<std::span<T>> activePower(activePowerSpans);
            std::span<std::span<T>> reactivePower(reactivePowerSpans);
            std::span<std::span<T>> apparentPower(apparentPowerSpans);
            std::span<std::span<T>> rmsVoltage(rmsVoltageSpans);
            std::span<std::span<T>> rmsCurrent(rmsCurrentSpans);

            // call processBulk
            expect(block.processBulk(spanInVoltage, spanInCurrent, //
                       activePower, reactivePower, apparentPower,  //
                       rmsVoltage, rmsCurrent) == gr::work::Status::OK);

            // Check outputs - compute expected values
            float expected_rms_voltage = V_rms;
            float expected_rms_current = I_rms;
            float tolerance_voltage    = expected_rms_voltage * 0.05f; // 5% tolerance
            float tolerance_current    = expected_rms_current * 0.05f; // 5% tolerance

            for (std::size_t phase = 0; phase < kNPhases; ++phase) {
                float cos_phi = std::cos(current_phase_delays[phase]);
                float sin_phi = std::sin(current_phase_delays[phase]);

                float expected_active_power   = V_rms * I_rms * cos_phi;
                float expected_reactive_power = V_rms * I_rms * sin_phi;
                float expected_apparent_power = V_rms * I_rms;

                float tolerance_power = expected_apparent_power * 0.1f; // 10% tolerance due to noise and filter settling

                // last output sample
                const std::size_t lastIdx = nSamplesOut - 1;
                if constexpr (std::is_same_v<T, float>) { // T = float
                    expect(approx(activePower[phase][lastIdx], expected_active_power, tolerance_power)) << std::format("{}: active power mismatch", testName);
                    expect(approx(reactivePower[phase][lastIdx], expected_reactive_power, tolerance_power)) << std::format("{}: reactive power mismatch", testName);
                    expect(approx(apparentPower[phase][lastIdx], expected_apparent_power, tolerance_power)) << std::format("{}: Apparent power mismatch", testName);
                    expect(approx(rmsVoltage[phase][lastIdx], expected_rms_voltage, tolerance_voltage)) << std::format("{}: RMS voltage mismatch", testName);
                    expect(approx(rmsCurrent[phase][lastIdx], expected_rms_current, tolerance_current)) << std::format("{}: RMS current mismatch", testName);
                } else { // T = gr::UncertainValue<float>
                    expect(approx(gr::value(activePower[phase][lastIdx]), expected_active_power, tolerance_power)) << std::format("{}: Active power value mismatch", testName);
                    expect(gt(gr::uncertainty(activePower[phase][lastIdx]), 0.0f)) << std::format("{}: Active power uncertainty is zero", testName, activePower[phase][lastIdx]);

                    expect(approx(gr::value(reactivePower[phase][lastIdx]), expected_reactive_power, tolerance_power)) << std::format("{}: Reactive power value mismatch", testName);
                    expect(gt(gr::uncertainty(reactivePower[phase][lastIdx]), 0.0f)) << std::format("{}: Reactive power uncertainty {} is zero", testName, reactivePower[phase][lastIdx]);

                    expect(approx(gr::value(apparentPower[phase][lastIdx]), expected_apparent_power, tolerance_power)) << std::format("{}: Apparent power value mismatch", testName);
                    expect(gt(gr::uncertainty(apparentPower[phase][lastIdx]), 0.0f)) << std::format("{}: Apparent power uncertainty {} is zero", testName, apparentPower[phase][lastIdx]);

                    expect(approx(gr::value(rmsVoltage[phase][lastIdx]), expected_rms_voltage, tolerance_voltage)) << std::format("{}: RMS voltage value mismatch", testName);
                    expect(gt(gr::uncertainty(rmsVoltage[phase][lastIdx]), 0.0f)) << std::format("{}: RMS voltage uncertainty {} is zero", testName, rmsVoltage[phase][lastIdx]);

                    expect(approx(gr::value(rmsCurrent[phase][lastIdx]), expected_rms_current, tolerance_current)) << std::format("{}: RMS current value mismatch", testName);
                    expect(gt(gr::uncertainty(rmsCurrent[phase][lastIdx]), 0.0f)) << std::format("{}: RMS current uncertainty {} is zero", testName, rmsCurrent[phase][lastIdx]);
                }
            }
        } |
        std::tuple{
            std::pair<float, std::integral_constant<std::size_t, 1>>{},                     //
            std::pair<gr::UncertainValue<float>, std::integral_constant<std::size_t, 1>>{}, //
            std::pair<float, std::integral_constant<std::size_t, 3>>{},                     //
            std::pair<gr::UncertainValue<float>, std::integral_constant<std::size_t, 3>>{}  //
        };

    "PowerFactor"_test = []<typename TestParam>() {
        using T                        = typename TestParam::first_type;
        constexpr std::size_t kNPhases = TestParam::second_type::value;

        PowerFactor<T, kNPhases> block;

        std::vector<std::vector<T>> vecInActivePower(kNPhases);   // P
        std::vector<std::vector<T>> vecInApparentPower(kNPhases); // S
        std::vector<std::vector<T>> vecOutPowerFactor(kNPhases);
        std::vector<std::vector<T>> vecOutPhaseAngle(kNPhases);
        std::vector<T>              expectedPowerFactor(kNPhases);
        std::vector<T>              expectedPhaseAngle(kNPhases);

        for (std::size_t phaseIdx = 0; phaseIdx < kNPhases; ++phaseIdx) {
            const auto cos_phi = static_cast<T>(std::cos(phaseOffset[phaseIdx]));
            const auto S_i     = static_cast<T>(V_rms * I_rms);

            vecInApparentPower[phaseIdx].push_back(S_i);
            vecInActivePower[phaseIdx].push_back(S_i * cos_phi);
            vecOutPowerFactor[phaseIdx].resize(1UZ);
            vecOutPhaseAngle[phaseIdx].resize(1UZ);

            expectedPowerFactor[phaseIdx] = cos_phi;
            expectedPhaseAngle[phaseIdx]  = static_cast<T>(phaseOffset[phaseIdx]);
        }

        // prepare input and output spans
        std::vector<std::span<const T>> spanInP;
        std::vector<std::span<const T>> spanInS;
        std::vector<std::span<T>>       spanOutPowerFactor;
        std::vector<std::span<T>>       spanOutPhaseAngle;
        for (std::size_t i = 0; i < kNPhases; ++i) {
            spanInP.push_back(vecInActivePower[i]);
            spanInS.push_back(vecInApparentPower[i]);
            spanOutPowerFactor.push_back(vecOutPowerFactor[i]);
            spanOutPhaseAngle.push_back(vecOutPhaseAngle[i]);
        }

        expect(block.processBulk(std::span{spanInP}, std::span{spanInS}, std::span{spanOutPowerFactor}, std::span{spanOutPhaseAngle}) == gr::work::Status::OK);

        for (std::size_t phaseIdx = 0; phaseIdx < kNPhases; ++phaseIdx) {
            expect(approx(spanOutPowerFactor[phaseIdx][0], expectedPowerFactor[phaseIdx], 1e-5)) << std::format("power factor mismatch for phase {}", phaseIdx);
            expect(approx(spanOutPhaseAngle[phaseIdx][0], expectedPhaseAngle[phaseIdx], 1e-5)) << std::format("phase angle mismatch for phase {}", phaseIdx);
        }
    } | std::tuple{std::pair<double, std::integral_constant<std::size_t, 1>>{}, std::pair<double, std::integral_constant<std::size_t, 3>>{}};

    "SystemUnbalance"_test = [] {
        using T                        = double;
        constexpr std::size_t kNPhases = 3;

        SystemUnbalance<T, kNPhases> block;

        // slight unbalance in voltages and currents
        std::array<T, kNPhases> inU_rms = {T(V_rms), T(V_rms * 1.01f), T(V_rms * 0.99f)};
        std::array<T, kNPhases> inI_rms = {T(I_rms), T(I_rms * 1.02f), T(I_rms * 0.98f)};

        T                       expectedTotalP = 0.0;
        std::array<T, kNPhases> activePower{};
        for (std::size_t i = 0; i < kNPhases; ++i) {
            T cosPhi       = static_cast<T>(std::cos(phaseOffset[i]));
            T P_i          = inU_rms[i] * inI_rms[i] * cosPhi;
            activePower[i] = P_i;
            expectedTotalP += P_i;
        }

        // expected voltage and current unbalance
        const T U_avg                    = std::accumulate(inU_rms.begin(), inU_rms.end(), T(0)) / static_cast<T>(kNPhases);
        const T I_avg                    = std::accumulate(inI_rms.begin(), inI_rms.end(), T(0)) / static_cast<T>(kNPhases);
        const T deltaU_max               = std::abs(*std::ranges::max_element(inU_rms, [U_avg](T a, T b) { return std::abs(a - U_avg) < std::abs(b - U_avg); }) - U_avg);
        const T VoltageUnbalanceExpected = (deltaU_max / U_avg) * 100.0;
        const T Delta_I_max              = std::abs(*std::ranges::max_element(inI_rms, [I_avg](T a, T b) { return std::abs(a - I_avg) < std::abs(b - I_avg); }) - I_avg);

        T CurrentUnbalanceExpected = (Delta_I_max / I_avg) * 100.0;

        // prepare input and output spans
        std::vector<std::vector<T>> vecInU_rms(kNPhases);
        std::vector<std::vector<T>> vecInI_rms(kNPhases);
        std::vector<std::vector<T>> vecInActivePower(kNPhases);
        std::vector<T>              vecOutTotalP(1UZ);
        std::vector<T>              vecOutUnbalancedU(1UZ);
        std::vector<T>              vecOutUnbalancedI(1UZ);

        for (std::size_t i = 0UZ; i < kNPhases; ++i) {
            vecInU_rms[i].push_back(inU_rms[i]);
            vecInI_rms[i].push_back(inI_rms[i]);
            vecInActivePower[i].push_back(activePower[i]);
        }

        std::vector<std::span<const T>> spanInU_rms;
        std::vector<std::span<const T>> spanInI_rms;
        std::vector<std::span<const T>> spanInP_in;
        for (std::size_t i = 0; i < kNPhases; ++i) {
            spanInU_rms.emplace_back(vecInU_rms[i]);
            spanInI_rms.emplace_back(vecInI_rms[i]);
            spanInP_in.emplace_back(vecInActivePower[i]);
        }
        std::span<std::span<const T>> spanSpanInU_rms(spanInU_rms);
        std::span<std::span<const T>> spanSpanInI_rms(spanInI_rms);
        std::span<std::span<const T>> spanSpanInP_in(spanInP_in);
        std::span<T>                  spanOutTotalP(vecOutTotalP);
        std::span<T>                  spanOutUnbalancedU(vecOutUnbalancedU);
        std::span<T>                  spanOutUnbalancedI(vecOutUnbalancedI);

        expect(block.processBulk(spanSpanInU_rms, spanSpanInI_rms, spanSpanInP_in, // inputs
                   spanOutTotalP, spanOutUnbalancedU, spanOutUnbalancedI) == gr::work::Status::OK);

        expect(approx(vecOutTotalP[0], expectedTotalP, expectedTotalP * 0.01)) << "total active power mismatch";
        expect(approx(vecOutUnbalancedU[0], VoltageUnbalanceExpected, 0.1)) << "voltage unbalance mismatch";
        expect(approx(vecOutUnbalancedI[0], CurrentUnbalanceExpected, 0.1)) << "current unbalance mismatch";
    };
};

int main() { /* not needed for UT */ }
