#ifndef GNURADIO_LOCALISATION_CLOCK_REFERENCED_DELAY_HPP
#define GNURADIO_LOCALISATION_CLOCK_REFERENCED_DELAY_HPP

#include <cstdint>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <limits>

namespace gr::localisation {

[[nodiscard]] constexpr double clockReferencedDelaySeconds(std::uint64_t emissionNs, std::uint64_t arrivalNs, double interfaceLatencySeconds) noexcept {
    const double elapsedNs = arrivalNs >= emissionNs ? static_cast<double>(arrivalNs - emissionNs) : -static_cast<double>(emissionNs - arrivalNs);
    return elapsedNs * 1.e-9 - interfaceLatencySeconds;
}

GR_REGISTER_BLOCK(gr::localisation::ClockReferencedDelay, [T], [ float, double ])

template<std::floating_point T>
struct ClockReferencedDelay : Block<ClockReferencedDelay<T>, NoTagPropagation> {
    using Description = Doc<"Delay of each arrival trigger_time tag on `in` after the latest emission trigger_time on `clk`, less interface_latency; first arrival per sample wins.">;

    PortIn<T>            in;
    PortIn<std::uint8_t> clk;
    PortOut<double>      out;

    Annotated<double, "interface_latency", Unit<"s">, Doc<"output-to-input latency of the interface, subtracted from each delay">> interface_latency = 0.0;

    GR_MAKE_REFLECTABLE(ClockReferencedDelay, in, clk, out, interface_latency);

    std::uint64_t _emissionTimeNs = 0U;
    bool          _hasEmission    = false;

    void start() { _hasEmission = false; }

    work::Status processBulk(InputSpanLike auto& input, InputSpanLike auto& clock, OutputSpanLike auto& output) {
        const std::size_t count            = std::min({input.size(), clock.size(), output.size()});
        const auto        emissions        = clock.tags();
        auto              emission         = emissions.begin();
        const auto        advanceEmissions = [&](std::size_t through) {
            while (emission != emissions.end()) {
                const auto& [index, map] = *emission;
                const std::size_t offset = index < 0 ? 0UZ : static_cast<std::size_t>(index);
                if (offset >= count || offset > through) {
                    break;
                }
                if (map.get().contains(gr::tag::kTriggerTime)) {
                    _emissionTimeNs = map.get().template value_or<std::uint64_t>(gr::tag::kTriggerTime, 0U);
                    _hasEmission    = true;
                }
                ++emission;
            }
        };
        std::size_t emitted         = 0UZ;
        std::size_t previousArrival = std::numeric_limits<std::size_t>::max();
        for (const auto& [index, map] : input.tags()) {
            const std::size_t offset = index < 0 ? 0UZ : static_cast<std::size_t>(index);
            if (offset >= count) {
                break;
            }
            advanceEmissions(offset);
            if (!_hasEmission || previousArrival == offset || !map.get().contains(gr::tag::kTriggerTime)) {
                continue;
            }
            const auto arrival = map.get().template value_or<std::uint64_t>(gr::tag::kTriggerTime, 0U);
            output[emitted]    = clockReferencedDelaySeconds(_emissionTimeNs, arrival, interface_latency);
            output.publishTag(property_map{{gr::tag::kTriggerTime, arrival}}, emitted);
            ++emitted;
            previousArrival = offset;
        }
        if (count > 0UZ) {
            advanceEmissions(count - 1UZ);
        }
        std::ignore = input.consume(count);
        std::ignore = clock.consume(count);
        output.publish(emitted);
        return work::Status::OK;
    }
};

} // namespace gr::localisation
#endif // GNURADIO_LOCALISATION_CLOCK_REFERENCED_DELAY_HPP
