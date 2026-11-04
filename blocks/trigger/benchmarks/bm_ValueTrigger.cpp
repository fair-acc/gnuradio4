#include <benchmark.hpp>

#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/trigger/ValueTrigger.hpp>

#include "TriggerBenchmark.hpp"

namespace {
using gr::trigger_benchmark::kSamples;
using gr::trigger_benchmark::runThrough;
using namespace gr::blocks::trigger;
using enum ValueCondition;

const gr::property_map kLevel{{"threshold", 0.f}, {"hysteresis", 0.1f}, {"sample_rate", 1000.f}};
const gr::property_map kBand{{"threshold", -0.5f}, {"threshold_upper", 0.5f}, {"hysteresis", 0.05f}, {"sample_rate", 1000.f}};
const gr::property_map kWidth{{"threshold", 0.f}, {"hysteresis", 0.1f}, {"duration_condition", std::string("inside")}, {"duration_min", 16.}, {"duration_max", 48.}, {"sample_rate", 1000.f}};
} // namespace

inline const boost::ut::suite<"ValueTrigger"> _valueTrigger = [] {
    using namespace benchmark;
    "control: Copy<float>"_benchmark.repeat<5>(kSamples)                              = [] { runThrough<gr::testing::Copy<float>>({}); };
    "ValueTrigger<float, level>"_benchmark.repeat<5>(kSamples)                        = [] { runThrough<ValueTrigger<float, level>>(kLevel); };
    "ValueTrigger<float, window>"_benchmark.repeat<5>(kSamples)                       = [] { runThrough<ValueTrigger<float, window>>(kBand); };
    "ValueTrigger<float, pulseWidth>, 16 to 48 samples"_benchmark.repeat<5>(kSamples) = [] { runThrough<ValueTrigger<float, pulseWidth>>(kWidth); };
    "ValueTrigger<float, runt>"_benchmark.repeat<5>(kSamples)                         = [] { runThrough<ValueTrigger<float, runt>>(kBand); };
    "ValueTrigger<float, transition>"_benchmark.repeat<5>(kSamples)                   = [] { runThrough<ValueTrigger<float, transition>>(kBand); };
};

int main() {}
