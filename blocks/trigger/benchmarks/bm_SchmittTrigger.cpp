#include <benchmark.hpp>

#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/trigger/SchmittTrigger.hpp>

#include "TriggerBenchmark.hpp"

namespace {
using gr::blocks::trigger::SchmittTrigger;
using gr::trigger_benchmark::kSamples;
using gr::trigger_benchmark::runThrough;
using enum gr::trigger::InterpolationMethod;

const gr::property_map kBand{{"offset", 0.f}, {"threshold", 0.1f}};
} // namespace

inline const boost::ut::suite<"SchmittTrigger"> _schmittTrigger = [] {
    using namespace benchmark;
    "control: Copy<float>"_benchmark.repeat<5>(kSamples)                     = [] { runThrough<gr::testing::Copy<float>>({}); };
    "SchmittTrigger<float, none>"_benchmark.repeat<5>(kSamples)              = [] { runThrough<SchmittTrigger<float, NO_INTERPOLATION>>(kBand); };
    "SchmittTrigger<float, basic linear>"_benchmark.repeat<5>(kSamples)      = [] { runThrough<SchmittTrigger<float, BASIC_LINEAR_INTERPOLATION>>(kBand); };
    "SchmittTrigger<float, linear regression>"_benchmark.repeat<5>(kSamples) = [] { runThrough<SchmittTrigger<float, LINEAR_INTERPOLATION>>(kBand); };
    "SchmittTrigger<float, polynomial>"_benchmark.repeat<5>(kSamples)        = [] { runThrough<SchmittTrigger<float, POLYNOMIAL_INTERPOLATION>>(kBand); };
};

int main() {}
