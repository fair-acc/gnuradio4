// expect-file: ok
#ifndef NAMING_FIXTURE_GAUGES_HPP
#define NAMING_FIXTURE_GAUGES_HPP // expect: ok

namespace fixture {
struct GaugeSet { // expect: ok
    int size = 1; // expect: ok
};
int forEachGauge(int side); // expect: ok
} // namespace fixture

#endif
