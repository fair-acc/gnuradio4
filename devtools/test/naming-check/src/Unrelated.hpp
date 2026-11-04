// expect-file: R14
#ifndef NAMING_FIXTURE_UNRELATED_HPP
#define NAMING_FIXTURE_UNRELATED_HPP // expect: ok

namespace fixture {
struct GaugeNeedle { // expect: ok
    int size = 2;    // expect: ok
};
} // namespace fixture

#endif
