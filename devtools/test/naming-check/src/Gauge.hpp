// expect-file: ok
#ifndef NAMING_FIXTURE_GAUGE_HPP
#define NAMING_FIXTURE_GAUGE_HPP // expect: ok

namespace fixture {
struct GaugeFace { // expect: ok
    int size = 1;  // expect: ok
};
int gaugeArea(int side); // expect: ok
} // namespace fixture

#endif
