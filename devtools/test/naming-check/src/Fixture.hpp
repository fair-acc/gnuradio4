// expect-file: ok
#ifndef NAMING_FIXTURE_FIXTURE_HPP
#define NAMING_FIXTURE_FIXTURE_HPP // expect: ok

namespace fixture {
inline int fixtureEntry() { return 0; } // expect: ok
} // namespace fixture

#endif
