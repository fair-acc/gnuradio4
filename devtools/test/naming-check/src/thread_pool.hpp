// expect-file: ok
#ifndef NAMING_FIXTURE_THREAD_POOL_HPP
#define NAMING_FIXTURE_THREAD_POOL_HPP // expect: ok

#include <cstddef>

namespace fixture_std {
inline int thread_pool_size() { return 4; } // expect: ok
} // namespace fixture_std

#endif
