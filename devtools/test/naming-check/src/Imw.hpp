// expect-file: ok
#ifndef NAMING_FIXTURE_IMW_HPP
#define NAMING_FIXTURE_IMW_HPP // expect: ok

namespace fixture::imw { // the namespace a header opens is something it declares
struct Font {            // expect: ok
    int size = 1;        // expect: ok
};
} // namespace fixture::imw

#endif
