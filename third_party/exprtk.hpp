#ifndef GNURADIO_EXPRTK_HPP
#define GNURADIO_EXPRTK_HPP

/**
 * @brief Wrapper for the ExprTk library with precompiled template instantiations.
 *
 * This header provides a precompiled static library (`libexprtk.a`) interface for
 * the ExprTk library, optimising build times by avoiding redundant template
 * instantiations for 'float' and 'double' across multiple compilation units.
 *
 * ## Measured Performance
 * 1. Build the Static Library: time cmake --build . --target exprtk
 *     [..]
 *     real    1m55.642s
 *     user    1m53.015s
 *     sys     0m2.604s
 * 2. Build a Dependent Target:
 *    time cmake --build . --target exprtk_example0
 *    [..]    w/ static lib  w/o static lib
 *    real    0m22.068s      1m0.512s
 *    user    0m20.023s      0m58.802s
 *    sys     0m1.347s       0m1.636s
 *    time cmake --build . --target exprtk_example1
 *    [..]    w/ static lib  w/o static lib
 *    real    0m17.774s      1m0.234s
 *    user    0m16.541s      0m58.608s
 *    sys     0m1.224s       0m1.582s
 *
 * ## Usage
 * * Link your target against `libexprtk.a` - CMake snippet:
 *   target_link_libraries(<target> PRIVATE exprtk <other deps>)
 * * include the exprtk.hpp wrapper:
 *   @code
 *   #include <exprtk.hpp>
 *
 *   exprtk::symbol_table<float> table;
 *   exprtk::expression<float> expr;
 *   exprtk::parser<float> parser;
 *   // [..] <your code> [..]
 *   @endcode
 */

#include <cstdint>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif

#include "exprtk/exprtk.hpp" // include the original ExprTk header

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// explicit template instantiations for commonly used types
extern template class exprtk::expression<float>;
extern template class exprtk::expression<double>;

extern template class exprtk::parser<float>;
extern template class exprtk::parser<double>;

extern template class exprtk::symbol_table<float>;
extern template class exprtk::symbol_table<double>;

extern template class exprtk::function_compositor<float>;
extern template class exprtk::function_compositor<double>;

#endif // GNURADIO_EXPRTK_HPP
