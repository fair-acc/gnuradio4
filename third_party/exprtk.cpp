#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif

#include "exprtk/exprtk.hpp" // include the original ExprTk header

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// explicit instantiations for supported types
template class exprtk::expression<float>;
template class exprtk::expression<double>;

template class exprtk::parser<float>;
template class exprtk::parser<double>;

template class exprtk::symbol_table<float>;
template class exprtk::symbol_table<double>;

template class exprtk::function_compositor<float>;
template class exprtk::function_compositor<double>;
