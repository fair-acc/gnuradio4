#ifndef GNURADIO_META_EXECUTION_POLICY_HPP
#define GNURADIO_META_EXECUTION_POLICY_HPP

#include <numeric>

// this mocks the execution policy until Emscripten's libc++ does support this (Clang already does)
#if not defined(__GLIBCXX__) && (defined(__EMSCRIPTEN__) || defined(__clang__))

namespace std {

namespace execution {
class mock_execution_policy {};

inline constexpr mock_execution_policy seq{};
inline constexpr mock_execution_policy unseq{};
inline constexpr mock_execution_policy par{};
} // namespace execution

template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp1, typename BinaryOp2>
inline T transform_reduce(auto, InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2) {
    return std::transform_reduce(first1, last1, first2, init, binary_op1, binary_op2);
}

} // namespace std
#endif

#endif // GNURADIO_META_EXECUTION_POLICY_HPP
