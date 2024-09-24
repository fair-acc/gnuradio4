#ifndef GNURADIO_REFLECTION_HPP
#define GNURADIO_REFLECTION_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <vir/reflect-light.h>
#pragma GCC diagnostic pop
#include <gnuradio-4.0/meta/typelist.hpp>

#define GR_MAKE_REFLECTABLE VIR_MAKE_REFLECTABLE

namespace gr::refl {

using std::size_t;

using namespace vir::refl;

namespace detail {
template<typename IdxSeq, auto Fun>
struct make_typelist_from_index_sequence_impl;

template<size_t... Is, auto Fun>
struct make_typelist_from_index_sequence_impl<std::index_sequence<Is...>, Fun> {
    using type = meta::concat<decltype(Fun(std::integral_constant<size_t, Is>{}))...>;
};
} // namespace detail

/**
 * Constructs a meta::typelist via concatenation of all type lists returned from applying \p Fun to each index in the
 * given std::index_sequence \p IdxSeq.
 *
 * \tparam IdxSeq  The sequence of indexes to pass to \p Fun.
 * \tparam Fun     A function object (e.g. Lambda) that is called for every integer in \p IdxSeq. It is passed an
 *                 std::integral_constant<std::size_t, Idx> and needs to return a meta::typelist object. The return
 *                 types of all \p Fun invocations are then concatenated (meta::concat) to the resulting typelist.
 */
template <typename IdxSeq, auto Fun>
using make_typelist_from_index_sequence = typename detail::make_typelist_from_index_sequence_impl<IdxSeq, Fun>::type;

} // namespace gr::refl

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // GNURADIO_REFLECTION_HPP
