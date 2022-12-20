/*
    Copyright © 2022 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
                     Matthias Kretz <m.kretz@gsi.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VIR_DETAILS_H
#define VIR_DETAILS_H

#include "simd.h"
#include <type_traits>
#include <bit>

#if defined _GLIBCXX_EXPERIMENTAL_SIMD_H && defined __cpp_lib_experimental_parallel_simd
#define VIR_GLIBCXX_STDX_SIMD 1
#else
#define VIR_GLIBCXX_STDX_SIMD 0
#endif

namespace vir::meta
{
  template <typename T>
    using is_simd_or_mask = std::disjunction<stdx::is_simd<T>, stdx::is_simd_mask<T>>;

  template <typename T>
    inline constexpr bool is_simd_or_mask_v = std::disjunction_v<stdx::is_simd<T>,
                                                                 stdx::is_simd_mask<T>>;

    template <typename T>
      struct type_identity
      { using type = T; };

    template <typename T>
      using type_identity_t = typename type_identity<T>::type;

    template <typename T, typename U = long long, bool = (sizeof(T) == sizeof(U))>
      struct as_int;

    template <typename T, typename U>
      struct as_int<T, U, true>
      { using type = U; };

    template <typename T>
      struct as_int<T, long long, false>
      : as_int<T, long> {};

    template <typename T>
      struct as_int<T, long, false>
      : as_int<T, int> {};

    template <typename T>
      struct as_int<T, int, false>
      : as_int<T, short> {};

    template <typename T>
      struct as_int<T, short, false>
      : as_int<T, signed char> {};

    template <typename T>
      struct as_int<T, signed char, false>
  #ifdef __SIZEOF_INT128__
      : as_int<T, __int128> {};

    template <typename T>
      struct as_int<T, __int128, false>
  #endif // __SIZEOF_INT128__
      {};

    template <typename T>
      using as_int_t = typename as_int<T>::type;

    template <typename T, typename U = unsigned long long, bool = (sizeof(T) == sizeof(U))>
      struct as_unsigned;

    template <typename T, typename U>
      struct as_unsigned<T, U, true>
      { using type = U; };

    template <typename T>
      struct as_unsigned<T, unsigned long long, false>
      : as_unsigned<T, unsigned long> {};

    template <typename T>
      struct as_unsigned<T, unsigned long, false>
      : as_unsigned<T, unsigned int> {};

    template <typename T>
      struct as_unsigned<T, unsigned int, false>
      : as_unsigned<T, unsigned short> {};

    template <typename T>
      struct as_unsigned<T, unsigned short, false>
      : as_unsigned<T, unsigned char> {};

    template <typename T>
      struct as_unsigned<T, unsigned char, false>
  #ifdef __SIZEOF_INT128__
      : as_unsigned<T, unsigned __int128> {};

    template <typename T>
      struct as_unsigned<T, unsigned __int128, false>
  #endif // __SIZEOF_INT128__
      {};

    template <typename T>
      using as_unsigned_t = typename as_unsigned<T>::type;
}

namespace vir::detail
{
  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    using FloatingPoint = T;

  using namespace vir::stdx;

  template <typename T, int N>
    using deduced_simd = stdx::simd<T, stdx::simd_abi::deduce_t<T, N>>;

  template <typename T, int N>
    using deduced_simd_mask = stdx::simd_mask<T, stdx::simd_abi::deduce_t<T, N>>;

  template <typename To, typename From>
#ifdef __cpp_lib_bit_cast
    constexpr
#endif
    std::enable_if_t<sizeof(To) == sizeof(From), To>
    bit_cast(const From& x)
    {
#ifdef __cpp_lib_bit_cast
      return std::bit_cast<To>(x);
#else
      To r;
      std::memcpy(&r, &x, sizeof(x));
      return r;
#endif
    }
}

#endif // VIR_DETAILS_H
