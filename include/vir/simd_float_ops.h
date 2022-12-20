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

#ifndef VIR_SIMD_FLOAT_OPS_H_
#define VIR_SIMD_FLOAT_OPS_H_

#include "detail.h"
#include <type_traits>

namespace vir
{
  namespace simd_float_ops
  {
    template <typename T, typename A>
      constexpr inline stdx::simd<T, A>
      operator&(const stdx::simd<detail::FloatingPoint<T>, A> a,
                const stdx::simd<T, A> b) noexcept
      {
        if constexpr (sizeof(T) <= sizeof(long long))
          {
            using V = stdx::simd<T, A>;
            using I = stdx::rebind_simd_t<meta::as_unsigned_t<T>, V>;
            return detail::bit_cast<V>(detail::bit_cast<I>(a) & detail::bit_cast<I>(b));
          }
        else
          return stdx::simd<T, A>([&](auto i) {
            using I = meta::as_unsigned_t<T>;
            return detail::bit_cast<T>(detail::bit_cast<I>(a[i]) & detail::bit_cast<I>(b[i]));
          });
      }

    template <typename T, typename A>
      constexpr inline stdx::simd<T, A>
      operator|(const stdx::simd<detail::FloatingPoint<T>, A> a,
                const stdx::simd<T, A> b) noexcept
      {
        if constexpr (sizeof(T) <= sizeof(long long))
          {
            using V = stdx::simd<T, A>;
            using I = stdx::rebind_simd_t<meta::as_unsigned_t<T>, V>;
            return detail::bit_cast<V>(detail::bit_cast<I>(a) | detail::bit_cast<I>(b));
          }
        else
          return stdx::simd<T, A>([&](auto i) {
            using I = meta::as_unsigned_t<T>;
            return detail::bit_cast<T>(detail::bit_cast<I>(a[i]) | detail::bit_cast<I>(b[i]));
          });
      }

    template <typename T, typename A>
      constexpr inline stdx::simd<T, A>
      operator^(const stdx::simd<detail::FloatingPoint<T>, A> a,
                const stdx::simd<T, A> b) noexcept
      {
        if constexpr (sizeof(T) <= sizeof(long long))
          {
            using V = stdx::simd<T, A>;
            using I = stdx::rebind_simd_t<meta::as_unsigned_t<T>, V>;
            return detail::bit_cast<V>(detail::bit_cast<I>(a) ^ detail::bit_cast<I>(b));
          }
        else
          return stdx::simd<T, A>([&](auto i) {
            using I = meta::as_unsigned_t<T>;
            return detail::bit_cast<T>(detail::bit_cast<I>(a[i]) ^ detail::bit_cast<I>(b[i]));
          });
      }
  }
}

#endif // VIR_SIMD_FLOAT_OPS_H_
