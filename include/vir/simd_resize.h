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

#ifndef VIR_SIMD_RESIZE_H
#define VIR_SIMD_RESIZE_H

#include "simd.h"
#include "detail.h"

namespace vir
{
  template <int N, typename T, typename A>
    constexpr vir::detail::deduced_simd<T, N>
    simd_resize(const stdx::simd<T, A>& x)
    {
      constexpr int xn = stdx::simd_size_v<T, A>;
      return vir::detail::deduced_simd<T, N>([&x](auto i) {
               if constexpr (i < xn)
                 return x[i];
               else
                 return T();
             });
    }

  template <int N, typename T, typename A>
    constexpr vir::detail::deduced_simd_mask<T, N>
    simd_resize(const stdx::simd_mask<T, A>& x)
    {
      constexpr int xn = stdx::simd_size_v<T, A>;
      return vir::detail::deduced_simd_mask<T, N>([&x](auto i) {
               if constexpr (i < xn)
                 return x[i];
               else
                 return false;
             });
    }

  template <typename V, typename A>
    constexpr std::enable_if_t<stdx::is_simd_v<V>, V>
    simd_size_cast(const stdx::simd<typename V::value_type, A>& x)
    {
      using T = typename V::value_type;
      constexpr int xn = stdx::simd_size_v<T, A>;
      return V([&x](auto i) {
               if constexpr (i < xn)
                 return x[i];
               else
                 return false;
             });
    }

  template <typename M, typename A>
    constexpr std::enable_if_t<stdx::is_simd_mask_v<M>, M>
    simd_size_cast(const stdx::simd_mask<typename M::simd_type::value_type, A>& x)
    {
#if VIR_GLIBCXX_STDX_SIMD
      return std::experimental::parallelism_v2::__proposed::resizing_simd_cast<M>(x);
#else
      using T = typename M::simd_type::value_type;
      constexpr int xn = stdx::simd_size_v<T, A>;
      return M([&x](auto i) {
               if constexpr (i < xn)
                 return x[i];
               else
                 return false;
             });
#endif
    }
}

#endif // VIR_SIMD_RESIZE_H
