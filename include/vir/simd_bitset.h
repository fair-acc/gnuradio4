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

#ifndef VIR_SIMD_BITSET_H_
#define VIR_SIMD_BITSET_H_

#include <bitset>
#include "simd.h"
#include "detail.h"

namespace vir
{
  template <typename T, typename A>
    std::bitset<stdx::simd_size_v<T, A>>
    to_bitset(const stdx::simd_mask<T, A>& k)
    {
#if VIR_GLIBCXX_STDX_SIMD
      return k.__to_bitset();
#else
      if constexpr (stdx::simd_size_v<T, A> == 1)
        return k[0];
      else
        {
          constexpr int N = stdx::simd_size_v<T, A>;
          std::bitset<N> bits;
          for (int i = 0; i < N; ++i)
            bits[i] = k[i];
          return bits;
        }
#endif
    }

  template <typename M>
    std::enable_if_t<stdx::is_simd_mask_v<M>, M>
    to_simd_mask(std::bitset<M::size()> bits)
    {
#if VIR_GLIBCXX_STDX_SIMD
      return M::__from_bitset(bits);
#else
      if constexpr (M::size() == 1)
        return M(bits[0]);
      else
        return M([bits](auto i) -> bool { return bits[i]; });
#endif
    }

  template <typename T, typename A>
    auto
    to_simd_mask(std::bitset<stdx::simd_size_v<T, A>> bits)
    -> decltype(to_simd_mask<stdx::simd_mask<T, A>>(bits))
    { return to_simd_mask<stdx::simd_mask<T, A>>(bits); }

  template <typename T, std::size_t N>
    auto
    to_simd_mask(std::bitset<N> bits)
    -> decltype(to_simd_mask<stdx::simd_mask<T, stdx::simd_abi::deduce_t<T, N>>>(bits))
    { return to_simd_mask<stdx::simd_mask<T, stdx::simd_abi::deduce_t<T, N>>>(bits); }
}

#endif // VIR_SIMD_BITSET_H_
