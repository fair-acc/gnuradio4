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

#ifndef VIR_SIMD_BIT_H
#define VIR_SIMD_BIT_H

#include "simd.h"
#include "detail.h"

namespace vir
{
  template <typename To, typename From>
    constexpr
    std::enable_if_t<std::conjunction_v<std::integral_constant<bool, sizeof(To) == sizeof(From)>,
                                        meta::is_simd_or_mask<To>,
                                        meta::is_simd_or_mask<From>>, To>
    simd_bit_cast(const From& x)
    {
#if VIR_GLIBCXX_STDX_SIMD
      return std::experimental::__proposed::simd_bit_cast<To>(x);
#else
      return detail::bit_cast<To>(x);
#endif
    }
}

#endif // VIR_SIMD_BIT_H
