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

#ifndef VIR_SIMD_CAST_H
#define VIR_SIMD_CAST_H

#include "simd.h"
#include "detail.h"

namespace vir
{
#if VIR_GLIBCXX_STDX_SIMD
  using std::experimental::parallelism_v2::__proposed::static_simd_cast;
#else
  using vir::stdx::static_simd_cast;
#endif
}

#endif // VIR_SIMD_CAST_H
