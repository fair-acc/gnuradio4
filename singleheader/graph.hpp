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

#ifndef VIR_SIMD_H_
#define VIR_SIMD_H_

#if __cplusplus < 201703L
#error "simd requires C++17 or later"
#endif

#if __has_include (<experimental/simd>) && !defined VIR_DISABLE_STDX_SIMD
#include <experimental/simd>
#endif

#if defined __cpp_lib_experimental_parallel_simd && __cpp_lib_experimental_parallel_simd >= 201803

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}

#else

#include <cmath>
#include <cstring>
#ifdef _GLIBCXX_DEBUG_UB
#include <cstdio>
#endif
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef VIR_SIMD_TS_DROPIN
namespace std::experimental
{
  inline namespace parallelism_v2
#else
namespace vir::stdx
#endif
{
  using std::size_t;

  namespace detail
  {
    template <typename T>
      struct type_identity
      { using type = T; };

    template <typename T>
      using type_identity_t = typename type_identity<T>::type;

    constexpr size_t
    bit_ceil(size_t x)
    {
      size_t r = 1;
      while (r < x)
        r <<= 1;
      return r;
    }

    constexpr size_t
    bit_floor(size_t x)
    {
      size_t r = x;
      do {
        r = x;
        x &= x - 1;
      } while (x);
      return r;
    }

    template <typename T>
      typename T::value_type
      value_type_or_identity_impl(int);

    template <typename T>
      T
      value_type_or_identity_impl(float);

    template <typename T>
      using value_type_or_identity_t
        = decltype(value_type_or_identity_impl<T>(int()));

    class ExactBool
    {
      const bool data;

    public:
      constexpr ExactBool(bool b) : data(b) {}

      ExactBool(int) = delete;

      constexpr operator bool() const { return data; }
    };

    template <typename... Args>
      [[noreturn]] [[gnu::always_inline]] inline void
      invoke_ub([[maybe_unused]] const char* msg,
                [[maybe_unused]] const Args&... args)
      {
#ifdef _GLIBCXX_DEBUG_UB
        std::fprintf(stderr, msg, args...);
        __builtin_trap();
#else
        __builtin_unreachable();
#endif
      }

    template <typename T>
      using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename T>
      using L = std::numeric_limits<T>;

    template <bool B>
      using BoolConstant = std::integral_constant<bool, B>;

    template <size_t X>
      using SizeConstant = std::integral_constant<size_t, X>;

    template <size_t I, typename T, typename... Ts>
      constexpr auto
      pack_simd_subscript(const T& x0, const Ts&... xs)
      {
        if constexpr (I >= T::size())
          return pack_simd_subscript<I - T::size()>(xs...);
        else
          return x0[I];
      }

    template <class T>
      struct is_vectorizable : std::is_arithmetic<T>
      {};

    template <>
      struct is_vectorizable<bool> : std::false_type
      {};

    template <class T>
      inline constexpr bool is_vectorizable_v = is_vectorizable<T>::value;

    // Deduces to a vectorizable type
    template <typename T, typename = std::enable_if_t<is_vectorizable_v<T>>>
      using Vectorizable = T;

    // Deduces to a floating-point type
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
      using FloatingPoint = T;

    // Deduces to a signed integer type
    template <typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>,
                                                                         std::is_signed<T>>>>
      using SignedIntegral = T;

    // is_higher_integer_rank<T, U> (T has higher or equal integer rank than U)
    template <typename T, typename U, bool = (sizeof(T) > sizeof(U)),
              bool = (sizeof(T) == sizeof(U))>
      struct is_higher_integer_rank;

    template <typename T>
      struct is_higher_integer_rank<T, T, false, true>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, true, false>
      : public std::true_type
      {};

    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, false>
      : public std::false_type
      {};

    // this may fail for char -> short if sizeof(char) == sizeof(short)
    template <typename T, typename U>
      struct is_higher_integer_rank<T, U, false, true>
      : public std::is_same<decltype(std::declval<T>() + std::declval<U>()), T>
      {};

    // is_value_preserving<From, To>
    template <typename From, typename To, bool = std::is_arithmetic_v<From>,
              bool = std::is_arithmetic_v<To>>
      struct is_value_preserving;

    // ignore "signed/unsigned mismatch" in the following trait.
    // The implicit conversions will do the right thing here.
    template <typename From, typename To>
      struct is_value_preserving<From, To, true, true>
      : public BoolConstant<L<From>::digits <= L<To>::digits
                              && L<From>::max() <= L<To>::max()
                              && L<From>::lowest() >= L<To>::lowest()
                              && !(std::is_signed_v<From> && std::is_unsigned_v<To>)> {};

    template <typename T>
      struct is_value_preserving<T, bool, true, true>
      : public std::false_type {};

    template <>
      struct is_value_preserving<bool, bool, true, true>
      : public std::true_type {};

    template <typename T>
      struct is_value_preserving<T, T, true, true>
      : public std::true_type {};

    template <typename From, typename To>
      struct is_value_preserving<From, To, false, true>
      : public std::is_convertible<From, To> {};

    template <typename From, typename To,
              typename = std::enable_if_t<is_value_preserving<remove_cvref_t<From>, To>::value>>
      using ValuePreserving = From;

    template <typename From, typename To,
              typename DecayedFrom = remove_cvref_t<From>,
              typename = std::enable_if_t<std::conjunction<
                                            std::is_convertible<From, To>,
                                            std::disjunction<
                                              std::is_same<DecayedFrom, To>,
                                              std::is_same<DecayedFrom, int>,
                                              std::conjunction<std::is_same<DecayedFrom, unsigned>,
                                                               std::is_unsigned<To>>,
                                              is_value_preserving<DecayedFrom, To>>>::value>>
      using ValuePreservingOrInt = From;

    // LoadStorePtr / is_possible_loadstore_conversion
    template <typename Ptr, typename ValueType>
      struct is_possible_loadstore_conversion
      : std::conjunction<is_vectorizable<Ptr>, is_vectorizable<ValueType>>
      {};

    template <>
      struct is_possible_loadstore_conversion<bool, bool> : std::true_type {};

    // Deduces to a type allowed for load/store with the given value type.
    template <typename Ptr, typename ValueType,
              typename = std::enable_if_t<
                           is_possible_loadstore_conversion<Ptr, ValueType>::value>>
      using LoadStorePtr = Ptr;
  }

  namespace simd_abi
  {
    struct scalar
    {};

    template <typename>
      inline constexpr int max_fixed_size = 32;

    template <int N>
      struct fixed_size
      {};

    template <class T>
      using native =
        std::conditional_t<(sizeof(T) > 8),
                           scalar,
                           fixed_size<
#ifdef __AVX512F__
                             64
#elif defined __AVX2__
                             32
#elif defined __AVX__
                             std::is_floating_point_v<T> ? 32 : 16
#else
                             16
#endif
                               / sizeof(T)
                           >
                          >;

    template <class T>
      using compatible = std::conditional_t<(sizeof(T) > 8),
                                            scalar,
                                            fixed_size<16 / sizeof(T)>>;

    template <typename T, size_t N, typename...>
      struct deduce
      { using type = std::conditional_t<N == 1, scalar, fixed_size<int(N)>>; };

    template <typename T, size_t N, typename... Abis>
      using deduce_t = typename deduce<T, N, Abis...>::type;
  }

  // flags //
  struct element_aligned_tag
  {};

  struct vector_aligned_tag
  {};

  template <size_t>
    struct overaligned_tag
    {};

  inline constexpr element_aligned_tag element_aligned{};

  inline constexpr vector_aligned_tag vector_aligned{};

  template <size_t N>
    inline constexpr overaligned_tag<N> overaligned{};

  // fwd decls //
  template <class T, class A = simd_abi::compatible<T>>
    class simd;

  template <class T, class A = simd_abi::compatible<T>>
    class simd_mask;

  // aliases //
  template <class T>
    using native_simd = simd<T, simd_abi::native<T>>;

  template <class T>
    using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

  template <class T, int N>
    using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;

  template <class T, int N>
    using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

  // Traits //
  template <class T>
    struct is_abi_tag : std::false_type
    {};

  template <class T>
    inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

  template <>
    struct is_abi_tag<simd_abi::scalar> : std::true_type
    {};

  template <int N>
    struct is_abi_tag<simd_abi::fixed_size<N>> : std::true_type
    {};

  template <class T>
    struct is_simd : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_v = is_simd<T>::value;

  template <class T, class A>
    struct is_simd<simd<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_mask : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

  template <class T, class A>
    struct is_simd_mask<simd_mask<T, A>>
    : std::conjunction<detail::is_vectorizable<T>, is_abi_tag<A>>
    {};

  template <class T>
    struct is_simd_flag_type : std::false_type
    {};

  template <class T>
    inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

  template <class T, class A = simd_abi::compatible<T>>
    struct simd_size;

  template <class T, class A = simd_abi::compatible<T>>
    inline constexpr size_t simd_size_v = simd_size<T, A>::value;

  template <class T>
    struct simd_size<detail::Vectorizable<T>, simd_abi::scalar>
    : std::integral_constant<size_t, 1>
    {};

  template <class T, int N>
    struct simd_size<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    : std::integral_constant<size_t, N>
    {};

  template <class T, class U = typename T::value_type>
    struct memory_alignment;

  template <class T, class U = typename T::value_type>
    inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

  template <class T, class A, class U>
    struct memory_alignment<simd<T, A>, detail::Vectorizable<U>>
    : std::integral_constant<size_t, alignof(U)>
    {};

  template <class T, class A>
    struct memory_alignment<simd_mask<T, A>, bool>
    : std::integral_constant<size_t, alignof(bool)>
    {};

  template <class T, class V,
            class = typename std::conjunction<detail::is_vectorizable<T>,
                                              std::disjunction<is_simd<V>, is_simd_mask<V>>>::type>
    struct rebind_simd;

  template <class T, class V>
    using rebind_simd_t = typename rebind_simd<T, V>::type;

  template <class T, class U, class A>
    struct rebind_simd<T, simd<U, A>, std::true_type>
    { using type = simd<T, A>; };

  template <class T, class U, class A>
    struct rebind_simd<T, simd_mask<U, A>, std::true_type>
    { using type = simd_mask<T, A>; };

  template <int N, class V,
            class = typename std::conjunction<
                               detail::BoolConstant<(N > 0)>,
                               std::disjunction<is_simd<V>, is_simd_mask<V>>
                             >::type>
    struct resize_simd;

  template <int N, class V>
    using resize_simd_t = typename resize_simd<N, V>::type;

  template <int N, class T, class A>
    struct resize_simd<N, simd<T, A>, std::true_type>
    {
      using type = simd<T, std::conditional_t<N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  template <int N, class T, class A>
    struct resize_simd<N, simd_mask<T, A>, std::true_type>
    {
      using type = simd_mask<T, std::conditional_t<
                                  N == 1, simd_abi::scalar, simd_abi::fixed_size<N>>>;
    };

  // simd_mask (scalar)
  template <class T>
    class simd_mask<detail::Vectorizable<T>, simd_abi::scalar>
    {
      bool data;

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::scalar;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : data(x) {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : data(mem[0])
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, simd_mask k, Flags)
        : data(k ? mem[0] : false)
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { data = mem[0]; }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // negation
      constexpr simd_mask
      operator!() const
      { return simd_mask(not data); }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data && y.data); }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data || y.data); }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data & y.data); }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data | y.data); }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data ^ y.data); }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        x.data &= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        x.data |= y.data;
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        x.data ^= y.data;
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data == y.data); }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      { return simd_mask(x.data != y.data); }
    };

  // simd_mask (fixed_size)
  template <class T, int N>
    class simd_mask<detail::Vectorizable<T>, simd_abi::fixed_size<N>>
    {
    private:
      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
        split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<M>>&);

      bool data[N];

      template <typename F, size_t... Is>
        constexpr
        simd_mask(std::index_sequence<Is...>, F&& init)
        : data {init(detail::SizeConstant<Is>())...}
        {}

    public:
      using value_type = bool;
      using reference = bool&;
      using abi_type = simd_abi::fixed_size<N>;
      using simd_type = simd<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd_mask() = default;
      constexpr simd_mask(const simd_mask&) = default;
      constexpr simd_mask(simd_mask&&) noexcept = default;
      constexpr simd_mask& operator=(const simd_mask&) = default;
      constexpr simd_mask& operator=(simd_mask&&) noexcept = default;

      // explicit broadcast constructor
      explicit constexpr
      simd_mask(bool x)
      : simd_mask([x](size_t) { return x; })
      {}

      template <typename F>
        explicit constexpr
        simd_mask(F&& gen, std::enable_if_t<
                             std::is_same_v<decltype(std::declval<F>()(detail::SizeConstant<0>())),
                                            value_type>>* = nullptr)
        : simd_mask(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // implicit conversions
      template <typename U>
        constexpr
        simd_mask(const simd_mask<U, abi_type>& x)
        : simd_mask([&x](auto i) { return x[i]; })
        {}

      // load constructor
      template <typename Flags>
        simd_mask(const value_type* mem, Flags)
        : simd_mask([mem](size_t i) { return mem[i]; })
        {}

      template <typename Flags>
        simd_mask(const value_type* mem, const simd_mask& k, Flags)
        : simd_mask([mem, &k](size_t i) { return k[i] ? mem[i] : false; })
        {}

      // loads [simd_mask.load]
      template <typename Flags>
        void
        copy_from(const value_type* mem, Flags)
        { std::memcpy(data, mem, N * sizeof(bool)); }

      // stores [simd_mask.store]
      template <typename Flags>
        void
        copy_to(value_type* mem, Flags) const
        { std::memcpy(mem, data, N * sizeof(bool)); }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // negation
      constexpr simd_mask
      operator!() const
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = !data[i];
        return r;
      }

      // simd_mask binary operators [simd_mask.binary]
      friend constexpr simd_mask
      operator&&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator||(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator&(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] & y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator|(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] | y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator^(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] ^ y.data[i];
        return r;
      }

      friend constexpr simd_mask&
      operator&=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] &= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator|=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] |= y.data[i];
        return x;
      }

      friend constexpr simd_mask&
      operator^=(simd_mask& x, const simd_mask& y)
      {
        for (int i = 0; i < N; ++i)
          x.data[i] ^= y.data[i];
        return x;
      }

      // simd_mask compares [simd_mask.comparison]
      friend constexpr simd_mask
      operator==(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] == y.data[i];
        return r;
      }

      friend constexpr simd_mask
      operator!=(const simd_mask& x, const simd_mask& y)
      {
        simd_mask r {};
        for (int i = 0; i < N; ++i)
          r.data[i] = x.data[i] != y.data[i];
        return r;
      }
    };

  // simd_mask reductions [simd_mask.reductions]
  template <typename T>
    constexpr bool
    all_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    any_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return k[0]; }

  template <typename T>
    constexpr bool
    none_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return not k[0]; }

  template <typename T>
    constexpr bool
    some_of(simd_mask<T, simd_abi::scalar> k) noexcept
    { return false; }

  template <typename T>
    constexpr int
    popcount(simd_mask<T, simd_abi::scalar> k) noexcept
    { return static_cast<int>(k[0]); }

  template <typename T>
    constexpr int
    find_first_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_first_set(empty mask) is UB");
      return 0;
    }

  template <typename T>
    constexpr int
    find_last_set(simd_mask<T, simd_abi::scalar> k) noexcept
    {
      if (not k[0])
        detail::invoke_ub("find_last_set(empty mask) is UB");
      return 0;
    }

  template <typename T, int N>
    constexpr bool
    all_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (not k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    any_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr bool
    none_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return false;
        }
      return true;
    }

  template <typename T, int N>
    constexpr bool
    some_of(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      bool last = k[0];
      for (int i = 1; i < N; ++i)
        {
          if (last != k[i])
            return true;
        }
      return false;
    }

  template <typename T, int N>
    constexpr int
    popcount(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      int cnt = k[0];
      for (int i = 1; i < N; ++i)
        cnt += k[i];
      return cnt;
    }

  template <typename T, int N>
    constexpr int
    find_first_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = 0; i < N; ++i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_first_set(empty mask) is UB");
    }

  template <typename T, int N>
    constexpr int
    find_last_set(const simd_mask<T, simd_abi::fixed_size<N>>& k) noexcept
    {
      for (int i = N - 1; i >= 0; --i)
        {
          if (k[i])
            return i;
        }
      detail::invoke_ub("find_last_set(empty mask) is UB");
    }

  constexpr bool
  all_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  any_of(detail::ExactBool x) noexcept
  { return x; }

  constexpr bool
  none_of(detail::ExactBool x) noexcept
  { return !x; }

  constexpr bool
  some_of(detail::ExactBool) noexcept
  { return false; }

  constexpr int
  popcount(detail::ExactBool x) noexcept
  { return x; }

  constexpr int
  find_first_set(detail::ExactBool)
  { return 0; }

  constexpr int
  find_last_set(detail::ExactBool)
  { return 0; }

  // scalar_simd_int_base
  template <class T, bool = std::is_integral_v<T>>
    class scalar_simd_int_base
    {};

  template <class T>
    class scalar_simd_int_base<T, true>
    {
      using Derived = simd<T, simd_abi::scalar>;

      constexpr T&
      d() noexcept
      { return static_cast<Derived*>(this)->data; }

      constexpr const T&
      d() const noexcept
      { return static_cast<const Derived*>(this)->data; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, Derived x)
      {
        lhs.d() %= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, Derived x)
      {
        lhs.d() &= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, Derived x)
      {
        lhs.d() |= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, Derived x)
      {
        lhs.d() ^= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, Derived x)
      {
        lhs.d() <<= x.d();
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, Derived x)
      {
        lhs.d() >>= x.d();
        return lhs;
      }

      friend constexpr Derived
      operator%(Derived x, Derived y)
      {
        x.d() %= y.d();
        return x;
      }

      friend constexpr Derived
      operator&(Derived x, Derived y)
      {
        x.d() &= y.d();
        return x;
      }

      friend constexpr Derived
      operator|(Derived x, Derived y)
      {
        x.d() |= y.d();
        return x;
      }

      friend constexpr Derived
      operator^(Derived x, Derived y)
      {
        x.d() ^= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, Derived y)
      {
        x.d() <<= y.d();
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, Derived y)
      {
        x.d() >>= y.d();
        return x;
      }

      friend constexpr Derived
      operator<<(Derived x, int y)
      {
        x.d() <<= y;
        return x;
      }

      friend constexpr Derived
      operator>>(Derived x, int y)
      {
        x.d() >>= y;
        return x;
      }

      constexpr Derived
      operator~() const
      { return Derived(static_cast<T>(~d())); }
    };

  // simd (scalar)
  template <class T>
    class simd<T, simd_abi::scalar>
    : public scalar_simd_int_base<T>
    {
      friend class scalar_simd_int_base<T>;

      T data;

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::scalar;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return 1; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : data(value)
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : data(gen(detail::SizeConstant<0>()))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : data(mem[0])
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        { data = mem[0]; }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        { mem[0] = data; }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data;
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        ++data;
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        ++data;
        return r;
      }

      constexpr simd&
      operator--()
      {
        --data;
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        --data;
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type(not data); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return -data; }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      { return lhs = lhs + x; }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      { return lhs = lhs - x; }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      { return lhs = lhs * x; }

      constexpr friend simd&
        operator/=(simd& lhs, const simd& x)
      { return lhs = lhs / x; }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { simd r = x; r.data += y.data; return r; }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { simd r = x; r.data -= y.data; return r; }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { simd r = x; r.data *= y.data; return r; }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { simd r = x; r.data /= y.data; return r; }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type(x.data == y.data); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type(x.data != y.data); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type(x.data < y.data); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type(x.data <= y.data); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type(x.data > y.data); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type(x.data >= y.data); }
    };

  // fixed_simd_int_base
  template <class T, int N, bool = std::is_integral_v<T>>
    class fixed_simd_int_base
    {};

  template <class T, int N>
    class fixed_simd_int_base<T, N, true>
    {
      using Derived = simd<T, simd_abi::fixed_size<N>>;

      constexpr T&
      d(int i) noexcept
      { return static_cast<Derived*>(this)->data[i]; }

      constexpr const T&
      d(int i) const noexcept
      { return static_cast<const Derived*>(this)->data[i]; }

    public:
      friend constexpr Derived&
      operator%=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) %= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator&=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) &= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator|=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) |= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator^=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) ^= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator<<=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) <<= x.d(i);
        return lhs;
      }

      friend constexpr Derived&
      operator>>=(Derived& lhs, const Derived& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.d(i) >>= x.d(i);
        return lhs;
      }

      friend constexpr Derived
      operator%(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] % y[i]; }); }

      friend constexpr Derived
      operator&(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] & y[i]; }); }

      friend constexpr Derived
      operator|(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] | y[i]; }); }

      friend constexpr Derived
      operator^(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] ^ y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] << y[i]; }); }

      friend constexpr Derived
      operator>>(const Derived& x, const Derived& y)
      { return Derived([&](auto i) -> T { return x[i] >> y[i]; }); }

      friend constexpr Derived
      operator<<(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] << y; }); }

      friend constexpr Derived
      operator>>(const Derived& x, int y)
      { return Derived([&](auto i) -> T { return x[i] >> y; }); }

      constexpr Derived
      operator~() const
      { return Derived([&](auto i) -> T { return ~d(i); }); }
    };

  // simd (fixed_size)
  template <class T, int N>
    class simd<T, simd_abi::fixed_size<N>>
    : public fixed_simd_int_base<T, N>
    {
    private:
      friend class fixed_simd_int_base<T, N>;

      template <typename V, int M, size_t Parts>
        friend std::enable_if_t<M == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
        split(const simd<typename V::value_type, simd_abi::fixed_size<M>>&);

      template <size_t... Sizes, typename U>
        friend std::tuple<simd<U, simd_abi::deduce_t<U, int(Sizes)>>...>
        split(const simd<U, simd_abi::fixed_size<int((Sizes + ...))>>&);

      T data[N];

      template <typename F, size_t... Is>
        constexpr
        simd(std::index_sequence<Is...>, F&& init)
        : data {static_cast<value_type>(init(detail::SizeConstant<Is>()))...}
        {}

    public:
      using value_type = T;
      using reference = T&;
      using abi_type = simd_abi::fixed_size<N>;
      using mask_type = simd_mask<T, abi_type>;

      static constexpr size_t size() noexcept
      { return N; }

      constexpr simd() = default;
      constexpr simd(const simd&) = default;
      constexpr simd(simd&&) noexcept = default;
      constexpr simd& operator=(const simd&) = default;
      constexpr simd& operator=(simd&&) noexcept = default;

      // simd constructors
      template <typename U>
        constexpr
        simd(detail::ValuePreservingOrInt<U, value_type>&& value) noexcept
        : simd([v = static_cast<value_type>(value)](size_t) { return v; })
        {}

      // conversion constructors
      template <typename U,
                typename = std::enable_if_t<
                             std::conjunction_v<detail::is_value_preserving<U, value_type>,
                                                detail::is_higher_integer_rank<value_type, U>>>>
        constexpr
        simd(const simd<U, abi_type>& x)
        : simd([&x](auto i) { return static_cast<value_type>(x[i]); })
        {}

      // generator constructor
      template <typename F>
        explicit constexpr
        simd(F&& gen, detail::ValuePreservingOrInt<
                        decltype(std::declval<F>()(std::declval<detail::SizeConstant<0>&>())),
                        value_type>* = nullptr)
        : simd(std::make_index_sequence<N>(), std::forward<F>(gen))
        {}

      // load constructor
      template <typename U, typename Flags>
        simd(const U* mem, Flags)
        : simd([mem](auto i) -> value_type { return mem[i]; })
        {}

      // loads [simd.load]
      template <typename U, typename Flags>
        void
        copy_from(const detail::Vectorizable<U>* mem, Flags)
        {
          for (int i = 0; i < N; ++i)
            data[i] = mem[i];
        }

      // stores [simd.store]
      template <typename U, typename Flags>
        void
        copy_to(detail::Vectorizable<U>* mem, Flags) const
        {
          for (int i = 0; i < N; ++i)
            mem[i] = data[i];
        }

      // scalar access
      constexpr reference
      operator[](size_t i)
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      constexpr value_type
      operator[](size_t i) const
      {
        if (i >= size())
          detail::invoke_ub("Subscript %d is out of range [0, %d]", i, size() - 1);
        return data[i];
      }

      // increment and decrement:
      constexpr simd&
      operator++()
      {
        for (int i = 0; i < N; ++i)
          ++data[i];
        return *this;
      }

      constexpr simd
      operator++(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          ++data[i];
        return r;
      }

      constexpr simd&
      operator--()
      {
        for (int i = 0; i < N; ++i)
          --data[i];
        return *this;
      }

      constexpr simd
      operator--(int)
      {
        simd r = *this;
        for (int i = 0; i < N; ++i)
          --data[i];
        return r;
      }

      // unary operators
      constexpr mask_type
      operator!() const
      { return mask_type([&](auto i) { return !data[i]; }); }

      constexpr simd
      operator+() const
      { return *this; }

      constexpr simd
      operator-() const
      { return simd([&](auto i) -> value_type { return -data[i]; }); }

      // compound assignment [simd.cassign]
      constexpr friend simd&
      operator+=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] += x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator-=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] -= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator*=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] *= x.data[i];
        return lhs;
      }

      constexpr friend simd&
      operator/=(simd& lhs, const simd& x)
      {
        for (int i = 0; i < N; ++i)
          lhs.data[i] /= x.data[i];
        return lhs;
      }

      // binary operators [simd.binary]
      constexpr friend simd
      operator+(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] + y.data[i]; }); }

      constexpr friend simd
      operator-(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] - y.data[i]; }); }

      constexpr friend simd
      operator*(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] * y.data[i]; }); }

      constexpr friend simd
      operator/(const simd& x, const simd& y)
      { return simd([&](auto i) { return x.data[i] / y.data[i]; }); }

      // compares [simd.comparison]
      constexpr friend mask_type
      operator==(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] == y.data[i]; }); }

      constexpr friend mask_type
      operator!=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] != y.data[i]; }); }

      constexpr friend mask_type
      operator<(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] < y.data[i]; }); }

      constexpr friend mask_type
      operator<=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] <= y.data[i]; }); }

      constexpr friend mask_type
      operator>(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] > y.data[i]; }); }

      constexpr friend mask_type
      operator>=(const simd& x, const simd& y)
      { return mask_type([&](auto i) { return x.data[i] >= y.data[i]; }); }
    };

  // casts [simd.casts]
  // static_simd_cast
  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd<T, A>
    static_simd_cast(const simd<U, A>& x)
    { return simd<T, A>([&x](auto i) { return static_cast<T>(x[i]); }); }

  template <typename V, typename U, typename A,
            typename = std::enable_if_t<is_simd_v<V>>>
    constexpr V
    static_simd_cast(const simd<U, A>& x)
    { return V([&x](auto i) { return static_cast<typename V::value_type>(x[i]); }); }

  template <typename T, typename U, typename A,
            typename = std::enable_if_t<detail::is_vectorizable_v<T>>>
    constexpr simd_mask<T, A>
    static_simd_cast(const simd_mask<U, A>& x)
    { return simd_mask<T, A>([&x](auto i) { return x[i]; }); }

  template <typename M, typename U, typename A,
            typename = std::enable_if_t<M::size() == simd_size_v<U, A>>>
    constexpr M
    static_simd_cast(const simd_mask<U, A>& x)
    { return M([&x](auto i) { return x[i]; }); }

  // simd_cast
  template <typename T, typename U, typename A,
            typename To = detail::value_type_or_identity_t<T>>
    constexpr auto
    simd_cast(const simd<detail::ValuePreserving<U, To>, A>& x)
    -> decltype(static_simd_cast<T>(x))
    { return static_simd_cast<T>(x); }

  // to_fixed_size
  template <typename T, int N>
    constexpr fixed_size_simd<T, N>
    to_fixed_size(const fixed_size_simd<T, N>& x)
    { return x; }

  template <typename T, int N>
    constexpr fixed_size_simd_mask<T, N>
    to_fixed_size(const fixed_size_simd_mask<T, N>& x)
    { return x; }

  template <typename T>
    constexpr fixed_size_simd<T, 1>
    to_fixed_size(const simd<T> x)
    { return x[0]; }

  template <typename T>
    constexpr fixed_size_simd_mask<T, 1>
    to_fixed_size(const simd_mask<T> x)
    { return fixed_size_simd_mask<T, 1>(x[0]); }

  // to_native
  template <typename T>
    constexpr simd<T>
    to_native(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_native(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // to_compatible
  template <typename T>
    constexpr simd<T>
    to_compatible(const fixed_size_simd<T, 1> x)
    { return x[0]; }

  template <typename T>
    constexpr simd_mask<T>
    to_compatible(const fixed_size_simd_mask<T, 1> x)
    { return simd_mask<T>(x[0]); }

  // split(simd)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_v<V>, std::array<V, Parts>>
    split(const simd<typename V::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split(simd_mask)
  template <typename V, int N, size_t Parts = N / V::size()>
    std::enable_if_t<N == Parts * V::size() && is_simd_mask_v<V>, std::array<V, Parts>>
    split(const simd_mask<typename V::simd_type::value_type, simd_abi::fixed_size<N>>& x)
    {
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>)
               -> std::array<V, Parts> {
                 return {V(data + Is * V::size(), element_aligned)...};
               }(std::make_index_sequence<Parts>());
    }

  // split<Sizes...>
  template <size_t... Sizes, typename T>
    std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>
    split(const simd<T, simd_abi::fixed_size<int((Sizes + ...))>>& x)
    {
      using R = std::tuple<simd<T, simd_abi::deduce_t<T, int(Sizes)>>...>;
      const auto* data = x.data;
      return [&]<size_t... Is>(std::index_sequence<Is...>) -> R {
        constexpr size_t offsets[sizeof...(Sizes)] = {
          []<size_t... Js>(std::index_sequence<Js...>) {
            constexpr size_t sizes[sizeof...(Sizes)] = {Sizes...};
            return (sizes[Js] + ... + 0);
          }(std::make_index_sequence<Is>())...
        };
        return {simd<T, simd_abi::deduce_t<T, int(Sizes)>>(data + offsets[Is],
                                                           element_aligned)...};
      }(std::make_index_sequence<sizeof...(Sizes)>());
    }

  // split<V>(V)
  template <typename V>
    std::enable_if_t<std::disjunction_v<is_simd<V>, is_simd_mask<V>>, std::array<V, 1>>
    split(const V& x)
    { return {x}; }

  // concat(simd...)
  template <typename T, typename... As>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd<T, As>&... xs)
    {
      using R = simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(simd_mask...)
  template <typename T, typename... As>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>
    concat(const simd_mask<T, As>&... xs)
    {
      using R = simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>;
      return R([&](auto i) -> bool {
               return detail::pack_simd_subscript<i>(xs...);
             });
    }

  // concat(array<simd>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) {
               return x[i / K][i % K];
             });
    }

  // concat(array<simd_mask>)
  template <typename T, typename A, size_t N>
    inline constexpr
    simd_mask<T, simd_abi::deduce_t<T, N * simd_size_v<T, A>>>
    concat(const std::array<simd_mask<T, A>, N>& x)
    {
      constexpr int K = simd_size_v<T, A>;
      using R = simd_mask<T, simd_abi::deduce_t<T, N * K>>;
      return R([&](auto i) -> bool {
               return x[i / K][i % K];
             });
    }

  // const_where_expression<M, T>
  template <typename M, typename V>
    class const_where_expression
    {
      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const M& m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const M& kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      {
        return V([&](auto i) {
                 return m_k[i] ? static_cast<value_type>(-m_value[i]) : m_value[i];
               });
      }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          return V([&](auto i) {
                   return m_k[i] ? static_cast<value_type>(mem[i]) : m_value[i];
                 });
        }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                mem[i] = static_cast<Up>(m_value[i]);
            }
        }
    };

  // const_where_expression<bool, T>
  template <typename V>
    class const_where_expression<bool, V>
    {
      using M = bool;

      static_assert(std::is_same_v<V, detail::remove_cvref_t<V>>);

      struct Wrapper { using value_type = V; };

    protected:
      using value_type =
        typename std::conditional_t<std::is_arithmetic_v<V>, Wrapper, V>::value_type;

      friend const M&
      get_mask(const const_where_expression& x)
      { return x.m_k; }

      friend const V&
      get_lvalue(const const_where_expression& x)
      { return x.m_value; }

      const bool m_k;
      V& m_value;

    public:
      const_where_expression(const const_where_expression&) = delete;
      const_where_expression& operator=(const const_where_expression&) = delete;

      constexpr const_where_expression(const bool kk, const V& dd)
      : m_k(kk), m_value(const_cast<V&>(dd)) {}

      constexpr V
      operator-() const &&
      { return m_k ? -m_value : m_value; }

      template <typename Up, typename Flags>
        [[nodiscard]] constexpr V
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        { return m_k ? static_cast<V>(mem[0]) : m_value; }

      template <typename Up, typename Flags>
        constexpr void
        copy_to(detail::LoadStorePtr<Up, value_type>* mem, Flags) const &&
        {
          if (m_k)
            mem[0] = m_value;
        }
    };

  // where_expression<M, T>
  template <typename M, typename V>
    class where_expression : public const_where_expression<M, V>
    {
      static_assert(not std::is_const_v<V>,
                    "where_expression may only be instantiated with a non-const V parameter");

      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

      static_assert(std::is_same_v<typename M::abi_type, typename V::abi_type>);
      static_assert(M::size() == V::size());

      friend V&
      get_lvalue(where_expression& x)
      { return x.m_value; }

      template <typename Up>
        constexpr auto
        as_simd(Up&& x)
        {
          using UU = detail::remove_cvref_t<Up>;
          if constexpr (std::is_same_v<V, UU>)
            return x;
          else if constexpr (std::is_convertible_v<Up&&, value_type>)
            return V(static_cast<value_type>(static_cast<Up&&>(x)));
          else if constexpr (std::is_convertible_v<Up&&, V>)
            return static_cast<V>(static_cast<Up&&>(x));
          else
            return static_simd_cast<V>(static_cast<Up&&>(x));
        }

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd)
      {}

      template <typename Up>
        constexpr void
        operator=(Up&& x) &&
        {
          const V& rhs = as_simd(x);
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = rhs[i];
            }
        }

#define SIMD_OP_(op)                              \
      template <typename Up>                      \
        constexpr void                            \
        operator op##=(Up&& x) &&                 \
        {                                         \
          const V& rhs = as_simd(x);              \
          for (size_t i = 0; i < V::size(); ++i)  \
            {                                     \
              if (m_k[i])                         \
                m_value[i] op##= rhs[i];          \
            }                                     \
        }                                         \
      static_assert(true)
      SIMD_OP_(+);
      SIMD_OP_(-);
      SIMD_OP_(*);
      SIMD_OP_(/);
      SIMD_OP_(%);
      SIMD_OP_(&);
      SIMD_OP_(|);
      SIMD_OP_(^);
      SIMD_OP_(<<);
      SIMD_OP_(>>);
#undef SIMD_OP_

      constexpr void operator++() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator++(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              ++m_value[i];
          }
      }

      constexpr void operator--() &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      constexpr void operator--(int) &&
      {
        for (size_t i = 0; i < V::size(); ++i)
          {
            if (m_k[i])
              --m_value[i];
          }
      }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        {
          for (size_t i = 0; i < V::size(); ++i)
            {
              if (m_k[i])
                m_value[i] = mem[i];
            }
        }
    };

  // where_expression<bool, T>
  template <typename V>
    class where_expression<bool, V> : public const_where_expression<bool, V>
    {
      using M = bool;
      using typename const_where_expression<M, V>::value_type;
      using const_where_expression<M, V>::m_k;
      using const_where_expression<M, V>::m_value;

    public:
      where_expression(const where_expression&) = delete;
      where_expression& operator=(const where_expression&) = delete;

      constexpr where_expression(const M& kk, V& dd)
      : const_where_expression<M, V>(kk, dd) {}

#define SIMD_OP_(op)                                \
      template <typename Up>                        \
        constexpr void operator op(Up&& x) &&       \
        { if (m_k) m_value op static_cast<Up&&>(x); }

      SIMD_OP_(=)
      SIMD_OP_(+=)
      SIMD_OP_(-=)
      SIMD_OP_(*=)
      SIMD_OP_(/=)
      SIMD_OP_(%=)
      SIMD_OP_(&=)
      SIMD_OP_(|=)
      SIMD_OP_(^=)
      SIMD_OP_(<<=)
      SIMD_OP_(>>=)
#undef SIMD_OP_

      constexpr void operator++() &&
      { if (m_k) ++m_value; }

      constexpr void operator++(int) &&
      { if (m_k) ++m_value; }

      constexpr void operator--() &&
      { if (m_k) --m_value; }

      constexpr void operator--(int) &&
      { if (m_k) --m_value; }

      // intentionally hides const_where_expression::copy_from
      template <typename Up, typename Flags>
        constexpr void
        copy_from(const detail::LoadStorePtr<Up, value_type>* mem, Flags) &&
        { if (m_k) m_value = mem[0]; }
    };

  // where
  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k, simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd<Tp, Ap>>
    where(const typename simd<Tp, Ap>::mask_type& k,
          const simd<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr const_where_expression<simd_mask<Tp, Ap>, simd_mask<Tp, Ap>>
    where(const std::remove_const_t<simd_mask<Tp, Ap>>& k,
          const simd_mask<Tp, Ap>& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr where_expression<bool, Tp>
    where(detail::ExactBool k, Tp& value)
    { return {k, value}; }

  template <typename Tp>
    constexpr const_where_expression<bool, Tp>
    where(detail::ExactBool k, const Tp& value)
    { return {k, value}; }

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, simd<Tp, Ap>& value) = delete;

  template <typename Tp, typename Ap>
    constexpr void
    where(bool k, const simd<Tp, Ap>& value) = delete;

  // reductions [simd.reductions]
  template <typename T, typename A, typename BinaryOperation = std::plus<>>
    constexpr T
    reduce(const simd<T, A>& v,
           BinaryOperation binary_op = BinaryOperation())
    {
      constexpr int N = simd_size_v<T, A>;
      if constexpr (N > 3)
        {
          constexpr int N2 = detail::bit_floor(N / 2);
          constexpr int NRem = N - 2 * N2;
          if constexpr (NRem > 0)
            {
              const auto [l, r, rem] = split<N2, N2, N - 2 * N2>(v);
              return binary_op(reduce(binary_op(l, r), binary_op), reduce(rem, binary_op));
            }
          else
            {
              const auto [l, r] = split<N2, N2>(v);
              return reduce(binary_op(l, r), binary_op);
            }
        }
      else
        {
          T r = v[0];
          for (size_t i = 1; i < simd_size_v<T, A>; ++i)
            r = binary_op(r, v[i]);
          return r;
        }
    }

  template <typename M, typename V, typename BinaryOperation = std::plus<>>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x,
        typename V::value_type identity_element,
        BinaryOperation binary_op)
    {
      const M& k = get_mask(x);
      const V& v = get_lvalue(x);
      auto r = identity_element;
      if (any_of(k)) [[likely]]
        {
          for (size_t i = 0; i < V::size(); ++i)
            if (k[i])
              r = binary_op(r, v[i]);
        }
      return r;
    }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::plus<> binary_op = {})
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::multiplies<> binary_op)
    { return reduce(x, 1, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_and<> binary_op)
    { return reduce(x, ~typename V::value_type(), binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_or<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename M, typename V>
    typename V::value_type
    reduce(const const_where_expression<M, V>& x, std::bit_xor<> binary_op)
    { return reduce(x, 0, binary_op); }

  template <typename T, typename A>
    constexpr T
    hmin(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename T, typename A>
    constexpr T
    hmax(const simd<T, A>& v) noexcept
    {
      return reduce(v, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  template <typename M, typename V>
    constexpr typename V::value_type
    hmin(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::max();
#else
        std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::min;
               return min(l, r);
             });
    }

  template <typename M, typename V>
    constexpr
    typename V::value_type
    hmax(const const_where_expression<M, V>& x) noexcept
    {
      using T = typename V::value_type;
      constexpr T id_elem =
#ifdef __FINITE_MATH_ONLY__
        std::numeric_limits<T>::lowest();
#else
        -std::numeric_limits<T>::infinity();
#endif
      return reduce(x, id_elem, [](const auto& l, const auto& r) {
               using std::max;
               return max(l, r);
             });
    }

  // algorithms [simd.alg]
  template <typename T, typename A>
    constexpr simd<T, A>
    min(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::min(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    max(const simd<T, A>& a, const simd<T, A>& b)
    { return simd<T, A>([&](auto i) { return std::max(a[i], b[i]); }); }

  template <typename T, typename A>
    constexpr
    std::pair<simd<T, A>, simd<T, A>>
    minmax(const simd<T, A>& a, const simd<T, A>& b)
    { return {min(a, b), max(a, b)}; }

  template <typename T, typename A>
    constexpr simd<T, A>
    clamp(const simd<T, A>& v, const simd<T, A>& lo,
        const simd<T, A>& hi)
    { return simd<T, A>([&](auto i) { return std::clamp(v[i], lo[i], hi[i]); }); }

  // math
#define SIMD_MATH_1ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return return_temp<T, A>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_1ARG_FIXED(name, R)                                                              \
  template <typename T, typename A>                                                                \
    constexpr fixed_size_simd<R, simd_size_v<T, A>>                                                \
    name(const simd<detail::FloatingPoint<T>, A>& x) noexcept                                      \
    { return fixed_size_simd<R, simd_size_v<T, A>>([&x](auto i) { return std::name(x[i]); }); }

#define SIMD_MATH_2ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x, const simd<T, A>& y) noexcept                 \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }                   \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i]); }); }

#define SIMD_MATH_3ARG(name, return_temp)                                                          \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const simd<T, A>& y, const simd<T, A> &z) noexcept                                        \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const simd<detail::FloatingPoint<T>, A>& x,                                               \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const simd<detail::FloatingPoint<T>, A>& y,                                               \
         const detail::type_identity_t<simd<T, A>> &z) noexcept                                    \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }             \
                                                                                                   \
  template <typename T, typename A>                                                                \
    constexpr return_temp<T, A>                                                                    \
    name(const detail::type_identity_t<simd<T, A>>& x,                                             \
         const detail::type_identity_t<simd<T, A>>& y,                                             \
         const simd<detail::FloatingPoint<T>, A> &z) noexcept                                      \
    { return return_temp<T, A>([&](auto i) { return std::name(x[i], y[i], z[i]); }); }

  template <typename T, typename A, typename U = detail::SignedIntegral<T>>
    constexpr simd<T, A>
    abs(const simd<T, A>& x) noexcept
    { return simd<T, A>([&x](auto i) { return std::abs(x[i]); }); }

  SIMD_MATH_1ARG(abs, simd)
  SIMD_MATH_1ARG(isnan, simd_mask)
  SIMD_MATH_1ARG(isfinite, simd_mask)
  SIMD_MATH_1ARG(isinf, simd_mask)
  SIMD_MATH_1ARG(isnormal, simd_mask)
  SIMD_MATH_1ARG(signbit, simd_mask)
  SIMD_MATH_1ARG_FIXED(fpclassify, int)

  SIMD_MATH_2ARG(hypot, simd)
  SIMD_MATH_3ARG(hypot, simd)

  template <typename T, typename A>
    constexpr simd<T, A>
    remquo(const simd<T, A>& x, const simd<T, A>& y,
           fixed_size_simd<int, simd_size_v<T, A>>* quo) noexcept
    { return simd<T, A>([&x, &y, quo](auto i) { return std::remquo(x[i], y[i], &(*quo)[i]); }); }

  SIMD_MATH_1ARG(erf, simd)
  SIMD_MATH_1ARG(erfc, simd)
  SIMD_MATH_1ARG(tgamma, simd)
  SIMD_MATH_1ARG(lgamma, simd)

  SIMD_MATH_2ARG(pow, simd)
  SIMD_MATH_2ARG(fmod, simd)
  SIMD_MATH_2ARG(remainder, simd)
  SIMD_MATH_2ARG(nextafter, simd)
  SIMD_MATH_2ARG(copysign, simd)
  SIMD_MATH_2ARG(fdim, simd)
  SIMD_MATH_2ARG(fmax, simd)
  SIMD_MATH_2ARG(fmin, simd)
  SIMD_MATH_2ARG(isgreater, simd_mask)
  SIMD_MATH_2ARG(isgreaterequal, simd_mask)
  SIMD_MATH_2ARG(isless, simd_mask)
  SIMD_MATH_2ARG(islessequal, simd_mask)
  SIMD_MATH_2ARG(islessgreater, simd_mask)
  SIMD_MATH_2ARG(isunordered, simd_mask)

  template <typename T, typename A>
    constexpr simd<T, A>
    modf(const simd<detail::FloatingPoint<T>, A>& x, simd<T, A>* iptr) noexcept
    { return simd<T, A>([&x, iptr](auto i) { return std::modf(x[i], &(*iptr)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    frexp(const simd<detail::FloatingPoint<T>, A>& x,
          fixed_size_simd<int, simd_size_v<T, A>>* exp) noexcept
    { return simd<T, A>([&x, exp](auto i) { return std::frexp(x[i], &(*exp)[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbln(const simd<detail::FloatingPoint<T>, A>& x,
            const fixed_size_simd<long int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbln(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    scalbn(const simd<detail::FloatingPoint<T>, A>& x,
           const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::scalbn(x[i], exp[i]); }); }

  template <typename T, typename A>
    constexpr simd<T, A>
    ldexp(const simd<detail::FloatingPoint<T>, A>& x,
          const fixed_size_simd<int, simd_size_v<T, A>>& exp) noexcept
    { return simd<T, A>([&x, &exp](auto i) { return std::ldexp(x[i], exp[i]); }); }

  SIMD_MATH_1ARG(sqrt, simd)

  SIMD_MATH_3ARG(fma, simd)

  SIMD_MATH_1ARG(trunc, simd)
  SIMD_MATH_1ARG(ceil, simd)
  SIMD_MATH_1ARG(floor, simd)
  SIMD_MATH_1ARG(round, simd)
  SIMD_MATH_1ARG_FIXED(lround, long)
  SIMD_MATH_1ARG_FIXED(llround, long long)
  SIMD_MATH_1ARG(nearbyint, simd)
  SIMD_MATH_1ARG(rint, simd)
  SIMD_MATH_1ARG_FIXED(lrint, long)
  SIMD_MATH_1ARG_FIXED(llrint, long long)
  SIMD_MATH_1ARG_FIXED(ilogb, int)

  // trig functions
  SIMD_MATH_1ARG(sin, simd)
  SIMD_MATH_1ARG(cos, simd)
  SIMD_MATH_1ARG(tan, simd)
  SIMD_MATH_1ARG(asin, simd)
  SIMD_MATH_1ARG(acos, simd)
  SIMD_MATH_1ARG(atan, simd)
  SIMD_MATH_2ARG(atan2, simd)
  SIMD_MATH_1ARG(sinh, simd)
  SIMD_MATH_1ARG(cosh, simd)
  SIMD_MATH_1ARG(tanh, simd)
  SIMD_MATH_1ARG(asinh, simd)
  SIMD_MATH_1ARG(acosh, simd)
  SIMD_MATH_1ARG(atanh, simd)

  // logarithms
  SIMD_MATH_1ARG(log, simd)
  SIMD_MATH_1ARG(log10, simd)
  SIMD_MATH_1ARG(log1p, simd)
  SIMD_MATH_1ARG(log2, simd)
  SIMD_MATH_1ARG(logb, simd)

#undef SIMD_MATH_1ARG
#undef SIMD_MATH_1ARG_FIXED
#undef SIMD_MATH_2ARG
#undef SIMD_MATH_3ARG
}
#ifdef VIR_SIMD_TS_DROPIN
}

namespace vir::stdx
{
  using namespace std::experimental::parallelism_v2;
}
#endif

#endif
#endif  // VIR_SIMD_H_
#ifndef GNURADIO_GRAPH_UTILS_HPP
#define GNURADIO_GRAPH_UTILS_HPP

#include <functional>
#include <iostream>
#include <string>
#include <string_view>


#ifndef __EMSCRIPTEN__
#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#endif

namespace fair::meta {

template<typename... Ts>
struct print_types;

template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N            = SIZE;
    CharT                        _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0)
            for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
    }

    [[nodiscard]] constexpr std::size_t
    size() const noexcept {
        return N;
    }

    [[nodiscard]] constexpr bool
    empty() const noexcept {
        return N == 0;
    }

    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return { _data, N }; }

    [[nodiscard]] explicit           operator std::string() const noexcept { return { _data, N }; }

    [[nodiscard]]                    operator const char *() const noexcept { return _data; }

    [[nodiscard]] constexpr bool
    operator==(const fixed_string &other) const noexcept {
        return std::string_view{ _data, N } == std::string_view(other);
    }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool
    operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
        return false;
    }
};

template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

template<typename T>
[[nodiscard]] constexpr std::string
type_name() noexcept {
#ifndef __EMSCRIPTEN__
    std::string type_name = typeid(T).name();
    int         status;
    char       *demangled_name = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        return demangled_name;
    } else {
        return typeid(T).name();
    }
#else
    return typeid(T).name(); // TODO: to be replaced by refl-cpp
#endif
}

template<fixed_string val>
struct message_type {};

template<class... T>
constexpr bool always_false = false;

struct dummy_t {};

template<typename F, typename... Args>
auto
invoke_void_wrapped(F &&f, Args &&...args) {
    if constexpr (std::is_same_v<void, std::invoke_result_t<F, Args...>>) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        return dummy_t{};
    } else {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
}

static_assert(std::is_same_v<decltype(invoke_void_wrapped([] {})), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([] { return 42; })), int>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int) {}, 42)), dummy_t>);
static_assert(std::is_same_v<decltype(invoke_void_wrapped([](int i) { return i; }, 42)), int>);

#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':' << loc.column() << ": `" << loc.function_name() << "`\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure(loc);
}
#else
[[gnu::always_inline]] inline void
precondition(bool cond) {
    struct handle {
        [[noreturn]] static void
        failure() {
            std::clog << "failed precondition\n";
            __builtin_trap();
        }
    };

    if (not cond) [[unlikely]]
        handle::failure();
}
#endif

namespace stdx = vir::stdx;

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V> && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable = std::constructible_from<stdx::simd<T>>;

template<typename A, typename B>
struct wider_native_simd_size : std::conditional<(stdx::native_simd<A>::size() > stdx::native_simd<B>::size()), A, B> {};

template<typename A>
struct wider_native_simd_size<A, A> {
    using type = A;
};

template<typename V>
struct rebind_simd_helper {
    template<typename T>
    using rebind = stdx::rebind_simd_t<T, V>;
};

struct simd_load_ctor {
    template<any_simd W>
    static constexpr W
    apply(typename W::value_type const *ptr) {
        return W(ptr, stdx::element_aligned);
    }
};

template<typename List>
using reduce_to_widest_simd = stdx::native_simd<meta::reduce<wider_native_simd_size, List>>;

template<typename V, typename List>
using transform_by_rebind_simd = meta::transform_types<rebind_simd_helper<V>::template rebind, List>;

template<typename List>
using transform_to_widest_simd = transform_by_rebind_simd<reduce_to_widest_simd<List>, List>;

template<typename Node>
concept source_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                          {
                              [](Node &n, auto &inputs) {
                                  if constexpr (Node::input_port_types::size > 0) {
                                      return []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>)->decltype(n_inside.process_one(std::get<Is>(tup)...)) { return {}; }
                                      (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                                  } else {
                                      return n.process_one();
                                  }
                              }(node, inputs)
                              } -> std::same_as<typename Node::return_type>;
                      };

template<typename Node>
concept sink_node = requires(Node &node, typename Node::input_port_types::tuple_type const &inputs) {
                        {
                            [](Node &n, auto &inputs) {
                                []<std::size_t... Is>(Node & n_inside, auto const &tup, std::index_sequence<Is...>) {
                                    if constexpr (Node::output_port_types::size > 0) {
                                        auto a [[maybe_unused]] = n_inside.process_one(std::get<Is>(tup)...);
                                    } else {
                                        n_inside.process_one(std::get<Is>(tup)...);
                                    }
                                }
                                (n, inputs, std::make_index_sequence<Node::input_port_types::size>());
                            }(node, inputs)
                        };
                    };

template<typename Node>
concept any_node = source_node<Node> || sink_node<Node>;

template<typename Node>
concept node_can_process_simd = any_node<Node> && requires(Node &n, typename transform_to_widest_simd<typename Node::input_port_types>::template apply<std::tuple> const &inputs) {
                                                      {
                                                          []<std::size_t... Is>(Node & n, auto const &tup, std::index_sequence<Is...>)->decltype(n.process_one(std::get<Is>(tup)...)) { return {}; }
                                                          (n, inputs, std::make_index_sequence<Node::input_port_types::size>())
                                                          } -> any_simd<typename Node::return_type>;
                                                  };

template<fixed_string Name, typename PortList>
consteval int
indexForName() {
    auto helper = []<std::size_t... Ids>(std::index_sequence<Ids...>) {
        int result = -1;
        ((PortList::template at<Ids>::static_name() == Name ? (result = Ids) : 0), ...);
        return result;
    };
    return helper(std::make_index_sequence<PortList::size>());
}

template<typename... Lambdas>
struct overloaded : Lambdas... {
    using Lambdas::operator()...;
};

template<typename... Lambdas>
overloaded(Lambdas...) -> overloaded<Lambdas...>;

namespace detail {
template<template<typename...> typename Mapper, template<typename...> typename Wrapper, typename... Args>
Wrapper<Mapper<Args>...> *
type_transform_impl(Wrapper<Args...> *);

template<template<typename...> typename Mapper, typename T>
Mapper<T> *
type_transform_impl(T *);

template<template<typename...> typename Mapper>
void *
type_transform_impl(void *);

template<template<typename...> typename Mapper>
fair::meta::dummy_t *
type_transform_impl(fair::meta::dummy_t *);
} // namespace detail

template<template<typename...> typename Mapper, typename T>
using type_transform = std::remove_pointer_t<decltype(detail::type_transform_impl<Mapper>(static_cast<T *>(nullptr)))>;

static_assert(std::is_same_v<std::vector<int>, type_transform<std::vector, int>>);
static_assert(std::is_same_v<std::tuple<std::vector<int>, std::vector<float>>, type_transform<std::vector, std::tuple<int, float>>>);
static_assert(std::is_same_v<void, type_transform<std::vector, void>>);
static_assert(std::is_same_v<dummy_t, type_transform<std::vector, dummy_t>>);

} // namespace fair::meta

#endif // include guard
#ifndef GNURADIO_TYPELIST_HPP
#define GNURADIO_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <string_view>
#include <string>

namespace fair::meta {

template<typename... Ts>
struct typelist;

// concat ///////////////
namespace detail {
template<typename...>
struct concat_impl;

template<>
struct concat_impl<> {
    using type = typelist<>;
};

template<typename A>
struct concat_impl<A> {
    using type = typelist<A>;
};

template<typename... As>
struct concat_impl<typelist<As...>> {
    using type = typelist<As...>;
};

template<typename A, typename B>
struct concat_impl<A, B> {
    using type = typelist<A, B>;
};

template<typename... As, typename B>
struct concat_impl<typelist<As...>, B> {
    using type = typelist<As..., B>;
};

template<typename A, typename... Bs>
struct concat_impl<A, typelist<Bs...>> {
    using type = typelist<A, Bs...>;
};

template<typename... As, typename... Bs>
struct concat_impl<typelist<As...>, typelist<Bs...>> {
    using type = typelist<As..., Bs...>;
};

template<typename A, typename B, typename C>
struct concat_impl<A, B, C> {
    using type = typename concat_impl<typename concat_impl<A, B>::type, C>::type;
};

template<typename A, typename B, typename C, typename D, typename... More>
struct concat_impl<A, B, C, D, More...> {
    using type =
            typename concat_impl<typename concat_impl<A, B>::type, typename concat_impl<C, D>::type,
                                 typename concat_impl<More...>::type>::type;
};
} // namespace detail

template<typename... Ts>
using concat = typename detail::concat_impl<Ts...>::type;

// split_at, left_of, right_of ////////////////
namespace detail {
template<unsigned N>
struct splitter;

template<>
struct splitter<0> {
    template<typename...>
    using first = typelist<>;
    template<typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<1> {
    template<typename T0, typename...>
    using first = typelist<T0>;
    template<typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<2> {
    template<typename T0, typename T1, typename...>
    using first = typelist<T0, T1>;
    template<typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<4> {
    template<typename T0, typename T1, typename T2, typename T3, typename...>
    using first = typelist<T0, T1, T2, T3>;
    template<typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<8> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7>;

    template<typename, typename, typename, typename, typename, typename, typename, typename,
             typename... Ts>
    using second = typelist<Ts...>;
};

template<>
struct splitter<16> {
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15, typename...>
    using first = typelist<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>;

    template<typename, typename, typename, typename, typename, typename, typename, typename, typename,
             typename, typename, typename, typename, typename, typename, typename, typename... Ts>
    using second = typelist<Ts...>;
};

template<unsigned N>
struct splitter {
    static constexpr unsigned FirstSplit = std::has_single_bit(N) ? N / 2 : std::bit_floor(N);
    using A                              = splitter<FirstSplit>;
    using B                              = splitter<N - FirstSplit>;

    template<typename... Ts>
    using first = concat<typename A::template first<Ts...>,
                         typename B::template first<typename A::template second<Ts...>>>;

    template<typename... Ts>
    using second = typename B::template second<typename A::template second<Ts...>>;
};
} // namespace detail

template<unsigned N, typename List>
struct split_at;

template<unsigned N, typename... Ts>
struct split_at<N, typelist<Ts...>> {
    using first  = typename detail::splitter<N>::template first<Ts...>;
    using second = typename detail::splitter<N>::template second<Ts...>;
};

template<std::size_t N, typename List>
using left_of = typename split_at<N, List>::first;

template<std::size_t N, typename List>
using right_of = typename split_at<N + 1, List>::second;

// remove_at /////////////
template<std::size_t Idx, typename List>
using remove_at = concat<left_of<Idx, List>, right_of<Idx, List>>;

// first_type ////////////
namespace detail {
template<typename List>
struct first_type_impl {};

template<typename T0, typename... Ts>
struct first_type_impl<typelist<T0, Ts...>> {
    using type = T0;
};
} // namespace detail

template<typename List>
using first_type = typename detail::first_type_impl<List>::type;

// transform_types ////////////
namespace detail {
template<template<typename> class Template, typename List>
struct transform_types_impl;

template<template<typename> class Template, typename... Ts>
struct transform_types_impl<Template, typelist<Ts...>> {
    using type = typelist<Template<Ts>...>;
};
} // namespace detail

template<template<typename> class Template, typename List>
using transform_types = typename detail::transform_types_impl<Template, List>::type;

// transform_value_type
template<typename T>
using transform_value_type = typename T::value_type;

// reduce ////////////////
namespace detail {
template<template<typename, typename> class Method, typename List>
struct reduce_impl;

template<template<typename, typename> class Method, typename T0>
struct reduce_impl<Method, typelist<T0>> {
    using type = T0;
};

template<template<typename, typename> class Method, typename T0, typename T1, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, Ts...>>
    : public reduce_impl<Method, typelist<typename Method<T0, T1>::type, Ts...>> {};

template<template<typename, typename> class Method, typename T0, typename T1, typename T2,
         typename T3, typename... Ts>
struct reduce_impl<Method, typelist<T0, T1, T2, T3, Ts...>>
    : public reduce_impl<
              Method, typelist<typename Method<T0, T1>::type, typename Method<T2, T3>::type, Ts...>> {
};
} // namespace detail

template<template<typename, typename> class Method, typename List>
using reduce = typename detail::reduce_impl<Method, List>::type;

// typelist /////////////////
template<typename T>
concept is_typelist_v = requires { typename T::typelist_tag; };

template<typename... Ts>
struct typelist {
    using this_t = typelist<Ts...>;
    using typelist_tag = std::true_type;

    static inline constexpr std::integral_constant<std::size_t, sizeof...(Ts)> size = {};

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<std::size_t I>
    using at = first_type<typename detail::splitter<I>::template second<Ts...>>;

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Ts> && ...);

    template<typename F, typename Tup>
        requires(sizeof...(Ts) == std::tuple_size_v<std::remove_cvref_t<Tup>>)
    static constexpr auto
    construct(Tup &&args_tuple) {
        return std::apply(
                []<typename... Args>(Args &&...args) {
                    return std::make_tuple(F::template apply<Ts>(std::forward<Args>(args))...);
                },
                std::forward<Tup>(args_tuple));
    }

    template<template<typename> typename Trafo>
    using transform = meta::transform_types<Trafo, this_t>;

    template<template<typename...> typename Pred>
    constexpr static bool all_of = (Pred<Ts>::value && ...);

    template<template<typename...> typename Pred>
    constexpr static bool none_of = (!Pred<Ts>::value && ...);

    template<template<typename...> typename Predicate>
    using filter = concat<std::conditional_t<Predicate<Ts>::value, typelist<Ts>, typelist<>>...>;

    using tuple_type    = std::tuple<Ts...>;
    using tuple_or_type = std::remove_pointer_t<decltype(
            [] {
                if constexpr (sizeof...(Ts) == 0) {
                    return static_cast<void*>(nullptr);
                } else if constexpr (sizeof...(Ts) == 1) {
                    return static_cast<at<0>*>(nullptr);
                } else {
                    return static_cast<tuple_type*>(nullptr);
                }
            }())>;

};




} // namespace fair::meta

#endif // include guard
#ifndef GNURADIO_WAIT_STRATEGY_HPP
#define GNURADIO_WAIT_STRATEGY_HPP

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef GNURADIO_SEQUENCE_HPP
#endif

namespace gr {
// clang-format off
/**
 * Wait for the given sequence to be available.  It is possible for this method to return a value less than the sequence number supplied depending on the implementation of the WaitStrategy.
 * A common use for this is to signal a timeout.Any EventProcessor that is using a WaitStrategy to get notifications about message becoming available should remember to handle this case.
 * The BatchEventProcessor<T> explicitly handles this case and will signal a timeout if required.
 *
 * \param sequence sequence to be waited on.
 * \param cursor Ring buffer cursor on which to wait.
 * \param dependentSequence on which to wait.
 * \param barrier barrier the IEventProcessor is waiting on.
 * \returns the sequence that is available which may be greater than the requested sequence.
 */
template<typename T>
inline constexpr bool isWaitStrategy = requires(T /*const*/ t, const std::int64_t sequence, const Sequence &cursor, std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
    { t.waitFor(sequence, cursor, dependentSequences) } -> std::same_as<std::int64_t>;
};
static_assert(!isWaitStrategy<int>);

/**
 * signal those waiting that the cursor has advanced.
 */
template<typename T>
inline constexpr bool hasSignalAllWhenBlocking = requires(T /*const*/ t) {
    { t.signalAllWhenBlocking() } -> std::same_as<void>;
};
static_assert(!hasSignalAllWhenBlocking<int>);

template<typename T>
concept WaitStrategy = isWaitStrategy<T>;



/**
 * Blocking strategy that uses a lock and condition variable for IEventProcessor's waiting on a barrier.
 * This strategy should be used when performance and low-latency are not as important as CPU resource.
 */
class BlockingWaitStrategy {
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert
                _conditionVariable.wait(uniqueLock);
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<BlockingWaitStrategy>);

/**
 * Busy Spin strategy that uses a busy spin loop for IEventProcessor's waiting on a barrier.
 * This strategy will use CPU resource to avoid syscalls which can introduce latency jitter.
 * It is best used when threads can be bound to specific CPU cores.
 */
struct BusySpinWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }
        return availableSequence;
    }
};
static_assert(WaitStrategy<BusySpinWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<BusySpinWaitStrategy>);

/**
 * Sleeping strategy that initially spins, then uses a std::this_thread::yield(), and eventually sleep. T
 * his strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
class SleepingWaitStrategy {
    static const std::int32_t _defaultRetries = 200;
    std::int32_t              _retries        = 0;

public:
    explicit SleepingWaitStrategy(std::int32_t retries = _defaultRetries)
        : _retries(retries) {
    }

    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _retries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter > 100) {
                --counter;
            } else if (counter > 0) {
                --counter;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(0));
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SleepingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SleepingWaitStrategy>);

struct TimeoutException : public std::runtime_error {
    TimeoutException() : std::runtime_error("TimeoutException") {}
};

class TimeoutBlockingWaitStrategy {
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    Clock::duration             _timeout;
    std::recursive_mutex        _gate;
    std::condition_variable_any _conditionVariable;

public:
    explicit TimeoutBlockingWaitStrategy(Clock::duration timeout)
        : _timeout(timeout) {}

    std::int64_t waitFor(const std::int64_t sequence, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) {
        auto timeSpan = std::chrono::duration_cast<std::chrono::microseconds>(_timeout);

        if (cursor.value() < sequence) {
            std::unique_lock uniqueLock(_gate);

            while (cursor.value() < sequence) {
                // optional: barrier check alert

                if (_conditionVariable.wait_for(uniqueLock, timeSpan) == std::cv_status::timeout) {
                    throw TimeoutException();
                }
            }
        }

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            // optional: barrier check alert
        }

        return availableSequence;
    }

    void signalAllWhenBlocking() {
        std::unique_lock uniqueLock(_gate);
        _conditionVariable.notify_all();
    }
};
static_assert(WaitStrategy<TimeoutBlockingWaitStrategy>);
static_assert(hasSignalAllWhenBlocking<TimeoutBlockingWaitStrategy>);

/**
 * Yielding strategy that uses a Thread.Yield() for IEventProcessors waiting on a barrier after an initially spinning.
 * This strategy is a good compromise between performance and CPU resource without incurring significant latency spikes.
 */
class YieldingWaitStrategy {
    const std::size_t _spinTries = 100;

public:
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequences) const {
        auto       counter    = _spinTries;
        const auto waitMethod = [&counter]() {
            // optional: barrier check alert

            if (counter == 0) {
                std::this_thread::yield();
            } else {
                --counter;
            }
        };

        std::int64_t availableSequence;
        while ((availableSequence = detail::getMinimumSequence(dependentSequences)) < sequence) {
            waitMethod();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<YieldingWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<YieldingWaitStrategy>);

struct NoWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> & /*dependentSequences*/) const {
        // wait for nothing
        return sequence;
    }
};
static_assert(WaitStrategy<NoWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<NoWaitStrategy>);


/**
 *
 * SpinWait is meant to be used as a tool for waiting in situations where the thread is not allowed to block.
 *
 * In order to get the maximum performance, the implementation first spins for `YIELD_THRESHOLD` times, and then
 * alternates between yielding, spinning and putting the thread to sleep, to allow other threads to be scheduled
 * by the kernel to avoid potential CPU contention.
 *
 * The number of spins, yielding, and sleeping for either '0 ms' or '1 ms' is controlled by the NTTP constants
 * @tparam YIELD_THRESHOLD
 * @tparam SLEEP_0_EVERY_HOW_MANY_TIMES
 * @tparam SLEEP_1_EVERY_HOW_MANY_TIMES
 */
template<std::int32_t YIELD_THRESHOLD = 10, std::int32_t SLEEP_0_EVERY_HOW_MANY_TIMES = 5, std::int32_t SLEEP_1_EVERY_HOW_MANY_TIMES = 20>
class SpinWait {
    using Clock         = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    std::int32_t _count = 0;
    static void  spinWaitInternal(std::int32_t iterationCount) noexcept {
        for (auto i = 0; i < iterationCount; i++) {
            yieldProcessor();
        }
    }
#ifndef __EMSCRIPTEN__
    static void yieldProcessor() noexcept { asm volatile("rep\nnop"); }
#else
    static void yieldProcessor() noexcept { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
#endif

public:
    SpinWait() = default;

    [[nodiscard]] std::int32_t count() const noexcept { return _count; }
    [[nodiscard]] bool         nextSpinWillYield() const noexcept { return _count > YIELD_THRESHOLD; }

    void                       spinOnce() {
        if (nextSpinWillYield()) {
            auto num = _count >= YIELD_THRESHOLD ? _count - 10 : _count;
            if (num % SLEEP_1_EVERY_HOW_MANY_TIMES == SLEEP_1_EVERY_HOW_MANY_TIMES - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                if (num % SLEEP_0_EVERY_HOW_MANY_TIMES == SLEEP_0_EVERY_HOW_MANY_TIMES - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(0));
                } else {
                    std::this_thread::yield();
                }
            }
        } else {
            spinWaitInternal(4 << _count);
        }

        if (_count == std::numeric_limits<std::int32_t>::max()) {
            _count = YIELD_THRESHOLD;
        } else {
            ++_count;
        }
    }

    void reset() noexcept { _count = 0; }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition) const { return spinUntil(condition, -1); }

    template<typename T>
    requires std::is_nothrow_invocable_r_v<bool, T>
    bool
    spinUntil(const T &condition, std::int64_t millisecondsTimeout) const {
        if (millisecondsTimeout < -1) {
            throw std::out_of_range("Timeout value is out of range");
        }

        std::int64_t num = 0;
        if (millisecondsTimeout != 0 && millisecondsTimeout != -1) {
            num = getTickCount();
        }

        SpinWait spinWait;
        while (!condition()) {
            if (millisecondsTimeout == 0) {
                return false;
            }

            spinWait.spinOnce();

            if (millisecondsTimeout != 1 && spinWait.nextSpinWillYield() && millisecondsTimeout <= (getTickCount() - num)) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] static std::int64_t getTickCount() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }
};

/**
 * Spin strategy that uses a SpinWait for IEventProcessors waiting on a barrier.
 * This strategy is a good compromise between performance and CPU resource.
 * Latency spikes can occur after quiet periods.
 */
struct SpinWaitWaitStrategy {
    std::int64_t waitFor(const std::int64_t sequence, const Sequence & /*cursor*/, const std::vector<std::shared_ptr<Sequence>> &dependentSequence) const {
        std::int64_t availableSequence;

        SpinWait     spinWait;
        while ((availableSequence = detail::getMinimumSequence(dependentSequence)) < sequence) {
            // optional: barrier check alert
            spinWait.spinOnce();
        }

        return availableSequence;
    }
};
static_assert(WaitStrategy<SpinWaitWaitStrategy>);
static_assert(!hasSignalAllWhenBlocking<SpinWaitWaitStrategy>);

struct NO_SPIN_WAIT {};

template<typename SPIN_WAIT = NO_SPIN_WAIT>
class AtomicMutex {
    std::atomic_flag _lock = ATOMIC_FLAG_INIT;
    SPIN_WAIT        _spin_wait;

public:
    AtomicMutex()                    = default;
    AtomicMutex(const AtomicMutex &) = delete;
    AtomicMutex &operator=(const AtomicMutex &) = delete;

    //
    void lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {
            if constexpr (requires { _spin_wait.spin_once(); }) {
                _spin_wait.spin_once();
            }
        }
        if constexpr (requires { _spin_wait.spin_once(); }) {
            _spin_wait.reset();
        }
    }
    void unlock() { _lock.clear(std::memory_order::release); }
};


// clang-format on
} // namespace gr


#endif // GNURADIO_WAIT_STRATEGY_HPP
/*
 Formatting library for C++

 Copyright (c) 2012 - present, Victor Zverovich

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 --- Optional exception to the license ---

 As an exception, if, as a result of your compiling your source code, portions
 of this Software are embedded into a machine-executable object form of such
 source code, you may redistribute such embedded portions in such object form
 without including the above copyright and permission notices.
 */

#ifndef FMT_FORMAT_H_
#define FMT_FORMAT_H_

#include <cmath>         // std::signbit
#include <cstdint>       // uint32_t
#include <limits>        // std::numeric_limits
#include <memory>        // std::uninitialized_copy
#include <stdexcept>     // std::runtime_error
#include <system_error>  // std::system_error
#include <utility>       // std::swap

#ifdef __cpp_lib_bit_cast
#  include <bit>  // std::bitcast
#endif


#if FMT_GCC_VERSION
#  define FMT_GCC_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#else
#  define FMT_GCC_VISIBILITY_HIDDEN
#endif

#ifdef __NVCC__
#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
#else
#  define FMT_CUDA_VERSION 0
#endif

#ifdef __has_builtin
#  define FMT_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define FMT_HAS_BUILTIN(x) 0
#endif

#if FMT_GCC_VERSION || FMT_CLANG_VERSION
#  define FMT_NOINLINE __attribute__((noinline))
#else
#  define FMT_NOINLINE
#endif

#if FMT_MSC_VER
#  define FMT_MSC_DEFAULT = default
#else
#  define FMT_MSC_DEFAULT
#endif

#ifndef FMT_THROW
#  if FMT_EXCEPTIONS
#    if FMT_MSC_VER || FMT_NVCC
FMT_BEGIN_NAMESPACE
namespace detail {
template <typename Exception> inline void do_throw(const Exception& x) {
  // Silence unreachable code warnings in MSVC and NVCC because these
  // are nearly impossible to fix in a generic code.
  volatile bool b = true;
  if (b) throw x;
}
}  // namespace detail
FMT_END_NAMESPACE
#      define FMT_THROW(x) detail::do_throw(x)
#    else
#      define FMT_THROW(x) throw x
#    endif
#  else
#    define FMT_THROW(x)               \
      do {                             \
        FMT_ASSERT(false, (x).what()); \
      } while (false)
#  endif
#endif

#if FMT_EXCEPTIONS
#  define FMT_TRY try
#  define FMT_CATCH(x) catch (x)
#else
#  define FMT_TRY if (true)
#  define FMT_CATCH(x) if (false)
#endif

#ifndef FMT_MAYBE_UNUSED
#  if FMT_HAS_CPP17_ATTRIBUTE(maybe_unused)
#    define FMT_MAYBE_UNUSED [[maybe_unused]]
#  else
#    define FMT_MAYBE_UNUSED
#  endif
#endif

// Workaround broken [[deprecated]] in the Intel, PGI and NVCC compilers.
#if FMT_ICC_VERSION || defined(__PGI) || FMT_NVCC
#  define FMT_DEPRECATED_ALIAS
#else
#  define FMT_DEPRECATED_ALIAS FMT_DEPRECATED
#endif

#ifndef FMT_USE_USER_DEFINED_LITERALS
// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
#  if (FMT_HAS_FEATURE(cxx_user_literals) || FMT_GCC_VERSION >= 407 || \
       FMT_MSC_VER >= 1900) &&                                         \
      (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= /* UDL feature */ 480)
#    define FMT_USE_USER_DEFINED_LITERALS 1
#  else
#    define FMT_USE_USER_DEFINED_LITERALS 0
#  endif
#endif

// Defining FMT_REDUCE_INT_INSTANTIATIONS to 1, will reduce the number of
// integer formatter template instantiations to just one by only using the
// largest integer type. This results in a reduction in binary size but will
// cause a decrease in integer formatting performance.
#if !defined(FMT_REDUCE_INT_INSTANTIATIONS)
#  define FMT_REDUCE_INT_INSTANTIATIONS 0
#endif

// __builtin_clz is broken in clang with Microsoft CodeGen:
// https://github.com/fmtlib/fmt/issues/519.
#if !FMT_MSC_VER
#  if FMT_HAS_BUILTIN(__builtin_clz) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CLZ(n) __builtin_clz(n)
#  endif
#  if FMT_HAS_BUILTIN(__builtin_clzll) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CLZLL(n) __builtin_clzll(n)
#  endif
#endif

// __builtin_ctz is broken in Intel Compiler Classic on Windows:
// https://github.com/fmtlib/fmt/issues/2510.
#ifndef __ICL
#  if FMT_HAS_BUILTIN(__builtin_ctz) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CTZ(n) __builtin_ctz(n)
#  endif
#  if FMT_HAS_BUILTIN(__builtin_ctzll) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CTZLL(n) __builtin_ctzll(n)
#  endif
#endif

#if FMT_MSC_VER
#  include <intrin.h>  // _BitScanReverse[64], _BitScanForward[64], _umul128
#endif

// Some compilers masquerade as both MSVC and GCC-likes or otherwise support
// __builtin_clz and __builtin_clzll, so only define FMT_BUILTIN_CLZ using the
// MSVC intrinsics if the clz and clzll builtins are not available.
#if FMT_MSC_VER && !defined(FMT_BUILTIN_CLZLL) && !defined(FMT_BUILTIN_CTZLL)
FMT_BEGIN_NAMESPACE
namespace detail {
// Avoid Clang with Microsoft CodeGen's -Wunknown-pragmas warning.
#  if !defined(__clang__)
#    pragma intrinsic(_BitScanForward)
#    pragma intrinsic(_BitScanReverse)
#    if defined(_WIN64)
#      pragma intrinsic(_BitScanForward64)
#      pragma intrinsic(_BitScanReverse64)
#    endif
#  endif

inline auto clz(uint32_t x) -> int {
  unsigned long r = 0;
  _BitScanReverse(&r, x);
  FMT_ASSERT(x != 0, "");
  // Static analysis complains about using uninitialized data
  // "r", but the only way that can happen is if "x" is 0,
  // which the callers guarantee to not happen.
  FMT_MSC_WARNING(suppress : 6102)
  return 31 ^ static_cast<int>(r);
}
#  define FMT_BUILTIN_CLZ(n) detail::clz(n)

inline auto clzll(uint64_t x) -> int {
  unsigned long r = 0;
#  ifdef _WIN64
  _BitScanReverse64(&r, x);
#  else
  // Scan the high 32 bits.
  if (_BitScanReverse(&r, static_cast<uint32_t>(x >> 32))) return 63 ^ (r + 32);
  // Scan the low 32 bits.
  _BitScanReverse(&r, static_cast<uint32_t>(x));
#  endif
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
  return 63 ^ static_cast<int>(r);
}
#  define FMT_BUILTIN_CLZLL(n) detail::clzll(n)

inline auto ctz(uint32_t x) -> int {
  unsigned long r = 0;
  _BitScanForward(&r, x);
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
  return static_cast<int>(r);
}
#  define FMT_BUILTIN_CTZ(n) detail::ctz(n)

inline auto ctzll(uint64_t x) -> int {
  unsigned long r = 0;
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
#  ifdef _WIN64
  _BitScanForward64(&r, x);
#  else
  // Scan the low 32 bits.
  if (_BitScanForward(&r, static_cast<uint32_t>(x))) return static_cast<int>(r);
  // Scan the high 32 bits.
  _BitScanForward(&r, static_cast<uint32_t>(x >> 32));
  r += 32;
#  endif
  return static_cast<int>(r);
}
#  define FMT_BUILTIN_CTZLL(n) detail::ctzll(n)
}  // namespace detail
FMT_END_NAMESPACE
#endif

#ifdef FMT_HEADER_ONLY
#  define FMT_HEADER_ONLY_CONSTEXPR20 FMT_CONSTEXPR20
#else
#  define FMT_HEADER_ONLY_CONSTEXPR20
#endif

FMT_BEGIN_NAMESPACE
namespace detail {

template <typename Streambuf> class formatbuf : public Streambuf {
 private:
  using char_type = typename Streambuf::char_type;
  using streamsize = decltype(std::declval<Streambuf>().sputn(nullptr, 0));
  using int_type = typename Streambuf::int_type;
  using traits_type = typename Streambuf::traits_type;

  buffer<char_type>& buffer_;

 public:
  explicit formatbuf(buffer<char_type>& buf) : buffer_(buf) {}

 protected:
  // The put area is always empty. This makes the implementation simpler and has
  // the advantage that the streambuf and the buffer are always in sync and
  // sputc never writes into uninitialized memory. A disadvantage is that each
  // call to sputc always results in a (virtual) call to overflow. There is no
  // disadvantage here for sputn since this always results in a call to xsputn.

  auto overflow(int_type ch) -> int_type override {
    if (!traits_type::eq_int_type(ch, traits_type::eof()))
      buffer_.push_back(static_cast<char_type>(ch));
    return ch;
  }

  auto xsputn(const char_type* s, streamsize count) -> streamsize override {
    buffer_.append(s, s + count);
    return count;
  }
};

// Implementation of std::bit_cast for pre-C++20.
template <typename To, typename From>
FMT_CONSTEXPR20 auto bit_cast(const From& from) -> To {
  static_assert(sizeof(To) == sizeof(From), "size mismatch");
#ifdef __cpp_lib_bit_cast
  if (is_constant_evaluated()) return std::bit_cast<To>(from);
#endif
  auto to = To();
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

inline auto is_big_endian() -> bool {
#ifdef _WIN32
  return false;
#elif defined(__BIG_ENDIAN__)
  return true;
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
  return __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
#else
  struct bytes {
    char data[sizeof(int)];
  };
  return bit_cast<bytes>(1).data[0] == 0;
#endif
}

// A fallback implementation of uintptr_t for systems that lack it.
struct fallback_uintptr {
  unsigned char value[sizeof(void*)];

  fallback_uintptr() = default;
  explicit fallback_uintptr(const void* p) {
    *this = bit_cast<fallback_uintptr>(p);
    if (const_check(is_big_endian())) {
      for (size_t i = 0, j = sizeof(void*) - 1; i < j; ++i, --j)
        std::swap(value[i], value[j]);
    }
  }
};
#ifdef UINTPTR_MAX
using uintptr_t = ::uintptr_t;
inline auto to_uintptr(const void* p) -> uintptr_t {
  return bit_cast<uintptr_t>(p);
}
#else
using uintptr_t = fallback_uintptr;
inline auto to_uintptr(const void* p) -> fallback_uintptr {
  return fallback_uintptr(p);
}
#endif

// Returns the largest possible value for type T. Same as
// std::numeric_limits<T>::max() but shorter and not affected by the max macro.
template <typename T> constexpr auto max_value() -> T {
  return (std::numeric_limits<T>::max)();
}
template <typename T> constexpr auto num_bits() -> int {
  return std::numeric_limits<T>::digits;
}
// std::numeric_limits<T>::digits may return 0 for 128-bit ints.
template <> constexpr auto num_bits<int128_t>() -> int { return 128; }
template <> constexpr auto num_bits<uint128_t>() -> int { return 128; }
template <> constexpr auto num_bits<fallback_uintptr>() -> int {
  return static_cast<int>(sizeof(void*) *
                          std::numeric_limits<unsigned char>::digits);
}

FMT_INLINE void assume(bool condition) {
  (void)condition;
#if FMT_HAS_BUILTIN(__builtin_assume)
  __builtin_assume(condition);
#endif
}

// An approximation of iterator_t for pre-C++20 systems.
template <typename T>
using iterator_t = decltype(std::begin(std::declval<T&>()));
template <typename T> using sentinel_t = decltype(std::end(std::declval<T&>()));

// A workaround for std::string not having mutable data() until C++17.
template <typename Char>
inline auto get_data(std::basic_string<Char>& s) -> Char* {
  return &s[0];
}
template <typename Container>
inline auto get_data(Container& c) -> typename Container::value_type* {
  return c.data();
}

#if defined(_SECURE_SCL) && _SECURE_SCL
// Make a checked iterator to avoid MSVC warnings.
template <typename T> using checked_ptr = stdext::checked_array_iterator<T*>;
template <typename T>
constexpr auto make_checked(T* p, size_t size) -> checked_ptr<T> {
  return {p, size};
}
#else
template <typename T> using checked_ptr = T*;
template <typename T> constexpr auto make_checked(T* p, size_t) -> T* {
  return p;
}
#endif

// Attempts to reserve space for n extra characters in the output range.
// Returns a pointer to the reserved range or a reference to it.
template <typename Container, FMT_ENABLE_IF(is_contiguous<Container>::value)>
#if FMT_CLANG_VERSION >= 307 && !FMT_ICC_VERSION
__attribute__((no_sanitize("undefined")))
#endif
inline auto
reserve(std::back_insert_iterator<Container> it, size_t n)
    -> checked_ptr<typename Container::value_type> {
  Container& c = get_container(it);
  size_t size = c.size();
  c.resize(size + n);
  return make_checked(get_data(c) + size, n);
}

template <typename T>
inline auto reserve(buffer_appender<T> it, size_t n) -> buffer_appender<T> {
  buffer<T>& buf = get_container(it);
  buf.try_reserve(buf.size() + n);
  return it;
}

template <typename Iterator>
constexpr auto reserve(Iterator& it, size_t) -> Iterator& {
  return it;
}

template <typename OutputIt>
using reserve_iterator =
    remove_reference_t<decltype(reserve(std::declval<OutputIt&>(), 0))>;

template <typename T, typename OutputIt>
constexpr auto to_pointer(OutputIt, size_t) -> T* {
  return nullptr;
}
template <typename T> auto to_pointer(buffer_appender<T> it, size_t n) -> T* {
  buffer<T>& buf = get_container(it);
  auto size = buf.size();
  if (buf.capacity() < size + n) return nullptr;
  buf.try_resize(size + n);
  return buf.data() + size;
}

template <typename Container, FMT_ENABLE_IF(is_contiguous<Container>::value)>
inline auto base_iterator(std::back_insert_iterator<Container>& it,
                          checked_ptr<typename Container::value_type>)
    -> std::back_insert_iterator<Container> {
  return it;
}

template <typename Iterator>
constexpr auto base_iterator(Iterator, Iterator it) -> Iterator {
  return it;
}

// <algorithm> is spectacularly slow to compile in C++20 so use a simple fill_n
// instead (#1998).
template <typename OutputIt, typename Size, typename T>
FMT_CONSTEXPR auto fill_n(OutputIt out, Size count, const T& value)
    -> OutputIt {
  for (Size i = 0; i < count; ++i) *out++ = value;
  return out;
}
template <typename T, typename Size>
FMT_CONSTEXPR20 auto fill_n(T* out, Size count, char value) -> T* {
  if (is_constant_evaluated()) {
    return fill_n<T*, Size, T>(out, count, value);
  }
  std::memset(out, value, to_unsigned(count));
  return out + count;
}

#ifdef __cpp_char8_t
using char8_type = char8_t;
#else
enum char8_type : unsigned char {};
#endif

template <typename OutChar, typename InputIt, typename OutputIt>
FMT_CONSTEXPR FMT_NOINLINE auto copy_str_noinline(InputIt begin, InputIt end,
                                                  OutputIt out) -> OutputIt {
  return copy_str<OutChar>(begin, end, out);
}

// A public domain branchless UTF-8 decoder by Christopher Wellons:
// https://github.com/skeeto/branchless-utf8
/* Decode the next character, c, from s, reporting errors in e.
 *
 * Since this is a branchless decoder, four bytes will be read from the
 * buffer regardless of the actual length of the next character. This
 * means the buffer _must_ have at least three bytes of zero padding
 * following the end of the data stream.
 *
 * Errors are reported in e, which will be non-zero if the parsed
 * character was somehow invalid: invalid byte sequence, non-canonical
 * encoding, or a surrogate half.
 *
 * The function returns a pointer to the next character. When an error
 * occurs, this pointer will be a guess that depends on the particular
 * error, but it will always advance at least one byte.
 */
FMT_CONSTEXPR inline auto utf8_decode(const char* s, uint32_t* c, int* e)
    -> const char* {
  constexpr const int masks[] = {0x00, 0x7f, 0x1f, 0x0f, 0x07};
  constexpr const uint32_t mins[] = {4194304, 0, 128, 2048, 65536};
  constexpr const int shiftc[] = {0, 18, 12, 6, 0};
  constexpr const int shifte[] = {0, 6, 4, 2, 0};

  int len = code_point_length(s);
  const char* next = s + len;

  // Assume a four-byte character and load four bytes. Unused bits are
  // shifted out.
  *c = uint32_t(s[0] & masks[len]) << 18;
  *c |= uint32_t(s[1] & 0x3f) << 12;
  *c |= uint32_t(s[2] & 0x3f) << 6;
  *c |= uint32_t(s[3] & 0x3f) << 0;
  *c >>= shiftc[len];

  // Accumulate the various error conditions.
  using uchar = unsigned char;
  *e = (*c < mins[len]) << 6;       // non-canonical encoding
  *e |= ((*c >> 11) == 0x1b) << 7;  // surrogate half?
  *e |= (*c > 0x10FFFF) << 8;       // out of range?
  *e |= (uchar(s[1]) & 0xc0) >> 2;
  *e |= (uchar(s[2]) & 0xc0) >> 4;
  *e |= uchar(s[3]) >> 6;
  *e ^= 0x2a;  // top two bits of each tail byte correct?
  *e >>= shifte[len];

  return next;
}

constexpr uint32_t invalid_code_point = ~uint32_t();

// Invokes f(cp, sv) for every code point cp in s with sv being the string view
// corresponding to the code point. cp is invalid_code_point on error.
template <typename F>
FMT_CONSTEXPR void for_each_codepoint(string_view s, F f) {
  auto decode = [f](const char* buf_ptr, const char* ptr) {
    auto cp = uint32_t();
    auto error = 0;
    auto end = utf8_decode(buf_ptr, &cp, &error);
    bool result = f(error ? invalid_code_point : cp,
                    string_view(ptr, to_unsigned(end - buf_ptr)));
    return result ? end : nullptr;
  };
  auto p = s.data();
  const size_t block_size = 4;  // utf8_decode always reads blocks of 4 chars.
  if (s.size() >= block_size) {
    for (auto end = p + s.size() - block_size + 1; p < end;) {
      p = decode(p, p);
      if (!p) return;
    }
  }
  if (auto num_chars_left = s.data() + s.size() - p) {
    char buf[2 * block_size - 1] = {};
    copy_str<char>(p, p + num_chars_left, buf);
    const char* buf_ptr = buf;
    do {
      auto end = decode(buf_ptr, p);
      if (!end) return;
      p += end - buf_ptr;
      buf_ptr = end;
    } while (buf_ptr - buf < num_chars_left);
  }
}

template <typename Char>
inline auto compute_width(basic_string_view<Char> s) -> size_t {
  return s.size();
}

// Computes approximate display width of a UTF-8 string.
FMT_CONSTEXPR inline size_t compute_width(string_view s) {
  size_t num_code_points = 0;
  // It is not a lambda for compatibility with C++14.
  struct count_code_points {
    size_t* count;
    FMT_CONSTEXPR auto operator()(uint32_t cp, string_view) const -> bool {
      *count += detail::to_unsigned(
          1 +
          (cp >= 0x1100 &&
           (cp <= 0x115f ||  // Hangul Jamo init. consonants
            cp == 0x2329 ||  // LEFT-POINTING ANGLE BRACKET
            cp == 0x232a ||  // RIGHT-POINTING ANGLE BRACKET
            // CJK ... Yi except IDEOGRAPHIC HALF FILL SPACE:
            (cp >= 0x2e80 && cp <= 0xa4cf && cp != 0x303f) ||
            (cp >= 0xac00 && cp <= 0xd7a3) ||    // Hangul Syllables
            (cp >= 0xf900 && cp <= 0xfaff) ||    // CJK Compatibility Ideographs
            (cp >= 0xfe10 && cp <= 0xfe19) ||    // Vertical Forms
            (cp >= 0xfe30 && cp <= 0xfe6f) ||    // CJK Compatibility Forms
            (cp >= 0xff00 && cp <= 0xff60) ||    // Fullwidth Forms
            (cp >= 0xffe0 && cp <= 0xffe6) ||    // Fullwidth Forms
            (cp >= 0x20000 && cp <= 0x2fffd) ||  // CJK
            (cp >= 0x30000 && cp <= 0x3fffd) ||
            // Miscellaneous Symbols and Pictographs + Emoticons:
            (cp >= 0x1f300 && cp <= 0x1f64f) ||
            // Supplemental Symbols and Pictographs:
            (cp >= 0x1f900 && cp <= 0x1f9ff))));
      return true;
    }
  };
  for_each_codepoint(s, count_code_points{&num_code_points});
  return num_code_points;
}

inline auto compute_width(basic_string_view<char8_type> s) -> size_t {
  return compute_width(basic_string_view<char>(
      reinterpret_cast<const char*>(s.data()), s.size()));
}

template <typename Char>
inline auto code_point_index(basic_string_view<Char> s, size_t n) -> size_t {
  size_t size = s.size();
  return n < size ? n : size;
}

// Calculates the index of the nth code point in a UTF-8 string.
inline auto code_point_index(basic_string_view<char8_type> s, size_t n)
    -> size_t {
  const char8_type* data = s.data();
  size_t num_code_points = 0;
  for (size_t i = 0, size = s.size(); i != size; ++i) {
    if ((data[i] & 0xc0) != 0x80 && ++num_code_points > n) return i;
  }
  return s.size();
}

template <typename T, bool = std::is_floating_point<T>::value>
struct is_fast_float : bool_constant<std::numeric_limits<T>::is_iec559 &&
                                     sizeof(T) <= sizeof(double)> {};
template <typename T> struct is_fast_float<T, false> : std::false_type {};

#ifndef FMT_USE_FULL_CACHE_DRAGONBOX
#  define FMT_USE_FULL_CACHE_DRAGONBOX 0
#endif

template <typename T>
template <typename U>
void buffer<T>::append(const U* begin, const U* end) {
  while (begin != end) {
    auto count = to_unsigned(end - begin);
    try_reserve(size_ + count);
    auto free_cap = capacity_ - size_;
    if (free_cap < count) count = free_cap;
    std::uninitialized_copy_n(begin, count, make_checked(ptr_ + size_, count));
    size_ += count;
    begin += count;
  }
}

template <typename T, typename Enable = void>
struct is_locale : std::false_type {};
template <typename T>
struct is_locale<T, void_t<decltype(T::classic())>> : std::true_type {};
}  // namespace detail

FMT_MODULE_EXPORT_BEGIN

// The number of characters to store in the basic_memory_buffer object itself
// to avoid dynamic memory allocation.
enum { inline_buffer_size = 500 };

/**
  \rst
  A dynamically growing memory buffer for trivially copyable/constructible types
  with the first ``SIZE`` elements stored in the object itself.

  You can use the ``memory_buffer`` type alias for ``char`` instead.

  **Example**::

     auto out = fmt::memory_buffer();
     format_to(std::back_inserter(out), "The answer is {}.", 42);

  This will append the following output to the ``out`` object:

  .. code-block:: none

     The answer is 42.

  The output can be converted to an ``std::string`` with ``to_string(out)``.
  \endrst
 */
template <typename T, size_t SIZE = inline_buffer_size,
          typename Allocator = std::allocator<T>>
class basic_memory_buffer final : public detail::buffer<T> {
 private:
  T store_[SIZE];

  // Don't inherit from Allocator avoid generating type_info for it.
  Allocator alloc_;

  // Deallocate memory allocated by the buffer.
  FMT_CONSTEXPR20 void deallocate() {
    T* data = this->data();
    if (data != store_) alloc_.deallocate(data, this->capacity());
  }

 protected:
  FMT_CONSTEXPR20 void grow(size_t size) override;

 public:
  using value_type = T;
  using const_reference = const T&;

  FMT_CONSTEXPR20 explicit basic_memory_buffer(
      const Allocator& alloc = Allocator())
      : alloc_(alloc) {
    this->set(store_, SIZE);
    if (detail::is_constant_evaluated()) {
      detail::fill_n(store_, SIZE, T{});
    }
  }
  FMT_CONSTEXPR20 ~basic_memory_buffer() { deallocate(); }

 private:
  // Move data from other to this buffer.
  FMT_CONSTEXPR20 void move(basic_memory_buffer& other) {
    alloc_ = std::move(other.alloc_);
    T* data = other.data();
    size_t size = other.size(), capacity = other.capacity();
    if (data == other.store_) {
      this->set(store_, capacity);
      if (detail::is_constant_evaluated()) {
        detail::copy_str<T>(other.store_, other.store_ + size,
                            detail::make_checked(store_, capacity));
      } else {
        std::uninitialized_copy(other.store_, other.store_ + size,
                                detail::make_checked(store_, capacity));
      }
    } else {
      this->set(data, capacity);
      // Set pointer to the inline array so that delete is not called
      // when deallocating.
      other.set(other.store_, 0);
    }
    this->resize(size);
  }

 public:
  /**
    \rst
    Constructs a :class:`fmt::basic_memory_buffer` object moving the content
    of the other object to it.
    \endrst
   */
  FMT_CONSTEXPR20 basic_memory_buffer(basic_memory_buffer&& other)
      FMT_NOEXCEPT {
    move(other);
  }

  /**
    \rst
    Moves the content of the other ``basic_memory_buffer`` object to this one.
    \endrst
   */
  auto operator=(basic_memory_buffer&& other) FMT_NOEXCEPT
      -> basic_memory_buffer& {
    FMT_ASSERT(this != &other, "");
    deallocate();
    move(other);
    return *this;
  }

  // Returns a copy of the allocator associated with this buffer.
  auto get_allocator() const -> Allocator { return alloc_; }

  /**
    Resizes the buffer to contain *count* elements. If T is a POD type new
    elements may not be initialized.
   */
  FMT_CONSTEXPR20 void resize(size_t count) { this->try_resize(count); }

  /** Increases the buffer capacity to *new_capacity*. */
  void reserve(size_t new_capacity) { this->try_reserve(new_capacity); }

  // Directly append data into the buffer
  using detail::buffer<T>::append;
  template <typename ContiguousRange>
  void append(const ContiguousRange& range) {
    append(range.data(), range.data() + range.size());
  }
};

template <typename T, size_t SIZE, typename Allocator>
FMT_CONSTEXPR20 void basic_memory_buffer<T, SIZE, Allocator>::grow(
    size_t size) {
#ifdef FMT_FUZZ
  if (size > 5000) throw std::runtime_error("fuzz mode - won't grow that much");
#endif
  const size_t max_size = std::allocator_traits<Allocator>::max_size(alloc_);
  size_t old_capacity = this->capacity();
  size_t new_capacity = old_capacity + old_capacity / 2;
  if (size > new_capacity)
    new_capacity = size;
  else if (new_capacity > max_size)
    new_capacity = size > max_size ? size : max_size;
  T* old_data = this->data();
  T* new_data =
      std::allocator_traits<Allocator>::allocate(alloc_, new_capacity);
  // The following code doesn't throw, so the raw pointer above doesn't leak.
  std::uninitialized_copy(old_data, old_data + this->size(),
                          detail::make_checked(new_data, new_capacity));
  this->set(new_data, new_capacity);
  // deallocate must not throw according to the standard, but even if it does,
  // the buffer already uses the new storage and will deallocate it in
  // destructor.
  if (old_data != store_) alloc_.deallocate(old_data, old_capacity);
}

using memory_buffer = basic_memory_buffer<char>;

template <typename T, size_t SIZE, typename Allocator>
struct is_contiguous<basic_memory_buffer<T, SIZE, Allocator>> : std::true_type {
};

namespace detail {
FMT_API void print(std::FILE*, string_view);
}

/** A formatting error such as invalid format string. */
FMT_CLASS_API
class FMT_API format_error : public std::runtime_error {
 public:
  explicit format_error(const char* message) : std::runtime_error(message) {}
  explicit format_error(const std::string& message)
      : std::runtime_error(message) {}
  format_error(const format_error&) = default;
  format_error& operator=(const format_error&) = default;
  format_error(format_error&&) = default;
  format_error& operator=(format_error&&) = default;
  ~format_error() FMT_NOEXCEPT override FMT_MSC_DEFAULT;
};

/**
  \rst
  Constructs a `~fmt::format_arg_store` object that contains references
  to arguments and can be implicitly converted to `~fmt::format_args`.
  If ``fmt`` is a compile-time string then `make_args_checked` checks
  its validity at compile time.
  \endrst
 */
template <typename... Args, typename S, typename Char = char_t<S>>
FMT_INLINE auto make_args_checked(const S& fmt,
                                  const remove_reference_t<Args>&... args)
    -> format_arg_store<buffer_context<Char>, remove_reference_t<Args>...> {
  static_assert(
      detail::count<(
              std::is_base_of<detail::view, remove_reference_t<Args>>::value &&
              std::is_reference<Args>::value)...>() == 0,
      "passing views as lvalues is disallowed");
  detail::check_format_string<Args...>(fmt);
  return {args...};
}

// compile-time support
namespace detail_exported {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <typename Char, size_t N> struct fixed_string {
  constexpr fixed_string(const Char (&str)[N]) {
    detail::copy_str<Char, const Char*, Char*>(static_cast<const Char*>(str),
                                               str + N, data);
  }
  Char data[N]{};
};
#endif

// Converts a compile-time string to basic_string_view.
template <typename Char, size_t N>
constexpr auto compile_string_to_view(const Char (&s)[N])
    -> basic_string_view<Char> {
  // Remove trailing NUL character if needed. Won't be present if this is used
  // with a raw character array (i.e. not defined as a string).
  return {s, N - (std::char_traits<Char>::to_int_type(s[N - 1]) == 0 ? 1 : 0)};
}
template <typename Char>
constexpr auto compile_string_to_view(detail::std_string_view<Char> s)
    -> basic_string_view<Char> {
  return {s.data(), s.size()};
}
}  // namespace detail_exported

FMT_BEGIN_DETAIL_NAMESPACE

template <typename T> struct is_integral : std::is_integral<T> {};
template <> struct is_integral<int128_t> : std::true_type {};
template <> struct is_integral<uint128_t> : std::true_type {};

template <typename T>
using is_signed =
    std::integral_constant<bool, std::numeric_limits<T>::is_signed ||
                                     std::is_same<T, int128_t>::value>;

// Returns true if value is negative, false otherwise.
// Same as `value < 0` but doesn't produce warnings if T is an unsigned type.
template <typename T, FMT_ENABLE_IF(is_signed<T>::value)>
FMT_CONSTEXPR auto is_negative(T value) -> bool {
  return value < 0;
}
template <typename T, FMT_ENABLE_IF(!is_signed<T>::value)>
FMT_CONSTEXPR auto is_negative(T) -> bool {
  return false;
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR auto is_supported_floating_point(T) -> uint16_t {
  return (std::is_same<T, float>::value && FMT_USE_FLOAT) ||
         (std::is_same<T, double>::value && FMT_USE_DOUBLE) ||
         (std::is_same<T, long double>::value && FMT_USE_LONG_DOUBLE);
}

// Smallest of uint32_t, uint64_t, uint128_t that is large enough to
// represent all values of an integral type T.
template <typename T>
using uint32_or_64_or_128_t =
    conditional_t<num_bits<T>() <= 32 && !FMT_REDUCE_INT_INSTANTIATIONS,
                  uint32_t,
                  conditional_t<num_bits<T>() <= 64, uint64_t, uint128_t>>;
template <typename T>
using uint64_or_128_t = conditional_t<num_bits<T>() <= 64, uint64_t, uint128_t>;

#define FMT_POWERS_OF_10(factor)                                             \
  factor * 10, (factor)*100, (factor)*1000, (factor)*10000, (factor)*100000, \
      (factor)*1000000, (factor)*10000000, (factor)*100000000,               \
      (factor)*1000000000

// Converts value in the range [0, 100) to a string.
constexpr const char* digits2(size_t value) {
  // GCC generates slightly better code when value is pointer-size.
  return &"0001020304050607080910111213141516171819"
         "2021222324252627282930313233343536373839"
         "4041424344454647484950515253545556575859"
         "6061626364656667686970717273747576777879"
         "8081828384858687888990919293949596979899"[value * 2];
}

// Sign is a template parameter to workaround a bug in gcc 4.8.
template <typename Char, typename Sign> constexpr Char sign(Sign s) {
#if !FMT_GCC_VERSION || FMT_GCC_VERSION >= 604
  static_assert(std::is_same<Sign, sign_t>::value, "");
#endif
  return static_cast<Char>("\0-+ "[s]);
}

template <typename T> FMT_CONSTEXPR auto count_digits_fallback(T n) -> int {
  int count = 1;
  for (;;) {
    // Integer division is slow so do it for a group of four digits instead
    // of for every digit. The idea comes from the talk by Alexandrescu
    // "Three Optimization Tips for C++". See speed-test for a comparison.
    if (n < 10) return count;
    if (n < 100) return count + 1;
    if (n < 1000) return count + 2;
    if (n < 10000) return count + 3;
    n /= 10000u;
    count += 4;
  }
}
#if FMT_USE_INT128
FMT_CONSTEXPR inline auto count_digits(uint128_t n) -> int {
  return count_digits_fallback(n);
}
#endif

#ifdef FMT_BUILTIN_CLZLL
// It is a separate function rather than a part of count_digits to workaround
// the lack of static constexpr in constexpr functions.
inline auto do_count_digits(uint64_t n) -> int {
  // This has comparable performance to the version by Kendall Willets
  // (https://github.com/fmtlib/format-benchmark/blob/master/digits10)
  // but uses smaller tables.
  // Maps bsr(n) to ceil(log10(pow(2, bsr(n) + 1) - 1)).
  static constexpr uint8_t bsr2log10[] = {
      1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,
      6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10, 10,
      10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15,
      15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20};
  auto t = bsr2log10[FMT_BUILTIN_CLZLL(n | 1) ^ 63];
  static constexpr const uint64_t zero_or_powers_of_10[] = {
      0, 0, FMT_POWERS_OF_10(1U), FMT_POWERS_OF_10(1000000000ULL),
      10000000000000000000ULL};
  return t - (n < zero_or_powers_of_10[t]);
}
#endif

// Returns the number of decimal digits in n. Leading zeros are not counted
// except for n == 0 in which case count_digits returns 1.
FMT_CONSTEXPR20 inline auto count_digits(uint64_t n) -> int {
#ifdef FMT_BUILTIN_CLZLL
  if (!is_constant_evaluated()) {
    return do_count_digits(n);
  }
#endif
  return count_digits_fallback(n);
}

// Counts the number of digits in n. BITS = log2(radix).
template <int BITS, typename UInt>
FMT_CONSTEXPR auto count_digits(UInt n) -> int {
#ifdef FMT_BUILTIN_CLZ
  if (num_bits<UInt>() == 32)
    return (FMT_BUILTIN_CLZ(static_cast<uint32_t>(n) | 1) ^ 31) / BITS + 1;
#endif
  // Lambda avoids unreachable code warnings from NVHPC.
  return [](UInt m) {
    int num_digits = 0;
    do {
      ++num_digits;
    } while ((m >>= BITS) != 0);
    return num_digits;
  }(n);
}

template <> auto count_digits<4>(detail::fallback_uintptr n) -> int;

#ifdef FMT_BUILTIN_CLZ
// It is a separate function rather than a part of count_digits to workaround
// the lack of static constexpr in constexpr functions.
FMT_INLINE auto do_count_digits(uint32_t n) -> int {
// An optimization by Kendall Willets from https://bit.ly/3uOIQrB.
// This increments the upper 32 bits (log10(T) - 1) when >= T is added.
#  define FMT_INC(T) (((sizeof(#  T) - 1ull) << 32) - T)
  static constexpr uint64_t table[] = {
      FMT_INC(0),          FMT_INC(0),          FMT_INC(0),           // 8
      FMT_INC(10),         FMT_INC(10),         FMT_INC(10),          // 64
      FMT_INC(100),        FMT_INC(100),        FMT_INC(100),         // 512
      FMT_INC(1000),       FMT_INC(1000),       FMT_INC(1000),        // 4096
      FMT_INC(10000),      FMT_INC(10000),      FMT_INC(10000),       // 32k
      FMT_INC(100000),     FMT_INC(100000),     FMT_INC(100000),      // 256k
      FMT_INC(1000000),    FMT_INC(1000000),    FMT_INC(1000000),     // 2048k
      FMT_INC(10000000),   FMT_INC(10000000),   FMT_INC(10000000),    // 16M
      FMT_INC(100000000),  FMT_INC(100000000),  FMT_INC(100000000),   // 128M
      FMT_INC(1000000000), FMT_INC(1000000000), FMT_INC(1000000000),  // 1024M
      FMT_INC(1000000000), FMT_INC(1000000000)                        // 4B
  };
  auto inc = table[FMT_BUILTIN_CLZ(n | 1) ^ 31];
  return static_cast<int>((n + inc) >> 32);
}
#endif

// Optional version of count_digits for better performance on 32-bit platforms.
FMT_CONSTEXPR20 inline auto count_digits(uint32_t n) -> int {
#ifdef FMT_BUILTIN_CLZ
  if (!is_constant_evaluated()) {
    return do_count_digits(n);
  }
#endif
  return count_digits_fallback(n);
}

template <typename Int> constexpr auto digits10() FMT_NOEXCEPT -> int {
  return std::numeric_limits<Int>::digits10;
}
template <> constexpr auto digits10<int128_t>() FMT_NOEXCEPT -> int {
  return 38;
}
template <> constexpr auto digits10<uint128_t>() FMT_NOEXCEPT -> int {
  return 38;
}

template <typename Char> struct thousands_sep_result {
  std::string grouping;
  Char thousands_sep;
};

template <typename Char>
FMT_API auto thousands_sep_impl(locale_ref loc) -> thousands_sep_result<Char>;
template <typename Char>
inline auto thousands_sep(locale_ref loc) -> thousands_sep_result<Char> {
  auto result = thousands_sep_impl<char>(loc);
  return {result.grouping, Char(result.thousands_sep)};
}
template <>
inline auto thousands_sep(locale_ref loc) -> thousands_sep_result<wchar_t> {
  return thousands_sep_impl<wchar_t>(loc);
}

template <typename Char>
FMT_API auto decimal_point_impl(locale_ref loc) -> Char;
template <typename Char> inline auto decimal_point(locale_ref loc) -> Char {
  return Char(decimal_point_impl<char>(loc));
}
template <> inline auto decimal_point(locale_ref loc) -> wchar_t {
  return decimal_point_impl<wchar_t>(loc);
}

// Compares two characters for equality.
template <typename Char> auto equal2(const Char* lhs, const char* rhs) -> bool {
  return lhs[0] == Char(rhs[0]) && lhs[1] == Char(rhs[1]);
}
inline auto equal2(const char* lhs, const char* rhs) -> bool {
  return memcmp(lhs, rhs, 2) == 0;
}

// Copies two characters from src to dst.
template <typename Char>
FMT_CONSTEXPR20 FMT_INLINE void copy2(Char* dst, const char* src) {
  if (!is_constant_evaluated() && sizeof(Char) == sizeof(char)) {
    memcpy(dst, src, 2);
    return;
  }
  *dst++ = static_cast<Char>(*src++);
  *dst = static_cast<Char>(*src);
}

template <typename Iterator> struct format_decimal_result {
  Iterator begin;
  Iterator end;
};

// Formats a decimal unsigned integer value writing into out pointing to a
// buffer of specified size. The caller must ensure that the buffer is large
// enough.
template <typename Char, typename UInt>
FMT_CONSTEXPR20 auto format_decimal(Char* out, UInt value, int size)
    -> format_decimal_result<Char*> {
  FMT_ASSERT(size >= count_digits(value), "invalid digit count");
  out += size;
  Char* end = out;
  while (value >= 100) {
    // Integer division is slow so do it for a group of two digits instead
    // of for every digit. The idea comes from the talk by Alexandrescu
    // "Three Optimization Tips for C++". See speed-test for a comparison.
    out -= 2;
    copy2(out, digits2(static_cast<size_t>(value % 100)));
    value /= 100;
  }
  if (value < 10) {
    *--out = static_cast<Char>('0' + value);
    return {out, end};
  }
  out -= 2;
  copy2(out, digits2(static_cast<size_t>(value)));
  return {out, end};
}

template <typename Char, typename UInt, typename Iterator,
          FMT_ENABLE_IF(!std::is_pointer<remove_cvref_t<Iterator>>::value)>
inline auto format_decimal(Iterator out, UInt value, int size)
    -> format_decimal_result<Iterator> {
  // Buffer is large enough to hold all digits (digits10 + 1).
  Char buffer[digits10<UInt>() + 1];
  auto end = format_decimal(buffer, value, size).end;
  return {out, detail::copy_str_noinline<Char>(buffer, end, out)};
}

template <unsigned BASE_BITS, typename Char, typename UInt>
FMT_CONSTEXPR auto format_uint(Char* buffer, UInt value, int num_digits,
                               bool upper = false) -> Char* {
  buffer += num_digits;
  Char* end = buffer;
  do {
    const char* digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    unsigned digit = (value & ((1 << BASE_BITS) - 1));
    *--buffer = static_cast<Char>(BASE_BITS < 4 ? static_cast<char>('0' + digit)
                                                : digits[digit]);
  } while ((value >>= BASE_BITS) != 0);
  return end;
}

template <unsigned BASE_BITS, typename Char>
auto format_uint(Char* buffer, detail::fallback_uintptr n, int num_digits,
                 bool = false) -> Char* {
  auto char_digits = std::numeric_limits<unsigned char>::digits / 4;
  int start = (num_digits + char_digits - 1) / char_digits - 1;
  if (int start_digits = num_digits % char_digits) {
    unsigned value = n.value[start--];
    buffer = format_uint<BASE_BITS>(buffer, value, start_digits);
  }
  for (; start >= 0; --start) {
    unsigned value = n.value[start];
    buffer += char_digits;
    auto p = buffer;
    for (int i = 0; i < char_digits; ++i) {
      unsigned digit = (value & ((1 << BASE_BITS) - 1));
      *--p = static_cast<Char>("0123456789abcdef"[digit]);
      value >>= BASE_BITS;
    }
  }
  return buffer;
}

template <unsigned BASE_BITS, typename Char, typename It, typename UInt>
inline auto format_uint(It out, UInt value, int num_digits, bool upper = false)
    -> It {
  if (auto ptr = to_pointer<Char>(out, to_unsigned(num_digits))) {
    format_uint<BASE_BITS>(ptr, value, num_digits, upper);
    return out;
  }
  // Buffer should be large enough to hold all digits (digits / BASE_BITS + 1).
  char buffer[num_bits<UInt>() / BASE_BITS + 1];
  format_uint<BASE_BITS>(buffer, value, num_digits, upper);
  return detail::copy_str_noinline<Char>(buffer, buffer + num_digits, out);
}

// A converter from UTF-8 to UTF-16.
class utf8_to_utf16 {
 private:
  basic_memory_buffer<wchar_t> buffer_;

 public:
  FMT_API explicit utf8_to_utf16(string_view s);
  operator basic_string_view<wchar_t>() const { return {&buffer_[0], size()}; }
  auto size() const -> size_t { return buffer_.size() - 1; }
  auto c_str() const -> const wchar_t* { return &buffer_[0]; }
  auto str() const -> std::wstring { return {&buffer_[0], size()}; }
};

namespace dragonbox {

// Type-specific information that Dragonbox uses.
template <class T> struct float_info;

template <> struct float_info<float> {
  using carrier_uint = uint32_t;
  static const int significand_bits = 23;
  static const int exponent_bits = 8;
  static const int min_exponent = -126;
  static const int max_exponent = 127;
  static const int exponent_bias = -127;
  static const int decimal_digits = 9;
  static const int kappa = 1;
  static const int big_divisor = 100;
  static const int small_divisor = 10;
  static const int min_k = -31;
  static const int max_k = 46;
  static const int cache_bits = 64;
  static const int divisibility_check_by_5_threshold = 39;
  static const int case_fc_pm_half_lower_threshold = -1;
  static const int case_fc_pm_half_upper_threshold = 6;
  static const int case_fc_lower_threshold = -2;
  static const int case_fc_upper_threshold = 6;
  static const int case_shorter_interval_left_endpoint_lower_threshold = 2;
  static const int case_shorter_interval_left_endpoint_upper_threshold = 3;
  static const int shorter_interval_tie_lower_threshold = -35;
  static const int shorter_interval_tie_upper_threshold = -35;
  static const int max_trailing_zeros = 7;
};

template <> struct float_info<double> {
  using carrier_uint = uint64_t;
  static const int significand_bits = 52;
  static const int exponent_bits = 11;
  static const int min_exponent = -1022;
  static const int max_exponent = 1023;
  static const int exponent_bias = -1023;
  static const int decimal_digits = 17;
  static const int kappa = 2;
  static const int big_divisor = 1000;
  static const int small_divisor = 100;
  static const int min_k = -292;
  static const int max_k = 326;
  static const int cache_bits = 128;
  static const int divisibility_check_by_5_threshold = 86;
  static const int case_fc_pm_half_lower_threshold = -2;
  static const int case_fc_pm_half_upper_threshold = 9;
  static const int case_fc_lower_threshold = -4;
  static const int case_fc_upper_threshold = 9;
  static const int case_shorter_interval_left_endpoint_lower_threshold = 2;
  static const int case_shorter_interval_left_endpoint_upper_threshold = 3;
  static const int shorter_interval_tie_lower_threshold = -77;
  static const int shorter_interval_tie_upper_threshold = -77;
  static const int max_trailing_zeros = 16;
};

template <typename T> struct decimal_fp {
  using significand_type = typename float_info<T>::carrier_uint;
  significand_type significand;
  int exponent;
};

template <typename T>
FMT_API auto to_decimal(T x) FMT_NOEXCEPT -> decimal_fp<T>;
}  // namespace dragonbox

template <typename T>
constexpr auto exponent_mask() ->
    typename dragonbox::float_info<T>::carrier_uint {
  using uint = typename dragonbox::float_info<T>::carrier_uint;
  return ((uint(1) << dragonbox::float_info<T>::exponent_bits) - 1)
         << dragonbox::float_info<T>::significand_bits;
}

// Writes the exponent exp in the form "[+-]d{2,3}" to buffer.
template <typename Char, typename It>
FMT_CONSTEXPR auto write_exponent(int exp, It it) -> It {
  FMT_ASSERT(-10000 < exp && exp < 10000, "exponent out of range");
  if (exp < 0) {
    *it++ = static_cast<Char>('-');
    exp = -exp;
  } else {
    *it++ = static_cast<Char>('+');
  }
  if (exp >= 100) {
    const char* top = digits2(to_unsigned(exp / 100));
    if (exp >= 1000) *it++ = static_cast<Char>(top[0]);
    *it++ = static_cast<Char>(top[1]);
    exp %= 100;
  }
  const char* d = digits2(to_unsigned(exp));
  *it++ = static_cast<Char>(d[0]);
  *it++ = static_cast<Char>(d[1]);
  return it;
}

template <typename T>
FMT_HEADER_ONLY_CONSTEXPR20 auto format_float(T value, int precision,
                                              float_specs specs,
                                              buffer<char>& buf) -> int;

// Formats a floating-point number with snprintf.
template <typename T>
auto snprintf_float(T value, int precision, float_specs specs,
                    buffer<char>& buf) -> int;

template <typename T> constexpr auto promote_float(T value) -> T {
  return value;
}
constexpr auto promote_float(float value) -> double {
  return static_cast<double>(value);
}

template <typename OutputIt, typename Char>
FMT_NOINLINE FMT_CONSTEXPR auto fill(OutputIt it, size_t n,
                                     const fill_t<Char>& fill) -> OutputIt {
  auto fill_size = fill.size();
  if (fill_size == 1) return detail::fill_n(it, n, fill[0]);
  auto data = fill.data();
  for (size_t i = 0; i < n; ++i)
    it = copy_str<Char>(data, data + fill_size, it);
  return it;
}

// Writes the output of f, padded according to format specifications in specs.
// size: output size in code units.
// width: output display width in (terminal) column positions.
template <align::type align = align::left, typename OutputIt, typename Char,
          typename F>
FMT_CONSTEXPR auto write_padded(OutputIt out,
                                const basic_format_specs<Char>& specs,
                                size_t size, size_t width, F&& f) -> OutputIt {
  static_assert(align == align::left || align == align::right, "");
  unsigned spec_width = to_unsigned(specs.width);
  size_t padding = spec_width > width ? spec_width - width : 0;
  // Shifts are encoded as string literals because static constexpr is not
  // supported in constexpr functions.
  auto* shifts = align == align::left ? "\x1f\x1f\x00\x01" : "\x00\x1f\x00\x01";
  size_t left_padding = padding >> shifts[specs.align];
  size_t right_padding = padding - left_padding;
  auto it = reserve(out, size + padding * specs.fill.size());
  if (left_padding != 0) it = fill(it, left_padding, specs.fill);
  it = f(it);
  if (right_padding != 0) it = fill(it, right_padding, specs.fill);
  return base_iterator(out, it);
}

template <align::type align = align::left, typename OutputIt, typename Char,
          typename F>
constexpr auto write_padded(OutputIt out, const basic_format_specs<Char>& specs,
                            size_t size, F&& f) -> OutputIt {
  return write_padded<align>(out, specs, size, size, f);
}

template <align::type align = align::left, typename Char, typename OutputIt>
FMT_CONSTEXPR auto write_bytes(OutputIt out, string_view bytes,
                               const basic_format_specs<Char>& specs)
    -> OutputIt {
  return write_padded<align>(
      out, specs, bytes.size(), [bytes](reserve_iterator<OutputIt> it) {
        const char* data = bytes.data();
        return copy_str<Char>(data, data + bytes.size(), it);
      });
}

template <typename Char, typename OutputIt, typename UIntPtr>
auto write_ptr(OutputIt out, UIntPtr value,
               const basic_format_specs<Char>* specs) -> OutputIt {
  int num_digits = count_digits<4>(value);
  auto size = to_unsigned(num_digits) + size_t(2);
  auto write = [=](reserve_iterator<OutputIt> it) {
    *it++ = static_cast<Char>('0');
    *it++ = static_cast<Char>('x');
    return format_uint<4, Char>(it, value, num_digits);
  };
  return specs ? write_padded<align::right>(out, *specs, size, write)
               : base_iterator(out, write(reserve(out, size)));
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write_char(OutputIt out, Char value,
                              const basic_format_specs<Char>& specs)
    -> OutputIt {
  return write_padded(out, specs, 1, [=](reserve_iterator<OutputIt> it) {
    *it++ = value;
    return it;
  });
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, Char value,
                         const basic_format_specs<Char>& specs,
                         locale_ref loc = {}) -> OutputIt {
  return check_char_specs(specs)
             ? write_char(out, value, specs)
             : write(out, static_cast<int>(value), specs, loc);
}

// Data for write_int that doesn't depend on output iterator type. It is used to
// avoid template code bloat.
template <typename Char> struct write_int_data {
  size_t size;
  size_t padding;

  FMT_CONSTEXPR write_int_data(int num_digits, unsigned prefix,
                               const basic_format_specs<Char>& specs)
      : size((prefix >> 24) + to_unsigned(num_digits)), padding(0) {
    if (specs.align == align::numeric) {
      auto width = to_unsigned(specs.width);
      if (width > size) {
        padding = width - size;
        size = width;
      }
    } else if (specs.precision > num_digits) {
      size = (prefix >> 24) + to_unsigned(specs.precision);
      padding = to_unsigned(specs.precision - num_digits);
    }
  }
};

// Writes an integer in the format
//   <left-padding><prefix><numeric-padding><digits><right-padding>
// where <digits> are written by write_digits(it).
// prefix contains chars in three lower bytes and the size in the fourth byte.
template <typename OutputIt, typename Char, typename W>
FMT_CONSTEXPR FMT_INLINE auto write_int(OutputIt out, int num_digits,
                                        unsigned prefix,
                                        const basic_format_specs<Char>& specs,
                                        W write_digits) -> OutputIt {
  // Slightly faster check for specs.width == 0 && specs.precision == -1.
  if ((specs.width | (specs.precision + 1)) == 0) {
    auto it = reserve(out, to_unsigned(num_digits) + (prefix >> 24));
    if (prefix != 0) {
      for (unsigned p = prefix & 0xffffff; p != 0; p >>= 8)
        *it++ = static_cast<Char>(p & 0xff);
    }
    return base_iterator(out, write_digits(it));
  }
  auto data = write_int_data<Char>(num_digits, prefix, specs);
  return write_padded<align::right>(
      out, specs, data.size, [=](reserve_iterator<OutputIt> it) {
        for (unsigned p = prefix & 0xffffff; p != 0; p >>= 8)
          *it++ = static_cast<Char>(p & 0xff);
        it = detail::fill_n(it, data.padding, static_cast<Char>('0'));
        return write_digits(it);
      });
}

template <typename Char> class digit_grouping {
 private:
  thousands_sep_result<Char> sep_;

  struct next_state {
    std::string::const_iterator group;
    int pos;
  };
  next_state initial_state() const { return {sep_.grouping.begin(), 0}; }

  // Returns the next digit group separator position.
  int next(next_state& state) const {
    if (!sep_.thousands_sep) return max_value<int>();
    if (state.group == sep_.grouping.end())
      return state.pos += sep_.grouping.back();
    if (*state.group <= 0 || *state.group == max_value<char>())
      return max_value<int>();
    state.pos += *state.group++;
    return state.pos;
  }

 public:
  explicit digit_grouping(locale_ref loc, bool localized = true) {
    if (localized)
      sep_ = thousands_sep<Char>(loc);
    else
      sep_.thousands_sep = Char();
  }
  explicit digit_grouping(thousands_sep_result<Char> sep) : sep_(sep) {}

  Char separator() const { return sep_.thousands_sep; }

  int count_separators(int num_digits) const {
    int count = 0;
    auto state = initial_state();
    while (num_digits > next(state)) ++count;
    return count;
  }

  // Applies grouping to digits and write the output to out.
  template <typename Out, typename C>
  Out apply(Out out, basic_string_view<C> digits) const {
    auto num_digits = static_cast<int>(digits.size());
    auto separators = basic_memory_buffer<int>();
    separators.push_back(0);
    auto state = initial_state();
    while (int i = next(state)) {
      if (i >= num_digits) break;
      separators.push_back(i);
    }
    for (int i = 0, sep_index = static_cast<int>(separators.size() - 1);
         i < num_digits; ++i) {
      if (num_digits - i == separators[sep_index]) {
        *out++ = separator();
        --sep_index;
      }
      *out++ = static_cast<Char>(digits[to_unsigned(i)]);
    }
    return out;
  }
};

template <typename OutputIt, typename UInt, typename Char>
auto write_int_localized(OutputIt out, UInt value, unsigned prefix,
                         const basic_format_specs<Char>& specs,
                         const digit_grouping<Char>& grouping) -> OutputIt {
  static_assert(std::is_same<uint64_or_128_t<UInt>, UInt>::value, "");
  int num_digits = count_digits(value);
  char digits[40];
  format_decimal(digits, value, num_digits);
  unsigned size = to_unsigned((prefix != 0 ? 1 : 0) + num_digits +
                              grouping.count_separators(num_digits));
  return write_padded<align::right>(
      out, specs, size, size, [&](reserve_iterator<OutputIt> it) {
        if (prefix != 0) *it++ = static_cast<Char>(prefix);
        return grouping.apply(it, string_view(digits, to_unsigned(num_digits)));
      });
}

template <typename OutputIt, typename UInt, typename Char>
auto write_int_localized(OutputIt& out, UInt value, unsigned prefix,
                         const basic_format_specs<Char>& specs, locale_ref loc)
    -> bool {
  auto grouping = digit_grouping<Char>(loc);
  out = write_int_localized(out, value, prefix, specs, grouping);
  return true;
}

FMT_CONSTEXPR inline void prefix_append(unsigned& prefix, unsigned value) {
  prefix |= prefix != 0 ? value << 8 : value;
  prefix += (1u + (value > 0xff ? 1 : 0)) << 24;
}

template <typename UInt> struct write_int_arg {
  UInt abs_value;
  unsigned prefix;
};

template <typename T>
FMT_CONSTEXPR auto make_write_int_arg(T value, sign_t sign)
    -> write_int_arg<uint32_or_64_or_128_t<T>> {
  auto prefix = 0u;
  auto abs_value = static_cast<uint32_or_64_or_128_t<T>>(value);
  if (is_negative(value)) {
    prefix = 0x01000000 | '-';
    abs_value = 0 - abs_value;
  } else {
    constexpr const unsigned prefixes[4] = {0, 0, 0x1000000u | '+',
                                            0x1000000u | ' '};
    prefix = prefixes[sign];
  }
  return {abs_value, prefix};
}

template <typename Char, typename OutputIt, typename T>
FMT_CONSTEXPR FMT_INLINE auto write_int(OutputIt out, write_int_arg<T> arg,
                                        const basic_format_specs<Char>& specs,
                                        locale_ref loc) -> OutputIt {
  static_assert(std::is_same<T, uint32_or_64_or_128_t<T>>::value, "");
  auto abs_value = arg.abs_value;
  auto prefix = arg.prefix;
  switch (specs.type) {
  case presentation_type::none:
  case presentation_type::dec: {
    if (specs.localized &&
        write_int_localized(out, static_cast<uint64_or_128_t<T>>(abs_value),
                            prefix, specs, loc)) {
      return out;
    }
    auto num_digits = count_digits(abs_value);
    return write_int(
        out, num_digits, prefix, specs, [=](reserve_iterator<OutputIt> it) {
          return format_decimal<Char>(it, abs_value, num_digits).end;
        });
  }
  case presentation_type::hex_lower:
  case presentation_type::hex_upper: {
    bool upper = specs.type == presentation_type::hex_upper;
    if (specs.alt)
      prefix_append(prefix, unsigned(upper ? 'X' : 'x') << 8 | '0');
    int num_digits = count_digits<4>(abs_value);
    return write_int(
        out, num_digits, prefix, specs, [=](reserve_iterator<OutputIt> it) {
          return format_uint<4, Char>(it, abs_value, num_digits, upper);
        });
  }
  case presentation_type::bin_lower:
  case presentation_type::bin_upper: {
    bool upper = specs.type == presentation_type::bin_upper;
    if (specs.alt)
      prefix_append(prefix, unsigned(upper ? 'B' : 'b') << 8 | '0');
    int num_digits = count_digits<1>(abs_value);
    return write_int(out, num_digits, prefix, specs,
                     [=](reserve_iterator<OutputIt> it) {
                       return format_uint<1, Char>(it, abs_value, num_digits);
                     });
  }
  case presentation_type::oct: {
    int num_digits = count_digits<3>(abs_value);
    // Octal prefix '0' is counted as a digit, so only add it if precision
    // is not greater than the number of digits.
    if (specs.alt && specs.precision <= num_digits && abs_value != 0)
      prefix_append(prefix, '0');
    return write_int(out, num_digits, prefix, specs,
                     [=](reserve_iterator<OutputIt> it) {
                       return format_uint<3, Char>(it, abs_value, num_digits);
                     });
  }
  case presentation_type::chr:
    return write_char(out, static_cast<Char>(abs_value), specs);
  default:
    throw_format_error("invalid type specifier");
  }
  return out;
}
template <typename Char, typename OutputIt, typename T>
FMT_CONSTEXPR FMT_NOINLINE auto write_int_noinline(
    OutputIt out, write_int_arg<T> arg, const basic_format_specs<Char>& specs,
    locale_ref loc) -> OutputIt {
  return write_int(out, arg, specs, loc);
}
template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        std::is_same<OutputIt, buffer_appender<Char>>::value)>
FMT_CONSTEXPR FMT_INLINE auto write(OutputIt out, T value,
                                    const basic_format_specs<Char>& specs,
                                    locale_ref loc) -> OutputIt {
  return write_int_noinline(out, make_write_int_arg(value, specs.sign), specs,
                            loc);
}
// An inlined version of write used in format string compilation.
template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        !std::is_same<OutputIt, buffer_appender<Char>>::value)>
FMT_CONSTEXPR FMT_INLINE auto write(OutputIt out, T value,
                                    const basic_format_specs<Char>& specs,
                                    locale_ref loc) -> OutputIt {
  return write_int(out, make_write_int_arg(value, specs.sign), specs, loc);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, basic_string_view<Char> s,
                         const basic_format_specs<Char>& specs) -> OutputIt {
  auto data = s.data();
  auto size = s.size();
  if (specs.precision >= 0 && to_unsigned(specs.precision) < size)
    size = code_point_index(s, to_unsigned(specs.precision));
  auto width =
      specs.width != 0 ? compute_width(basic_string_view<Char>(data, size)) : 0;
  return write_padded(out, specs, size, width,
                      [=](reserve_iterator<OutputIt> it) {
                        return copy_str<Char>(data, data + size, it);
                      });
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out,
                         basic_string_view<type_identity_t<Char>> s,
                         const basic_format_specs<Char>& specs, locale_ref)
    -> OutputIt {
  check_string_type_spec(specs.type);
  return write(out, s, specs);
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, const Char* s,
                         const basic_format_specs<Char>& specs, locale_ref)
    -> OutputIt {
  return check_cstring_type_spec(specs.type)
             ? write(out, basic_string_view<Char>(s), specs, {})
             : write_ptr<Char>(out, to_uintptr(s), &specs);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR20 auto write_nonfinite(OutputIt out, bool isinf,
                                     basic_format_specs<Char> specs,
                                     const float_specs& fspecs) -> OutputIt {
  auto str =
      isinf ? (fspecs.upper ? "INF" : "inf") : (fspecs.upper ? "NAN" : "nan");
  constexpr size_t str_size = 3;
  auto sign = fspecs.sign;
  auto size = str_size + (sign ? 1 : 0);
  // Replace '0'-padding with space for non-finite values.
  const bool is_zero_fill =
      specs.fill.size() == 1 && *specs.fill.data() == static_cast<Char>('0');
  if (is_zero_fill) specs.fill[0] = static_cast<Char>(' ');
  return write_padded(out, specs, size, [=](reserve_iterator<OutputIt> it) {
    if (sign) *it++ = detail::sign<Char>(sign);
    return copy_str<Char>(str, str + str_size, it);
  });
}

// A decimal floating-point number significand * pow(10, exp).
struct big_decimal_fp {
  const char* significand;
  int significand_size;
  int exponent;
};

constexpr auto get_significand_size(const big_decimal_fp& fp) -> int {
  return fp.significand_size;
}
template <typename T>
inline auto get_significand_size(const dragonbox::decimal_fp<T>& fp) -> int {
  return count_digits(fp.significand);
}

template <typename Char, typename OutputIt>
constexpr auto write_significand(OutputIt out, const char* significand,
                                 int significand_size) -> OutputIt {
  return copy_str<Char>(significand, significand + significand_size, out);
}
template <typename Char, typename OutputIt, typename UInt>
inline auto write_significand(OutputIt out, UInt significand,
                              int significand_size) -> OutputIt {
  return format_decimal<Char>(out, significand, significand_size).end;
}
template <typename Char, typename OutputIt, typename T, typename Grouping>
FMT_CONSTEXPR20 auto write_significand(OutputIt out, T significand,
                                       int significand_size, int exponent,
                                       const Grouping& grouping) -> OutputIt {
  if (!grouping.separator()) {
    out = write_significand<Char>(out, significand, significand_size);
    return detail::fill_n(out, exponent, static_cast<Char>('0'));
  }
  auto buffer = memory_buffer();
  write_significand<char>(appender(buffer), significand, significand_size);
  detail::fill_n(appender(buffer), exponent, '0');
  return grouping.apply(out, string_view(buffer.data(), buffer.size()));
}

template <typename Char, typename UInt,
          FMT_ENABLE_IF(std::is_integral<UInt>::value)>
inline auto write_significand(Char* out, UInt significand, int significand_size,
                              int integral_size, Char decimal_point) -> Char* {
  if (!decimal_point)
    return format_decimal(out, significand, significand_size).end;
  out += significand_size + 1;
  Char* end = out;
  int floating_size = significand_size - integral_size;
  for (int i = floating_size / 2; i > 0; --i) {
    out -= 2;
    copy2(out, digits2(significand % 100));
    significand /= 100;
  }
  if (floating_size % 2 != 0) {
    *--out = static_cast<Char>('0' + significand % 10);
    significand /= 10;
  }
  *--out = decimal_point;
  format_decimal(out - integral_size, significand, integral_size);
  return end;
}

template <typename OutputIt, typename UInt, typename Char,
          FMT_ENABLE_IF(!std::is_pointer<remove_cvref_t<OutputIt>>::value)>
inline auto write_significand(OutputIt out, UInt significand,
                              int significand_size, int integral_size,
                              Char decimal_point) -> OutputIt {
  // Buffer is large enough to hold digits (digits10 + 1) and a decimal point.
  Char buffer[digits10<UInt>() + 2];
  auto end = write_significand(buffer, significand, significand_size,
                               integral_size, decimal_point);
  return detail::copy_str_noinline<Char>(buffer, end, out);
}

template <typename OutputIt, typename Char>
FMT_CONSTEXPR auto write_significand(OutputIt out, const char* significand,
                                     int significand_size, int integral_size,
                                     Char decimal_point) -> OutputIt {
  out = detail::copy_str_noinline<Char>(significand,
                                        significand + integral_size, out);
  if (!decimal_point) return out;
  *out++ = decimal_point;
  return detail::copy_str_noinline<Char>(significand + integral_size,
                                         significand + significand_size, out);
}

template <typename OutputIt, typename Char, typename T, typename Grouping>
FMT_CONSTEXPR20 auto write_significand(OutputIt out, T significand,
                                       int significand_size, int integral_size,
                                       Char decimal_point,
                                       const Grouping& grouping) -> OutputIt {
  if (!grouping.separator()) {
    return write_significand(out, significand, significand_size, integral_size,
                             decimal_point);
  }
  auto buffer = basic_memory_buffer<Char>();
  write_significand(buffer_appender<Char>(buffer), significand,
                    significand_size, integral_size, decimal_point);
  grouping.apply(
      out, basic_string_view<Char>(buffer.data(), to_unsigned(integral_size)));
  return detail::copy_str_noinline<Char>(buffer.data() + integral_size,
                                         buffer.end(), out);
}

template <typename OutputIt, typename DecimalFP, typename Char,
          typename Grouping = digit_grouping<Char>>
FMT_CONSTEXPR20 auto do_write_float(OutputIt out, const DecimalFP& fp,
                                    const basic_format_specs<Char>& specs,
                                    float_specs fspecs, locale_ref loc)
    -> OutputIt {
  auto significand = fp.significand;
  int significand_size = get_significand_size(fp);
  constexpr Char zero = static_cast<Char>('0');
  auto sign = fspecs.sign;
  size_t size = to_unsigned(significand_size) + (sign ? 1 : 0);
  using iterator = reserve_iterator<OutputIt>;

  Char decimal_point =
      fspecs.locale ? detail::decimal_point<Char>(loc) : static_cast<Char>('.');

  int output_exp = fp.exponent + significand_size - 1;
  auto use_exp_format = [=]() {
    if (fspecs.format == float_format::exp) return true;
    if (fspecs.format != float_format::general) return false;
    // Use the fixed notation if the exponent is in [exp_lower, exp_upper),
    // e.g. 0.0001 instead of 1e-04. Otherwise use the exponent notation.
    const int exp_lower = -4, exp_upper = 16;
    return output_exp < exp_lower ||
           output_exp >= (fspecs.precision > 0 ? fspecs.precision : exp_upper);
  };
  if (use_exp_format()) {
    int num_zeros = 0;
    if (fspecs.showpoint) {
      num_zeros = fspecs.precision - significand_size;
      if (num_zeros < 0) num_zeros = 0;
      size += to_unsigned(num_zeros);
    } else if (significand_size == 1) {
      decimal_point = Char();
    }
    auto abs_output_exp = output_exp >= 0 ? output_exp : -output_exp;
    int exp_digits = 2;
    if (abs_output_exp >= 100) exp_digits = abs_output_exp >= 1000 ? 4 : 3;

    size += to_unsigned((decimal_point ? 1 : 0) + 2 + exp_digits);
    char exp_char = fspecs.upper ? 'E' : 'e';
    auto write = [=](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      // Insert a decimal point after the first digit and add an exponent.
      it = write_significand(it, significand, significand_size, 1,
                             decimal_point);
      if (num_zeros > 0) it = detail::fill_n(it, num_zeros, zero);
      *it++ = static_cast<Char>(exp_char);
      return write_exponent<Char>(output_exp, it);
    };
    return specs.width > 0 ? write_padded<align::right>(out, specs, size, write)
                           : base_iterator(out, write(reserve(out, size)));
  }

  int exp = fp.exponent + significand_size;
  if (fp.exponent >= 0) {
    // 1234e5 -> 123400000[.0+]
    size += to_unsigned(fp.exponent);
    int num_zeros = fspecs.precision - exp;
#ifdef FMT_FUZZ
    if (num_zeros > 5000)
      throw std::runtime_error("fuzz mode - avoiding excessive cpu use");
#endif
    if (fspecs.showpoint) {
      if (num_zeros <= 0 && fspecs.format != float_format::fixed) num_zeros = 1;
      if (num_zeros > 0) size += to_unsigned(num_zeros) + 1;
    }
    auto grouping = Grouping(loc, fspecs.locale);
    size += to_unsigned(grouping.count_separators(significand_size));
    return write_padded<align::right>(out, specs, size, [&](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      it = write_significand<Char>(it, significand, significand_size,
                                   fp.exponent, grouping);
      if (!fspecs.showpoint) return it;
      *it++ = decimal_point;
      return num_zeros > 0 ? detail::fill_n(it, num_zeros, zero) : it;
    });
  } else if (exp > 0) {
    // 1234e-2 -> 12.34[0+]
    int num_zeros = fspecs.showpoint ? fspecs.precision - significand_size : 0;
    size += 1 + to_unsigned(num_zeros > 0 ? num_zeros : 0);
    auto grouping = Grouping(loc, fspecs.locale);
    size += to_unsigned(grouping.count_separators(significand_size));
    return write_padded<align::right>(out, specs, size, [&](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      it = write_significand(it, significand, significand_size, exp,
                             decimal_point, grouping);
      return num_zeros > 0 ? detail::fill_n(it, num_zeros, zero) : it;
    });
  }
  // 1234e-6 -> 0.001234
  int num_zeros = -exp;
  if (significand_size == 0 && fspecs.precision >= 0 &&
      fspecs.precision < num_zeros) {
    num_zeros = fspecs.precision;
  }
  bool pointy = num_zeros != 0 || significand_size != 0 || fspecs.showpoint;
  size += 1 + (pointy ? 1 : 0) + to_unsigned(num_zeros);
  return write_padded<align::right>(out, specs, size, [&](iterator it) {
    if (sign) *it++ = detail::sign<Char>(sign);
    *it++ = zero;
    if (!pointy) return it;
    *it++ = decimal_point;
    it = detail::fill_n(it, num_zeros, zero);
    return write_significand<Char>(it, significand, significand_size);
  });
}

template <typename Char> class fallback_digit_grouping {
 public:
  constexpr fallback_digit_grouping(locale_ref, bool) {}

  constexpr Char separator() const { return Char(); }

  constexpr int count_separators(int) const { return 0; }

  template <typename Out, typename C>
  constexpr Out apply(Out out, basic_string_view<C>) const {
    return out;
  }
};

template <typename OutputIt, typename DecimalFP, typename Char>
FMT_CONSTEXPR20 auto write_float(OutputIt out, const DecimalFP& fp,
                                 const basic_format_specs<Char>& specs,
                                 float_specs fspecs, locale_ref loc)
    -> OutputIt {
  if (is_constant_evaluated()) {
    return do_write_float<OutputIt, DecimalFP, Char,
                          fallback_digit_grouping<Char>>(out, fp, specs, fspecs,
                                                         loc);
  } else {
    return do_write_float(out, fp, specs, fspecs, loc);
  }
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 bool isinf(T value) {
  if (is_constant_evaluated()) {
#if defined(__cpp_if_constexpr)
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      constexpr auto significand_bits =
          dragonbox::float_info<double>::significand_bits;
      return (bits & exponent_mask<double>()) &&
             !(bits & ((uint64_t(1) << significand_bits) - 1));
    }
#endif
  }
  return std::isinf(value);
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 bool isfinite(T value) {
  if (is_constant_evaluated()) {
#if defined(__cpp_if_constexpr)
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      return (bits & exponent_mask<double>()) != exponent_mask<double>();
    }
#endif
  }
  return std::isfinite(value);
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_INLINE FMT_CONSTEXPR bool signbit(T value) {
  if (is_constant_evaluated()) {
#ifdef __cpp_if_constexpr
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      return (bits & (uint64_t(1) << (num_bits<uint64_t>() - 1))) != 0;
    }
#endif
  }
  return std::signbit(value);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 auto write(OutputIt out, T value,
                           basic_format_specs<Char> specs, locale_ref loc = {})
    -> OutputIt {
  if (const_check(!is_supported_floating_point(value))) return out;
  float_specs fspecs = parse_float_type_spec(specs);
  fspecs.sign = specs.sign;
  if (detail::signbit(value)) {  // value < 0 is false for NaN so use signbit.
    fspecs.sign = sign::minus;
    value = -value;
  } else if (fspecs.sign == sign::minus) {
    fspecs.sign = sign::none;
  }

  if (!detail::isfinite(value))
    return write_nonfinite(out, detail::isinf(value), specs, fspecs);

  if (specs.align == align::numeric && fspecs.sign) {
    auto it = reserve(out, 1);
    *it++ = detail::sign<Char>(fspecs.sign);
    out = base_iterator(out, it);
    fspecs.sign = sign::none;
    if (specs.width != 0) --specs.width;
  }

  memory_buffer buffer;
  if (fspecs.format == float_format::hex) {
    if (fspecs.sign) buffer.push_back(detail::sign<char>(fspecs.sign));
    snprintf_float(promote_float(value), specs.precision, fspecs, buffer);
    return write_bytes<align::right>(out, {buffer.data(), buffer.size()},
                                     specs);
  }
  int precision = specs.precision >= 0 || specs.type == presentation_type::none
                      ? specs.precision
                      : 6;
  if (fspecs.format == float_format::exp) {
    if (precision == max_value<int>())
      throw_format_error("number is too big");
    else
      ++precision;
  }
  if (const_check(std::is_same<T, float>())) fspecs.binary32 = true;
  if (!is_fast_float<T>()) fspecs.fallback = true;
  int exp = format_float(promote_float(value), precision, fspecs, buffer);
  fspecs.precision = precision;
  auto fp = big_decimal_fp{buffer.data(), static_cast<int>(buffer.size()), exp};
  return write_float(out, fp, specs, fspecs, loc);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_fast_float<T>::value)>
FMT_CONSTEXPR20 auto write(OutputIt out, T value) -> OutputIt {
  if (is_constant_evaluated()) {
    return write(out, value, basic_format_specs<Char>());
  }

  if (const_check(!is_supported_floating_point(value))) return out;

  using floaty = conditional_t<std::is_same<T, long double>::value, double, T>;
  using uint = typename dragonbox::float_info<floaty>::carrier_uint;
  auto bits = bit_cast<uint>(value);

  auto fspecs = float_specs();
  if (detail::signbit(value)) {
    fspecs.sign = sign::minus;
    value = -value;
  }

  constexpr auto specs = basic_format_specs<Char>();
  uint mask = exponent_mask<floaty>();
  if ((bits & mask) == mask)
    return write_nonfinite(out, std::isinf(value), specs, fspecs);

  auto dec = dragonbox::to_decimal(static_cast<floaty>(value));
  return write_float(out, dec, specs, fspecs, {});
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_floating_point<T>::value &&
                        !is_fast_float<T>::value)>
inline auto write(OutputIt out, T value) -> OutputIt {
  return write(out, value, basic_format_specs<Char>());
}

template <typename Char, typename OutputIt>
auto write(OutputIt out, monostate, basic_format_specs<Char> = {},
           locale_ref = {}) -> OutputIt {
  FMT_ASSERT(false, "");
  return out;
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, basic_string_view<Char> value)
    -> OutputIt {
  auto it = reserve(out, value.size());
  it = copy_str_noinline<Char>(value.begin(), value.end(), it);
  return base_iterator(out, it);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_string<T>::value)>
constexpr auto write(OutputIt out, const T& value) -> OutputIt {
  return write<Char>(out, to_string_view(value));
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        !std::is_same<T, Char>::value)>
FMT_CONSTEXPR auto write(OutputIt out, T value) -> OutputIt {
  auto abs_value = static_cast<uint32_or_64_or_128_t<T>>(value);
  bool negative = is_negative(value);
  // Don't do -abs_value since it trips unsigned-integer-overflow sanitizer.
  if (negative) abs_value = ~abs_value + 1;
  int num_digits = count_digits(abs_value);
  auto size = (negative ? 1 : 0) + static_cast<size_t>(num_digits);
  auto it = reserve(out, size);
  if (auto ptr = to_pointer<Char>(it, size)) {
    if (negative) *ptr++ = static_cast<Char>('-');
    format_decimal<Char>(ptr, abs_value, num_digits);
    return out;
  }
  if (negative) *it++ = static_cast<Char>('-');
  it = format_decimal<Char>(it, abs_value, num_digits).end;
  return base_iterator(out, it);
}

// FMT_ENABLE_IF() condition separated to workaround an MSVC bug.
template <
    typename Char, typename OutputIt, typename T,
    bool check =
        std::is_enum<T>::value && !std::is_same<T, Char>::value &&
        mapped_type_constant<T, basic_format_context<OutputIt, Char>>::value !=
            type::custom_type,
    FMT_ENABLE_IF(check)>
FMT_CONSTEXPR auto write(OutputIt out, T value) -> OutputIt {
  return write<Char>(
      out, static_cast<typename std::underlying_type<T>::type>(value));
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_same<T, bool>::value)>
FMT_CONSTEXPR auto write(OutputIt out, T value,
                         const basic_format_specs<Char>& specs = {},
                         locale_ref = {}) -> OutputIt {
  return specs.type != presentation_type::none &&
                 specs.type != presentation_type::string
             ? write(out, value ? 1 : 0, specs, {})
             : write_bytes(out, value ? "true" : "false", specs);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, Char value) -> OutputIt {
  auto it = reserve(out, 1);
  *it++ = value;
  return base_iterator(out, it);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR_CHAR_TRAITS auto write(OutputIt out, const Char* value)
    -> OutputIt {
  if (!value) {
    throw_format_error("string pointer is null");
  } else {
    out = write(out, basic_string_view<Char>(value));
  }
  return out;
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_same<T, void>::value)>
auto write(OutputIt out, const T* value,
           const basic_format_specs<Char>& specs = {}, locale_ref = {})
    -> OutputIt {
  check_pointer_type_spec(specs.type, error_handler());
  return write_ptr<Char>(out, to_uintptr(value), &specs);
}

// A write overload that handles implicit conversions.
template <typename Char, typename OutputIt, typename T,
          typename Context = basic_format_context<OutputIt, Char>>
FMT_CONSTEXPR auto write(OutputIt out, const T& value) -> enable_if_t<
    std::is_class<T>::value && !is_string<T>::value &&
        !std::is_same<T, Char>::value &&
        !std::is_same<const T&,
                      decltype(arg_mapper<Context>().map(value))>::value,
    OutputIt> {
  return write<Char>(out, arg_mapper<Context>().map(value));
}

template <typename Char, typename OutputIt, typename T,
          typename Context = basic_format_context<OutputIt, Char>>
FMT_CONSTEXPR auto write(OutputIt out, const T& value)
    -> enable_if_t<mapped_type_constant<T, Context>::value == type::custom_type,
                   OutputIt> {
  using formatter_type =
      conditional_t<has_formatter<T, Context>::value,
                    typename Context::template formatter_type<T>,
                    fallback_formatter<T, Char>>;
  auto ctx = Context(out, {}, {});
  return formatter_type().format(value, ctx);
}

// An argument visitor that formats the argument and writes it via the output
// iterator. It's a class and not a generic lambda for compatibility with C++11.
template <typename Char> struct default_arg_formatter {
  using iterator = buffer_appender<Char>;
  using context = buffer_context<Char>;

  iterator out;
  basic_format_args<context> args;
  locale_ref loc;

  template <typename T> auto operator()(T value) -> iterator {
    return write<Char>(out, value);
  }
  auto operator()(typename basic_format_arg<context>::handle h) -> iterator {
    basic_format_parse_context<Char> parse_ctx({});
    context format_ctx(out, args, loc);
    h.format(parse_ctx, format_ctx);
    return format_ctx.out();
  }
};

template <typename Char> struct arg_formatter {
  using iterator = buffer_appender<Char>;
  using context = buffer_context<Char>;

  iterator out;
  const basic_format_specs<Char>& specs;
  locale_ref locale;

  template <typename T>
  FMT_CONSTEXPR FMT_INLINE auto operator()(T value) -> iterator {
    return detail::write(out, value, specs, locale);
  }
  auto operator()(typename basic_format_arg<context>::handle) -> iterator {
    // User-defined types are handled separately because they require access
    // to the parse context.
    return out;
  }
};

template <typename Char> struct custom_formatter {
  basic_format_parse_context<Char>& parse_ctx;
  buffer_context<Char>& ctx;

  void operator()(
      typename basic_format_arg<buffer_context<Char>>::handle h) const {
    h.format(parse_ctx, ctx);
  }
  template <typename T> void operator()(T) const {}
};

template <typename T>
using is_integer =
    bool_constant<is_integral<T>::value && !std::is_same<T, bool>::value &&
                  !std::is_same<T, char>::value &&
                  !std::is_same<T, wchar_t>::value>;

template <typename ErrorHandler> class width_checker {
 public:
  explicit FMT_CONSTEXPR width_checker(ErrorHandler& eh) : handler_(eh) {}

  template <typename T, FMT_ENABLE_IF(is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T value) -> unsigned long long {
    if (is_negative(value)) handler_.on_error("negative width");
    return static_cast<unsigned long long>(value);
  }

  template <typename T, FMT_ENABLE_IF(!is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T) -> unsigned long long {
    handler_.on_error("width is not integer");
    return 0;
  }

 private:
  ErrorHandler& handler_;
};

template <typename ErrorHandler> class precision_checker {
 public:
  explicit FMT_CONSTEXPR precision_checker(ErrorHandler& eh) : handler_(eh) {}

  template <typename T, FMT_ENABLE_IF(is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T value) -> unsigned long long {
    if (is_negative(value)) handler_.on_error("negative precision");
    return static_cast<unsigned long long>(value);
  }

  template <typename T, FMT_ENABLE_IF(!is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T) -> unsigned long long {
    handler_.on_error("precision is not integer");
    return 0;
  }

 private:
  ErrorHandler& handler_;
};

template <template <typename> class Handler, typename FormatArg,
          typename ErrorHandler>
FMT_CONSTEXPR auto get_dynamic_spec(FormatArg arg, ErrorHandler eh) -> int {
  unsigned long long value = visit_format_arg(Handler<ErrorHandler>(eh), arg);
  if (value > to_unsigned(max_value<int>())) eh.on_error("number is too big");
  return static_cast<int>(value);
}

template <typename Context, typename ID>
FMT_CONSTEXPR auto get_arg(Context& ctx, ID id) ->
    typename Context::format_arg {
  auto arg = ctx.arg(id);
  if (!arg) ctx.on_error("argument not found");
  return arg;
}

// The standard format specifier handler with checking.
template <typename Char> class specs_handler : public specs_setter<Char> {
 private:
  basic_format_parse_context<Char>& parse_context_;
  buffer_context<Char>& context_;

  // This is only needed for compatibility with gcc 4.4.
  using format_arg = basic_format_arg<buffer_context<Char>>;

  FMT_CONSTEXPR auto get_arg(auto_id) -> format_arg {
    return detail::get_arg(context_, parse_context_.next_arg_id());
  }

  FMT_CONSTEXPR auto get_arg(int arg_id) -> format_arg {
    parse_context_.check_arg_id(arg_id);
    return detail::get_arg(context_, arg_id);
  }

  FMT_CONSTEXPR auto get_arg(basic_string_view<Char> arg_id) -> format_arg {
    parse_context_.check_arg_id(arg_id);
    return detail::get_arg(context_, arg_id);
  }

 public:
  FMT_CONSTEXPR specs_handler(basic_format_specs<Char>& specs,
                              basic_format_parse_context<Char>& parse_ctx,
                              buffer_context<Char>& ctx)
      : specs_setter<Char>(specs), parse_context_(parse_ctx), context_(ctx) {}

  template <typename Id> FMT_CONSTEXPR void on_dynamic_width(Id arg_id) {
    this->specs_.width = get_dynamic_spec<width_checker>(
        get_arg(arg_id), context_.error_handler());
  }

  template <typename Id> FMT_CONSTEXPR void on_dynamic_precision(Id arg_id) {
    this->specs_.precision = get_dynamic_spec<precision_checker>(
        get_arg(arg_id), context_.error_handler());
  }

  void on_error(const char* message) { context_.on_error(message); }
};

template <template <typename> class Handler, typename Context>
FMT_CONSTEXPR void handle_dynamic_spec(int& value,
                                       arg_ref<typename Context::char_type> ref,
                                       Context& ctx) {
  switch (ref.kind) {
  case arg_id_kind::none:
    break;
  case arg_id_kind::index:
    value = detail::get_dynamic_spec<Handler>(ctx.arg(ref.val.index),
                                              ctx.error_handler());
    break;
  case arg_id_kind::name:
    value = detail::get_dynamic_spec<Handler>(ctx.arg(ref.val.name),
                                              ctx.error_handler());
    break;
  }
}

#define FMT_STRING_IMPL(s, base, explicit)                                 \
  [] {                                                                     \
    /* Use the hidden visibility as a workaround for a GCC bug (#1973). */ \
    /* Use a macro-like name to avoid shadowing warnings. */               \
    struct FMT_GCC_VISIBILITY_HIDDEN FMT_COMPILE_STRING : base {           \
      using char_type = fmt::remove_cvref_t<decltype(s[0])>;               \
      FMT_MAYBE_UNUSED FMT_CONSTEXPR explicit                              \
      operator fmt::basic_string_view<char_type>() const {                 \
        return fmt::detail_exported::compile_string_to_view<char_type>(s); \
      }                                                                    \
    };                                                                     \
    return FMT_COMPILE_STRING();                                           \
  }()

/**
  \rst
  Constructs a compile-time format string from a string literal *s*.

  **Example**::

    // A compile-time error because 'd' is an invalid specifier for strings.
    std::string s = fmt::format(FMT_STRING("{:d}"), "foo");
  \endrst
 */
#define FMT_STRING(s) FMT_STRING_IMPL(s, fmt::compile_string, )

#if FMT_USE_USER_DEFINED_LITERALS
template <typename Char> struct udl_formatter {
  basic_string_view<Char> str;

  template <typename... T>
  auto operator()(T&&... args) const -> std::basic_string<Char> {
    return vformat(str, fmt::make_args_checked<T...>(str, args...));
  }
};

#  if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct statically_named_arg : view {
  static constexpr auto name = Str.data;

  const T& value;
  statically_named_arg(const T& v) : value(v) {}
};

template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct is_named_arg<statically_named_arg<T, Char, N, Str>> : std::true_type {};

template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct is_statically_named_arg<statically_named_arg<T, Char, N, Str>>
    : std::true_type {};

template <typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct udl_arg {
  template <typename T> auto operator=(T&& value) const {
    return statically_named_arg<T, Char, N, Str>(std::forward<T>(value));
  }
};
#  else
template <typename Char> struct udl_arg {
  const Char* str;

  template <typename T> auto operator=(T&& value) const -> named_arg<Char, T> {
    return {str, std::forward<T>(value)};
  }
};
#  endif
#endif  // FMT_USE_USER_DEFINED_LITERALS

template <typename Locale, typename Char>
auto vformat(const Locale& loc, basic_string_view<Char> format_str,
             basic_format_args<buffer_context<type_identity_t<Char>>> args)
    -> std::basic_string<Char> {
  basic_memory_buffer<Char> buffer;
  detail::vformat_to(buffer, format_str, args, detail::locale_ref(loc));
  return {buffer.data(), buffer.size()};
}

using format_func = void (*)(detail::buffer<char>&, int, const char*);

FMT_API void format_error_code(buffer<char>& out, int error_code,
                               string_view message) FMT_NOEXCEPT;

FMT_API void report_error(format_func func, int error_code,
                          const char* message) FMT_NOEXCEPT;
FMT_END_DETAIL_NAMESPACE

FMT_API auto vsystem_error(int error_code, string_view format_str,
                           format_args args) -> std::system_error;

/**
 \rst
 Constructs :class:`std::system_error` with a message formatted with
 ``fmt::format(fmt, args...)``.
  *error_code* is a system error code as given by ``errno``.

 **Example**::

   // This throws std::system_error with the description
   //   cannot open file 'madeup': No such file or directory
   // or similar (system message may vary).
   const char* filename = "madeup";
   std::FILE* file = std::fopen(filename, "r");
   if (!file)
     throw fmt::system_error(errno, "cannot open file '{}'", filename);
 \endrst
*/
template <typename... T>
auto system_error(int error_code, format_string<T...> fmt, T&&... args)
    -> std::system_error {
  return vsystem_error(error_code, fmt, fmt::make_format_args(args...));
}

/**
  \rst
  Formats an error message for an error returned by an operating system or a
  language runtime, for example a file opening error, and writes it to *out*.
  The format is the same as the one used by ``std::system_error(ec, message)``
  where ``ec`` is ``std::error_code(error_code, std::generic_category()})``.
  It is implementation-defined but normally looks like:

  .. parsed-literal::
     *<message>*: *<system-message>*

  where *<message>* is the passed message and *<system-message>* is the system
  message corresponding to the error code.
  *error_code* is a system error code as given by ``errno``.
  \endrst
 */
FMT_API void format_system_error(detail::buffer<char>& out, int error_code,
                                 const char* message) FMT_NOEXCEPT;

// Reports a system error without throwing an exception.
// Can be used to report errors from destructors.
FMT_API void report_system_error(int error_code,
                                 const char* message) FMT_NOEXCEPT;

/** Fast integer formatter. */
class format_int {
 private:
  // Buffer should be large enough to hold all digits (digits10 + 1),
  // a sign and a null character.
  enum { buffer_size = std::numeric_limits<unsigned long long>::digits10 + 3 };
  mutable char buffer_[buffer_size];
  char* str_;

  template <typename UInt> auto format_unsigned(UInt value) -> char* {
    auto n = static_cast<detail::uint32_or_64_or_128_t<UInt>>(value);
    return detail::format_decimal(buffer_, n, buffer_size - 1).begin;
  }

  template <typename Int> auto format_signed(Int value) -> char* {
    auto abs_value = static_cast<detail::uint32_or_64_or_128_t<Int>>(value);
    bool negative = value < 0;
    if (negative) abs_value = 0 - abs_value;
    auto begin = format_unsigned(abs_value);
    if (negative) *--begin = '-';
    return begin;
  }

 public:
  explicit format_int(int value) : str_(format_signed(value)) {}
  explicit format_int(long value) : str_(format_signed(value)) {}
  explicit format_int(long long value) : str_(format_signed(value)) {}
  explicit format_int(unsigned value) : str_(format_unsigned(value)) {}
  explicit format_int(unsigned long value) : str_(format_unsigned(value)) {}
  explicit format_int(unsigned long long value)
      : str_(format_unsigned(value)) {}

  /** Returns the number of characters written to the output buffer. */
  auto size() const -> size_t {
    return detail::to_unsigned(buffer_ - str_ + buffer_size - 1);
  }

  /**
    Returns a pointer to the output buffer content. No terminating null
    character is appended.
   */
  auto data() const -> const char* { return str_; }

  /**
    Returns a pointer to the output buffer content with terminating null
    character appended.
   */
  auto c_str() const -> const char* {
    buffer_[buffer_size - 1] = '\0';
    return str_;
  }

  /**
    \rst
    Returns the content of the output buffer as an ``std::string``.
    \endrst
   */
  auto str() const -> std::string { return std::string(str_, size()); }
};

template <typename T, typename Char>
template <typename FormatContext>
FMT_CONSTEXPR FMT_INLINE auto
formatter<T, Char,
          enable_if_t<detail::type_constant<T, Char>::value !=
                      detail::type::custom_type>>::format(const T& val,
                                                          FormatContext& ctx)
    const -> decltype(ctx.out()) {
  if (specs_.width_ref.kind != detail::arg_id_kind::none ||
      specs_.precision_ref.kind != detail::arg_id_kind::none) {
    auto specs = specs_;
    detail::handle_dynamic_spec<detail::width_checker>(specs.width,
                                                       specs.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs.precision, specs.precision_ref, ctx);
    return detail::write<Char>(ctx.out(), val, specs, ctx.locale());
  }
  return detail::write<Char>(ctx.out(), val, specs_, ctx.locale());
}

#define FMT_FORMAT_AS(Type, Base)                                        \
  template <typename Char>                                               \
  struct formatter<Type, Char> : formatter<Base, Char> {                 \
    template <typename FormatContext>                                    \
    auto format(Type const& val, FormatContext& ctx) const               \
        -> decltype(ctx.out()) {                                         \
      return formatter<Base, Char>::format(static_cast<Base>(val), ctx); \
    }                                                                    \
  }

FMT_FORMAT_AS(signed char, int);
FMT_FORMAT_AS(unsigned char, unsigned);
FMT_FORMAT_AS(short, int);
FMT_FORMAT_AS(unsigned short, unsigned);
FMT_FORMAT_AS(long, long long);
FMT_FORMAT_AS(unsigned long, unsigned long long);
FMT_FORMAT_AS(Char*, const Char*);
FMT_FORMAT_AS(std::basic_string<Char>, basic_string_view<Char>);
FMT_FORMAT_AS(std::nullptr_t, const void*);
FMT_FORMAT_AS(detail::byte, unsigned char);
FMT_FORMAT_AS(detail::std_string_view<Char>, basic_string_view<Char>);

template <typename Char>
struct formatter<void*, Char> : formatter<const void*, Char> {
  template <typename FormatContext>
  auto format(void* val, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<const void*, Char>::format(val, ctx);
  }
};

template <typename Char, size_t N>
struct formatter<Char[N], Char> : formatter<basic_string_view<Char>, Char> {
  template <typename FormatContext>
  FMT_CONSTEXPR auto format(const Char* val, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    return formatter<basic_string_view<Char>, Char>::format(val, ctx);
  }
};

// A formatter for types known only at run time such as variant alternatives.
//
// Usage:
//   using variant = std::variant<int, std::string>;
//   template <>
//   struct formatter<variant>: dynamic_formatter<> {
//     auto format(const variant& v, format_context& ctx) {
//       return visit([&](const auto& val) {
//           return dynamic_formatter<>::format(val, ctx);
//       }, v);
//     }
//   };
template <typename Char = char> class dynamic_formatter {
 private:
  detail::dynamic_format_specs<Char> specs_;
  const Char* format_str_;

  struct null_handler : detail::error_handler {
    void on_align(align_t) {}
    void on_sign(sign_t) {}
    void on_hash() {}
  };

  template <typename Context> void handle_specs(Context& ctx) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
  }

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    format_str_ = ctx.begin();
    // Checks are deferred to formatting time when the argument type is known.
    detail::dynamic_specs_handler<ParseContext> handler(specs_, ctx);
    return detail::parse_format_specs(ctx.begin(), ctx.end(), handler);
  }

  template <typename T, typename FormatContext>
  auto format(const T& val, FormatContext& ctx) -> decltype(ctx.out()) {
    handle_specs(ctx);
    detail::specs_checker<null_handler> checker(
        null_handler(), detail::mapped_type_constant<T, FormatContext>::value);
    checker.on_align(specs_.align);
    if (specs_.sign != sign::none) checker.on_sign(specs_.sign);
    if (specs_.alt) checker.on_hash();
    if (specs_.precision >= 0) checker.end_precision();
    return detail::write<Char>(ctx.out(), val, specs_, ctx.locale());
  }
};

/**
  \rst
  Converts ``p`` to ``const void*`` for pointer formatting.

  **Example**::

    auto s = fmt::format("{}", fmt::ptr(p));
  \endrst
 */
template <typename T> auto ptr(T p) -> const void* {
  static_assert(std::is_pointer<T>::value, "");
  return detail::bit_cast<const void*>(p);
}
template <typename T> auto ptr(const std::unique_ptr<T>& p) -> const void* {
  return p.get();
}
template <typename T> auto ptr(const std::shared_ptr<T>& p) -> const void* {
  return p.get();
}

class bytes {
 private:
  string_view data_;
  friend struct formatter<bytes>;

 public:
  explicit bytes(string_view data) : data_(data) {}
};

template <> struct formatter<bytes> {
 private:
  detail::dynamic_format_specs<char> specs_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    detail::specs_checker<handler_type> handler(handler_type(specs_, ctx),
                                                detail::type::string_type);
    auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
    detail::check_string_type_spec(specs_.type, ctx.error_handler());
    return it;
  }

  template <typename FormatContext>
  auto format(bytes b, FormatContext& ctx) -> decltype(ctx.out()) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
    return detail::write_bytes(ctx.out(), b.data_, specs_);
  }
};

// group_digits_view is not derived from view because it copies the argument.
template <typename T> struct group_digits_view { T value; };

/**
  \rst
  Returns a view that formats an integer value using ',' as a locale-independent
  thousands separator.

  **Example**::

    fmt::print("{}", fmt::group_digits(12345));
    // Output: "12,345"
  \endrst
 */
template <typename T> auto group_digits(T value) -> group_digits_view<T> {
  return {value};
}

template <typename T> struct formatter<group_digits_view<T>> : formatter<T> {
 private:
  detail::dynamic_format_specs<char> specs_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    detail::specs_checker<handler_type> handler(handler_type(specs_, ctx),
                                                detail::type::int_type);
    auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
    detail::check_string_type_spec(specs_.type, ctx.error_handler());
    return it;
  }

  template <typename FormatContext>
  auto format(group_digits_view<T> t, FormatContext& ctx)
      -> decltype(ctx.out()) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
    return detail::write_int_localized(
        ctx.out(), static_cast<detail::uint64_or_128_t<T>>(t.value), 0, specs_,
        detail::digit_grouping<char>({"\3", ','}));
  }
};

template <typename It, typename Sentinel, typename Char = char>
struct join_view : detail::view {
  It begin;
  Sentinel end;
  basic_string_view<Char> sep;

  join_view(It b, Sentinel e, basic_string_view<Char> s)
      : begin(b), end(e), sep(s) {}
};

template <typename It, typename Sentinel, typename Char>
using arg_join FMT_DEPRECATED_ALIAS = join_view<It, Sentinel, Char>;

template <typename It, typename Sentinel, typename Char>
struct formatter<join_view<It, Sentinel, Char>, Char> {
 private:
  using value_type =
#ifdef __cpp_lib_ranges
      std::iter_value_t<It>;
#else
      typename std::iterator_traits<It>::value_type;
#endif
  using context = buffer_context<Char>;
  using mapper = detail::arg_mapper<context>;

  template <typename T, FMT_ENABLE_IF(has_formatter<T, context>::value)>
  static auto map(const T& value) -> const T& {
    return value;
  }
  template <typename T, FMT_ENABLE_IF(!has_formatter<T, context>::value)>
  static auto map(const T& value) -> decltype(mapper().map(value)) {
    return mapper().map(value);
  }

  using formatter_type =
      conditional_t<is_formattable<value_type, Char>::value,
                    formatter<remove_cvref_t<decltype(map(
                                  std::declval<const value_type&>()))>,
                              Char>,
                    detail::fallback_formatter<value_type, Char>>;

  formatter_type value_formatter_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return value_formatter_.parse(ctx);
  }

  template <typename FormatContext>
  auto format(const join_view<It, Sentinel, Char>& value, FormatContext& ctx)
      -> decltype(ctx.out()) {
    auto it = value.begin;
    auto out = ctx.out();
    if (it != value.end) {
      out = value_formatter_.format(map(*it), ctx);
      ++it;
      while (it != value.end) {
        out = detail::copy_str<Char>(value.sep.begin(), value.sep.end(), out);
        ctx.advance_to(out);
        out = value_formatter_.format(map(*it), ctx);
        ++it;
      }
    }
    return out;
  }
};

/**
  Returns a view that formats the iterator range `[begin, end)` with elements
  separated by `sep`.
 */
template <typename It, typename Sentinel>
auto join(It begin, Sentinel end, string_view sep) -> join_view<It, Sentinel> {
  return {begin, end, sep};
}

/**
  \rst
  Returns a view that formats `range` with elements separated by `sep`.

  **Example**::

    std::vector<int> v = {1, 2, 3};
    fmt::print("{}", fmt::join(v, ", "));
    // Output: "1, 2, 3"

  ``fmt::join`` applies passed format specifiers to the range elements::

    fmt::print("{:02}", fmt::join(v, ", "));
    // Output: "01, 02, 03"
  \endrst
 */
template <typename Range>
auto join(Range&& range, string_view sep)
    -> join_view<detail::iterator_t<Range>, detail::sentinel_t<Range>> {
  return join(std::begin(range), std::end(range), sep);
}

/**
  \rst
  Converts *value* to ``std::string`` using the default format for type *T*.

  **Example**::

    #include <fmt/format.h>

    std::string answer = fmt::to_string(42);
  \endrst
 */
template <typename T, FMT_ENABLE_IF(!std::is_integral<T>::value)>
inline auto to_string(const T& value) -> std::string {
  auto result = std::string();
  detail::write<char>(std::back_inserter(result), value);
  return result;
}

template <typename T, FMT_ENABLE_IF(std::is_integral<T>::value)>
FMT_NODISCARD inline auto to_string(T value) -> std::string {
  // The buffer should be large enough to store the number including the sign
  // or "false" for bool.
  constexpr int max_size = detail::digits10<T>() + 2;
  char buffer[max_size > 5 ? static_cast<unsigned>(max_size) : 5];
  char* begin = buffer;
  return std::string(begin, detail::write<char>(begin, value));
}

template <typename Char, size_t SIZE>
FMT_NODISCARD auto to_string(const basic_memory_buffer<Char, SIZE>& buf)
    -> std::basic_string<Char> {
  auto size = buf.size();
  detail::assume(size < std::basic_string<Char>().max_size());
  return std::basic_string<Char>(buf.data(), size);
}

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Char>
void vformat_to(
    buffer<Char>& buf, basic_string_view<Char> fmt,
    basic_format_args<FMT_BUFFER_CONTEXT(type_identity_t<Char>)> args,
    locale_ref loc) {
  // workaround for msvc bug regarding name-lookup in module
  // link names into function scope
  using detail::arg_formatter;
  using detail::buffer_appender;
  using detail::custom_formatter;
  using detail::default_arg_formatter;
  using detail::get_arg;
  using detail::locale_ref;
  using detail::parse_format_specs;
  using detail::specs_checker;
  using detail::specs_handler;
  using detail::to_unsigned;
  using detail::type;
  using detail::write;
  auto out = buffer_appender<Char>(buf);
  if (fmt.size() == 2 && equal2(fmt.data(), "{}")) {
    auto arg = args.get(0);
    if (!arg) error_handler().on_error("argument not found");
    visit_format_arg(default_arg_formatter<Char>{out, args, loc}, arg);
    return;
  }

  struct format_handler : error_handler {
    basic_format_parse_context<Char> parse_context;
    buffer_context<Char> context;

    format_handler(buffer_appender<Char> out, basic_string_view<Char> str,
                   basic_format_args<buffer_context<Char>> args, locale_ref loc)
        : parse_context(str), context(out, args, loc) {}

    void on_text(const Char* begin, const Char* end) {
      auto text = basic_string_view<Char>(begin, to_unsigned(end - begin));
      context.advance_to(write<Char>(context.out(), text));
    }

    FMT_CONSTEXPR auto on_arg_id() -> int {
      return parse_context.next_arg_id();
    }
    FMT_CONSTEXPR auto on_arg_id(int id) -> int {
      return parse_context.check_arg_id(id), id;
    }
    FMT_CONSTEXPR auto on_arg_id(basic_string_view<Char> id) -> int {
      int arg_id = context.arg_id(id);
      if (arg_id < 0) on_error("argument not found");
      return arg_id;
    }

    FMT_INLINE void on_replacement_field(int id, const Char*) {
      auto arg = get_arg(context, id);
      context.advance_to(visit_format_arg(
          default_arg_formatter<Char>{context.out(), context.args(),
                                      context.locale()},
          arg));
    }

    auto on_format_specs(int id, const Char* begin, const Char* end)
        -> const Char* {
      auto arg = get_arg(context, id);
      if (arg.type() == type::custom_type) {
        parse_context.advance_to(parse_context.begin() +
                                 (begin - &*parse_context.begin()));
        visit_format_arg(custom_formatter<Char>{parse_context, context}, arg);
        return parse_context.begin();
      }
      auto specs = basic_format_specs<Char>();
      specs_checker<specs_handler<Char>> handler(
          specs_handler<Char>(specs, parse_context, context), arg.type());
      begin = parse_format_specs(begin, end, handler);
      if (begin == end || *begin != '}')
        on_error("missing '}' in format string");
      auto f = arg_formatter<Char>{context.out(), specs, context.locale()};
      context.advance_to(visit_format_arg(f, arg));
      return begin;
    }
  };
  detail::parse_format_string<false>(fmt, format_handler(out, fmt, args, loc));
}

#ifndef FMT_HEADER_ONLY
extern template FMT_API auto thousands_sep_impl<char>(locale_ref)
    -> thousands_sep_result<char>;
extern template FMT_API auto thousands_sep_impl<wchar_t>(locale_ref)
    -> thousands_sep_result<wchar_t>;
extern template FMT_API auto decimal_point_impl(locale_ref) -> char;
extern template FMT_API auto decimal_point_impl(locale_ref) -> wchar_t;
extern template auto format_float<double>(double value, int precision,
                                          float_specs specs, buffer<char>& buf)
    -> int;
extern template auto format_float<long double>(long double value, int precision,
                                               float_specs specs,
                                               buffer<char>& buf) -> int;
void snprintf_float(float, int, float_specs, buffer<char>&) = delete;
extern template auto snprintf_float<double>(double value, int precision,
                                            float_specs specs,
                                            buffer<char>& buf) -> int;
extern template auto snprintf_float<long double>(long double value,
                                                 int precision,
                                                 float_specs specs,
                                                 buffer<char>& buf) -> int;
#endif  // FMT_HEADER_ONLY

FMT_END_DETAIL_NAMESPACE

#if FMT_USE_USER_DEFINED_LITERALS
inline namespace literals {
/**
  \rst
  User-defined literal equivalent of :func:`fmt::arg`.

  **Example**::

    using namespace fmt::literals;
    fmt::print("Elapsed time: {s:.2f} seconds", "s"_a=1.23);
  \endrst
 */
#  if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <detail_exported::fixed_string Str>
constexpr auto operator""_a()
    -> detail::udl_arg<remove_cvref_t<decltype(Str.data[0])>,
                       sizeof(Str.data) / sizeof(decltype(Str.data[0])), Str> {
  return {};
}
#  else
constexpr auto operator"" _a(const char* s, size_t) -> detail::udl_arg<char> {
  return {s};
}
#  endif

// DEPRECATED!
// User-defined literal equivalent of fmt::format.
FMT_DEPRECATED constexpr auto operator"" _format(const char* s, size_t n)
    -> detail::udl_formatter<char> {
  return {{s, n}};
}
}  // namespace literals
#endif  // FMT_USE_USER_DEFINED_LITERALS

template <typename Locale, FMT_ENABLE_IF(detail::is_locale<Locale>::value)>
inline auto vformat(const Locale& loc, string_view fmt, format_args args)
    -> std::string {
  return detail::vformat(loc, fmt, args);
}

template <typename Locale, typename... T,
          FMT_ENABLE_IF(detail::is_locale<Locale>::value)>
inline auto format(const Locale& loc, format_string<T...> fmt, T&&... args)
    -> std::string {
  return vformat(loc, string_view(fmt), fmt::make_format_args(args...));
}

template <typename... T, size_t SIZE, typename Allocator>
FMT_DEPRECATED auto format_to(basic_memory_buffer<char, SIZE, Allocator>& buf,
                              format_string<T...> fmt, T&&... args)
    -> appender {
  detail::vformat_to(buf, string_view(fmt), fmt::make_format_args(args...));
  return appender(buf);
}

template <typename OutputIt, typename Locale,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value&&
                            detail::is_locale<Locale>::value)>
auto vformat_to(OutputIt out, const Locale& loc, string_view fmt,
                format_args args) -> OutputIt {
  using detail::get_buffer;
  auto&& buf = get_buffer<char>(out);
  detail::vformat_to(buf, fmt, args, detail::locale_ref(loc));
  return detail::get_iterator(buf);
}

template <typename OutputIt, typename Locale, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value&&
                            detail::is_locale<Locale>::value)>
FMT_INLINE auto format_to(OutputIt out, const Locale& loc,
                          format_string<T...> fmt, T&&... args) -> OutputIt {
  return vformat_to(out, loc, fmt, fmt::make_format_args(args...));
}

FMT_MODULE_EXPORT_END
FMT_END_NAMESPACE

#ifdef FMT_DEPRECATED_INCLUDE_XCHAR
#  include "xchar.h"
#endif

#ifdef FMT_HEADER_ONLY
#  define FMT_FUNC inline
#  include "format-inl.h"
#else
#  define FMT_FUNC
#endif

#endif  // FMT_FORMAT_H_
// Formatting library for C++ - std::ostream support
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_OSTREAM_H_
#define FMT_OSTREAM_H_

#include <ostream>


FMT_BEGIN_NAMESPACE

template <typename OutputIt, typename Char> class basic_printf_context;

namespace detail {

// Checks if T has a user-defined operator<<.
template <typename T, typename Char, typename Enable = void>
class is_streamable {
 private:
  template <typename U>
  static auto test(int)
      -> bool_constant<sizeof(std::declval<std::basic_ostream<Char>&>()
                              << std::declval<U>()) != 0>;

  template <typename> static auto test(...) -> std::false_type;

  using result = decltype(test<T>(0));

 public:
  is_streamable() = default;

  static const bool value = result::value;
};

// Formatting of built-in types and arrays is intentionally disabled because
// it's handled by standard (non-ostream) formatters.
template <typename T, typename Char>
struct is_streamable<
    T, Char,
    enable_if_t<
        std::is_arithmetic<T>::value || std::is_array<T>::value ||
        std::is_pointer<T>::value || std::is_same<T, char8_type>::value ||
        std::is_same<T, std::basic_string<Char>>::value ||
        std::is_same<T, std_string_view<Char>>::value ||
        (std::is_convertible<T, int>::value && !std::is_enum<T>::value)>>
    : std::false_type {};

// Write the content of buf to os.
// It is a separate function rather than a part of vprint to simplify testing.
template <typename Char>
void write_buffer(std::basic_ostream<Char>& os, buffer<Char>& buf) {
  const Char* buf_data = buf.data();
  using unsigned_streamsize = std::make_unsigned<std::streamsize>::type;
  unsigned_streamsize size = buf.size();
  unsigned_streamsize max_size = to_unsigned(max_value<std::streamsize>());
  do {
    unsigned_streamsize n = size <= max_size ? size : max_size;
    os.write(buf_data, static_cast<std::streamsize>(n));
    buf_data += n;
    size -= n;
  } while (size != 0);
}

template <typename Char, typename T>
void format_value(buffer<Char>& buf, const T& value,
                  locale_ref loc = locale_ref()) {
  auto&& format_buf = formatbuf<std::basic_streambuf<Char>>(buf);
  auto&& output = std::basic_ostream<Char>(&format_buf);
#if !defined(FMT_STATIC_THOUSANDS_SEPARATOR)
  if (loc) output.imbue(loc.get<std::locale>());
#endif
  output << value;
  output.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  buf.try_resize(buf.size());
}

// Formats an object of type T that has an overloaded ostream operator<<.
template <typename T, typename Char>
struct fallback_formatter<T, Char, enable_if_t<is_streamable<T, Char>::value>>
    : private formatter<basic_string_view<Char>, Char> {
  using formatter<basic_string_view<Char>, Char>::parse;

  template <typename OutputIt>
  auto format(const T& value, basic_format_context<OutputIt, Char>& ctx)
      -> OutputIt {
    auto buffer = basic_memory_buffer<Char>();
    format_value(buffer, value, ctx.locale());
    return formatter<basic_string_view<Char>, Char>::format(
        {buffer.data(), buffer.size()}, ctx);
  }

  // DEPRECATED!
  template <typename OutputIt>
  auto format(const T& value, basic_printf_context<OutputIt, Char>& ctx)
      -> OutputIt {
    auto buffer = basic_memory_buffer<Char>();
    format_value(buffer, value, ctx.locale());
    return std::copy(buffer.begin(), buffer.end(), ctx.out());
  }
};
}  // namespace detail

FMT_MODULE_EXPORT
template <typename Char>
void vprint(std::basic_ostream<Char>& os, basic_string_view<Char> format_str,
            basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  auto buffer = basic_memory_buffer<Char>();
  detail::vformat_to(buffer, format_str, args);
  detail::write_buffer(os, buffer);
}

/**
  \rst
  Prints formatted data to the stream *os*.

  **Example**::

    fmt::print(cerr, "Don't {}!", "panic");
  \endrst
 */
FMT_MODULE_EXPORT
template <typename S, typename... Args,
          typename Char = enable_if_t<detail::is_string<S>::value, char_t<S>>>
void print(std::basic_ostream<Char>& os, const S& format_str, Args&&... args) {
  vprint(os, to_string_view(format_str),
         fmt::make_args_checked<Args...>(format_str, args...));
}
FMT_END_NAMESPACE

#endif  // FMT_OSTREAM_H_
// Formatting library for C++ - the core API for char/UTF-8
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_CORE_H_
#define FMT_CORE_H_

#include <cstddef>  // std::byte
#include <cstdio>   // std::FILE
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

// The fmt library version in the form major * 10000 + minor * 100 + patch.
#define FMT_VERSION 80101

#if defined(__clang__) && !defined(__ibmxl__)
#  define FMT_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#else
#  define FMT_CLANG_VERSION 0
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__NVCOMPILER)
#  define FMT_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#  define FMT_GCC_VERSION 0
#endif

#ifndef FMT_GCC_PRAGMA
// Workaround _Pragma bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59884.
#  if FMT_GCC_VERSION >= 504
#    define FMT_GCC_PRAGMA(arg) _Pragma(arg)
#  else
#    define FMT_GCC_PRAGMA(arg)
#  endif
#endif

#ifdef __ICL
#  define FMT_ICC_VERSION __ICL
#elif defined(__INTEL_COMPILER)
#  define FMT_ICC_VERSION __INTEL_COMPILER
#else
#  define FMT_ICC_VERSION 0
#endif

#ifdef __NVCC__
#  define FMT_NVCC __NVCC__
#else
#  define FMT_NVCC 0
#endif

#ifdef _MSC_VER
#  define FMT_MSC_VER _MSC_VER
#  define FMT_MSC_WARNING(...) __pragma(warning(__VA_ARGS__))
#else
#  define FMT_MSC_VER 0
#  define FMT_MSC_WARNING(...)
#endif

#ifdef __has_feature
#  define FMT_HAS_FEATURE(x) __has_feature(x)
#else
#  define FMT_HAS_FEATURE(x) 0
#endif

#if defined(__has_include) &&                             \
    (!defined(__INTELLISENSE__) || FMT_MSC_VER > 1900) && \
    (!FMT_ICC_VERSION || FMT_ICC_VERSION >= 1600)
#  define FMT_HAS_INCLUDE(x) __has_include(x)
#else
#  define FMT_HAS_INCLUDE(x) 0
#endif

#ifdef __has_cpp_attribute
#  define FMT_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#  define FMT_HAS_CPP_ATTRIBUTE(x) 0
#endif

#ifdef _MSVC_LANG
#  define FMT_CPLUSPLUS _MSVC_LANG
#else
#  define FMT_CPLUSPLUS __cplusplus
#endif

#define FMT_HAS_CPP14_ATTRIBUTE(attribute) \
  (FMT_CPLUSPLUS >= 201402L && FMT_HAS_CPP_ATTRIBUTE(attribute))

#define FMT_HAS_CPP17_ATTRIBUTE(attribute) \
  (FMT_CPLUSPLUS >= 201703L && FMT_HAS_CPP_ATTRIBUTE(attribute))

// Check if relaxed C++14 constexpr is supported.
// GCC doesn't allow throw in constexpr until version 6 (bug 67371).
#ifndef FMT_USE_CONSTEXPR
#  define FMT_USE_CONSTEXPR                                           \
    (FMT_HAS_FEATURE(cxx_relaxed_constexpr) || FMT_MSC_VER >= 1912 || \
     (FMT_GCC_VERSION >= 600 && __cplusplus >= 201402L)) &&           \
        !FMT_NVCC && !FMT_ICC_VERSION
#endif
#if FMT_USE_CONSTEXPR
#  define FMT_CONSTEXPR constexpr
#  define FMT_CONSTEXPR_DECL constexpr
#else
#  define FMT_CONSTEXPR
#  define FMT_CONSTEXPR_DECL
#endif

#if ((__cplusplus >= 202002L) &&                              \
     (!defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE > 9)) || \
    (__cplusplus >= 201709L && FMT_GCC_VERSION >= 1002)
#  define FMT_CONSTEXPR20 constexpr
#else
#  define FMT_CONSTEXPR20
#endif

// Check if constexpr std::char_traits<>::compare,length is supported.
#if defined(__GLIBCXX__)
#  if __cplusplus >= 201703L && defined(_GLIBCXX_RELEASE) && \
      _GLIBCXX_RELEASE >= 7  // GCC 7+ libstdc++ has _GLIBCXX_RELEASE.
#    define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#  endif
#elif defined(_LIBCPP_VERSION) && __cplusplus >= 201703L && \
    _LIBCPP_VERSION >= 4000
#  define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#elif FMT_MSC_VER >= 1914 && _MSVC_LANG >= 201703L
#  define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#endif
#ifndef FMT_CONSTEXPR_CHAR_TRAITS
#  define FMT_CONSTEXPR_CHAR_TRAITS
#endif

// Check if exceptions are disabled.
#ifndef FMT_EXCEPTIONS
#  if (defined(__GNUC__) && !defined(__EXCEPTIONS)) || \
      FMT_MSC_VER && !_HAS_EXCEPTIONS
#    define FMT_EXCEPTIONS 0
#  else
#    define FMT_EXCEPTIONS 1
#  endif
#endif

// Define FMT_USE_NOEXCEPT to make fmt use noexcept (C++11 feature).
#ifndef FMT_USE_NOEXCEPT
#  define FMT_USE_NOEXCEPT 0
#endif

#if FMT_USE_NOEXCEPT || FMT_HAS_FEATURE(cxx_noexcept) || \
    FMT_GCC_VERSION >= 408 || FMT_MSC_VER >= 1900
#  define FMT_DETECTED_NOEXCEPT noexcept
#  define FMT_HAS_CXX11_NOEXCEPT 1
#else
#  define FMT_DETECTED_NOEXCEPT throw()
#  define FMT_HAS_CXX11_NOEXCEPT 0
#endif

#ifndef FMT_NOEXCEPT
#  if FMT_EXCEPTIONS || FMT_HAS_CXX11_NOEXCEPT
#    define FMT_NOEXCEPT FMT_DETECTED_NOEXCEPT
#  else
#    define FMT_NOEXCEPT
#  endif
#endif

// [[noreturn]] is disabled on MSVC and NVCC because of bogus unreachable code
// warnings.
#if FMT_EXCEPTIONS && FMT_HAS_CPP_ATTRIBUTE(noreturn) && !FMT_MSC_VER && \
    !FMT_NVCC
#  define FMT_NORETURN [[noreturn]]
#else
#  define FMT_NORETURN
#endif

#if __cplusplus == 201103L || __cplusplus == 201402L
#  if defined(__INTEL_COMPILER) || defined(__PGI)
#    define FMT_FALLTHROUGH
#  elif defined(__clang__)
#    define FMT_FALLTHROUGH [[clang::fallthrough]]
#  elif FMT_GCC_VERSION >= 700 && \
      (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 520)
#    define FMT_FALLTHROUGH [[gnu::fallthrough]]
#  else
#    define FMT_FALLTHROUGH
#  endif
#elif FMT_HAS_CPP17_ATTRIBUTE(fallthrough)
#  define FMT_FALLTHROUGH [[fallthrough]]
#else
#  define FMT_FALLTHROUGH
#endif

#ifndef FMT_NODISCARD
#  if FMT_HAS_CPP17_ATTRIBUTE(nodiscard)
#    define FMT_NODISCARD [[nodiscard]]
#  else
#    define FMT_NODISCARD
#  endif
#endif

#ifndef FMT_USE_FLOAT
#  define FMT_USE_FLOAT 1
#endif
#ifndef FMT_USE_DOUBLE
#  define FMT_USE_DOUBLE 1
#endif
#ifndef FMT_USE_LONG_DOUBLE
#  define FMT_USE_LONG_DOUBLE 1
#endif

#ifndef FMT_INLINE
#  if FMT_GCC_VERSION || FMT_CLANG_VERSION
#    define FMT_INLINE inline __attribute__((always_inline))
#  else
#    define FMT_INLINE inline
#  endif
#endif

#ifndef FMT_DEPRECATED
#  if FMT_HAS_CPP14_ATTRIBUTE(deprecated) || FMT_MSC_VER >= 1900
#    define FMT_DEPRECATED [[deprecated]]
#  else
#    if (defined(__GNUC__) && !defined(__LCC__)) || defined(__clang__)
#      define FMT_DEPRECATED __attribute__((deprecated))
#    elif FMT_MSC_VER
#      define FMT_DEPRECATED __declspec(deprecated)
#    else
#      define FMT_DEPRECATED /* deprecated */
#    endif
#  endif
#endif

#ifndef FMT_BEGIN_NAMESPACE
#  define FMT_BEGIN_NAMESPACE \
    namespace fmt {           \
    inline namespace v8 {
#  define FMT_END_NAMESPACE \
    }                       \
    }
#endif

#ifndef FMT_MODULE_EXPORT
#  define FMT_MODULE_EXPORT
#  define FMT_MODULE_EXPORT_BEGIN
#  define FMT_MODULE_EXPORT_END
#  define FMT_BEGIN_DETAIL_NAMESPACE namespace detail {
#  define FMT_END_DETAIL_NAMESPACE }
#endif

#if !defined(FMT_HEADER_ONLY) && defined(_WIN32)
#  define FMT_CLASS_API FMT_MSC_WARNING(suppress : 4275)
#  ifdef FMT_EXPORT
#    define FMT_API __declspec(dllexport)
#  elif defined(FMT_SHARED)
#    define FMT_API __declspec(dllimport)
#  endif
#else
#  define FMT_CLASS_API
#  if defined(FMT_EXPORT) || defined(FMT_SHARED)
#    if defined(__GNUC__) || defined(__clang__)
#      define FMT_API __attribute__((visibility("default")))
#    endif
#  endif
#endif
#ifndef FMT_API
#  define FMT_API
#endif

// libc++ supports string_view in pre-c++17.
#if (FMT_HAS_INCLUDE(<string_view>) &&                       \
     (__cplusplus > 201402L || defined(_LIBCPP_VERSION))) || \
    (defined(_MSVC_LANG) && _MSVC_LANG > 201402L && _MSC_VER >= 1910)
#  include <string_view>
#  define FMT_USE_STRING_VIEW
#elif FMT_HAS_INCLUDE("experimental/string_view") && __cplusplus >= 201402L
#  include <experimental/string_view>
#  define FMT_USE_EXPERIMENTAL_STRING_VIEW
#endif

#ifndef FMT_UNICODE
#  define FMT_UNICODE !FMT_MSC_VER
#endif

#ifndef FMT_CONSTEVAL
#  if ((FMT_GCC_VERSION >= 1000 || FMT_CLANG_VERSION >= 1101) &&      \
       __cplusplus > 201703L && !defined(__apple_build_version__)) || \
      (defined(__cpp_consteval) &&                                    \
       (!FMT_MSC_VER || _MSC_FULL_VER >= 193030704))
// consteval is broken in MSVC before VS2022 and Apple clang 13.
#    define FMT_CONSTEVAL consteval
#    define FMT_HAS_CONSTEVAL
#  else
#    define FMT_CONSTEVAL
#  endif
#endif

#ifndef FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
#  if defined(__cpp_nontype_template_args) &&                \
      ((FMT_GCC_VERSION >= 903 && __cplusplus >= 201709L) || \
       __cpp_nontype_template_args >= 201911L)
#    define FMT_USE_NONTYPE_TEMPLATE_PARAMETERS 1
#  else
#    define FMT_USE_NONTYPE_TEMPLATE_PARAMETERS 0
#  endif
#endif

// Enable minimal optimizations for more compact code in debug mode.
FMT_GCC_PRAGMA("GCC push_options")
#ifndef __OPTIMIZE__
FMT_GCC_PRAGMA("GCC optimize(\"Og\")")
#endif

FMT_BEGIN_NAMESPACE
FMT_MODULE_EXPORT_BEGIN

// Implementations of enable_if_t and other metafunctions for older systems.
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;
template <bool B> using bool_constant = std::integral_constant<bool, B>;
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <typename T>
using remove_const_t = typename std::remove_const<T>::type;
template <typename T>
using remove_cvref_t = typename std::remove_cv<remove_reference_t<T>>::type;
template <typename T> struct type_identity { using type = T; };
template <typename T> using type_identity_t = typename type_identity<T>::type;

struct monostate {
  constexpr monostate() {}
};

// An enable_if helper to be used in template parameters which results in much
// shorter symbols: https://godbolt.org/z/sWw4vP. Extra parentheses are needed
// to workaround a bug in MSVC 2019 (see #1140 and #1186).
#ifdef FMT_DOC
#  define FMT_ENABLE_IF(...)
#else
#  define FMT_ENABLE_IF(...) enable_if_t<(__VA_ARGS__), int> = 0
#endif

FMT_BEGIN_DETAIL_NAMESPACE

// Suppress "unused variable" warnings with the method described in
// https://herbsutter.com/2009/10/18/mailbag-shutting-up-compiler-warnings/.
// (void)var does not work on many Intel compilers.
template <typename... T> FMT_CONSTEXPR void ignore_unused(const T&...) {}

constexpr FMT_INLINE auto is_constant_evaluated(bool default_value = false)
    FMT_NOEXCEPT -> bool {
#ifdef __cpp_lib_is_constant_evaluated
  ignore_unused(default_value);
  return std::is_constant_evaluated();
#else
  return default_value;
#endif
}

// A function to suppress "conditional expression is constant" warnings.
template <typename T> constexpr FMT_INLINE auto const_check(T value) -> T {
  return value;
}

FMT_NORETURN FMT_API void assert_fail(const char* file, int line,
                                      const char* message);

#ifndef FMT_ASSERT
#  ifdef NDEBUG
// FMT_ASSERT is not empty to avoid -Werror=empty-body.
#    define FMT_ASSERT(condition, message) \
      ::fmt::detail::ignore_unused((condition), (message))
#  else
#    define FMT_ASSERT(condition, message)                                    \
      ((condition) /* void() fails with -Winvalid-constexpr on clang 4.0.1 */ \
           ? (void)0                                                          \
           : ::fmt::detail::assert_fail(__FILE__, __LINE__, (message)))
#  endif
#endif

#ifdef __cpp_lib_byte
using byte = std::byte;
#else
enum class byte : unsigned char {};
#endif

#if defined(FMT_USE_STRING_VIEW)
template <typename Char> using std_string_view = std::basic_string_view<Char>;
#elif defined(FMT_USE_EXPERIMENTAL_STRING_VIEW)
template <typename Char>
using std_string_view = std::experimental::basic_string_view<Char>;
#else
template <typename T> struct std_string_view {};
#endif

#ifdef FMT_USE_INT128
// Do nothing.
#elif defined(__SIZEOF_INT128__) && !FMT_NVCC && \
    !(FMT_CLANG_VERSION && FMT_MSC_VER)
#  define FMT_USE_INT128 1
using int128_t = __int128_t;
using uint128_t = __uint128_t;
template <typename T> inline auto convert_for_visit(T value) -> T {
  return value;
}
#else
#  define FMT_USE_INT128 0
#endif
#if !FMT_USE_INT128
enum class int128_t {};
enum class uint128_t {};
// Reduce template instantiations.
template <typename T> inline auto convert_for_visit(T) -> monostate {
  return {};
}
#endif

// Casts a nonnegative integer to unsigned.
template <typename Int>
FMT_CONSTEXPR auto to_unsigned(Int value) ->
    typename std::make_unsigned<Int>::type {
  FMT_ASSERT(value >= 0, "negative value");
  return static_cast<typename std::make_unsigned<Int>::type>(value);
}

FMT_MSC_WARNING(suppress : 4566) constexpr unsigned char micro[] = "\u00B5";

constexpr auto is_utf8() -> bool {
  // Avoid buggy sign extensions in MSVC's constant evaluation mode.
  // https://developercommunity.visualstudio.com/t/C-difference-in-behavior-for-unsigned/1233612
  using uchar = unsigned char;
  return FMT_UNICODE || (sizeof(micro) == 3 && uchar(micro[0]) == 0xC2 &&
                         uchar(micro[1]) == 0xB5);
}
FMT_END_DETAIL_NAMESPACE

/**
  An implementation of ``std::basic_string_view`` for pre-C++17. It provides a
  subset of the API. ``fmt::basic_string_view`` is used for format strings even
  if ``std::string_view`` is available to prevent issues when a library is
  compiled with a different ``-std`` option than the client code (which is not
  recommended).
 */
template <typename Char> class basic_string_view {
 private:
  const Char* data_;
  size_t size_;

 public:
  using value_type = Char;
  using iterator = const Char*;

  constexpr basic_string_view() FMT_NOEXCEPT : data_(nullptr), size_(0) {}

  /** Constructs a string reference object from a C string and a size. */
  constexpr basic_string_view(const Char* s, size_t count) FMT_NOEXCEPT
      : data_(s),
        size_(count) {}

  /**
    \rst
    Constructs a string reference object from a C string computing
    the size with ``std::char_traits<Char>::length``.
    \endrst
   */
  FMT_CONSTEXPR_CHAR_TRAITS
  FMT_INLINE
  basic_string_view(const Char* s)
      : data_(s),
        size_(detail::const_check(std::is_same<Char, char>::value &&
                                  !detail::is_constant_evaluated(true))
                  ? std::strlen(reinterpret_cast<const char*>(s))
                  : std::char_traits<Char>::length(s)) {}

  /** Constructs a string reference from a ``std::basic_string`` object. */
  template <typename Traits, typename Alloc>
  FMT_CONSTEXPR basic_string_view(
      const std::basic_string<Char, Traits, Alloc>& s) FMT_NOEXCEPT
      : data_(s.data()),
        size_(s.size()) {}

  template <typename S, FMT_ENABLE_IF(std::is_same<
                                      S, detail::std_string_view<Char>>::value)>
  FMT_CONSTEXPR basic_string_view(S s) FMT_NOEXCEPT : data_(s.data()),
                                                      size_(s.size()) {}

  /** Returns a pointer to the string data. */
  constexpr auto data() const FMT_NOEXCEPT -> const Char* { return data_; }

  /** Returns the string size. */
  constexpr auto size() const FMT_NOEXCEPT -> size_t { return size_; }

  constexpr auto begin() const FMT_NOEXCEPT -> iterator { return data_; }
  constexpr auto end() const FMT_NOEXCEPT -> iterator { return data_ + size_; }

  constexpr auto operator[](size_t pos) const FMT_NOEXCEPT -> const Char& {
    return data_[pos];
  }

  FMT_CONSTEXPR void remove_prefix(size_t n) FMT_NOEXCEPT {
    data_ += n;
    size_ -= n;
  }

  // Lexicographically compare this string reference to other.
  FMT_CONSTEXPR_CHAR_TRAITS auto compare(basic_string_view other) const -> int {
    size_t str_size = size_ < other.size_ ? size_ : other.size_;
    int result = std::char_traits<Char>::compare(data_, other.data_, str_size);
    if (result == 0)
      result = size_ == other.size_ ? 0 : (size_ < other.size_ ? -1 : 1);
    return result;
  }

  FMT_CONSTEXPR_CHAR_TRAITS friend auto operator==(basic_string_view lhs,
                                                   basic_string_view rhs)
      -> bool {
    return lhs.compare(rhs) == 0;
  }
  friend auto operator!=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) != 0;
  }
  friend auto operator<(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) < 0;
  }
  friend auto operator<=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) <= 0;
  }
  friend auto operator>(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) > 0;
  }
  friend auto operator>=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) >= 0;
  }
};

using string_view = basic_string_view<char>;

/** Specifies if ``T`` is a character type. Can be specialized by users. */
template <typename T> struct is_char : std::false_type {};
template <> struct is_char<char> : std::true_type {};

// Returns a string view of `s`.
template <typename Char, FMT_ENABLE_IF(is_char<Char>::value)>
FMT_INLINE auto to_string_view(const Char* s) -> basic_string_view<Char> {
  return s;
}
template <typename Char, typename Traits, typename Alloc>
inline auto to_string_view(const std::basic_string<Char, Traits, Alloc>& s)
    -> basic_string_view<Char> {
  return s;
}
template <typename Char>
constexpr auto to_string_view(basic_string_view<Char> s)
    -> basic_string_view<Char> {
  return s;
}
template <typename Char,
          FMT_ENABLE_IF(!std::is_empty<detail::std_string_view<Char>>::value)>
inline auto to_string_view(detail::std_string_view<Char> s)
    -> basic_string_view<Char> {
  return s;
}

// A base class for compile-time strings. It is defined in the fmt namespace to
// make formatting functions visible via ADL, e.g. format(FMT_STRING("{}"), 42).
struct compile_string {};

template <typename S>
struct is_compile_string : std::is_base_of<compile_string, S> {};

template <typename S, FMT_ENABLE_IF(is_compile_string<S>::value)>
constexpr auto to_string_view(const S& s)
    -> basic_string_view<typename S::char_type> {
  return basic_string_view<typename S::char_type>(s);
}

FMT_BEGIN_DETAIL_NAMESPACE

void to_string_view(...);
using fmt::to_string_view;

// Specifies whether S is a string type convertible to fmt::basic_string_view.
// It should be a constexpr function but MSVC 2017 fails to compile it in
// enable_if and MSVC 2015 fails to compile it as an alias template.
template <typename S>
struct is_string : std::is_class<decltype(to_string_view(std::declval<S>()))> {
};

template <typename S, typename = void> struct char_t_impl {};
template <typename S> struct char_t_impl<S, enable_if_t<is_string<S>::value>> {
  using result = decltype(to_string_view(std::declval<S>()));
  using type = typename result::value_type;
};

// Reports a compile-time error if S is not a valid format string.
template <typename..., typename S, FMT_ENABLE_IF(!is_compile_string<S>::value)>
FMT_INLINE void check_format_string(const S&) {
#ifdef FMT_ENFORCE_COMPILE_STRING
  static_assert(is_compile_string<S>::value,
                "FMT_ENFORCE_COMPILE_STRING requires all format strings to use "
                "FMT_STRING.");
#endif
}
template <typename..., typename S, FMT_ENABLE_IF(is_compile_string<S>::value)>
void check_format_string(S);

FMT_NORETURN FMT_API void throw_format_error(const char* message);

struct error_handler {
  constexpr error_handler() = default;
  constexpr error_handler(const error_handler&) = default;

  // This function is intentionally not constexpr to give a compile-time error.
  FMT_NORETURN FMT_API void on_error(const char* message);
};
FMT_END_DETAIL_NAMESPACE

/** String's character type. */
template <typename S> using char_t = typename detail::char_t_impl<S>::type;

/**
  \rst
  Parsing context consisting of a format string range being parsed and an
  argument counter for automatic indexing.
  You can use the ``format_parse_context`` type alias for ``char`` instead.
  \endrst
 */
template <typename Char, typename ErrorHandler = detail::error_handler>
class basic_format_parse_context : private ErrorHandler {
 private:
  basic_string_view<Char> format_str_;
  int next_arg_id_;

 public:
  using char_type = Char;
  using iterator = typename basic_string_view<Char>::iterator;

  explicit constexpr basic_format_parse_context(
      basic_string_view<Char> format_str, ErrorHandler eh = {},
      int next_arg_id = 0)
      : ErrorHandler(eh), format_str_(format_str), next_arg_id_(next_arg_id) {}

  /**
    Returns an iterator to the beginning of the format string range being
    parsed.
   */
  constexpr auto begin() const FMT_NOEXCEPT -> iterator {
    return format_str_.begin();
  }

  /**
    Returns an iterator past the end of the format string range being parsed.
   */
  constexpr auto end() const FMT_NOEXCEPT -> iterator {
    return format_str_.end();
  }

  /** Advances the begin iterator to ``it``. */
  FMT_CONSTEXPR void advance_to(iterator it) {
    format_str_.remove_prefix(detail::to_unsigned(it - begin()));
  }

  /**
    Reports an error if using the manual argument indexing; otherwise returns
    the next argument index and switches to the automatic indexing.
   */
  FMT_CONSTEXPR auto next_arg_id() -> int {
    // Don't check if the argument id is valid to avoid overhead and because it
    // will be checked during formatting anyway.
    if (next_arg_id_ >= 0) return next_arg_id_++;
    on_error("cannot switch from manual to automatic argument indexing");
    return 0;
  }

  /**
    Reports an error if using the automatic argument indexing; otherwise
    switches to the manual indexing.
   */
  FMT_CONSTEXPR void check_arg_id(int) {
    if (next_arg_id_ > 0)
      on_error("cannot switch from automatic to manual argument indexing");
    else
      next_arg_id_ = -1;
  }

  FMT_CONSTEXPR void check_arg_id(basic_string_view<Char>) {}

  FMT_CONSTEXPR void on_error(const char* message) {
    ErrorHandler::on_error(message);
  }

  constexpr auto error_handler() const -> ErrorHandler { return *this; }
};

using format_parse_context = basic_format_parse_context<char>;

template <typename Context> class basic_format_arg;
template <typename Context> class basic_format_args;
template <typename Context> class dynamic_format_arg_store;

// A formatter for objects of type T.
template <typename T, typename Char = char, typename Enable = void>
struct formatter {
  // A deleted default constructor indicates a disabled formatter.
  formatter() = delete;
};

// Specifies if T has an enabled formatter specialization. A type can be
// formattable even if it doesn't have a formatter e.g. via a conversion.
template <typename T, typename Context>
using has_formatter =
    std::is_constructible<typename Context::template formatter_type<T>>;

// Checks whether T is a container with contiguous storage.
template <typename T> struct is_contiguous : std::false_type {};
template <typename Char>
struct is_contiguous<std::basic_string<Char>> : std::true_type {};

class appender;

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Context, typename T>
constexpr auto has_const_formatter_impl(T*)
    -> decltype(typename Context::template formatter_type<T>().format(
                    std::declval<const T&>(), std::declval<Context&>()),
                true) {
  return true;
}
template <typename Context>
constexpr auto has_const_formatter_impl(...) -> bool {
  return false;
}
template <typename T, typename Context>
constexpr auto has_const_formatter() -> bool {
  return has_const_formatter_impl<Context>(static_cast<T*>(nullptr));
}

// Extracts a reference to the container from back_insert_iterator.
template <typename Container>
inline auto get_container(std::back_insert_iterator<Container> it)
    -> Container& {
  using bi_iterator = std::back_insert_iterator<Container>;
  struct accessor : bi_iterator {
    accessor(bi_iterator iter) : bi_iterator(iter) {}
    using bi_iterator::container;
  };
  return *accessor(it).container;
}

template <typename Char, typename InputIt, typename OutputIt>
FMT_CONSTEXPR auto copy_str(InputIt begin, InputIt end, OutputIt out)
    -> OutputIt {
  while (begin != end) *out++ = static_cast<Char>(*begin++);
  return out;
}

template <typename Char, typename T, typename U,
          FMT_ENABLE_IF(
              std::is_same<remove_const_t<T>, U>::value&& is_char<U>::value)>
FMT_CONSTEXPR auto copy_str(T* begin, T* end, U* out) -> U* {
  if (is_constant_evaluated()) return copy_str<Char, T*, U*>(begin, end, out);
  auto size = to_unsigned(end - begin);
  memcpy(out, begin, size * sizeof(U));
  return out + size;
}

/**
  \rst
  A contiguous memory buffer with an optional growing ability. It is an internal
  class and shouldn't be used directly, only via `~fmt::basic_memory_buffer`.
  \endrst
 */
template <typename T> class buffer {
 private:
  T* ptr_;
  size_t size_;
  size_t capacity_;

 protected:
  // Don't initialize ptr_ since it is not accessed to save a few cycles.
  FMT_MSC_WARNING(suppress : 26495)
  buffer(size_t sz) FMT_NOEXCEPT : size_(sz), capacity_(sz) {}

  FMT_CONSTEXPR20 buffer(T* p = nullptr, size_t sz = 0,
                         size_t cap = 0) FMT_NOEXCEPT : ptr_(p),
                                                        size_(sz),
                                                        capacity_(cap) {}

  FMT_CONSTEXPR20 ~buffer() = default;
  buffer(buffer&&) = default;

  /** Sets the buffer data and capacity. */
  FMT_CONSTEXPR void set(T* buf_data, size_t buf_capacity) FMT_NOEXCEPT {
    ptr_ = buf_data;
    capacity_ = buf_capacity;
  }

  /** Increases the buffer capacity to hold at least *capacity* elements. */
  virtual FMT_CONSTEXPR20 void grow(size_t capacity) = 0;

 public:
  using value_type = T;
  using const_reference = const T&;

  buffer(const buffer&) = delete;
  void operator=(const buffer&) = delete;

  auto begin() FMT_NOEXCEPT -> T* { return ptr_; }
  auto end() FMT_NOEXCEPT -> T* { return ptr_ + size_; }

  auto begin() const FMT_NOEXCEPT -> const T* { return ptr_; }
  auto end() const FMT_NOEXCEPT -> const T* { return ptr_ + size_; }

  /** Returns the size of this buffer. */
  constexpr auto size() const FMT_NOEXCEPT -> size_t { return size_; }

  /** Returns the capacity of this buffer. */
  constexpr auto capacity() const FMT_NOEXCEPT -> size_t { return capacity_; }

  /** Returns a pointer to the buffer data. */
  FMT_CONSTEXPR auto data() FMT_NOEXCEPT -> T* { return ptr_; }

  /** Returns a pointer to the buffer data. */
  FMT_CONSTEXPR auto data() const FMT_NOEXCEPT -> const T* { return ptr_; }

  /** Clears this buffer. */
  void clear() { size_ = 0; }

  // Tries resizing the buffer to contain *count* elements. If T is a POD type
  // the new elements may not be initialized.
  FMT_CONSTEXPR20 void try_resize(size_t count) {
    try_reserve(count);
    size_ = count <= capacity_ ? count : capacity_;
  }

  // Tries increasing the buffer capacity to *new_capacity*. It can increase the
  // capacity by a smaller amount than requested but guarantees there is space
  // for at least one additional element either by increasing the capacity or by
  // flushing the buffer if it is full.
  FMT_CONSTEXPR20 void try_reserve(size_t new_capacity) {
    if (new_capacity > capacity_) grow(new_capacity);
  }

  FMT_CONSTEXPR20 void push_back(const T& value) {
    try_reserve(size_ + 1);
    ptr_[size_++] = value;
  }

  /** Appends data to the end of the buffer. */
  template <typename U> void append(const U* begin, const U* end);

  template <typename I> FMT_CONSTEXPR auto operator[](I index) -> T& {
    return ptr_[index];
  }
  template <typename I>
  FMT_CONSTEXPR auto operator[](I index) const -> const T& {
    return ptr_[index];
  }
};

struct buffer_traits {
  explicit buffer_traits(size_t) {}
  auto count() const -> size_t { return 0; }
  auto limit(size_t size) -> size_t { return size; }
};

class fixed_buffer_traits {
 private:
  size_t count_ = 0;
  size_t limit_;

 public:
  explicit fixed_buffer_traits(size_t limit) : limit_(limit) {}
  auto count() const -> size_t { return count_; }
  auto limit(size_t size) -> size_t {
    size_t n = limit_ > count_ ? limit_ - count_ : 0;
    count_ += size;
    return size < n ? size : n;
  }
};

// A buffer that writes to an output iterator when flushed.
template <typename OutputIt, typename T, typename Traits = buffer_traits>
class iterator_buffer final : public Traits, public buffer<T> {
 private:
  OutputIt out_;
  enum { buffer_size = 256 };
  T data_[buffer_size];

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() == buffer_size) flush();
  }

  void flush() {
    auto size = this->size();
    this->clear();
    out_ = copy_str<T>(data_, data_ + this->limit(size), out_);
  }

 public:
  explicit iterator_buffer(OutputIt out, size_t n = buffer_size)
      : Traits(n), buffer<T>(data_, 0, buffer_size), out_(out) {}
  iterator_buffer(iterator_buffer&& other)
      : Traits(other), buffer<T>(data_, 0, buffer_size), out_(other.out_) {}
  ~iterator_buffer() { flush(); }

  auto out() -> OutputIt {
    flush();
    return out_;
  }
  auto count() const -> size_t { return Traits::count() + this->size(); }
};

template <typename T>
class iterator_buffer<T*, T, fixed_buffer_traits> final
    : public fixed_buffer_traits,
      public buffer<T> {
 private:
  T* out_;
  enum { buffer_size = 256 };
  T data_[buffer_size];

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() == this->capacity()) flush();
  }

  void flush() {
    size_t n = this->limit(this->size());
    if (this->data() == out_) {
      out_ += n;
      this->set(data_, buffer_size);
    }
    this->clear();
  }

 public:
  explicit iterator_buffer(T* out, size_t n = buffer_size)
      : fixed_buffer_traits(n), buffer<T>(out, 0, n), out_(out) {}
  iterator_buffer(iterator_buffer&& other)
      : fixed_buffer_traits(other),
        buffer<T>(std::move(other)),
        out_(other.out_) {
    if (this->data() != out_) {
      this->set(data_, buffer_size);
      this->clear();
    }
  }
  ~iterator_buffer() { flush(); }

  auto out() -> T* {
    flush();
    return out_;
  }
  auto count() const -> size_t {
    return fixed_buffer_traits::count() + this->size();
  }
};

template <typename T> class iterator_buffer<T*, T> final : public buffer<T> {
 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {}

 public:
  explicit iterator_buffer(T* out, size_t = 0) : buffer<T>(out, 0, ~size_t()) {}

  auto out() -> T* { return &*this->end(); }
};

// A buffer that writes to a container with the contiguous storage.
template <typename Container>
class iterator_buffer<std::back_insert_iterator<Container>,
                      enable_if_t<is_contiguous<Container>::value,
                                  typename Container::value_type>>
    final : public buffer<typename Container::value_type> {
 private:
  Container& container_;

 protected:
  FMT_CONSTEXPR20 void grow(size_t capacity) override {
    container_.resize(capacity);
    this->set(&container_[0], capacity);
  }

 public:
  explicit iterator_buffer(Container& c)
      : buffer<typename Container::value_type>(c.size()), container_(c) {}
  explicit iterator_buffer(std::back_insert_iterator<Container> out, size_t = 0)
      : iterator_buffer(get_container(out)) {}
  auto out() -> std::back_insert_iterator<Container> {
    return std::back_inserter(container_);
  }
};

// A buffer that counts the number of code units written discarding the output.
template <typename T = char> class counting_buffer final : public buffer<T> {
 private:
  enum { buffer_size = 256 };
  T data_[buffer_size];
  size_t count_ = 0;

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() != buffer_size) return;
    count_ += this->size();
    this->clear();
  }

 public:
  counting_buffer() : buffer<T>(data_, 0, buffer_size) {}

  auto count() -> size_t { return count_ + this->size(); }
};

template <typename T>
using buffer_appender = conditional_t<std::is_same<T, char>::value, appender,
                                      std::back_insert_iterator<buffer<T>>>;

// Maps an output iterator to a buffer.
template <typename T, typename OutputIt>
auto get_buffer(OutputIt out) -> iterator_buffer<OutputIt, T> {
  return iterator_buffer<OutputIt, T>(out);
}

template <typename Buffer>
auto get_iterator(Buffer& buf) -> decltype(buf.out()) {
  return buf.out();
}
template <typename T> auto get_iterator(buffer<T>& buf) -> buffer_appender<T> {
  return buffer_appender<T>(buf);
}

template <typename T, typename Char = char, typename Enable = void>
struct fallback_formatter {
  fallback_formatter() = delete;
};

// Specifies if T has an enabled fallback_formatter specialization.
template <typename T, typename Char>
using has_fallback_formatter =
    std::is_constructible<fallback_formatter<T, Char>>;

struct view {};

template <typename Char, typename T> struct named_arg : view {
  const Char* name;
  const T& value;
  named_arg(const Char* n, const T& v) : name(n), value(v) {}
};

template <typename Char> struct named_arg_info {
  const Char* name;
  int id;
};

template <typename T, typename Char, size_t NUM_ARGS, size_t NUM_NAMED_ARGS>
struct arg_data {
  // args_[0].named_args points to named_args_ to avoid bloating format_args.
  // +1 to workaround a bug in gcc 7.5 that causes duplicated-branches warning.
  T args_[1 + (NUM_ARGS != 0 ? NUM_ARGS : +1)];
  named_arg_info<Char> named_args_[NUM_NAMED_ARGS];

  template <typename... U>
  arg_data(const U&... init) : args_{T(named_args_, NUM_NAMED_ARGS), init...} {}
  arg_data(const arg_data& other) = delete;
  auto args() const -> const T* { return args_ + 1; }
  auto named_args() -> named_arg_info<Char>* { return named_args_; }
};

template <typename T, typename Char, size_t NUM_ARGS>
struct arg_data<T, Char, NUM_ARGS, 0> {
  // +1 to workaround a bug in gcc 7.5 that causes duplicated-branches warning.
  T args_[NUM_ARGS != 0 ? NUM_ARGS : +1];

  template <typename... U>
  FMT_CONSTEXPR FMT_INLINE arg_data(const U&... init) : args_{init...} {}
  FMT_CONSTEXPR FMT_INLINE auto args() const -> const T* { return args_; }
  FMT_CONSTEXPR FMT_INLINE auto named_args() -> std::nullptr_t {
    return nullptr;
  }
};

template <typename Char>
inline void init_named_args(named_arg_info<Char>*, int, int) {}

template <typename T> struct is_named_arg : std::false_type {};
template <typename T> struct is_statically_named_arg : std::false_type {};

template <typename T, typename Char>
struct is_named_arg<named_arg<Char, T>> : std::true_type {};

template <typename Char, typename T, typename... Tail,
          FMT_ENABLE_IF(!is_named_arg<T>::value)>
void init_named_args(named_arg_info<Char>* named_args, int arg_count,
                     int named_arg_count, const T&, const Tail&... args) {
  init_named_args(named_args, arg_count + 1, named_arg_count, args...);
}

template <typename Char, typename T, typename... Tail,
          FMT_ENABLE_IF(is_named_arg<T>::value)>
void init_named_args(named_arg_info<Char>* named_args, int arg_count,
                     int named_arg_count, const T& arg, const Tail&... args) {
  named_args[named_arg_count++] = {arg.name, arg_count};
  init_named_args(named_args, arg_count + 1, named_arg_count, args...);
}

template <typename... Args>
FMT_CONSTEXPR FMT_INLINE void init_named_args(std::nullptr_t, int, int,
                                              const Args&...) {}

template <bool B = false> constexpr auto count() -> size_t { return B ? 1 : 0; }
template <bool B1, bool B2, bool... Tail> constexpr auto count() -> size_t {
  return (B1 ? 1 : 0) + count<B2, Tail...>();
}

template <typename... Args> constexpr auto count_named_args() -> size_t {
  return count<is_named_arg<Args>::value...>();
}

template <typename... Args>
constexpr auto count_statically_named_args() -> size_t {
  return count<is_statically_named_arg<Args>::value...>();
}

enum class type {
  none_type,
  // Integer types should go first,
  int_type,
  uint_type,
  long_long_type,
  ulong_long_type,
  int128_type,
  uint128_type,
  bool_type,
  char_type,
  last_integer_type = char_type,
  // followed by floating-point types.
  float_type,
  double_type,
  long_double_type,
  last_numeric_type = long_double_type,
  cstring_type,
  string_type,
  pointer_type,
  custom_type
};

// Maps core type T to the corresponding type enum constant.
template <typename T, typename Char>
struct type_constant : std::integral_constant<type, type::custom_type> {};

#define FMT_TYPE_CONSTANT(Type, constant) \
  template <typename Char>                \
  struct type_constant<Type, Char>        \
      : std::integral_constant<type, type::constant> {}

FMT_TYPE_CONSTANT(int, int_type);
FMT_TYPE_CONSTANT(unsigned, uint_type);
FMT_TYPE_CONSTANT(long long, long_long_type);
FMT_TYPE_CONSTANT(unsigned long long, ulong_long_type);
FMT_TYPE_CONSTANT(int128_t, int128_type);
FMT_TYPE_CONSTANT(uint128_t, uint128_type);
FMT_TYPE_CONSTANT(bool, bool_type);
FMT_TYPE_CONSTANT(Char, char_type);
FMT_TYPE_CONSTANT(float, float_type);
FMT_TYPE_CONSTANT(double, double_type);
FMT_TYPE_CONSTANT(long double, long_double_type);
FMT_TYPE_CONSTANT(const Char*, cstring_type);
FMT_TYPE_CONSTANT(basic_string_view<Char>, string_type);
FMT_TYPE_CONSTANT(const void*, pointer_type);

constexpr bool is_integral_type(type t) {
  return t > type::none_type && t <= type::last_integer_type;
}

constexpr bool is_arithmetic_type(type t) {
  return t > type::none_type && t <= type::last_numeric_type;
}

struct unformattable {};
struct unformattable_char : unformattable {};
struct unformattable_const : unformattable {};
struct unformattable_pointer : unformattable {};

template <typename Char> struct string_value {
  const Char* data;
  size_t size;
};

template <typename Char> struct named_arg_value {
  const named_arg_info<Char>* data;
  size_t size;
};

template <typename Context> struct custom_value {
  using parse_context = typename Context::parse_context_type;
  void* value;
  void (*format)(void* arg, parse_context& parse_ctx, Context& ctx);
};

// A formatting argument value.
template <typename Context> class value {
 public:
  using char_type = typename Context::char_type;

  union {
    monostate no_value;
    int int_value;
    unsigned uint_value;
    long long long_long_value;
    unsigned long long ulong_long_value;
    int128_t int128_value;
    uint128_t uint128_value;
    bool bool_value;
    char_type char_value;
    float float_value;
    double double_value;
    long double long_double_value;
    const void* pointer;
    string_value<char_type> string;
    custom_value<Context> custom;
    named_arg_value<char_type> named_args;
  };

  constexpr FMT_INLINE value() : no_value() {}
  constexpr FMT_INLINE value(int val) : int_value(val) {}
  constexpr FMT_INLINE value(unsigned val) : uint_value(val) {}
  constexpr FMT_INLINE value(long long val) : long_long_value(val) {}
  constexpr FMT_INLINE value(unsigned long long val) : ulong_long_value(val) {}
  FMT_INLINE value(int128_t val) : int128_value(val) {}
  FMT_INLINE value(uint128_t val) : uint128_value(val) {}
  constexpr FMT_INLINE value(float val) : float_value(val) {}
  constexpr FMT_INLINE value(double val) : double_value(val) {}
  FMT_INLINE value(long double val) : long_double_value(val) {}
  constexpr FMT_INLINE value(bool val) : bool_value(val) {}
  constexpr FMT_INLINE value(char_type val) : char_value(val) {}
  FMT_CONSTEXPR FMT_INLINE value(const char_type* val) {
    string.data = val;
    if (is_constant_evaluated()) string.size = {};
  }
  FMT_CONSTEXPR FMT_INLINE value(basic_string_view<char_type> val) {
    string.data = val.data();
    string.size = val.size();
  }
  FMT_INLINE value(const void* val) : pointer(val) {}
  FMT_INLINE value(const named_arg_info<char_type>* args, size_t size)
      : named_args{args, size} {}

  template <typename T> FMT_CONSTEXPR FMT_INLINE value(T& val) {
    using value_type = remove_cvref_t<T>;
    custom.value = const_cast<value_type*>(&val);
    // Get the formatter type through the context to allow different contexts
    // have different extension points, e.g. `formatter<T>` for `format` and
    // `printf_formatter<T>` for `printf`.
    custom.format = format_custom_arg<
        value_type,
        conditional_t<has_formatter<value_type, Context>::value,
                      typename Context::template formatter_type<value_type>,
                      fallback_formatter<value_type, char_type>>>;
  }
  value(unformattable);
  value(unformattable_char);
  value(unformattable_const);
  value(unformattable_pointer);

 private:
  // Formats an argument of a custom type, such as a user-defined class.
  template <typename T, typename Formatter>
  static void format_custom_arg(void* arg,
                                typename Context::parse_context_type& parse_ctx,
                                Context& ctx) {
    auto f = Formatter();
    parse_ctx.advance_to(f.parse(parse_ctx));
    using qualified_type =
        conditional_t<has_const_formatter<T, Context>(), const T, T>;
    ctx.advance_to(f.format(*static_cast<qualified_type*>(arg), ctx));
  }
};

template <typename Context, typename T>
FMT_CONSTEXPR auto make_arg(const T& value) -> basic_format_arg<Context>;

// To minimize the number of types we need to deal with, long is translated
// either to int or to long long depending on its size.
enum { long_short = sizeof(long) == sizeof(int) };
using long_type = conditional_t<long_short, int, long long>;
using ulong_type = conditional_t<long_short, unsigned, unsigned long long>;

// Maps formatting arguments to core types.
// arg_mapper reports errors by returning unformattable instead of using
// static_assert because it's used in the is_formattable trait.
template <typename Context> struct arg_mapper {
  using char_type = typename Context::char_type;

  FMT_CONSTEXPR FMT_INLINE auto map(signed char val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned char val) -> unsigned {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(short val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned short val) -> unsigned {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(int val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned val) -> unsigned { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(long val) -> long_type { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned long val) -> ulong_type {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(long long val) -> long long { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned long long val)
      -> unsigned long long {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(int128_t val) -> int128_t { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(uint128_t val) -> uint128_t { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(bool val) -> bool { return val; }

  template <typename T, FMT_ENABLE_IF(std::is_same<T, char>::value ||
                                      std::is_same<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(T val) -> char_type {
    return val;
  }
  template <typename T, enable_if_t<(std::is_same<T, wchar_t>::value ||
#ifdef __cpp_char8_t
                                     std::is_same<T, char8_t>::value ||
#endif
                                     std::is_same<T, char16_t>::value ||
                                     std::is_same<T, char32_t>::value) &&
                                        !std::is_same<T, char_type>::value,
                                    int> = 0>
  FMT_CONSTEXPR FMT_INLINE auto map(T) -> unformattable_char {
    return {};
  }

  FMT_CONSTEXPR FMT_INLINE auto map(float val) -> float { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(double val) -> double { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(long double val) -> long double {
    return val;
  }

  FMT_CONSTEXPR FMT_INLINE auto map(char_type* val) -> const char_type* {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(const char_type* val) -> const char_type* {
    return val;
  }
  template <typename T,
            FMT_ENABLE_IF(is_string<T>::value && !std::is_pointer<T>::value &&
                          std::is_same<char_type, char_t<T>>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return to_string_view(val);
  }
  template <typename T,
            FMT_ENABLE_IF(is_string<T>::value && !std::is_pointer<T>::value &&
                          !std::is_same<char_type, char_t<T>>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T&) -> unformattable_char {
    return {};
  }
  template <typename T,
            FMT_ENABLE_IF(
                std::is_constructible<basic_string_view<char_type>, T>::value &&
                !is_string<T>::value && !has_formatter<T, Context>::value &&
                !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return basic_string_view<char_type>(val);
  }
  template <
      typename T,
      FMT_ENABLE_IF(
          std::is_constructible<std_string_view<char_type>, T>::value &&
          !std::is_constructible<basic_string_view<char_type>, T>::value &&
          !is_string<T>::value && !has_formatter<T, Context>::value &&
          !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return std_string_view<char_type>(val);
  }

  using cstring_result = conditional_t<std::is_same<char_type, char>::value,
                                       const char*, unformattable_pointer>;

  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(const signed char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(const unsigned char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(signed char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(unsigned char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }

  FMT_CONSTEXPR FMT_INLINE auto map(void* val) -> const void* { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(const void* val) -> const void* {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(std::nullptr_t val) -> const void* {
    return val;
  }

  // We use SFINAE instead of a const T* parameter to avoid conflicting with
  // the C array overload.
  template <
      typename T,
      FMT_ENABLE_IF(
          std::is_member_pointer<T>::value ||
          std::is_function<typename std::remove_pointer<T>::type>::value ||
          (std::is_convertible<const T&, const void*>::value &&
           !std::is_convertible<const T&, const char_type*>::value))>
  FMT_CONSTEXPR auto map(const T&) -> unformattable_pointer {
    return {};
  }

  template <typename T, std::size_t N,
            FMT_ENABLE_IF(!std::is_same<T, wchar_t>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T (&values)[N]) -> const T (&)[N] {
    return values;
  }

  template <typename T,
            FMT_ENABLE_IF(
                std::is_enum<T>::value&& std::is_convertible<T, int>::value &&
                !has_formatter<T, Context>::value &&
                !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> decltype(std::declval<arg_mapper>().map(
          static_cast<typename std::underlying_type<T>::type>(val))) {
    return map(static_cast<typename std::underlying_type<T>::type>(val));
  }

  FMT_CONSTEXPR FMT_INLINE auto map(detail::byte val) -> unsigned {
    return map(static_cast<unsigned char>(val));
  }

  template <typename T, typename U = remove_cvref_t<T>>
  struct formattable
      : bool_constant<has_const_formatter<U, Context>() ||
                      !std::is_const<remove_reference_t<T>>::value ||
                      has_fallback_formatter<U, char_type>::value> {};

#if FMT_MSC_VER != 0 && FMT_MSC_VER < 1910
  // Workaround a bug in MSVC.
  template <typename T> FMT_CONSTEXPR FMT_INLINE auto do_map(T&& val) -> T& {
    return val;
  }
#else
  template <typename T, FMT_ENABLE_IF(formattable<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto do_map(T&& val) -> T& {
    return val;
  }
  template <typename T, FMT_ENABLE_IF(!formattable<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto do_map(T&&) -> unformattable_const {
    return {};
  }
#endif

  template <typename T, typename U = remove_cvref_t<T>,
            FMT_ENABLE_IF(!is_string<U>::value && !is_char<U>::value &&
                          !std::is_array<U>::value &&
                          (has_formatter<U, Context>::value ||
                           has_fallback_formatter<U, char_type>::value))>
  FMT_CONSTEXPR FMT_INLINE auto map(T&& val)
      -> decltype(this->do_map(std::forward<T>(val))) {
    return do_map(std::forward<T>(val));
  }

  template <typename T, FMT_ENABLE_IF(is_named_arg<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& named_arg)
      -> decltype(std::declval<arg_mapper>().map(named_arg.value)) {
    return map(named_arg.value);
  }

  auto map(...) -> unformattable { return {}; }
};

// A type constant after applying arg_mapper<Context>.
template <typename T, typename Context>
using mapped_type_constant =
    type_constant<decltype(arg_mapper<Context>().map(std::declval<const T&>())),
                  typename Context::char_type>;

enum { packed_arg_bits = 4 };
// Maximum number of arguments with packed types.
enum { max_packed_args = 62 / packed_arg_bits };
enum : unsigned long long { is_unpacked_bit = 1ULL << 63 };
enum : unsigned long long { has_named_args_bit = 1ULL << 62 };

FMT_END_DETAIL_NAMESPACE

// An output iterator that appends to a buffer.
// It is used to reduce symbol sizes for the common case.
class appender : public std::back_insert_iterator<detail::buffer<char>> {
  using base = std::back_insert_iterator<detail::buffer<char>>;

  template <typename T>
  friend auto get_buffer(appender out) -> detail::buffer<char>& {
    return detail::get_container(out);
  }

 public:
  using std::back_insert_iterator<detail::buffer<char>>::back_insert_iterator;
  appender(base it) FMT_NOEXCEPT : base(it) {}
  using _Unchecked_type = appender;  // Mark iterator as checked.

  auto operator++() FMT_NOEXCEPT -> appender& { return *this; }

  auto operator++(int) FMT_NOEXCEPT -> appender { return *this; }
};

// A formatting argument. It is a trivially copyable/constructible type to
// allow storage in basic_memory_buffer.
template <typename Context> class basic_format_arg {
 private:
  detail::value<Context> value_;
  detail::type type_;

  template <typename ContextType, typename T>
  friend FMT_CONSTEXPR auto detail::make_arg(const T& value)
      -> basic_format_arg<ContextType>;

  template <typename Visitor, typename Ctx>
  friend FMT_CONSTEXPR auto visit_format_arg(Visitor&& vis,
                                             const basic_format_arg<Ctx>& arg)
      -> decltype(vis(0));

  friend class basic_format_args<Context>;
  friend class dynamic_format_arg_store<Context>;

  using char_type = typename Context::char_type;

  template <typename T, typename Char, size_t NUM_ARGS, size_t NUM_NAMED_ARGS>
  friend struct detail::arg_data;

  basic_format_arg(const detail::named_arg_info<char_type>* args, size_t size)
      : value_(args, size) {}

 public:
  class handle {
   public:
    explicit handle(detail::custom_value<Context> custom) : custom_(custom) {}

    void format(typename Context::parse_context_type& parse_ctx,
                Context& ctx) const {
      custom_.format(custom_.value, parse_ctx, ctx);
    }

   private:
    detail::custom_value<Context> custom_;
  };

  constexpr basic_format_arg() : type_(detail::type::none_type) {}

  constexpr explicit operator bool() const FMT_NOEXCEPT {
    return type_ != detail::type::none_type;
  }

  auto type() const -> detail::type { return type_; }

  auto is_integral() const -> bool { return detail::is_integral_type(type_); }
  auto is_arithmetic() const -> bool {
    return detail::is_arithmetic_type(type_);
  }
};

/**
  \rst
  Visits an argument dispatching to the appropriate visit method based on
  the argument type. For example, if the argument type is ``double`` then
  ``vis(value)`` will be called with the value of type ``double``.
  \endrst
 */
template <typename Visitor, typename Context>
FMT_CONSTEXPR FMT_INLINE auto visit_format_arg(
    Visitor&& vis, const basic_format_arg<Context>& arg) -> decltype(vis(0)) {
  switch (arg.type_) {
  case detail::type::none_type:
    break;
  case detail::type::int_type:
    return vis(arg.value_.int_value);
  case detail::type::uint_type:
    return vis(arg.value_.uint_value);
  case detail::type::long_long_type:
    return vis(arg.value_.long_long_value);
  case detail::type::ulong_long_type:
    return vis(arg.value_.ulong_long_value);
  case detail::type::int128_type:
    return vis(detail::convert_for_visit(arg.value_.int128_value));
  case detail::type::uint128_type:
    return vis(detail::convert_for_visit(arg.value_.uint128_value));
  case detail::type::bool_type:
    return vis(arg.value_.bool_value);
  case detail::type::char_type:
    return vis(arg.value_.char_value);
  case detail::type::float_type:
    return vis(arg.value_.float_value);
  case detail::type::double_type:
    return vis(arg.value_.double_value);
  case detail::type::long_double_type:
    return vis(arg.value_.long_double_value);
  case detail::type::cstring_type:
    return vis(arg.value_.string.data);
  case detail::type::string_type:
    using sv = basic_string_view<typename Context::char_type>;
    return vis(sv(arg.value_.string.data, arg.value_.string.size));
  case detail::type::pointer_type:
    return vis(arg.value_.pointer);
  case detail::type::custom_type:
    return vis(typename basic_format_arg<Context>::handle(arg.value_.custom));
  }
  return vis(monostate());
}

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Char, typename InputIt>
auto copy_str(InputIt begin, InputIt end, appender out) -> appender {
  get_container(out).append(begin, end);
  return out;
}

#if FMT_GCC_VERSION && FMT_GCC_VERSION < 500
// A workaround for gcc 4.8 to make void_t work in a SFINAE context.
template <typename... Ts> struct void_t_impl { using type = void; };
template <typename... Ts>
using void_t = typename detail::void_t_impl<Ts...>::type;
#else
template <typename...> using void_t = void;
#endif

template <typename It, typename T, typename Enable = void>
struct is_output_iterator : std::false_type {};

template <typename It, typename T>
struct is_output_iterator<
    It, T,
    void_t<typename std::iterator_traits<It>::iterator_category,
           decltype(*std::declval<It>() = std::declval<T>())>>
    : std::true_type {};

template <typename OutputIt>
struct is_back_insert_iterator : std::false_type {};
template <typename Container>
struct is_back_insert_iterator<std::back_insert_iterator<Container>>
    : std::true_type {};

template <typename OutputIt>
struct is_contiguous_back_insert_iterator : std::false_type {};
template <typename Container>
struct is_contiguous_back_insert_iterator<std::back_insert_iterator<Container>>
    : is_contiguous<Container> {};
template <>
struct is_contiguous_back_insert_iterator<appender> : std::true_type {};

// A type-erased reference to an std::locale to avoid heavy <locale> include.
class locale_ref {
 private:
  const void* locale_;  // A type-erased pointer to std::locale.

 public:
  constexpr locale_ref() : locale_(nullptr) {}
  template <typename Locale> explicit locale_ref(const Locale& loc);

  explicit operator bool() const FMT_NOEXCEPT { return locale_ != nullptr; }

  template <typename Locale> auto get() const -> Locale;
};

template <typename> constexpr auto encode_types() -> unsigned long long {
  return 0;
}

template <typename Context, typename Arg, typename... Args>
constexpr auto encode_types() -> unsigned long long {
  return static_cast<unsigned>(mapped_type_constant<Arg, Context>::value) |
         (encode_types<Context, Args...>() << packed_arg_bits);
}

template <typename Context, typename T>
FMT_CONSTEXPR auto make_arg(const T& value) -> basic_format_arg<Context> {
  basic_format_arg<Context> arg;
  arg.type_ = mapped_type_constant<T, Context>::value;
  arg.value_ = arg_mapper<Context>().map(value);
  return arg;
}

// The type template parameter is there to avoid an ODR violation when using
// a fallback formatter in one translation unit and an implicit conversion in
// another (not recommended).
template <bool IS_PACKED, typename Context, type, typename T,
          FMT_ENABLE_IF(IS_PACKED)>
FMT_CONSTEXPR FMT_INLINE auto make_arg(T&& val) -> value<Context> {
  const auto& arg = arg_mapper<Context>().map(std::forward<T>(val));

  constexpr bool formattable_char =
      !std::is_same<decltype(arg), const unformattable_char&>::value;
  static_assert(formattable_char, "Mixing character types is disallowed.");

  constexpr bool formattable_const =
      !std::is_same<decltype(arg), const unformattable_const&>::value;
  static_assert(formattable_const, "Cannot format a const argument.");

  // Formatting of arbitrary pointers is disallowed. If you want to output
  // a pointer cast it to "void *" or "const void *". In particular, this
  // forbids formatting of "[const] volatile char *" which is printed as bool
  // by iostreams.
  constexpr bool formattable_pointer =
      !std::is_same<decltype(arg), const unformattable_pointer&>::value;
  static_assert(formattable_pointer,
                "Formatting of non-void pointers is disallowed.");

  constexpr bool formattable =
      !std::is_same<decltype(arg), const unformattable&>::value;
  static_assert(
      formattable,
      "Cannot format an argument. To make type T formattable provide a "
      "formatter<T> specialization: https://fmt.dev/latest/api.html#udt");
  return {arg};
}

template <bool IS_PACKED, typename Context, type, typename T,
          FMT_ENABLE_IF(!IS_PACKED)>
inline auto make_arg(const T& value) -> basic_format_arg<Context> {
  return make_arg<Context>(value);
}
FMT_END_DETAIL_NAMESPACE

// Formatting context.
template <typename OutputIt, typename Char> class basic_format_context {
 public:
  /** The character type for the output. */
  using char_type = Char;

 private:
  OutputIt out_;
  basic_format_args<basic_format_context> args_;
  detail::locale_ref loc_;

 public:
  using iterator = OutputIt;
  using format_arg = basic_format_arg<basic_format_context>;
  using parse_context_type = basic_format_parse_context<Char>;
  template <typename T> using formatter_type = formatter<T, char_type>;

  basic_format_context(basic_format_context&&) = default;
  basic_format_context(const basic_format_context&) = delete;
  void operator=(const basic_format_context&) = delete;
  /**
   Constructs a ``basic_format_context`` object. References to the arguments are
   stored in the object so make sure they have appropriate lifetimes.
   */
  constexpr basic_format_context(
      OutputIt out, basic_format_args<basic_format_context> ctx_args,
      detail::locale_ref loc = detail::locale_ref())
      : out_(out), args_(ctx_args), loc_(loc) {}

  constexpr auto arg(int id) const -> format_arg { return args_.get(id); }
  FMT_CONSTEXPR auto arg(basic_string_view<char_type> name) -> format_arg {
    return args_.get(name);
  }
  FMT_CONSTEXPR auto arg_id(basic_string_view<char_type> name) -> int {
    return args_.get_id(name);
  }
  auto args() const -> const basic_format_args<basic_format_context>& {
    return args_;
  }

  FMT_CONSTEXPR auto error_handler() -> detail::error_handler { return {}; }
  void on_error(const char* message) { error_handler().on_error(message); }

  // Returns an iterator to the beginning of the output range.
  FMT_CONSTEXPR auto out() -> iterator { return out_; }

  // Advances the begin iterator to ``it``.
  void advance_to(iterator it) {
    if (!detail::is_back_insert_iterator<iterator>()) out_ = it;
  }

  FMT_CONSTEXPR auto locale() -> detail::locale_ref { return loc_; }
};

template <typename Char>
using buffer_context =
    basic_format_context<detail::buffer_appender<Char>, Char>;
using format_context = buffer_context<char>;

// Workaround an alias issue: https://stackoverflow.com/q/62767544/471164.
#define FMT_BUFFER_CONTEXT(Char) \
  basic_format_context<detail::buffer_appender<Char>, Char>

template <typename T, typename Char = char>
using is_formattable = bool_constant<
    !std::is_base_of<detail::unformattable,
                     decltype(detail::arg_mapper<buffer_context<Char>>().map(
                         std::declval<T>()))>::value &&
    !detail::has_fallback_formatter<T, Char>::value>;

/**
  \rst
  An array of references to arguments. It can be implicitly converted into
  `~fmt::basic_format_args` for passing into type-erased formatting functions
  such as `~fmt::vformat`.
  \endrst
 */
template <typename Context, typename... Args>
class format_arg_store
#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
    // Workaround a GCC template argument substitution bug.
    : public basic_format_args<Context>
#endif
{
 private:
  static const size_t num_args = sizeof...(Args);
  static const size_t num_named_args = detail::count_named_args<Args...>();
  static const bool is_packed = num_args <= detail::max_packed_args;

  using value_type = conditional_t<is_packed, detail::value<Context>,
                                   basic_format_arg<Context>>;

  detail::arg_data<value_type, typename Context::char_type, num_args,
                   num_named_args>
      data_;

  friend class basic_format_args<Context>;

  static constexpr unsigned long long desc =
      (is_packed ? detail::encode_types<Context, Args...>()
                 : detail::is_unpacked_bit | num_args) |
      (num_named_args != 0
           ? static_cast<unsigned long long>(detail::has_named_args_bit)
           : 0);

 public:
  template <typename... T>
  FMT_CONSTEXPR FMT_INLINE format_arg_store(T&&... args)
      :
#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
        basic_format_args<Context>(*this),
#endif
        data_{detail::make_arg<
            is_packed, Context,
            detail::mapped_type_constant<remove_cvref_t<T>, Context>::value>(
            std::forward<T>(args))...} {
    detail::init_named_args(data_.named_args(), 0, 0, args...);
  }
};

/**
  \rst
  Constructs a `~fmt::format_arg_store` object that contains references to
  arguments and can be implicitly converted to `~fmt::format_args`. `Context`
  can be omitted in which case it defaults to `~fmt::context`.
  See `~fmt::arg` for lifetime considerations.
  \endrst
 */
template <typename Context = format_context, typename... Args>
constexpr auto make_format_args(Args&&... args)
    -> format_arg_store<Context, remove_cvref_t<Args>...> {
  return {std::forward<Args>(args)...};
}

/**
  \rst
  Returns a named argument to be used in a formatting function.
  It should only be used in a call to a formatting function or
  `dynamic_format_arg_store::push_back`.

  **Example**::

    fmt::print("Elapsed time: {s:.2f} seconds", fmt::arg("s", 1.23));
  \endrst
 */
template <typename Char, typename T>
inline auto arg(const Char* name, const T& arg) -> detail::named_arg<Char, T> {
  static_assert(!detail::is_named_arg<T>(), "nested named arguments");
  return {name, arg};
}

/**
  \rst
  A view of a collection of formatting arguments. To avoid lifetime issues it
  should only be used as a parameter type in type-erased functions such as
  ``vformat``::

    void vlog(string_view format_str, format_args args);  // OK
    format_args args = make_format_args(42);  // Error: dangling reference
  \endrst
 */
template <typename Context> class basic_format_args {
 public:
  using size_type = int;
  using format_arg = basic_format_arg<Context>;

 private:
  // A descriptor that contains information about formatting arguments.
  // If the number of arguments is less or equal to max_packed_args then
  // argument types are passed in the descriptor. This reduces binary code size
  // per formatting function call.
  unsigned long long desc_;
  union {
    // If is_packed() returns true then argument values are stored in values_;
    // otherwise they are stored in args_. This is done to improve cache
    // locality and reduce compiled code size since storing larger objects
    // may require more code (at least on x86-64) even if the same amount of
    // data is actually copied to stack. It saves ~10% on the bloat test.
    const detail::value<Context>* values_;
    const format_arg* args_;
  };

  constexpr auto is_packed() const -> bool {
    return (desc_ & detail::is_unpacked_bit) == 0;
  }
  auto has_named_args() const -> bool {
    return (desc_ & detail::has_named_args_bit) != 0;
  }

  FMT_CONSTEXPR auto type(int index) const -> detail::type {
    int shift = index * detail::packed_arg_bits;
    unsigned int mask = (1 << detail::packed_arg_bits) - 1;
    return static_cast<detail::type>((desc_ >> shift) & mask);
  }

  constexpr FMT_INLINE basic_format_args(unsigned long long desc,
                                         const detail::value<Context>* values)
      : desc_(desc), values_(values) {}
  constexpr basic_format_args(unsigned long long desc, const format_arg* args)
      : desc_(desc), args_(args) {}

 public:
  constexpr basic_format_args() : desc_(0), args_(nullptr) {}

  /**
   \rst
   Constructs a `basic_format_args` object from `~fmt::format_arg_store`.
   \endrst
   */
  template <typename... Args>
  constexpr FMT_INLINE basic_format_args(
      const format_arg_store<Context, Args...>& store)
      : basic_format_args(format_arg_store<Context, Args...>::desc,
                          store.data_.args()) {}

  /**
   \rst
   Constructs a `basic_format_args` object from
   `~fmt::dynamic_format_arg_store`.
   \endrst
   */
  constexpr FMT_INLINE basic_format_args(
      const dynamic_format_arg_store<Context>& store)
      : basic_format_args(store.get_types(), store.data()) {}

  /**
   \rst
   Constructs a `basic_format_args` object from a dynamic set of arguments.
   \endrst
   */
  constexpr basic_format_args(const format_arg* args, int count)
      : basic_format_args(detail::is_unpacked_bit | detail::to_unsigned(count),
                          args) {}

  /** Returns the argument with the specified id. */
  FMT_CONSTEXPR auto get(int id) const -> format_arg {
    format_arg arg;
    if (!is_packed()) {
      if (id < max_size()) arg = args_[id];
      return arg;
    }
    if (id >= detail::max_packed_args) return arg;
    arg.type_ = type(id);
    if (arg.type_ == detail::type::none_type) return arg;
    arg.value_ = values_[id];
    return arg;
  }

  template <typename Char>
  auto get(basic_string_view<Char> name) const -> format_arg {
    int id = get_id(name);
    return id >= 0 ? get(id) : format_arg();
  }

  template <typename Char>
  auto get_id(basic_string_view<Char> name) const -> int {
    if (!has_named_args()) return -1;
    const auto& named_args =
        (is_packed() ? values_[-1] : args_[-1].value_).named_args;
    for (size_t i = 0; i < named_args.size; ++i) {
      if (named_args.data[i].name == name) return named_args.data[i].id;
    }
    return -1;
  }

  auto max_size() const -> int {
    unsigned long long max_packed = detail::max_packed_args;
    return static_cast<int>(is_packed() ? max_packed
                                        : desc_ & ~detail::is_unpacked_bit);
  }
};

/** An alias to ``basic_format_args<format_context>``. */
// A separate type would result in shorter symbols but break ABI compatibility
// between clang and gcc on ARM (#1919).
using format_args = basic_format_args<format_context>;

// We cannot use enum classes as bit fields because of a gcc bug
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61414.
namespace align {
enum type { none, left, right, center, numeric };
}
using align_t = align::type;
namespace sign {
enum type { none, minus, plus, space };
}
using sign_t = sign::type;

FMT_BEGIN_DETAIL_NAMESPACE

// Workaround an array initialization issue in gcc 4.8.
template <typename Char> struct fill_t {
 private:
  enum { max_size = 4 };
  Char data_[max_size] = {Char(' '), Char(0), Char(0), Char(0)};
  unsigned char size_ = 1;

 public:
  FMT_CONSTEXPR void operator=(basic_string_view<Char> s) {
    auto size = s.size();
    if (size > max_size) return throw_format_error("invalid fill");
    for (size_t i = 0; i < size; ++i) data_[i] = s[i];
    size_ = static_cast<unsigned char>(size);
  }

  constexpr auto size() const -> size_t { return size_; }
  constexpr auto data() const -> const Char* { return data_; }

  FMT_CONSTEXPR auto operator[](size_t index) -> Char& { return data_[index]; }
  FMT_CONSTEXPR auto operator[](size_t index) const -> const Char& {
    return data_[index];
  }
};
FMT_END_DETAIL_NAMESPACE

enum class presentation_type : unsigned char {
  none,
  // Integer types should go first,
  dec,             // 'd'
  oct,             // 'o'
  hex_lower,       // 'x'
  hex_upper,       // 'X'
  bin_lower,       // 'b'
  bin_upper,       // 'B'
  hexfloat_lower,  // 'a'
  hexfloat_upper,  // 'A'
  exp_lower,       // 'e'
  exp_upper,       // 'E'
  fixed_lower,     // 'f'
  fixed_upper,     // 'F'
  general_lower,   // 'g'
  general_upper,   // 'G'
  chr,             // 'c'
  string,          // 's'
  pointer          // 'p'
};

// Format specifiers for built-in and string types.
template <typename Char> struct basic_format_specs {
  int width;
  int precision;
  presentation_type type;
  align_t align : 4;
  sign_t sign : 3;
  bool alt : 1;  // Alternate form ('#').
  bool localized : 1;
  detail::fill_t<Char> fill;

  constexpr basic_format_specs()
      : width(0),
        precision(-1),
        type(presentation_type::none),
        align(align::none),
        sign(sign::none),
        alt(false),
        localized(false) {}
};

using format_specs = basic_format_specs<char>;

FMT_BEGIN_DETAIL_NAMESPACE

enum class arg_id_kind { none, index, name };

// An argument reference.
template <typename Char> struct arg_ref {
  FMT_CONSTEXPR arg_ref() : kind(arg_id_kind::none), val() {}

  FMT_CONSTEXPR explicit arg_ref(int index)
      : kind(arg_id_kind::index), val(index) {}
  FMT_CONSTEXPR explicit arg_ref(basic_string_view<Char> name)
      : kind(arg_id_kind::name), val(name) {}

  FMT_CONSTEXPR auto operator=(int idx) -> arg_ref& {
    kind = arg_id_kind::index;
    val.index = idx;
    return *this;
  }

  arg_id_kind kind;
  union value {
    FMT_CONSTEXPR value(int id = 0) : index{id} {}
    FMT_CONSTEXPR value(basic_string_view<Char> n) : name(n) {}

    int index;
    basic_string_view<Char> name;
  } val;
};

// Format specifiers with width and precision resolved at formatting rather
// than parsing time to allow re-using the same parsed specifiers with
// different sets of arguments (precompilation of format strings).
template <typename Char>
struct dynamic_format_specs : basic_format_specs<Char> {
  arg_ref<Char> width_ref;
  arg_ref<Char> precision_ref;
};

struct auto_id {};

// A format specifier handler that sets fields in basic_format_specs.
template <typename Char> class specs_setter {
 protected:
  basic_format_specs<Char>& specs_;

 public:
  explicit FMT_CONSTEXPR specs_setter(basic_format_specs<Char>& specs)
      : specs_(specs) {}

  FMT_CONSTEXPR specs_setter(const specs_setter& other)
      : specs_(other.specs_) {}

  FMT_CONSTEXPR void on_align(align_t align) { specs_.align = align; }
  FMT_CONSTEXPR void on_fill(basic_string_view<Char> fill) {
    specs_.fill = fill;
  }
  FMT_CONSTEXPR void on_sign(sign_t s) { specs_.sign = s; }
  FMT_CONSTEXPR void on_hash() { specs_.alt = true; }
  FMT_CONSTEXPR void on_localized() { specs_.localized = true; }

  FMT_CONSTEXPR void on_zero() {
    if (specs_.align == align::none) specs_.align = align::numeric;
    specs_.fill[0] = Char('0');
  }

  FMT_CONSTEXPR void on_width(int width) { specs_.width = width; }
  FMT_CONSTEXPR void on_precision(int precision) {
    specs_.precision = precision;
  }
  FMT_CONSTEXPR void end_precision() {}

  FMT_CONSTEXPR void on_type(presentation_type type) { specs_.type = type; }
};

// Format spec handler that saves references to arguments representing dynamic
// width and precision to be resolved at formatting time.
template <typename ParseContext>
class dynamic_specs_handler
    : public specs_setter<typename ParseContext::char_type> {
 public:
  using char_type = typename ParseContext::char_type;

  FMT_CONSTEXPR dynamic_specs_handler(dynamic_format_specs<char_type>& specs,
                                      ParseContext& ctx)
      : specs_setter<char_type>(specs), specs_(specs), context_(ctx) {}

  FMT_CONSTEXPR dynamic_specs_handler(const dynamic_specs_handler& other)
      : specs_setter<char_type>(other),
        specs_(other.specs_),
        context_(other.context_) {}

  template <typename Id> FMT_CONSTEXPR void on_dynamic_width(Id arg_id) {
    specs_.width_ref = make_arg_ref(arg_id);
  }

  template <typename Id> FMT_CONSTEXPR void on_dynamic_precision(Id arg_id) {
    specs_.precision_ref = make_arg_ref(arg_id);
  }

  FMT_CONSTEXPR void on_error(const char* message) {
    context_.on_error(message);
  }

 private:
  dynamic_format_specs<char_type>& specs_;
  ParseContext& context_;

  using arg_ref_type = arg_ref<char_type>;

  FMT_CONSTEXPR auto make_arg_ref(int arg_id) -> arg_ref_type {
    context_.check_arg_id(arg_id);
    return arg_ref_type(arg_id);
  }

  FMT_CONSTEXPR auto make_arg_ref(auto_id) -> arg_ref_type {
    return arg_ref_type(context_.next_arg_id());
  }

  FMT_CONSTEXPR auto make_arg_ref(basic_string_view<char_type> arg_id)
      -> arg_ref_type {
    context_.check_arg_id(arg_id);
    basic_string_view<char_type> format_str(
        context_.begin(), to_unsigned(context_.end() - context_.begin()));
    return arg_ref_type(arg_id);
  }
};

template <typename Char> constexpr bool is_ascii_letter(Char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Converts a character to ASCII. Returns a number > 127 on conversion failure.
template <typename Char, FMT_ENABLE_IF(std::is_integral<Char>::value)>
constexpr auto to_ascii(Char value) -> Char {
  return value;
}
template <typename Char, FMT_ENABLE_IF(std::is_enum<Char>::value)>
constexpr auto to_ascii(Char value) ->
    typename std::underlying_type<Char>::type {
  return value;
}

template <typename Char>
FMT_CONSTEXPR auto code_point_length(const Char* begin) -> int {
  if (const_check(sizeof(Char) != 1)) return 1;
  auto lengths =
      "\1\1\1\1\1\1\1\1\1\1\1\1\1\1\1\1\0\0\0\0\0\0\0\0\2\2\2\2\3\3\4";
  int len = lengths[static_cast<unsigned char>(*begin) >> 3];

  // Compute the pointer to the next character early so that the next
  // iteration can start working on the next character. Neither Clang
  // nor GCC figure out this reordering on their own.
  return len + !len;
}

// Return the result via the out param to workaround gcc bug 77539.
template <bool IS_CONSTEXPR, typename T, typename Ptr = const T*>
FMT_CONSTEXPR auto find(Ptr first, Ptr last, T value, Ptr& out) -> bool {
  for (out = first; out != last; ++out) {
    if (*out == value) return true;
  }
  return false;
}

template <>
inline auto find<false, char>(const char* first, const char* last, char value,
                              const char*& out) -> bool {
  out = static_cast<const char*>(
      std::memchr(first, value, to_unsigned(last - first)));
  return out != nullptr;
}

// Parses the range [begin, end) as an unsigned integer. This function assumes
// that the range is non-empty and the first character is a digit.
template <typename Char>
FMT_CONSTEXPR auto parse_nonnegative_int(const Char*& begin, const Char* end,
                                         int error_value) noexcept -> int {
  FMT_ASSERT(begin != end && '0' <= *begin && *begin <= '9', "");
  unsigned value = 0, prev = 0;
  auto p = begin;
  do {
    prev = value;
    value = value * 10 + unsigned(*p - '0');
    ++p;
  } while (p != end && '0' <= *p && *p <= '9');
  auto num_digits = p - begin;
  begin = p;
  if (num_digits <= std::numeric_limits<int>::digits10)
    return static_cast<int>(value);
  // Check for overflow.
  const unsigned max = to_unsigned((std::numeric_limits<int>::max)());
  return num_digits == std::numeric_limits<int>::digits10 + 1 &&
                 prev * 10ull + unsigned(p[-1] - '0') <= max
             ? static_cast<int>(value)
             : error_value;
}

// Parses fill and alignment.
template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_align(const Char* begin, const Char* end,
                               Handler&& handler) -> const Char* {
  FMT_ASSERT(begin != end, "");
  auto align = align::none;
  auto p = begin + code_point_length(begin);
  if (p >= end) p = begin;
  for (;;) {
    switch (to_ascii(*p)) {
    case '<':
      align = align::left;
      break;
    case '>':
      align = align::right;
      break;
    case '^':
      align = align::center;
      break;
    default:
      break;
    }
    if (align != align::none) {
      if (p != begin) {
        auto c = *begin;
        if (c == '{')
          return handler.on_error("invalid fill character '{'"), begin;
        handler.on_fill(basic_string_view<Char>(begin, to_unsigned(p - begin)));
        begin = p + 1;
      } else
        ++begin;
      handler.on_align(align);
      break;
    } else if (p == begin) {
      break;
    }
    p = begin;
  }
  return begin;
}

template <typename Char> FMT_CONSTEXPR bool is_name_start(Char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || '_' == c;
}

template <typename Char, typename IDHandler>
FMT_CONSTEXPR auto do_parse_arg_id(const Char* begin, const Char* end,
                                   IDHandler&& handler) -> const Char* {
  FMT_ASSERT(begin != end, "");
  Char c = *begin;
  if (c >= '0' && c <= '9') {
    int index = 0;
    if (c != '0')
      index =
          parse_nonnegative_int(begin, end, (std::numeric_limits<int>::max)());
    else
      ++begin;
    if (begin == end || (*begin != '}' && *begin != ':'))
      handler.on_error("invalid format string");
    else
      handler(index);
    return begin;
  }
  if (!is_name_start(c)) {
    handler.on_error("invalid format string");
    return begin;
  }
  auto it = begin;
  do {
    ++it;
  } while (it != end && (is_name_start(c = *it) || ('0' <= c && c <= '9')));
  handler(basic_string_view<Char>(begin, to_unsigned(it - begin)));
  return it;
}

template <typename Char, typename IDHandler>
FMT_CONSTEXPR FMT_INLINE auto parse_arg_id(const Char* begin, const Char* end,
                                           IDHandler&& handler) -> const Char* {
  Char c = *begin;
  if (c != '}' && c != ':') return do_parse_arg_id(begin, end, handler);
  handler();
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_width(const Char* begin, const Char* end,
                               Handler&& handler) -> const Char* {
  using detail::auto_id;
  struct width_adapter {
    Handler& handler;

    FMT_CONSTEXPR void operator()() { handler.on_dynamic_width(auto_id()); }
    FMT_CONSTEXPR void operator()(int id) { handler.on_dynamic_width(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      handler.on_dynamic_width(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  FMT_ASSERT(begin != end, "");
  if ('0' <= *begin && *begin <= '9') {
    int width = parse_nonnegative_int(begin, end, -1);
    if (width != -1)
      handler.on_width(width);
    else
      handler.on_error("number is too big");
  } else if (*begin == '{') {
    ++begin;
    if (begin != end) begin = parse_arg_id(begin, end, width_adapter{handler});
    if (begin == end || *begin != '}')
      return handler.on_error("invalid format string"), begin;
    ++begin;
  }
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_precision(const Char* begin, const Char* end,
                                   Handler&& handler) -> const Char* {
  using detail::auto_id;
  struct precision_adapter {
    Handler& handler;

    FMT_CONSTEXPR void operator()() { handler.on_dynamic_precision(auto_id()); }
    FMT_CONSTEXPR void operator()(int id) { handler.on_dynamic_precision(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      handler.on_dynamic_precision(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  ++begin;
  auto c = begin != end ? *begin : Char();
  if ('0' <= c && c <= '9') {
    auto precision = parse_nonnegative_int(begin, end, -1);
    if (precision != -1)
      handler.on_precision(precision);
    else
      handler.on_error("number is too big");
  } else if (c == '{') {
    ++begin;
    if (begin != end)
      begin = parse_arg_id(begin, end, precision_adapter{handler});
    if (begin == end || *begin++ != '}')
      return handler.on_error("invalid format string"), begin;
  } else {
    return handler.on_error("missing precision specifier"), begin;
  }
  handler.end_precision();
  return begin;
}

template <typename Char>
FMT_CONSTEXPR auto parse_presentation_type(Char type) -> presentation_type {
  switch (to_ascii(type)) {
  case 'd':
    return presentation_type::dec;
  case 'o':
    return presentation_type::oct;
  case 'x':
    return presentation_type::hex_lower;
  case 'X':
    return presentation_type::hex_upper;
  case 'b':
    return presentation_type::bin_lower;
  case 'B':
    return presentation_type::bin_upper;
  case 'a':
    return presentation_type::hexfloat_lower;
  case 'A':
    return presentation_type::hexfloat_upper;
  case 'e':
    return presentation_type::exp_lower;
  case 'E':
    return presentation_type::exp_upper;
  case 'f':
    return presentation_type::fixed_lower;
  case 'F':
    return presentation_type::fixed_upper;
  case 'g':
    return presentation_type::general_lower;
  case 'G':
    return presentation_type::general_upper;
  case 'c':
    return presentation_type::chr;
  case 's':
    return presentation_type::string;
  case 'p':
    return presentation_type::pointer;
  default:
    return presentation_type::none;
  }
}

// Parses standard format specifiers and sends notifications about parsed
// components to handler.
template <typename Char, typename SpecHandler>
FMT_CONSTEXPR FMT_INLINE auto parse_format_specs(const Char* begin,
                                                 const Char* end,
                                                 SpecHandler&& handler)
    -> const Char* {
  if (1 < end - begin && begin[1] == '}' && is_ascii_letter(*begin) &&
      *begin != 'L') {
    presentation_type type = parse_presentation_type(*begin++);
    if (type == presentation_type::none)
      handler.on_error("invalid type specifier");
    handler.on_type(type);
    return begin;
  }

  if (begin == end) return begin;

  begin = parse_align(begin, end, handler);
  if (begin == end) return begin;

  // Parse sign.
  switch (to_ascii(*begin)) {
  case '+':
    handler.on_sign(sign::plus);
    ++begin;
    break;
  case '-':
    handler.on_sign(sign::minus);
    ++begin;
    break;
  case ' ':
    handler.on_sign(sign::space);
    ++begin;
    break;
  default:
    break;
  }
  if (begin == end) return begin;

  if (*begin == '#') {
    handler.on_hash();
    if (++begin == end) return begin;
  }

  // Parse zero flag.
  if (*begin == '0') {
    handler.on_zero();
    if (++begin == end) return begin;
  }

  begin = parse_width(begin, end, handler);
  if (begin == end) return begin;

  // Parse precision.
  if (*begin == '.') {
    begin = parse_precision(begin, end, handler);
    if (begin == end) return begin;
  }

  if (*begin == 'L') {
    handler.on_localized();
    ++begin;
  }

  // Parse type.
  if (begin != end && *begin != '}') {
    presentation_type type = parse_presentation_type(*begin++);
    if (type == presentation_type::none)
      handler.on_error("invalid type specifier");
    handler.on_type(type);
  }
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_replacement_field(const Char* begin, const Char* end,
                                           Handler&& handler) -> const Char* {
  struct id_adapter {
    Handler& handler;
    int arg_id;

    FMT_CONSTEXPR void operator()() { arg_id = handler.on_arg_id(); }
    FMT_CONSTEXPR void operator()(int id) { arg_id = handler.on_arg_id(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      arg_id = handler.on_arg_id(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  ++begin;
  if (begin == end) return handler.on_error("invalid format string"), end;
  if (*begin == '}') {
    handler.on_replacement_field(handler.on_arg_id(), begin);
  } else if (*begin == '{') {
    handler.on_text(begin, begin + 1);
  } else {
    auto adapter = id_adapter{handler, 0};
    begin = parse_arg_id(begin, end, adapter);
    Char c = begin != end ? *begin : Char();
    if (c == '}') {
      handler.on_replacement_field(adapter.arg_id, begin);
    } else if (c == ':') {
      begin = handler.on_format_specs(adapter.arg_id, begin + 1, end);
      if (begin == end || *begin != '}')
        return handler.on_error("unknown format specifier"), end;
    } else {
      return handler.on_error("missing '}' in format string"), end;
    }
  }
  return begin + 1;
}

template <bool IS_CONSTEXPR, typename Char, typename Handler>
FMT_CONSTEXPR FMT_INLINE void parse_format_string(
    basic_string_view<Char> format_str, Handler&& handler) {
  // Workaround a name-lookup bug in MSVC's modules implementation.
  using detail::find;

  auto begin = format_str.data();
  auto end = begin + format_str.size();
  if (end - begin < 32) {
    // Use a simple loop instead of memchr for small strings.
    const Char* p = begin;
    while (p != end) {
      auto c = *p++;
      if (c == '{') {
        handler.on_text(begin, p - 1);
        begin = p = parse_replacement_field(p - 1, end, handler);
      } else if (c == '}') {
        if (p == end || *p != '}')
          return handler.on_error("unmatched '}' in format string");
        handler.on_text(begin, p);
        begin = ++p;
      }
    }
    handler.on_text(begin, end);
    return;
  }
  struct writer {
    FMT_CONSTEXPR void operator()(const Char* pbegin, const Char* pend) {
      if (pbegin == pend) return;
      for (;;) {
        const Char* p = nullptr;
        if (!find<IS_CONSTEXPR>(pbegin, pend, Char('}'), p))
          return handler_.on_text(pbegin, pend);
        ++p;
        if (p == pend || *p != '}')
          return handler_.on_error("unmatched '}' in format string");
        handler_.on_text(pbegin, p);
        pbegin = p + 1;
      }
    }
    Handler& handler_;
  } write{handler};
  while (begin != end) {
    // Doing two passes with memchr (one for '{' and another for '}') is up to
    // 2.5x faster than the naive one-pass implementation on big format strings.
    const Char* p = begin;
    if (*begin != '{' && !find<IS_CONSTEXPR>(begin + 1, end, Char('{'), p))
      return write(begin, end);
    write(begin, p);
    begin = parse_replacement_field(p, end, handler);
  }
}

template <typename T, typename ParseContext>
FMT_CONSTEXPR auto parse_format_specs(ParseContext& ctx)
    -> decltype(ctx.begin()) {
  using char_type = typename ParseContext::char_type;
  using context = buffer_context<char_type>;
  using mapped_type = conditional_t<
      mapped_type_constant<T, context>::value != type::custom_type,
      decltype(arg_mapper<context>().map(std::declval<const T&>())), T>;
  auto f = conditional_t<has_formatter<mapped_type, context>::value,
                         formatter<mapped_type, char_type>,
                         fallback_formatter<T, char_type>>();
  return f.parse(ctx);
}

// A parse context with extra argument id checks. It is only used at compile
// time because adding checks at runtime would introduce substantial overhead
// and would be redundant since argument ids are checked when arguments are
// retrieved anyway.
template <typename Char, typename ErrorHandler = error_handler>
class compile_parse_context
    : public basic_format_parse_context<Char, ErrorHandler> {
 private:
  int num_args_;
  using base = basic_format_parse_context<Char, ErrorHandler>;

 public:
  explicit FMT_CONSTEXPR compile_parse_context(
      basic_string_view<Char> format_str,
      int num_args = (std::numeric_limits<int>::max)(), ErrorHandler eh = {})
      : base(format_str, eh), num_args_(num_args) {}

  FMT_CONSTEXPR auto next_arg_id() -> int {
    int id = base::next_arg_id();
    if (id >= num_args_) this->on_error("argument not found");
    return id;
  }

  FMT_CONSTEXPR void check_arg_id(int id) {
    base::check_arg_id(id);
    if (id >= num_args_) this->on_error("argument not found");
  }
  using base::check_arg_id;
};

template <typename ErrorHandler>
FMT_CONSTEXPR void check_int_type_spec(presentation_type type,
                                       ErrorHandler&& eh) {
  if (type > presentation_type::bin_upper && type != presentation_type::chr)
    eh.on_error("invalid type specifier");
}

// Checks char specs and returns true if the type spec is char (and not int).
template <typename Char, typename ErrorHandler = error_handler>
FMT_CONSTEXPR auto check_char_specs(const basic_format_specs<Char>& specs,
                                    ErrorHandler&& eh = {}) -> bool {
  if (specs.type != presentation_type::none &&
      specs.type != presentation_type::chr) {
    check_int_type_spec(specs.type, eh);
    return false;
  }
  if (specs.align == align::numeric || specs.sign != sign::none || specs.alt)
    eh.on_error("invalid format specifier for char");
  return true;
}

// A floating-point presentation format.
enum class float_format : unsigned char {
  general,  // General: exponent notation or fixed point based on magnitude.
  exp,      // Exponent notation with the default precision of 6, e.g. 1.2e-3.
  fixed,    // Fixed point with the default precision of 6, e.g. 0.0012.
  hex
};

struct float_specs {
  int precision;
  float_format format : 8;
  sign_t sign : 8;
  bool upper : 1;
  bool locale : 1;
  bool binary32 : 1;
  bool fallback : 1;
  bool showpoint : 1;
};

template <typename ErrorHandler = error_handler, typename Char>
FMT_CONSTEXPR auto parse_float_type_spec(const basic_format_specs<Char>& specs,
                                         ErrorHandler&& eh = {})
    -> float_specs {
  auto result = float_specs();
  result.showpoint = specs.alt;
  result.locale = specs.localized;
  switch (specs.type) {
  case presentation_type::none:
    result.format = float_format::general;
    break;
  case presentation_type::general_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::general_lower:
    result.format = float_format::general;
    break;
  case presentation_type::exp_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::exp_lower:
    result.format = float_format::exp;
    result.showpoint |= specs.precision != 0;
    break;
  case presentation_type::fixed_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::fixed_lower:
    result.format = float_format::fixed;
    result.showpoint |= specs.precision != 0;
    break;
  case presentation_type::hexfloat_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::hexfloat_lower:
    result.format = float_format::hex;
    break;
  default:
    eh.on_error("invalid type specifier");
    break;
  }
  return result;
}

template <typename ErrorHandler = error_handler>
FMT_CONSTEXPR auto check_cstring_type_spec(presentation_type type,
                                           ErrorHandler&& eh = {}) -> bool {
  if (type == presentation_type::none || type == presentation_type::string)
    return true;
  if (type != presentation_type::pointer) eh.on_error("invalid type specifier");
  return false;
}

template <typename ErrorHandler = error_handler>
FMT_CONSTEXPR void check_string_type_spec(presentation_type type,
                                          ErrorHandler&& eh = {}) {
  if (type != presentation_type::none && type != presentation_type::string)
    eh.on_error("invalid type specifier");
}

template <typename ErrorHandler>
FMT_CONSTEXPR void check_pointer_type_spec(presentation_type type,
                                           ErrorHandler&& eh) {
  if (type != presentation_type::none && type != presentation_type::pointer)
    eh.on_error("invalid type specifier");
}

// A parse_format_specs handler that checks if specifiers are consistent with
// the argument type.
template <typename Handler> class specs_checker : public Handler {
 private:
  detail::type arg_type_;

  FMT_CONSTEXPR void require_numeric_argument() {
    if (!is_arithmetic_type(arg_type_))
      this->on_error("format specifier requires numeric argument");
  }

 public:
  FMT_CONSTEXPR specs_checker(const Handler& handler, detail::type arg_type)
      : Handler(handler), arg_type_(arg_type) {}

  FMT_CONSTEXPR void on_align(align_t align) {
    if (align == align::numeric) require_numeric_argument();
    Handler::on_align(align);
  }

  FMT_CONSTEXPR void on_sign(sign_t s) {
    require_numeric_argument();
    if (is_integral_type(arg_type_) && arg_type_ != type::int_type &&
        arg_type_ != type::long_long_type && arg_type_ != type::char_type) {
      this->on_error("format specifier requires signed argument");
    }
    Handler::on_sign(s);
  }

  FMT_CONSTEXPR void on_hash() {
    require_numeric_argument();
    Handler::on_hash();
  }

  FMT_CONSTEXPR void on_localized() {
    require_numeric_argument();
    Handler::on_localized();
  }

  FMT_CONSTEXPR void on_zero() {
    require_numeric_argument();
    Handler::on_zero();
  }

  FMT_CONSTEXPR void end_precision() {
    if (is_integral_type(arg_type_) || arg_type_ == type::pointer_type)
      this->on_error("precision not allowed for this argument type");
  }
};

constexpr int invalid_arg_index = -1;

#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <int N, typename T, typename... Args, typename Char>
constexpr auto get_arg_index_by_name(basic_string_view<Char> name) -> int {
  if constexpr (detail::is_statically_named_arg<T>()) {
    if (name == T::name) return N;
  }
  if constexpr (sizeof...(Args) > 0)
    return get_arg_index_by_name<N + 1, Args...>(name);
  (void)name;  // Workaround an MSVC bug about "unused" parameter.
  return invalid_arg_index;
}
#endif

template <typename... Args, typename Char>
FMT_CONSTEXPR auto get_arg_index_by_name(basic_string_view<Char> name) -> int {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
  if constexpr (sizeof...(Args) > 0)
    return get_arg_index_by_name<0, Args...>(name);
#endif
  (void)name;
  return invalid_arg_index;
}

template <typename Char, typename ErrorHandler, typename... Args>
class format_string_checker {
 private:
  using parse_context_type = compile_parse_context<Char, ErrorHandler>;
  enum { num_args = sizeof...(Args) };

  // Format specifier parsing function.
  using parse_func = const Char* (*)(parse_context_type&);

  parse_context_type context_;
  parse_func parse_funcs_[num_args > 0 ? num_args : 1];

 public:
  explicit FMT_CONSTEXPR format_string_checker(
      basic_string_view<Char> format_str, ErrorHandler eh)
      : context_(format_str, num_args, eh),
        parse_funcs_{&parse_format_specs<Args, parse_context_type>...} {}

  FMT_CONSTEXPR void on_text(const Char*, const Char*) {}

  FMT_CONSTEXPR auto on_arg_id() -> int { return context_.next_arg_id(); }
  FMT_CONSTEXPR auto on_arg_id(int id) -> int {
    return context_.check_arg_id(id), id;
  }
  FMT_CONSTEXPR auto on_arg_id(basic_string_view<Char> id) -> int {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
    auto index = get_arg_index_by_name<Args...>(id);
    if (index == invalid_arg_index) on_error("named argument is not found");
    return context_.check_arg_id(index), index;
#else
    (void)id;
    on_error("compile-time checks for named arguments require C++20 support");
    return 0;
#endif
  }

  FMT_CONSTEXPR void on_replacement_field(int, const Char*) {}

  FMT_CONSTEXPR auto on_format_specs(int id, const Char* begin, const Char*)
      -> const Char* {
    context_.advance_to(context_.begin() + (begin - &*context_.begin()));
    // id >= 0 check is a workaround for gcc 10 bug (#2065).
    return id >= 0 && id < num_args ? parse_funcs_[id](context_) : begin;
  }

  FMT_CONSTEXPR void on_error(const char* message) {
    context_.on_error(message);
  }
};

template <typename... Args, typename S,
          enable_if_t<(is_compile_string<S>::value), int>>
void check_format_string(S format_str) {
  FMT_CONSTEXPR auto s = to_string_view(format_str);
  using checker = format_string_checker<typename S::char_type, error_handler,
                                        remove_cvref_t<Args>...>;
  FMT_CONSTEXPR bool invalid_format =
      (parse_format_string<true>(s, checker(s, {})), true);
  ignore_unused(invalid_format);
}

template <typename Char>
void vformat_to(
    buffer<Char>& buf, basic_string_view<Char> fmt,
    basic_format_args<FMT_BUFFER_CONTEXT(type_identity_t<Char>)> args,
    locale_ref loc = {});

FMT_API void vprint_mojibake(std::FILE*, string_view, format_args);
#ifndef _WIN32
inline void vprint_mojibake(std::FILE*, string_view, format_args) {}
#endif
FMT_END_DETAIL_NAMESPACE

// A formatter specialization for the core types corresponding to detail::type
// constants.
template <typename T, typename Char>
struct formatter<T, Char,
                 enable_if_t<detail::type_constant<T, Char>::value !=
                             detail::type::custom_type>> {
 private:
  detail::dynamic_format_specs<Char> specs_;

 public:
  // Parses format specifiers stopping either at the end of the range or at the
  // terminating '}'.
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto begin = ctx.begin(), end = ctx.end();
    if (begin == end) return begin;
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    auto type = detail::type_constant<T, Char>::value;
    auto checker =
        detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
    auto it = detail::parse_format_specs(begin, end, checker);
    auto eh = ctx.error_handler();
    switch (type) {
    case detail::type::none_type:
      FMT_ASSERT(false, "invalid argument type");
      break;
    case detail::type::bool_type:
      if (specs_.type == presentation_type::none ||
          specs_.type == presentation_type::string) {
        break;
      }
      FMT_FALLTHROUGH;
    case detail::type::int_type:
    case detail::type::uint_type:
    case detail::type::long_long_type:
    case detail::type::ulong_long_type:
    case detail::type::int128_type:
    case detail::type::uint128_type:
      detail::check_int_type_spec(specs_.type, eh);
      break;
    case detail::type::char_type:
      detail::check_char_specs(specs_, eh);
      break;
    case detail::type::float_type:
      if (detail::const_check(FMT_USE_FLOAT))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "float support disabled");
      break;
    case detail::type::double_type:
      if (detail::const_check(FMT_USE_DOUBLE))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "double support disabled");
      break;
    case detail::type::long_double_type:
      if (detail::const_check(FMT_USE_LONG_DOUBLE))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "long double support disabled");
      break;
    case detail::type::cstring_type:
      detail::check_cstring_type_spec(specs_.type, eh);
      break;
    case detail::type::string_type:
      detail::check_string_type_spec(specs_.type, eh);
      break;
    case detail::type::pointer_type:
      detail::check_pointer_type_spec(specs_.type, eh);
      break;
    case detail::type::custom_type:
      // Custom format specifiers are checked in parse functions of
      // formatter specializations.
      break;
    }
    return it;
  }

  template <typename FormatContext>
  FMT_CONSTEXPR auto format(const T& val, FormatContext& ctx) const
      -> decltype(ctx.out());
};

template <typename Char> struct basic_runtime { basic_string_view<Char> str; };

/** A compile-time format string. */
template <typename Char, typename... Args> class basic_format_string {
 private:
  basic_string_view<Char> str_;

 public:
  template <typename S,
            FMT_ENABLE_IF(
                std::is_convertible<const S&, basic_string_view<Char>>::value)>
  FMT_CONSTEVAL FMT_INLINE basic_format_string(const S& s) : str_(s) {
    static_assert(
        detail::count<
            (std::is_base_of<detail::view, remove_reference_t<Args>>::value &&
             std::is_reference<Args>::value)...>() == 0,
        "passing views as lvalues is disallowed");
#ifdef FMT_HAS_CONSTEVAL
    if constexpr (detail::count_named_args<Args...>() ==
                  detail::count_statically_named_args<Args...>()) {
      using checker = detail::format_string_checker<Char, detail::error_handler,
                                                    remove_cvref_t<Args>...>;
      detail::parse_format_string<true>(str_, checker(s, {}));
    }
#else
    detail::check_format_string<Args...>(s);
#endif
  }
  basic_format_string(basic_runtime<Char> r) : str_(r.str) {}

  FMT_INLINE operator basic_string_view<Char>() const { return str_; }
};

#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
// Workaround broken conversion on older gcc.
template <typename... Args> using format_string = string_view;
template <typename S> auto runtime(const S& s) -> basic_string_view<char_t<S>> {
  return s;
}
#else
template <typename... Args>
using format_string = basic_format_string<char, type_identity_t<Args>...>;
/**
  \rst
  Creates a runtime format string.

  **Example**::

    // Check format string at runtime instead of compile-time.
    fmt::print(fmt::runtime("{:d}"), "I am not a number");
  \endrst
 */
template <typename S> auto runtime(const S& s) -> basic_runtime<char_t<S>> {
  return {{s}};
}
#endif

FMT_API auto vformat(string_view fmt, format_args args) -> std::string;

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and returns the result
  as a string.

  **Example**::

    #include <fmt/core.h>
    std::string message = fmt::format("The answer is {}.", 42);
  \endrst
*/
template <typename... T>
FMT_NODISCARD FMT_INLINE auto format(format_string<T...> fmt, T&&... args)
    -> std::string {
  return vformat(fmt, fmt::make_format_args(args...));
}

/** Formats a string and writes the output to ``out``. */
template <typename OutputIt,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
auto vformat_to(OutputIt out, string_view fmt, format_args args) -> OutputIt {
  using detail::get_buffer;
  auto&& buf = get_buffer<char>(out);
  detail::vformat_to(buf, fmt, args, {});
  return detail::get_iterator(buf);
}

/**
 \rst
 Formats ``args`` according to specifications in ``fmt``, writes the result to
 the output iterator ``out`` and returns the iterator past the end of the output
 range. `format_to` does not append a terminating null character.

 **Example**::

   auto out = std::vector<char>();
   fmt::format_to(std::back_inserter(out), "{}", 42);
 \endrst
 */
template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
FMT_INLINE auto format_to(OutputIt out, format_string<T...> fmt, T&&... args)
    -> OutputIt {
  return vformat_to(out, fmt, fmt::make_format_args(args...));
}

template <typename OutputIt> struct format_to_n_result {
  /** Iterator past the end of the output range. */
  OutputIt out;
  /** Total (not truncated) output size. */
  size_t size;
};

template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
auto vformat_to_n(OutputIt out, size_t n, string_view fmt, format_args args)
    -> format_to_n_result<OutputIt> {
  using traits = detail::fixed_buffer_traits;
  auto buf = detail::iterator_buffer<OutputIt, char, traits>(out, n);
  detail::vformat_to(buf, fmt, args, {});
  return {buf.out(), buf.count()};
}

/**
  \rst
  Formats ``args`` according to specifications in ``fmt``, writes up to ``n``
  characters of the result to the output iterator ``out`` and returns the total
  (not truncated) output size and the iterator past the end of the output range.
  `format_to_n` does not append a terminating null character.
  \endrst
 */
template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
FMT_INLINE auto format_to_n(OutputIt out, size_t n, format_string<T...> fmt,
                            T&&... args) -> format_to_n_result<OutputIt> {
  return vformat_to_n(out, n, fmt, fmt::make_format_args(args...));
}

/** Returns the number of chars in the output of ``format(fmt, args...)``. */
template <typename... T>
FMT_NODISCARD FMT_INLINE auto formatted_size(format_string<T...> fmt,
                                             T&&... args) -> size_t {
  auto buf = detail::counting_buffer<>();
  detail::vformat_to(buf, string_view(fmt), fmt::make_format_args(args...), {});
  return buf.count();
}

FMT_API void vprint(string_view fmt, format_args args);
FMT_API void vprint(std::FILE* f, string_view fmt, format_args args);

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and writes the output
  to ``stdout``.

  **Example**::

    fmt::print("Elapsed time: {0:.2f} seconds", 1.23);
  \endrst
 */
template <typename... T>
FMT_INLINE void print(format_string<T...> fmt, T&&... args) {
  const auto& vargs = fmt::make_format_args(args...);
  return detail::is_utf8() ? vprint(fmt, vargs)
                           : detail::vprint_mojibake(stdout, fmt, vargs);
}

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and writes the
  output to the file ``f``.

  **Example**::

    fmt::print(stderr, "Don't {}!", "panic");
  \endrst
 */
template <typename... T>
FMT_INLINE void print(std::FILE* f, format_string<T...> fmt, T&&... args) {
  const auto& vargs = fmt::make_format_args(args...);
  return detail::is_utf8() ? vprint(f, fmt, vargs)
                           : detail::vprint_mojibake(f, fmt, vargs);
}

FMT_MODULE_EXPORT_END
FMT_GCC_PRAGMA("GCC pop_options")
FMT_END_NAMESPACE

#ifdef FMT_HEADER_ONLY
#  include "format.h"
#endif
#endif  // FMT_CORE_H_
#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

namespace gr {

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

static constexpr const std::size_t kCacheLine
        = 64; // waiting for clang: std::hardware_destructive_interference_size
static constexpr const std::int64_t kInitialCursorValue = -1L;

/**
 * Concurrent sequence class used for tracking the progress of the ring buffer and event
 * processors. Support a number of concurrent operations including CAS and order writes.
 * Also attempts to be more efficient with regards to false sharing by adding padding
 * around the volatile field.
 */
// clang-format off
class Sequence
{
    alignas(kCacheLine) std::atomic<std::int64_t> _fieldsValue{};

public:
    Sequence(const Sequence&) = delete;
    Sequence(const Sequence&&) = delete;
    void operator=(const Sequence&) = delete;
    explicit Sequence(std::int64_t initialValue = kInitialCursorValue) noexcept
        : _fieldsValue(initialValue)
    {
    }

    [[nodiscard]] forceinline std::int64_t value() const noexcept
    {
        return std::atomic_load_explicit(&_fieldsValue, std::memory_order_acquire);
    }

    forceinline void setValue(const std::int64_t value) noexcept
    {
        std::atomic_store_explicit(&_fieldsValue, value, std::memory_order_release);
    }

    [[nodiscard]] forceinline bool compareAndSet(std::int64_t expectedSequence,
                                                 std::int64_t nextSequence) noexcept
    {
        // atomically set the value to the given updated value if the current value == the
        // expected value (true, otherwise folse).
        return std::atomic_compare_exchange_strong(
            &_fieldsValue, &expectedSequence, nextSequence);
    }

    [[nodiscard]] forceinline std::int64_t incrementAndGet() noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, 1L) + 1L;
    }

    [[nodiscard]] forceinline std::int64_t addAndGet(std::int64_t value) noexcept
    {
        return std::atomic_fetch_add(&_fieldsValue, value) + value;
    }
};

namespace detail {
/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum an initial default minimum.  If the array is empty this value will
 * returned. \returns the minimum sequence found or lon.MaxValue if the array is empty.
 */
inline std::int64_t getMinimumSequence(
    const std::vector<std::shared_ptr<Sequence>>& sequences,
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max()) noexcept
{
    if (sequences.empty()) {
        return minimum;
    }
#if not defined(_LIBCPP_VERSION)
    return std::min(minimum, std::ranges::min(sequences, std::less{}, [](const auto &sequence) noexcept { return sequence->value(); })->value());
#else
    std::vector<int64_t> v(sequences.size());
    std::transform(sequences.cbegin(), sequences.cend(), v.begin(), [](auto val) { return val->value(); });
    auto min = std::min(v.begin(), v.end());
    return std::min(*min, minimum);
#endif
}

inline void addSequences(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences,
             const Sequence& cursor,
             const std::vector<std::shared_ptr<Sequence>>& sequencesToAdd)
{
    std::int64_t cursorSequence;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> updatedSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> currentSequences;

    do {
        currentSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
        updatedSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(currentSequences->size() + sequencesToAdd.size());

#if not defined(_LIBCPP_VERSION)
        std::ranges::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#else
        std::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#endif

        cursorSequence = cursor.value();

        auto index = currentSequences->size();
        for (auto&& sequence : sequencesToAdd) {
            sequence->setValue(cursorSequence);
            (*updatedSequences)[index] = sequence;
            index++;
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &currentSequences, updatedSequences)); // xTODO: explicit memory order

    cursorSequence = cursor.value();

    for (auto&& sequence : sequencesToAdd) {
        sequence->setValue(cursorSequence);
    }
}

inline bool removeSequence(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences, const std::shared_ptr<Sequence>& sequence)
{
    std::uint32_t numToRemove;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> oldSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> newSequences;

    do {
        oldSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
#if not defined(_LIBCPP_VERSION)
        numToRemove = static_cast<std::uint32_t>(std::ranges::count(*oldSequences, sequence)); // specifically uses identity
#else
        numToRemove = static_cast<std::uint32_t>(std::count((*oldSequences).begin(), (*oldSequences).end(), sequence)); // specifically uses identity
#endif
        if (numToRemove == 0) {
            break;
        }

        auto oldSize = static_cast<std::uint32_t>(oldSequences->size());
        newSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(
            oldSize - numToRemove);

        for (auto i = 0U, pos = 0U; i < oldSize; ++i) {
            const auto& testSequence = (*oldSequences)[i];
            if (sequence != testSequence) {
                (*newSequences)[pos] = testSequence;
                pos++;
            }
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &oldSequences, newSequences));

    return numToRemove != 0;
}

// clang-format on

} // namespace detail

} // namespace gr

#ifdef FMT_FORMAT_H_
#include <fmt/core.h>
#include <fmt/ostream.h>

template<>
struct fmt::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto
    format(gr::Sequence const &value, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream &
operator<<(std::ostream &os, const Sequence &v) {
    return os << fmt::format("{}", v);
}
} // namespace gr

#endif // FMT_FORMAT_H_

#endif // GNURADIO_SEQUENCE_HPP
#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#define GNURADIO_CLAIM_STRATEGY_HPP

#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef GNURADIO_SEQUENCE_HPP
#endif
#ifndef GNURADIO_WAIT_STRATEGY_HPP
#endif

namespace gr {

struct NoCapacityException : public std::runtime_error {
    NoCapacityException() : std::runtime_error("NoCapacityException"){};
};

// clang-format off

template<typename T>
concept ClaimStrategy = requires(T /*const*/ t, const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity,
        const std::int64_t cursorValue, const std::int64_t sequence, const std::int64_t availableSequence, const std::int32_t n_slots_to_claim) {
    { t.hasAvailableCapacity(dependents, requiredCapacity, cursorValue) } -> std::same_as<bool>;
    { t.next(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.tryNext(dependents, n_slots_to_claim) } -> std::same_as<std::int64_t>;
    { t.getRemainingCapacity(dependents) } -> std::same_as<std::int64_t>;
    { t.publish(sequence) } -> std::same_as<void>;
    { t.isAvailable(sequence) } -> std::same_as<bool>;
    { t.getHighestPublishedSequence(sequence, availableSequence) } -> std::same_as<std::int64_t>;
};

namespace claim_strategy::util {
constexpr unsigned    floorlog2(std::size_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }
constexpr unsigned    ceillog2(std::size_t x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }
}

template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class alignas(kCacheLine) SingleThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::int64_t _nextValue{ kInitialCursorValue }; // N.B. no need for atomics since this is called by a single publisher
    alignas(kCacheLine) mutable std::int64_t _cachedValue{ kInitialCursorValue };

public:
    SingleThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy){};
    SingleThreadedStrategy(const SingleThreadedStrategy &)  = delete;
    SingleThreadedStrategy(const SingleThreadedStrategy &&) = delete;
    void operator=(const SingleThreadedStrategy &) = delete;

    bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const int requiredCapacity, const std::int64_t /*cursorValue*/) const noexcept {
        if (const std::int64_t wrapPoint = (_nextValue + requiredCapacity) - static_cast<std::int64_t>(_size); wrapPoint > _cachedValue || _cachedValue > _nextValue) {
            auto minSequence = detail::getMinimumSequence(dependents, _nextValue);
            _cachedValue     = minSequence;
            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int32_t n_slots_to_claim = 1) noexcept {
        assert((n_slots_to_claim > 0 && n_slots_to_claim <= static_cast<std::int32_t>(_size)) && "n_slots_to_claim must be > 0 and <= bufferSize");

        auto nextSequence = _nextValue + n_slots_to_claim;
        auto wrapPoint    = nextSequence - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _cachedValue; wrapPoint > cachedGatingSequence || cachedGatingSequence > _nextValue) {
            _cursor.setValue(_nextValue);

            SpinWait     spinWait;
            std::int64_t minSequence;
            while (wrapPoint > (minSequence = detail::getMinimumSequence(dependents, _nextValue))) {
                if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                    _waitStrategy.signalAllWhenBlocking();
                }
                spinWait.spinOnce();
            }
            _cachedValue = minSequence;
        }
        _nextValue = nextSequence;

        return nextSequence;
    }

    std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::size_t n_slots_to_claim) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        if (!hasAvailableCapacity(dependents, n_slots_to_claim, 0 /* unused cursor value */)) {
            throw NoCapacityException();
        }

        const auto nextSequence = _nextValue + n_slots_to_claim;
        _nextValue              = nextSequence;

        return nextSequence;
    }

    std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto consumed = detail::getMinimumSequence(dependents, _nextValue);
        const auto produced = _nextValue;

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        _cursor.setValue(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept { return sequence <= _cursor.value(); }
    [[nodiscard]] std::int64_t     getHighestPublishedSequence(std::int64_t /*nextSequence*/, std::int64_t availableSequence) const noexcept { return availableSequence; }
};
static_assert(ClaimStrategy<SingleThreadedStrategy<1024, NoWaitStrategy>>);

/**
 * Claim strategy for claiming sequences for access to a data structure while tracking dependent Sequences.
 * Suitable for use for sequencing across multiple publisher threads.
 * Note on cursor:  With this sequencer the cursor value is updated after the call to SequencerBase::next(),
 * to determine the highest available sequence that can be read, then getHighestPublishedSequence should be used.
 */
template<std::size_t SIZE = std::dynamic_extent, WaitStrategy WAIT_STRATEGY = BusySpinWaitStrategy>
class MultiThreadedStrategy {
    alignas(kCacheLine) const std::size_t _size;
    alignas(kCacheLine) Sequence &_cursor;
    alignas(kCacheLine) WAIT_STRATEGY &_waitStrategy;
    alignas(kCacheLine) std::vector<std::int32_t> _availableBuffer; // tracks the state of each ringbuffer slot
    alignas(kCacheLine) std::shared_ptr<Sequence> _gatingSequenceCache = std::make_shared<Sequence>();
    const std::int32_t _indexMask;
    const std::int32_t _indexShift;

public:
    MultiThreadedStrategy() = delete;
    explicit MultiThreadedStrategy(Sequence &cursor, WAIT_STRATEGY &waitStrategy, const std::size_t buffer_size = SIZE)
        : _size(buffer_size), _cursor(cursor), _waitStrategy(waitStrategy), _availableBuffer(_size),
        _indexMask(_size - 1), _indexShift(claim_strategy::util::ceillog2(_size)) {
        for (std::size_t i = _size - 1; i != 0; i--) {
            setAvailableBufferValue(i, -1);
        }
        setAvailableBufferValue(0, -1);
    }
    MultiThreadedStrategy(const MultiThreadedStrategy &)  = delete;
    MultiThreadedStrategy(const MultiThreadedStrategy &&) = delete;
    void               operator=(const MultiThreadedStrategy &) = delete;

    [[nodiscard]] bool hasAvailableCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents, const std::int64_t requiredCapacity, const std::int64_t cursorValue) const noexcept {
        const auto wrapPoint = (cursorValue + requiredCapacity) - static_cast<std::int64_t>(_size);

        if (const auto cachedGatingSequence = _gatingSequenceCache->value(); wrapPoint > cachedGatingSequence || cachedGatingSequence > cursorValue) {
            const auto minSequence = detail::getMinimumSequence(dependents, cursorValue);
            _gatingSequenceCache->setValue(minSequence);

            if (wrapPoint > minSequence) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::int64_t next(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        SpinWait     spinWait;
        do {
            current                           = _cursor.value();
            next                              = current + n_slots_to_claim;

            std::int64_t wrapPoint            = next - static_cast<std::int64_t>(_size);
            std::int64_t cachedGatingSequence = _gatingSequenceCache->value();

            if (wrapPoint > cachedGatingSequence || cachedGatingSequence > current) {
                std::int64_t gatingSequence = detail::getMinimumSequence(dependents, current);

                if (wrapPoint > gatingSequence) {
                    if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
                        _waitStrategy.signalAllWhenBlocking();
                    }
                    spinWait.spinOnce();
                    continue;
                }

                _gatingSequenceCache->setValue(gatingSequence);
            } else if (_cursor.compareAndSet(current, next)) {
                break;
            }
        } while (true);

        return next;
    }

    [[nodiscard]] std::int64_t tryNext(const std::vector<std::shared_ptr<Sequence>> &dependents, std::size_t n_slots_to_claim = 1) {
        assert((n_slots_to_claim > 0) && "n_slots_to_claim must be > 0");

        std::int64_t current;
        std::int64_t next;

        do {
            current = _cursor.value();
            next    = current + n_slots_to_claim;

            if (!hasAvailableCapacity(dependents, n_slots_to_claim, current)) {
                throw NoCapacityException();
            }
        } while (!_cursor.compareAndSet(current, next));

        return next;
    }

    [[nodiscard]] std::int64_t getRemainingCapacity(const std::vector<std::shared_ptr<Sequence>> &dependents) const noexcept {
        const auto produced = _cursor.value();
        const auto consumed = detail::getMinimumSequence(dependents, produced);

        return static_cast<std::int64_t>(_size) - (produced - consumed);
    }

    void publish(std::int64_t sequence) {
        setAvailable(sequence);
        if constexpr (hasSignalAllWhenBlocking<WAIT_STRATEGY>) {
            _waitStrategy.signalAllWhenBlocking();
        }
    }

    [[nodiscard]] forceinline bool isAvailable(std::int64_t sequence) const noexcept {
        const auto index = calculateIndex(sequence);
        const auto flag  = calculateAvailabilityFlag(sequence);

        return _availableBuffer[static_cast<std::size_t>(index)] == flag;
    }

    [[nodiscard]] forceinline std::int64_t getHighestPublishedSequence(const std::int64_t lowerBound, const std::int64_t availableSequence) const noexcept {
        for (std::int64_t sequence = lowerBound; sequence <= availableSequence; sequence++) {
            if (!isAvailable(sequence)) {
                return sequence - 1;
            }
        }

        return availableSequence;
    }

private:
    void                      setAvailable(std::int64_t sequence) noexcept { setAvailableBufferValue(calculateIndex(sequence), calculateAvailabilityFlag(sequence)); }
    forceinline void          setAvailableBufferValue(std::size_t index, std::int32_t flag) noexcept { _availableBuffer[index] = flag; }
    [[nodiscard]] forceinline std::int32_t calculateAvailabilityFlag(const std::int64_t sequence) const noexcept { return static_cast<std::int32_t>(static_cast<std::uint64_t>(sequence) >> _indexShift); }
    [[nodiscard]] forceinline std::size_t calculateIndex(const std::int64_t sequence) const noexcept { return static_cast<std::size_t>(static_cast<std::int32_t>(sequence) & _indexMask); }
};
static_assert(ClaimStrategy<MultiThreadedStrategy<1024, NoWaitStrategy>>);
// clang-format on

enum class ProducerType {
    /**
     * creates a buffer assuming a single producer-thread and multiple consumer
     */
    Single,

    /**
     * creates a buffer assuming multiple producer-threads and multiple consumer
     */
    Multi
};

namespace detail {
template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
struct producer_type;

template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Single, WAIT_STRATEGY> {
    using value_type = SingleThreadedStrategy<size, WAIT_STRATEGY>;
};
template <std::size_t size, WaitStrategy WAIT_STRATEGY>
struct producer_type<size, ProducerType::Multi, WAIT_STRATEGY> {
    using value_type = MultiThreadedStrategy<size, WAIT_STRATEGY>;
};

template <std::size_t size, ProducerType producerType, WaitStrategy WAIT_STRATEGY>
using producer_type_v = typename producer_type<size, producerType, WAIT_STRATEGY>::value_type;

} // namespace detail

} // namespace gr


#endif // GNURADIO_CLAIM_STRATEGY_HPP
// Formatting library for C++ - the core API for char/UTF-8
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_CORE_H_
#define FMT_CORE_H_

#include <cstddef>  // std::byte
#include <cstdio>   // std::FILE
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>

// The fmt library version in the form major * 10000 + minor * 100 + patch.
#define FMT_VERSION 80101

#if defined(__clang__) && !defined(__ibmxl__)
#  define FMT_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#else
#  define FMT_CLANG_VERSION 0
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__NVCOMPILER)
#  define FMT_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#  define FMT_GCC_VERSION 0
#endif

#ifndef FMT_GCC_PRAGMA
// Workaround _Pragma bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59884.
#  if FMT_GCC_VERSION >= 504
#    define FMT_GCC_PRAGMA(arg) _Pragma(arg)
#  else
#    define FMT_GCC_PRAGMA(arg)
#  endif
#endif

#ifdef __ICL
#  define FMT_ICC_VERSION __ICL
#elif defined(__INTEL_COMPILER)
#  define FMT_ICC_VERSION __INTEL_COMPILER
#else
#  define FMT_ICC_VERSION 0
#endif

#ifdef __NVCC__
#  define FMT_NVCC __NVCC__
#else
#  define FMT_NVCC 0
#endif

#ifdef _MSC_VER
#  define FMT_MSC_VER _MSC_VER
#  define FMT_MSC_WARNING(...) __pragma(warning(__VA_ARGS__))
#else
#  define FMT_MSC_VER 0
#  define FMT_MSC_WARNING(...)
#endif

#ifdef __has_feature
#  define FMT_HAS_FEATURE(x) __has_feature(x)
#else
#  define FMT_HAS_FEATURE(x) 0
#endif

#if defined(__has_include) &&                             \
    (!defined(__INTELLISENSE__) || FMT_MSC_VER > 1900) && \
    (!FMT_ICC_VERSION || FMT_ICC_VERSION >= 1600)
#  define FMT_HAS_INCLUDE(x) __has_include(x)
#else
#  define FMT_HAS_INCLUDE(x) 0
#endif

#ifdef __has_cpp_attribute
#  define FMT_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#  define FMT_HAS_CPP_ATTRIBUTE(x) 0
#endif

#ifdef _MSVC_LANG
#  define FMT_CPLUSPLUS _MSVC_LANG
#else
#  define FMT_CPLUSPLUS __cplusplus
#endif

#define FMT_HAS_CPP14_ATTRIBUTE(attribute) \
  (FMT_CPLUSPLUS >= 201402L && FMT_HAS_CPP_ATTRIBUTE(attribute))

#define FMT_HAS_CPP17_ATTRIBUTE(attribute) \
  (FMT_CPLUSPLUS >= 201703L && FMT_HAS_CPP_ATTRIBUTE(attribute))

// Check if relaxed C++14 constexpr is supported.
// GCC doesn't allow throw in constexpr until version 6 (bug 67371).
#ifndef FMT_USE_CONSTEXPR
#  define FMT_USE_CONSTEXPR                                           \
    (FMT_HAS_FEATURE(cxx_relaxed_constexpr) || FMT_MSC_VER >= 1912 || \
     (FMT_GCC_VERSION >= 600 && __cplusplus >= 201402L)) &&           \
        !FMT_NVCC && !FMT_ICC_VERSION
#endif
#if FMT_USE_CONSTEXPR
#  define FMT_CONSTEXPR constexpr
#  define FMT_CONSTEXPR_DECL constexpr
#else
#  define FMT_CONSTEXPR
#  define FMT_CONSTEXPR_DECL
#endif

#if ((__cplusplus >= 202002L) &&                              \
     (!defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE > 9)) || \
    (__cplusplus >= 201709L && FMT_GCC_VERSION >= 1002)
#  define FMT_CONSTEXPR20 constexpr
#else
#  define FMT_CONSTEXPR20
#endif

// Check if constexpr std::char_traits<>::compare,length is supported.
#if defined(__GLIBCXX__)
#  if __cplusplus >= 201703L && defined(_GLIBCXX_RELEASE) && \
      _GLIBCXX_RELEASE >= 7  // GCC 7+ libstdc++ has _GLIBCXX_RELEASE.
#    define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#  endif
#elif defined(_LIBCPP_VERSION) && __cplusplus >= 201703L && \
    _LIBCPP_VERSION >= 4000
#  define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#elif FMT_MSC_VER >= 1914 && _MSVC_LANG >= 201703L
#  define FMT_CONSTEXPR_CHAR_TRAITS constexpr
#endif
#ifndef FMT_CONSTEXPR_CHAR_TRAITS
#  define FMT_CONSTEXPR_CHAR_TRAITS
#endif

// Check if exceptions are disabled.
#ifndef FMT_EXCEPTIONS
#  if (defined(__GNUC__) && !defined(__EXCEPTIONS)) || \
      FMT_MSC_VER && !_HAS_EXCEPTIONS
#    define FMT_EXCEPTIONS 0
#  else
#    define FMT_EXCEPTIONS 1
#  endif
#endif

// Define FMT_USE_NOEXCEPT to make fmt use noexcept (C++11 feature).
#ifndef FMT_USE_NOEXCEPT
#  define FMT_USE_NOEXCEPT 0
#endif

#if FMT_USE_NOEXCEPT || FMT_HAS_FEATURE(cxx_noexcept) || \
    FMT_GCC_VERSION >= 408 || FMT_MSC_VER >= 1900
#  define FMT_DETECTED_NOEXCEPT noexcept
#  define FMT_HAS_CXX11_NOEXCEPT 1
#else
#  define FMT_DETECTED_NOEXCEPT throw()
#  define FMT_HAS_CXX11_NOEXCEPT 0
#endif

#ifndef FMT_NOEXCEPT
#  if FMT_EXCEPTIONS || FMT_HAS_CXX11_NOEXCEPT
#    define FMT_NOEXCEPT FMT_DETECTED_NOEXCEPT
#  else
#    define FMT_NOEXCEPT
#  endif
#endif

// [[noreturn]] is disabled on MSVC and NVCC because of bogus unreachable code
// warnings.
#if FMT_EXCEPTIONS && FMT_HAS_CPP_ATTRIBUTE(noreturn) && !FMT_MSC_VER && \
    !FMT_NVCC
#  define FMT_NORETURN [[noreturn]]
#else
#  define FMT_NORETURN
#endif

#if __cplusplus == 201103L || __cplusplus == 201402L
#  if defined(__INTEL_COMPILER) || defined(__PGI)
#    define FMT_FALLTHROUGH
#  elif defined(__clang__)
#    define FMT_FALLTHROUGH [[clang::fallthrough]]
#  elif FMT_GCC_VERSION >= 700 && \
      (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 520)
#    define FMT_FALLTHROUGH [[gnu::fallthrough]]
#  else
#    define FMT_FALLTHROUGH
#  endif
#elif FMT_HAS_CPP17_ATTRIBUTE(fallthrough)
#  define FMT_FALLTHROUGH [[fallthrough]]
#else
#  define FMT_FALLTHROUGH
#endif

#ifndef FMT_NODISCARD
#  if FMT_HAS_CPP17_ATTRIBUTE(nodiscard)
#    define FMT_NODISCARD [[nodiscard]]
#  else
#    define FMT_NODISCARD
#  endif
#endif

#ifndef FMT_USE_FLOAT
#  define FMT_USE_FLOAT 1
#endif
#ifndef FMT_USE_DOUBLE
#  define FMT_USE_DOUBLE 1
#endif
#ifndef FMT_USE_LONG_DOUBLE
#  define FMT_USE_LONG_DOUBLE 1
#endif

#ifndef FMT_INLINE
#  if FMT_GCC_VERSION || FMT_CLANG_VERSION
#    define FMT_INLINE inline __attribute__((always_inline))
#  else
#    define FMT_INLINE inline
#  endif
#endif

#ifndef FMT_DEPRECATED
#  if FMT_HAS_CPP14_ATTRIBUTE(deprecated) || FMT_MSC_VER >= 1900
#    define FMT_DEPRECATED [[deprecated]]
#  else
#    if (defined(__GNUC__) && !defined(__LCC__)) || defined(__clang__)
#      define FMT_DEPRECATED __attribute__((deprecated))
#    elif FMT_MSC_VER
#      define FMT_DEPRECATED __declspec(deprecated)
#    else
#      define FMT_DEPRECATED /* deprecated */
#    endif
#  endif
#endif

#ifndef FMT_BEGIN_NAMESPACE
#  define FMT_BEGIN_NAMESPACE \
    namespace fmt {           \
    inline namespace v8 {
#  define FMT_END_NAMESPACE \
    }                       \
    }
#endif

#ifndef FMT_MODULE_EXPORT
#  define FMT_MODULE_EXPORT
#  define FMT_MODULE_EXPORT_BEGIN
#  define FMT_MODULE_EXPORT_END
#  define FMT_BEGIN_DETAIL_NAMESPACE namespace detail {
#  define FMT_END_DETAIL_NAMESPACE }
#endif

#if !defined(FMT_HEADER_ONLY) && defined(_WIN32)
#  define FMT_CLASS_API FMT_MSC_WARNING(suppress : 4275)
#  ifdef FMT_EXPORT
#    define FMT_API __declspec(dllexport)
#  elif defined(FMT_SHARED)
#    define FMT_API __declspec(dllimport)
#  endif
#else
#  define FMT_CLASS_API
#  if defined(FMT_EXPORT) || defined(FMT_SHARED)
#    if defined(__GNUC__) || defined(__clang__)
#      define FMT_API __attribute__((visibility("default")))
#    endif
#  endif
#endif
#ifndef FMT_API
#  define FMT_API
#endif

// libc++ supports string_view in pre-c++17.
#if (FMT_HAS_INCLUDE(<string_view>) &&                       \
     (__cplusplus > 201402L || defined(_LIBCPP_VERSION))) || \
    (defined(_MSVC_LANG) && _MSVC_LANG > 201402L && _MSC_VER >= 1910)
#  include <string_view>
#  define FMT_USE_STRING_VIEW
#elif FMT_HAS_INCLUDE("experimental/string_view") && __cplusplus >= 201402L
#  include <experimental/string_view>
#  define FMT_USE_EXPERIMENTAL_STRING_VIEW
#endif

#ifndef FMT_UNICODE
#  define FMT_UNICODE !FMT_MSC_VER
#endif

#ifndef FMT_CONSTEVAL
#  if ((FMT_GCC_VERSION >= 1000 || FMT_CLANG_VERSION >= 1101) &&      \
       __cplusplus > 201703L && !defined(__apple_build_version__)) || \
      (defined(__cpp_consteval) &&                                    \
       (!FMT_MSC_VER || _MSC_FULL_VER >= 193030704))
// consteval is broken in MSVC before VS2022 and Apple clang 13.
#    define FMT_CONSTEVAL consteval
#    define FMT_HAS_CONSTEVAL
#  else
#    define FMT_CONSTEVAL
#  endif
#endif

#ifndef FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
#  if defined(__cpp_nontype_template_args) &&                \
      ((FMT_GCC_VERSION >= 903 && __cplusplus >= 201709L) || \
       __cpp_nontype_template_args >= 201911L)
#    define FMT_USE_NONTYPE_TEMPLATE_PARAMETERS 1
#  else
#    define FMT_USE_NONTYPE_TEMPLATE_PARAMETERS 0
#  endif
#endif

// Enable minimal optimizations for more compact code in debug mode.
FMT_GCC_PRAGMA("GCC push_options")
#ifndef __OPTIMIZE__
FMT_GCC_PRAGMA("GCC optimize(\"Og\")")
#endif

FMT_BEGIN_NAMESPACE
FMT_MODULE_EXPORT_BEGIN

// Implementations of enable_if_t and other metafunctions for older systems.
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;
template <bool B> using bool_constant = std::integral_constant<bool, B>;
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <typename T>
using remove_const_t = typename std::remove_const<T>::type;
template <typename T>
using remove_cvref_t = typename std::remove_cv<remove_reference_t<T>>::type;
template <typename T> struct type_identity { using type = T; };
template <typename T> using type_identity_t = typename type_identity<T>::type;

struct monostate {
  constexpr monostate() {}
};

// An enable_if helper to be used in template parameters which results in much
// shorter symbols: https://godbolt.org/z/sWw4vP. Extra parentheses are needed
// to workaround a bug in MSVC 2019 (see #1140 and #1186).
#ifdef FMT_DOC
#  define FMT_ENABLE_IF(...)
#else
#  define FMT_ENABLE_IF(...) enable_if_t<(__VA_ARGS__), int> = 0
#endif

FMT_BEGIN_DETAIL_NAMESPACE

// Suppress "unused variable" warnings with the method described in
// https://herbsutter.com/2009/10/18/mailbag-shutting-up-compiler-warnings/.
// (void)var does not work on many Intel compilers.
template <typename... T> FMT_CONSTEXPR void ignore_unused(const T&...) {}

constexpr FMT_INLINE auto is_constant_evaluated(bool default_value = false)
    FMT_NOEXCEPT -> bool {
#ifdef __cpp_lib_is_constant_evaluated
  ignore_unused(default_value);
  return std::is_constant_evaluated();
#else
  return default_value;
#endif
}

// A function to suppress "conditional expression is constant" warnings.
template <typename T> constexpr FMT_INLINE auto const_check(T value) -> T {
  return value;
}

FMT_NORETURN FMT_API void assert_fail(const char* file, int line,
                                      const char* message);

#ifndef FMT_ASSERT
#  ifdef NDEBUG
// FMT_ASSERT is not empty to avoid -Werror=empty-body.
#    define FMT_ASSERT(condition, message) \
      ::fmt::detail::ignore_unused((condition), (message))
#  else
#    define FMT_ASSERT(condition, message)                                    \
      ((condition) /* void() fails with -Winvalid-constexpr on clang 4.0.1 */ \
           ? (void)0                                                          \
           : ::fmt::detail::assert_fail(__FILE__, __LINE__, (message)))
#  endif
#endif

#ifdef __cpp_lib_byte
using byte = std::byte;
#else
enum class byte : unsigned char {};
#endif

#if defined(FMT_USE_STRING_VIEW)
template <typename Char> using std_string_view = std::basic_string_view<Char>;
#elif defined(FMT_USE_EXPERIMENTAL_STRING_VIEW)
template <typename Char>
using std_string_view = std::experimental::basic_string_view<Char>;
#else
template <typename T> struct std_string_view {};
#endif

#ifdef FMT_USE_INT128
// Do nothing.
#elif defined(__SIZEOF_INT128__) && !FMT_NVCC && \
    !(FMT_CLANG_VERSION && FMT_MSC_VER)
#  define FMT_USE_INT128 1
using int128_t = __int128_t;
using uint128_t = __uint128_t;
template <typename T> inline auto convert_for_visit(T value) -> T {
  return value;
}
#else
#  define FMT_USE_INT128 0
#endif
#if !FMT_USE_INT128
enum class int128_t {};
enum class uint128_t {};
// Reduce template instantiations.
template <typename T> inline auto convert_for_visit(T) -> monostate {
  return {};
}
#endif

// Casts a nonnegative integer to unsigned.
template <typename Int>
FMT_CONSTEXPR auto to_unsigned(Int value) ->
    typename std::make_unsigned<Int>::type {
  FMT_ASSERT(value >= 0, "negative value");
  return static_cast<typename std::make_unsigned<Int>::type>(value);
}

FMT_MSC_WARNING(suppress : 4566) constexpr unsigned char micro[] = "\u00B5";

constexpr auto is_utf8() -> bool {
  // Avoid buggy sign extensions in MSVC's constant evaluation mode.
  // https://developercommunity.visualstudio.com/t/C-difference-in-behavior-for-unsigned/1233612
  using uchar = unsigned char;
  return FMT_UNICODE || (sizeof(micro) == 3 && uchar(micro[0]) == 0xC2 &&
                         uchar(micro[1]) == 0xB5);
}
FMT_END_DETAIL_NAMESPACE

/**
  An implementation of ``std::basic_string_view`` for pre-C++17. It provides a
  subset of the API. ``fmt::basic_string_view`` is used for format strings even
  if ``std::string_view`` is available to prevent issues when a library is
  compiled with a different ``-std`` option than the client code (which is not
  recommended).
 */
template <typename Char> class basic_string_view {
 private:
  const Char* data_;
  size_t size_;

 public:
  using value_type = Char;
  using iterator = const Char*;

  constexpr basic_string_view() FMT_NOEXCEPT : data_(nullptr), size_(0) {}

  /** Constructs a string reference object from a C string and a size. */
  constexpr basic_string_view(const Char* s, size_t count) FMT_NOEXCEPT
      : data_(s),
        size_(count) {}

  /**
    \rst
    Constructs a string reference object from a C string computing
    the size with ``std::char_traits<Char>::length``.
    \endrst
   */
  FMT_CONSTEXPR_CHAR_TRAITS
  FMT_INLINE
  basic_string_view(const Char* s)
      : data_(s),
        size_(detail::const_check(std::is_same<Char, char>::value &&
                                  !detail::is_constant_evaluated(true))
                  ? std::strlen(reinterpret_cast<const char*>(s))
                  : std::char_traits<Char>::length(s)) {}

  /** Constructs a string reference from a ``std::basic_string`` object. */
  template <typename Traits, typename Alloc>
  FMT_CONSTEXPR basic_string_view(
      const std::basic_string<Char, Traits, Alloc>& s) FMT_NOEXCEPT
      : data_(s.data()),
        size_(s.size()) {}

  template <typename S, FMT_ENABLE_IF(std::is_same<
                                      S, detail::std_string_view<Char>>::value)>
  FMT_CONSTEXPR basic_string_view(S s) FMT_NOEXCEPT : data_(s.data()),
                                                      size_(s.size()) {}

  /** Returns a pointer to the string data. */
  constexpr auto data() const FMT_NOEXCEPT -> const Char* { return data_; }

  /** Returns the string size. */
  constexpr auto size() const FMT_NOEXCEPT -> size_t { return size_; }

  constexpr auto begin() const FMT_NOEXCEPT -> iterator { return data_; }
  constexpr auto end() const FMT_NOEXCEPT -> iterator { return data_ + size_; }

  constexpr auto operator[](size_t pos) const FMT_NOEXCEPT -> const Char& {
    return data_[pos];
  }

  FMT_CONSTEXPR void remove_prefix(size_t n) FMT_NOEXCEPT {
    data_ += n;
    size_ -= n;
  }

  // Lexicographically compare this string reference to other.
  FMT_CONSTEXPR_CHAR_TRAITS auto compare(basic_string_view other) const -> int {
    size_t str_size = size_ < other.size_ ? size_ : other.size_;
    int result = std::char_traits<Char>::compare(data_, other.data_, str_size);
    if (result == 0)
      result = size_ == other.size_ ? 0 : (size_ < other.size_ ? -1 : 1);
    return result;
  }

  FMT_CONSTEXPR_CHAR_TRAITS friend auto operator==(basic_string_view lhs,
                                                   basic_string_view rhs)
      -> bool {
    return lhs.compare(rhs) == 0;
  }
  friend auto operator!=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) != 0;
  }
  friend auto operator<(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) < 0;
  }
  friend auto operator<=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) <= 0;
  }
  friend auto operator>(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) > 0;
  }
  friend auto operator>=(basic_string_view lhs, basic_string_view rhs) -> bool {
    return lhs.compare(rhs) >= 0;
  }
};

using string_view = basic_string_view<char>;

/** Specifies if ``T`` is a character type. Can be specialized by users. */
template <typename T> struct is_char : std::false_type {};
template <> struct is_char<char> : std::true_type {};

// Returns a string view of `s`.
template <typename Char, FMT_ENABLE_IF(is_char<Char>::value)>
FMT_INLINE auto to_string_view(const Char* s) -> basic_string_view<Char> {
  return s;
}
template <typename Char, typename Traits, typename Alloc>
inline auto to_string_view(const std::basic_string<Char, Traits, Alloc>& s)
    -> basic_string_view<Char> {
  return s;
}
template <typename Char>
constexpr auto to_string_view(basic_string_view<Char> s)
    -> basic_string_view<Char> {
  return s;
}
template <typename Char,
          FMT_ENABLE_IF(!std::is_empty<detail::std_string_view<Char>>::value)>
inline auto to_string_view(detail::std_string_view<Char> s)
    -> basic_string_view<Char> {
  return s;
}

// A base class for compile-time strings. It is defined in the fmt namespace to
// make formatting functions visible via ADL, e.g. format(FMT_STRING("{}"), 42).
struct compile_string {};

template <typename S>
struct is_compile_string : std::is_base_of<compile_string, S> {};

template <typename S, FMT_ENABLE_IF(is_compile_string<S>::value)>
constexpr auto to_string_view(const S& s)
    -> basic_string_view<typename S::char_type> {
  return basic_string_view<typename S::char_type>(s);
}

FMT_BEGIN_DETAIL_NAMESPACE

void to_string_view(...);
using fmt::to_string_view;

// Specifies whether S is a string type convertible to fmt::basic_string_view.
// It should be a constexpr function but MSVC 2017 fails to compile it in
// enable_if and MSVC 2015 fails to compile it as an alias template.
template <typename S>
struct is_string : std::is_class<decltype(to_string_view(std::declval<S>()))> {
};

template <typename S, typename = void> struct char_t_impl {};
template <typename S> struct char_t_impl<S, enable_if_t<is_string<S>::value>> {
  using result = decltype(to_string_view(std::declval<S>()));
  using type = typename result::value_type;
};

// Reports a compile-time error if S is not a valid format string.
template <typename..., typename S, FMT_ENABLE_IF(!is_compile_string<S>::value)>
FMT_INLINE void check_format_string(const S&) {
#ifdef FMT_ENFORCE_COMPILE_STRING
  static_assert(is_compile_string<S>::value,
                "FMT_ENFORCE_COMPILE_STRING requires all format strings to use "
                "FMT_STRING.");
#endif
}
template <typename..., typename S, FMT_ENABLE_IF(is_compile_string<S>::value)>
void check_format_string(S);

FMT_NORETURN FMT_API void throw_format_error(const char* message);

struct error_handler {
  constexpr error_handler() = default;
  constexpr error_handler(const error_handler&) = default;

  // This function is intentionally not constexpr to give a compile-time error.
  FMT_NORETURN FMT_API void on_error(const char* message);
};
FMT_END_DETAIL_NAMESPACE

/** String's character type. */
template <typename S> using char_t = typename detail::char_t_impl<S>::type;

/**
  \rst
  Parsing context consisting of a format string range being parsed and an
  argument counter for automatic indexing.
  You can use the ``format_parse_context`` type alias for ``char`` instead.
  \endrst
 */
template <typename Char, typename ErrorHandler = detail::error_handler>
class basic_format_parse_context : private ErrorHandler {
 private:
  basic_string_view<Char> format_str_;
  int next_arg_id_;

 public:
  using char_type = Char;
  using iterator = typename basic_string_view<Char>::iterator;

  explicit constexpr basic_format_parse_context(
      basic_string_view<Char> format_str, ErrorHandler eh = {},
      int next_arg_id = 0)
      : ErrorHandler(eh), format_str_(format_str), next_arg_id_(next_arg_id) {}

  /**
    Returns an iterator to the beginning of the format string range being
    parsed.
   */
  constexpr auto begin() const FMT_NOEXCEPT -> iterator {
    return format_str_.begin();
  }

  /**
    Returns an iterator past the end of the format string range being parsed.
   */
  constexpr auto end() const FMT_NOEXCEPT -> iterator {
    return format_str_.end();
  }

  /** Advances the begin iterator to ``it``. */
  FMT_CONSTEXPR void advance_to(iterator it) {
    format_str_.remove_prefix(detail::to_unsigned(it - begin()));
  }

  /**
    Reports an error if using the manual argument indexing; otherwise returns
    the next argument index and switches to the automatic indexing.
   */
  FMT_CONSTEXPR auto next_arg_id() -> int {
    // Don't check if the argument id is valid to avoid overhead and because it
    // will be checked during formatting anyway.
    if (next_arg_id_ >= 0) return next_arg_id_++;
    on_error("cannot switch from manual to automatic argument indexing");
    return 0;
  }

  /**
    Reports an error if using the automatic argument indexing; otherwise
    switches to the manual indexing.
   */
  FMT_CONSTEXPR void check_arg_id(int) {
    if (next_arg_id_ > 0)
      on_error("cannot switch from automatic to manual argument indexing");
    else
      next_arg_id_ = -1;
  }

  FMT_CONSTEXPR void check_arg_id(basic_string_view<Char>) {}

  FMT_CONSTEXPR void on_error(const char* message) {
    ErrorHandler::on_error(message);
  }

  constexpr auto error_handler() const -> ErrorHandler { return *this; }
};

using format_parse_context = basic_format_parse_context<char>;

template <typename Context> class basic_format_arg;
template <typename Context> class basic_format_args;
template <typename Context> class dynamic_format_arg_store;

// A formatter for objects of type T.
template <typename T, typename Char = char, typename Enable = void>
struct formatter {
  // A deleted default constructor indicates a disabled formatter.
  formatter() = delete;
};

// Specifies if T has an enabled formatter specialization. A type can be
// formattable even if it doesn't have a formatter e.g. via a conversion.
template <typename T, typename Context>
using has_formatter =
    std::is_constructible<typename Context::template formatter_type<T>>;

// Checks whether T is a container with contiguous storage.
template <typename T> struct is_contiguous : std::false_type {};
template <typename Char>
struct is_contiguous<std::basic_string<Char>> : std::true_type {};

class appender;

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Context, typename T>
constexpr auto has_const_formatter_impl(T*)
    -> decltype(typename Context::template formatter_type<T>().format(
                    std::declval<const T&>(), std::declval<Context&>()),
                true) {
  return true;
}
template <typename Context>
constexpr auto has_const_formatter_impl(...) -> bool {
  return false;
}
template <typename T, typename Context>
constexpr auto has_const_formatter() -> bool {
  return has_const_formatter_impl<Context>(static_cast<T*>(nullptr));
}

// Extracts a reference to the container from back_insert_iterator.
template <typename Container>
inline auto get_container(std::back_insert_iterator<Container> it)
    -> Container& {
  using bi_iterator = std::back_insert_iterator<Container>;
  struct accessor : bi_iterator {
    accessor(bi_iterator iter) : bi_iterator(iter) {}
    using bi_iterator::container;
  };
  return *accessor(it).container;
}

template <typename Char, typename InputIt, typename OutputIt>
FMT_CONSTEXPR auto copy_str(InputIt begin, InputIt end, OutputIt out)
    -> OutputIt {
  while (begin != end) *out++ = static_cast<Char>(*begin++);
  return out;
}

template <typename Char, typename T, typename U,
          FMT_ENABLE_IF(
              std::is_same<remove_const_t<T>, U>::value&& is_char<U>::value)>
FMT_CONSTEXPR auto copy_str(T* begin, T* end, U* out) -> U* {
  if (is_constant_evaluated()) return copy_str<Char, T*, U*>(begin, end, out);
  auto size = to_unsigned(end - begin);
  memcpy(out, begin, size * sizeof(U));
  return out + size;
}

/**
  \rst
  A contiguous memory buffer with an optional growing ability. It is an internal
  class and shouldn't be used directly, only via `~fmt::basic_memory_buffer`.
  \endrst
 */
template <typename T> class buffer {
 private:
  T* ptr_;
  size_t size_;
  size_t capacity_;

 protected:
  // Don't initialize ptr_ since it is not accessed to save a few cycles.
  FMT_MSC_WARNING(suppress : 26495)
  buffer(size_t sz) FMT_NOEXCEPT : size_(sz), capacity_(sz) {}

  FMT_CONSTEXPR20 buffer(T* p = nullptr, size_t sz = 0,
                         size_t cap = 0) FMT_NOEXCEPT : ptr_(p),
                                                        size_(sz),
                                                        capacity_(cap) {}

  FMT_CONSTEXPR20 ~buffer() = default;
  buffer(buffer&&) = default;

  /** Sets the buffer data and capacity. */
  FMT_CONSTEXPR void set(T* buf_data, size_t buf_capacity) FMT_NOEXCEPT {
    ptr_ = buf_data;
    capacity_ = buf_capacity;
  }

  /** Increases the buffer capacity to hold at least *capacity* elements. */
  virtual FMT_CONSTEXPR20 void grow(size_t capacity) = 0;

 public:
  using value_type = T;
  using const_reference = const T&;

  buffer(const buffer&) = delete;
  void operator=(const buffer&) = delete;

  auto begin() FMT_NOEXCEPT -> T* { return ptr_; }
  auto end() FMT_NOEXCEPT -> T* { return ptr_ + size_; }

  auto begin() const FMT_NOEXCEPT -> const T* { return ptr_; }
  auto end() const FMT_NOEXCEPT -> const T* { return ptr_ + size_; }

  /** Returns the size of this buffer. */
  constexpr auto size() const FMT_NOEXCEPT -> size_t { return size_; }

  /** Returns the capacity of this buffer. */
  constexpr auto capacity() const FMT_NOEXCEPT -> size_t { return capacity_; }

  /** Returns a pointer to the buffer data. */
  FMT_CONSTEXPR auto data() FMT_NOEXCEPT -> T* { return ptr_; }

  /** Returns a pointer to the buffer data. */
  FMT_CONSTEXPR auto data() const FMT_NOEXCEPT -> const T* { return ptr_; }

  /** Clears this buffer. */
  void clear() { size_ = 0; }

  // Tries resizing the buffer to contain *count* elements. If T is a POD type
  // the new elements may not be initialized.
  FMT_CONSTEXPR20 void try_resize(size_t count) {
    try_reserve(count);
    size_ = count <= capacity_ ? count : capacity_;
  }

  // Tries increasing the buffer capacity to *new_capacity*. It can increase the
  // capacity by a smaller amount than requested but guarantees there is space
  // for at least one additional element either by increasing the capacity or by
  // flushing the buffer if it is full.
  FMT_CONSTEXPR20 void try_reserve(size_t new_capacity) {
    if (new_capacity > capacity_) grow(new_capacity);
  }

  FMT_CONSTEXPR20 void push_back(const T& value) {
    try_reserve(size_ + 1);
    ptr_[size_++] = value;
  }

  /** Appends data to the end of the buffer. */
  template <typename U> void append(const U* begin, const U* end);

  template <typename I> FMT_CONSTEXPR auto operator[](I index) -> T& {
    return ptr_[index];
  }
  template <typename I>
  FMT_CONSTEXPR auto operator[](I index) const -> const T& {
    return ptr_[index];
  }
};

struct buffer_traits {
  explicit buffer_traits(size_t) {}
  auto count() const -> size_t { return 0; }
  auto limit(size_t size) -> size_t { return size; }
};

class fixed_buffer_traits {
 private:
  size_t count_ = 0;
  size_t limit_;

 public:
  explicit fixed_buffer_traits(size_t limit) : limit_(limit) {}
  auto count() const -> size_t { return count_; }
  auto limit(size_t size) -> size_t {
    size_t n = limit_ > count_ ? limit_ - count_ : 0;
    count_ += size;
    return size < n ? size : n;
  }
};

// A buffer that writes to an output iterator when flushed.
template <typename OutputIt, typename T, typename Traits = buffer_traits>
class iterator_buffer final : public Traits, public buffer<T> {
 private:
  OutputIt out_;
  enum { buffer_size = 256 };
  T data_[buffer_size];

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() == buffer_size) flush();
  }

  void flush() {
    auto size = this->size();
    this->clear();
    out_ = copy_str<T>(data_, data_ + this->limit(size), out_);
  }

 public:
  explicit iterator_buffer(OutputIt out, size_t n = buffer_size)
      : Traits(n), buffer<T>(data_, 0, buffer_size), out_(out) {}
  iterator_buffer(iterator_buffer&& other)
      : Traits(other), buffer<T>(data_, 0, buffer_size), out_(other.out_) {}
  ~iterator_buffer() { flush(); }

  auto out() -> OutputIt {
    flush();
    return out_;
  }
  auto count() const -> size_t { return Traits::count() + this->size(); }
};

template <typename T>
class iterator_buffer<T*, T, fixed_buffer_traits> final
    : public fixed_buffer_traits,
      public buffer<T> {
 private:
  T* out_;
  enum { buffer_size = 256 };
  T data_[buffer_size];

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() == this->capacity()) flush();
  }

  void flush() {
    size_t n = this->limit(this->size());
    if (this->data() == out_) {
      out_ += n;
      this->set(data_, buffer_size);
    }
    this->clear();
  }

 public:
  explicit iterator_buffer(T* out, size_t n = buffer_size)
      : fixed_buffer_traits(n), buffer<T>(out, 0, n), out_(out) {}
  iterator_buffer(iterator_buffer&& other)
      : fixed_buffer_traits(other),
        buffer<T>(std::move(other)),
        out_(other.out_) {
    if (this->data() != out_) {
      this->set(data_, buffer_size);
      this->clear();
    }
  }
  ~iterator_buffer() { flush(); }

  auto out() -> T* {
    flush();
    return out_;
  }
  auto count() const -> size_t {
    return fixed_buffer_traits::count() + this->size();
  }
};

template <typename T> class iterator_buffer<T*, T> final : public buffer<T> {
 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {}

 public:
  explicit iterator_buffer(T* out, size_t = 0) : buffer<T>(out, 0, ~size_t()) {}

  auto out() -> T* { return &*this->end(); }
};

// A buffer that writes to a container with the contiguous storage.
template <typename Container>
class iterator_buffer<std::back_insert_iterator<Container>,
                      enable_if_t<is_contiguous<Container>::value,
                                  typename Container::value_type>>
    final : public buffer<typename Container::value_type> {
 private:
  Container& container_;

 protected:
  FMT_CONSTEXPR20 void grow(size_t capacity) override {
    container_.resize(capacity);
    this->set(&container_[0], capacity);
  }

 public:
  explicit iterator_buffer(Container& c)
      : buffer<typename Container::value_type>(c.size()), container_(c) {}
  explicit iterator_buffer(std::back_insert_iterator<Container> out, size_t = 0)
      : iterator_buffer(get_container(out)) {}
  auto out() -> std::back_insert_iterator<Container> {
    return std::back_inserter(container_);
  }
};

// A buffer that counts the number of code units written discarding the output.
template <typename T = char> class counting_buffer final : public buffer<T> {
 private:
  enum { buffer_size = 256 };
  T data_[buffer_size];
  size_t count_ = 0;

 protected:
  FMT_CONSTEXPR20 void grow(size_t) override {
    if (this->size() != buffer_size) return;
    count_ += this->size();
    this->clear();
  }

 public:
  counting_buffer() : buffer<T>(data_, 0, buffer_size) {}

  auto count() -> size_t { return count_ + this->size(); }
};

template <typename T>
using buffer_appender = conditional_t<std::is_same<T, char>::value, appender,
                                      std::back_insert_iterator<buffer<T>>>;

// Maps an output iterator to a buffer.
template <typename T, typename OutputIt>
auto get_buffer(OutputIt out) -> iterator_buffer<OutputIt, T> {
  return iterator_buffer<OutputIt, T>(out);
}

template <typename Buffer>
auto get_iterator(Buffer& buf) -> decltype(buf.out()) {
  return buf.out();
}
template <typename T> auto get_iterator(buffer<T>& buf) -> buffer_appender<T> {
  return buffer_appender<T>(buf);
}

template <typename T, typename Char = char, typename Enable = void>
struct fallback_formatter {
  fallback_formatter() = delete;
};

// Specifies if T has an enabled fallback_formatter specialization.
template <typename T, typename Char>
using has_fallback_formatter =
    std::is_constructible<fallback_formatter<T, Char>>;

struct view {};

template <typename Char, typename T> struct named_arg : view {
  const Char* name;
  const T& value;
  named_arg(const Char* n, const T& v) : name(n), value(v) {}
};

template <typename Char> struct named_arg_info {
  const Char* name;
  int id;
};

template <typename T, typename Char, size_t NUM_ARGS, size_t NUM_NAMED_ARGS>
struct arg_data {
  // args_[0].named_args points to named_args_ to avoid bloating format_args.
  // +1 to workaround a bug in gcc 7.5 that causes duplicated-branches warning.
  T args_[1 + (NUM_ARGS != 0 ? NUM_ARGS : +1)];
  named_arg_info<Char> named_args_[NUM_NAMED_ARGS];

  template <typename... U>
  arg_data(const U&... init) : args_{T(named_args_, NUM_NAMED_ARGS), init...} {}
  arg_data(const arg_data& other) = delete;
  auto args() const -> const T* { return args_ + 1; }
  auto named_args() -> named_arg_info<Char>* { return named_args_; }
};

template <typename T, typename Char, size_t NUM_ARGS>
struct arg_data<T, Char, NUM_ARGS, 0> {
  // +1 to workaround a bug in gcc 7.5 that causes duplicated-branches warning.
  T args_[NUM_ARGS != 0 ? NUM_ARGS : +1];

  template <typename... U>
  FMT_CONSTEXPR FMT_INLINE arg_data(const U&... init) : args_{init...} {}
  FMT_CONSTEXPR FMT_INLINE auto args() const -> const T* { return args_; }
  FMT_CONSTEXPR FMT_INLINE auto named_args() -> std::nullptr_t {
    return nullptr;
  }
};

template <typename Char>
inline void init_named_args(named_arg_info<Char>*, int, int) {}

template <typename T> struct is_named_arg : std::false_type {};
template <typename T> struct is_statically_named_arg : std::false_type {};

template <typename T, typename Char>
struct is_named_arg<named_arg<Char, T>> : std::true_type {};

template <typename Char, typename T, typename... Tail,
          FMT_ENABLE_IF(!is_named_arg<T>::value)>
void init_named_args(named_arg_info<Char>* named_args, int arg_count,
                     int named_arg_count, const T&, const Tail&... args) {
  init_named_args(named_args, arg_count + 1, named_arg_count, args...);
}

template <typename Char, typename T, typename... Tail,
          FMT_ENABLE_IF(is_named_arg<T>::value)>
void init_named_args(named_arg_info<Char>* named_args, int arg_count,
                     int named_arg_count, const T& arg, const Tail&... args) {
  named_args[named_arg_count++] = {arg.name, arg_count};
  init_named_args(named_args, arg_count + 1, named_arg_count, args...);
}

template <typename... Args>
FMT_CONSTEXPR FMT_INLINE void init_named_args(std::nullptr_t, int, int,
                                              const Args&...) {}

template <bool B = false> constexpr auto count() -> size_t { return B ? 1 : 0; }
template <bool B1, bool B2, bool... Tail> constexpr auto count() -> size_t {
  return (B1 ? 1 : 0) + count<B2, Tail...>();
}

template <typename... Args> constexpr auto count_named_args() -> size_t {
  return count<is_named_arg<Args>::value...>();
}

template <typename... Args>
constexpr auto count_statically_named_args() -> size_t {
  return count<is_statically_named_arg<Args>::value...>();
}

enum class type {
  none_type,
  // Integer types should go first,
  int_type,
  uint_type,
  long_long_type,
  ulong_long_type,
  int128_type,
  uint128_type,
  bool_type,
  char_type,
  last_integer_type = char_type,
  // followed by floating-point types.
  float_type,
  double_type,
  long_double_type,
  last_numeric_type = long_double_type,
  cstring_type,
  string_type,
  pointer_type,
  custom_type
};

// Maps core type T to the corresponding type enum constant.
template <typename T, typename Char>
struct type_constant : std::integral_constant<type, type::custom_type> {};

#define FMT_TYPE_CONSTANT(Type, constant) \
  template <typename Char>                \
  struct type_constant<Type, Char>        \
      : std::integral_constant<type, type::constant> {}

FMT_TYPE_CONSTANT(int, int_type);
FMT_TYPE_CONSTANT(unsigned, uint_type);
FMT_TYPE_CONSTANT(long long, long_long_type);
FMT_TYPE_CONSTANT(unsigned long long, ulong_long_type);
FMT_TYPE_CONSTANT(int128_t, int128_type);
FMT_TYPE_CONSTANT(uint128_t, uint128_type);
FMT_TYPE_CONSTANT(bool, bool_type);
FMT_TYPE_CONSTANT(Char, char_type);
FMT_TYPE_CONSTANT(float, float_type);
FMT_TYPE_CONSTANT(double, double_type);
FMT_TYPE_CONSTANT(long double, long_double_type);
FMT_TYPE_CONSTANT(const Char*, cstring_type);
FMT_TYPE_CONSTANT(basic_string_view<Char>, string_type);
FMT_TYPE_CONSTANT(const void*, pointer_type);

constexpr bool is_integral_type(type t) {
  return t > type::none_type && t <= type::last_integer_type;
}

constexpr bool is_arithmetic_type(type t) {
  return t > type::none_type && t <= type::last_numeric_type;
}

struct unformattable {};
struct unformattable_char : unformattable {};
struct unformattable_const : unformattable {};
struct unformattable_pointer : unformattable {};

template <typename Char> struct string_value {
  const Char* data;
  size_t size;
};

template <typename Char> struct named_arg_value {
  const named_arg_info<Char>* data;
  size_t size;
};

template <typename Context> struct custom_value {
  using parse_context = typename Context::parse_context_type;
  void* value;
  void (*format)(void* arg, parse_context& parse_ctx, Context& ctx);
};

// A formatting argument value.
template <typename Context> class value {
 public:
  using char_type = typename Context::char_type;

  union {
    monostate no_value;
    int int_value;
    unsigned uint_value;
    long long long_long_value;
    unsigned long long ulong_long_value;
    int128_t int128_value;
    uint128_t uint128_value;
    bool bool_value;
    char_type char_value;
    float float_value;
    double double_value;
    long double long_double_value;
    const void* pointer;
    string_value<char_type> string;
    custom_value<Context> custom;
    named_arg_value<char_type> named_args;
  };

  constexpr FMT_INLINE value() : no_value() {}
  constexpr FMT_INLINE value(int val) : int_value(val) {}
  constexpr FMT_INLINE value(unsigned val) : uint_value(val) {}
  constexpr FMT_INLINE value(long long val) : long_long_value(val) {}
  constexpr FMT_INLINE value(unsigned long long val) : ulong_long_value(val) {}
  FMT_INLINE value(int128_t val) : int128_value(val) {}
  FMT_INLINE value(uint128_t val) : uint128_value(val) {}
  constexpr FMT_INLINE value(float val) : float_value(val) {}
  constexpr FMT_INLINE value(double val) : double_value(val) {}
  FMT_INLINE value(long double val) : long_double_value(val) {}
  constexpr FMT_INLINE value(bool val) : bool_value(val) {}
  constexpr FMT_INLINE value(char_type val) : char_value(val) {}
  FMT_CONSTEXPR FMT_INLINE value(const char_type* val) {
    string.data = val;
    if (is_constant_evaluated()) string.size = {};
  }
  FMT_CONSTEXPR FMT_INLINE value(basic_string_view<char_type> val) {
    string.data = val.data();
    string.size = val.size();
  }
  FMT_INLINE value(const void* val) : pointer(val) {}
  FMT_INLINE value(const named_arg_info<char_type>* args, size_t size)
      : named_args{args, size} {}

  template <typename T> FMT_CONSTEXPR FMT_INLINE value(T& val) {
    using value_type = remove_cvref_t<T>;
    custom.value = const_cast<value_type*>(&val);
    // Get the formatter type through the context to allow different contexts
    // have different extension points, e.g. `formatter<T>` for `format` and
    // `printf_formatter<T>` for `printf`.
    custom.format = format_custom_arg<
        value_type,
        conditional_t<has_formatter<value_type, Context>::value,
                      typename Context::template formatter_type<value_type>,
                      fallback_formatter<value_type, char_type>>>;
  }
  value(unformattable);
  value(unformattable_char);
  value(unformattable_const);
  value(unformattable_pointer);

 private:
  // Formats an argument of a custom type, such as a user-defined class.
  template <typename T, typename Formatter>
  static void format_custom_arg(void* arg,
                                typename Context::parse_context_type& parse_ctx,
                                Context& ctx) {
    auto f = Formatter();
    parse_ctx.advance_to(f.parse(parse_ctx));
    using qualified_type =
        conditional_t<has_const_formatter<T, Context>(), const T, T>;
    ctx.advance_to(f.format(*static_cast<qualified_type*>(arg), ctx));
  }
};

template <typename Context, typename T>
FMT_CONSTEXPR auto make_arg(const T& value) -> basic_format_arg<Context>;

// To minimize the number of types we need to deal with, long is translated
// either to int or to long long depending on its size.
enum { long_short = sizeof(long) == sizeof(int) };
using long_type = conditional_t<long_short, int, long long>;
using ulong_type = conditional_t<long_short, unsigned, unsigned long long>;

// Maps formatting arguments to core types.
// arg_mapper reports errors by returning unformattable instead of using
// static_assert because it's used in the is_formattable trait.
template <typename Context> struct arg_mapper {
  using char_type = typename Context::char_type;

  FMT_CONSTEXPR FMT_INLINE auto map(signed char val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned char val) -> unsigned {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(short val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned short val) -> unsigned {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(int val) -> int { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned val) -> unsigned { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(long val) -> long_type { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned long val) -> ulong_type {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(long long val) -> long long { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(unsigned long long val)
      -> unsigned long long {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(int128_t val) -> int128_t { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(uint128_t val) -> uint128_t { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(bool val) -> bool { return val; }

  template <typename T, FMT_ENABLE_IF(std::is_same<T, char>::value ||
                                      std::is_same<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(T val) -> char_type {
    return val;
  }
  template <typename T, enable_if_t<(std::is_same<T, wchar_t>::value ||
#ifdef __cpp_char8_t
                                     std::is_same<T, char8_t>::value ||
#endif
                                     std::is_same<T, char16_t>::value ||
                                     std::is_same<T, char32_t>::value) &&
                                        !std::is_same<T, char_type>::value,
                                    int> = 0>
  FMT_CONSTEXPR FMT_INLINE auto map(T) -> unformattable_char {
    return {};
  }

  FMT_CONSTEXPR FMT_INLINE auto map(float val) -> float { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(double val) -> double { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(long double val) -> long double {
    return val;
  }

  FMT_CONSTEXPR FMT_INLINE auto map(char_type* val) -> const char_type* {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(const char_type* val) -> const char_type* {
    return val;
  }
  template <typename T,
            FMT_ENABLE_IF(is_string<T>::value && !std::is_pointer<T>::value &&
                          std::is_same<char_type, char_t<T>>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return to_string_view(val);
  }
  template <typename T,
            FMT_ENABLE_IF(is_string<T>::value && !std::is_pointer<T>::value &&
                          !std::is_same<char_type, char_t<T>>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T&) -> unformattable_char {
    return {};
  }
  template <typename T,
            FMT_ENABLE_IF(
                std::is_constructible<basic_string_view<char_type>, T>::value &&
                !is_string<T>::value && !has_formatter<T, Context>::value &&
                !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return basic_string_view<char_type>(val);
  }
  template <
      typename T,
      FMT_ENABLE_IF(
          std::is_constructible<std_string_view<char_type>, T>::value &&
          !std::is_constructible<basic_string_view<char_type>, T>::value &&
          !is_string<T>::value && !has_formatter<T, Context>::value &&
          !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> basic_string_view<char_type> {
    return std_string_view<char_type>(val);
  }

  using cstring_result = conditional_t<std::is_same<char_type, char>::value,
                                       const char*, unformattable_pointer>;

  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(const signed char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(const unsigned char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(signed char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }
  FMT_DEPRECATED FMT_CONSTEXPR FMT_INLINE auto map(unsigned char* val)
      -> cstring_result {
    return map(reinterpret_cast<const char*>(val));
  }

  FMT_CONSTEXPR FMT_INLINE auto map(void* val) -> const void* { return val; }
  FMT_CONSTEXPR FMT_INLINE auto map(const void* val) -> const void* {
    return val;
  }
  FMT_CONSTEXPR FMT_INLINE auto map(std::nullptr_t val) -> const void* {
    return val;
  }

  // We use SFINAE instead of a const T* parameter to avoid conflicting with
  // the C array overload.
  template <
      typename T,
      FMT_ENABLE_IF(
          std::is_member_pointer<T>::value ||
          std::is_function<typename std::remove_pointer<T>::type>::value ||
          (std::is_convertible<const T&, const void*>::value &&
           !std::is_convertible<const T&, const char_type*>::value))>
  FMT_CONSTEXPR auto map(const T&) -> unformattable_pointer {
    return {};
  }

  template <typename T, std::size_t N,
            FMT_ENABLE_IF(!std::is_same<T, wchar_t>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T (&values)[N]) -> const T (&)[N] {
    return values;
  }

  template <typename T,
            FMT_ENABLE_IF(
                std::is_enum<T>::value&& std::is_convertible<T, int>::value &&
                !has_formatter<T, Context>::value &&
                !has_fallback_formatter<T, char_type>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& val)
      -> decltype(std::declval<arg_mapper>().map(
          static_cast<typename std::underlying_type<T>::type>(val))) {
    return map(static_cast<typename std::underlying_type<T>::type>(val));
  }

  FMT_CONSTEXPR FMT_INLINE auto map(detail::byte val) -> unsigned {
    return map(static_cast<unsigned char>(val));
  }

  template <typename T, typename U = remove_cvref_t<T>>
  struct formattable
      : bool_constant<has_const_formatter<U, Context>() ||
                      !std::is_const<remove_reference_t<T>>::value ||
                      has_fallback_formatter<U, char_type>::value> {};

#if FMT_MSC_VER != 0 && FMT_MSC_VER < 1910
  // Workaround a bug in MSVC.
  template <typename T> FMT_CONSTEXPR FMT_INLINE auto do_map(T&& val) -> T& {
    return val;
  }
#else
  template <typename T, FMT_ENABLE_IF(formattable<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto do_map(T&& val) -> T& {
    return val;
  }
  template <typename T, FMT_ENABLE_IF(!formattable<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto do_map(T&&) -> unformattable_const {
    return {};
  }
#endif

  template <typename T, typename U = remove_cvref_t<T>,
            FMT_ENABLE_IF(!is_string<U>::value && !is_char<U>::value &&
                          !std::is_array<U>::value &&
                          (has_formatter<U, Context>::value ||
                           has_fallback_formatter<U, char_type>::value))>
  FMT_CONSTEXPR FMT_INLINE auto map(T&& val)
      -> decltype(this->do_map(std::forward<T>(val))) {
    return do_map(std::forward<T>(val));
  }

  template <typename T, FMT_ENABLE_IF(is_named_arg<T>::value)>
  FMT_CONSTEXPR FMT_INLINE auto map(const T& named_arg)
      -> decltype(std::declval<arg_mapper>().map(named_arg.value)) {
    return map(named_arg.value);
  }

  auto map(...) -> unformattable { return {}; }
};

// A type constant after applying arg_mapper<Context>.
template <typename T, typename Context>
using mapped_type_constant =
    type_constant<decltype(arg_mapper<Context>().map(std::declval<const T&>())),
                  typename Context::char_type>;

enum { packed_arg_bits = 4 };
// Maximum number of arguments with packed types.
enum { max_packed_args = 62 / packed_arg_bits };
enum : unsigned long long { is_unpacked_bit = 1ULL << 63 };
enum : unsigned long long { has_named_args_bit = 1ULL << 62 };

FMT_END_DETAIL_NAMESPACE

// An output iterator that appends to a buffer.
// It is used to reduce symbol sizes for the common case.
class appender : public std::back_insert_iterator<detail::buffer<char>> {
  using base = std::back_insert_iterator<detail::buffer<char>>;

  template <typename T>
  friend auto get_buffer(appender out) -> detail::buffer<char>& {
    return detail::get_container(out);
  }

 public:
  using std::back_insert_iterator<detail::buffer<char>>::back_insert_iterator;
  appender(base it) FMT_NOEXCEPT : base(it) {}
  using _Unchecked_type = appender;  // Mark iterator as checked.

  auto operator++() FMT_NOEXCEPT -> appender& { return *this; }

  auto operator++(int) FMT_NOEXCEPT -> appender { return *this; }
};

// A formatting argument. It is a trivially copyable/constructible type to
// allow storage in basic_memory_buffer.
template <typename Context> class basic_format_arg {
 private:
  detail::value<Context> value_;
  detail::type type_;

  template <typename ContextType, typename T>
  friend FMT_CONSTEXPR auto detail::make_arg(const T& value)
      -> basic_format_arg<ContextType>;

  template <typename Visitor, typename Ctx>
  friend FMT_CONSTEXPR auto visit_format_arg(Visitor&& vis,
                                             const basic_format_arg<Ctx>& arg)
      -> decltype(vis(0));

  friend class basic_format_args<Context>;
  friend class dynamic_format_arg_store<Context>;

  using char_type = typename Context::char_type;

  template <typename T, typename Char, size_t NUM_ARGS, size_t NUM_NAMED_ARGS>
  friend struct detail::arg_data;

  basic_format_arg(const detail::named_arg_info<char_type>* args, size_t size)
      : value_(args, size) {}

 public:
  class handle {
   public:
    explicit handle(detail::custom_value<Context> custom) : custom_(custom) {}

    void format(typename Context::parse_context_type& parse_ctx,
                Context& ctx) const {
      custom_.format(custom_.value, parse_ctx, ctx);
    }

   private:
    detail::custom_value<Context> custom_;
  };

  constexpr basic_format_arg() : type_(detail::type::none_type) {}

  constexpr explicit operator bool() const FMT_NOEXCEPT {
    return type_ != detail::type::none_type;
  }

  auto type() const -> detail::type { return type_; }

  auto is_integral() const -> bool { return detail::is_integral_type(type_); }
  auto is_arithmetic() const -> bool {
    return detail::is_arithmetic_type(type_);
  }
};

/**
  \rst
  Visits an argument dispatching to the appropriate visit method based on
  the argument type. For example, if the argument type is ``double`` then
  ``vis(value)`` will be called with the value of type ``double``.
  \endrst
 */
template <typename Visitor, typename Context>
FMT_CONSTEXPR FMT_INLINE auto visit_format_arg(
    Visitor&& vis, const basic_format_arg<Context>& arg) -> decltype(vis(0)) {
  switch (arg.type_) {
  case detail::type::none_type:
    break;
  case detail::type::int_type:
    return vis(arg.value_.int_value);
  case detail::type::uint_type:
    return vis(arg.value_.uint_value);
  case detail::type::long_long_type:
    return vis(arg.value_.long_long_value);
  case detail::type::ulong_long_type:
    return vis(arg.value_.ulong_long_value);
  case detail::type::int128_type:
    return vis(detail::convert_for_visit(arg.value_.int128_value));
  case detail::type::uint128_type:
    return vis(detail::convert_for_visit(arg.value_.uint128_value));
  case detail::type::bool_type:
    return vis(arg.value_.bool_value);
  case detail::type::char_type:
    return vis(arg.value_.char_value);
  case detail::type::float_type:
    return vis(arg.value_.float_value);
  case detail::type::double_type:
    return vis(arg.value_.double_value);
  case detail::type::long_double_type:
    return vis(arg.value_.long_double_value);
  case detail::type::cstring_type:
    return vis(arg.value_.string.data);
  case detail::type::string_type:
    using sv = basic_string_view<typename Context::char_type>;
    return vis(sv(arg.value_.string.data, arg.value_.string.size));
  case detail::type::pointer_type:
    return vis(arg.value_.pointer);
  case detail::type::custom_type:
    return vis(typename basic_format_arg<Context>::handle(arg.value_.custom));
  }
  return vis(monostate());
}

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Char, typename InputIt>
auto copy_str(InputIt begin, InputIt end, appender out) -> appender {
  get_container(out).append(begin, end);
  return out;
}

#if FMT_GCC_VERSION && FMT_GCC_VERSION < 500
// A workaround for gcc 4.8 to make void_t work in a SFINAE context.
template <typename... Ts> struct void_t_impl { using type = void; };
template <typename... Ts>
using void_t = typename detail::void_t_impl<Ts...>::type;
#else
template <typename...> using void_t = void;
#endif

template <typename It, typename T, typename Enable = void>
struct is_output_iterator : std::false_type {};

template <typename It, typename T>
struct is_output_iterator<
    It, T,
    void_t<typename std::iterator_traits<It>::iterator_category,
           decltype(*std::declval<It>() = std::declval<T>())>>
    : std::true_type {};

template <typename OutputIt>
struct is_back_insert_iterator : std::false_type {};
template <typename Container>
struct is_back_insert_iterator<std::back_insert_iterator<Container>>
    : std::true_type {};

template <typename OutputIt>
struct is_contiguous_back_insert_iterator : std::false_type {};
template <typename Container>
struct is_contiguous_back_insert_iterator<std::back_insert_iterator<Container>>
    : is_contiguous<Container> {};
template <>
struct is_contiguous_back_insert_iterator<appender> : std::true_type {};

// A type-erased reference to an std::locale to avoid heavy <locale> include.
class locale_ref {
 private:
  const void* locale_;  // A type-erased pointer to std::locale.

 public:
  constexpr locale_ref() : locale_(nullptr) {}
  template <typename Locale> explicit locale_ref(const Locale& loc);

  explicit operator bool() const FMT_NOEXCEPT { return locale_ != nullptr; }

  template <typename Locale> auto get() const -> Locale;
};

template <typename> constexpr auto encode_types() -> unsigned long long {
  return 0;
}

template <typename Context, typename Arg, typename... Args>
constexpr auto encode_types() -> unsigned long long {
  return static_cast<unsigned>(mapped_type_constant<Arg, Context>::value) |
         (encode_types<Context, Args...>() << packed_arg_bits);
}

template <typename Context, typename T>
FMT_CONSTEXPR auto make_arg(const T& value) -> basic_format_arg<Context> {
  basic_format_arg<Context> arg;
  arg.type_ = mapped_type_constant<T, Context>::value;
  arg.value_ = arg_mapper<Context>().map(value);
  return arg;
}

// The type template parameter is there to avoid an ODR violation when using
// a fallback formatter in one translation unit and an implicit conversion in
// another (not recommended).
template <bool IS_PACKED, typename Context, type, typename T,
          FMT_ENABLE_IF(IS_PACKED)>
FMT_CONSTEXPR FMT_INLINE auto make_arg(T&& val) -> value<Context> {
  const auto& arg = arg_mapper<Context>().map(std::forward<T>(val));

  constexpr bool formattable_char =
      !std::is_same<decltype(arg), const unformattable_char&>::value;
  static_assert(formattable_char, "Mixing character types is disallowed.");

  constexpr bool formattable_const =
      !std::is_same<decltype(arg), const unformattable_const&>::value;
  static_assert(formattable_const, "Cannot format a const argument.");

  // Formatting of arbitrary pointers is disallowed. If you want to output
  // a pointer cast it to "void *" or "const void *". In particular, this
  // forbids formatting of "[const] volatile char *" which is printed as bool
  // by iostreams.
  constexpr bool formattable_pointer =
      !std::is_same<decltype(arg), const unformattable_pointer&>::value;
  static_assert(formattable_pointer,
                "Formatting of non-void pointers is disallowed.");

  constexpr bool formattable =
      !std::is_same<decltype(arg), const unformattable&>::value;
  static_assert(
      formattable,
      "Cannot format an argument. To make type T formattable provide a "
      "formatter<T> specialization: https://fmt.dev/latest/api.html#udt");
  return {arg};
}

template <bool IS_PACKED, typename Context, type, typename T,
          FMT_ENABLE_IF(!IS_PACKED)>
inline auto make_arg(const T& value) -> basic_format_arg<Context> {
  return make_arg<Context>(value);
}
FMT_END_DETAIL_NAMESPACE

// Formatting context.
template <typename OutputIt, typename Char> class basic_format_context {
 public:
  /** The character type for the output. */
  using char_type = Char;

 private:
  OutputIt out_;
  basic_format_args<basic_format_context> args_;
  detail::locale_ref loc_;

 public:
  using iterator = OutputIt;
  using format_arg = basic_format_arg<basic_format_context>;
  using parse_context_type = basic_format_parse_context<Char>;
  template <typename T> using formatter_type = formatter<T, char_type>;

  basic_format_context(basic_format_context&&) = default;
  basic_format_context(const basic_format_context&) = delete;
  void operator=(const basic_format_context&) = delete;
  /**
   Constructs a ``basic_format_context`` object. References to the arguments are
   stored in the object so make sure they have appropriate lifetimes.
   */
  constexpr basic_format_context(
      OutputIt out, basic_format_args<basic_format_context> ctx_args,
      detail::locale_ref loc = detail::locale_ref())
      : out_(out), args_(ctx_args), loc_(loc) {}

  constexpr auto arg(int id) const -> format_arg { return args_.get(id); }
  FMT_CONSTEXPR auto arg(basic_string_view<char_type> name) -> format_arg {
    return args_.get(name);
  }
  FMT_CONSTEXPR auto arg_id(basic_string_view<char_type> name) -> int {
    return args_.get_id(name);
  }
  auto args() const -> const basic_format_args<basic_format_context>& {
    return args_;
  }

  FMT_CONSTEXPR auto error_handler() -> detail::error_handler { return {}; }
  void on_error(const char* message) { error_handler().on_error(message); }

  // Returns an iterator to the beginning of the output range.
  FMT_CONSTEXPR auto out() -> iterator { return out_; }

  // Advances the begin iterator to ``it``.
  void advance_to(iterator it) {
    if (!detail::is_back_insert_iterator<iterator>()) out_ = it;
  }

  FMT_CONSTEXPR auto locale() -> detail::locale_ref { return loc_; }
};

template <typename Char>
using buffer_context =
    basic_format_context<detail::buffer_appender<Char>, Char>;
using format_context = buffer_context<char>;

// Workaround an alias issue: https://stackoverflow.com/q/62767544/471164.
#define FMT_BUFFER_CONTEXT(Char) \
  basic_format_context<detail::buffer_appender<Char>, Char>

template <typename T, typename Char = char>
using is_formattable = bool_constant<
    !std::is_base_of<detail::unformattable,
                     decltype(detail::arg_mapper<buffer_context<Char>>().map(
                         std::declval<T>()))>::value &&
    !detail::has_fallback_formatter<T, Char>::value>;

/**
  \rst
  An array of references to arguments. It can be implicitly converted into
  `~fmt::basic_format_args` for passing into type-erased formatting functions
  such as `~fmt::vformat`.
  \endrst
 */
template <typename Context, typename... Args>
class format_arg_store
#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
    // Workaround a GCC template argument substitution bug.
    : public basic_format_args<Context>
#endif
{
 private:
  static const size_t num_args = sizeof...(Args);
  static const size_t num_named_args = detail::count_named_args<Args...>();
  static const bool is_packed = num_args <= detail::max_packed_args;

  using value_type = conditional_t<is_packed, detail::value<Context>,
                                   basic_format_arg<Context>>;

  detail::arg_data<value_type, typename Context::char_type, num_args,
                   num_named_args>
      data_;

  friend class basic_format_args<Context>;

  static constexpr unsigned long long desc =
      (is_packed ? detail::encode_types<Context, Args...>()
                 : detail::is_unpacked_bit | num_args) |
      (num_named_args != 0
           ? static_cast<unsigned long long>(detail::has_named_args_bit)
           : 0);

 public:
  template <typename... T>
  FMT_CONSTEXPR FMT_INLINE format_arg_store(T&&... args)
      :
#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
        basic_format_args<Context>(*this),
#endif
        data_{detail::make_arg<
            is_packed, Context,
            detail::mapped_type_constant<remove_cvref_t<T>, Context>::value>(
            std::forward<T>(args))...} {
    detail::init_named_args(data_.named_args(), 0, 0, args...);
  }
};

/**
  \rst
  Constructs a `~fmt::format_arg_store` object that contains references to
  arguments and can be implicitly converted to `~fmt::format_args`. `Context`
  can be omitted in which case it defaults to `~fmt::context`.
  See `~fmt::arg` for lifetime considerations.
  \endrst
 */
template <typename Context = format_context, typename... Args>
constexpr auto make_format_args(Args&&... args)
    -> format_arg_store<Context, remove_cvref_t<Args>...> {
  return {std::forward<Args>(args)...};
}

/**
  \rst
  Returns a named argument to be used in a formatting function.
  It should only be used in a call to a formatting function or
  `dynamic_format_arg_store::push_back`.

  **Example**::

    fmt::print("Elapsed time: {s:.2f} seconds", fmt::arg("s", 1.23));
  \endrst
 */
template <typename Char, typename T>
inline auto arg(const Char* name, const T& arg) -> detail::named_arg<Char, T> {
  static_assert(!detail::is_named_arg<T>(), "nested named arguments");
  return {name, arg};
}

/**
  \rst
  A view of a collection of formatting arguments. To avoid lifetime issues it
  should only be used as a parameter type in type-erased functions such as
  ``vformat``::

    void vlog(string_view format_str, format_args args);  // OK
    format_args args = make_format_args(42);  // Error: dangling reference
  \endrst
 */
template <typename Context> class basic_format_args {
 public:
  using size_type = int;
  using format_arg = basic_format_arg<Context>;

 private:
  // A descriptor that contains information about formatting arguments.
  // If the number of arguments is less or equal to max_packed_args then
  // argument types are passed in the descriptor. This reduces binary code size
  // per formatting function call.
  unsigned long long desc_;
  union {
    // If is_packed() returns true then argument values are stored in values_;
    // otherwise they are stored in args_. This is done to improve cache
    // locality and reduce compiled code size since storing larger objects
    // may require more code (at least on x86-64) even if the same amount of
    // data is actually copied to stack. It saves ~10% on the bloat test.
    const detail::value<Context>* values_;
    const format_arg* args_;
  };

  constexpr auto is_packed() const -> bool {
    return (desc_ & detail::is_unpacked_bit) == 0;
  }
  auto has_named_args() const -> bool {
    return (desc_ & detail::has_named_args_bit) != 0;
  }

  FMT_CONSTEXPR auto type(int index) const -> detail::type {
    int shift = index * detail::packed_arg_bits;
    unsigned int mask = (1 << detail::packed_arg_bits) - 1;
    return static_cast<detail::type>((desc_ >> shift) & mask);
  }

  constexpr FMT_INLINE basic_format_args(unsigned long long desc,
                                         const detail::value<Context>* values)
      : desc_(desc), values_(values) {}
  constexpr basic_format_args(unsigned long long desc, const format_arg* args)
      : desc_(desc), args_(args) {}

 public:
  constexpr basic_format_args() : desc_(0), args_(nullptr) {}

  /**
   \rst
   Constructs a `basic_format_args` object from `~fmt::format_arg_store`.
   \endrst
   */
  template <typename... Args>
  constexpr FMT_INLINE basic_format_args(
      const format_arg_store<Context, Args...>& store)
      : basic_format_args(format_arg_store<Context, Args...>::desc,
                          store.data_.args()) {}

  /**
   \rst
   Constructs a `basic_format_args` object from
   `~fmt::dynamic_format_arg_store`.
   \endrst
   */
  constexpr FMT_INLINE basic_format_args(
      const dynamic_format_arg_store<Context>& store)
      : basic_format_args(store.get_types(), store.data()) {}

  /**
   \rst
   Constructs a `basic_format_args` object from a dynamic set of arguments.
   \endrst
   */
  constexpr basic_format_args(const format_arg* args, int count)
      : basic_format_args(detail::is_unpacked_bit | detail::to_unsigned(count),
                          args) {}

  /** Returns the argument with the specified id. */
  FMT_CONSTEXPR auto get(int id) const -> format_arg {
    format_arg arg;
    if (!is_packed()) {
      if (id < max_size()) arg = args_[id];
      return arg;
    }
    if (id >= detail::max_packed_args) return arg;
    arg.type_ = type(id);
    if (arg.type_ == detail::type::none_type) return arg;
    arg.value_ = values_[id];
    return arg;
  }

  template <typename Char>
  auto get(basic_string_view<Char> name) const -> format_arg {
    int id = get_id(name);
    return id >= 0 ? get(id) : format_arg();
  }

  template <typename Char>
  auto get_id(basic_string_view<Char> name) const -> int {
    if (!has_named_args()) return -1;
    const auto& named_args =
        (is_packed() ? values_[-1] : args_[-1].value_).named_args;
    for (size_t i = 0; i < named_args.size; ++i) {
      if (named_args.data[i].name == name) return named_args.data[i].id;
    }
    return -1;
  }

  auto max_size() const -> int {
    unsigned long long max_packed = detail::max_packed_args;
    return static_cast<int>(is_packed() ? max_packed
                                        : desc_ & ~detail::is_unpacked_bit);
  }
};

/** An alias to ``basic_format_args<format_context>``. */
// A separate type would result in shorter symbols but break ABI compatibility
// between clang and gcc on ARM (#1919).
using format_args = basic_format_args<format_context>;

// We cannot use enum classes as bit fields because of a gcc bug
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61414.
namespace align {
enum type { none, left, right, center, numeric };
}
using align_t = align::type;
namespace sign {
enum type { none, minus, plus, space };
}
using sign_t = sign::type;

FMT_BEGIN_DETAIL_NAMESPACE

// Workaround an array initialization issue in gcc 4.8.
template <typename Char> struct fill_t {
 private:
  enum { max_size = 4 };
  Char data_[max_size] = {Char(' '), Char(0), Char(0), Char(0)};
  unsigned char size_ = 1;

 public:
  FMT_CONSTEXPR void operator=(basic_string_view<Char> s) {
    auto size = s.size();
    if (size > max_size) return throw_format_error("invalid fill");
    for (size_t i = 0; i < size; ++i) data_[i] = s[i];
    size_ = static_cast<unsigned char>(size);
  }

  constexpr auto size() const -> size_t { return size_; }
  constexpr auto data() const -> const Char* { return data_; }

  FMT_CONSTEXPR auto operator[](size_t index) -> Char& { return data_[index]; }
  FMT_CONSTEXPR auto operator[](size_t index) const -> const Char& {
    return data_[index];
  }
};
FMT_END_DETAIL_NAMESPACE

enum class presentation_type : unsigned char {
  none,
  // Integer types should go first,
  dec,             // 'd'
  oct,             // 'o'
  hex_lower,       // 'x'
  hex_upper,       // 'X'
  bin_lower,       // 'b'
  bin_upper,       // 'B'
  hexfloat_lower,  // 'a'
  hexfloat_upper,  // 'A'
  exp_lower,       // 'e'
  exp_upper,       // 'E'
  fixed_lower,     // 'f'
  fixed_upper,     // 'F'
  general_lower,   // 'g'
  general_upper,   // 'G'
  chr,             // 'c'
  string,          // 's'
  pointer          // 'p'
};

// Format specifiers for built-in and string types.
template <typename Char> struct basic_format_specs {
  int width;
  int precision;
  presentation_type type;
  align_t align : 4;
  sign_t sign : 3;
  bool alt : 1;  // Alternate form ('#').
  bool localized : 1;
  detail::fill_t<Char> fill;

  constexpr basic_format_specs()
      : width(0),
        precision(-1),
        type(presentation_type::none),
        align(align::none),
        sign(sign::none),
        alt(false),
        localized(false) {}
};

using format_specs = basic_format_specs<char>;

FMT_BEGIN_DETAIL_NAMESPACE

enum class arg_id_kind { none, index, name };

// An argument reference.
template <typename Char> struct arg_ref {
  FMT_CONSTEXPR arg_ref() : kind(arg_id_kind::none), val() {}

  FMT_CONSTEXPR explicit arg_ref(int index)
      : kind(arg_id_kind::index), val(index) {}
  FMT_CONSTEXPR explicit arg_ref(basic_string_view<Char> name)
      : kind(arg_id_kind::name), val(name) {}

  FMT_CONSTEXPR auto operator=(int idx) -> arg_ref& {
    kind = arg_id_kind::index;
    val.index = idx;
    return *this;
  }

  arg_id_kind kind;
  union value {
    FMT_CONSTEXPR value(int id = 0) : index{id} {}
    FMT_CONSTEXPR value(basic_string_view<Char> n) : name(n) {}

    int index;
    basic_string_view<Char> name;
  } val;
};

// Format specifiers with width and precision resolved at formatting rather
// than parsing time to allow re-using the same parsed specifiers with
// different sets of arguments (precompilation of format strings).
template <typename Char>
struct dynamic_format_specs : basic_format_specs<Char> {
  arg_ref<Char> width_ref;
  arg_ref<Char> precision_ref;
};

struct auto_id {};

// A format specifier handler that sets fields in basic_format_specs.
template <typename Char> class specs_setter {
 protected:
  basic_format_specs<Char>& specs_;

 public:
  explicit FMT_CONSTEXPR specs_setter(basic_format_specs<Char>& specs)
      : specs_(specs) {}

  FMT_CONSTEXPR specs_setter(const specs_setter& other)
      : specs_(other.specs_) {}

  FMT_CONSTEXPR void on_align(align_t align) { specs_.align = align; }
  FMT_CONSTEXPR void on_fill(basic_string_view<Char> fill) {
    specs_.fill = fill;
  }
  FMT_CONSTEXPR void on_sign(sign_t s) { specs_.sign = s; }
  FMT_CONSTEXPR void on_hash() { specs_.alt = true; }
  FMT_CONSTEXPR void on_localized() { specs_.localized = true; }

  FMT_CONSTEXPR void on_zero() {
    if (specs_.align == align::none) specs_.align = align::numeric;
    specs_.fill[0] = Char('0');
  }

  FMT_CONSTEXPR void on_width(int width) { specs_.width = width; }
  FMT_CONSTEXPR void on_precision(int precision) {
    specs_.precision = precision;
  }
  FMT_CONSTEXPR void end_precision() {}

  FMT_CONSTEXPR void on_type(presentation_type type) { specs_.type = type; }
};

// Format spec handler that saves references to arguments representing dynamic
// width and precision to be resolved at formatting time.
template <typename ParseContext>
class dynamic_specs_handler
    : public specs_setter<typename ParseContext::char_type> {
 public:
  using char_type = typename ParseContext::char_type;

  FMT_CONSTEXPR dynamic_specs_handler(dynamic_format_specs<char_type>& specs,
                                      ParseContext& ctx)
      : specs_setter<char_type>(specs), specs_(specs), context_(ctx) {}

  FMT_CONSTEXPR dynamic_specs_handler(const dynamic_specs_handler& other)
      : specs_setter<char_type>(other),
        specs_(other.specs_),
        context_(other.context_) {}

  template <typename Id> FMT_CONSTEXPR void on_dynamic_width(Id arg_id) {
    specs_.width_ref = make_arg_ref(arg_id);
  }

  template <typename Id> FMT_CONSTEXPR void on_dynamic_precision(Id arg_id) {
    specs_.precision_ref = make_arg_ref(arg_id);
  }

  FMT_CONSTEXPR void on_error(const char* message) {
    context_.on_error(message);
  }

 private:
  dynamic_format_specs<char_type>& specs_;
  ParseContext& context_;

  using arg_ref_type = arg_ref<char_type>;

  FMT_CONSTEXPR auto make_arg_ref(int arg_id) -> arg_ref_type {
    context_.check_arg_id(arg_id);
    return arg_ref_type(arg_id);
  }

  FMT_CONSTEXPR auto make_arg_ref(auto_id) -> arg_ref_type {
    return arg_ref_type(context_.next_arg_id());
  }

  FMT_CONSTEXPR auto make_arg_ref(basic_string_view<char_type> arg_id)
      -> arg_ref_type {
    context_.check_arg_id(arg_id);
    basic_string_view<char_type> format_str(
        context_.begin(), to_unsigned(context_.end() - context_.begin()));
    return arg_ref_type(arg_id);
  }
};

template <typename Char> constexpr bool is_ascii_letter(Char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Converts a character to ASCII. Returns a number > 127 on conversion failure.
template <typename Char, FMT_ENABLE_IF(std::is_integral<Char>::value)>
constexpr auto to_ascii(Char value) -> Char {
  return value;
}
template <typename Char, FMT_ENABLE_IF(std::is_enum<Char>::value)>
constexpr auto to_ascii(Char value) ->
    typename std::underlying_type<Char>::type {
  return value;
}

template <typename Char>
FMT_CONSTEXPR auto code_point_length(const Char* begin) -> int {
  if (const_check(sizeof(Char) != 1)) return 1;
  auto lengths =
      "\1\1\1\1\1\1\1\1\1\1\1\1\1\1\1\1\0\0\0\0\0\0\0\0\2\2\2\2\3\3\4";
  int len = lengths[static_cast<unsigned char>(*begin) >> 3];

  // Compute the pointer to the next character early so that the next
  // iteration can start working on the next character. Neither Clang
  // nor GCC figure out this reordering on their own.
  return len + !len;
}

// Return the result via the out param to workaround gcc bug 77539.
template <bool IS_CONSTEXPR, typename T, typename Ptr = const T*>
FMT_CONSTEXPR auto find(Ptr first, Ptr last, T value, Ptr& out) -> bool {
  for (out = first; out != last; ++out) {
    if (*out == value) return true;
  }
  return false;
}

template <>
inline auto find<false, char>(const char* first, const char* last, char value,
                              const char*& out) -> bool {
  out = static_cast<const char*>(
      std::memchr(first, value, to_unsigned(last - first)));
  return out != nullptr;
}

// Parses the range [begin, end) as an unsigned integer. This function assumes
// that the range is non-empty and the first character is a digit.
template <typename Char>
FMT_CONSTEXPR auto parse_nonnegative_int(const Char*& begin, const Char* end,
                                         int error_value) noexcept -> int {
  FMT_ASSERT(begin != end && '0' <= *begin && *begin <= '9', "");
  unsigned value = 0, prev = 0;
  auto p = begin;
  do {
    prev = value;
    value = value * 10 + unsigned(*p - '0');
    ++p;
  } while (p != end && '0' <= *p && *p <= '9');
  auto num_digits = p - begin;
  begin = p;
  if (num_digits <= std::numeric_limits<int>::digits10)
    return static_cast<int>(value);
  // Check for overflow.
  const unsigned max = to_unsigned((std::numeric_limits<int>::max)());
  return num_digits == std::numeric_limits<int>::digits10 + 1 &&
                 prev * 10ull + unsigned(p[-1] - '0') <= max
             ? static_cast<int>(value)
             : error_value;
}

// Parses fill and alignment.
template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_align(const Char* begin, const Char* end,
                               Handler&& handler) -> const Char* {
  FMT_ASSERT(begin != end, "");
  auto align = align::none;
  auto p = begin + code_point_length(begin);
  if (p >= end) p = begin;
  for (;;) {
    switch (to_ascii(*p)) {
    case '<':
      align = align::left;
      break;
    case '>':
      align = align::right;
      break;
    case '^':
      align = align::center;
      break;
    default:
      break;
    }
    if (align != align::none) {
      if (p != begin) {
        auto c = *begin;
        if (c == '{')
          return handler.on_error("invalid fill character '{'"), begin;
        handler.on_fill(basic_string_view<Char>(begin, to_unsigned(p - begin)));
        begin = p + 1;
      } else
        ++begin;
      handler.on_align(align);
      break;
    } else if (p == begin) {
      break;
    }
    p = begin;
  }
  return begin;
}

template <typename Char> FMT_CONSTEXPR bool is_name_start(Char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || '_' == c;
}

template <typename Char, typename IDHandler>
FMT_CONSTEXPR auto do_parse_arg_id(const Char* begin, const Char* end,
                                   IDHandler&& handler) -> const Char* {
  FMT_ASSERT(begin != end, "");
  Char c = *begin;
  if (c >= '0' && c <= '9') {
    int index = 0;
    if (c != '0')
      index =
          parse_nonnegative_int(begin, end, (std::numeric_limits<int>::max)());
    else
      ++begin;
    if (begin == end || (*begin != '}' && *begin != ':'))
      handler.on_error("invalid format string");
    else
      handler(index);
    return begin;
  }
  if (!is_name_start(c)) {
    handler.on_error("invalid format string");
    return begin;
  }
  auto it = begin;
  do {
    ++it;
  } while (it != end && (is_name_start(c = *it) || ('0' <= c && c <= '9')));
  handler(basic_string_view<Char>(begin, to_unsigned(it - begin)));
  return it;
}

template <typename Char, typename IDHandler>
FMT_CONSTEXPR FMT_INLINE auto parse_arg_id(const Char* begin, const Char* end,
                                           IDHandler&& handler) -> const Char* {
  Char c = *begin;
  if (c != '}' && c != ':') return do_parse_arg_id(begin, end, handler);
  handler();
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_width(const Char* begin, const Char* end,
                               Handler&& handler) -> const Char* {
  using detail::auto_id;
  struct width_adapter {
    Handler& handler;

    FMT_CONSTEXPR void operator()() { handler.on_dynamic_width(auto_id()); }
    FMT_CONSTEXPR void operator()(int id) { handler.on_dynamic_width(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      handler.on_dynamic_width(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  FMT_ASSERT(begin != end, "");
  if ('0' <= *begin && *begin <= '9') {
    int width = parse_nonnegative_int(begin, end, -1);
    if (width != -1)
      handler.on_width(width);
    else
      handler.on_error("number is too big");
  } else if (*begin == '{') {
    ++begin;
    if (begin != end) begin = parse_arg_id(begin, end, width_adapter{handler});
    if (begin == end || *begin != '}')
      return handler.on_error("invalid format string"), begin;
    ++begin;
  }
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_precision(const Char* begin, const Char* end,
                                   Handler&& handler) -> const Char* {
  using detail::auto_id;
  struct precision_adapter {
    Handler& handler;

    FMT_CONSTEXPR void operator()() { handler.on_dynamic_precision(auto_id()); }
    FMT_CONSTEXPR void operator()(int id) { handler.on_dynamic_precision(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      handler.on_dynamic_precision(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  ++begin;
  auto c = begin != end ? *begin : Char();
  if ('0' <= c && c <= '9') {
    auto precision = parse_nonnegative_int(begin, end, -1);
    if (precision != -1)
      handler.on_precision(precision);
    else
      handler.on_error("number is too big");
  } else if (c == '{') {
    ++begin;
    if (begin != end)
      begin = parse_arg_id(begin, end, precision_adapter{handler});
    if (begin == end || *begin++ != '}')
      return handler.on_error("invalid format string"), begin;
  } else {
    return handler.on_error("missing precision specifier"), begin;
  }
  handler.end_precision();
  return begin;
}

template <typename Char>
FMT_CONSTEXPR auto parse_presentation_type(Char type) -> presentation_type {
  switch (to_ascii(type)) {
  case 'd':
    return presentation_type::dec;
  case 'o':
    return presentation_type::oct;
  case 'x':
    return presentation_type::hex_lower;
  case 'X':
    return presentation_type::hex_upper;
  case 'b':
    return presentation_type::bin_lower;
  case 'B':
    return presentation_type::bin_upper;
  case 'a':
    return presentation_type::hexfloat_lower;
  case 'A':
    return presentation_type::hexfloat_upper;
  case 'e':
    return presentation_type::exp_lower;
  case 'E':
    return presentation_type::exp_upper;
  case 'f':
    return presentation_type::fixed_lower;
  case 'F':
    return presentation_type::fixed_upper;
  case 'g':
    return presentation_type::general_lower;
  case 'G':
    return presentation_type::general_upper;
  case 'c':
    return presentation_type::chr;
  case 's':
    return presentation_type::string;
  case 'p':
    return presentation_type::pointer;
  default:
    return presentation_type::none;
  }
}

// Parses standard format specifiers and sends notifications about parsed
// components to handler.
template <typename Char, typename SpecHandler>
FMT_CONSTEXPR FMT_INLINE auto parse_format_specs(const Char* begin,
                                                 const Char* end,
                                                 SpecHandler&& handler)
    -> const Char* {
  if (1 < end - begin && begin[1] == '}' && is_ascii_letter(*begin) &&
      *begin != 'L') {
    presentation_type type = parse_presentation_type(*begin++);
    if (type == presentation_type::none)
      handler.on_error("invalid type specifier");
    handler.on_type(type);
    return begin;
  }

  if (begin == end) return begin;

  begin = parse_align(begin, end, handler);
  if (begin == end) return begin;

  // Parse sign.
  switch (to_ascii(*begin)) {
  case '+':
    handler.on_sign(sign::plus);
    ++begin;
    break;
  case '-':
    handler.on_sign(sign::minus);
    ++begin;
    break;
  case ' ':
    handler.on_sign(sign::space);
    ++begin;
    break;
  default:
    break;
  }
  if (begin == end) return begin;

  if (*begin == '#') {
    handler.on_hash();
    if (++begin == end) return begin;
  }

  // Parse zero flag.
  if (*begin == '0') {
    handler.on_zero();
    if (++begin == end) return begin;
  }

  begin = parse_width(begin, end, handler);
  if (begin == end) return begin;

  // Parse precision.
  if (*begin == '.') {
    begin = parse_precision(begin, end, handler);
    if (begin == end) return begin;
  }

  if (*begin == 'L') {
    handler.on_localized();
    ++begin;
  }

  // Parse type.
  if (begin != end && *begin != '}') {
    presentation_type type = parse_presentation_type(*begin++);
    if (type == presentation_type::none)
      handler.on_error("invalid type specifier");
    handler.on_type(type);
  }
  return begin;
}

template <typename Char, typename Handler>
FMT_CONSTEXPR auto parse_replacement_field(const Char* begin, const Char* end,
                                           Handler&& handler) -> const Char* {
  struct id_adapter {
    Handler& handler;
    int arg_id;

    FMT_CONSTEXPR void operator()() { arg_id = handler.on_arg_id(); }
    FMT_CONSTEXPR void operator()(int id) { arg_id = handler.on_arg_id(id); }
    FMT_CONSTEXPR void operator()(basic_string_view<Char> id) {
      arg_id = handler.on_arg_id(id);
    }
    FMT_CONSTEXPR void on_error(const char* message) {
      if (message) handler.on_error(message);
    }
  };

  ++begin;
  if (begin == end) return handler.on_error("invalid format string"), end;
  if (*begin == '}') {
    handler.on_replacement_field(handler.on_arg_id(), begin);
  } else if (*begin == '{') {
    handler.on_text(begin, begin + 1);
  } else {
    auto adapter = id_adapter{handler, 0};
    begin = parse_arg_id(begin, end, adapter);
    Char c = begin != end ? *begin : Char();
    if (c == '}') {
      handler.on_replacement_field(adapter.arg_id, begin);
    } else if (c == ':') {
      begin = handler.on_format_specs(adapter.arg_id, begin + 1, end);
      if (begin == end || *begin != '}')
        return handler.on_error("unknown format specifier"), end;
    } else {
      return handler.on_error("missing '}' in format string"), end;
    }
  }
  return begin + 1;
}

template <bool IS_CONSTEXPR, typename Char, typename Handler>
FMT_CONSTEXPR FMT_INLINE void parse_format_string(
    basic_string_view<Char> format_str, Handler&& handler) {
  // Workaround a name-lookup bug in MSVC's modules implementation.
  using detail::find;

  auto begin = format_str.data();
  auto end = begin + format_str.size();
  if (end - begin < 32) {
    // Use a simple loop instead of memchr for small strings.
    const Char* p = begin;
    while (p != end) {
      auto c = *p++;
      if (c == '{') {
        handler.on_text(begin, p - 1);
        begin = p = parse_replacement_field(p - 1, end, handler);
      } else if (c == '}') {
        if (p == end || *p != '}')
          return handler.on_error("unmatched '}' in format string");
        handler.on_text(begin, p);
        begin = ++p;
      }
    }
    handler.on_text(begin, end);
    return;
  }
  struct writer {
    FMT_CONSTEXPR void operator()(const Char* pbegin, const Char* pend) {
      if (pbegin == pend) return;
      for (;;) {
        const Char* p = nullptr;
        if (!find<IS_CONSTEXPR>(pbegin, pend, Char('}'), p))
          return handler_.on_text(pbegin, pend);
        ++p;
        if (p == pend || *p != '}')
          return handler_.on_error("unmatched '}' in format string");
        handler_.on_text(pbegin, p);
        pbegin = p + 1;
      }
    }
    Handler& handler_;
  } write{handler};
  while (begin != end) {
    // Doing two passes with memchr (one for '{' and another for '}') is up to
    // 2.5x faster than the naive one-pass implementation on big format strings.
    const Char* p = begin;
    if (*begin != '{' && !find<IS_CONSTEXPR>(begin + 1, end, Char('{'), p))
      return write(begin, end);
    write(begin, p);
    begin = parse_replacement_field(p, end, handler);
  }
}

template <typename T, typename ParseContext>
FMT_CONSTEXPR auto parse_format_specs(ParseContext& ctx)
    -> decltype(ctx.begin()) {
  using char_type = typename ParseContext::char_type;
  using context = buffer_context<char_type>;
  using mapped_type = conditional_t<
      mapped_type_constant<T, context>::value != type::custom_type,
      decltype(arg_mapper<context>().map(std::declval<const T&>())), T>;
  auto f = conditional_t<has_formatter<mapped_type, context>::value,
                         formatter<mapped_type, char_type>,
                         fallback_formatter<T, char_type>>();
  return f.parse(ctx);
}

// A parse context with extra argument id checks. It is only used at compile
// time because adding checks at runtime would introduce substantial overhead
// and would be redundant since argument ids are checked when arguments are
// retrieved anyway.
template <typename Char, typename ErrorHandler = error_handler>
class compile_parse_context
    : public basic_format_parse_context<Char, ErrorHandler> {
 private:
  int num_args_;
  using base = basic_format_parse_context<Char, ErrorHandler>;

 public:
  explicit FMT_CONSTEXPR compile_parse_context(
      basic_string_view<Char> format_str,
      int num_args = (std::numeric_limits<int>::max)(), ErrorHandler eh = {})
      : base(format_str, eh), num_args_(num_args) {}

  FMT_CONSTEXPR auto next_arg_id() -> int {
    int id = base::next_arg_id();
    if (id >= num_args_) this->on_error("argument not found");
    return id;
  }

  FMT_CONSTEXPR void check_arg_id(int id) {
    base::check_arg_id(id);
    if (id >= num_args_) this->on_error("argument not found");
  }
  using base::check_arg_id;
};

template <typename ErrorHandler>
FMT_CONSTEXPR void check_int_type_spec(presentation_type type,
                                       ErrorHandler&& eh) {
  if (type > presentation_type::bin_upper && type != presentation_type::chr)
    eh.on_error("invalid type specifier");
}

// Checks char specs and returns true if the type spec is char (and not int).
template <typename Char, typename ErrorHandler = error_handler>
FMT_CONSTEXPR auto check_char_specs(const basic_format_specs<Char>& specs,
                                    ErrorHandler&& eh = {}) -> bool {
  if (specs.type != presentation_type::none &&
      specs.type != presentation_type::chr) {
    check_int_type_spec(specs.type, eh);
    return false;
  }
  if (specs.align == align::numeric || specs.sign != sign::none || specs.alt)
    eh.on_error("invalid format specifier for char");
  return true;
}

// A floating-point presentation format.
enum class float_format : unsigned char {
  general,  // General: exponent notation or fixed point based on magnitude.
  exp,      // Exponent notation with the default precision of 6, e.g. 1.2e-3.
  fixed,    // Fixed point with the default precision of 6, e.g. 0.0012.
  hex
};

struct float_specs {
  int precision;
  float_format format : 8;
  sign_t sign : 8;
  bool upper : 1;
  bool locale : 1;
  bool binary32 : 1;
  bool fallback : 1;
  bool showpoint : 1;
};

template <typename ErrorHandler = error_handler, typename Char>
FMT_CONSTEXPR auto parse_float_type_spec(const basic_format_specs<Char>& specs,
                                         ErrorHandler&& eh = {})
    -> float_specs {
  auto result = float_specs();
  result.showpoint = specs.alt;
  result.locale = specs.localized;
  switch (specs.type) {
  case presentation_type::none:
    result.format = float_format::general;
    break;
  case presentation_type::general_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::general_lower:
    result.format = float_format::general;
    break;
  case presentation_type::exp_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::exp_lower:
    result.format = float_format::exp;
    result.showpoint |= specs.precision != 0;
    break;
  case presentation_type::fixed_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::fixed_lower:
    result.format = float_format::fixed;
    result.showpoint |= specs.precision != 0;
    break;
  case presentation_type::hexfloat_upper:
    result.upper = true;
    FMT_FALLTHROUGH;
  case presentation_type::hexfloat_lower:
    result.format = float_format::hex;
    break;
  default:
    eh.on_error("invalid type specifier");
    break;
  }
  return result;
}

template <typename ErrorHandler = error_handler>
FMT_CONSTEXPR auto check_cstring_type_spec(presentation_type type,
                                           ErrorHandler&& eh = {}) -> bool {
  if (type == presentation_type::none || type == presentation_type::string)
    return true;
  if (type != presentation_type::pointer) eh.on_error("invalid type specifier");
  return false;
}

template <typename ErrorHandler = error_handler>
FMT_CONSTEXPR void check_string_type_spec(presentation_type type,
                                          ErrorHandler&& eh = {}) {
  if (type != presentation_type::none && type != presentation_type::string)
    eh.on_error("invalid type specifier");
}

template <typename ErrorHandler>
FMT_CONSTEXPR void check_pointer_type_spec(presentation_type type,
                                           ErrorHandler&& eh) {
  if (type != presentation_type::none && type != presentation_type::pointer)
    eh.on_error("invalid type specifier");
}

// A parse_format_specs handler that checks if specifiers are consistent with
// the argument type.
template <typename Handler> class specs_checker : public Handler {
 private:
  detail::type arg_type_;

  FMT_CONSTEXPR void require_numeric_argument() {
    if (!is_arithmetic_type(arg_type_))
      this->on_error("format specifier requires numeric argument");
  }

 public:
  FMT_CONSTEXPR specs_checker(const Handler& handler, detail::type arg_type)
      : Handler(handler), arg_type_(arg_type) {}

  FMT_CONSTEXPR void on_align(align_t align) {
    if (align == align::numeric) require_numeric_argument();
    Handler::on_align(align);
  }

  FMT_CONSTEXPR void on_sign(sign_t s) {
    require_numeric_argument();
    if (is_integral_type(arg_type_) && arg_type_ != type::int_type &&
        arg_type_ != type::long_long_type && arg_type_ != type::char_type) {
      this->on_error("format specifier requires signed argument");
    }
    Handler::on_sign(s);
  }

  FMT_CONSTEXPR void on_hash() {
    require_numeric_argument();
    Handler::on_hash();
  }

  FMT_CONSTEXPR void on_localized() {
    require_numeric_argument();
    Handler::on_localized();
  }

  FMT_CONSTEXPR void on_zero() {
    require_numeric_argument();
    Handler::on_zero();
  }

  FMT_CONSTEXPR void end_precision() {
    if (is_integral_type(arg_type_) || arg_type_ == type::pointer_type)
      this->on_error("precision not allowed for this argument type");
  }
};

constexpr int invalid_arg_index = -1;

#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <int N, typename T, typename... Args, typename Char>
constexpr auto get_arg_index_by_name(basic_string_view<Char> name) -> int {
  if constexpr (detail::is_statically_named_arg<T>()) {
    if (name == T::name) return N;
  }
  if constexpr (sizeof...(Args) > 0)
    return get_arg_index_by_name<N + 1, Args...>(name);
  (void)name;  // Workaround an MSVC bug about "unused" parameter.
  return invalid_arg_index;
}
#endif

template <typename... Args, typename Char>
FMT_CONSTEXPR auto get_arg_index_by_name(basic_string_view<Char> name) -> int {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
  if constexpr (sizeof...(Args) > 0)
    return get_arg_index_by_name<0, Args...>(name);
#endif
  (void)name;
  return invalid_arg_index;
}

template <typename Char, typename ErrorHandler, typename... Args>
class format_string_checker {
 private:
  using parse_context_type = compile_parse_context<Char, ErrorHandler>;
  enum { num_args = sizeof...(Args) };

  // Format specifier parsing function.
  using parse_func = const Char* (*)(parse_context_type&);

  parse_context_type context_;
  parse_func parse_funcs_[num_args > 0 ? num_args : 1];

 public:
  explicit FMT_CONSTEXPR format_string_checker(
      basic_string_view<Char> format_str, ErrorHandler eh)
      : context_(format_str, num_args, eh),
        parse_funcs_{&parse_format_specs<Args, parse_context_type>...} {}

  FMT_CONSTEXPR void on_text(const Char*, const Char*) {}

  FMT_CONSTEXPR auto on_arg_id() -> int { return context_.next_arg_id(); }
  FMT_CONSTEXPR auto on_arg_id(int id) -> int {
    return context_.check_arg_id(id), id;
  }
  FMT_CONSTEXPR auto on_arg_id(basic_string_view<Char> id) -> int {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
    auto index = get_arg_index_by_name<Args...>(id);
    if (index == invalid_arg_index) on_error("named argument is not found");
    return context_.check_arg_id(index), index;
#else
    (void)id;
    on_error("compile-time checks for named arguments require C++20 support");
    return 0;
#endif
  }

  FMT_CONSTEXPR void on_replacement_field(int, const Char*) {}

  FMT_CONSTEXPR auto on_format_specs(int id, const Char* begin, const Char*)
      -> const Char* {
    context_.advance_to(context_.begin() + (begin - &*context_.begin()));
    // id >= 0 check is a workaround for gcc 10 bug (#2065).
    return id >= 0 && id < num_args ? parse_funcs_[id](context_) : begin;
  }

  FMT_CONSTEXPR void on_error(const char* message) {
    context_.on_error(message);
  }
};

template <typename... Args, typename S,
          enable_if_t<(is_compile_string<S>::value), int>>
void check_format_string(S format_str) {
  FMT_CONSTEXPR auto s = to_string_view(format_str);
  using checker = format_string_checker<typename S::char_type, error_handler,
                                        remove_cvref_t<Args>...>;
  FMT_CONSTEXPR bool invalid_format =
      (parse_format_string<true>(s, checker(s, {})), true);
  ignore_unused(invalid_format);
}

template <typename Char>
void vformat_to(
    buffer<Char>& buf, basic_string_view<Char> fmt,
    basic_format_args<FMT_BUFFER_CONTEXT(type_identity_t<Char>)> args,
    locale_ref loc = {});

FMT_API void vprint_mojibake(std::FILE*, string_view, format_args);
#ifndef _WIN32
inline void vprint_mojibake(std::FILE*, string_view, format_args) {}
#endif
FMT_END_DETAIL_NAMESPACE

// A formatter specialization for the core types corresponding to detail::type
// constants.
template <typename T, typename Char>
struct formatter<T, Char,
                 enable_if_t<detail::type_constant<T, Char>::value !=
                             detail::type::custom_type>> {
 private:
  detail::dynamic_format_specs<Char> specs_;

 public:
  // Parses format specifiers stopping either at the end of the range or at the
  // terminating '}'.
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto begin = ctx.begin(), end = ctx.end();
    if (begin == end) return begin;
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    auto type = detail::type_constant<T, Char>::value;
    auto checker =
        detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
    auto it = detail::parse_format_specs(begin, end, checker);
    auto eh = ctx.error_handler();
    switch (type) {
    case detail::type::none_type:
      FMT_ASSERT(false, "invalid argument type");
      break;
    case detail::type::bool_type:
      if (specs_.type == presentation_type::none ||
          specs_.type == presentation_type::string) {
        break;
      }
      FMT_FALLTHROUGH;
    case detail::type::int_type:
    case detail::type::uint_type:
    case detail::type::long_long_type:
    case detail::type::ulong_long_type:
    case detail::type::int128_type:
    case detail::type::uint128_type:
      detail::check_int_type_spec(specs_.type, eh);
      break;
    case detail::type::char_type:
      detail::check_char_specs(specs_, eh);
      break;
    case detail::type::float_type:
      if (detail::const_check(FMT_USE_FLOAT))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "float support disabled");
      break;
    case detail::type::double_type:
      if (detail::const_check(FMT_USE_DOUBLE))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "double support disabled");
      break;
    case detail::type::long_double_type:
      if (detail::const_check(FMT_USE_LONG_DOUBLE))
        detail::parse_float_type_spec(specs_, eh);
      else
        FMT_ASSERT(false, "long double support disabled");
      break;
    case detail::type::cstring_type:
      detail::check_cstring_type_spec(specs_.type, eh);
      break;
    case detail::type::string_type:
      detail::check_string_type_spec(specs_.type, eh);
      break;
    case detail::type::pointer_type:
      detail::check_pointer_type_spec(specs_.type, eh);
      break;
    case detail::type::custom_type:
      // Custom format specifiers are checked in parse functions of
      // formatter specializations.
      break;
    }
    return it;
  }

  template <typename FormatContext>
  FMT_CONSTEXPR auto format(const T& val, FormatContext& ctx) const
      -> decltype(ctx.out());
};

template <typename Char> struct basic_runtime { basic_string_view<Char> str; };

/** A compile-time format string. */
template <typename Char, typename... Args> class basic_format_string {
 private:
  basic_string_view<Char> str_;

 public:
  template <typename S,
            FMT_ENABLE_IF(
                std::is_convertible<const S&, basic_string_view<Char>>::value)>
  FMT_CONSTEVAL FMT_INLINE basic_format_string(const S& s) : str_(s) {
    static_assert(
        detail::count<
            (std::is_base_of<detail::view, remove_reference_t<Args>>::value &&
             std::is_reference<Args>::value)...>() == 0,
        "passing views as lvalues is disallowed");
#ifdef FMT_HAS_CONSTEVAL
    if constexpr (detail::count_named_args<Args...>() ==
                  detail::count_statically_named_args<Args...>()) {
      using checker = detail::format_string_checker<Char, detail::error_handler,
                                                    remove_cvref_t<Args>...>;
      detail::parse_format_string<true>(str_, checker(s, {}));
    }
#else
    detail::check_format_string<Args...>(s);
#endif
  }
  basic_format_string(basic_runtime<Char> r) : str_(r.str) {}

  FMT_INLINE operator basic_string_view<Char>() const { return str_; }
};

#if FMT_GCC_VERSION && FMT_GCC_VERSION < 409
// Workaround broken conversion on older gcc.
template <typename... Args> using format_string = string_view;
template <typename S> auto runtime(const S& s) -> basic_string_view<char_t<S>> {
  return s;
}
#else
template <typename... Args>
using format_string = basic_format_string<char, type_identity_t<Args>...>;
/**
  \rst
  Creates a runtime format string.

  **Example**::

    // Check format string at runtime instead of compile-time.
    fmt::print(fmt::runtime("{:d}"), "I am not a number");
  \endrst
 */
template <typename S> auto runtime(const S& s) -> basic_runtime<char_t<S>> {
  return {{s}};
}
#endif

FMT_API auto vformat(string_view fmt, format_args args) -> std::string;

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and returns the result
  as a string.

  **Example**::

    #include <fmt/core.h>
    std::string message = fmt::format("The answer is {}.", 42);
  \endrst
*/
template <typename... T>
FMT_NODISCARD FMT_INLINE auto format(format_string<T...> fmt, T&&... args)
    -> std::string {
  return vformat(fmt, fmt::make_format_args(args...));
}

/** Formats a string and writes the output to ``out``. */
template <typename OutputIt,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
auto vformat_to(OutputIt out, string_view fmt, format_args args) -> OutputIt {
  using detail::get_buffer;
  auto&& buf = get_buffer<char>(out);
  detail::vformat_to(buf, fmt, args, {});
  return detail::get_iterator(buf);
}

/**
 \rst
 Formats ``args`` according to specifications in ``fmt``, writes the result to
 the output iterator ``out`` and returns the iterator past the end of the output
 range. `format_to` does not append a terminating null character.

 **Example**::

   auto out = std::vector<char>();
   fmt::format_to(std::back_inserter(out), "{}", 42);
 \endrst
 */
template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
FMT_INLINE auto format_to(OutputIt out, format_string<T...> fmt, T&&... args)
    -> OutputIt {
  return vformat_to(out, fmt, fmt::make_format_args(args...));
}

template <typename OutputIt> struct format_to_n_result {
  /** Iterator past the end of the output range. */
  OutputIt out;
  /** Total (not truncated) output size. */
  size_t size;
};

template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
auto vformat_to_n(OutputIt out, size_t n, string_view fmt, format_args args)
    -> format_to_n_result<OutputIt> {
  using traits = detail::fixed_buffer_traits;
  auto buf = detail::iterator_buffer<OutputIt, char, traits>(out, n);
  detail::vformat_to(buf, fmt, args, {});
  return {buf.out(), buf.count()};
}

/**
  \rst
  Formats ``args`` according to specifications in ``fmt``, writes up to ``n``
  characters of the result to the output iterator ``out`` and returns the total
  (not truncated) output size and the iterator past the end of the output range.
  `format_to_n` does not append a terminating null character.
  \endrst
 */
template <typename OutputIt, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value)>
FMT_INLINE auto format_to_n(OutputIt out, size_t n, format_string<T...> fmt,
                            T&&... args) -> format_to_n_result<OutputIt> {
  return vformat_to_n(out, n, fmt, fmt::make_format_args(args...));
}

/** Returns the number of chars in the output of ``format(fmt, args...)``. */
template <typename... T>
FMT_NODISCARD FMT_INLINE auto formatted_size(format_string<T...> fmt,
                                             T&&... args) -> size_t {
  auto buf = detail::counting_buffer<>();
  detail::vformat_to(buf, string_view(fmt), fmt::make_format_args(args...), {});
  return buf.count();
}

FMT_API void vprint(string_view fmt, format_args args);
FMT_API void vprint(std::FILE* f, string_view fmt, format_args args);

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and writes the output
  to ``stdout``.

  **Example**::

    fmt::print("Elapsed time: {0:.2f} seconds", 1.23);
  \endrst
 */
template <typename... T>
FMT_INLINE void print(format_string<T...> fmt, T&&... args) {
  const auto& vargs = fmt::make_format_args(args...);
  return detail::is_utf8() ? vprint(fmt, vargs)
                           : detail::vprint_mojibake(stdout, fmt, vargs);
}

/**
  \rst
  Formats ``args`` according to specifications in ``fmt`` and writes the
  output to the file ``f``.

  **Example**::

    fmt::print(stderr, "Don't {}!", "panic");
  \endrst
 */
template <typename... T>
FMT_INLINE void print(std::FILE* f, format_string<T...> fmt, T&&... args) {
  const auto& vargs = fmt::make_format_args(args...);
  return detail::is_utf8() ? vprint(f, fmt, vargs)
                           : detail::vprint_mojibake(f, fmt, vargs);
}

FMT_MODULE_EXPORT_END
FMT_GCC_PRAGMA("GCC pop_options")
FMT_END_NAMESPACE

#ifdef FMT_HEADER_ONLY
#  include "format.h"
#endif
#endif  // FMT_CORE_H_
/*
 Formatting library for C++

 Copyright (c) 2012 - present, Victor Zverovich

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 --- Optional exception to the license ---

 As an exception, if, as a result of your compiling your source code, portions
 of this Software are embedded into a machine-executable object form of such
 source code, you may redistribute such embedded portions in such object form
 without including the above copyright and permission notices.
 */

#ifndef FMT_FORMAT_H_
#define FMT_FORMAT_H_

#include <cmath>         // std::signbit
#include <cstdint>       // uint32_t
#include <limits>        // std::numeric_limits
#include <memory>        // std::uninitialized_copy
#include <stdexcept>     // std::runtime_error
#include <system_error>  // std::system_error
#include <utility>       // std::swap

#ifdef __cpp_lib_bit_cast
#  include <bit>  // std::bitcast
#endif


#if FMT_GCC_VERSION
#  define FMT_GCC_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#else
#  define FMT_GCC_VISIBILITY_HIDDEN
#endif

#ifdef __NVCC__
#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
#else
#  define FMT_CUDA_VERSION 0
#endif

#ifdef __has_builtin
#  define FMT_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define FMT_HAS_BUILTIN(x) 0
#endif

#if FMT_GCC_VERSION || FMT_CLANG_VERSION
#  define FMT_NOINLINE __attribute__((noinline))
#else
#  define FMT_NOINLINE
#endif

#if FMT_MSC_VER
#  define FMT_MSC_DEFAULT = default
#else
#  define FMT_MSC_DEFAULT
#endif

#ifndef FMT_THROW
#  if FMT_EXCEPTIONS
#    if FMT_MSC_VER || FMT_NVCC
FMT_BEGIN_NAMESPACE
namespace detail {
template <typename Exception> inline void do_throw(const Exception& x) {
  // Silence unreachable code warnings in MSVC and NVCC because these
  // are nearly impossible to fix in a generic code.
  volatile bool b = true;
  if (b) throw x;
}
}  // namespace detail
FMT_END_NAMESPACE
#      define FMT_THROW(x) detail::do_throw(x)
#    else
#      define FMT_THROW(x) throw x
#    endif
#  else
#    define FMT_THROW(x)               \
      do {                             \
        FMT_ASSERT(false, (x).what()); \
      } while (false)
#  endif
#endif

#if FMT_EXCEPTIONS
#  define FMT_TRY try
#  define FMT_CATCH(x) catch (x)
#else
#  define FMT_TRY if (true)
#  define FMT_CATCH(x) if (false)
#endif

#ifndef FMT_MAYBE_UNUSED
#  if FMT_HAS_CPP17_ATTRIBUTE(maybe_unused)
#    define FMT_MAYBE_UNUSED [[maybe_unused]]
#  else
#    define FMT_MAYBE_UNUSED
#  endif
#endif

// Workaround broken [[deprecated]] in the Intel, PGI and NVCC compilers.
#if FMT_ICC_VERSION || defined(__PGI) || FMT_NVCC
#  define FMT_DEPRECATED_ALIAS
#else
#  define FMT_DEPRECATED_ALIAS FMT_DEPRECATED
#endif

#ifndef FMT_USE_USER_DEFINED_LITERALS
// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
#  if (FMT_HAS_FEATURE(cxx_user_literals) || FMT_GCC_VERSION >= 407 || \
       FMT_MSC_VER >= 1900) &&                                         \
      (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= /* UDL feature */ 480)
#    define FMT_USE_USER_DEFINED_LITERALS 1
#  else
#    define FMT_USE_USER_DEFINED_LITERALS 0
#  endif
#endif

// Defining FMT_REDUCE_INT_INSTANTIATIONS to 1, will reduce the number of
// integer formatter template instantiations to just one by only using the
// largest integer type. This results in a reduction in binary size but will
// cause a decrease in integer formatting performance.
#if !defined(FMT_REDUCE_INT_INSTANTIATIONS)
#  define FMT_REDUCE_INT_INSTANTIATIONS 0
#endif

// __builtin_clz is broken in clang with Microsoft CodeGen:
// https://github.com/fmtlib/fmt/issues/519.
#if !FMT_MSC_VER
#  if FMT_HAS_BUILTIN(__builtin_clz) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CLZ(n) __builtin_clz(n)
#  endif
#  if FMT_HAS_BUILTIN(__builtin_clzll) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CLZLL(n) __builtin_clzll(n)
#  endif
#endif

// __builtin_ctz is broken in Intel Compiler Classic on Windows:
// https://github.com/fmtlib/fmt/issues/2510.
#ifndef __ICL
#  if FMT_HAS_BUILTIN(__builtin_ctz) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CTZ(n) __builtin_ctz(n)
#  endif
#  if FMT_HAS_BUILTIN(__builtin_ctzll) || FMT_GCC_VERSION || FMT_ICC_VERSION
#    define FMT_BUILTIN_CTZLL(n) __builtin_ctzll(n)
#  endif
#endif

#if FMT_MSC_VER
#  include <intrin.h>  // _BitScanReverse[64], _BitScanForward[64], _umul128
#endif

// Some compilers masquerade as both MSVC and GCC-likes or otherwise support
// __builtin_clz and __builtin_clzll, so only define FMT_BUILTIN_CLZ using the
// MSVC intrinsics if the clz and clzll builtins are not available.
#if FMT_MSC_VER && !defined(FMT_BUILTIN_CLZLL) && !defined(FMT_BUILTIN_CTZLL)
FMT_BEGIN_NAMESPACE
namespace detail {
// Avoid Clang with Microsoft CodeGen's -Wunknown-pragmas warning.
#  if !defined(__clang__)
#    pragma intrinsic(_BitScanForward)
#    pragma intrinsic(_BitScanReverse)
#    if defined(_WIN64)
#      pragma intrinsic(_BitScanForward64)
#      pragma intrinsic(_BitScanReverse64)
#    endif
#  endif

inline auto clz(uint32_t x) -> int {
  unsigned long r = 0;
  _BitScanReverse(&r, x);
  FMT_ASSERT(x != 0, "");
  // Static analysis complains about using uninitialized data
  // "r", but the only way that can happen is if "x" is 0,
  // which the callers guarantee to not happen.
  FMT_MSC_WARNING(suppress : 6102)
  return 31 ^ static_cast<int>(r);
}
#  define FMT_BUILTIN_CLZ(n) detail::clz(n)

inline auto clzll(uint64_t x) -> int {
  unsigned long r = 0;
#  ifdef _WIN64
  _BitScanReverse64(&r, x);
#  else
  // Scan the high 32 bits.
  if (_BitScanReverse(&r, static_cast<uint32_t>(x >> 32))) return 63 ^ (r + 32);
  // Scan the low 32 bits.
  _BitScanReverse(&r, static_cast<uint32_t>(x));
#  endif
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
  return 63 ^ static_cast<int>(r);
}
#  define FMT_BUILTIN_CLZLL(n) detail::clzll(n)

inline auto ctz(uint32_t x) -> int {
  unsigned long r = 0;
  _BitScanForward(&r, x);
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
  return static_cast<int>(r);
}
#  define FMT_BUILTIN_CTZ(n) detail::ctz(n)

inline auto ctzll(uint64_t x) -> int {
  unsigned long r = 0;
  FMT_ASSERT(x != 0, "");
  FMT_MSC_WARNING(suppress : 6102)  // Suppress a bogus static analysis warning.
#  ifdef _WIN64
  _BitScanForward64(&r, x);
#  else
  // Scan the low 32 bits.
  if (_BitScanForward(&r, static_cast<uint32_t>(x))) return static_cast<int>(r);
  // Scan the high 32 bits.
  _BitScanForward(&r, static_cast<uint32_t>(x >> 32));
  r += 32;
#  endif
  return static_cast<int>(r);
}
#  define FMT_BUILTIN_CTZLL(n) detail::ctzll(n)
}  // namespace detail
FMT_END_NAMESPACE
#endif

#ifdef FMT_HEADER_ONLY
#  define FMT_HEADER_ONLY_CONSTEXPR20 FMT_CONSTEXPR20
#else
#  define FMT_HEADER_ONLY_CONSTEXPR20
#endif

FMT_BEGIN_NAMESPACE
namespace detail {

template <typename Streambuf> class formatbuf : public Streambuf {
 private:
  using char_type = typename Streambuf::char_type;
  using streamsize = decltype(std::declval<Streambuf>().sputn(nullptr, 0));
  using int_type = typename Streambuf::int_type;
  using traits_type = typename Streambuf::traits_type;

  buffer<char_type>& buffer_;

 public:
  explicit formatbuf(buffer<char_type>& buf) : buffer_(buf) {}

 protected:
  // The put area is always empty. This makes the implementation simpler and has
  // the advantage that the streambuf and the buffer are always in sync and
  // sputc never writes into uninitialized memory. A disadvantage is that each
  // call to sputc always results in a (virtual) call to overflow. There is no
  // disadvantage here for sputn since this always results in a call to xsputn.

  auto overflow(int_type ch) -> int_type override {
    if (!traits_type::eq_int_type(ch, traits_type::eof()))
      buffer_.push_back(static_cast<char_type>(ch));
    return ch;
  }

  auto xsputn(const char_type* s, streamsize count) -> streamsize override {
    buffer_.append(s, s + count);
    return count;
  }
};

// Implementation of std::bit_cast for pre-C++20.
template <typename To, typename From>
FMT_CONSTEXPR20 auto bit_cast(const From& from) -> To {
  static_assert(sizeof(To) == sizeof(From), "size mismatch");
#ifdef __cpp_lib_bit_cast
  if (is_constant_evaluated()) return std::bit_cast<To>(from);
#endif
  auto to = To();
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

inline auto is_big_endian() -> bool {
#ifdef _WIN32
  return false;
#elif defined(__BIG_ENDIAN__)
  return true;
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
  return __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
#else
  struct bytes {
    char data[sizeof(int)];
  };
  return bit_cast<bytes>(1).data[0] == 0;
#endif
}

// A fallback implementation of uintptr_t for systems that lack it.
struct fallback_uintptr {
  unsigned char value[sizeof(void*)];

  fallback_uintptr() = default;
  explicit fallback_uintptr(const void* p) {
    *this = bit_cast<fallback_uintptr>(p);
    if (const_check(is_big_endian())) {
      for (size_t i = 0, j = sizeof(void*) - 1; i < j; ++i, --j)
        std::swap(value[i], value[j]);
    }
  }
};
#ifdef UINTPTR_MAX
using uintptr_t = ::uintptr_t;
inline auto to_uintptr(const void* p) -> uintptr_t {
  return bit_cast<uintptr_t>(p);
}
#else
using uintptr_t = fallback_uintptr;
inline auto to_uintptr(const void* p) -> fallback_uintptr {
  return fallback_uintptr(p);
}
#endif

// Returns the largest possible value for type T. Same as
// std::numeric_limits<T>::max() but shorter and not affected by the max macro.
template <typename T> constexpr auto max_value() -> T {
  return (std::numeric_limits<T>::max)();
}
template <typename T> constexpr auto num_bits() -> int {
  return std::numeric_limits<T>::digits;
}
// std::numeric_limits<T>::digits may return 0 for 128-bit ints.
template <> constexpr auto num_bits<int128_t>() -> int { return 128; }
template <> constexpr auto num_bits<uint128_t>() -> int { return 128; }
template <> constexpr auto num_bits<fallback_uintptr>() -> int {
  return static_cast<int>(sizeof(void*) *
                          std::numeric_limits<unsigned char>::digits);
}

FMT_INLINE void assume(bool condition) {
  (void)condition;
#if FMT_HAS_BUILTIN(__builtin_assume)
  __builtin_assume(condition);
#endif
}

// An approximation of iterator_t for pre-C++20 systems.
template <typename T>
using iterator_t = decltype(std::begin(std::declval<T&>()));
template <typename T> using sentinel_t = decltype(std::end(std::declval<T&>()));

// A workaround for std::string not having mutable data() until C++17.
template <typename Char>
inline auto get_data(std::basic_string<Char>& s) -> Char* {
  return &s[0];
}
template <typename Container>
inline auto get_data(Container& c) -> typename Container::value_type* {
  return c.data();
}

#if defined(_SECURE_SCL) && _SECURE_SCL
// Make a checked iterator to avoid MSVC warnings.
template <typename T> using checked_ptr = stdext::checked_array_iterator<T*>;
template <typename T>
constexpr auto make_checked(T* p, size_t size) -> checked_ptr<T> {
  return {p, size};
}
#else
template <typename T> using checked_ptr = T*;
template <typename T> constexpr auto make_checked(T* p, size_t) -> T* {
  return p;
}
#endif

// Attempts to reserve space for n extra characters in the output range.
// Returns a pointer to the reserved range or a reference to it.
template <typename Container, FMT_ENABLE_IF(is_contiguous<Container>::value)>
#if FMT_CLANG_VERSION >= 307 && !FMT_ICC_VERSION
__attribute__((no_sanitize("undefined")))
#endif
inline auto
reserve(std::back_insert_iterator<Container> it, size_t n)
    -> checked_ptr<typename Container::value_type> {
  Container& c = get_container(it);
  size_t size = c.size();
  c.resize(size + n);
  return make_checked(get_data(c) + size, n);
}

template <typename T>
inline auto reserve(buffer_appender<T> it, size_t n) -> buffer_appender<T> {
  buffer<T>& buf = get_container(it);
  buf.try_reserve(buf.size() + n);
  return it;
}

template <typename Iterator>
constexpr auto reserve(Iterator& it, size_t) -> Iterator& {
  return it;
}

template <typename OutputIt>
using reserve_iterator =
    remove_reference_t<decltype(reserve(std::declval<OutputIt&>(), 0))>;

template <typename T, typename OutputIt>
constexpr auto to_pointer(OutputIt, size_t) -> T* {
  return nullptr;
}
template <typename T> auto to_pointer(buffer_appender<T> it, size_t n) -> T* {
  buffer<T>& buf = get_container(it);
  auto size = buf.size();
  if (buf.capacity() < size + n) return nullptr;
  buf.try_resize(size + n);
  return buf.data() + size;
}

template <typename Container, FMT_ENABLE_IF(is_contiguous<Container>::value)>
inline auto base_iterator(std::back_insert_iterator<Container>& it,
                          checked_ptr<typename Container::value_type>)
    -> std::back_insert_iterator<Container> {
  return it;
}

template <typename Iterator>
constexpr auto base_iterator(Iterator, Iterator it) -> Iterator {
  return it;
}

// <algorithm> is spectacularly slow to compile in C++20 so use a simple fill_n
// instead (#1998).
template <typename OutputIt, typename Size, typename T>
FMT_CONSTEXPR auto fill_n(OutputIt out, Size count, const T& value)
    -> OutputIt {
  for (Size i = 0; i < count; ++i) *out++ = value;
  return out;
}
template <typename T, typename Size>
FMT_CONSTEXPR20 auto fill_n(T* out, Size count, char value) -> T* {
  if (is_constant_evaluated()) {
    return fill_n<T*, Size, T>(out, count, value);
  }
  std::memset(out, value, to_unsigned(count));
  return out + count;
}

#ifdef __cpp_char8_t
using char8_type = char8_t;
#else
enum char8_type : unsigned char {};
#endif

template <typename OutChar, typename InputIt, typename OutputIt>
FMT_CONSTEXPR FMT_NOINLINE auto copy_str_noinline(InputIt begin, InputIt end,
                                                  OutputIt out) -> OutputIt {
  return copy_str<OutChar>(begin, end, out);
}

// A public domain branchless UTF-8 decoder by Christopher Wellons:
// https://github.com/skeeto/branchless-utf8
/* Decode the next character, c, from s, reporting errors in e.
 *
 * Since this is a branchless decoder, four bytes will be read from the
 * buffer regardless of the actual length of the next character. This
 * means the buffer _must_ have at least three bytes of zero padding
 * following the end of the data stream.
 *
 * Errors are reported in e, which will be non-zero if the parsed
 * character was somehow invalid: invalid byte sequence, non-canonical
 * encoding, or a surrogate half.
 *
 * The function returns a pointer to the next character. When an error
 * occurs, this pointer will be a guess that depends on the particular
 * error, but it will always advance at least one byte.
 */
FMT_CONSTEXPR inline auto utf8_decode(const char* s, uint32_t* c, int* e)
    -> const char* {
  constexpr const int masks[] = {0x00, 0x7f, 0x1f, 0x0f, 0x07};
  constexpr const uint32_t mins[] = {4194304, 0, 128, 2048, 65536};
  constexpr const int shiftc[] = {0, 18, 12, 6, 0};
  constexpr const int shifte[] = {0, 6, 4, 2, 0};

  int len = code_point_length(s);
  const char* next = s + len;

  // Assume a four-byte character and load four bytes. Unused bits are
  // shifted out.
  *c = uint32_t(s[0] & masks[len]) << 18;
  *c |= uint32_t(s[1] & 0x3f) << 12;
  *c |= uint32_t(s[2] & 0x3f) << 6;
  *c |= uint32_t(s[3] & 0x3f) << 0;
  *c >>= shiftc[len];

  // Accumulate the various error conditions.
  using uchar = unsigned char;
  *e = (*c < mins[len]) << 6;       // non-canonical encoding
  *e |= ((*c >> 11) == 0x1b) << 7;  // surrogate half?
  *e |= (*c > 0x10FFFF) << 8;       // out of range?
  *e |= (uchar(s[1]) & 0xc0) >> 2;
  *e |= (uchar(s[2]) & 0xc0) >> 4;
  *e |= uchar(s[3]) >> 6;
  *e ^= 0x2a;  // top two bits of each tail byte correct?
  *e >>= shifte[len];

  return next;
}

constexpr uint32_t invalid_code_point = ~uint32_t();

// Invokes f(cp, sv) for every code point cp in s with sv being the string view
// corresponding to the code point. cp is invalid_code_point on error.
template <typename F>
FMT_CONSTEXPR void for_each_codepoint(string_view s, F f) {
  auto decode = [f](const char* buf_ptr, const char* ptr) {
    auto cp = uint32_t();
    auto error = 0;
    auto end = utf8_decode(buf_ptr, &cp, &error);
    bool result = f(error ? invalid_code_point : cp,
                    string_view(ptr, to_unsigned(end - buf_ptr)));
    return result ? end : nullptr;
  };
  auto p = s.data();
  const size_t block_size = 4;  // utf8_decode always reads blocks of 4 chars.
  if (s.size() >= block_size) {
    for (auto end = p + s.size() - block_size + 1; p < end;) {
      p = decode(p, p);
      if (!p) return;
    }
  }
  if (auto num_chars_left = s.data() + s.size() - p) {
    char buf[2 * block_size - 1] = {};
    copy_str<char>(p, p + num_chars_left, buf);
    const char* buf_ptr = buf;
    do {
      auto end = decode(buf_ptr, p);
      if (!end) return;
      p += end - buf_ptr;
      buf_ptr = end;
    } while (buf_ptr - buf < num_chars_left);
  }
}

template <typename Char>
inline auto compute_width(basic_string_view<Char> s) -> size_t {
  return s.size();
}

// Computes approximate display width of a UTF-8 string.
FMT_CONSTEXPR inline size_t compute_width(string_view s) {
  size_t num_code_points = 0;
  // It is not a lambda for compatibility with C++14.
  struct count_code_points {
    size_t* count;
    FMT_CONSTEXPR auto operator()(uint32_t cp, string_view) const -> bool {
      *count += detail::to_unsigned(
          1 +
          (cp >= 0x1100 &&
           (cp <= 0x115f ||  // Hangul Jamo init. consonants
            cp == 0x2329 ||  // LEFT-POINTING ANGLE BRACKET
            cp == 0x232a ||  // RIGHT-POINTING ANGLE BRACKET
            // CJK ... Yi except IDEOGRAPHIC HALF FILL SPACE:
            (cp >= 0x2e80 && cp <= 0xa4cf && cp != 0x303f) ||
            (cp >= 0xac00 && cp <= 0xd7a3) ||    // Hangul Syllables
            (cp >= 0xf900 && cp <= 0xfaff) ||    // CJK Compatibility Ideographs
            (cp >= 0xfe10 && cp <= 0xfe19) ||    // Vertical Forms
            (cp >= 0xfe30 && cp <= 0xfe6f) ||    // CJK Compatibility Forms
            (cp >= 0xff00 && cp <= 0xff60) ||    // Fullwidth Forms
            (cp >= 0xffe0 && cp <= 0xffe6) ||    // Fullwidth Forms
            (cp >= 0x20000 && cp <= 0x2fffd) ||  // CJK
            (cp >= 0x30000 && cp <= 0x3fffd) ||
            // Miscellaneous Symbols and Pictographs + Emoticons:
            (cp >= 0x1f300 && cp <= 0x1f64f) ||
            // Supplemental Symbols and Pictographs:
            (cp >= 0x1f900 && cp <= 0x1f9ff))));
      return true;
    }
  };
  for_each_codepoint(s, count_code_points{&num_code_points});
  return num_code_points;
}

inline auto compute_width(basic_string_view<char8_type> s) -> size_t {
  return compute_width(basic_string_view<char>(
      reinterpret_cast<const char*>(s.data()), s.size()));
}

template <typename Char>
inline auto code_point_index(basic_string_view<Char> s, size_t n) -> size_t {
  size_t size = s.size();
  return n < size ? n : size;
}

// Calculates the index of the nth code point in a UTF-8 string.
inline auto code_point_index(basic_string_view<char8_type> s, size_t n)
    -> size_t {
  const char8_type* data = s.data();
  size_t num_code_points = 0;
  for (size_t i = 0, size = s.size(); i != size; ++i) {
    if ((data[i] & 0xc0) != 0x80 && ++num_code_points > n) return i;
  }
  return s.size();
}

template <typename T, bool = std::is_floating_point<T>::value>
struct is_fast_float : bool_constant<std::numeric_limits<T>::is_iec559 &&
                                     sizeof(T) <= sizeof(double)> {};
template <typename T> struct is_fast_float<T, false> : std::false_type {};

#ifndef FMT_USE_FULL_CACHE_DRAGONBOX
#  define FMT_USE_FULL_CACHE_DRAGONBOX 0
#endif

template <typename T>
template <typename U>
void buffer<T>::append(const U* begin, const U* end) {
  while (begin != end) {
    auto count = to_unsigned(end - begin);
    try_reserve(size_ + count);
    auto free_cap = capacity_ - size_;
    if (free_cap < count) count = free_cap;
    std::uninitialized_copy_n(begin, count, make_checked(ptr_ + size_, count));
    size_ += count;
    begin += count;
  }
}

template <typename T, typename Enable = void>
struct is_locale : std::false_type {};
template <typename T>
struct is_locale<T, void_t<decltype(T::classic())>> : std::true_type {};
}  // namespace detail

FMT_MODULE_EXPORT_BEGIN

// The number of characters to store in the basic_memory_buffer object itself
// to avoid dynamic memory allocation.
enum { inline_buffer_size = 500 };

/**
  \rst
  A dynamically growing memory buffer for trivially copyable/constructible types
  with the first ``SIZE`` elements stored in the object itself.

  You can use the ``memory_buffer`` type alias for ``char`` instead.

  **Example**::

     auto out = fmt::memory_buffer();
     format_to(std::back_inserter(out), "The answer is {}.", 42);

  This will append the following output to the ``out`` object:

  .. code-block:: none

     The answer is 42.

  The output can be converted to an ``std::string`` with ``to_string(out)``.
  \endrst
 */
template <typename T, size_t SIZE = inline_buffer_size,
          typename Allocator = std::allocator<T>>
class basic_memory_buffer final : public detail::buffer<T> {
 private:
  T store_[SIZE];

  // Don't inherit from Allocator avoid generating type_info for it.
  Allocator alloc_;

  // Deallocate memory allocated by the buffer.
  FMT_CONSTEXPR20 void deallocate() {
    T* data = this->data();
    if (data != store_) alloc_.deallocate(data, this->capacity());
  }

 protected:
  FMT_CONSTEXPR20 void grow(size_t size) override;

 public:
  using value_type = T;
  using const_reference = const T&;

  FMT_CONSTEXPR20 explicit basic_memory_buffer(
      const Allocator& alloc = Allocator())
      : alloc_(alloc) {
    this->set(store_, SIZE);
    if (detail::is_constant_evaluated()) {
      detail::fill_n(store_, SIZE, T{});
    }
  }
  FMT_CONSTEXPR20 ~basic_memory_buffer() { deallocate(); }

 private:
  // Move data from other to this buffer.
  FMT_CONSTEXPR20 void move(basic_memory_buffer& other) {
    alloc_ = std::move(other.alloc_);
    T* data = other.data();
    size_t size = other.size(), capacity = other.capacity();
    if (data == other.store_) {
      this->set(store_, capacity);
      if (detail::is_constant_evaluated()) {
        detail::copy_str<T>(other.store_, other.store_ + size,
                            detail::make_checked(store_, capacity));
      } else {
        std::uninitialized_copy(other.store_, other.store_ + size,
                                detail::make_checked(store_, capacity));
      }
    } else {
      this->set(data, capacity);
      // Set pointer to the inline array so that delete is not called
      // when deallocating.
      other.set(other.store_, 0);
    }
    this->resize(size);
  }

 public:
  /**
    \rst
    Constructs a :class:`fmt::basic_memory_buffer` object moving the content
    of the other object to it.
    \endrst
   */
  FMT_CONSTEXPR20 basic_memory_buffer(basic_memory_buffer&& other)
      FMT_NOEXCEPT {
    move(other);
  }

  /**
    \rst
    Moves the content of the other ``basic_memory_buffer`` object to this one.
    \endrst
   */
  auto operator=(basic_memory_buffer&& other) FMT_NOEXCEPT
      -> basic_memory_buffer& {
    FMT_ASSERT(this != &other, "");
    deallocate();
    move(other);
    return *this;
  }

  // Returns a copy of the allocator associated with this buffer.
  auto get_allocator() const -> Allocator { return alloc_; }

  /**
    Resizes the buffer to contain *count* elements. If T is a POD type new
    elements may not be initialized.
   */
  FMT_CONSTEXPR20 void resize(size_t count) { this->try_resize(count); }

  /** Increases the buffer capacity to *new_capacity*. */
  void reserve(size_t new_capacity) { this->try_reserve(new_capacity); }

  // Directly append data into the buffer
  using detail::buffer<T>::append;
  template <typename ContiguousRange>
  void append(const ContiguousRange& range) {
    append(range.data(), range.data() + range.size());
  }
};

template <typename T, size_t SIZE, typename Allocator>
FMT_CONSTEXPR20 void basic_memory_buffer<T, SIZE, Allocator>::grow(
    size_t size) {
#ifdef FMT_FUZZ
  if (size > 5000) throw std::runtime_error("fuzz mode - won't grow that much");
#endif
  const size_t max_size = std::allocator_traits<Allocator>::max_size(alloc_);
  size_t old_capacity = this->capacity();
  size_t new_capacity = old_capacity + old_capacity / 2;
  if (size > new_capacity)
    new_capacity = size;
  else if (new_capacity > max_size)
    new_capacity = size > max_size ? size : max_size;
  T* old_data = this->data();
  T* new_data =
      std::allocator_traits<Allocator>::allocate(alloc_, new_capacity);
  // The following code doesn't throw, so the raw pointer above doesn't leak.
  std::uninitialized_copy(old_data, old_data + this->size(),
                          detail::make_checked(new_data, new_capacity));
  this->set(new_data, new_capacity);
  // deallocate must not throw according to the standard, but even if it does,
  // the buffer already uses the new storage and will deallocate it in
  // destructor.
  if (old_data != store_) alloc_.deallocate(old_data, old_capacity);
}

using memory_buffer = basic_memory_buffer<char>;

template <typename T, size_t SIZE, typename Allocator>
struct is_contiguous<basic_memory_buffer<T, SIZE, Allocator>> : std::true_type {
};

namespace detail {
FMT_API void print(std::FILE*, string_view);
}

/** A formatting error such as invalid format string. */
FMT_CLASS_API
class FMT_API format_error : public std::runtime_error {
 public:
  explicit format_error(const char* message) : std::runtime_error(message) {}
  explicit format_error(const std::string& message)
      : std::runtime_error(message) {}
  format_error(const format_error&) = default;
  format_error& operator=(const format_error&) = default;
  format_error(format_error&&) = default;
  format_error& operator=(format_error&&) = default;
  ~format_error() FMT_NOEXCEPT override FMT_MSC_DEFAULT;
};

/**
  \rst
  Constructs a `~fmt::format_arg_store` object that contains references
  to arguments and can be implicitly converted to `~fmt::format_args`.
  If ``fmt`` is a compile-time string then `make_args_checked` checks
  its validity at compile time.
  \endrst
 */
template <typename... Args, typename S, typename Char = char_t<S>>
FMT_INLINE auto make_args_checked(const S& fmt,
                                  const remove_reference_t<Args>&... args)
    -> format_arg_store<buffer_context<Char>, remove_reference_t<Args>...> {
  static_assert(
      detail::count<(
              std::is_base_of<detail::view, remove_reference_t<Args>>::value &&
              std::is_reference<Args>::value)...>() == 0,
      "passing views as lvalues is disallowed");
  detail::check_format_string<Args...>(fmt);
  return {args...};
}

// compile-time support
namespace detail_exported {
#if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <typename Char, size_t N> struct fixed_string {
  constexpr fixed_string(const Char (&str)[N]) {
    detail::copy_str<Char, const Char*, Char*>(static_cast<const Char*>(str),
                                               str + N, data);
  }
  Char data[N]{};
};
#endif

// Converts a compile-time string to basic_string_view.
template <typename Char, size_t N>
constexpr auto compile_string_to_view(const Char (&s)[N])
    -> basic_string_view<Char> {
  // Remove trailing NUL character if needed. Won't be present if this is used
  // with a raw character array (i.e. not defined as a string).
  return {s, N - (std::char_traits<Char>::to_int_type(s[N - 1]) == 0 ? 1 : 0)};
}
template <typename Char>
constexpr auto compile_string_to_view(detail::std_string_view<Char> s)
    -> basic_string_view<Char> {
  return {s.data(), s.size()};
}
}  // namespace detail_exported

FMT_BEGIN_DETAIL_NAMESPACE

template <typename T> struct is_integral : std::is_integral<T> {};
template <> struct is_integral<int128_t> : std::true_type {};
template <> struct is_integral<uint128_t> : std::true_type {};

template <typename T>
using is_signed =
    std::integral_constant<bool, std::numeric_limits<T>::is_signed ||
                                     std::is_same<T, int128_t>::value>;

// Returns true if value is negative, false otherwise.
// Same as `value < 0` but doesn't produce warnings if T is an unsigned type.
template <typename T, FMT_ENABLE_IF(is_signed<T>::value)>
FMT_CONSTEXPR auto is_negative(T value) -> bool {
  return value < 0;
}
template <typename T, FMT_ENABLE_IF(!is_signed<T>::value)>
FMT_CONSTEXPR auto is_negative(T) -> bool {
  return false;
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR auto is_supported_floating_point(T) -> uint16_t {
  return (std::is_same<T, float>::value && FMT_USE_FLOAT) ||
         (std::is_same<T, double>::value && FMT_USE_DOUBLE) ||
         (std::is_same<T, long double>::value && FMT_USE_LONG_DOUBLE);
}

// Smallest of uint32_t, uint64_t, uint128_t that is large enough to
// represent all values of an integral type T.
template <typename T>
using uint32_or_64_or_128_t =
    conditional_t<num_bits<T>() <= 32 && !FMT_REDUCE_INT_INSTANTIATIONS,
                  uint32_t,
                  conditional_t<num_bits<T>() <= 64, uint64_t, uint128_t>>;
template <typename T>
using uint64_or_128_t = conditional_t<num_bits<T>() <= 64, uint64_t, uint128_t>;

#define FMT_POWERS_OF_10(factor)                                             \
  factor * 10, (factor)*100, (factor)*1000, (factor)*10000, (factor)*100000, \
      (factor)*1000000, (factor)*10000000, (factor)*100000000,               \
      (factor)*1000000000

// Converts value in the range [0, 100) to a string.
constexpr const char* digits2(size_t value) {
  // GCC generates slightly better code when value is pointer-size.
  return &"0001020304050607080910111213141516171819"
         "2021222324252627282930313233343536373839"
         "4041424344454647484950515253545556575859"
         "6061626364656667686970717273747576777879"
         "8081828384858687888990919293949596979899"[value * 2];
}

// Sign is a template parameter to workaround a bug in gcc 4.8.
template <typename Char, typename Sign> constexpr Char sign(Sign s) {
#if !FMT_GCC_VERSION || FMT_GCC_VERSION >= 604
  static_assert(std::is_same<Sign, sign_t>::value, "");
#endif
  return static_cast<Char>("\0-+ "[s]);
}

template <typename T> FMT_CONSTEXPR auto count_digits_fallback(T n) -> int {
  int count = 1;
  for (;;) {
    // Integer division is slow so do it for a group of four digits instead
    // of for every digit. The idea comes from the talk by Alexandrescu
    // "Three Optimization Tips for C++". See speed-test for a comparison.
    if (n < 10) return count;
    if (n < 100) return count + 1;
    if (n < 1000) return count + 2;
    if (n < 10000) return count + 3;
    n /= 10000u;
    count += 4;
  }
}
#if FMT_USE_INT128
FMT_CONSTEXPR inline auto count_digits(uint128_t n) -> int {
  return count_digits_fallback(n);
}
#endif

#ifdef FMT_BUILTIN_CLZLL
// It is a separate function rather than a part of count_digits to workaround
// the lack of static constexpr in constexpr functions.
inline auto do_count_digits(uint64_t n) -> int {
  // This has comparable performance to the version by Kendall Willets
  // (https://github.com/fmtlib/format-benchmark/blob/master/digits10)
  // but uses smaller tables.
  // Maps bsr(n) to ceil(log10(pow(2, bsr(n) + 1) - 1)).
  static constexpr uint8_t bsr2log10[] = {
      1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,
      6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10, 10,
      10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15,
      15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20};
  auto t = bsr2log10[FMT_BUILTIN_CLZLL(n | 1) ^ 63];
  static constexpr const uint64_t zero_or_powers_of_10[] = {
      0, 0, FMT_POWERS_OF_10(1U), FMT_POWERS_OF_10(1000000000ULL),
      10000000000000000000ULL};
  return t - (n < zero_or_powers_of_10[t]);
}
#endif

// Returns the number of decimal digits in n. Leading zeros are not counted
// except for n == 0 in which case count_digits returns 1.
FMT_CONSTEXPR20 inline auto count_digits(uint64_t n) -> int {
#ifdef FMT_BUILTIN_CLZLL
  if (!is_constant_evaluated()) {
    return do_count_digits(n);
  }
#endif
  return count_digits_fallback(n);
}

// Counts the number of digits in n. BITS = log2(radix).
template <int BITS, typename UInt>
FMT_CONSTEXPR auto count_digits(UInt n) -> int {
#ifdef FMT_BUILTIN_CLZ
  if (num_bits<UInt>() == 32)
    return (FMT_BUILTIN_CLZ(static_cast<uint32_t>(n) | 1) ^ 31) / BITS + 1;
#endif
  // Lambda avoids unreachable code warnings from NVHPC.
  return [](UInt m) {
    int num_digits = 0;
    do {
      ++num_digits;
    } while ((m >>= BITS) != 0);
    return num_digits;
  }(n);
}

template <> auto count_digits<4>(detail::fallback_uintptr n) -> int;

#ifdef FMT_BUILTIN_CLZ
// It is a separate function rather than a part of count_digits to workaround
// the lack of static constexpr in constexpr functions.
FMT_INLINE auto do_count_digits(uint32_t n) -> int {
// An optimization by Kendall Willets from https://bit.ly/3uOIQrB.
// This increments the upper 32 bits (log10(T) - 1) when >= T is added.
#  define FMT_INC(T) (((sizeof(#  T) - 1ull) << 32) - T)
  static constexpr uint64_t table[] = {
      FMT_INC(0),          FMT_INC(0),          FMT_INC(0),           // 8
      FMT_INC(10),         FMT_INC(10),         FMT_INC(10),          // 64
      FMT_INC(100),        FMT_INC(100),        FMT_INC(100),         // 512
      FMT_INC(1000),       FMT_INC(1000),       FMT_INC(1000),        // 4096
      FMT_INC(10000),      FMT_INC(10000),      FMT_INC(10000),       // 32k
      FMT_INC(100000),     FMT_INC(100000),     FMT_INC(100000),      // 256k
      FMT_INC(1000000),    FMT_INC(1000000),    FMT_INC(1000000),     // 2048k
      FMT_INC(10000000),   FMT_INC(10000000),   FMT_INC(10000000),    // 16M
      FMT_INC(100000000),  FMT_INC(100000000),  FMT_INC(100000000),   // 128M
      FMT_INC(1000000000), FMT_INC(1000000000), FMT_INC(1000000000),  // 1024M
      FMT_INC(1000000000), FMT_INC(1000000000)                        // 4B
  };
  auto inc = table[FMT_BUILTIN_CLZ(n | 1) ^ 31];
  return static_cast<int>((n + inc) >> 32);
}
#endif

// Optional version of count_digits for better performance on 32-bit platforms.
FMT_CONSTEXPR20 inline auto count_digits(uint32_t n) -> int {
#ifdef FMT_BUILTIN_CLZ
  if (!is_constant_evaluated()) {
    return do_count_digits(n);
  }
#endif
  return count_digits_fallback(n);
}

template <typename Int> constexpr auto digits10() FMT_NOEXCEPT -> int {
  return std::numeric_limits<Int>::digits10;
}
template <> constexpr auto digits10<int128_t>() FMT_NOEXCEPT -> int {
  return 38;
}
template <> constexpr auto digits10<uint128_t>() FMT_NOEXCEPT -> int {
  return 38;
}

template <typename Char> struct thousands_sep_result {
  std::string grouping;
  Char thousands_sep;
};

template <typename Char>
FMT_API auto thousands_sep_impl(locale_ref loc) -> thousands_sep_result<Char>;
template <typename Char>
inline auto thousands_sep(locale_ref loc) -> thousands_sep_result<Char> {
  auto result = thousands_sep_impl<char>(loc);
  return {result.grouping, Char(result.thousands_sep)};
}
template <>
inline auto thousands_sep(locale_ref loc) -> thousands_sep_result<wchar_t> {
  return thousands_sep_impl<wchar_t>(loc);
}

template <typename Char>
FMT_API auto decimal_point_impl(locale_ref loc) -> Char;
template <typename Char> inline auto decimal_point(locale_ref loc) -> Char {
  return Char(decimal_point_impl<char>(loc));
}
template <> inline auto decimal_point(locale_ref loc) -> wchar_t {
  return decimal_point_impl<wchar_t>(loc);
}

// Compares two characters for equality.
template <typename Char> auto equal2(const Char* lhs, const char* rhs) -> bool {
  return lhs[0] == Char(rhs[0]) && lhs[1] == Char(rhs[1]);
}
inline auto equal2(const char* lhs, const char* rhs) -> bool {
  return memcmp(lhs, rhs, 2) == 0;
}

// Copies two characters from src to dst.
template <typename Char>
FMT_CONSTEXPR20 FMT_INLINE void copy2(Char* dst, const char* src) {
  if (!is_constant_evaluated() && sizeof(Char) == sizeof(char)) {
    memcpy(dst, src, 2);
    return;
  }
  *dst++ = static_cast<Char>(*src++);
  *dst = static_cast<Char>(*src);
}

template <typename Iterator> struct format_decimal_result {
  Iterator begin;
  Iterator end;
};

// Formats a decimal unsigned integer value writing into out pointing to a
// buffer of specified size. The caller must ensure that the buffer is large
// enough.
template <typename Char, typename UInt>
FMT_CONSTEXPR20 auto format_decimal(Char* out, UInt value, int size)
    -> format_decimal_result<Char*> {
  FMT_ASSERT(size >= count_digits(value), "invalid digit count");
  out += size;
  Char* end = out;
  while (value >= 100) {
    // Integer division is slow so do it for a group of two digits instead
    // of for every digit. The idea comes from the talk by Alexandrescu
    // "Three Optimization Tips for C++". See speed-test for a comparison.
    out -= 2;
    copy2(out, digits2(static_cast<size_t>(value % 100)));
    value /= 100;
  }
  if (value < 10) {
    *--out = static_cast<Char>('0' + value);
    return {out, end};
  }
  out -= 2;
  copy2(out, digits2(static_cast<size_t>(value)));
  return {out, end};
}

template <typename Char, typename UInt, typename Iterator,
          FMT_ENABLE_IF(!std::is_pointer<remove_cvref_t<Iterator>>::value)>
inline auto format_decimal(Iterator out, UInt value, int size)
    -> format_decimal_result<Iterator> {
  // Buffer is large enough to hold all digits (digits10 + 1).
  Char buffer[digits10<UInt>() + 1];
  auto end = format_decimal(buffer, value, size).end;
  return {out, detail::copy_str_noinline<Char>(buffer, end, out)};
}

template <unsigned BASE_BITS, typename Char, typename UInt>
FMT_CONSTEXPR auto format_uint(Char* buffer, UInt value, int num_digits,
                               bool upper = false) -> Char* {
  buffer += num_digits;
  Char* end = buffer;
  do {
    const char* digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    unsigned digit = (value & ((1 << BASE_BITS) - 1));
    *--buffer = static_cast<Char>(BASE_BITS < 4 ? static_cast<char>('0' + digit)
                                                : digits[digit]);
  } while ((value >>= BASE_BITS) != 0);
  return end;
}

template <unsigned BASE_BITS, typename Char>
auto format_uint(Char* buffer, detail::fallback_uintptr n, int num_digits,
                 bool = false) -> Char* {
  auto char_digits = std::numeric_limits<unsigned char>::digits / 4;
  int start = (num_digits + char_digits - 1) / char_digits - 1;
  if (int start_digits = num_digits % char_digits) {
    unsigned value = n.value[start--];
    buffer = format_uint<BASE_BITS>(buffer, value, start_digits);
  }
  for (; start >= 0; --start) {
    unsigned value = n.value[start];
    buffer += char_digits;
    auto p = buffer;
    for (int i = 0; i < char_digits; ++i) {
      unsigned digit = (value & ((1 << BASE_BITS) - 1));
      *--p = static_cast<Char>("0123456789abcdef"[digit]);
      value >>= BASE_BITS;
    }
  }
  return buffer;
}

template <unsigned BASE_BITS, typename Char, typename It, typename UInt>
inline auto format_uint(It out, UInt value, int num_digits, bool upper = false)
    -> It {
  if (auto ptr = to_pointer<Char>(out, to_unsigned(num_digits))) {
    format_uint<BASE_BITS>(ptr, value, num_digits, upper);
    return out;
  }
  // Buffer should be large enough to hold all digits (digits / BASE_BITS + 1).
  char buffer[num_bits<UInt>() / BASE_BITS + 1];
  format_uint<BASE_BITS>(buffer, value, num_digits, upper);
  return detail::copy_str_noinline<Char>(buffer, buffer + num_digits, out);
}

// A converter from UTF-8 to UTF-16.
class utf8_to_utf16 {
 private:
  basic_memory_buffer<wchar_t> buffer_;

 public:
  FMT_API explicit utf8_to_utf16(string_view s);
  operator basic_string_view<wchar_t>() const { return {&buffer_[0], size()}; }
  auto size() const -> size_t { return buffer_.size() - 1; }
  auto c_str() const -> const wchar_t* { return &buffer_[0]; }
  auto str() const -> std::wstring { return {&buffer_[0], size()}; }
};

namespace dragonbox {

// Type-specific information that Dragonbox uses.
template <class T> struct float_info;

template <> struct float_info<float> {
  using carrier_uint = uint32_t;
  static const int significand_bits = 23;
  static const int exponent_bits = 8;
  static const int min_exponent = -126;
  static const int max_exponent = 127;
  static const int exponent_bias = -127;
  static const int decimal_digits = 9;
  static const int kappa = 1;
  static const int big_divisor = 100;
  static const int small_divisor = 10;
  static const int min_k = -31;
  static const int max_k = 46;
  static const int cache_bits = 64;
  static const int divisibility_check_by_5_threshold = 39;
  static const int case_fc_pm_half_lower_threshold = -1;
  static const int case_fc_pm_half_upper_threshold = 6;
  static const int case_fc_lower_threshold = -2;
  static const int case_fc_upper_threshold = 6;
  static const int case_shorter_interval_left_endpoint_lower_threshold = 2;
  static const int case_shorter_interval_left_endpoint_upper_threshold = 3;
  static const int shorter_interval_tie_lower_threshold = -35;
  static const int shorter_interval_tie_upper_threshold = -35;
  static const int max_trailing_zeros = 7;
};

template <> struct float_info<double> {
  using carrier_uint = uint64_t;
  static const int significand_bits = 52;
  static const int exponent_bits = 11;
  static const int min_exponent = -1022;
  static const int max_exponent = 1023;
  static const int exponent_bias = -1023;
  static const int decimal_digits = 17;
  static const int kappa = 2;
  static const int big_divisor = 1000;
  static const int small_divisor = 100;
  static const int min_k = -292;
  static const int max_k = 326;
  static const int cache_bits = 128;
  static const int divisibility_check_by_5_threshold = 86;
  static const int case_fc_pm_half_lower_threshold = -2;
  static const int case_fc_pm_half_upper_threshold = 9;
  static const int case_fc_lower_threshold = -4;
  static const int case_fc_upper_threshold = 9;
  static const int case_shorter_interval_left_endpoint_lower_threshold = 2;
  static const int case_shorter_interval_left_endpoint_upper_threshold = 3;
  static const int shorter_interval_tie_lower_threshold = -77;
  static const int shorter_interval_tie_upper_threshold = -77;
  static const int max_trailing_zeros = 16;
};

template <typename T> struct decimal_fp {
  using significand_type = typename float_info<T>::carrier_uint;
  significand_type significand;
  int exponent;
};

template <typename T>
FMT_API auto to_decimal(T x) FMT_NOEXCEPT -> decimal_fp<T>;
}  // namespace dragonbox

template <typename T>
constexpr auto exponent_mask() ->
    typename dragonbox::float_info<T>::carrier_uint {
  using uint = typename dragonbox::float_info<T>::carrier_uint;
  return ((uint(1) << dragonbox::float_info<T>::exponent_bits) - 1)
         << dragonbox::float_info<T>::significand_bits;
}

// Writes the exponent exp in the form "[+-]d{2,3}" to buffer.
template <typename Char, typename It>
FMT_CONSTEXPR auto write_exponent(int exp, It it) -> It {
  FMT_ASSERT(-10000 < exp && exp < 10000, "exponent out of range");
  if (exp < 0) {
    *it++ = static_cast<Char>('-');
    exp = -exp;
  } else {
    *it++ = static_cast<Char>('+');
  }
  if (exp >= 100) {
    const char* top = digits2(to_unsigned(exp / 100));
    if (exp >= 1000) *it++ = static_cast<Char>(top[0]);
    *it++ = static_cast<Char>(top[1]);
    exp %= 100;
  }
  const char* d = digits2(to_unsigned(exp));
  *it++ = static_cast<Char>(d[0]);
  *it++ = static_cast<Char>(d[1]);
  return it;
}

template <typename T>
FMT_HEADER_ONLY_CONSTEXPR20 auto format_float(T value, int precision,
                                              float_specs specs,
                                              buffer<char>& buf) -> int;

// Formats a floating-point number with snprintf.
template <typename T>
auto snprintf_float(T value, int precision, float_specs specs,
                    buffer<char>& buf) -> int;

template <typename T> constexpr auto promote_float(T value) -> T {
  return value;
}
constexpr auto promote_float(float value) -> double {
  return static_cast<double>(value);
}

template <typename OutputIt, typename Char>
FMT_NOINLINE FMT_CONSTEXPR auto fill(OutputIt it, size_t n,
                                     const fill_t<Char>& fill) -> OutputIt {
  auto fill_size = fill.size();
  if (fill_size == 1) return detail::fill_n(it, n, fill[0]);
  auto data = fill.data();
  for (size_t i = 0; i < n; ++i)
    it = copy_str<Char>(data, data + fill_size, it);
  return it;
}

// Writes the output of f, padded according to format specifications in specs.
// size: output size in code units.
// width: output display width in (terminal) column positions.
template <align::type align = align::left, typename OutputIt, typename Char,
          typename F>
FMT_CONSTEXPR auto write_padded(OutputIt out,
                                const basic_format_specs<Char>& specs,
                                size_t size, size_t width, F&& f) -> OutputIt {
  static_assert(align == align::left || align == align::right, "");
  unsigned spec_width = to_unsigned(specs.width);
  size_t padding = spec_width > width ? spec_width - width : 0;
  // Shifts are encoded as string literals because static constexpr is not
  // supported in constexpr functions.
  auto* shifts = align == align::left ? "\x1f\x1f\x00\x01" : "\x00\x1f\x00\x01";
  size_t left_padding = padding >> shifts[specs.align];
  size_t right_padding = padding - left_padding;
  auto it = reserve(out, size + padding * specs.fill.size());
  if (left_padding != 0) it = fill(it, left_padding, specs.fill);
  it = f(it);
  if (right_padding != 0) it = fill(it, right_padding, specs.fill);
  return base_iterator(out, it);
}

template <align::type align = align::left, typename OutputIt, typename Char,
          typename F>
constexpr auto write_padded(OutputIt out, const basic_format_specs<Char>& specs,
                            size_t size, F&& f) -> OutputIt {
  return write_padded<align>(out, specs, size, size, f);
}

template <align::type align = align::left, typename Char, typename OutputIt>
FMT_CONSTEXPR auto write_bytes(OutputIt out, string_view bytes,
                               const basic_format_specs<Char>& specs)
    -> OutputIt {
  return write_padded<align>(
      out, specs, bytes.size(), [bytes](reserve_iterator<OutputIt> it) {
        const char* data = bytes.data();
        return copy_str<Char>(data, data + bytes.size(), it);
      });
}

template <typename Char, typename OutputIt, typename UIntPtr>
auto write_ptr(OutputIt out, UIntPtr value,
               const basic_format_specs<Char>* specs) -> OutputIt {
  int num_digits = count_digits<4>(value);
  auto size = to_unsigned(num_digits) + size_t(2);
  auto write = [=](reserve_iterator<OutputIt> it) {
    *it++ = static_cast<Char>('0');
    *it++ = static_cast<Char>('x');
    return format_uint<4, Char>(it, value, num_digits);
  };
  return specs ? write_padded<align::right>(out, *specs, size, write)
               : base_iterator(out, write(reserve(out, size)));
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write_char(OutputIt out, Char value,
                              const basic_format_specs<Char>& specs)
    -> OutputIt {
  return write_padded(out, specs, 1, [=](reserve_iterator<OutputIt> it) {
    *it++ = value;
    return it;
  });
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, Char value,
                         const basic_format_specs<Char>& specs,
                         locale_ref loc = {}) -> OutputIt {
  return check_char_specs(specs)
             ? write_char(out, value, specs)
             : write(out, static_cast<int>(value), specs, loc);
}

// Data for write_int that doesn't depend on output iterator type. It is used to
// avoid template code bloat.
template <typename Char> struct write_int_data {
  size_t size;
  size_t padding;

  FMT_CONSTEXPR write_int_data(int num_digits, unsigned prefix,
                               const basic_format_specs<Char>& specs)
      : size((prefix >> 24) + to_unsigned(num_digits)), padding(0) {
    if (specs.align == align::numeric) {
      auto width = to_unsigned(specs.width);
      if (width > size) {
        padding = width - size;
        size = width;
      }
    } else if (specs.precision > num_digits) {
      size = (prefix >> 24) + to_unsigned(specs.precision);
      padding = to_unsigned(specs.precision - num_digits);
    }
  }
};

// Writes an integer in the format
//   <left-padding><prefix><numeric-padding><digits><right-padding>
// where <digits> are written by write_digits(it).
// prefix contains chars in three lower bytes and the size in the fourth byte.
template <typename OutputIt, typename Char, typename W>
FMT_CONSTEXPR FMT_INLINE auto write_int(OutputIt out, int num_digits,
                                        unsigned prefix,
                                        const basic_format_specs<Char>& specs,
                                        W write_digits) -> OutputIt {
  // Slightly faster check for specs.width == 0 && specs.precision == -1.
  if ((specs.width | (specs.precision + 1)) == 0) {
    auto it = reserve(out, to_unsigned(num_digits) + (prefix >> 24));
    if (prefix != 0) {
      for (unsigned p = prefix & 0xffffff; p != 0; p >>= 8)
        *it++ = static_cast<Char>(p & 0xff);
    }
    return base_iterator(out, write_digits(it));
  }
  auto data = write_int_data<Char>(num_digits, prefix, specs);
  return write_padded<align::right>(
      out, specs, data.size, [=](reserve_iterator<OutputIt> it) {
        for (unsigned p = prefix & 0xffffff; p != 0; p >>= 8)
          *it++ = static_cast<Char>(p & 0xff);
        it = detail::fill_n(it, data.padding, static_cast<Char>('0'));
        return write_digits(it);
      });
}

template <typename Char> class digit_grouping {
 private:
  thousands_sep_result<Char> sep_;

  struct next_state {
    std::string::const_iterator group;
    int pos;
  };
  next_state initial_state() const { return {sep_.grouping.begin(), 0}; }

  // Returns the next digit group separator position.
  int next(next_state& state) const {
    if (!sep_.thousands_sep) return max_value<int>();
    if (state.group == sep_.grouping.end())
      return state.pos += sep_.grouping.back();
    if (*state.group <= 0 || *state.group == max_value<char>())
      return max_value<int>();
    state.pos += *state.group++;
    return state.pos;
  }

 public:
  explicit digit_grouping(locale_ref loc, bool localized = true) {
    if (localized)
      sep_ = thousands_sep<Char>(loc);
    else
      sep_.thousands_sep = Char();
  }
  explicit digit_grouping(thousands_sep_result<Char> sep) : sep_(sep) {}

  Char separator() const { return sep_.thousands_sep; }

  int count_separators(int num_digits) const {
    int count = 0;
    auto state = initial_state();
    while (num_digits > next(state)) ++count;
    return count;
  }

  // Applies grouping to digits and write the output to out.
  template <typename Out, typename C>
  Out apply(Out out, basic_string_view<C> digits) const {
    auto num_digits = static_cast<int>(digits.size());
    auto separators = basic_memory_buffer<int>();
    separators.push_back(0);
    auto state = initial_state();
    while (int i = next(state)) {
      if (i >= num_digits) break;
      separators.push_back(i);
    }
    for (int i = 0, sep_index = static_cast<int>(separators.size() - 1);
         i < num_digits; ++i) {
      if (num_digits - i == separators[sep_index]) {
        *out++ = separator();
        --sep_index;
      }
      *out++ = static_cast<Char>(digits[to_unsigned(i)]);
    }
    return out;
  }
};

template <typename OutputIt, typename UInt, typename Char>
auto write_int_localized(OutputIt out, UInt value, unsigned prefix,
                         const basic_format_specs<Char>& specs,
                         const digit_grouping<Char>& grouping) -> OutputIt {
  static_assert(std::is_same<uint64_or_128_t<UInt>, UInt>::value, "");
  int num_digits = count_digits(value);
  char digits[40];
  format_decimal(digits, value, num_digits);
  unsigned size = to_unsigned((prefix != 0 ? 1 : 0) + num_digits +
                              grouping.count_separators(num_digits));
  return write_padded<align::right>(
      out, specs, size, size, [&](reserve_iterator<OutputIt> it) {
        if (prefix != 0) *it++ = static_cast<Char>(prefix);
        return grouping.apply(it, string_view(digits, to_unsigned(num_digits)));
      });
}

template <typename OutputIt, typename UInt, typename Char>
auto write_int_localized(OutputIt& out, UInt value, unsigned prefix,
                         const basic_format_specs<Char>& specs, locale_ref loc)
    -> bool {
  auto grouping = digit_grouping<Char>(loc);
  out = write_int_localized(out, value, prefix, specs, grouping);
  return true;
}

FMT_CONSTEXPR inline void prefix_append(unsigned& prefix, unsigned value) {
  prefix |= prefix != 0 ? value << 8 : value;
  prefix += (1u + (value > 0xff ? 1 : 0)) << 24;
}

template <typename UInt> struct write_int_arg {
  UInt abs_value;
  unsigned prefix;
};

template <typename T>
FMT_CONSTEXPR auto make_write_int_arg(T value, sign_t sign)
    -> write_int_arg<uint32_or_64_or_128_t<T>> {
  auto prefix = 0u;
  auto abs_value = static_cast<uint32_or_64_or_128_t<T>>(value);
  if (is_negative(value)) {
    prefix = 0x01000000 | '-';
    abs_value = 0 - abs_value;
  } else {
    constexpr const unsigned prefixes[4] = {0, 0, 0x1000000u | '+',
                                            0x1000000u | ' '};
    prefix = prefixes[sign];
  }
  return {abs_value, prefix};
}

template <typename Char, typename OutputIt, typename T>
FMT_CONSTEXPR FMT_INLINE auto write_int(OutputIt out, write_int_arg<T> arg,
                                        const basic_format_specs<Char>& specs,
                                        locale_ref loc) -> OutputIt {
  static_assert(std::is_same<T, uint32_or_64_or_128_t<T>>::value, "");
  auto abs_value = arg.abs_value;
  auto prefix = arg.prefix;
  switch (specs.type) {
  case presentation_type::none:
  case presentation_type::dec: {
    if (specs.localized &&
        write_int_localized(out, static_cast<uint64_or_128_t<T>>(abs_value),
                            prefix, specs, loc)) {
      return out;
    }
    auto num_digits = count_digits(abs_value);
    return write_int(
        out, num_digits, prefix, specs, [=](reserve_iterator<OutputIt> it) {
          return format_decimal<Char>(it, abs_value, num_digits).end;
        });
  }
  case presentation_type::hex_lower:
  case presentation_type::hex_upper: {
    bool upper = specs.type == presentation_type::hex_upper;
    if (specs.alt)
      prefix_append(prefix, unsigned(upper ? 'X' : 'x') << 8 | '0');
    int num_digits = count_digits<4>(abs_value);
    return write_int(
        out, num_digits, prefix, specs, [=](reserve_iterator<OutputIt> it) {
          return format_uint<4, Char>(it, abs_value, num_digits, upper);
        });
  }
  case presentation_type::bin_lower:
  case presentation_type::bin_upper: {
    bool upper = specs.type == presentation_type::bin_upper;
    if (specs.alt)
      prefix_append(prefix, unsigned(upper ? 'B' : 'b') << 8 | '0');
    int num_digits = count_digits<1>(abs_value);
    return write_int(out, num_digits, prefix, specs,
                     [=](reserve_iterator<OutputIt> it) {
                       return format_uint<1, Char>(it, abs_value, num_digits);
                     });
  }
  case presentation_type::oct: {
    int num_digits = count_digits<3>(abs_value);
    // Octal prefix '0' is counted as a digit, so only add it if precision
    // is not greater than the number of digits.
    if (specs.alt && specs.precision <= num_digits && abs_value != 0)
      prefix_append(prefix, '0');
    return write_int(out, num_digits, prefix, specs,
                     [=](reserve_iterator<OutputIt> it) {
                       return format_uint<3, Char>(it, abs_value, num_digits);
                     });
  }
  case presentation_type::chr:
    return write_char(out, static_cast<Char>(abs_value), specs);
  default:
    throw_format_error("invalid type specifier");
  }
  return out;
}
template <typename Char, typename OutputIt, typename T>
FMT_CONSTEXPR FMT_NOINLINE auto write_int_noinline(
    OutputIt out, write_int_arg<T> arg, const basic_format_specs<Char>& specs,
    locale_ref loc) -> OutputIt {
  return write_int(out, arg, specs, loc);
}
template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        std::is_same<OutputIt, buffer_appender<Char>>::value)>
FMT_CONSTEXPR FMT_INLINE auto write(OutputIt out, T value,
                                    const basic_format_specs<Char>& specs,
                                    locale_ref loc) -> OutputIt {
  return write_int_noinline(out, make_write_int_arg(value, specs.sign), specs,
                            loc);
}
// An inlined version of write used in format string compilation.
template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        !std::is_same<OutputIt, buffer_appender<Char>>::value)>
FMT_CONSTEXPR FMT_INLINE auto write(OutputIt out, T value,
                                    const basic_format_specs<Char>& specs,
                                    locale_ref loc) -> OutputIt {
  return write_int(out, make_write_int_arg(value, specs.sign), specs, loc);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, basic_string_view<Char> s,
                         const basic_format_specs<Char>& specs) -> OutputIt {
  auto data = s.data();
  auto size = s.size();
  if (specs.precision >= 0 && to_unsigned(specs.precision) < size)
    size = code_point_index(s, to_unsigned(specs.precision));
  auto width =
      specs.width != 0 ? compute_width(basic_string_view<Char>(data, size)) : 0;
  return write_padded(out, specs, size, width,
                      [=](reserve_iterator<OutputIt> it) {
                        return copy_str<Char>(data, data + size, it);
                      });
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out,
                         basic_string_view<type_identity_t<Char>> s,
                         const basic_format_specs<Char>& specs, locale_ref)
    -> OutputIt {
  check_string_type_spec(specs.type);
  return write(out, s, specs);
}
template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, const Char* s,
                         const basic_format_specs<Char>& specs, locale_ref)
    -> OutputIt {
  return check_cstring_type_spec(specs.type)
             ? write(out, basic_string_view<Char>(s), specs, {})
             : write_ptr<Char>(out, to_uintptr(s), &specs);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR20 auto write_nonfinite(OutputIt out, bool isinf,
                                     basic_format_specs<Char> specs,
                                     const float_specs& fspecs) -> OutputIt {
  auto str =
      isinf ? (fspecs.upper ? "INF" : "inf") : (fspecs.upper ? "NAN" : "nan");
  constexpr size_t str_size = 3;
  auto sign = fspecs.sign;
  auto size = str_size + (sign ? 1 : 0);
  // Replace '0'-padding with space for non-finite values.
  const bool is_zero_fill =
      specs.fill.size() == 1 && *specs.fill.data() == static_cast<Char>('0');
  if (is_zero_fill) specs.fill[0] = static_cast<Char>(' ');
  return write_padded(out, specs, size, [=](reserve_iterator<OutputIt> it) {
    if (sign) *it++ = detail::sign<Char>(sign);
    return copy_str<Char>(str, str + str_size, it);
  });
}

// A decimal floating-point number significand * pow(10, exp).
struct big_decimal_fp {
  const char* significand;
  int significand_size;
  int exponent;
};

constexpr auto get_significand_size(const big_decimal_fp& fp) -> int {
  return fp.significand_size;
}
template <typename T>
inline auto get_significand_size(const dragonbox::decimal_fp<T>& fp) -> int {
  return count_digits(fp.significand);
}

template <typename Char, typename OutputIt>
constexpr auto write_significand(OutputIt out, const char* significand,
                                 int significand_size) -> OutputIt {
  return copy_str<Char>(significand, significand + significand_size, out);
}
template <typename Char, typename OutputIt, typename UInt>
inline auto write_significand(OutputIt out, UInt significand,
                              int significand_size) -> OutputIt {
  return format_decimal<Char>(out, significand, significand_size).end;
}
template <typename Char, typename OutputIt, typename T, typename Grouping>
FMT_CONSTEXPR20 auto write_significand(OutputIt out, T significand,
                                       int significand_size, int exponent,
                                       const Grouping& grouping) -> OutputIt {
  if (!grouping.separator()) {
    out = write_significand<Char>(out, significand, significand_size);
    return detail::fill_n(out, exponent, static_cast<Char>('0'));
  }
  auto buffer = memory_buffer();
  write_significand<char>(appender(buffer), significand, significand_size);
  detail::fill_n(appender(buffer), exponent, '0');
  return grouping.apply(out, string_view(buffer.data(), buffer.size()));
}

template <typename Char, typename UInt,
          FMT_ENABLE_IF(std::is_integral<UInt>::value)>
inline auto write_significand(Char* out, UInt significand, int significand_size,
                              int integral_size, Char decimal_point) -> Char* {
  if (!decimal_point)
    return format_decimal(out, significand, significand_size).end;
  out += significand_size + 1;
  Char* end = out;
  int floating_size = significand_size - integral_size;
  for (int i = floating_size / 2; i > 0; --i) {
    out -= 2;
    copy2(out, digits2(significand % 100));
    significand /= 100;
  }
  if (floating_size % 2 != 0) {
    *--out = static_cast<Char>('0' + significand % 10);
    significand /= 10;
  }
  *--out = decimal_point;
  format_decimal(out - integral_size, significand, integral_size);
  return end;
}

template <typename OutputIt, typename UInt, typename Char,
          FMT_ENABLE_IF(!std::is_pointer<remove_cvref_t<OutputIt>>::value)>
inline auto write_significand(OutputIt out, UInt significand,
                              int significand_size, int integral_size,
                              Char decimal_point) -> OutputIt {
  // Buffer is large enough to hold digits (digits10 + 1) and a decimal point.
  Char buffer[digits10<UInt>() + 2];
  auto end = write_significand(buffer, significand, significand_size,
                               integral_size, decimal_point);
  return detail::copy_str_noinline<Char>(buffer, end, out);
}

template <typename OutputIt, typename Char>
FMT_CONSTEXPR auto write_significand(OutputIt out, const char* significand,
                                     int significand_size, int integral_size,
                                     Char decimal_point) -> OutputIt {
  out = detail::copy_str_noinline<Char>(significand,
                                        significand + integral_size, out);
  if (!decimal_point) return out;
  *out++ = decimal_point;
  return detail::copy_str_noinline<Char>(significand + integral_size,
                                         significand + significand_size, out);
}

template <typename OutputIt, typename Char, typename T, typename Grouping>
FMT_CONSTEXPR20 auto write_significand(OutputIt out, T significand,
                                       int significand_size, int integral_size,
                                       Char decimal_point,
                                       const Grouping& grouping) -> OutputIt {
  if (!grouping.separator()) {
    return write_significand(out, significand, significand_size, integral_size,
                             decimal_point);
  }
  auto buffer = basic_memory_buffer<Char>();
  write_significand(buffer_appender<Char>(buffer), significand,
                    significand_size, integral_size, decimal_point);
  grouping.apply(
      out, basic_string_view<Char>(buffer.data(), to_unsigned(integral_size)));
  return detail::copy_str_noinline<Char>(buffer.data() + integral_size,
                                         buffer.end(), out);
}

template <typename OutputIt, typename DecimalFP, typename Char,
          typename Grouping = digit_grouping<Char>>
FMT_CONSTEXPR20 auto do_write_float(OutputIt out, const DecimalFP& fp,
                                    const basic_format_specs<Char>& specs,
                                    float_specs fspecs, locale_ref loc)
    -> OutputIt {
  auto significand = fp.significand;
  int significand_size = get_significand_size(fp);
  constexpr Char zero = static_cast<Char>('0');
  auto sign = fspecs.sign;
  size_t size = to_unsigned(significand_size) + (sign ? 1 : 0);
  using iterator = reserve_iterator<OutputIt>;

  Char decimal_point =
      fspecs.locale ? detail::decimal_point<Char>(loc) : static_cast<Char>('.');

  int output_exp = fp.exponent + significand_size - 1;
  auto use_exp_format = [=]() {
    if (fspecs.format == float_format::exp) return true;
    if (fspecs.format != float_format::general) return false;
    // Use the fixed notation if the exponent is in [exp_lower, exp_upper),
    // e.g. 0.0001 instead of 1e-04. Otherwise use the exponent notation.
    const int exp_lower = -4, exp_upper = 16;
    return output_exp < exp_lower ||
           output_exp >= (fspecs.precision > 0 ? fspecs.precision : exp_upper);
  };
  if (use_exp_format()) {
    int num_zeros = 0;
    if (fspecs.showpoint) {
      num_zeros = fspecs.precision - significand_size;
      if (num_zeros < 0) num_zeros = 0;
      size += to_unsigned(num_zeros);
    } else if (significand_size == 1) {
      decimal_point = Char();
    }
    auto abs_output_exp = output_exp >= 0 ? output_exp : -output_exp;
    int exp_digits = 2;
    if (abs_output_exp >= 100) exp_digits = abs_output_exp >= 1000 ? 4 : 3;

    size += to_unsigned((decimal_point ? 1 : 0) + 2 + exp_digits);
    char exp_char = fspecs.upper ? 'E' : 'e';
    auto write = [=](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      // Insert a decimal point after the first digit and add an exponent.
      it = write_significand(it, significand, significand_size, 1,
                             decimal_point);
      if (num_zeros > 0) it = detail::fill_n(it, num_zeros, zero);
      *it++ = static_cast<Char>(exp_char);
      return write_exponent<Char>(output_exp, it);
    };
    return specs.width > 0 ? write_padded<align::right>(out, specs, size, write)
                           : base_iterator(out, write(reserve(out, size)));
  }

  int exp = fp.exponent + significand_size;
  if (fp.exponent >= 0) {
    // 1234e5 -> 123400000[.0+]
    size += to_unsigned(fp.exponent);
    int num_zeros = fspecs.precision - exp;
#ifdef FMT_FUZZ
    if (num_zeros > 5000)
      throw std::runtime_error("fuzz mode - avoiding excessive cpu use");
#endif
    if (fspecs.showpoint) {
      if (num_zeros <= 0 && fspecs.format != float_format::fixed) num_zeros = 1;
      if (num_zeros > 0) size += to_unsigned(num_zeros) + 1;
    }
    auto grouping = Grouping(loc, fspecs.locale);
    size += to_unsigned(grouping.count_separators(significand_size));
    return write_padded<align::right>(out, specs, size, [&](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      it = write_significand<Char>(it, significand, significand_size,
                                   fp.exponent, grouping);
      if (!fspecs.showpoint) return it;
      *it++ = decimal_point;
      return num_zeros > 0 ? detail::fill_n(it, num_zeros, zero) : it;
    });
  } else if (exp > 0) {
    // 1234e-2 -> 12.34[0+]
    int num_zeros = fspecs.showpoint ? fspecs.precision - significand_size : 0;
    size += 1 + to_unsigned(num_zeros > 0 ? num_zeros : 0);
    auto grouping = Grouping(loc, fspecs.locale);
    size += to_unsigned(grouping.count_separators(significand_size));
    return write_padded<align::right>(out, specs, size, [&](iterator it) {
      if (sign) *it++ = detail::sign<Char>(sign);
      it = write_significand(it, significand, significand_size, exp,
                             decimal_point, grouping);
      return num_zeros > 0 ? detail::fill_n(it, num_zeros, zero) : it;
    });
  }
  // 1234e-6 -> 0.001234
  int num_zeros = -exp;
  if (significand_size == 0 && fspecs.precision >= 0 &&
      fspecs.precision < num_zeros) {
    num_zeros = fspecs.precision;
  }
  bool pointy = num_zeros != 0 || significand_size != 0 || fspecs.showpoint;
  size += 1 + (pointy ? 1 : 0) + to_unsigned(num_zeros);
  return write_padded<align::right>(out, specs, size, [&](iterator it) {
    if (sign) *it++ = detail::sign<Char>(sign);
    *it++ = zero;
    if (!pointy) return it;
    *it++ = decimal_point;
    it = detail::fill_n(it, num_zeros, zero);
    return write_significand<Char>(it, significand, significand_size);
  });
}

template <typename Char> class fallback_digit_grouping {
 public:
  constexpr fallback_digit_grouping(locale_ref, bool) {}

  constexpr Char separator() const { return Char(); }

  constexpr int count_separators(int) const { return 0; }

  template <typename Out, typename C>
  constexpr Out apply(Out out, basic_string_view<C>) const {
    return out;
  }
};

template <typename OutputIt, typename DecimalFP, typename Char>
FMT_CONSTEXPR20 auto write_float(OutputIt out, const DecimalFP& fp,
                                 const basic_format_specs<Char>& specs,
                                 float_specs fspecs, locale_ref loc)
    -> OutputIt {
  if (is_constant_evaluated()) {
    return do_write_float<OutputIt, DecimalFP, Char,
                          fallback_digit_grouping<Char>>(out, fp, specs, fspecs,
                                                         loc);
  } else {
    return do_write_float(out, fp, specs, fspecs, loc);
  }
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 bool isinf(T value) {
  if (is_constant_evaluated()) {
#if defined(__cpp_if_constexpr)
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      constexpr auto significand_bits =
          dragonbox::float_info<double>::significand_bits;
      return (bits & exponent_mask<double>()) &&
             !(bits & ((uint64_t(1) << significand_bits) - 1));
    }
#endif
  }
  return std::isinf(value);
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 bool isfinite(T value) {
  if (is_constant_evaluated()) {
#if defined(__cpp_if_constexpr)
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      return (bits & exponent_mask<double>()) != exponent_mask<double>();
    }
#endif
  }
  return std::isfinite(value);
}

template <typename T, FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_INLINE FMT_CONSTEXPR bool signbit(T value) {
  if (is_constant_evaluated()) {
#ifdef __cpp_if_constexpr
    if constexpr (std::numeric_limits<double>::is_iec559) {
      auto bits = detail::bit_cast<uint64_t>(static_cast<double>(value));
      return (bits & (uint64_t(1) << (num_bits<uint64_t>() - 1))) != 0;
    }
#endif
  }
  return std::signbit(value);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_floating_point<T>::value)>
FMT_CONSTEXPR20 auto write(OutputIt out, T value,
                           basic_format_specs<Char> specs, locale_ref loc = {})
    -> OutputIt {
  if (const_check(!is_supported_floating_point(value))) return out;
  float_specs fspecs = parse_float_type_spec(specs);
  fspecs.sign = specs.sign;
  if (detail::signbit(value)) {  // value < 0 is false for NaN so use signbit.
    fspecs.sign = sign::minus;
    value = -value;
  } else if (fspecs.sign == sign::minus) {
    fspecs.sign = sign::none;
  }

  if (!detail::isfinite(value))
    return write_nonfinite(out, detail::isinf(value), specs, fspecs);

  if (specs.align == align::numeric && fspecs.sign) {
    auto it = reserve(out, 1);
    *it++ = detail::sign<Char>(fspecs.sign);
    out = base_iterator(out, it);
    fspecs.sign = sign::none;
    if (specs.width != 0) --specs.width;
  }

  memory_buffer buffer;
  if (fspecs.format == float_format::hex) {
    if (fspecs.sign) buffer.push_back(detail::sign<char>(fspecs.sign));
    snprintf_float(promote_float(value), specs.precision, fspecs, buffer);
    return write_bytes<align::right>(out, {buffer.data(), buffer.size()},
                                     specs);
  }
  int precision = specs.precision >= 0 || specs.type == presentation_type::none
                      ? specs.precision
                      : 6;
  if (fspecs.format == float_format::exp) {
    if (precision == max_value<int>())
      throw_format_error("number is too big");
    else
      ++precision;
  }
  if (const_check(std::is_same<T, float>())) fspecs.binary32 = true;
  if (!is_fast_float<T>()) fspecs.fallback = true;
  int exp = format_float(promote_float(value), precision, fspecs, buffer);
  fspecs.precision = precision;
  auto fp = big_decimal_fp{buffer.data(), static_cast<int>(buffer.size()), exp};
  return write_float(out, fp, specs, fspecs, loc);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_fast_float<T>::value)>
FMT_CONSTEXPR20 auto write(OutputIt out, T value) -> OutputIt {
  if (is_constant_evaluated()) {
    return write(out, value, basic_format_specs<Char>());
  }

  if (const_check(!is_supported_floating_point(value))) return out;

  using floaty = conditional_t<std::is_same<T, long double>::value, double, T>;
  using uint = typename dragonbox::float_info<floaty>::carrier_uint;
  auto bits = bit_cast<uint>(value);

  auto fspecs = float_specs();
  if (detail::signbit(value)) {
    fspecs.sign = sign::minus;
    value = -value;
  }

  constexpr auto specs = basic_format_specs<Char>();
  uint mask = exponent_mask<floaty>();
  if ((bits & mask) == mask)
    return write_nonfinite(out, std::isinf(value), specs, fspecs);

  auto dec = dragonbox::to_decimal(static_cast<floaty>(value));
  return write_float(out, dec, specs, fspecs, {});
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_floating_point<T>::value &&
                        !is_fast_float<T>::value)>
inline auto write(OutputIt out, T value) -> OutputIt {
  return write(out, value, basic_format_specs<Char>());
}

template <typename Char, typename OutputIt>
auto write(OutputIt out, monostate, basic_format_specs<Char> = {},
           locale_ref = {}) -> OutputIt {
  FMT_ASSERT(false, "");
  return out;
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, basic_string_view<Char> value)
    -> OutputIt {
  auto it = reserve(out, value.size());
  it = copy_str_noinline<Char>(value.begin(), value.end(), it);
  return base_iterator(out, it);
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_string<T>::value)>
constexpr auto write(OutputIt out, const T& value) -> OutputIt {
  return write<Char>(out, to_string_view(value));
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        !std::is_same<T, Char>::value)>
FMT_CONSTEXPR auto write(OutputIt out, T value) -> OutputIt {
  auto abs_value = static_cast<uint32_or_64_or_128_t<T>>(value);
  bool negative = is_negative(value);
  // Don't do -abs_value since it trips unsigned-integer-overflow sanitizer.
  if (negative) abs_value = ~abs_value + 1;
  int num_digits = count_digits(abs_value);
  auto size = (negative ? 1 : 0) + static_cast<size_t>(num_digits);
  auto it = reserve(out, size);
  if (auto ptr = to_pointer<Char>(it, size)) {
    if (negative) *ptr++ = static_cast<Char>('-');
    format_decimal<Char>(ptr, abs_value, num_digits);
    return out;
  }
  if (negative) *it++ = static_cast<Char>('-');
  it = format_decimal<Char>(it, abs_value, num_digits).end;
  return base_iterator(out, it);
}

// FMT_ENABLE_IF() condition separated to workaround an MSVC bug.
template <
    typename Char, typename OutputIt, typename T,
    bool check =
        std::is_enum<T>::value && !std::is_same<T, Char>::value &&
        mapped_type_constant<T, basic_format_context<OutputIt, Char>>::value !=
            type::custom_type,
    FMT_ENABLE_IF(check)>
FMT_CONSTEXPR auto write(OutputIt out, T value) -> OutputIt {
  return write<Char>(
      out, static_cast<typename std::underlying_type<T>::type>(value));
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_same<T, bool>::value)>
FMT_CONSTEXPR auto write(OutputIt out, T value,
                         const basic_format_specs<Char>& specs = {},
                         locale_ref = {}) -> OutputIt {
  return specs.type != presentation_type::none &&
                 specs.type != presentation_type::string
             ? write(out, value ? 1 : 0, specs, {})
             : write_bytes(out, value ? "true" : "false", specs);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR auto write(OutputIt out, Char value) -> OutputIt {
  auto it = reserve(out, 1);
  *it++ = value;
  return base_iterator(out, it);
}

template <typename Char, typename OutputIt>
FMT_CONSTEXPR_CHAR_TRAITS auto write(OutputIt out, const Char* value)
    -> OutputIt {
  if (!value) {
    throw_format_error("string pointer is null");
  } else {
    out = write(out, basic_string_view<Char>(value));
  }
  return out;
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_same<T, void>::value)>
auto write(OutputIt out, const T* value,
           const basic_format_specs<Char>& specs = {}, locale_ref = {})
    -> OutputIt {
  check_pointer_type_spec(specs.type, error_handler());
  return write_ptr<Char>(out, to_uintptr(value), &specs);
}

// A write overload that handles implicit conversions.
template <typename Char, typename OutputIt, typename T,
          typename Context = basic_format_context<OutputIt, Char>>
FMT_CONSTEXPR auto write(OutputIt out, const T& value) -> enable_if_t<
    std::is_class<T>::value && !is_string<T>::value &&
        !std::is_same<T, Char>::value &&
        !std::is_same<const T&,
                      decltype(arg_mapper<Context>().map(value))>::value,
    OutputIt> {
  return write<Char>(out, arg_mapper<Context>().map(value));
}

template <typename Char, typename OutputIt, typename T,
          typename Context = basic_format_context<OutputIt, Char>>
FMT_CONSTEXPR auto write(OutputIt out, const T& value)
    -> enable_if_t<mapped_type_constant<T, Context>::value == type::custom_type,
                   OutputIt> {
  using formatter_type =
      conditional_t<has_formatter<T, Context>::value,
                    typename Context::template formatter_type<T>,
                    fallback_formatter<T, Char>>;
  auto ctx = Context(out, {}, {});
  return formatter_type().format(value, ctx);
}

// An argument visitor that formats the argument and writes it via the output
// iterator. It's a class and not a generic lambda for compatibility with C++11.
template <typename Char> struct default_arg_formatter {
  using iterator = buffer_appender<Char>;
  using context = buffer_context<Char>;

  iterator out;
  basic_format_args<context> args;
  locale_ref loc;

  template <typename T> auto operator()(T value) -> iterator {
    return write<Char>(out, value);
  }
  auto operator()(typename basic_format_arg<context>::handle h) -> iterator {
    basic_format_parse_context<Char> parse_ctx({});
    context format_ctx(out, args, loc);
    h.format(parse_ctx, format_ctx);
    return format_ctx.out();
  }
};

template <typename Char> struct arg_formatter {
  using iterator = buffer_appender<Char>;
  using context = buffer_context<Char>;

  iterator out;
  const basic_format_specs<Char>& specs;
  locale_ref locale;

  template <typename T>
  FMT_CONSTEXPR FMT_INLINE auto operator()(T value) -> iterator {
    return detail::write(out, value, specs, locale);
  }
  auto operator()(typename basic_format_arg<context>::handle) -> iterator {
    // User-defined types are handled separately because they require access
    // to the parse context.
    return out;
  }
};

template <typename Char> struct custom_formatter {
  basic_format_parse_context<Char>& parse_ctx;
  buffer_context<Char>& ctx;

  void operator()(
      typename basic_format_arg<buffer_context<Char>>::handle h) const {
    h.format(parse_ctx, ctx);
  }
  template <typename T> void operator()(T) const {}
};

template <typename T>
using is_integer =
    bool_constant<is_integral<T>::value && !std::is_same<T, bool>::value &&
                  !std::is_same<T, char>::value &&
                  !std::is_same<T, wchar_t>::value>;

template <typename ErrorHandler> class width_checker {
 public:
  explicit FMT_CONSTEXPR width_checker(ErrorHandler& eh) : handler_(eh) {}

  template <typename T, FMT_ENABLE_IF(is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T value) -> unsigned long long {
    if (is_negative(value)) handler_.on_error("negative width");
    return static_cast<unsigned long long>(value);
  }

  template <typename T, FMT_ENABLE_IF(!is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T) -> unsigned long long {
    handler_.on_error("width is not integer");
    return 0;
  }

 private:
  ErrorHandler& handler_;
};

template <typename ErrorHandler> class precision_checker {
 public:
  explicit FMT_CONSTEXPR precision_checker(ErrorHandler& eh) : handler_(eh) {}

  template <typename T, FMT_ENABLE_IF(is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T value) -> unsigned long long {
    if (is_negative(value)) handler_.on_error("negative precision");
    return static_cast<unsigned long long>(value);
  }

  template <typename T, FMT_ENABLE_IF(!is_integer<T>::value)>
  FMT_CONSTEXPR auto operator()(T) -> unsigned long long {
    handler_.on_error("precision is not integer");
    return 0;
  }

 private:
  ErrorHandler& handler_;
};

template <template <typename> class Handler, typename FormatArg,
          typename ErrorHandler>
FMT_CONSTEXPR auto get_dynamic_spec(FormatArg arg, ErrorHandler eh) -> int {
  unsigned long long value = visit_format_arg(Handler<ErrorHandler>(eh), arg);
  if (value > to_unsigned(max_value<int>())) eh.on_error("number is too big");
  return static_cast<int>(value);
}

template <typename Context, typename ID>
FMT_CONSTEXPR auto get_arg(Context& ctx, ID id) ->
    typename Context::format_arg {
  auto arg = ctx.arg(id);
  if (!arg) ctx.on_error("argument not found");
  return arg;
}

// The standard format specifier handler with checking.
template <typename Char> class specs_handler : public specs_setter<Char> {
 private:
  basic_format_parse_context<Char>& parse_context_;
  buffer_context<Char>& context_;

  // This is only needed for compatibility with gcc 4.4.
  using format_arg = basic_format_arg<buffer_context<Char>>;

  FMT_CONSTEXPR auto get_arg(auto_id) -> format_arg {
    return detail::get_arg(context_, parse_context_.next_arg_id());
  }

  FMT_CONSTEXPR auto get_arg(int arg_id) -> format_arg {
    parse_context_.check_arg_id(arg_id);
    return detail::get_arg(context_, arg_id);
  }

  FMT_CONSTEXPR auto get_arg(basic_string_view<Char> arg_id) -> format_arg {
    parse_context_.check_arg_id(arg_id);
    return detail::get_arg(context_, arg_id);
  }

 public:
  FMT_CONSTEXPR specs_handler(basic_format_specs<Char>& specs,
                              basic_format_parse_context<Char>& parse_ctx,
                              buffer_context<Char>& ctx)
      : specs_setter<Char>(specs), parse_context_(parse_ctx), context_(ctx) {}

  template <typename Id> FMT_CONSTEXPR void on_dynamic_width(Id arg_id) {
    this->specs_.width = get_dynamic_spec<width_checker>(
        get_arg(arg_id), context_.error_handler());
  }

  template <typename Id> FMT_CONSTEXPR void on_dynamic_precision(Id arg_id) {
    this->specs_.precision = get_dynamic_spec<precision_checker>(
        get_arg(arg_id), context_.error_handler());
  }

  void on_error(const char* message) { context_.on_error(message); }
};

template <template <typename> class Handler, typename Context>
FMT_CONSTEXPR void handle_dynamic_spec(int& value,
                                       arg_ref<typename Context::char_type> ref,
                                       Context& ctx) {
  switch (ref.kind) {
  case arg_id_kind::none:
    break;
  case arg_id_kind::index:
    value = detail::get_dynamic_spec<Handler>(ctx.arg(ref.val.index),
                                              ctx.error_handler());
    break;
  case arg_id_kind::name:
    value = detail::get_dynamic_spec<Handler>(ctx.arg(ref.val.name),
                                              ctx.error_handler());
    break;
  }
}

#define FMT_STRING_IMPL(s, base, explicit)                                 \
  [] {                                                                     \
    /* Use the hidden visibility as a workaround for a GCC bug (#1973). */ \
    /* Use a macro-like name to avoid shadowing warnings. */               \
    struct FMT_GCC_VISIBILITY_HIDDEN FMT_COMPILE_STRING : base {           \
      using char_type = fmt::remove_cvref_t<decltype(s[0])>;               \
      FMT_MAYBE_UNUSED FMT_CONSTEXPR explicit                              \
      operator fmt::basic_string_view<char_type>() const {                 \
        return fmt::detail_exported::compile_string_to_view<char_type>(s); \
      }                                                                    \
    };                                                                     \
    return FMT_COMPILE_STRING();                                           \
  }()

/**
  \rst
  Constructs a compile-time format string from a string literal *s*.

  **Example**::

    // A compile-time error because 'd' is an invalid specifier for strings.
    std::string s = fmt::format(FMT_STRING("{:d}"), "foo");
  \endrst
 */
#define FMT_STRING(s) FMT_STRING_IMPL(s, fmt::compile_string, )

#if FMT_USE_USER_DEFINED_LITERALS
template <typename Char> struct udl_formatter {
  basic_string_view<Char> str;

  template <typename... T>
  auto operator()(T&&... args) const -> std::basic_string<Char> {
    return vformat(str, fmt::make_args_checked<T...>(str, args...));
  }
};

#  if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct statically_named_arg : view {
  static constexpr auto name = Str.data;

  const T& value;
  statically_named_arg(const T& v) : value(v) {}
};

template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct is_named_arg<statically_named_arg<T, Char, N, Str>> : std::true_type {};

template <typename T, typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct is_statically_named_arg<statically_named_arg<T, Char, N, Str>>
    : std::true_type {};

template <typename Char, size_t N,
          fmt::detail_exported::fixed_string<Char, N> Str>
struct udl_arg {
  template <typename T> auto operator=(T&& value) const {
    return statically_named_arg<T, Char, N, Str>(std::forward<T>(value));
  }
};
#  else
template <typename Char> struct udl_arg {
  const Char* str;

  template <typename T> auto operator=(T&& value) const -> named_arg<Char, T> {
    return {str, std::forward<T>(value)};
  }
};
#  endif
#endif  // FMT_USE_USER_DEFINED_LITERALS

template <typename Locale, typename Char>
auto vformat(const Locale& loc, basic_string_view<Char> format_str,
             basic_format_args<buffer_context<type_identity_t<Char>>> args)
    -> std::basic_string<Char> {
  basic_memory_buffer<Char> buffer;
  detail::vformat_to(buffer, format_str, args, detail::locale_ref(loc));
  return {buffer.data(), buffer.size()};
}

using format_func = void (*)(detail::buffer<char>&, int, const char*);

FMT_API void format_error_code(buffer<char>& out, int error_code,
                               string_view message) FMT_NOEXCEPT;

FMT_API void report_error(format_func func, int error_code,
                          const char* message) FMT_NOEXCEPT;
FMT_END_DETAIL_NAMESPACE

FMT_API auto vsystem_error(int error_code, string_view format_str,
                           format_args args) -> std::system_error;

/**
 \rst
 Constructs :class:`std::system_error` with a message formatted with
 ``fmt::format(fmt, args...)``.
  *error_code* is a system error code as given by ``errno``.

 **Example**::

   // This throws std::system_error with the description
   //   cannot open file 'madeup': No such file or directory
   // or similar (system message may vary).
   const char* filename = "madeup";
   std::FILE* file = std::fopen(filename, "r");
   if (!file)
     throw fmt::system_error(errno, "cannot open file '{}'", filename);
 \endrst
*/
template <typename... T>
auto system_error(int error_code, format_string<T...> fmt, T&&... args)
    -> std::system_error {
  return vsystem_error(error_code, fmt, fmt::make_format_args(args...));
}

/**
  \rst
  Formats an error message for an error returned by an operating system or a
  language runtime, for example a file opening error, and writes it to *out*.
  The format is the same as the one used by ``std::system_error(ec, message)``
  where ``ec`` is ``std::error_code(error_code, std::generic_category()})``.
  It is implementation-defined but normally looks like:

  .. parsed-literal::
     *<message>*: *<system-message>*

  where *<message>* is the passed message and *<system-message>* is the system
  message corresponding to the error code.
  *error_code* is a system error code as given by ``errno``.
  \endrst
 */
FMT_API void format_system_error(detail::buffer<char>& out, int error_code,
                                 const char* message) FMT_NOEXCEPT;

// Reports a system error without throwing an exception.
// Can be used to report errors from destructors.
FMT_API void report_system_error(int error_code,
                                 const char* message) FMT_NOEXCEPT;

/** Fast integer formatter. */
class format_int {
 private:
  // Buffer should be large enough to hold all digits (digits10 + 1),
  // a sign and a null character.
  enum { buffer_size = std::numeric_limits<unsigned long long>::digits10 + 3 };
  mutable char buffer_[buffer_size];
  char* str_;

  template <typename UInt> auto format_unsigned(UInt value) -> char* {
    auto n = static_cast<detail::uint32_or_64_or_128_t<UInt>>(value);
    return detail::format_decimal(buffer_, n, buffer_size - 1).begin;
  }

  template <typename Int> auto format_signed(Int value) -> char* {
    auto abs_value = static_cast<detail::uint32_or_64_or_128_t<Int>>(value);
    bool negative = value < 0;
    if (negative) abs_value = 0 - abs_value;
    auto begin = format_unsigned(abs_value);
    if (negative) *--begin = '-';
    return begin;
  }

 public:
  explicit format_int(int value) : str_(format_signed(value)) {}
  explicit format_int(long value) : str_(format_signed(value)) {}
  explicit format_int(long long value) : str_(format_signed(value)) {}
  explicit format_int(unsigned value) : str_(format_unsigned(value)) {}
  explicit format_int(unsigned long value) : str_(format_unsigned(value)) {}
  explicit format_int(unsigned long long value)
      : str_(format_unsigned(value)) {}

  /** Returns the number of characters written to the output buffer. */
  auto size() const -> size_t {
    return detail::to_unsigned(buffer_ - str_ + buffer_size - 1);
  }

  /**
    Returns a pointer to the output buffer content. No terminating null
    character is appended.
   */
  auto data() const -> const char* { return str_; }

  /**
    Returns a pointer to the output buffer content with terminating null
    character appended.
   */
  auto c_str() const -> const char* {
    buffer_[buffer_size - 1] = '\0';
    return str_;
  }

  /**
    \rst
    Returns the content of the output buffer as an ``std::string``.
    \endrst
   */
  auto str() const -> std::string { return std::string(str_, size()); }
};

template <typename T, typename Char>
template <typename FormatContext>
FMT_CONSTEXPR FMT_INLINE auto
formatter<T, Char,
          enable_if_t<detail::type_constant<T, Char>::value !=
                      detail::type::custom_type>>::format(const T& val,
                                                          FormatContext& ctx)
    const -> decltype(ctx.out()) {
  if (specs_.width_ref.kind != detail::arg_id_kind::none ||
      specs_.precision_ref.kind != detail::arg_id_kind::none) {
    auto specs = specs_;
    detail::handle_dynamic_spec<detail::width_checker>(specs.width,
                                                       specs.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs.precision, specs.precision_ref, ctx);
    return detail::write<Char>(ctx.out(), val, specs, ctx.locale());
  }
  return detail::write<Char>(ctx.out(), val, specs_, ctx.locale());
}

#define FMT_FORMAT_AS(Type, Base)                                        \
  template <typename Char>                                               \
  struct formatter<Type, Char> : formatter<Base, Char> {                 \
    template <typename FormatContext>                                    \
    auto format(Type const& val, FormatContext& ctx) const               \
        -> decltype(ctx.out()) {                                         \
      return formatter<Base, Char>::format(static_cast<Base>(val), ctx); \
    }                                                                    \
  }

FMT_FORMAT_AS(signed char, int);
FMT_FORMAT_AS(unsigned char, unsigned);
FMT_FORMAT_AS(short, int);
FMT_FORMAT_AS(unsigned short, unsigned);
FMT_FORMAT_AS(long, long long);
FMT_FORMAT_AS(unsigned long, unsigned long long);
FMT_FORMAT_AS(Char*, const Char*);
FMT_FORMAT_AS(std::basic_string<Char>, basic_string_view<Char>);
FMT_FORMAT_AS(std::nullptr_t, const void*);
FMT_FORMAT_AS(detail::byte, unsigned char);
FMT_FORMAT_AS(detail::std_string_view<Char>, basic_string_view<Char>);

template <typename Char>
struct formatter<void*, Char> : formatter<const void*, Char> {
  template <typename FormatContext>
  auto format(void* val, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<const void*, Char>::format(val, ctx);
  }
};

template <typename Char, size_t N>
struct formatter<Char[N], Char> : formatter<basic_string_view<Char>, Char> {
  template <typename FormatContext>
  FMT_CONSTEXPR auto format(const Char* val, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    return formatter<basic_string_view<Char>, Char>::format(val, ctx);
  }
};

// A formatter for types known only at run time such as variant alternatives.
//
// Usage:
//   using variant = std::variant<int, std::string>;
//   template <>
//   struct formatter<variant>: dynamic_formatter<> {
//     auto format(const variant& v, format_context& ctx) {
//       return visit([&](const auto& val) {
//           return dynamic_formatter<>::format(val, ctx);
//       }, v);
//     }
//   };
template <typename Char = char> class dynamic_formatter {
 private:
  detail::dynamic_format_specs<Char> specs_;
  const Char* format_str_;

  struct null_handler : detail::error_handler {
    void on_align(align_t) {}
    void on_sign(sign_t) {}
    void on_hash() {}
  };

  template <typename Context> void handle_specs(Context& ctx) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
  }

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    format_str_ = ctx.begin();
    // Checks are deferred to formatting time when the argument type is known.
    detail::dynamic_specs_handler<ParseContext> handler(specs_, ctx);
    return detail::parse_format_specs(ctx.begin(), ctx.end(), handler);
  }

  template <typename T, typename FormatContext>
  auto format(const T& val, FormatContext& ctx) -> decltype(ctx.out()) {
    handle_specs(ctx);
    detail::specs_checker<null_handler> checker(
        null_handler(), detail::mapped_type_constant<T, FormatContext>::value);
    checker.on_align(specs_.align);
    if (specs_.sign != sign::none) checker.on_sign(specs_.sign);
    if (specs_.alt) checker.on_hash();
    if (specs_.precision >= 0) checker.end_precision();
    return detail::write<Char>(ctx.out(), val, specs_, ctx.locale());
  }
};

/**
  \rst
  Converts ``p`` to ``const void*`` for pointer formatting.

  **Example**::

    auto s = fmt::format("{}", fmt::ptr(p));
  \endrst
 */
template <typename T> auto ptr(T p) -> const void* {
  static_assert(std::is_pointer<T>::value, "");
  return detail::bit_cast<const void*>(p);
}
template <typename T> auto ptr(const std::unique_ptr<T>& p) -> const void* {
  return p.get();
}
template <typename T> auto ptr(const std::shared_ptr<T>& p) -> const void* {
  return p.get();
}

class bytes {
 private:
  string_view data_;
  friend struct formatter<bytes>;

 public:
  explicit bytes(string_view data) : data_(data) {}
};

template <> struct formatter<bytes> {
 private:
  detail::dynamic_format_specs<char> specs_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    detail::specs_checker<handler_type> handler(handler_type(specs_, ctx),
                                                detail::type::string_type);
    auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
    detail::check_string_type_spec(specs_.type, ctx.error_handler());
    return it;
  }

  template <typename FormatContext>
  auto format(bytes b, FormatContext& ctx) -> decltype(ctx.out()) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
    return detail::write_bytes(ctx.out(), b.data_, specs_);
  }
};

// group_digits_view is not derived from view because it copies the argument.
template <typename T> struct group_digits_view { T value; };

/**
  \rst
  Returns a view that formats an integer value using ',' as a locale-independent
  thousands separator.

  **Example**::

    fmt::print("{}", fmt::group_digits(12345));
    // Output: "12,345"
  \endrst
 */
template <typename T> auto group_digits(T value) -> group_digits_view<T> {
  return {value};
}

template <typename T> struct formatter<group_digits_view<T>> : formatter<T> {
 private:
  detail::dynamic_format_specs<char> specs_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    using handler_type = detail::dynamic_specs_handler<ParseContext>;
    detail::specs_checker<handler_type> handler(handler_type(specs_, ctx),
                                                detail::type::int_type);
    auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
    detail::check_string_type_spec(specs_.type, ctx.error_handler());
    return it;
  }

  template <typename FormatContext>
  auto format(group_digits_view<T> t, FormatContext& ctx)
      -> decltype(ctx.out()) {
    detail::handle_dynamic_spec<detail::width_checker>(specs_.width,
                                                       specs_.width_ref, ctx);
    detail::handle_dynamic_spec<detail::precision_checker>(
        specs_.precision, specs_.precision_ref, ctx);
    return detail::write_int_localized(
        ctx.out(), static_cast<detail::uint64_or_128_t<T>>(t.value), 0, specs_,
        detail::digit_grouping<char>({"\3", ','}));
  }
};

template <typename It, typename Sentinel, typename Char = char>
struct join_view : detail::view {
  It begin;
  Sentinel end;
  basic_string_view<Char> sep;

  join_view(It b, Sentinel e, basic_string_view<Char> s)
      : begin(b), end(e), sep(s) {}
};

template <typename It, typename Sentinel, typename Char>
using arg_join FMT_DEPRECATED_ALIAS = join_view<It, Sentinel, Char>;

template <typename It, typename Sentinel, typename Char>
struct formatter<join_view<It, Sentinel, Char>, Char> {
 private:
  using value_type =
#ifdef __cpp_lib_ranges
      std::iter_value_t<It>;
#else
      typename std::iterator_traits<It>::value_type;
#endif
  using context = buffer_context<Char>;
  using mapper = detail::arg_mapper<context>;

  template <typename T, FMT_ENABLE_IF(has_formatter<T, context>::value)>
  static auto map(const T& value) -> const T& {
    return value;
  }
  template <typename T, FMT_ENABLE_IF(!has_formatter<T, context>::value)>
  static auto map(const T& value) -> decltype(mapper().map(value)) {
    return mapper().map(value);
  }

  using formatter_type =
      conditional_t<is_formattable<value_type, Char>::value,
                    formatter<remove_cvref_t<decltype(map(
                                  std::declval<const value_type&>()))>,
                              Char>,
                    detail::fallback_formatter<value_type, Char>>;

  formatter_type value_formatter_;

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return value_formatter_.parse(ctx);
  }

  template <typename FormatContext>
  auto format(const join_view<It, Sentinel, Char>& value, FormatContext& ctx)
      -> decltype(ctx.out()) {
    auto it = value.begin;
    auto out = ctx.out();
    if (it != value.end) {
      out = value_formatter_.format(map(*it), ctx);
      ++it;
      while (it != value.end) {
        out = detail::copy_str<Char>(value.sep.begin(), value.sep.end(), out);
        ctx.advance_to(out);
        out = value_formatter_.format(map(*it), ctx);
        ++it;
      }
    }
    return out;
  }
};

/**
  Returns a view that formats the iterator range `[begin, end)` with elements
  separated by `sep`.
 */
template <typename It, typename Sentinel>
auto join(It begin, Sentinel end, string_view sep) -> join_view<It, Sentinel> {
  return {begin, end, sep};
}

/**
  \rst
  Returns a view that formats `range` with elements separated by `sep`.

  **Example**::

    std::vector<int> v = {1, 2, 3};
    fmt::print("{}", fmt::join(v, ", "));
    // Output: "1, 2, 3"

  ``fmt::join`` applies passed format specifiers to the range elements::

    fmt::print("{:02}", fmt::join(v, ", "));
    // Output: "01, 02, 03"
  \endrst
 */
template <typename Range>
auto join(Range&& range, string_view sep)
    -> join_view<detail::iterator_t<Range>, detail::sentinel_t<Range>> {
  return join(std::begin(range), std::end(range), sep);
}

/**
  \rst
  Converts *value* to ``std::string`` using the default format for type *T*.

  **Example**::

    #include <fmt/format.h>

    std::string answer = fmt::to_string(42);
  \endrst
 */
template <typename T, FMT_ENABLE_IF(!std::is_integral<T>::value)>
inline auto to_string(const T& value) -> std::string {
  auto result = std::string();
  detail::write<char>(std::back_inserter(result), value);
  return result;
}

template <typename T, FMT_ENABLE_IF(std::is_integral<T>::value)>
FMT_NODISCARD inline auto to_string(T value) -> std::string {
  // The buffer should be large enough to store the number including the sign
  // or "false" for bool.
  constexpr int max_size = detail::digits10<T>() + 2;
  char buffer[max_size > 5 ? static_cast<unsigned>(max_size) : 5];
  char* begin = buffer;
  return std::string(begin, detail::write<char>(begin, value));
}

template <typename Char, size_t SIZE>
FMT_NODISCARD auto to_string(const basic_memory_buffer<Char, SIZE>& buf)
    -> std::basic_string<Char> {
  auto size = buf.size();
  detail::assume(size < std::basic_string<Char>().max_size());
  return std::basic_string<Char>(buf.data(), size);
}

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Char>
void vformat_to(
    buffer<Char>& buf, basic_string_view<Char> fmt,
    basic_format_args<FMT_BUFFER_CONTEXT(type_identity_t<Char>)> args,
    locale_ref loc) {
  // workaround for msvc bug regarding name-lookup in module
  // link names into function scope
  using detail::arg_formatter;
  using detail::buffer_appender;
  using detail::custom_formatter;
  using detail::default_arg_formatter;
  using detail::get_arg;
  using detail::locale_ref;
  using detail::parse_format_specs;
  using detail::specs_checker;
  using detail::specs_handler;
  using detail::to_unsigned;
  using detail::type;
  using detail::write;
  auto out = buffer_appender<Char>(buf);
  if (fmt.size() == 2 && equal2(fmt.data(), "{}")) {
    auto arg = args.get(0);
    if (!arg) error_handler().on_error("argument not found");
    visit_format_arg(default_arg_formatter<Char>{out, args, loc}, arg);
    return;
  }

  struct format_handler : error_handler {
    basic_format_parse_context<Char> parse_context;
    buffer_context<Char> context;

    format_handler(buffer_appender<Char> out, basic_string_view<Char> str,
                   basic_format_args<buffer_context<Char>> args, locale_ref loc)
        : parse_context(str), context(out, args, loc) {}

    void on_text(const Char* begin, const Char* end) {
      auto text = basic_string_view<Char>(begin, to_unsigned(end - begin));
      context.advance_to(write<Char>(context.out(), text));
    }

    FMT_CONSTEXPR auto on_arg_id() -> int {
      return parse_context.next_arg_id();
    }
    FMT_CONSTEXPR auto on_arg_id(int id) -> int {
      return parse_context.check_arg_id(id), id;
    }
    FMT_CONSTEXPR auto on_arg_id(basic_string_view<Char> id) -> int {
      int arg_id = context.arg_id(id);
      if (arg_id < 0) on_error("argument not found");
      return arg_id;
    }

    FMT_INLINE void on_replacement_field(int id, const Char*) {
      auto arg = get_arg(context, id);
      context.advance_to(visit_format_arg(
          default_arg_formatter<Char>{context.out(), context.args(),
                                      context.locale()},
          arg));
    }

    auto on_format_specs(int id, const Char* begin, const Char* end)
        -> const Char* {
      auto arg = get_arg(context, id);
      if (arg.type() == type::custom_type) {
        parse_context.advance_to(parse_context.begin() +
                                 (begin - &*parse_context.begin()));
        visit_format_arg(custom_formatter<Char>{parse_context, context}, arg);
        return parse_context.begin();
      }
      auto specs = basic_format_specs<Char>();
      specs_checker<specs_handler<Char>> handler(
          specs_handler<Char>(specs, parse_context, context), arg.type());
      begin = parse_format_specs(begin, end, handler);
      if (begin == end || *begin != '}')
        on_error("missing '}' in format string");
      auto f = arg_formatter<Char>{context.out(), specs, context.locale()};
      context.advance_to(visit_format_arg(f, arg));
      return begin;
    }
  };
  detail::parse_format_string<false>(fmt, format_handler(out, fmt, args, loc));
}

#ifndef FMT_HEADER_ONLY
extern template FMT_API auto thousands_sep_impl<char>(locale_ref)
    -> thousands_sep_result<char>;
extern template FMT_API auto thousands_sep_impl<wchar_t>(locale_ref)
    -> thousands_sep_result<wchar_t>;
extern template FMT_API auto decimal_point_impl(locale_ref) -> char;
extern template FMT_API auto decimal_point_impl(locale_ref) -> wchar_t;
extern template auto format_float<double>(double value, int precision,
                                          float_specs specs, buffer<char>& buf)
    -> int;
extern template auto format_float<long double>(long double value, int precision,
                                               float_specs specs,
                                               buffer<char>& buf) -> int;
void snprintf_float(float, int, float_specs, buffer<char>&) = delete;
extern template auto snprintf_float<double>(double value, int precision,
                                            float_specs specs,
                                            buffer<char>& buf) -> int;
extern template auto snprintf_float<long double>(long double value,
                                                 int precision,
                                                 float_specs specs,
                                                 buffer<char>& buf) -> int;
#endif  // FMT_HEADER_ONLY

FMT_END_DETAIL_NAMESPACE

#if FMT_USE_USER_DEFINED_LITERALS
inline namespace literals {
/**
  \rst
  User-defined literal equivalent of :func:`fmt::arg`.

  **Example**::

    using namespace fmt::literals;
    fmt::print("Elapsed time: {s:.2f} seconds", "s"_a=1.23);
  \endrst
 */
#  if FMT_USE_NONTYPE_TEMPLATE_PARAMETERS
template <detail_exported::fixed_string Str>
constexpr auto operator""_a()
    -> detail::udl_arg<remove_cvref_t<decltype(Str.data[0])>,
                       sizeof(Str.data) / sizeof(decltype(Str.data[0])), Str> {
  return {};
}
#  else
constexpr auto operator"" _a(const char* s, size_t) -> detail::udl_arg<char> {
  return {s};
}
#  endif

// DEPRECATED!
// User-defined literal equivalent of fmt::format.
FMT_DEPRECATED constexpr auto operator"" _format(const char* s, size_t n)
    -> detail::udl_formatter<char> {
  return {{s, n}};
}
}  // namespace literals
#endif  // FMT_USE_USER_DEFINED_LITERALS

template <typename Locale, FMT_ENABLE_IF(detail::is_locale<Locale>::value)>
inline auto vformat(const Locale& loc, string_view fmt, format_args args)
    -> std::string {
  return detail::vformat(loc, fmt, args);
}

template <typename Locale, typename... T,
          FMT_ENABLE_IF(detail::is_locale<Locale>::value)>
inline auto format(const Locale& loc, format_string<T...> fmt, T&&... args)
    -> std::string {
  return vformat(loc, string_view(fmt), fmt::make_format_args(args...));
}

template <typename... T, size_t SIZE, typename Allocator>
FMT_DEPRECATED auto format_to(basic_memory_buffer<char, SIZE, Allocator>& buf,
                              format_string<T...> fmt, T&&... args)
    -> appender {
  detail::vformat_to(buf, string_view(fmt), fmt::make_format_args(args...));
  return appender(buf);
}

template <typename OutputIt, typename Locale,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value&&
                            detail::is_locale<Locale>::value)>
auto vformat_to(OutputIt out, const Locale& loc, string_view fmt,
                format_args args) -> OutputIt {
  using detail::get_buffer;
  auto&& buf = get_buffer<char>(out);
  detail::vformat_to(buf, fmt, args, detail::locale_ref(loc));
  return detail::get_iterator(buf);
}

template <typename OutputIt, typename Locale, typename... T,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, char>::value&&
                            detail::is_locale<Locale>::value)>
FMT_INLINE auto format_to(OutputIt out, const Locale& loc,
                          format_string<T...> fmt, T&&... args) -> OutputIt {
  return vformat_to(out, loc, fmt, fmt::make_format_args(args...));
}

FMT_MODULE_EXPORT_END
FMT_END_NAMESPACE

#ifdef FMT_DEPRECATED_INCLUDE_XCHAR
#  include "xchar.h"
#endif

#ifdef FMT_HEADER_ONLY
#  define FMT_FUNC inline
#  include "format-inl.h"
#else
#  define FMT_FUNC
#endif

#endif  // FMT_FORMAT_H_
#ifndef GNURADIO_CIRCULAR_BUFFER_HPP
#define GNURADIO_CIRCULAR_BUFFER_HPP

#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 16000
#include <experimental/memory_resource>

namespace std::pmr {
using memory_resource = std::experimental::pmr::memory_resource;
template<typename T>
using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
} // namespace std::pmr
#else
#include <memory_resource>
#endif
#include <algorithm>
#include <bit>
#include <cassert> // to assert if compiled for debugging
#include <functional>
#include <numeric>
#include <ranges>
#include <span>

#include <fmt/format.h>

// header for creating/opening or POSIX shared memory objects
#include <cerrno>
#include <fcntl.h>
#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gr {
static constexpr bool has_posix_mmap_interface = true;
}

#define HAS_POSIX_MAP_INTERFACE
#else
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif
#else // #if defined __has_include -- required for portability
namespace gr {
static constexpr bool has_posix_mmap_interface = false;
}
#endif

#ifndef GNURADIO_CLAIM_STRATEGY_HPP
#endif
#ifndef GNURADIO_SEQUENCE_HPP
#endif
#ifdef GNURADIO_WAIT_STRATEGY_HPP
#endif

namespace gr {

namespace util {
constexpr std::size_t
round_up(std::size_t num_to_round, std::size_t multiple) noexcept {
    if (multiple == 0) {
        return num_to_round;
    }
    const auto remainder = num_to_round % multiple;
    if (remainder == 0) {
        return num_to_round;
    }
    return num_to_round + multiple - remainder;
}
} // namespace util

// clang-format off
class double_mapped_memory_resource : public std::pmr::memory_resource {
#ifdef HAS_POSIX_MAP_INTERFACE
    [[nodiscard]] void* do_allocate(const std::size_t required_size, std::size_t alignment) override {

        const std::size_t size = 2 * required_size;
        if (size % 2LU != 0LU || size % static_cast<std::size_t>(getpagesize()) != 0LU) {
            throw std::runtime_error(fmt::format("incompatible buffer-byte-size: {} -> {} alignment: {} vs. page size: {}", required_size, size, alignment, getpagesize()));
        }
        const std::size_t size_half = size/2;

        static std::size_t _counter;
        const auto buffer_name = fmt::format("/double_mapped_memory_resource-{}-{}-{}", getpid(), size, _counter++);
        const auto memfd_create = [name = buffer_name.c_str()](unsigned int flags) -> long {
            return syscall(__NR_memfd_create, name, flags);
        };
        int shm_fd = static_cast<int>(memfd_create(0));
        if (shm_fd < 0) {
            throw std::runtime_error(fmt::format("{} - memfd_create error {}: {}",  buffer_name, errno, strerror(errno)));
        }

        if (ftruncate(shm_fd, static_cast<off_t>(size)) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - ftruncate {}: {}",  buffer_name, errno, strerror(errno)));
        }

        void* first_copy = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t>(0));
        if (first_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for first half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // unmap the 2nd half
        if (munmap(static_cast<char*>(first_copy) + size_half, size_half) == -1) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed munmap for second half {}: {}",  buffer_name, errno, strerror(errno)));
        }

        // map the first half into the now available hole where the
        if (const void* second_copy = mmap(static_cast<char*> (first_copy) + size_half, size_half, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, static_cast<off_t> (0)); second_copy == MAP_FAILED) {
            close(shm_fd);
            throw std::runtime_error(fmt::format("{} - failed mmap for second copy {}: {}",  buffer_name, errno, strerror(errno)));
        }

        close(shm_fd); // file-descriptor is no longer needed. The mapping is retained.
        return first_copy;
}
#else
    [[nodiscard]] void* do_allocate(const std::size_t, std::size_t) override {
        throw std::runtime_error("OS does not provide POSIX interface for mmap(...) and munmao(...)");
        // static_assert(false, "OS does not provide POSIX interface for mmap(...) and munmao(...)");
    }
#endif


#ifdef HAS_POSIX_MAP_INTERFACE
    void  do_deallocate(void* p, std::size_t size, size_t alignment) override {

        if (munmap(p, size) == -1) {
            throw std::runtime_error(fmt::format("double_mapped_memory_resource::do_deallocate(void*, {}, {}) - munmap(..) failed", size, alignment));
        }
    }
#else
    void  do_deallocate(void*, std::size_t, size_t) override { }
#endif

    bool  do_is_equal(const memory_resource& other) const noexcept override { return this == &other; }

public:
    static inline double_mapped_memory_resource* defaultAllocator() {
        static auto instance = double_mapped_memory_resource();
        return &instance;
    }

    template<typename T>
    requires (std::has_single_bit(sizeof(T)))
    static inline std::pmr::polymorphic_allocator<T> allocator()
    {
        return std::pmr::polymorphic_allocator<T>(gr::double_mapped_memory_resource::defaultAllocator());
    }
};



/**
 * @brief circular buffer implementation using double-mapped memory allocations
 * where the first SIZE-ed buffer is mirrored directly its end to mimic wrap-around
 * free bulk memory access. The buffer keeps a list of indices (using 'Sequence')
 * to keep track of which parts can be tread-safely read and/or written
 *
 *                          wrap-around point
 *                                 |
 *                                 v
 *  | buffer segment #1 (original) | buffer segment #2 (copy of #1) |
 *  0                            SIZE                            2*SIZE
 *                      writeIndex
 *                          v
 *  wrap-free write access  |<-  N_1 < SIZE   ->|
 *
 *  readIndex < writeIndex-N_2
 *      v
 *      |<- N_2 < SIZE ->|
 *
 * N.B N_AVAILABLE := (SIZE + writeIndex - readIndex ) % SIZE
 *
 * citation: <find appropriate first and educational reference>
 *
 * This implementation provides single- as well as multi-producer/consumer buffer
 * combinations for thread-safe CPU-to-CPU data transfer (optionally) using either
 * a) the POSIX mmaped(..)/munmapped(..) MMU interface, if available, and/or
 * b) the guaranteed portable standard C/C++ (de-)allocators as a fall-back.
 *
 * for more details see
 */
template <typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producer_type = ProducerType::Single, WaitStrategy WAIT_STRATEGY = SleepingWaitStrategy>
requires (std::has_single_bit(sizeof(T)))
class circular_buffer
{
    using Allocator         = std::pmr::polymorphic_allocator<T>;
    using BufferType        = circular_buffer<T, SIZE, producer_type, WAIT_STRATEGY>;
    using ClaimType         = detail::producer_type_v<SIZE, producer_type, WAIT_STRATEGY>;
    using DependendsType    = std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>;

    struct buffer_impl {
        alignas(kCacheLine) Allocator                   _allocator{};
        alignas(kCacheLine) const bool                  _is_mmap_allocated;
        alignas(kCacheLine) const std::size_t           _size;
        alignas(kCacheLine) std::vector<T, Allocator>   _data;
        alignas(kCacheLine) Sequence                    _cursor;
        alignas(kCacheLine) WAIT_STRATEGY               _wait_strategy = WAIT_STRATEGY();
        alignas(kCacheLine) ClaimType                   _claim_strategy;
        // list of dependent reader indices
        alignas(kCacheLine) DependendsType              _read_indices{ std::make_shared<std::vector<std::shared_ptr<Sequence>>>() };

        buffer_impl() = delete;
        buffer_impl(const std::size_t min_size, Allocator allocator) : _allocator(allocator), _is_mmap_allocated(dynamic_cast<double_mapped_memory_resource *>(_allocator.resource())),
            _size(align_with_page_size(min_size, _is_mmap_allocated)), _data(buffer_size(_size, _is_mmap_allocated), _allocator), _claim_strategy(ClaimType(_cursor, _wait_strategy, _size)) {
        }

#ifdef HAS_POSIX_MAP_INTERFACE
        static std::size_t align_with_page_size(const std::size_t min_size, bool _is_mmap_allocated) {
            return _is_mmap_allocated ? util::round_up(min_size * sizeof(T), static_cast<std::size_t>(getpagesize())) / sizeof(T) : min_size;
        }
#else
        static std::size_t align_with_page_size(const std::size_t min_size, bool) {
            return min_size; // mmap() & getpagesize() not supported for non-POSIX OS
        }
#endif

        static std::size_t buffer_size(const std::size_t size, bool _is_mmap_allocated) {
            // double-mmaped behaviour requires the different size/alloc strategy
            // i.e. the second buffer half may not default-constructed as it's identical to the first one
            // and would result in a double dealloc during the default destruction
            return _is_mmap_allocated ? size : 2 * size;
        }
    };

    template <typename U = T>
    class buffer_writer {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) bool                        _is_mmap_allocated;
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;
        alignas(kCacheLine) ClaimType*                  _claim_strategy;

    public:
        buffer_writer() = delete;
        buffer_writer(std::shared_ptr<buffer_impl> buffer) :
            _buffer(std::move(buffer)), _is_mmap_allocated(_buffer->_is_mmap_allocated),
            _size(_buffer->_size), _data(std::addressof(_buffer->_data)), _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer(buffer_writer&& other)
            : _buffer(std::move(other._buffer))
            , _is_mmap_allocated(_buffer->_is_mmap_allocated)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data))
            , _claim_strategy(std::addressof(_buffer->_claim_strategy)) { };
        buffer_writer& operator=(buffer_writer tmp) {
            std::swap(_buffer, tmp._buffer);
            _is_mmap_allocated = _buffer->_is_mmap_allocated;
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            _claim_strategy = std::addressof(_buffer->_claim_strategy);

            return *this;
        }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        [[nodiscard]] constexpr auto get(std::size_t n_slots_to_claim) noexcept -> std::pair<std::span<U>, std::pair<std::size_t, std::int64_t>> {
            try {
                const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim); // alt: try_next
                const std::size_t index = (sequence + _size - n_slots_to_claim) % _size;
                return {{ &(*_data)[index], n_slots_to_claim }, {index, sequence - n_slots_to_claim } };
            } catch (const NoCapacityException &) {
                return { { /* empty span */ }, { 0, 0 }};
            }
        }

        constexpr void publish(std::pair<std::size_t, std::int64_t> token, std::size_t n_produced) {
            if (!_is_mmap_allocated) {
                // mirror samples below/above the buffer's wrap-around point
                const std::size_t index = token.first;
                const size_t nFirstHalf = std::min(_size - index, n_produced);
                const size_t nSecondHalf = n_produced  - nFirstHalf;

                auto& data = *_data;
                std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
            }
            _claim_strategy->publish(token.second + n_produced); // points at first non-writable index
        }

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return;
            }
            const auto sequence = _claim_strategy->next(*_buffer->_read_indices, n_slots_to_claim);
            translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
        } // blocks until elements are available

        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr bool try_publish(Translator&& translator, std::size_t n_slots_to_claim = 1, Args&&... args) {
            if (n_slots_to_claim <= 0 || _buffer->_read_indices->empty()) {
                return true;
            }
            try {
                const auto sequence = _claim_strategy->tryNext(*_buffer->_read_indices, n_slots_to_claim);
                translate_and_publish(std::forward<Translator>(translator), n_slots_to_claim, sequence, std::forward<Args>(args)...);
                return true;
            } catch (const NoCapacityException &) {
                return false;
            }
        }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _claim_strategy->getRemainingCapacity(*_buffer->_read_indices);
        }

        private:
        template <typename... Args, WriterCallback<U, Args...> Translator>
        constexpr void translate_and_publish(Translator&& translator, const std::size_t n_slots_to_claim, const std::int64_t publishSequence, const Args&... args) {
            try {
                auto& data = *_data;
                const std::size_t index = (publishSequence + _size - n_slots_to_claim) % _size;
                std::span<U> writable_data(&data[index], n_slots_to_claim);
                if constexpr (std::is_invocable<Translator, std::span<T>&, std::int64_t, Args...>::value) {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), publishSequence - n_slots_to_claim, args...);
                } else {
                    std::invoke(std::forward<Translator>(translator), std::forward<std::span<T>&>(writable_data), args...);
                }

                if (!_is_mmap_allocated) {
                    // mirror samples below/above the buffer's wrap-around point
                    const size_t nFirstHalf = std::min(_size - index, n_slots_to_claim);
                    const size_t nSecondHalf = n_slots_to_claim  - nFirstHalf;

                    std::copy(&data[index], &data[index + nFirstHalf], &data[index+ _size]);
                    std::copy(&data[_size],  &data[_size + nSecondHalf], &data[0]);
                }
                _claim_strategy->publish(publishSequence); // points at first non-writable index
            } catch (const std::exception& e) {
                throw e;
            } catch (...) {
                throw std::runtime_error("circular_buffer::translate_and_publish() - unknown user exception thrown");
            }
        }
    };

    template<typename U = T>
    class buffer_reader
    {
        using BufferTypeLocal = std::shared_ptr<buffer_impl>;

        alignas(kCacheLine) std::shared_ptr<Sequence>   _read_index = std::make_shared<Sequence>();
        alignas(kCacheLine) std::int64_t                _read_index_cached;
        alignas(kCacheLine) BufferTypeLocal             _buffer; // controls buffer life-cycle, the rest are cache optimisations
        alignas(kCacheLine) std::size_t                 _size;
        alignas(kCacheLine) std::vector<U, Allocator>*  _data;

    public:
        buffer_reader() = delete;
        buffer_reader(std::shared_ptr<buffer_impl> buffer) :
            _buffer(buffer), _size(buffer->_size), _data(std::addressof(buffer->_data)) {
            gr::detail::addSequences(_buffer->_read_indices, _buffer->_cursor, {_read_index});
            _read_index_cached = _read_index->value();
        }
        buffer_reader(buffer_reader&& other)
            : _read_index(std::move(other._read_index))
            , _read_index_cached(std::exchange(other._read_index_cached, _read_index->value()))
            , _buffer(other._buffer)
            , _size(_buffer->_size)
            , _data(std::addressof(_buffer->_data)) {
        }
        buffer_reader& operator=(buffer_reader tmp) noexcept {
            std::swap(_read_index, tmp._read_index);
            std::swap(_read_index_cached, tmp._read_index_cached);
            std::swap(_buffer, tmp._buffer);
            _size = _buffer->_size;
            _data = std::addressof(_buffer->_data);
            return *this;
        };
        ~buffer_reader() { gr::detail::removeSequence( _buffer->_read_indices, _read_index); }

        [[nodiscard]] constexpr BufferType buffer() const noexcept { return circular_buffer(_buffer); };

        template <bool strict_check = true>
        [[nodiscard]] constexpr std::span<const U> get(const std::size_t n_requested = 0) const noexcept {
            const auto& data = *_data;
            if constexpr (strict_check) {
                const std::size_t n = n_requested > 0 ? std::min(n_requested, available()) : available();
                return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
            }
            const std::size_t n = n_requested > 0 ? n_requested : available();
            return { &data[static_cast<std::uint64_t>(_read_index_cached) % _size], n };
        }

        template <bool strict_check = true>
        [[nodiscard]] constexpr bool consume(const std::size_t n_elements = 1) noexcept {
            if constexpr (strict_check) {
                if (n_elements <= 0) {
                    return true;
                }
                if (n_elements > available()) {
                    return false;
                }
            }
            _read_index_cached = _read_index->addAndGet(static_cast<int64_t>(n_elements));
            return true;
        }

        [[nodiscard]] constexpr std::int64_t position() const noexcept { return _read_index_cached; }

        [[nodiscard]] constexpr std::size_t available() const noexcept {
            return _buffer->_cursor.value() - _read_index_cached;
        }
    };

    [[nodiscard]] constexpr static Allocator DefaultAllocator() {
        if constexpr (has_posix_mmap_interface) {
            return double_mapped_memory_resource::allocator<T>();
        } else {
            return Allocator();
        }
    }

    std::shared_ptr<buffer_impl> _shared_buffer_ptr;
    circular_buffer(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    circular_buffer() = delete;
    circular_buffer(std::size_t min_size, Allocator allocator = DefaultAllocator())
        : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size, allocator)) { }
    ~circular_buffer() = default;

    [[nodiscard]] std::size_t       size() const noexcept { return _shared_buffer_ptr->_size; }
    [[nodiscard]] BufferWriter auto new_writer() { return buffer_writer<T>(_shared_buffer_ptr); }
    [[nodiscard]] BufferReader auto new_reader() { return buffer_reader<T>(_shared_buffer_ptr); }

    // implementation specific interface -- not part of public Buffer / production-code API
    [[nodiscard]] auto n_readers()       { return _shared_buffer_ptr->_read_indices->size(); }
    [[nodiscard]] auto claim_strategy()  { return _shared_buffer_ptr->_claim_strategy; }
    [[nodiscard]] auto wait_strategy()   { return _shared_buffer_ptr->_wait_strategy; }
    [[nodiscard]] auto cursor_sequence() { return _shared_buffer_ptr->_cursor; }

};
static_assert(Buffer<circular_buffer<int32_t>>);
// clang-format on

} // namespace gr

#endif // GNURADIO_CIRCULAR_BUFFER_HPP
#ifndef GNURADIO_BUFFER2_H
#define GNURADIO_BUFFER2_H

#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <span>

namespace gr {
namespace util {
template <typename T, typename...>
struct first_template_arg_helper;

template <template <typename...> class TemplateType,
          typename ValueType,
          typename... OtherTypes>
struct first_template_arg_helper<TemplateType<ValueType, OtherTypes...>> {
    using value_type = ValueType;
};

template <typename T>
constexpr auto* value_type_helper()
{
    if constexpr (requires { typename T::value_type; }) {
        return static_cast<typename T::value_type*>(nullptr);
    }
    else {
        return static_cast<typename first_template_arg_helper<T>::value_type*>(nullptr);
    }
}

template <typename T>
using value_type_t = std::remove_pointer_t<decltype(value_type_helper<T>())>;

template <typename... A>
struct test_fallback {
};

template <typename, typename ValueType>
struct test_value_type {
    using value_type = ValueType;
};

static_assert(std::is_same_v<int, value_type_t<test_fallback<int, float, double>>>);
static_assert(std::is_same_v<float, value_type_t<test_value_type<int, float>>>);
static_assert(std::is_same_v<int, value_type_t<std::span<int>>>);
static_assert(std::is_same_v<double, value_type_t<std::array<double, 42>>>);

} // namespace util

// clang-format off
// disable formatting until clang-format (v16) supporting concepts
template<class T>
concept BufferReader = requires(T /*const*/ t, const std::size_t n_items) {
    { t.get(n_items) }     -> std::same_as<std::span<const util::value_type_t<T>>>;
    { t.consume(n_items) } -> std::same_as<bool>;
    { t.position() }       -> std::same_as<std::int64_t>;
    { t.available() }      -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class Fn, typename T, typename ...Args>
concept WriterCallback = std::is_invocable_v<Fn, std::span<T>&, std::int64_t, Args...> || std::is_invocable_v<Fn, std::span<T>&, Args...>;

template<class T, typename ...Args>
concept BufferWriter = requires(T t, const std::size_t n_items, std::pair<std::size_t, std::int64_t> token, Args ...args) {
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                                 -> std::same_as<void>;
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }   -> std::same_as<void>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                             -> std::same_as<bool>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::int64_t /* writePos */, Args ...) { /* */  }, n_items, args...) }-> std::same_as<bool>;
    { t.get(n_items) } -> std::same_as<std::pair<std::span<util::value_type_t<T>>, std::pair<std::size_t, std::int64_t>>>;
    { t.publish(token, n_items) } -> std::same_as<void>;
    { t.available() }         -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class T, typename ...Args>
concept Buffer = requires(T t, const std::size_t min_size, Args ...args) {
    { T(min_size, args...) };
    { t.size() }       -> std::same_as<std::size_t>;
    { t.new_reader() } -> BufferReader;
    { t.new_writer() } -> BufferWriter;
};

// compile-time unit-tests
namespace test {
template <typename T>
struct non_compliant_class {
};
template <typename T, typename... Args>
using WithSequenceParameter = decltype([](std::span<T>&, std::int64_t, Args...) { /* */ });
template <typename T, typename... Args>
using NoSequenceParameter = decltype([](std::span<T>&, Args...) { /* */ });
} // namespace test

static_assert(!Buffer<test::non_compliant_class<int>>);
static_assert(!BufferReader<test::non_compliant_class<int>>);
static_assert(!BufferWriter<test::non_compliant_class<int>>);

static_assert(WriterCallback<test::WithSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::WithSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::WithSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int>, int>);
static_assert(!WriterCallback<test::NoSequenceParameter<int>, int, std::span<bool>>);
static_assert(WriterCallback<test::NoSequenceParameter<int, std::span<bool>>, int, std::span<bool>>);
// clang-format on
} // namespace gr

#endif // GNURADIO_BUFFER2_H
#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP


#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <ranges>
#include <tuple>
#include <variant>

#if !__has_include(<source_location>)
#define HAVE_SOURCE_LOCATION 0
#else

#include <source_location>

#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace fair::graph {

namespace stdx = vir::stdx;

using fair::meta::fixed_string;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t { STREAM, MESSAGE }; // TODO: Think of a better name
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
                   typename T::value_type;
                   { t.pmt_type() } -> std::same_as<supported_type>;
                   { t.type() } -> std::same_as<port_type_t>;
                   { t.direction() } -> std::same_as<port_direction_t>;
                   { t.name() } -> std::same_as<std::string_view>;
                   { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
                   { t.disconnect() } -> std::same_as<connection_result_t>;
               };

class dynamic_port;

template<typename T>
concept has_fixed_port_info_v = requires {
                                    typename T::value_type;
                                    { T::static_name() };
                                    { T::direction() } -> std::same_as<port_direction_t>;
                                    { T::type() } -> std::same_as<port_type_t>;
                                };

template<typename T>
using has_fixed_port_info = std::integral_constant<bool, has_fixed_port_info_v<T>>;

template<typename T>
struct has_fixed_port_info_or_is_typelist : std::false_type {};

template<typename T>
    requires has_fixed_port_info_v<T>
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<typename T>
    requires(meta::is_typelist_v<T> and T::template all_of<has_fixed_port_info>)
struct has_fixed_port_info_or_is_typelist<T> : std::true_type {};

template<std::size_t _min, std::size_t _max>
struct limits {
    using limits_tag                 = std::true_type;
    static constexpr std::size_t min = _min;
    static constexpr std::size_t max = _max;
};

template<typename T>
concept is_limits_v = requires { typename T::limits_tag; };

static_assert(!is_limits_v<int>);
static_assert(!is_limits_v<std::size_t>);
static_assert(is_limits_v<limits<0, 1024>>);

template<typename T>
using is_limits = std::integral_constant<bool, is_limits_v<T>>;

template<typename Port>
using port_type = typename Port::value_type;

template<typename Port>
using is_in_port = std::integral_constant<bool, Port::direction() == port_direction_t::INPUT>;

template<typename Port>
concept InPort = is_in_port<Port>::value;

template<typename Port>
using is_out_port = std::integral_constant<bool, Port::direction() == port_direction_t::OUTPUT>;

template<typename Port>
concept OutPort = is_out_port<Port>::value;

enum class work_result { success, has_unprocessed_data, inputs_empty, writers_not_available, error };

namespace work_policies {
struct one_by_one { // TODO: remove -- benchmark indicate this being inefficient

    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        auto inputs_available = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return ((input_port<Idx>(&self).reader().available() > 0) && ...); }
        (std::make_index_sequence<Self::input_ports::size>());

        if (!inputs_available) {
            return work_result::inputs_empty;
        }

        auto results = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            auto result  = meta::invoke_void_wrapped([&self]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); }, input_port<Idx>(&self).reader().get(1)[0]...);

            bool success = true;
            ((success = success && input_port<Idx>(&self).reader().consume(1)), ...);
            if (!success) throw fmt::format("Buffers reported more available data than was available");

            return result;
        }
        (std::make_index_sequence<Self::input_ports::size>());

        if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
            // process_one returned void

        } else if constexpr (requires { std::get<0>(results); }) {
            static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
            [&self, &results]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((output_port<Idx>(&self).writer().publish([&results](auto &w) { w[0] = std::get<Idx>(std::move(results)); }, 1)), ...);
            }
            (std::make_index_sequence<Self::output_ports::size>());

        } else {
            static_assert(Self::output_ports::size == 1);
            output_port<0>(&self).writer().publish([&results](auto &w) { w[0] = std::move(results); }, 1);
        }

        return work_result::success;
    }
};

struct read_many_and_publish_one_by_one { // TODO: remove -- benchmark indicate this being inefficient

    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        auto available_values_count = [&self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return std::min({ std::numeric_limits<std::size_t>::max(), std::clamp((input_port<Idx>(&self).reader().available()), std::size_t{ 0 }, std::size_t{ 1024 })... }); // TODO min and max
        }
        (std::make_index_sequence<Self::input_ports::size>());

        if (available_values_count == 0) {
            return work_result::inputs_empty;
        }

        auto input_spans = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().get(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());

        for (std::size_t i = 0; i < available_values_count; ++i) {
            auto results = [&self, &input_spans, i]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                auto result = meta::invoke_void_wrapped([&self, &input_spans, i]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); },
                                                        std::get<Idx>(input_spans)[i]...);

                return result;
            }
            (std::make_index_sequence<Self::input_ports::size>());

            if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
                // process_one returned void

            } else if constexpr (requires { std::get<0>(results); }) {
                static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
                [&self, &results]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                    ((output_port<Idx>(&self).writer().publish([&results](auto &w) { w[0] = std::get<Idx>(std::move(results)); }, 1)), ...);
                }
                (std::make_index_sequence<Self::output_ports::size>());

            } else {
                static_assert(Self::output_ports::size == 1);
                output_port<0>(&self).writer().publish([&results](auto &w) { w[0] = std::move(results); }, 1);
            }
        }

        [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().consume(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());
    }
};

struct read_many_and_publish_many {
    template<typename Self>
    static work_result
    work(Self &self) noexcept {
        bool              at_least_one_input_has_data = false;
        const std::size_t available_values_count      = [&self, &at_least_one_input_has_data]() {
            if constexpr (Self::input_ports::size > 0) {
                const auto availableForPort = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept {
                    const std::size_t available = port.reader().available();
                    if (available > 0LU) at_least_one_input_has_data = true;
                    if (available < port.min_buffer_size()) {
                        return 0LU;
                    } else {
                        return std::min(available, port.max_buffer_size());
                    }
                };

                const auto availableInAll = [&self, &availableForPort]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                    const auto betterMin = []<typename Arg, typename... Args>(Arg arg, Args &&...args) noexcept -> std::size_t {
                        if constexpr (sizeof...(Args) == 0) {
                            return arg;
                        } else {
                            return std::min(arg, args...);
                        }
                    };
                    return betterMin(availableForPort(input_port<Idx>(&self))...);
                }
                (std::make_index_sequence<Self::input_ports::size>());

                if (availableInAll < self.min_samples()) {
                    return 0LU;
                } else {
                    return std::min(availableInAll, self.max_samples());
                }
            } else {
                (void) self;
                return std::size_t{ 1 };
            }
        }();

        if (available_values_count == 0) {
            return at_least_one_input_has_data ? work_result::has_unprocessed_data : work_result::inputs_empty;
        }

        bool all_writers_available = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return ((output_port<Idx>(&self).writer().available() >= available_values_count) && ... && true);
        }
        (std::make_index_sequence<Self::output_ports::size>());

        if (!all_writers_available) {
            return work_result::writers_not_available;
        }

        auto input_spans = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::make_tuple(input_port<Idx>(&self).reader().get(available_values_count)...); }
        (std::make_index_sequence<Self::input_ports::size>());

        auto writers_tuple = [&self, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
            return std::make_tuple(output_port<Idx>(&self).writer().get(available_values_count)...);
        }
        (std::make_index_sequence<Self::output_ports::size>());

        // TODO: check here whether a process_one(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> process_one(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> process_bulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
        for (std::size_t i = 0; i < available_values_count; ++i) {
            auto results = [&self, &input_spans, i]<std::size_t... Idx>(std::index_sequence<Idx...>) noexcept {
                return meta::invoke_void_wrapped([&self]<typename... Args>(Args... args) { return self.process_one(std::forward<Args>(args)...); }, std::get<Idx>(input_spans)[i]...);
            }
            (std::make_index_sequence<Self::input_ports::size>());

            if constexpr (std::is_same_v<decltype(results), meta::dummy_t>) {
                // process_one returned void

            } else if constexpr (requires { std::get<0>(results); }) {
                static_assert(std::tuple_size_v<decltype(results)> == Self::output_ports::size);
                [&self, &results, &writers_tuple, i]<std::size_t... Idx>(std::index_sequence<Idx...>) { ((std::get<Idx>(writers_tuple).first /*data*/[i] = std::get<Idx>(std::move(results))), ...); }
                (std::make_index_sequence<Self::output_ports::size>());

            } else {
                static_assert(Self::output_ports::size == 1);
                std::get<0>(writers_tuple).first /*data*/[i] = std::move(results);
            }
        }

        if constexpr (Self::output_ports::size > 0) {
            [&self, &writers_tuple, available_values_count]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((output_port<Idx>(&self).writer().publish(std::get<Idx>(writers_tuple).second, available_values_count)), ...);
            }
            (std::make_index_sequence<Self::output_ports::size>());
        }

        bool success = true;
        if constexpr (Self::input_ports::size > 0) {
            [&self, available_values_count, &success]<std::size_t... Idx>(std::index_sequence<Idx...>) {
                ((success = success && input_port<Idx>(&self).reader().consume(available_values_count)), ...);
            }
            (std::make_index_sequence<Self::input_ports::size>());
        }

        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self.name(), available_values_count);
        }

        return success ? work_result::success : work_result::error;
    }
};

using default_policy = read_many_and_publish_many;
} // namespace work_policies

template<typename T, fixed_string PortName, port_type_t Porttype, port_direction_t PortDirection, // TODO: sort default arguments
         std::size_t N_HISTORY = std::dynamic_extent, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, bool OPTIONAL = false,
         gr::Buffer BufferType = gr::circular_buffer<T>>
class port {
public:
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                = T;

    static constexpr bool IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;

    using port_tag                  = std::true_type;

private:
    using ReaderType          = decltype(std::declval<BufferType>().new_reader());
    using WriterType          = decltype(std::declval<BufferType>().new_writer());
    using IoType              = std::conditional_t<IS_INPUT, ReaderType, WriterType>;

    std::string  _name        = static_cast<std::string>(PortName);
    std::int16_t _priority    = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t  _n_history   = N_HISTORY;
    std::size_t  _min_samples = (MIN_SAMPLES == std::dynamic_extent ? 1 : MIN_SAMPLES);
    std::size_t  _max_samples = MAX_SAMPLES;
    bool         _connected   = false;

    IoType       _ioHandler   = new_io_handler();

public:
    [[nodiscard]] constexpr auto
    new_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(65536).new_reader();
        } else {
            return BufferType(65536).new_writer();
        }
    }

    [[nodiscard]] void *
    writer_handler_internal() noexcept {
        static_assert(IS_OUTPUT, "only to be used with output ports");
        return static_cast<void *>(std::addressof(_ioHandler));
    }

    [[nodiscard]] bool
    update_reader_internal(void *buffer_writer_handler_other) noexcept {
        static_assert(IS_INPUT, "only to be used with input ports");

        if (buffer_writer_handler_other == nullptr) {
            return false;
        }

        // TODO: If we want to allow ports with different buffer types to be mixed
        //       this will fail. We need to add a check that two ports that
        //       connect to each other use the same buffer type
        //       (std::any could be a viable approach)
        auto typed_buffer_writer = static_cast<WriterType *>(buffer_writer_handler_other);
        setBuffer(typed_buffer_writer->buffer());
        return true;
    }

public:
    port()             = default;

    port(const port &) = delete;

    auto
    operator=(const port &)
            = delete;

    port(std::string port_name, std::int16_t priority = 0, std::size_t n_history = 0, std::size_t min_samples = 0U, std::size_t max_samples = SIZE_MAX) noexcept
        : _name(std::move(port_name))
        , _priority{ priority }
        , _n_history(n_history)
        , _min_samples(min_samples)
        , _max_samples(max_samples) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr port(port &&other) noexcept
        : _name(std::move(other._name))
        , _priority{ other._priority }
        , _n_history(other._n_history)
        , _min_samples(other._min_samples)
        , _max_samples(other._max_samples) {}

    constexpr port &
    operator=(port &&other) {
        port tmp(std::move(other));
        std::swap(_name, tmp._name);
        std::swap(_priority, tmp._priority);
        std::swap(_n_history, tmp._n_history);
        std::swap(_min_samples, tmp._min_samples);
        std::swap(_max_samples, tmp._max_samples);
        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        return *this;
    }

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return Porttype;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    [[nodiscard]] constexpr supported_type
    pmt_type() const noexcept {
        return T();
    }

    [[nodiscard]] constexpr std::string_view
    name() const noexcept {
        if constexpr (!PortName.empty()) {
            return static_cast<std::string_view>(PortName);
        } else {
            return _name;
        }
    }

    [[nodiscard]] constexpr static bool
    is_optional() noexcept {
        return OPTIONAL;
    }

    [[nodiscard]] constexpr std::int16_t
    priority() const noexcept {
        return _priority;
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    n_history() const noexcept {
        if constexpr (N_HISTORY == std::dynamic_extent) {
            return _n_history;
        } else {
            return N_HISTORY;
        }
    }

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (MIN_SAMPLES == std::dynamic_extent) {
            return _min_samples;
        } else {
            return MIN_SAMPLES;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (MAX_SAMPLES == std::dynamic_extent) {
            return _max_samples;
        } else {
            return MAX_SAMPLES;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        if constexpr (IS_INPUT) {
            return connection_result_t::SUCCESS;
        } else {
            try {
                _ioHandler = BufferType(min_size).new_writer();
            } catch (...) {
                return connection_result_t::FAILED;
            }
        }
        return connection_result_t::SUCCESS;
    }

    [[nodiscard]] BufferType
    buffer() {
        return _ioHandler.buffer();
    }

    void
    setBuffer(gr::Buffer auto buffer) noexcept {
        if constexpr (IS_INPUT) {
            _ioHandler = std::move(buffer.new_reader());
            _connected = true;
        } else {
            _ioHandler = std::move(buffer.new_writer());
        }
    }

    [[nodiscard]] constexpr ReaderType &
    reader() const noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType &
    reader() noexcept {
        static_assert(!IS_OUTPUT, "reader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() const noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    writer() noexcept {
        static_assert(!IS_INPUT, "writer() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        if (_connected == false) {
            return connection_result_t::FAILED;
        }
        _ioHandler = new_io_handler();
        _connected = false;
        return connection_result_t::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] connection_result_t
    connect(Other &&other) noexcept {
        static_assert(IS_OUTPUT && std::remove_cvref_t<Other>::IS_INPUT);
        auto src_buffer = writer_handler_internal();
        return std::forward<Other>(other).update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
    }

    friend class dynamic_port;
};

namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, std::size_t... Is>
consteval fair::meta::typelist<just_t<T, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, typename T, fixed_string BaseName, port_type_t Porttype, port_direction_t PortDirection, std::size_t MIN_SAMPLES = std::dynamic_extent,
         std::size_t MAX_SAMPLES = std::dynamic_extent>
using repeated_ports = decltype(detail::repeated_ports_impl<port<T, BaseName, Porttype, PortDirection, MIN_SAMPLES, MAX_SAMPLES>>(std::make_index_sequence<Count>()));

template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN = port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT = port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using IN_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, fixed_string PortName = "", std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
using OUT_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float, "in">>);
static_assert(Port<decltype(IN<float>("in"))>);
static_assert(Port<OUT<float, "out">>);
static_assert(Port<IN_MSG<float, "in_msg">>);
static_assert(Port<OUT_MSG<float, "out_msg">>);

static_assert(IN<float, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name(); });

/**
 *  Runtime capable wrapper to be used within a block. It's primary purpose is to allow the runtime
 *  initialisation/connections between blocks that are not in the same compilation unit.
 *  Ownership is defined by if the strongly-typed port P is either passed
 *  a) as an lvalue (i.e. P& -> keep reference), or
 *  b) as an rvalue (P&& -> being moved into dyn_port).
 *
 *  N.B. the intended use is within the node/block interface where there is -- once initialised --
 *  always a strong-reference between the strongly-typed port and it's dyn_port wrapper. I.e no ports
 *  are added or removed after the initialisation and the port life-time is coupled to that of it's
 *  parent block/node.
 */
class dynamic_port {
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type
        pmt_type() const noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
                = 0;

        [[nodiscard]] virtual std::string_view
        name() const noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port) noexcept
                = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(void *buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] void *
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(void *buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(!"This works only on input ports");
                return false;
            }
        }

    public:
        wrapper()                = delete;

        wrapper(const wrapper &) = delete;

        auto &
        operator=(const wrapper &)
                = delete;

        auto &
        operator=(wrapper &&)
                = delete;

        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        ~wrapper() override = default;

        [[nodiscard]] constexpr supported_type
        pmt_type() const noexcept override {
            return _value.pmt_type();
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept override {
            return _value.name();
        }

        [[nodiscard]] connection_result_t
        resize_buffer(std::size_t min_size) noexcept override {
            return _value.resize_buffer(min_size);
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept override {
            return _value.disconnect();
        }

        [[nodiscard]] connection_result_t
        connect(dynamic_port &dst_port) noexcept override {
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
            } else {
                assert(!"This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(void *buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type         = void; // a sterile port

    constexpr dynamic_port() = delete;

    template<Port T>
    constexpr dynamic_port(const T &arg) = delete;

    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    pmt_type() const noexcept {
        return _accessor->pmt_type();
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _accessor->name();
    }

    [[nodiscard]] connection_result_t
    resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        return _accessor->disconnect();
    }

    [[nodiscard]] connection_result_t
    connect(dynamic_port &dst_port) noexcept {
        return _accessor->connect(dst_port);
    }
};

static_assert(Port<dynamic_port>);

template<typename...>
struct fixed_node_ports_data_helper;

template<meta::is_typelist_v InputPorts, meta::is_typelist_v OutputPorts>
    requires InputPorts::template
all_of<has_fixed_port_info> &&OutputPorts::template all_of<has_fixed_port_info> struct fixed_node_ports_data_helper<InputPorts, OutputPorts> {
    using input_ports       = InputPorts;
    using output_ports      = OutputPorts;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;

    using all_ports         = meta::concat<input_ports, output_ports>;
};

template<has_fixed_port_info_v... Ports>
struct fixed_node_ports_data_helper<Ports...> {
    using all_ports         = meta::concat<std::conditional_t<fair::meta::is_typelist_v<Ports>, Ports, meta::typelist<Ports>>...>;

    using input_ports       = typename all_ports ::template filter<is_in_port>;
    using output_ports      = typename all_ports ::template filter<is_out_port>;

    using input_port_types  = typename input_ports ::template transform<port_type>;
    using output_port_types = typename output_ports ::template transform<port_type>;
};

template<typename... Arguments>
using fixed_node_ports_data = typename meta::typelist<Arguments...>::template filter<has_fixed_port_info_or_is_typelist>::template apply<fixed_node_ports_data_helper>;

// Ports can either be a list of ports instances,
// or two typelists containing port instances -- one for input
// ports and one for output ports
template<typename Derived, typename... Arguments>
// class node: fixed_node_ports_data<Arguments...>::all_ports::tuple_type {
class node : protected std::tuple<Arguments...> {
public:
    using fixed_ports_data  = fixed_node_ports_data<Arguments...>;

    using all_ports         = typename fixed_ports_data::all_ports;
    using input_ports       = typename fixed_ports_data::input_ports;
    using output_ports      = typename fixed_ports_data::output_ports;
    using input_port_types  = typename fixed_ports_data::input_port_types;
    using output_port_types = typename fixed_ports_data::output_port_types;

    using return_type       = typename output_port_types::tuple_or_type;
    using work_policy       = work_policies::default_policy;
    friend work_policy;

    using min_max_limits = typename meta::typelist<Arguments...>::template filter<is_limits>;
    static_assert(min_max_limits::size <= 1);

private:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string _name{ std::string(fair::meta::type_name<Derived>()) };

    setting_map _exec_metrics{}; //  →  std::map<string, pmt> → fair scheduling, 'int' stand-in for pmtv

    friend class graph;

public:
    auto &
    self() {
        return *static_cast<Derived *>(this);
    }

    const auto &
    self() const {
        return *static_cast<const Derived *>(this);
    }

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _name;
    }

    void
    set_name(std::string name) noexcept {
        _name = std::move(name);
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(std::size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0                    = meta::first_type<input_port_types>;
            using V                      = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs                     = meta::transform_types<meta::rebind_simd_helper<V>::template rebind, input_port_types>;
            const std::tuple input_simds = Vs::template construct<meta::simd_load_ctor>(std::tuple{ in_ptr... });
            const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

#ifdef NOT_YET_PORTED_AS_IT_IS_UNUSED
    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...) && input_port_types::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto       &in0 = std::get<0>(std::tie(inputs...));
        const std::size_t n   = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ... && std::ranges::contiguous_range<Ins>) &&detail::vectorizable<return_type> && detail::node_can_process_simd<Derived>
                      && input_port_types ::template transform<stdx::native_simd>::template all_of<std::is_constructible>) {
            using V       = detail::reduce_to_widest_simd<input_port_types>;
            using Vs      = detail::transform_by_rebind_simd<V, input_port_types>;
            std::size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result      = std::apply([this](auto... args) { return self().process_one(args...); }, input_simds);
                result.copy_to(std::ranges::data(out_range) + i, stdx::element_aligned);
            }

            return process_batch_simd_epilogue<std::bit_ceil(V::size()) / 2>(n - i, std::ranges::data(out_range) + i, (std::ranges::data(inputs) + i)...);
        } else { // no explicit SIMD
            auto             out_it    = out_range.begin();
            std::tuple       it_tuple  = { std::ranges::begin(inputs)... };
            const std::tuple end_tuple = { std::ranges::end(inputs)... };
            while (std::get<0>(it_tuple) != std::get<0>(end_tuple)) {
                *out_it = std::apply([this](auto &...its) { return self().process_one((*its++)...); }, it_tuple);
                ++out_it;
            }
            return true;
        }
    }
#endif

    [[nodiscard]] setting_map &
    exec_metrics() noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] setting_map const &
    exec_metrics() const noexcept {
        return _exec_metrics;
    }

    [[nodiscard]] constexpr std::size_t
    min_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::min;
        } else {
            return 0;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_samples() const noexcept {
        if constexpr (min_max_limits::size == 1) {
            return min_max_limits::template at<0>::max;
        } else {
            return std::numeric_limits<std::size_t>::max();
        }
    }

    work_result
    work() noexcept {
        return work_policy::work(self());
    }
};

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::input_ports>();
    return std::get<typename Self::input_ports::template at<Index>>(*self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, typename Self::output_ports>();
    return std::get<typename Self::output_ports::template at<Index>>(*self);
}

#define ENABLE_PYTHON_INTEGRATION
#ifdef ENABLE_PYTHON_INTEGRATION

// TODO: Not yet implemented
class dynamic_node {
private:
    // TODO: replace the following with array<2, vector<dynamic_port>>
    using dynamic_ports = std::vector<dynamic_port>;
    dynamic_ports                                         _dynamic_input_ports;
    dynamic_ports                                         _dynamic_output_ports;

    std::function<void(dynamic_ports &, dynamic_ports &)> _process;

public:
    void
    work() {
        _process(_dynamic_input_ports, _dynamic_output_ports);
    }

    template<typename T>
    void
    add_port(T &&port) {
        switch (port.direction()) {
        case port_direction_t::INPUT:
            if (auto portID = port_index<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_input_ports.emplace_back(std::forward<T>(port));
            break;

        case port_direction_t::OUTPUT:
            if (auto portID = port_index<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_output_ports.emplace_back(std::forward<T>(port));
            break;

        default: assert(false && "cannot add port with ANY designation");
        }
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::size_t index) {
        return index < _dynamic_input_ports.size() ? std::optional{ &_dynamic_input_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_input_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_input_ports.cbegin(), _dynamic_input_ports.cend(), portNameMatches);
        return it != _dynamic_input_ports.cend() ? std::optional{ std::distance(_dynamic_input_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::string_view name) {
        if (const auto index = dynamic_input_port_index(name); index.has_value()) {
            return &_dynamic_input_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::size_t index) {
        return index < _dynamic_output_ports.size() ? std::optional{ &_dynamic_output_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_output_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_output_ports.cbegin(), _dynamic_output_ports.cend(), portNameMatches);
        return it != _dynamic_output_ports.cend() ? std::optional{ std::distance(_dynamic_output_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::string_view name) {
        if (const auto index = dynamic_output_port_index(name); index.has_value()) {
            return &_dynamic_output_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_input_ports() const noexcept {
        return _dynamic_input_ports;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_output_ports() const noexcept {
        return _dynamic_output_ports;
    }
};

#endif

template<meta::source_node Left, meta::sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename Left::input_ports, meta::remove_at<InId, typename Right::input_ports>>,
                                meta::concat<meta::remove_at<OutId, typename Left::output_ports>, typename Right::output_ports>> {
private:
    // copy-paste from above, keep in sync
    using base = node<merged_node<Left, Right, OutId, InId>, meta::concat<typename Left::input_ports, meta::remove_at<InId, typename Right::input_ports>>,
                      meta::concat<meta::remove_at<OutId, typename Left::output_ports>, typename Right::output_ports>>;

    Left  left;
    Right right;

    template<std::size_t I>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return left.process_one(std::get<Is>(input_tuple)...); }
        (std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = Left::input_port_types::size;
            constexpr std::size_t second_offset = Left::input_port_types::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp), std::get<second_offset + Js>(input_tuple)...);
        }
        (std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename base::input_port_types;
    using output_port_types = typename base::output_port_types;
    using return_type       = typename base::return_type;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    template<meta::any_simd... Ts>
        requires meta::vectorizable<return_type> && input_port_types::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> &&meta::node_can_process_simd<Left>
            &&meta::node_can_process_simd<Right> constexpr stdx::rebind_simd_t<return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
              process_one(Ts... inputs) {
        return apply_right<InId, Right::input_port_types::size() - InId - 1>(std::tie(inputs...), apply_left<Left::input_port_types::size()>(std::tie(inputs...)));
    }

    template<typename... Ts>
    // In order to have nicer error messages, this is checked in the function body
    // requires input_port_types::template are_equal<std::remove_cvref_t<Ts>...>
    constexpr return_type
    process_one(Ts &&...inputs) {
        if constexpr (!input_port_types::template are_equal<std::remove_cvref_t<Ts>...>) {
            meta::print_types<decltype(this), input_port_types, std::remove_cvref_t<Ts>...> error{};
        }

        if constexpr (Left::output_port_types::size == 1) { // only the result from the right node needs to be returned
            return apply_right<InId, Right::input_port_types::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                 apply_left<Left::input_port_types::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<Left::input_port_types::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, Right::input_port_types::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (Left::output_port_types::size == 2 && Right::output_port_types::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (Left::output_port_types::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (Right::output_port_types::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<Left::output_port_types::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...));
                }
                (std::make_index_sequence<OutId>(), std::make_index_sequence<Left::output_port_types::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    }
};

template<std::size_t OutId, std::size_t InId, meta::source_node A, meta::sink_node B>
[[gnu::always_inline]] constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>, typename std::remove_cvref_t<B>::input_port_types::template at<InId>>) {
        fair::meta::print_types<fair::meta::message_type<"OUTPUT_PORTS_ARE:">, typename std::remove_cvref_t<A>::output_port_types, std::integral_constant<int, OutId>,
                                typename std::remove_cvref_t<A>::output_port_types::template at<OutId>,

                                fair::meta::message_type<"INPUT_PORTS_ARE:">, typename std::remove_cvref_t<A>::input_port_types, std::integral_constant<int, InId>,
                                typename std::remove_cvref_t<A>::input_port_types::template at<InId>>{};
    }
    return { std::forward<A>(a), std::forward<B>(b) };
}

template<fixed_string OutName, fixed_string InName, meta::source_node A, meta::sink_node B>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) {
    constexpr std::size_t OutId = meta::indexForName<OutName, typename A::output_ports>();
    constexpr std::size_t InId  = meta::indexForName<InName, typename B::input_ports>();
    static_assert(OutId != -1);
    static_assert(InId != -1);
    static_assert(std::same_as<typename std::remove_cvref_t<A>::output_port_types::template at<OutId>, typename std::remove_cvref_t<B>::input_port_types::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

class graph {
private:
    class node_model {
    public:
        virtual ~node_model() = default;

        virtual std::string_view
        name() const
                = 0;

        virtual work_result
        work() = 0;

        virtual void *
        raw() const
                = 0;
    };

    template<typename T>
    class reference_node_wrapper final : public node_model {
    private:
        T *_node;

        auto &
        data() {
            return *_node;
        }

        const auto &
        data() const {
            return *_node;
        }

    public:
        reference_node_wrapper(const reference_node_wrapper &other) = delete;

        reference_node_wrapper &
        operator=(const reference_node_wrapper &other)
                = delete;

        reference_node_wrapper(reference_node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        reference_node_wrapper &
        operator=(reference_node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~reference_node_wrapper() override = default;

        template<typename In>
        reference_node_wrapper(In &&node) : _node(std::forward<In>(node)) {}

        work_result
        work() override {
            return data().work();
        }

        std::string_view
        name() const override {
            return data().name();
        }

        void *
        raw() const override {
            return _node;
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        std::unique_ptr<node_model> _src_node;
        std::unique_ptr<node_model> _dst_node;
        std::size_t                 _src_port_index;
        std::size_t                 _dst_port_index;
        int32_t                     _weight;
        std::string                 _name; // custom edge name
        bool                        _connected;

    public:
        edge()             = delete;

        edge(const edge &) = delete;

        edge &
        operator=(const edge &)
                = delete;

        edge(edge &&) noexcept = default;

        edge &
        operator=(edge &&) noexcept
                = default;

        edge(std::unique_ptr<node_model> src_node, std::size_t src_port_index, std::unique_ptr<node_model> dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(std::move(src_node))
            , _dst_node(std::move(dst_node))
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
            // if (!_src_node->port<OUTPUT>(_src_port_index)) {
            //     throw fmt::format("source node '{}' has not output port id {}", std::string() /* _src_node->name() */, _src_port_index);
            // }
            // if (!_dst_node->port<INPUT>(_dst_port_index)) {
            //     throw fmt::format("destination node '{}' has not output port id {}", std::string() /*_dst_node->name()*/, _dst_port_index);
            // }
            // const dynamic_port& src_port = *_src_node->port<OUTPUT>(_src_port_index).value();
            // const dynamic_port& dst_port = *_dst_node->port<INPUT>(_dst_port_index).value();
            // if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            //     throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
            //         std::string() /*_src_node->name()*/, std::string() /*src_port.name()*/, src_port.pmt_type().index(),
            //         std::string() /*_dst_node->name()*/, std::string() /*dst_port.name()*/, dst_port.pmt_type().index(),
            //         _weight, _name, dst_port.pmt_type().index());
            // }
        }

        // edge(std::shared_ptr<node_model> src_node, std::string_view src_port_name, std::shared_ptr<node_model> dst_node, std::string_view dst_port_name, int32_t weight, std::string_view name) :
        //         _src_node(src_node), _dst_node(dst_node), _weight(weight), _name(name) {
        //     const auto src_id = _src_node->port_index<OUTPUT>(src_port_name);
        //     const auto dst_id = _dst_node->port_index<INPUT>(dst_port_name);
        //     if (!src_id) {
        //         throw std::invalid_argument(fmt::format("source node '{}' has not output port '{}'", std::string() /*_src_node->name()*/, src_port_name));
        //     }
        //     if (!dst_id) {
        //         throw fmt::format("destination node '{}' has not output port '{}'", std::string() /*_dst_node->name()*/, dst_port_name);
        //     }
        //     _src_port_index = src_id.value();
        //     _dst_port_index = dst_id.value();
        //     const dynamic_port& src_port = *src_node->port<OUTPUT>(_src_port_index).value();
        //     const dynamic_port& dst_port = *dst_node->port<INPUT>(_dst_port_index).value();
        //     if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
        //         throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
        //                           std::string() /*_src_node->name()*/, src_port.name(), src_port.pmt_type().index(),
        //                           std::string() /*_dst_node->name()*/, dst_port.name(), dst_port.pmt_type().index(),
        //                           _weight, _name, dst_port.pmt_type().index());
        //     }
        // }

        [[nodiscard]] constexpr int32_t
        weight() const noexcept {
            return _weight;
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept {
            return _name;
        }

        [[nodiscard]] constexpr bool
        connected() const noexcept {
            return _connected;
        }

        [[nodiscard]] connection_result_t
        connect() noexcept {
            return connection_result_t::FAILED;
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept { /* return _dst_node->port<INPUT>(_dst_port_index).value()->disconnect(); */
            return connection_result_t::FAILED;
        }
    };

    std::vector<edge>                        _edges;
    std::vector<std::unique_ptr<node_model>> _nodes;

public:
    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect(Source_ &src_node_raw, Destination_ &dst_node_raw, int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
        using Source      = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        static_assert(std::is_same_v<typename Source::output_port_types::template at<src_port_index>, typename Destination::input_port_types::template at<dst_port_index>>,
                      "The source port type needs to match the sink port type");

        OutPort auto &source_port      = output_port<src_port_index>(&src_node_raw);
        InPort auto  &destination_port = input_port<dst_port_index>(&dst_node_raw);

        if (!std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) { return registered_node->raw() == std::addressof(src_node_raw); })
            || !std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) { return registered_node->raw() == std::addressof(dst_node_raw); })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_node_raw.name(), src_port_index, dst_node_raw.name(), dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            std::unique_ptr<node_model> src_node = std::make_unique<reference_node_wrapper<Source>>(std::addressof(src_node_raw));
            std::unique_ptr<node_model> dst_node = std::make_unique<reference_node_wrapper<Destination>>(std::addressof(dst_node_raw));
            _edges.emplace_back(std::move(src_node), src_port_index, std::move(dst_node), src_port_index, weight, name);
        }

        return result;
    }

    template<fixed_string src_port_name, fixed_string dst_port_name, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect(Source_ &&src_node_raw, Destination_ &&dst_node_raw, int32_t weight = 0, std::string_view name = "unnamed edge") noexcept {
        using Source      = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        return connect<meta::indexForName<src_port_name, typename Source::output_ports>(), meta::indexForName<dst_port_name, typename Destination::input_ports>()>(std::forward<Source_>(src_node_raw),
                                                                                                                                                                   std::forward<Destination_>(
                                                                                                                                                                           dst_node_raw),
                                                                                                                                                                   weight, name);
    }

    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node>
    void
    register_node(Node &node) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        _nodes.push_back(std::make_unique<reference_node_wrapper<Node>>(std::addressof(node)));
    }

    work_result
    work() {
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == work_result::error) {
                    return work_result::error;
                } else if (result == work_result::has_unprocessed_data) {
                    // nothing
                } else if (result == work_result::inputs_empty) {
                    // nothing
                } else if (result == work_result::success) {
                    something_happened = true;
                } else if (result == work_result::writers_not_available) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return work_result::inputs_empty;
    }
};

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const connection_result_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_type_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_direction_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_domain_t &value) {
    return os << static_cast<int>(value);
}

#ifndef __EMSCRIPTEN__
auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
auto
this_source_location() {
    return "not yet implemented";
}
#endif // __EMSCRIPTEN__

} // namespace fair::graph

#endif // include guard
