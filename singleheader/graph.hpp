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
#ifndef GRAPH_PROTOTYPE_TYPELIST_HPP
#define GRAPH_PROTOTYPE_TYPELIST_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>

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

template<int N, typename List>
using left_of = typename split_at<N, List>::first;

template<int N, typename List>
using right_of = typename split_at<N + 1, List>::second;

// remove_at /////////////
template<int Idx, typename List>
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
template<typename... Ts>
struct typelist {
    static inline constexpr std::integral_constant<int, sizeof...(Ts)> size = {};

    template<template<typename...> class Other>
    using apply = Other<Ts...>;

    template<int I>
    using at = first_type<typename detail::splitter<I>::template second<Ts...>>;

    template<typename... Other>
    static constexpr inline bool each_convertible_to = (std::convertible_to<Ts, Other> && ...);

    template<typename... Other>
    static constexpr inline bool each_convertible_from = (std::convertible_to<Other, Ts> && ...);

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
};
} // namespace fair::meta

#endif // GRAPH_PROTOTYPE_TYPELIST_HPP
#ifndef GRAPH_PROTOTYPE_GRAPH_HPP
#define GRAPH_PROTOTYPE_GRAPH_HPP

#ifndef GRAPH_PROTOTYPE_TYPELIST_HPP
#endif


#include <iostream>
#include <ranges>
#include <tuple>

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
using std::size_t;

namespace stdx = vir::stdx;

namespace detail {
#if HAVE_SOURCE_LOCATION
[[gnu::always_inline]] inline void
precondition(bool cond, const std::source_location loc = std::source_location::current()) {
    struct handle {
        [[noreturn]] static void
        failure(std::source_location const &loc) {
            std::clog << "failed precondition in " << loc.file_name() << ':' << loc.line() << ':'
                      << loc.column() << ": `" << loc.function_name() << "`\n";
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

template<typename V, typename T = void>
concept any_simd = stdx::is_simd_v<V>
                && (std::same_as<T, void> || std::same_as<T, typename V::value_type>);

template<typename V, typename T>
concept t_or_simd = std::same_as<V, T> || any_simd<V, T>;

template<typename T>
concept vectorizable = std::constructible_from<stdx::simd<T>>;

template<typename A, typename B>
struct wider_native_simd_size
    : std::conditional<(stdx::native_simd<A>::size() > stdx::native_simd<B>::size()), A, B> {};

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
    template<detail::any_simd W>
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
concept any_node = requires(Node &n, typename Node::input_ports::tuple_type const &inputs) {
    { n.in } -> std::same_as<typename Node::input_ports const &>;
    { n.out } -> std::same_as<typename Node::output_ports const &>;
    {
        []<size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                -> decltype(n.process_one(std::get<Is>(tup)...)) {
            return {};
        }(n, inputs, std::make_index_sequence<Node::input_ports::size>())
    } -> std::same_as<typename Node::return_type>;
};

template<typename Node>
concept node_can_process_simd
        = any_node<Node>
       && requires(Node &n,
                   typename transform_to_widest_simd<typename Node::input_ports::typelist>::
                           template apply<std::tuple> const &inputs) {
              {
                  []<size_t... Is>(Node &n, auto const &tup, std::index_sequence<Is...>)
                          -> decltype(n.process_one(std::get<Is>(tup)...)) {
                      return {};
                  }(n, inputs, std::make_index_sequence<Node::input_ports::size>())
              } -> detail::any_simd<typename Node::return_type>;
          };

// Workaround bug in Clang:
// std::is_constructible should be a valid template argument for the template template parameter
// `template <typename> class T`, but because std::is_constructible<T, Args...> has a parameter
// pack, Clang considers the program ill-formed. As so often, another indirection solves the
// problem. *Sigh*
template <typename T>
using is_constructible = std::is_constructible<T>;
} // namespace detail

enum class port_direction { in, out };

template<port_direction D, int I, typename T>
struct port_id {
    using type                                       = T;
    static inline constexpr port_direction direction = D;
    static inline constexpr int            id        = I;
};

template<port_direction D, typename... Types>
struct portlist {
    static inline constexpr port_direction                                direction = D;
    static inline constexpr std::integral_constant<int, sizeof...(Types)> size      = {};
    //
    using typelist      = meta::typelist<Types...>;
    using tuple_type    = std::tuple<Types...>;
    using tuple_or_type = std::conditional_t<sizeof...(Types) == 1,
                                             typename typelist::template at<0>, tuple_type>;

    template<int I>
    using at = port_id<D, I, typename typelist::template at<I>>;

    template<typename I>
    constexpr at<I::value>
    operator[](I) const {
        return {};
    }

    template<typename... Other>
    static constexpr inline bool are_equal = std::same_as<typelist, meta::typelist<Other...>>;

    template<typename... Other>
    static constexpr inline bool are_convertible_to = (std::convertible_to<Types, Other> && ...);

    template<typename... Other>
    static constexpr inline bool are_convertible_from = (std::convertible_to<Other, Types> && ...);
};

template<port_direction D, typename... Types>
struct portlist<D, meta::typelist<Types...>> : portlist<D, Types...> {};

template<typename... Types>
using make_input_ports = portlist<port_direction::in, Types...>;

template<typename... Types>
using make_output_ports = portlist<port_direction::out, Types...>;

// simple non-reentrant circular buffer
template<typename T, size_t Size>
class port_data {
    static_assert(std::has_single_bit(Size), "Size must be a power-of-2 value");
    alignas(64) std::array<T, Size> m_buffer      = {};
    size_t                         m_read_offset  = 0;
    size_t                         m_write_offset = 0;

    static inline constexpr size_t s_bitmask      = Size - 1;

public:
    static inline constexpr std::integral_constant<size_t, Size> size = {};

    size_t
    can_read() const {
        return m_write_offset >= m_read_offset ? m_write_offset - m_read_offset
                                               : size - m_read_offset;
    }

    size_t
    can_write() const {
        return m_write_offset >= m_read_offset ? size - m_write_offset
                                               : m_read_offset - m_write_offset;
    }

    std::span<const T>
    request_read() {
        return request_read(can_read());
    }

    std::span<const T>
    request_read(size_t n) {
        detail::precondition(can_read() >= n);
        const auto begin = m_buffer.begin() + m_read_offset;
        m_read_offset += n;
        m_read_offset &= s_bitmask;
        return std::span<const T>{ begin, n };
    }

    std::span<T>
    request_write() {
        return request_write(can_write());
    }

    std::span<T>
    request_write(size_t n) {
        detail::precondition(can_write() >= n);
        const auto begin = m_buffer.begin() + m_write_offset;
        m_write_offset += n;
        m_write_offset &= s_bitmask;
        return std::span<T>{ begin, n };
    }
};

template<typename Derived, typename InputPorts, typename OutputPorts>
class node {
public:
    using input_ports                        = InputPorts;
    using output_ports                       = OutputPorts;
    using return_type                        = typename output_ports::tuple_or_type;

    static inline constexpr input_ports  in  = {};
    static inline constexpr output_ports out = {};

    template<std::size_t N>
    [[gnu::always_inline]] constexpr bool
    process_batch_simd_epilogue(size_t n, auto out_ptr, auto... in_ptr) {
        if constexpr (N == 0) return true;
        else if (N <= n) {
            using In0 = meta::first_type<typename input_ports::typelist>;
            using V   = stdx::resize_simd_t<N, stdx::native_simd<In0>>;
            using Vs  = meta::transform_types<detail::rebind_simd_helper<V>::template rebind,
                                             typename input_ports::typelist>;
            const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                    std::tuple{ in_ptr... });
            const stdx::simd result = std::apply(
                    [this](auto... args) {
                        return static_cast<Derived *>(this)->process_one(args...);
                    },
                    input_simds);
            result.copy_to(out_ptr, stdx::element_aligned);
            return process_batch_simd_epilogue<N / 2>(n, out_ptr + N, (in_ptr + N)...);
        } else
            return process_batch_simd_epilogue<N / 2>(n, out_ptr, in_ptr...);
    }

    template<std::ranges::forward_range... Ins>
        requires(std::ranges::sized_range<Ins> && ...) && input_ports::template
    are_equal<std::ranges::range_value_t<Ins>...> constexpr bool
    process_batch(port_data<return_type, 1024> &out, Ins &&...inputs) {
        const auto  &in0 = std::get<0>(std::tie(inputs...));
        const size_t n   = std::ranges::size(in0);
        detail::precondition(((n == std::ranges::size(inputs)) && ...));
        auto &&out_range = out.request_write(n);
        // if SIMD makes sense (i.e. input and output ranges are contiguous and all types are
        // vectorizable)
        if constexpr ((std::ranges::contiguous_range<decltype(out_range)> && ...
                       && std::ranges::contiguous_range<Ins>) &&detail::vectorizable<return_type>
                      && detail::node_can_process_simd<Derived>
                      && meta::transform_types<detail::is_constructible,
                                               meta::transform_types<stdx::native_simd,
                                                                     typename input_ports::typelist>>::
                              template apply<std::conjunction>::value) {
            using V  = detail::reduce_to_widest_simd<typename input_ports::typelist>;
            using Vs = detail::transform_by_rebind_simd<V, typename input_ports::typelist>;
            size_t i = 0;
            for (i = 0; i + V::size() <= n; i += V::size()) {
                const std::tuple input_simds = Vs::template construct<detail::simd_load_ctor>(
                        std::tuple{ (std::ranges::data(inputs) + i)... });
                const stdx::simd result = std::apply(
                        [this](auto... args) {
                            return static_cast<Derived *>(this)->process_one(args...);
                        },
                        input_simds);
                result.copy_to(std::ranges::data(out_range) + i, stdx::element_aligned);
            }

            return process_batch_simd_epilogue<std::bit_ceil(V::size())
                                               / 2>(n - i, std::ranges::data(out_range) + i,
                                                    (std::ranges::data(inputs) + i)...);
        } else { // no explicit SIMD
            auto             out_it    = out_range.begin();
            std::tuple       it_tuple  = { std::ranges::begin(inputs)... };
            const std::tuple end_tuple = { std::ranges::end(inputs)... };
            while (std::get<0>(it_tuple) != std::get<0>(end_tuple)) {
                *out_it = std::apply(
                        [this](auto &...its) {
                            return static_cast<Derived *>(this)->process_one((*its++)...);
                        },
                        it_tuple);
                ++out_it;
            }
            return true;
        }
    }

protected:
    constexpr node() noexcept = default;
};

template<detail::any_node Left, detail::any_node Right, int OutId, int InId>
class merged_node
    : public node<merged_node<Left, Right, OutId, InId>,
                  make_input_ports<
                          meta::concat<typename Left::input_ports::typelist,
                                       meta::remove_at<InId, typename Right::input_ports::typelist>>>,
                  make_output_ports<
                          meta::concat<meta::remove_at<OutId, typename Left::output_ports::typelist>,
                                       typename Right::output_ports::typelist>>> {
private:
    using base = node<
            merged_node,
            make_input_ports<
                    meta::concat<typename Left::input_ports::typelist,
                                 meta::remove_at<InId, typename Right::input_ports::typelist>>>,
            make_output_ports<meta::concat<meta::remove_at<OutId, typename Left::output_ports::typelist>,
                                           typename Right::output_ports::typelist>>>;

    Left  left;
    Right right;

    template<std::size_t... Is>
    [[gnu::always_inline]] constexpr auto
    apply_left(auto &&input_tuple, std::index_sequence<Is...>) {
        return left.process_one(std::get<Is>(input_tuple)...);
    }

    template<std::size_t... Is, std::size_t... Js>
    [[gnu::always_inline]] constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp, std::index_sequence<Is...>,
                std::index_sequence<Js...>) {
        constexpr int first_offset  = Left::input_ports::size;
        constexpr int second_offset = Left::input_ports::size + sizeof...(Is);
        static_assert(second_offset + sizeof...(Js)
                      == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
        return right.process_one(std::get<first_offset + Is>(input_tuple)..., std::move(tmp),
                                 std::get<second_offset + Js>(input_tuple)...);
    }

public:
    using input_ports  = typename base::input_ports;
    using output_ports = typename base::output_ports;
    using return_type  = typename output_ports::tuple_or_type;

    [[gnu::always_inline]] constexpr merged_node(Left l, Right r)
        : left(std::move(l)), right(std::move(r)) {}

    template<detail::any_simd... Ts>
        requires detail::vectorizable<return_type> && input_ports::template
    are_equal<typename std::remove_cvref_t<Ts>::value_type...> &&detail::node_can_process_simd<Left>
            &&detail::node_can_process_simd<Right> constexpr stdx::rebind_simd_t<
                    return_type, meta::first_type<meta::typelist<std::remove_cvref_t<Ts>...>>>
            process_one(Ts... inputs) {
        return apply_right(std::tie(inputs...),
                           apply_left(std::tie(inputs...),
                                      std::make_index_sequence<Left::input_ports::size()>()),
                           std::make_index_sequence<InId>(),
                           std::make_index_sequence<Right::input_ports::size() - InId - 1>());
    }

    template<typename... Ts>
        requires input_ports::template
    are_equal<std::remove_cvref_t<Ts>...> constexpr typename output_ports::tuple_or_type
    process_one(Ts &&...inputs) {
        if constexpr (Left::output_ports::size
                      == 1) { // only the result from the right node needs to be returned
            return apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                               apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                          std::make_index_sequence<Left::input_ports::size()>()),
                               std::make_index_sequence<InId>(),
                               std::make_index_sequence<Right::input_ports::size() - InId - 1>());
        } else {
            // left produces a tuple
            auto left_out = apply_left(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                       std::make_index_sequence<Left::input_ports::size()>());
            auto right_out
                    = apply_right(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                  std::move(std::get<OutId>(left_out)),
                                  std::make_index_sequence<InId>(),
                                  std::make_index_sequence<Right::input_ports::size() - InId - 1>());

            if constexpr (Left::output_ports::size == 2 && Right::output_ports::size == 1)
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)),
                                       std::move(right_out));
            else if constexpr (Left::output_ports::size == 2)
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))),
                                      std::move(right_out));
            else if constexpr (Right::output_ports::size == 1)
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>,
                                                                 std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is...>(left_out)),
                                           std::move(std::get<OutId + 1 + Js...>(left_out)),
                                           std::move(right_out));
                }(std::make_index_sequence<OutId>(),
                       std::make_index_sequence<Left::output_ports::size - OutId - 1>());
            else
                return [&]<std::size_t... Is, std::size_t... Js,
                           std::size_t... Ks>(std::index_sequence<Is...>,
                                              std::index_sequence<Js...>,
                                              std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is...>(left_out)),
                                           std::move(std::get<OutId + 1 + Js...>(left_out)),
                                           std::move(std::get<Ks...>(right_out)));
                }(std::make_index_sequence<OutId>(),
                       std::make_index_sequence<Left::output_ports::size - OutId - 1>(),
                       std::make_index_sequence<Right::output_ports::size>());
        }
    }
};

template<int OutId, int InId, detail::any_node A, detail::any_node B>
    requires std::same_as<typename std::remove_cvref_t<A>::output_ports::template at<OutId>::type,
                          typename std::remove_cvref_t<B>::input_ports::template at<InId>::type>
[[gnu::always_inline]] constexpr auto
merge(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    return { std::forward<A>(a), std::forward<B>(b) };
}
} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_GRAPH_HPP
