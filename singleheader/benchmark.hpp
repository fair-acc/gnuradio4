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
// Formatting library for C++ - color support
//
// Copyright (c) 2018 - present, Victor Zverovich and fmt contributors
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_COLOR_H_
#define FMT_COLOR_H_


// __declspec(deprecated) is broken in some MSVC versions.
#if FMT_MSC_VER
#  define FMT_DEPRECATED_NONMSVC
#else
#  define FMT_DEPRECATED_NONMSVC FMT_DEPRECATED
#endif

FMT_BEGIN_NAMESPACE
FMT_MODULE_EXPORT_BEGIN

enum class color : uint32_t {
  alice_blue = 0xF0F8FF,               // rgb(240,248,255)
  antique_white = 0xFAEBD7,            // rgb(250,235,215)
  aqua = 0x00FFFF,                     // rgb(0,255,255)
  aquamarine = 0x7FFFD4,               // rgb(127,255,212)
  azure = 0xF0FFFF,                    // rgb(240,255,255)
  beige = 0xF5F5DC,                    // rgb(245,245,220)
  bisque = 0xFFE4C4,                   // rgb(255,228,196)
  black = 0x000000,                    // rgb(0,0,0)
  blanched_almond = 0xFFEBCD,          // rgb(255,235,205)
  blue = 0x0000FF,                     // rgb(0,0,255)
  blue_violet = 0x8A2BE2,              // rgb(138,43,226)
  brown = 0xA52A2A,                    // rgb(165,42,42)
  burly_wood = 0xDEB887,               // rgb(222,184,135)
  cadet_blue = 0x5F9EA0,               // rgb(95,158,160)
  chartreuse = 0x7FFF00,               // rgb(127,255,0)
  chocolate = 0xD2691E,                // rgb(210,105,30)
  coral = 0xFF7F50,                    // rgb(255,127,80)
  cornflower_blue = 0x6495ED,          // rgb(100,149,237)
  cornsilk = 0xFFF8DC,                 // rgb(255,248,220)
  crimson = 0xDC143C,                  // rgb(220,20,60)
  cyan = 0x00FFFF,                     // rgb(0,255,255)
  dark_blue = 0x00008B,                // rgb(0,0,139)
  dark_cyan = 0x008B8B,                // rgb(0,139,139)
  dark_golden_rod = 0xB8860B,          // rgb(184,134,11)
  dark_gray = 0xA9A9A9,                // rgb(169,169,169)
  dark_green = 0x006400,               // rgb(0,100,0)
  dark_khaki = 0xBDB76B,               // rgb(189,183,107)
  dark_magenta = 0x8B008B,             // rgb(139,0,139)
  dark_olive_green = 0x556B2F,         // rgb(85,107,47)
  dark_orange = 0xFF8C00,              // rgb(255,140,0)
  dark_orchid = 0x9932CC,              // rgb(153,50,204)
  dark_red = 0x8B0000,                 // rgb(139,0,0)
  dark_salmon = 0xE9967A,              // rgb(233,150,122)
  dark_sea_green = 0x8FBC8F,           // rgb(143,188,143)
  dark_slate_blue = 0x483D8B,          // rgb(72,61,139)
  dark_slate_gray = 0x2F4F4F,          // rgb(47,79,79)
  dark_turquoise = 0x00CED1,           // rgb(0,206,209)
  dark_violet = 0x9400D3,              // rgb(148,0,211)
  deep_pink = 0xFF1493,                // rgb(255,20,147)
  deep_sky_blue = 0x00BFFF,            // rgb(0,191,255)
  dim_gray = 0x696969,                 // rgb(105,105,105)
  dodger_blue = 0x1E90FF,              // rgb(30,144,255)
  fire_brick = 0xB22222,               // rgb(178,34,34)
  floral_white = 0xFFFAF0,             // rgb(255,250,240)
  forest_green = 0x228B22,             // rgb(34,139,34)
  fuchsia = 0xFF00FF,                  // rgb(255,0,255)
  gainsboro = 0xDCDCDC,                // rgb(220,220,220)
  ghost_white = 0xF8F8FF,              // rgb(248,248,255)
  gold = 0xFFD700,                     // rgb(255,215,0)
  golden_rod = 0xDAA520,               // rgb(218,165,32)
  gray = 0x808080,                     // rgb(128,128,128)
  green = 0x008000,                    // rgb(0,128,0)
  green_yellow = 0xADFF2F,             // rgb(173,255,47)
  honey_dew = 0xF0FFF0,                // rgb(240,255,240)
  hot_pink = 0xFF69B4,                 // rgb(255,105,180)
  indian_red = 0xCD5C5C,               // rgb(205,92,92)
  indigo = 0x4B0082,                   // rgb(75,0,130)
  ivory = 0xFFFFF0,                    // rgb(255,255,240)
  khaki = 0xF0E68C,                    // rgb(240,230,140)
  lavender = 0xE6E6FA,                 // rgb(230,230,250)
  lavender_blush = 0xFFF0F5,           // rgb(255,240,245)
  lawn_green = 0x7CFC00,               // rgb(124,252,0)
  lemon_chiffon = 0xFFFACD,            // rgb(255,250,205)
  light_blue = 0xADD8E6,               // rgb(173,216,230)
  light_coral = 0xF08080,              // rgb(240,128,128)
  light_cyan = 0xE0FFFF,               // rgb(224,255,255)
  light_golden_rod_yellow = 0xFAFAD2,  // rgb(250,250,210)
  light_gray = 0xD3D3D3,               // rgb(211,211,211)
  light_green = 0x90EE90,              // rgb(144,238,144)
  light_pink = 0xFFB6C1,               // rgb(255,182,193)
  light_salmon = 0xFFA07A,             // rgb(255,160,122)
  light_sea_green = 0x20B2AA,          // rgb(32,178,170)
  light_sky_blue = 0x87CEFA,           // rgb(135,206,250)
  light_slate_gray = 0x778899,         // rgb(119,136,153)
  light_steel_blue = 0xB0C4DE,         // rgb(176,196,222)
  light_yellow = 0xFFFFE0,             // rgb(255,255,224)
  lime = 0x00FF00,                     // rgb(0,255,0)
  lime_green = 0x32CD32,               // rgb(50,205,50)
  linen = 0xFAF0E6,                    // rgb(250,240,230)
  magenta = 0xFF00FF,                  // rgb(255,0,255)
  maroon = 0x800000,                   // rgb(128,0,0)
  medium_aquamarine = 0x66CDAA,        // rgb(102,205,170)
  medium_blue = 0x0000CD,              // rgb(0,0,205)
  medium_orchid = 0xBA55D3,            // rgb(186,85,211)
  medium_purple = 0x9370DB,            // rgb(147,112,219)
  medium_sea_green = 0x3CB371,         // rgb(60,179,113)
  medium_slate_blue = 0x7B68EE,        // rgb(123,104,238)
  medium_spring_green = 0x00FA9A,      // rgb(0,250,154)
  medium_turquoise = 0x48D1CC,         // rgb(72,209,204)
  medium_violet_red = 0xC71585,        // rgb(199,21,133)
  midnight_blue = 0x191970,            // rgb(25,25,112)
  mint_cream = 0xF5FFFA,               // rgb(245,255,250)
  misty_rose = 0xFFE4E1,               // rgb(255,228,225)
  moccasin = 0xFFE4B5,                 // rgb(255,228,181)
  navajo_white = 0xFFDEAD,             // rgb(255,222,173)
  navy = 0x000080,                     // rgb(0,0,128)
  old_lace = 0xFDF5E6,                 // rgb(253,245,230)
  olive = 0x808000,                    // rgb(128,128,0)
  olive_drab = 0x6B8E23,               // rgb(107,142,35)
  orange = 0xFFA500,                   // rgb(255,165,0)
  orange_red = 0xFF4500,               // rgb(255,69,0)
  orchid = 0xDA70D6,                   // rgb(218,112,214)
  pale_golden_rod = 0xEEE8AA,          // rgb(238,232,170)
  pale_green = 0x98FB98,               // rgb(152,251,152)
  pale_turquoise = 0xAFEEEE,           // rgb(175,238,238)
  pale_violet_red = 0xDB7093,          // rgb(219,112,147)
  papaya_whip = 0xFFEFD5,              // rgb(255,239,213)
  peach_puff = 0xFFDAB9,               // rgb(255,218,185)
  peru = 0xCD853F,                     // rgb(205,133,63)
  pink = 0xFFC0CB,                     // rgb(255,192,203)
  plum = 0xDDA0DD,                     // rgb(221,160,221)
  powder_blue = 0xB0E0E6,              // rgb(176,224,230)
  purple = 0x800080,                   // rgb(128,0,128)
  rebecca_purple = 0x663399,           // rgb(102,51,153)
  red = 0xFF0000,                      // rgb(255,0,0)
  rosy_brown = 0xBC8F8F,               // rgb(188,143,143)
  royal_blue = 0x4169E1,               // rgb(65,105,225)
  saddle_brown = 0x8B4513,             // rgb(139,69,19)
  salmon = 0xFA8072,                   // rgb(250,128,114)
  sandy_brown = 0xF4A460,              // rgb(244,164,96)
  sea_green = 0x2E8B57,                // rgb(46,139,87)
  sea_shell = 0xFFF5EE,                // rgb(255,245,238)
  sienna = 0xA0522D,                   // rgb(160,82,45)
  silver = 0xC0C0C0,                   // rgb(192,192,192)
  sky_blue = 0x87CEEB,                 // rgb(135,206,235)
  slate_blue = 0x6A5ACD,               // rgb(106,90,205)
  slate_gray = 0x708090,               // rgb(112,128,144)
  snow = 0xFFFAFA,                     // rgb(255,250,250)
  spring_green = 0x00FF7F,             // rgb(0,255,127)
  steel_blue = 0x4682B4,               // rgb(70,130,180)
  tan = 0xD2B48C,                      // rgb(210,180,140)
  teal = 0x008080,                     // rgb(0,128,128)
  thistle = 0xD8BFD8,                  // rgb(216,191,216)
  tomato = 0xFF6347,                   // rgb(255,99,71)
  turquoise = 0x40E0D0,                // rgb(64,224,208)
  violet = 0xEE82EE,                   // rgb(238,130,238)
  wheat = 0xF5DEB3,                    // rgb(245,222,179)
  white = 0xFFFFFF,                    // rgb(255,255,255)
  white_smoke = 0xF5F5F5,              // rgb(245,245,245)
  yellow = 0xFFFF00,                   // rgb(255,255,0)
  yellow_green = 0x9ACD32              // rgb(154,205,50)
};                                     // enum class color

enum class terminal_color : uint8_t {
  black = 30,
  red,
  green,
  yellow,
  blue,
  magenta,
  cyan,
  white,
  bright_black = 90,
  bright_red,
  bright_green,
  bright_yellow,
  bright_blue,
  bright_magenta,
  bright_cyan,
  bright_white
};

enum class emphasis : uint8_t {
  bold = 1,
  faint = 1 << 1,
  italic = 1 << 2,
  underline = 1 << 3,
  blink = 1 << 4,
  reverse = 1 << 5,
  conceal = 1 << 6,
  strikethrough = 1 << 7,
};

// rgb is a struct for red, green and blue colors.
// Using the name "rgb" makes some editors show the color in a tooltip.
struct rgb {
  FMT_CONSTEXPR rgb() : r(0), g(0), b(0) {}
  FMT_CONSTEXPR rgb(uint8_t r_, uint8_t g_, uint8_t b_) : r(r_), g(g_), b(b_) {}
  FMT_CONSTEXPR rgb(uint32_t hex)
      : r((hex >> 16) & 0xFF), g((hex >> 8) & 0xFF), b(hex & 0xFF) {}
  FMT_CONSTEXPR rgb(color hex)
      : r((uint32_t(hex) >> 16) & 0xFF),
        g((uint32_t(hex) >> 8) & 0xFF),
        b(uint32_t(hex) & 0xFF) {}
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

FMT_BEGIN_DETAIL_NAMESPACE

// color is a struct of either a rgb color or a terminal color.
struct color_type {
  FMT_CONSTEXPR color_type() FMT_NOEXCEPT : is_rgb(), value{} {}
  FMT_CONSTEXPR color_type(color rgb_color) FMT_NOEXCEPT : is_rgb(true),
                                                           value{} {
    value.rgb_color = static_cast<uint32_t>(rgb_color);
  }
  FMT_CONSTEXPR color_type(rgb rgb_color) FMT_NOEXCEPT : is_rgb(true), value{} {
    value.rgb_color = (static_cast<uint32_t>(rgb_color.r) << 16) |
                      (static_cast<uint32_t>(rgb_color.g) << 8) | rgb_color.b;
  }
  FMT_CONSTEXPR color_type(terminal_color term_color) FMT_NOEXCEPT : is_rgb(),
                                                                     value{} {
    value.term_color = static_cast<uint8_t>(term_color);
  }
  bool is_rgb;
  union color_union {
    uint8_t term_color;
    uint32_t rgb_color;
  } value;
};

FMT_END_DETAIL_NAMESPACE

/** A text style consisting of foreground and background colors and emphasis. */
class text_style {
 public:
  FMT_CONSTEXPR text_style(emphasis em = emphasis()) FMT_NOEXCEPT
      : set_foreground_color(),
        set_background_color(),
        ems(em) {}

  FMT_CONSTEXPR text_style& operator|=(const text_style& rhs) {
    if (!set_foreground_color) {
      set_foreground_color = rhs.set_foreground_color;
      foreground_color = rhs.foreground_color;
    } else if (rhs.set_foreground_color) {
      if (!foreground_color.is_rgb || !rhs.foreground_color.is_rgb)
        FMT_THROW(format_error("can't OR a terminal color"));
      foreground_color.value.rgb_color |= rhs.foreground_color.value.rgb_color;
    }

    if (!set_background_color) {
      set_background_color = rhs.set_background_color;
      background_color = rhs.background_color;
    } else if (rhs.set_background_color) {
      if (!background_color.is_rgb || !rhs.background_color.is_rgb)
        FMT_THROW(format_error("can't OR a terminal color"));
      background_color.value.rgb_color |= rhs.background_color.value.rgb_color;
    }

    ems = static_cast<emphasis>(static_cast<uint8_t>(ems) |
                                static_cast<uint8_t>(rhs.ems));
    return *this;
  }

  friend FMT_CONSTEXPR text_style operator|(text_style lhs,
                                            const text_style& rhs) {
    return lhs |= rhs;
  }

  FMT_DEPRECATED_NONMSVC FMT_CONSTEXPR text_style& operator&=(
      const text_style& rhs) {
    return and_assign(rhs);
  }

  FMT_DEPRECATED_NONMSVC friend FMT_CONSTEXPR text_style
  operator&(text_style lhs, const text_style& rhs) {
    return lhs.and_assign(rhs);
  }

  FMT_CONSTEXPR bool has_foreground() const FMT_NOEXCEPT {
    return set_foreground_color;
  }
  FMT_CONSTEXPR bool has_background() const FMT_NOEXCEPT {
    return set_background_color;
  }
  FMT_CONSTEXPR bool has_emphasis() const FMT_NOEXCEPT {
    return static_cast<uint8_t>(ems) != 0;
  }
  FMT_CONSTEXPR detail::color_type get_foreground() const FMT_NOEXCEPT {
    FMT_ASSERT(has_foreground(), "no foreground specified for this style");
    return foreground_color;
  }
  FMT_CONSTEXPR detail::color_type get_background() const FMT_NOEXCEPT {
    FMT_ASSERT(has_background(), "no background specified for this style");
    return background_color;
  }
  FMT_CONSTEXPR emphasis get_emphasis() const FMT_NOEXCEPT {
    FMT_ASSERT(has_emphasis(), "no emphasis specified for this style");
    return ems;
  }

 private:
  FMT_CONSTEXPR text_style(bool is_foreground,
                           detail::color_type text_color) FMT_NOEXCEPT
      : set_foreground_color(),
        set_background_color(),
        ems() {
    if (is_foreground) {
      foreground_color = text_color;
      set_foreground_color = true;
    } else {
      background_color = text_color;
      set_background_color = true;
    }
  }

  // DEPRECATED!
  FMT_CONSTEXPR text_style& and_assign(const text_style& rhs) {
    if (!set_foreground_color) {
      set_foreground_color = rhs.set_foreground_color;
      foreground_color = rhs.foreground_color;
    } else if (rhs.set_foreground_color) {
      if (!foreground_color.is_rgb || !rhs.foreground_color.is_rgb)
        FMT_THROW(format_error("can't AND a terminal color"));
      foreground_color.value.rgb_color &= rhs.foreground_color.value.rgb_color;
    }

    if (!set_background_color) {
      set_background_color = rhs.set_background_color;
      background_color = rhs.background_color;
    } else if (rhs.set_background_color) {
      if (!background_color.is_rgb || !rhs.background_color.is_rgb)
        FMT_THROW(format_error("can't AND a terminal color"));
      background_color.value.rgb_color &= rhs.background_color.value.rgb_color;
    }

    ems = static_cast<emphasis>(static_cast<uint8_t>(ems) &
                                static_cast<uint8_t>(rhs.ems));
    return *this;
  }

  friend FMT_CONSTEXPR_DECL text_style fg(detail::color_type foreground)
      FMT_NOEXCEPT;

  friend FMT_CONSTEXPR_DECL text_style bg(detail::color_type background)
      FMT_NOEXCEPT;

  detail::color_type foreground_color;
  detail::color_type background_color;
  bool set_foreground_color;
  bool set_background_color;
  emphasis ems;
};

/** Creates a text style from the foreground (text) color. */
FMT_CONSTEXPR inline text_style fg(detail::color_type foreground) FMT_NOEXCEPT {
  return text_style(true, foreground);
}

/** Creates a text style from the background color. */
FMT_CONSTEXPR inline text_style bg(detail::color_type background) FMT_NOEXCEPT {
  return text_style(false, background);
}

FMT_CONSTEXPR inline text_style operator|(emphasis lhs,
                                          emphasis rhs) FMT_NOEXCEPT {
  return text_style(lhs) | rhs;
}

FMT_BEGIN_DETAIL_NAMESPACE

template <typename Char> struct ansi_color_escape {
  FMT_CONSTEXPR ansi_color_escape(detail::color_type text_color,
                                  const char* esc) FMT_NOEXCEPT {
    // If we have a terminal color, we need to output another escape code
    // sequence.
    if (!text_color.is_rgb) {
      bool is_background = esc == string_view("\x1b[48;2;");
      uint32_t value = text_color.value.term_color;
      // Background ASCII codes are the same as the foreground ones but with
      // 10 more.
      if (is_background) value += 10u;

      size_t index = 0;
      buffer[index++] = static_cast<Char>('\x1b');
      buffer[index++] = static_cast<Char>('[');

      if (value >= 100u) {
        buffer[index++] = static_cast<Char>('1');
        value %= 100u;
      }
      buffer[index++] = static_cast<Char>('0' + value / 10u);
      buffer[index++] = static_cast<Char>('0' + value % 10u);

      buffer[index++] = static_cast<Char>('m');
      buffer[index++] = static_cast<Char>('\0');
      return;
    }

    for (int i = 0; i < 7; i++) {
      buffer[i] = static_cast<Char>(esc[i]);
    }
    rgb color(text_color.value.rgb_color);
    to_esc(color.r, buffer + 7, ';');
    to_esc(color.g, buffer + 11, ';');
    to_esc(color.b, buffer + 15, 'm');
    buffer[19] = static_cast<Char>(0);
  }
  FMT_CONSTEXPR ansi_color_escape(emphasis em) FMT_NOEXCEPT {
    uint8_t em_codes[num_emphases] = {};
    if (has_emphasis(em, emphasis::bold)) em_codes[0] = 1;
    if (has_emphasis(em, emphasis::faint)) em_codes[1] = 2;
    if (has_emphasis(em, emphasis::italic)) em_codes[2] = 3;
    if (has_emphasis(em, emphasis::underline)) em_codes[3] = 4;
    if (has_emphasis(em, emphasis::blink)) em_codes[4] = 5;
    if (has_emphasis(em, emphasis::reverse)) em_codes[5] = 7;
    if (has_emphasis(em, emphasis::conceal)) em_codes[6] = 8;
    if (has_emphasis(em, emphasis::strikethrough)) em_codes[7] = 9;

    size_t index = 0;
    for (size_t i = 0; i < num_emphases; ++i) {
      if (!em_codes[i]) continue;
      buffer[index++] = static_cast<Char>('\x1b');
      buffer[index++] = static_cast<Char>('[');
      buffer[index++] = static_cast<Char>('0' + em_codes[i]);
      buffer[index++] = static_cast<Char>('m');
    }
    buffer[index++] = static_cast<Char>(0);
  }
  FMT_CONSTEXPR operator const Char*() const FMT_NOEXCEPT { return buffer; }

  FMT_CONSTEXPR const Char* begin() const FMT_NOEXCEPT { return buffer; }
  FMT_CONSTEXPR_CHAR_TRAITS const Char* end() const FMT_NOEXCEPT {
    return buffer + std::char_traits<Char>::length(buffer);
  }

 private:
  static constexpr size_t num_emphases = 8;
  Char buffer[7u + 3u * num_emphases + 1u];

  static FMT_CONSTEXPR void to_esc(uint8_t c, Char* out,
                                   char delimiter) FMT_NOEXCEPT {
    out[0] = static_cast<Char>('0' + c / 100);
    out[1] = static_cast<Char>('0' + c / 10 % 10);
    out[2] = static_cast<Char>('0' + c % 10);
    out[3] = static_cast<Char>(delimiter);
  }
  static FMT_CONSTEXPR bool has_emphasis(emphasis em,
                                         emphasis mask) FMT_NOEXCEPT {
    return static_cast<uint8_t>(em) & static_cast<uint8_t>(mask);
  }
};

template <typename Char>
FMT_CONSTEXPR ansi_color_escape<Char> make_foreground_color(
    detail::color_type foreground) FMT_NOEXCEPT {
  return ansi_color_escape<Char>(foreground, "\x1b[38;2;");
}

template <typename Char>
FMT_CONSTEXPR ansi_color_escape<Char> make_background_color(
    detail::color_type background) FMT_NOEXCEPT {
  return ansi_color_escape<Char>(background, "\x1b[48;2;");
}

template <typename Char>
FMT_CONSTEXPR ansi_color_escape<Char> make_emphasis(emphasis em) FMT_NOEXCEPT {
  return ansi_color_escape<Char>(em);
}

template <typename Char>
inline void fputs(const Char* chars, FILE* stream) FMT_NOEXCEPT {
  std::fputs(chars, stream);
}

template <>
inline void fputs<wchar_t>(const wchar_t* chars, FILE* stream) FMT_NOEXCEPT {
  std::fputws(chars, stream);
}

template <typename Char> inline void reset_color(FILE* stream) FMT_NOEXCEPT {
  fputs("\x1b[0m", stream);
}

template <> inline void reset_color<wchar_t>(FILE* stream) FMT_NOEXCEPT {
  fputs(L"\x1b[0m", stream);
}

template <typename Char>
inline void reset_color(buffer<Char>& buffer) FMT_NOEXCEPT {
  auto reset_color = string_view("\x1b[0m");
  buffer.append(reset_color.begin(), reset_color.end());
}

template <typename Char>
void vformat_to(buffer<Char>& buf, const text_style& ts,
                basic_string_view<Char> format_str,
                basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  bool has_style = false;
  if (ts.has_emphasis()) {
    has_style = true;
    auto emphasis = detail::make_emphasis<Char>(ts.get_emphasis());
    buf.append(emphasis.begin(), emphasis.end());
  }
  if (ts.has_foreground()) {
    has_style = true;
    auto foreground = detail::make_foreground_color<Char>(ts.get_foreground());
    buf.append(foreground.begin(), foreground.end());
  }
  if (ts.has_background()) {
    has_style = true;
    auto background = detail::make_background_color<Char>(ts.get_background());
    buf.append(background.begin(), background.end());
  }
  detail::vformat_to(buf, format_str, args, {});
  if (has_style) detail::reset_color<Char>(buf);
}

FMT_END_DETAIL_NAMESPACE

template <typename S, typename Char = char_t<S>>
void vprint(std::FILE* f, const text_style& ts, const S& format,
            basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  basic_memory_buffer<Char> buf;
  detail::vformat_to(buf, ts, to_string_view(format), args);
  buf.push_back(Char(0));
  detail::fputs(buf.data(), f);
}

/**
  \rst
  Formats a string and prints it to the specified file stream using ANSI
  escape sequences to specify text formatting.

  **Example**::

    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
               "Elapsed time: {0:.2f} seconds", 1.23);
  \endrst
 */
template <typename S, typename... Args,
          FMT_ENABLE_IF(detail::is_string<S>::value)>
void print(std::FILE* f, const text_style& ts, const S& format_str,
           const Args&... args) {
  vprint(f, ts, format_str,
         fmt::make_args_checked<Args...>(format_str, args...));
}

/**
  \rst
  Formats a string and prints it to stdout using ANSI escape sequences to
  specify text formatting.

  **Example**::

    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
               "Elapsed time: {0:.2f} seconds", 1.23);
  \endrst
 */
template <typename S, typename... Args,
          FMT_ENABLE_IF(detail::is_string<S>::value)>
void print(const text_style& ts, const S& format_str, const Args&... args) {
  return print(stdout, ts, format_str, args...);
}

template <typename S, typename Char = char_t<S>>
inline std::basic_string<Char> vformat(
    const text_style& ts, const S& format_str,
    basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  basic_memory_buffer<Char> buf;
  detail::vformat_to(buf, ts, to_string_view(format_str), args);
  return fmt::to_string(buf);
}

/**
  \rst
  Formats arguments and returns the result as a string using ANSI
  escape sequences to specify text formatting.

  **Example**::

    #include <fmt/color.h>
    std::string message = fmt::format(fmt::emphasis::bold | fg(fmt::color::red),
                                      "The answer is {}", 42);
  \endrst
*/
template <typename S, typename... Args, typename Char = char_t<S>>
inline std::basic_string<Char> format(const text_style& ts, const S& format_str,
                                      const Args&... args) {
  return fmt::vformat(ts, to_string_view(format_str),
                      fmt::make_args_checked<Args...>(format_str, args...));
}

/**
  Formats a string with the given text_style and writes the output to ``out``.
 */
template <typename OutputIt, typename Char,
          FMT_ENABLE_IF(detail::is_output_iterator<OutputIt, Char>::value)>
OutputIt vformat_to(
    OutputIt out, const text_style& ts, basic_string_view<Char> format_str,
    basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  auto&& buf = detail::get_buffer<Char>(out);
  detail::vformat_to(buf, ts, format_str, args);
  return detail::get_iterator(buf);
}

/**
  \rst
  Formats arguments with the given text_style, writes the result to the output
  iterator ``out`` and returns the iterator past the end of the output range.

  **Example**::

    std::vector<char> out;
    fmt::format_to(std::back_inserter(out),
                   fmt::emphasis::bold | fg(fmt::color::red), "{}", 42);
  \endrst
*/
template <typename OutputIt, typename S, typename... Args,
          bool enable = detail::is_output_iterator<OutputIt, char_t<S>>::value&&
              detail::is_string<S>::value>
inline auto format_to(OutputIt out, const text_style& ts, const S& format_str,
                      Args&&... args) ->
    typename std::enable_if<enable, OutputIt>::type {
  return vformat_to(out, ts, to_string_view(format_str),
                    fmt::make_args_checked<Args...>(format_str, args...));
}

FMT_MODULE_EXPORT_END
FMT_END_NAMESPACE

#endif  // FMT_COLOR_H_
//
// Copyright (c) 2019-2021 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#if defined(__cpp_modules) && !defined(BOOST_UT_DISABLE_MODULE)
export module boost.ut;
export import std;
#else
#pragma once
#endif

#if __has_include(<iso646.h>)
#include <iso646.h>  // and, or, not, ...
#endif

#if not defined(__cpp_rvalue_references)
#error "[Boost::ext].UT requires support for rvalue references";
#elif not defined(__cpp_decltype)
#error "[Boost::ext].UT requires support for decltype";
#elif not defined(__cpp_return_type_deduction)
#error "[Boost::ext].UT requires support for return type deduction";
#elif not defined(__cpp_deduction_guides)
#error "[Boost::ext].UT requires support for return deduction guides";
#elif not defined(__cpp_generic_lambdas)
#error "[Boost::ext].UT requires support for generic lambdas";
#elif not defined(__cpp_constexpr)
#error "[Boost::ext].UT requires support for constexpr";
#elif not defined(__cpp_alias_templates)
#error "[Boost::ext].UT requires support for alias templates";
#elif not defined(__cpp_variadic_templates)
#error "[Boost::ext].UT requires support for variadic templates";
#elif not defined(__cpp_fold_expressions)
#error "[Boost::ext].UT requires support for return fold expressions";
#elif not defined(__cpp_static_assert)
#error "[Boost::ext].UT requires support for static assert";
#else
#define BOOST_UT_VERSION 1'1'9

#if defined(__has_builtin) and defined(__GNUC__) and (__GNUC__ < 10) and \
    not defined(__clang__)
#undef __has_builtin
#endif

#if not defined(__has_builtin)
#if defined(__GNUC__) and (__GNUC__ >= 9)
#define __has___builtin_FILE 1
#define __has___builtin_LINE 1
#endif
#define __has_builtin(...) __has_##__VA_ARGS__
#endif

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>
#if __has_include(<unistd.h>) and __has_include(<sys/wait.h>)
#include <sys/wait.h>
#include <unistd.h>
#endif
#if defined(__cpp_exceptions)
#include <exception>
#endif

#if __has_include(<source_location>)
#include <source_location>
#endif

#if defined(__cpp_modules) && !defined(BOOST_UT_DISABLE_MODULE)
export
#endif
    namespace boost::inline ext::ut::inline v1_1_9 {
namespace utility {
template <class>
class function;
template <class R, class... TArgs>
class function<R(TArgs...)> {
 public:
  constexpr function() = default;
  template <class T>
  constexpr /*explicit(false)*/ function(T data)
      : invoke_{invoke_impl<T>},
        destroy_{destroy_impl<T>},
        data_{new T{static_cast<T&&>(data)}} {}
  constexpr function(function&& other) noexcept
      : invoke_{static_cast<decltype(other.invoke_)&&>(other.invoke_)},
        destroy_{static_cast<decltype(other.destroy_)&&>(other.destroy_)},
        data_{static_cast<decltype(other.data_)&&>(other.data_)} {
    other.data_ = {};
  }
  constexpr function(const function&) = delete;
  ~function() { destroy_(data_); }

  constexpr function& operator=(const function&) = delete;
  constexpr function& operator=(function&&) = delete;
  [[nodiscard]] constexpr auto operator()(TArgs... args) -> R {
    return invoke_(data_, args...);
  }
  [[nodiscard]] constexpr auto operator()(TArgs... args) const -> R {
    return invoke_(data_, args...);
  }

 private:
  template <class T>
  [[nodiscard]] static auto invoke_impl(void* data, TArgs... args) -> R {
    return (*static_cast<T*>(data))(args...);
  }

  template <class T>
  static auto destroy_impl(void* data) -> void {
    delete static_cast<T*>(data);
  }

  R (*invoke_)(void*, TArgs...){};
  void (*destroy_)(void*){};
  void* data_{};
};

[[nodiscard]] inline auto is_match(std::string_view input,
                                   std::string_view pattern) -> bool {
  if (std::empty(pattern)) {
    return std::empty(input);
  }

  if (std::empty(input)) {
    return pattern[0] == '*' ? is_match(input, pattern.substr(1)) : false;
  }

  if (pattern[0] != '?' and pattern[0] != '*' and pattern[0] != input[0]) {
    return false;
  }

  if (pattern[0] == '*') {
    for (decltype(std::size(input)) i = 0u; i <= std::size(input); ++i) {
      if (is_match(input.substr(i), pattern.substr(1))) {
        return true;
      }
    }
    return false;
  }

  return is_match(input.substr(1), pattern.substr(1));
}

template <class TPattern, class TStr>
[[nodiscard]] constexpr auto match(const TPattern& pattern, const TStr& str)
    -> std::vector<TStr> {
  std::vector<TStr> groups{};
  auto pi = 0u;
  auto si = 0u;

  const auto matcher = [&](char b, char e, char c = 0) {
    const auto match = si;
    while (str[si] and str[si] != b and str[si] != c) {
      ++si;
    }
    groups.emplace_back(str.substr(match, si - match));
    while (pattern[pi] and pattern[pi] != e) {
      ++pi;
    }
    pi++;
  };

  while (pi < std::size(pattern) && si < std::size(str)) {
    if (pattern[pi] == '\'' and str[si] == '\'' and pattern[pi + 1] == '{') {
      ++si;
      matcher('\'', '}');
    } else if (pattern[pi] == '{') {
      matcher(' ', '}', ',');
    } else if (pattern[pi] != str[si]) {
      return {};
    }
    ++pi;
    ++si;
  }

  if (si < str.size() or pi < std::size(pattern)) {
    return {};
  }

  return groups;
}

template <class T = std::string_view, class TDelim>
[[nodiscard]] inline auto split(T input, TDelim delim) -> std::vector<T> {
  std::vector<T> output{};
  std::size_t first{};
  while (first < std::size(input)) {
    const auto second = input.find_first_of(delim, first);
    if (first != second) {
      output.emplace_back(input.substr(first, second - first));
    }
    if (second == T::npos) {
      break;
    }
    first = second + 1;
  }
  return output;
}
}  // namespace utility

namespace reflection {
#if defined(__cpp_lib_source_location)
using source_location = std::source_location;
#else
class source_location {
 public:
  [[nodiscard]] static constexpr auto current(
#if (__has_builtin(__builtin_FILE) and __has_builtin(__builtin_LINE))
      const char* file = __builtin_FILE(), int line = __builtin_LINE()
#else
      const char* file = "unknown", int line = {}
#endif
          ) noexcept {
    source_location sl{};
    sl.file_ = file;
    sl.line_ = line;
    return sl;
  }
  [[nodiscard]] constexpr auto file_name() const noexcept { return file_; }
  [[nodiscard]] constexpr auto line() const noexcept { return line_; }

 private:
  const char* file_{"unknown"};
  int line_{};
};
#endif

template <class T>
[[nodiscard]] constexpr auto type_name() -> std::string_view {
#if defined(_MSC_VER) and not defined(__clang__)
  return {&__FUNCSIG__[120], sizeof(__FUNCSIG__) - 128};
#elif defined(__clang_analyzer__)
  return {&__PRETTY_FUNCTION__[57], sizeof(__PRETTY_FUNCTION__) - 59};
#elif defined(__clang__) and (__clang_major__ >= 13) and defined(__APPLE__)
  return {&__PRETTY_FUNCTION__[57], sizeof(__PRETTY_FUNCTION__) - 59};
#elif defined(__clang__) and (__clang_major__ >= 12) and not defined(__APPLE__)
  return {&__PRETTY_FUNCTION__[57], sizeof(__PRETTY_FUNCTION__) - 59};
#elif defined(__clang__)
  return {&__PRETTY_FUNCTION__[70], sizeof(__PRETTY_FUNCTION__) - 72};
#elif defined(__GNUC__)
  return {&__PRETTY_FUNCTION__[85], sizeof(__PRETTY_FUNCTION__) - 136};
#endif
}
}  // namespace reflection

namespace math {
template <class T>
[[nodiscard]] constexpr auto abs(const T t) -> T {
  return t < T{} ? -t : t;
}

template <class T, class U>
[[nodiscard]] constexpr auto abs_diff(const T t, const U u)
    -> decltype(t < u ? u - t : t - u) {
  return t < u ? u - t : t - u;
}

template <class T>
[[nodiscard]] constexpr auto min_value(const T& lhs, const T& rhs) -> const T& {
  return (rhs < lhs) ? rhs : lhs;
}

template <class T, class TExp>
[[nodiscard]] constexpr auto pow(const T base, const TExp exp) -> T {
  return exp ? T(base * pow(base, exp - TExp(1))) : T(1);
}

template <class T, char... Cs>
[[nodiscard]] constexpr auto num() -> T {
  static_assert(
      ((Cs == '.' or Cs == '\'' or (Cs >= '0' and Cs <= '9')) and ...));
  T result{};
  for (const char c : std::array{Cs...}) {
    if (c == '.') {
      break;
    }
    if (c >= '0' and c <= '9') {
      result = result * T(10) + T(c - '0');
    }
  }
  return result;
}

template <class T, char... Cs>
[[nodiscard]] constexpr auto den() -> T {
  constexpr const std::array cs{Cs...};
  T result{};
  auto i = 0u;
  while (cs[i++] != '.') {
  }

  for (auto j = i; j < sizeof...(Cs); ++j) {
    result += pow(T(10), sizeof...(Cs) - j) * T(cs[j] - '0');
  }
  return result;
}

template <class T, char... Cs>
[[nodiscard]] constexpr auto den_size() -> T {
  constexpr const std::array cs{Cs...};
  T i{};
  while (cs[i++] != '.') {
  }

  return T(sizeof...(Cs)) - i + T(1);
}

template <class T, class TValue>
[[nodiscard]] constexpr auto den_size(TValue value) -> T {
  constexpr auto precision = TValue(1e-7);
  T result{};
  TValue tmp{};
  do {
    value *= 10;
    tmp = value - T(value);
    ++result;
  } while (tmp > precision);

  return result;
}

}  // namespace math

namespace type_traits {
template <class...>
struct list {};

template <class T, class...>
struct identity {
  using type = T;
};

template <class T>
struct function_traits : function_traits<decltype(&T::operator())> {};

template <class R, class... TArgs>
struct function_traits<R (*)(TArgs...)> {
  using result_type = R;
  using args = list<TArgs...>;
};

template <class R, class... TArgs>
struct function_traits<R(TArgs...)> {
  using result_type = R;
  using args = list<TArgs...>;
};

template <class R, class T, class... TArgs>
struct function_traits<R (T::*)(TArgs...)> {
  using result_type = R;
  using args = list<TArgs...>;
};

template <class R, class T, class... TArgs>
struct function_traits<R (T::*)(TArgs...) const> {
  using result_type = R;
  using args = list<TArgs...>;
};

template <class T>
T&& declval();
template <class... Ts, class TExpr>
constexpr auto is_valid(TExpr expr)
    -> decltype(expr(declval<Ts...>()), bool()) {
  return true;
}
template <class...>
constexpr auto is_valid(...) -> bool {
  return false;
}

template <class T>
static constexpr auto is_container_v =
    is_valid<T>([](auto t) -> decltype(t.begin(), t.end(), void()) {});

template <class T>
static constexpr auto has_user_print = is_valid<T>(
    [](auto t) -> decltype(void(declval<std::ostringstream&>() << t)) {});

template <class T, class = void>
struct has_static_member_object_value : std::false_type {};

template <class T>
struct has_static_member_object_value<T,
                                      std::void_t<decltype(declval<T>().value)>>
    : std::bool_constant<!std::is_member_pointer_v<decltype(&T::value)> &&
                         !std::is_function_v<decltype(T::value)>> {};

template <class T>
inline constexpr bool has_static_member_object_value_v =
    has_static_member_object_value<T>::value;

template <class T, class = void>
struct has_static_member_object_epsilon : std::false_type {};

template <class T>
struct has_static_member_object_epsilon<
    T, std::void_t<decltype(declval<T>().epsilon)>>
    : std::bool_constant<!std::is_member_pointer_v<decltype(&T::epsilon)> &&
                         !std::is_function_v<decltype(T::epsilon)>> {};

template <class T>
inline constexpr bool has_static_member_object_epsilon_v =
    has_static_member_object_epsilon<T>::value;

template <class T>
inline constexpr auto is_floating_point_v = false;
template <>
inline constexpr auto is_floating_point_v<float> = true;
template <>
inline constexpr auto is_floating_point_v<double> = true;
template <>
inline constexpr auto is_floating_point_v<long double> = true;

#if defined(__clang__) or defined(_MSC_VER)
template <class From, class To>
static constexpr auto is_convertible_v = __is_convertible_to(From, To);
#else
template <class From, class To>
constexpr auto is_convertible(int) -> decltype(bool(To(declval<From>()))) {
  return true;
}
template <class...>
constexpr auto is_convertible(...) {
  return false;
}
template <class From, class To>
constexpr auto is_convertible_v = is_convertible<From, To>(0);
#endif

template <bool>
struct requires_ {};
template <>
struct requires_<true> {
  using type = int;
};

template <bool Cond>
using requires_t = typename requires_<Cond>::type;
}  // namespace type_traits

struct none {};

namespace events {
struct test_begin {
  std::string_view type{};
  std::string_view name{};
  reflection::source_location location{};
};
template <class Test, class TArg = none>
struct test {
  std::string_view type{};
  std::string_view name{};
  std::vector<std::string_view> tag{};
  reflection::source_location location{};
  TArg arg{};
  Test run{};

  constexpr auto operator()() { run_impl(static_cast<Test&&>(run), arg); }
  constexpr auto operator()() const { run_impl(static_cast<Test&&>(run), arg); }

 private:
  static constexpr auto run_impl(Test test, const none&) { test(); }

  template <class T>
  static constexpr auto run_impl(T test, const TArg& arg)
      -> decltype(test(arg), void()) {
    test(arg);
  }

  template <class T>
  static constexpr auto run_impl(T test, const TArg&)
      -> decltype(test.template operator()<TArg>(), void()) {
    test.template operator()<TArg>();
  }
};
template <class Test, class TArg>
test(std::string_view, std::string_view, std::string_view,
     reflection::source_location, TArg, Test) -> test<Test, TArg>;
template <class TSuite>
struct suite {
  TSuite run{};
  constexpr auto operator()() { run(); }
  constexpr auto operator()() const { run(); }
};
template <class TSuite>
suite(TSuite) -> suite<TSuite>;
struct test_run {
  std::string_view type{};
  std::string_view name{};
};
template <class TArg = none>
struct skip {
  std::string_view type{};
  std::string_view name{};
  TArg arg{};
};
template <class TArg>
skip(std::string_view, std::string_view, TArg) -> skip<TArg>;
struct test_skip {
  std::string_view type{};
  std::string_view name{};
};
template <class TExpr>
struct assertion {
  TExpr expr{};
  reflection::source_location location{};
};
template <class TExpr>
assertion(TExpr, reflection::source_location) -> assertion<TExpr>;
template <class TExpr>
struct assertion_pass {
  TExpr expr{};
  reflection::source_location location{};
};
template <class TExpr>
assertion_pass(TExpr) -> assertion_pass<TExpr>;
template <class TExpr>
struct assertion_fail {
  TExpr expr{};
  reflection::source_location location{};
};
template <class TExpr>
assertion_fail(TExpr) -> assertion_fail<TExpr>;
struct test_end {
  std::string_view type{};
  std::string_view name{};
};
template <class TMsg>
struct log {
  TMsg msg{};
};
template <class TMsg = std::string_view>
log(TMsg) -> log<TMsg>;
struct fatal_assertion {};
struct exception {
  const char* msg{};
  [[nodiscard]] auto what() const -> const char* { return msg; }
};
struct summary {};
}  // namespace events

namespace detail {
struct op {};
struct fatal {};
struct cfg {
  static inline reflection::source_location location{};
  static inline bool wip{};
};

template <class T>
[[nodiscard]] constexpr auto get_impl(const T& t, int) -> decltype(t.get()) {
  return t.get();
}
template <class T>
[[nodiscard]] constexpr auto get_impl(const T& t, ...) -> decltype(auto) {
  return t;
}
template <class T>
[[nodiscard]] constexpr auto get(const T& t) {
  return get_impl(t, 0);
}

template <class T>
struct type_ : op {
  template <class TOther>
  [[nodiscard]] constexpr auto operator()(const TOther&) const
      -> const type_<TOther> {
    return {};
  }
  [[nodiscard]] constexpr auto operator==(type_<T>) -> bool { return true; }
  template <class TOther>
  [[nodiscard]] constexpr auto operator==(type_<TOther>) -> bool {
    return false;
  }
  template <class TOther>
  [[nodiscard]] constexpr auto operator==(const TOther&) -> bool {
    return std::is_same_v<TOther, T>;
  }
  [[nodiscard]] constexpr auto operator!=(type_<T>) -> bool { return false; }
  template <class TOther>
  [[nodiscard]] constexpr auto operator!=(type_<TOther>) -> bool {
    return true;
  }
  template <class TOther>
  [[nodiscard]] constexpr auto operator!=(const TOther&) -> bool {
    return not std::is_same_v<TOther, T>;
  }
};

template <class T, class = int>
struct value : op {
  using value_type = T;

  constexpr /*explicit(false)*/ value(const T& _value) : value_{_value} {}
  [[nodiscard]] constexpr explicit operator T() const { return value_; }
  [[nodiscard]] constexpr decltype(auto) get() const { return value_; }

  T value_{};
};

template <class T>
struct value<T, type_traits::requires_t<type_traits::is_floating_point_v<T>>>
    : op {
  using value_type = T;
  static inline auto epsilon = T{};

  constexpr value(const T& _value, const T precision) : value_{_value} {
    epsilon = precision;
  }

  constexpr /*explicit(false)*/ value(const T& val)
      : value{val, T(1) / math::pow(T(10),
                                    math::den_size<unsigned long long>(val))} {}
  [[nodiscard]] constexpr explicit operator T() const { return value_; }
  [[nodiscard]] constexpr decltype(auto) get() const { return value_; }

  T value_{};
};

template <class T>
class value_location : public detail::value<T> {
 public:
  constexpr /*explicit(false)*/ value_location(
      const T& t, const reflection::source_location& sl =
                      reflection::source_location::current())
      : detail::value<T>{t} {
    cfg::location = sl;
  }

  constexpr value_location(const T& t, const T precision,
                           const reflection::source_location& sl =
                               reflection::source_location::current())
      : detail::value<T>{t, precision} {
    cfg::location = sl;
  }
};

template <auto N>
struct integral_constant : op {
  using value_type = decltype(N);
  static constexpr auto value = N;

  [[nodiscard]] constexpr auto operator-() const {
    return integral_constant<-N>{};
  }
  [[nodiscard]] constexpr explicit operator value_type() const { return N; }
  [[nodiscard]] constexpr auto get() const { return N; }
};

template <class T, auto N, auto D, auto Size, auto P = 1>
struct floating_point_constant : op {
  using value_type = T;

  static constexpr auto epsilon = T(1) / math::pow(T(10), Size - 1);
  static constexpr auto value = T(P) * (T(N) + (T(D) / math::pow(T(10), Size)));

  [[nodiscard]] constexpr auto operator-() const {
    return floating_point_constant<T, N, D, Size, -1>{};
  }
  [[nodiscard]] constexpr explicit operator value_type() const { return value; }
  [[nodiscard]] constexpr auto get() const { return value; }
};

template <class TLhs, class TRhs>
struct eq_ : op {
  constexpr eq_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator==;
          using std::operator<;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value == TRhs::value;
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TLhs> and
                               type_traits::has_static_member_object_epsilon_v<
                                   TRhs>) {
            return math::abs(get(lhs) - get(rhs)) <
                   math::min_value(TLhs::epsilon, TRhs::epsilon);
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TLhs>) {
            return math::abs(get(lhs) - get(rhs)) < TLhs::epsilon;
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TRhs>) {
            return math::abs(get(lhs) - get(rhs)) < TRhs::epsilon;
          } else {
            return get(lhs) == get(rhs);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs, class TEpsilon>
struct approx_ : op {
  constexpr approx_(const TLhs& lhs = {}, const TRhs& rhs = {},
                    const TEpsilon& epsilon = {})
      : lhs_{lhs}, rhs_{rhs}, epsilon_{epsilon}, value_{[&] {
          using std::operator<;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs> and
                        type_traits::has_static_member_object_value_v<
                            TEpsilon>) {
            return math::abs_diff(TLhs::value, TRhs::value) < TEpsilon::value;
          } else {
            return math::abs_diff(get(lhs), get(rhs)) < get(epsilon);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }
  [[nodiscard]] constexpr auto epsilon() const { return get(epsilon_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const TEpsilon epsilon_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct neq_ : op {
  constexpr neq_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator==;
          using std::operator!=;
          using std::operator>;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value != TRhs::value;
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TLhs> and
                               type_traits::has_static_member_object_epsilon_v<
                                   TRhs>) {
            return math::abs(get(lhs_) - get(rhs_)) >
                   math::min_value(TLhs::epsilon, TRhs::epsilon);
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TLhs>) {
            return math::abs(get(lhs_) - get(rhs_)) > TLhs::epsilon;
          } else if constexpr (type_traits::has_static_member_object_epsilon_v<
                                   TRhs>) {
            return math::abs(get(lhs_) - get(rhs_)) > TRhs::epsilon;
          } else {
            return get(lhs_) != get(rhs_);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct gt_ : op {
  constexpr gt_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator>;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value > TRhs::value;
          } else {
            return get(lhs_) > get(rhs_);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct ge_ : op {
  constexpr ge_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator>=;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value >= TRhs::value;
          } else {
            return get(lhs_) >= get(rhs_);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct lt_ : op {
  constexpr lt_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator<;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value < TRhs::value;
          } else {
            return get(lhs_) < get(rhs_);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

 private:
  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct le_ : op {
  constexpr le_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs}, rhs_{rhs}, value_{[&] {
          using std::operator<=;

          if constexpr (type_traits::has_static_member_object_value_v<TLhs> and
                        type_traits::has_static_member_object_value_v<TRhs>) {
            return TLhs::value <= TRhs::value;
          } else {
            return get(lhs_) <= get(rhs_);
          }
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct and_ : op {
  constexpr and_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs},
        rhs_{rhs},
        value_{static_cast<bool>(lhs) and static_cast<bool>(rhs)} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class TLhs, class TRhs>
struct or_ : op {
  constexpr or_(const TLhs& lhs = {}, const TRhs& rhs = {})
      : lhs_{lhs},
        rhs_{rhs},
        value_{static_cast<bool>(lhs) or static_cast<bool>(rhs)} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto lhs() const { return get(lhs_); }
  [[nodiscard]] constexpr auto rhs() const { return get(rhs_); }

  const TLhs lhs_{};
  const TRhs rhs_{};
  const bool value_{};
};

template <class T>
struct not_ : op {
  explicit constexpr not_(const T& t = {})
      : t_{t}, value_{not static_cast<bool>(t)} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }
  [[nodiscard]] constexpr auto value() const { return get(t_); }

  const T t_{};
  const bool value_{};
};

template <class>
struct fatal_;

#if defined(__cpp_exceptions)
template <class TExpr, class TException = void>
struct throws_ : op {
  constexpr explicit throws_(const TExpr& expr)
      : value_{[&expr] {
          try {
            expr();
          } catch (const TException&) {
            return true;
          } catch (...) {
            return false;
          }
          return false;
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }

  const bool value_{};
};

template <class TExpr>
struct throws_<TExpr, void> : op {
  constexpr explicit throws_(const TExpr& expr)
      : value_{[&expr] {
          try {
            expr();
          } catch (...) {
            return true;
          }
          return false;
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }

  const bool value_{};
};

template <class TExpr>
struct nothrow_ : op {
  constexpr explicit nothrow_(const TExpr& expr)
      : value_{[&expr] {
          try {
            expr();
          } catch (...) {
            return false;
          }
          return true;
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }

  const bool value_{};
};
#endif

#if __has_include(<unistd.h>) and __has_include(<sys/wait.h>)
template <class TExpr>
struct aborts_ : op {
  constexpr explicit aborts_(const TExpr& expr)
      : value_{[&expr]() -> bool {
          if (const auto pid = fork(); not pid) {
            expr();
            exit(0);
          }
          auto exit_status = 0;
          wait(&exit_status);
          return exit_status;
        }()} {}

  [[nodiscard]] constexpr operator bool() const { return value_; }

  const bool value_{};
};
#endif
}  // namespace detail

namespace type_traits {
template <class T>
inline constexpr auto is_op_v = __is_base_of(detail::op, T);
}  // namespace type_traits

struct colors {
  std::string_view none = "\033[0m";
  std::string_view pass = "\033[32m";
  std::string_view fail = "\033[31m";
};

class printer {
  [[nodiscard]] inline auto color(const bool cond) {
    return cond ? colors_.pass : colors_.fail;
  }

 public:
  printer() = default;
  /*explicit(false)*/ printer(const colors colors) : colors_{colors} {}

  template <class T>
  auto& operator<<(const T& t) {
    out_ << detail::get(t);
    return *this;
  }

  template <class T,
            type_traits::requires_t<not type_traits::has_user_print<T> and
                                    type_traits::is_container_v<T>> = 0>
  auto& operator<<(T&& t) {
    *this << '{';
    auto first = true;
    for (const auto& arg : t) {
      *this << (first ? "" : ", ") << arg;
      first = false;
    }
    *this << '}';
    return *this;
  }

  auto& operator<<(std::string_view sv) {
    out_ << sv;
    return *this;
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::eq_<TLhs, TRhs>& op) {
    return (*this << color(op) << op.lhs() << " == " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs, class TEpsilon>
  auto& operator<<(const detail::approx_<TLhs, TRhs, TEpsilon>& op) {
    return (*this << color(op) << op.lhs() << " ~ (" << op.rhs() << " +/- "
                  << op.epsilon() << ')' << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::neq_<TLhs, TRhs>& op) {
    return (*this << color(op) << op.lhs() << " != " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::gt_<TLhs, TRhs>& op) {
    return (*this << color(op) << op.lhs() << " > " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::ge_<TLhs, TRhs>& op) {
    return (*this << color(op) << op.lhs() << " >= " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::lt_<TRhs, TLhs>& op) {
    return (*this << color(op) << op.lhs() << " < " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::le_<TRhs, TLhs>& op) {
    return (*this << color(op) << op.lhs() << " <= " << op.rhs()
                  << colors_.none);
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::and_<TLhs, TRhs>& op) {
    return (*this << '(' << op.lhs() << color(op) << " and " << colors_.none
                  << op.rhs() << ')');
  }

  template <class TLhs, class TRhs>
  auto& operator<<(const detail::or_<TLhs, TRhs>& op) {
    return (*this << '(' << op.lhs() << color(op) << " or " << colors_.none
                  << op.rhs() << ')');
  }

  template <class T>
  auto& operator<<(const detail::not_<T>& op) {
    return (*this << color(op) << "not " << op.value() << colors_.none);
  }

  template <class T>
  auto& operator<<(const detail::fatal_<T>& fatal) {
    return (*this << fatal.get());
  }

#if defined(__cpp_exceptions)
  template <class TExpr, class TException>
  auto& operator<<(const detail::throws_<TExpr, TException>& op) {
    return (*this << color(op) << "throws<"
                  << reflection::type_name<TException>() << ">"
                  << colors_.none);
  }

  template <class TExpr>
  auto& operator<<(const detail::throws_<TExpr, void>& op) {
    return (*this << color(op) << "throws" << colors_.none);
  }

  template <class TExpr>
  auto& operator<<(const detail::nothrow_<TExpr>& op) {
    return (*this << color(op) << "nothrow" << colors_.none);
  }
#endif

#if __has_include(<unistd.h>) and __has_include(<sys/wait.h>)
  template <class TExpr>
  auto& operator<<(const detail::aborts_<TExpr>& op) {
    return (*this << color(op) << "aborts" << colors_.none);
  }
#endif

  template <class T>
  auto& operator<<(const detail::type_<T>&) {
    return (*this << reflection::type_name<T>());
  }

  auto str() const { return out_.str(); }
  const auto& colors() const { return colors_; }

 private:
  ut::colors colors_{};
  std::ostringstream out_{};
};

template <class TPrinter = printer>
class reporter {
 public:
  constexpr auto operator=(TPrinter printer) {
    printer_ = static_cast<TPrinter&&>(printer);
  }

  auto on(events::test_begin test_begin) -> void {
    printer_ << "Running \"" << test_begin.name << "\"...";
    fails_ = asserts_.fail;
  }

  auto on(events::test_run test_run) -> void {
    printer_ << "\n \"" << test_run.name << "\"...";
  }

  auto on(events::test_skip test_skip) -> void {
    printer_ << test_skip.name << "...SKIPPED\n";
    ++tests_.skip;
  }

  auto on(events::test_end) -> void {
    if (asserts_.fail > fails_) {
      ++tests_.fail;
      printer_ << '\n'
               << printer_.colors().fail << "FAILED" << printer_.colors().none
               << '\n';
    } else {
      ++tests_.pass;
      printer_ << printer_.colors().pass << "PASSED" << printer_.colors().none
               << '\n';
    }
  }

  template <class TMsg>
  auto on(events::log<TMsg> l) -> void {
    printer_ << l.msg;
  }

  auto on(events::exception exception) -> void {
    printer_ << "\n  " << printer_.colors().fail
             << "Unexpected exception with message:\n"
             << exception.what() << printer_.colors().none;
    ++asserts_.fail;
  }

  template <class TExpr>
  auto on(events::assertion_pass<TExpr>) -> void {
    ++asserts_.pass;
  }

  template <class TExpr>
  auto on(events::assertion_fail<TExpr> assertion) -> void {
    constexpr auto short_name = [](std::string_view name) {
      return name.rfind('/') != std::string_view::npos
                 ? name.substr(name.rfind('/') + 1)
                 : name;
    };
    printer_ << "\n  " << short_name(assertion.location.file_name()) << ':'
             << assertion.location.line() << ':' << printer_.colors().fail
             << "FAILED" << printer_.colors().none << " [" << std::boolalpha
             << assertion.expr << printer_.colors().none << ']';
    ++asserts_.fail;
  }

  auto on(events::fatal_assertion) -> void {}

  auto on(events::summary) -> void {
    if (tests_.fail or asserts_.fail) {
      printer_ << "\n========================================================"
                  "=======================\n"
               << "tests:   " << (tests_.pass + tests_.fail) << " | "
               << printer_.colors().fail << tests_.fail << " failed"
               << printer_.colors().none << '\n'
               << "asserts: " << (asserts_.pass + asserts_.fail) << " | "
               << asserts_.pass << " passed"
               << " | " << printer_.colors().fail << asserts_.fail << " failed"
               << printer_.colors().none << '\n';
      std::cerr << printer_.str() << std::endl;
    } else {
      std::cout << printer_.colors().pass << "All tests passed"
                << printer_.colors().none << " (" << asserts_.pass
                << " asserts in " << tests_.pass << " tests)\n";

      if (tests_.skip) {
        std::cout << tests_.skip << " tests skipped\n";
      }

      std::cout.flush();
    }
  }

 protected:
  struct {
    std::size_t pass{};
    std::size_t fail{};
    std::size_t skip{};
  } tests_{};

  struct {
    std::size_t pass{};
    std::size_t fail{};
  } asserts_{};

  std::size_t fails_{};

  TPrinter printer_{};
};

struct options {
  std::string_view filter{};
  std::vector<std::string_view> tag{};
  ut::colors colors{};
  bool dry_run{};
};

struct run_cfg {
  bool report_errors{false};
};

template <class TReporter = reporter<printer>, auto MaxPathSize = 16>
class runner {
  class filter {
    static constexpr auto delim = ".";

   public:
    constexpr /*explicit(false)*/ filter(std::string_view _filter = {})
        : path_{utility::split(_filter, delim)} {}

    template <class TPath>
    constexpr auto operator()(const std::size_t level, const TPath& path) const
        -> bool {
      for (auto i = 0u; i < math::min_value(level + 1, std::size(path_)); ++i) {
        if (not utility::is_match(path[i], path_[i])) {
          return false;
        }
      }
      return true;
    }

   private:
    std::vector<std::string_view> path_{};
  };

 public:
  constexpr runner() = default;
  constexpr runner(TReporter reporter, std::size_t suites_size)
      : reporter_{std::move(reporter)}, suites_(suites_size) {}

  ~runner() {
    const auto should_run = not run_;

    if (should_run) {
      static_cast<void>(run());
    }

    if (not dry_run_) {
      report_summary();
    }

    if (should_run and fails_) {
      std::exit(-1);
    }
  }

  auto operator=(const options& options) {
    filter_ = options.filter;
    tag_ = options.tag;
    dry_run_ = options.dry_run;
    reporter_ = {options.colors};
  }

  template <class TSuite>
  auto on(events::suite<TSuite> suite) {
    suites_.push_back(suite.run);
  }

  template <class... Ts>
  auto on(events::test<Ts...> test) {
    path_[level_] = test.name;

    auto execute = std::empty(test.tag);
    for (const auto& tag_element : test.tag) {
      if (utility::is_match(tag_element, "skip")) {
        on(events::skip<>{.type = test.type, .name = test.name});
        return;
      }

      for (const auto& ftag : tag_) {
        if (utility::is_match(tag_element, ftag)) {
          execute = true;
          break;
        }
      }
    }

    if (not execute) {
      on(events::skip<>{.type = test.type, .name = test.name});
      return;
    }

    if (filter_(level_, path_)) {
      if (not level_++) {
        reporter_.on(events::test_begin{
            .type = test.type, .name = test.name, .location = test.location});
      } else {
        reporter_.on(events::test_run{.type = test.type, .name = test.name});
      }

      if (dry_run_) {
        for (auto i = 0u; i < level_; ++i) {
          std::cout << (i ? "." : "") << path_[i];
        }
        std::cout << '\n';
      }

#if defined(__cpp_exceptions)
      try {
#endif
        test();
#if defined(__cpp_exceptions)
      } catch (const events::fatal_assertion&) {
      } catch (const std::exception& exception) {
        ++fails_;
        reporter_.on(events::exception{exception.what()});
      } catch (...) {
        ++fails_;
        reporter_.on(events::exception{"Unknown exception"});
      }
#endif

      if (not --level_) {
        reporter_.on(events::test_end{.type = test.type, .name = test.name});
      }
    }
  }

  template <class... Ts>
  auto on(events::skip<Ts...> test) {
    reporter_.on(events::test_skip{.type = test.type, .name = test.name});
  }

  template <class TExpr>
  [[nodiscard]] auto on(events::assertion<TExpr> assertion) -> bool {
    if (dry_run_) {
      return true;
    }

    if (static_cast<bool>(assertion.expr)) {
      reporter_.on(events::assertion_pass<TExpr>{
          .expr = assertion.expr, .location = assertion.location});
      return true;
    }

    ++fails_;
    reporter_.on(events::assertion_fail<TExpr>{.expr = assertion.expr,
                                               .location = assertion.location});
    return false;
  }

  auto on(events::fatal_assertion fatal_assertion) {
    reporter_.on(fatal_assertion);

#if defined(__cpp_exceptions)
    if (not level_) {
      report_summary();
    }
    throw fatal_assertion;
#else
    if (level_) {
      reporter_.on(events::test_end{});
    }
    report_summary();
    std::abort();
#endif
  }

  template <class TMsg>
  auto on(events::log<TMsg> l) {
    reporter_.on(l);
  }

  [[nodiscard]] auto run(run_cfg rc = {}) -> bool {
    run_ = true;
    for (const auto& suite : suites_) {
      suite();
    }
    suites_.clear();

    if (rc.report_errors) {
      report_summary();
    }

    return fails_ > 0;
  }

  auto report_summary() -> void {
    if (static auto once = true; once) {
      once = false;
      reporter_.on(events::summary{});
    }
  }

 protected:
  TReporter reporter_{};
  std::vector<void (*)()> suites_{};
  std::size_t level_{};
  bool run_{};
  std::size_t fails_{};
  std::array<std::string_view, MaxPathSize> path_{};
  filter filter_{};
  std::vector<std::string_view> tag_{};
  bool dry_run_{};
};

struct override {};

template <class = override, class...>
[[maybe_unused]] inline auto cfg = runner<reporter<printer>>{};

namespace detail {
struct tag {
  std::vector<std::string_view> name{};
};

template <class... Ts, class TEvent>
[[nodiscard]] constexpr decltype(auto) on(TEvent&& event) {
  return ut::cfg<typename type_traits::identity<override, Ts...>::type>.on(
      static_cast<TEvent&&>(event));
}

template <class Test>
struct test_location {
  template <class T>
  constexpr test_location(const T& t,
                          const reflection::source_location& sl =
                              reflection::source_location::current())
      : test{t}, location{sl} {}

  Test test{};
  reflection::source_location location{};
};

struct test {
  std::string_view type{};
  std::string_view name{};
  std::vector<std::string_view> tag{};

  template <class... Ts>
  constexpr auto operator=(test_location<void (*)()> _test) {
    on<Ts...>(events::test<void (*)()>{.type = type,
                                       .name = name,
                                       .tag = tag,
                                       .location = _test.location,
                                       .arg = none{},
                                       .run = _test.test});
    return _test.test;
  }

  template <class Test,
            type_traits::requires_t<
                not type_traits::is_convertible_v<Test, void (*)()>> = 0>
  constexpr auto operator=(Test _test) ->
      typename type_traits::identity<Test, decltype(_test())>::type {
    on<Test>(events::test<Test>{.type = type,
                                .name = name,
                                .tag = tag,
                                .location = {},
                                .arg = none{},
                                .run = static_cast<Test&&>(_test)});
    return _test;
  }

  constexpr auto operator=(void (*_test)(std::string_view)) const {
    return _test(name);
  }

  template <class Test,
            type_traits::requires_t<not type_traits::is_convertible_v<
                Test, void (*)(std::string_view)>> = 0>
  constexpr auto operator=(Test _test)
      -> decltype(_test(type_traits::declval<std::string_view>())) {
    return _test(name);
  }
};

struct log {
  struct next {
    template <class TMsg>
    auto& operator<<(const TMsg& msg) {
      on<TMsg>(events::log{' '});
      on<TMsg>(events::log{msg});
      return *this;
    }
  };

  template <class TMsg>
  auto operator<<(const TMsg& msg) -> next {
    on<TMsg>(events::log{'\n'});
    on<TMsg>(events::log{msg});
    return next{};
  }
};

template <class TExpr>
class terse_ {
 public:
  constexpr explicit terse_(const TExpr& expr) : expr_{expr} { cfg::wip = {}; }

  ~terse_() noexcept(false) {
    if (static auto once = true; once and not cfg::wip) {
      once = {};
    } else {
      return;
    }

    cfg::wip = true;

    void(detail::on<TExpr>(
        events::assertion<TExpr>{.expr = expr_, .location = cfg::location}));
  }

 private:
  const TExpr& expr_;
};

struct that_ {
  template <class T>
  struct expr {
    using type = expr;

    constexpr explicit expr(const T& t) : t_{t} {}

    [[nodiscard]] constexpr auto operator!() const { return not_{*this}; }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator==(const TRhs& rhs) const {
      return eq_{t_, rhs};
    }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator!=(const TRhs& rhs) const {
      return neq_{t_, rhs};
    }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator>(const TRhs& rhs) const {
      return gt_{t_, rhs};
    }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator>=(const TRhs& rhs) const {
      return ge_{t_, rhs};
    }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator<(const TRhs& rhs) const {
      return lt_{t_, rhs};
    }

    template <class TRhs>
    [[nodiscard]] constexpr auto operator<=(const TRhs& rhs) const {
      return le_{t_, rhs};
    }

    [[nodiscard]] constexpr operator bool() const {
      return static_cast<bool>(t_);
    }

    const T t_{};
  };

  template <class T>
  [[nodiscard]] constexpr auto operator%(const T& t) const {
    return expr{t};
  }
};

template <class TExpr>
struct fatal_ : op {
  using type = fatal_;

  constexpr explicit fatal_(const TExpr& expr) : expr_{expr} {}

  [[nodiscard]] constexpr operator bool() const {
    if (static_cast<bool>(expr_)) {
    } else {
      cfg::wip = true;
      void(on<TExpr>(
          events::assertion<TExpr>{.expr = expr_, .location = cfg::location}));
      on<TExpr>(events::fatal_assertion{});
    }
    return static_cast<bool>(expr_);
  }

  [[nodiscard]] constexpr decltype(auto) get() const { return expr_; }

  TExpr expr_{};
};

template <class T>
struct expect_ {
  constexpr explicit expect_(bool value) : value_{value} { cfg::wip = {}; }

  template <class TMsg>
  auto& operator<<(const TMsg& msg) {
    if (not value_) {
      on<T>(events::log{' '});
      on<T>(events::log{msg});
    }
    return *this;
  }

  [[nodiscard]] constexpr operator bool() const { return value_; }

  bool value_{};
};
}  // namespace detail

namespace literals {
[[nodiscard]] inline auto operator""_test(const char* name,
                                          decltype(sizeof("")) size) {
  return detail::test{"test", std::string_view{name, size}};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_i() {
  return detail::integral_constant<math::num<int, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_s() {
  return detail::integral_constant<math::num<short, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_c() {
  return detail::integral_constant<math::num<char, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_sc() {
  return detail::integral_constant<math::num<signed char, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_l() {
  return detail::integral_constant<math::num<long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_ll() {
  return detail::integral_constant<math::num<long long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_u() {
  return detail::integral_constant<math::num<unsigned, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_uc() {
  return detail::integral_constant<math::num<unsigned char, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_us() {
  return detail::integral_constant<math::num<unsigned short, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_ul() {
  return detail::integral_constant<math::num<unsigned long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_ull() {
  return detail::integral_constant<math::num<unsigned long long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_i8() {
  return detail::integral_constant<math::num<std::int8_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_i16() {
  return detail::integral_constant<math::num<std::int16_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_i32() {
  return detail::integral_constant<math::num<std::int32_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_i64() {
  return detail::integral_constant<math::num<std::int64_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_u8() {
  return detail::integral_constant<math::num<std::uint8_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_u16() {
  return detail::integral_constant<math::num<std::uint16_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_u32() {
  return detail::integral_constant<math::num<std::uint32_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_u64() {
  return detail::integral_constant<math::num<std::uint64_t, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_f() {
  return detail::floating_point_constant<
      float, math::num<unsigned long, Cs...>(),
      math::den<unsigned long, Cs...>(),
      math::den_size<unsigned long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_d() {
  return detail::floating_point_constant<
      double, math::num<unsigned long, Cs...>(),
      math::den<unsigned long, Cs...>(),
      math::den_size<unsigned long, Cs...>()>{};
}

template <char... Cs>
[[nodiscard]] constexpr auto operator""_ld() {
  return detail::floating_point_constant<
      long double, math::num<unsigned long long, Cs...>(),
      math::den<unsigned long long, Cs...>(),
      math::den_size<unsigned long long, Cs...>()>{};
}

constexpr auto operator""_b(const char* name, decltype(sizeof("")) size) {
  struct named : std::string_view, detail::op {
    using value_type = bool;
    [[nodiscard]] constexpr operator value_type() const { return true; }
    [[nodiscard]] constexpr auto operator==(const named&) const { return true; }
    [[nodiscard]] constexpr auto operator==(const bool other) const {
      return other;
    }
  };
  return named{{name, size}, {}};
}
}  // namespace literals

namespace operators {
[[nodiscard]] constexpr auto operator==(std::string_view lhs,
                                        std::string_view rhs) {
  return detail::eq_{lhs, rhs};
}

[[nodiscard]] constexpr auto operator!=(std::string_view lhs,
                                        std::string_view rhs) {
  return detail::neq_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_container_v<T>> = 0>
[[nodiscard]] constexpr auto operator==(T&& lhs, T&& rhs) {
  return detail::eq_{static_cast<T&&>(lhs), static_cast<T&&>(rhs)};
}

template <class T, type_traits::requires_t<type_traits::is_container_v<T>> = 0>
[[nodiscard]] constexpr auto operator!=(T&& lhs, T&& rhs) {
  return detail::neq_{static_cast<T&&>(lhs), static_cast<T&&>(rhs)};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator==(const TLhs& lhs, const TRhs& rhs) {
  return detail::eq_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator!=(const TLhs& lhs, const TRhs& rhs) {
  return detail::neq_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator>(const TLhs& lhs, const TRhs& rhs) {
  return detail::gt_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator>=(const TLhs& lhs, const TRhs& rhs) {
  return detail::ge_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator<(const TLhs& lhs, const TRhs& rhs) {
  return detail::lt_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator<=(const TLhs& lhs, const TRhs& rhs) {
  return detail::le_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator and(const TLhs& lhs, const TRhs& rhs) {
  return detail::and_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
[[nodiscard]] constexpr auto operator or(const TLhs& lhs, const TRhs& rhs) {
  return detail::or_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
[[nodiscard]] constexpr auto operator not(const T& t) {
  return detail::not_{t};
}

template <class T>
[[nodiscard]] inline auto operator>>(
    const T& t, const detail::value_location<detail::fatal>&) {
  return detail::fatal_{t};
}

template <class Test>
[[nodiscard]] auto operator/(const detail::tag& tag, Test test) {
  for (const auto& name : tag.name) {
    test.tag.push_back(name);
  }
  return test;
}

[[nodiscard]] inline auto operator/(const detail::tag& lhs,
                                    const detail::tag& rhs) {
  std::vector<std::string_view> tag{};
  for (const auto& name : lhs.name) {
    tag.push_back(name);
  }
  for (const auto& name : rhs.name) {
    tag.push_back(name);
  }
  return detail::tag{tag};
}

template <class F, class T,
          type_traits::requires_t<type_traits::is_container_v<T>> = 0>
[[nodiscard]] constexpr auto operator|(const F& f, const T& t) {
  return [f, t](const auto name) {
    for (const auto& arg : t) {
      detail::on<F>(events::test<F, typename T::value_type>{.type = "test",
                                                            .name = name,
                                                            .tag = {},
                                                            .location = {},
                                                            .arg = arg,
                                                            .run = f});
    }
  };
}

template <
    class F, template <class...> class T, class... Ts,
    type_traits::requires_t<not type_traits::is_container_v<T<Ts...>>> = 0>
[[nodiscard]] constexpr auto operator|(const F& f, const T<Ts...>& t) {
  return [f, t](const auto name) {
    apply(
        [f, name](const auto&... args) {
          (detail::on<F>(events::test<F, Ts>{.type = "test",
                                             .name = name,
                                             .tag = {},
                                             .location = {},
                                             .arg = args,
                                             .run = f}),
           ...);
        },
        t);
  };
}

namespace terse {
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-comparison"
#endif

[[maybe_unused]] constexpr struct {
} _t;

template <class T>
constexpr auto operator%(const T& t, const decltype(_t)&) {
  return detail::value<T>{t};
}

template <class T>
inline auto operator>>(const T& t,
                       const detail::value_location<detail::fatal>&) {
  using fatal_t = detail::fatal_<T>;
  struct fatal_ : fatal_t, detail::log {
    using type = fatal_t;
    using fatal_t::fatal_t;
    const detail::terse_<type> _{*this};
  };
  return fatal_{t};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator==(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using eq_t = detail::eq_<T, detail::value_location<typename T::value_type>>;
  struct eq_ : eq_t, detail::log {
    using type = eq_t;
    using eq_t::eq_t;
    const detail::terse_<type> _{*this};
  };
  return eq_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator==(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using eq_t = detail::eq_<detail::value_location<typename T::value_type>, T>;
  struct eq_ : eq_t, detail::log {
    using type = eq_t;
    using eq_t::eq_t;
    const detail::terse_<type> _{*this};
  };
  return eq_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator!=(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using neq_t = detail::neq_<T, detail::value_location<typename T::value_type>>;
  struct neq_ : neq_t, detail::log {
    using type = neq_t;
    using neq_t::neq_t;
    const detail::terse_<type> _{*this};
  };
  return neq_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator!=(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using neq_t = detail::neq_<detail::value_location<typename T::value_type>, T>;
  struct neq_ : neq_t {
    using type = neq_t;
    using neq_t::neq_t;
    const detail::terse_<type> _{*this};
  };
  return neq_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator>(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using gt_t = detail::gt_<T, detail::value_location<typename T::value_type>>;
  struct gt_ : gt_t, detail::log {
    using type = gt_t;
    using gt_t::gt_t;
    const detail::terse_<type> _{*this};
  };
  return gt_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator>(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using gt_t = detail::gt_<detail::value_location<typename T::value_type>, T>;
  struct gt_ : gt_t, detail::log {
    using type = gt_t;
    using gt_t::gt_t;
    const detail::terse_<type> _{*this};
  };
  return gt_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator>=(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using ge_t = detail::ge_<T, detail::value_location<typename T::value_type>>;
  struct ge_ : ge_t, detail::log {
    using type = ge_t;
    using ge_t::ge_t;
    const detail::terse_<type> _{*this};
  };
  return ge_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator>=(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using ge_t = detail::ge_<detail::value_location<typename T::value_type>, T>;
  struct ge_ : ge_t, detail::log {
    using type = ge_t;
    using ge_t::ge_t;
    const detail::terse_<type> _{*this};
  };
  return ge_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator<(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using lt_t = detail::lt_<T, detail::value_location<typename T::value_type>>;
  struct lt_ : lt_t, detail::log {
    using type = lt_t;
    using lt_t::lt_t;
    const detail::terse_<type> _{*this};
  };
  return lt_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator<(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using lt_t = detail::lt_<detail::value_location<typename T::value_type>, T>;
  struct lt_ : lt_t, detail::log {
    using type = lt_t;
    using lt_t::lt_t;
    const detail::terse_<type> _{*this};
  };
  return lt_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator<=(
    const T& lhs, const detail::value_location<typename T::value_type>& rhs) {
  using le_t = detail::le_<T, detail::value_location<typename T::value_type>>;
  struct le_ : le_t, detail::log {
    using type = le_t;
    using le_t::le_t;
    const detail::terse_<type> _{*this};
  };
  return le_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator<=(
    const detail::value_location<typename T::value_type>& lhs, const T& rhs) {
  using le_t = detail::le_<detail::value_location<typename T::value_type>, T>;
  struct le_ : le_t {
    using type = le_t;
    using le_t::le_t;
    const detail::terse_<type> _{*this};
  };
  return le_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
constexpr auto operator and(const TLhs& lhs, const TRhs& rhs) {
  using and_t = detail::and_<typename TLhs::type, typename TRhs::type>;
  struct and_ : and_t, detail::log {
    using type = and_t;
    using and_t::and_t;
    const detail::terse_<type> _{*this};
  };
  return and_{lhs, rhs};
}

template <class TLhs, class TRhs,
          type_traits::requires_t<type_traits::is_op_v<TLhs> or
                                  type_traits::is_op_v<TRhs>> = 0>
constexpr auto operator or(const TLhs& lhs, const TRhs& rhs) {
  using or_t = detail::or_<typename TLhs::type, typename TRhs::type>;
  struct or_ : or_t, detail::log {
    using type = or_t;
    using or_t::or_t;
    const detail::terse_<type> _{*this};
  };
  return or_{lhs, rhs};
}

template <class T, type_traits::requires_t<type_traits::is_op_v<T>> = 0>
constexpr auto operator not(const T& t) {
  using not_t = detail::not_<typename T::type>;
  struct not_ : not_t, detail::log {
    using type = not_t;
    using not_t::not_t;
    const detail::terse_<type> _{*this};
  };
  return not_{t};
}

}  // namespace terse
}  // namespace operators

template <class TExpr, type_traits::requires_t<
                           type_traits::is_op_v<TExpr> or
                           type_traits::is_convertible_v<TExpr, bool>> = 0>
constexpr auto expect(const TExpr& expr,
                      const reflection::source_location& sl =
                          reflection::source_location::current()) {
  return detail::expect_<TExpr>{detail::on<TExpr>(
      events::assertion<TExpr>{.expr = expr, .location = sl})};
}

[[maybe_unused]] constexpr auto fatal = detail::fatal{};

#if defined(__cpp_nontype_template_parameter_class)
template <auto Constant>
#else
template <bool Constant>
#endif
constexpr auto constant = Constant;

#if defined(__cpp_exceptions)
template <class TException, class TExpr>
[[nodiscard]] constexpr auto throws(const TExpr& expr) {
  return detail::throws_<TExpr, TException>{expr};
}

template <class TExpr>
[[nodiscard]] constexpr auto throws(const TExpr& expr) {
  return detail::throws_<TExpr>{expr};
}

template <class TExpr>
[[nodiscard]] constexpr auto nothrow(const TExpr& expr) {
  return detail::nothrow_{expr};
}
#endif

#if __has_include(<unistd.h>) and __has_include(<sys/wait.h>)
template <class TExpr>
[[nodiscard]] constexpr auto aborts(const TExpr& expr) {
  return detail::aborts_{expr};
}
#endif

using _b = detail::value<bool>;
using _c = detail::value<char>;
using _sc = detail::value<signed char>;
using _s = detail::value<short>;
using _i = detail::value<int>;
using _l = detail::value<long>;
using _ll = detail::value<long long>;
using _u = detail::value<unsigned>;
using _uc = detail::value<unsigned char>;
using _us = detail::value<unsigned short>;
using _ul = detail::value<unsigned long>;
using _ull = detail::value<unsigned long long>;
using _i8 = detail::value<std::int8_t>;
using _i16 = detail::value<std::int16_t>;
using _i32 = detail::value<std::int32_t>;
using _i64 = detail::value<std::int64_t>;
using _u8 = detail::value<std::uint8_t>;
using _u16 = detail::value<std::uint16_t>;
using _u32 = detail::value<std::uint32_t>;
using _u64 = detail::value<std::uint64_t>;
using _f = detail::value<float>;
using _d = detail::value<double>;
using _ld = detail::value<long double>;

template <class T>
struct _t : detail::value<T> {
  constexpr explicit _t(const T& t) : detail::value<T>{t} {}
};

struct suite {
  template <class TSuite>
  constexpr /*explicit(false)*/ suite(TSuite _suite) {
    static_assert(1 == sizeof(_suite));
    detail::on<decltype(+_suite)>(
        events::suite<decltype(+_suite)>{.run = +_suite});
  }
};

[[maybe_unused]] inline auto log = detail::log{};
[[maybe_unused]] inline auto that = detail::that_{};
[[maybe_unused]] constexpr auto test = [](const auto name) {
  return detail::test{"test", name};
};
[[maybe_unused]] constexpr auto should = test;
[[maybe_unused]] inline auto tag = [](const auto name) {
  return detail::tag{{name}};
};
[[maybe_unused]] inline auto skip = tag("skip");
template <class T = void>
[[maybe_unused]] constexpr auto type = detail::type_<T>();

template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto eq(const TLhs& lhs, const TRhs& rhs) {
  return detail::eq_{lhs, rhs};
}
template <class TLhs, class TRhs, class TEpsilon>
[[nodiscard]] constexpr auto approx(const TLhs& lhs, const TRhs& rhs,
                                    const TEpsilon& epsilon) {
  return detail::approx_{lhs, rhs, epsilon};
}
template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto neq(const TLhs& lhs, const TRhs& rhs) {
  return detail::neq_{lhs, rhs};
}
template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto gt(const TLhs& lhs, const TRhs& rhs) {
  return detail::gt_{lhs, rhs};
}
template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto ge(const TLhs& lhs, const TRhs& rhs) {
  return detail::ge_{lhs, rhs};
}
template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto lt(const TLhs& lhs, const TRhs& rhs) {
  return detail::lt_{lhs, rhs};
}
template <class TLhs, class TRhs>
[[nodiscard]] constexpr auto le(const TLhs& lhs, const TRhs& rhs) {
  return detail::le_{lhs, rhs};
}

template <class T>
[[nodiscard]] constexpr auto mut(const T& t) noexcept -> T& {
  return const_cast<T&>(t);
}

namespace bdd {
[[maybe_unused]] constexpr auto feature = [](const auto name) {
  return detail::test{"feature", name};
};
[[maybe_unused]] constexpr auto scenario = [](const auto name) {
  return detail::test{"scenario", name};
};
[[maybe_unused]] constexpr auto given = [](const auto name) {
  return detail::test{"given", name};
};
[[maybe_unused]] constexpr auto when = [](const auto name) {
  return detail::test{"when", name};
};
[[maybe_unused]] constexpr auto then = [](const auto name) {
  return detail::test{"then", name};
};

namespace gherkin {
class steps {
  using step_t = std::string;
  using steps_t = void (*)(steps&);
  using gherkin_t = std::vector<step_t>;
  using call_step_t = utility::function<void(const std::string&)>;
  using call_steps_t = std::vector<std::pair<step_t, call_step_t>>;

  class step {
   public:
    template <class TPattern>
    step(steps& steps, const TPattern& pattern)
        : steps_{steps}, pattern_{pattern} {}

    ~step() { steps_.next(pattern_); }

    template <class TExpr>
    auto operator=(const TExpr& expr) -> void {
      for (const auto& [pattern, _] : steps_.call_steps()) {
        if (pattern_ == pattern) {
          return;
        }
      }

      steps_.call_steps().emplace_back(
          pattern_, [expr, pattern = pattern_](const auto& _step) {
            [=]<class... TArgs>(type_traits::list<TArgs...>) {
              log << _step;
              auto i = 0u;
              const auto& ms = utility::match(pattern, _step);
              expr(lexical_cast<TArgs>(ms[i++])...);
            }(typename type_traits::function_traits<TExpr>::args{});
          });
    }

   private:
    template <class T>
    static auto lexical_cast(const std::string& str) {
      T t{};
      std::istringstream iss{};
      iss.str(str);
      if constexpr (std::is_same_v<T, std::string>) {
        t = iss.str();
      } else {
        iss >> t;
      }
      return t;
    }

    steps& steps_;
    std::string pattern_{};
  };

 public:
  template <class TSteps>
  constexpr /*explicit(false)*/ steps(const TSteps& _steps) : steps_{_steps} {}

  template <class TGherkin>
  auto operator|(const TGherkin& gherkin) {
    gherkin_ = utility::split<std::string>(gherkin, '\n');
    for (auto& _step : gherkin_) {
      _step.erase(0, _step.find_first_not_of(" \t"));
    }

    return [this] {
      step_ = {};
      steps_(*this);
    };
  }
  auto feature(const std::string& pattern) {
    return step{*this, "Feature: " + pattern};
  }
  auto scenario(const std::string& pattern) {
    return step{*this, "Scenario: " + pattern};
  }
  auto given(const std::string& pattern) {
    return step{*this, "Given " + pattern};
  }
  auto when(const std::string& pattern) {
    return step{*this, "When " + pattern};
  }
  auto then(const std::string& pattern) {
    return step{*this, "Then " + pattern};
  }

 private:
  template <class TPattern>
  auto next(const TPattern& pattern) -> void {
    const auto is_scenario = [&pattern](const auto& _step) {
      constexpr auto scenario = "Scenario";
      return pattern.find(scenario) == std::string::npos and
             _step.find(scenario) != std::string::npos;
    };

    const auto call_steps = [this, is_scenario](const auto& _step,
                                                const auto i) {
      for (const auto& [name, call] : call_steps_) {
        if (is_scenario(_step)) {
          break;
        }

        if (utility::is_match(_step, name) or
            not std::empty(utility::match(name, _step))) {
          step_ = i;
          call(_step);
        }
      }
    };

    decltype(step_) i{};
    for (const auto& _step : gherkin_) {
      if (i++ == step_) {
        call_steps(_step, i);
      }
    }
  }

  auto call_steps() -> call_steps_t& { return call_steps_; }

  steps_t steps_{};
  gherkin_t gherkin_{};
  call_steps_t call_steps_{};
  decltype(sizeof("")) step_{};
};
}  // namespace gherkin
}  // namespace bdd

namespace spec {
[[maybe_unused]] constexpr auto describe = [](const auto name) {
  return detail::test{"describe", name};
};
[[maybe_unused]] constexpr auto it = [](const auto name) {
  return detail::test{"it", name};
};
}  // namespace spec

using literals::operator""_test;

using literals::operator""_b;
using literals::operator""_i;
using literals::operator""_s;
using literals::operator""_c;
using literals::operator""_sc;
using literals::operator""_l;
using literals::operator""_ll;
using literals::operator""_u;
using literals::operator""_uc;
using literals::operator""_us;
using literals::operator""_ul;
using literals::operator""_i8;
using literals::operator""_i16;
using literals::operator""_i32;
using literals::operator""_i64;
using literals::operator""_u8;
using literals::operator""_u16;
using literals::operator""_u32;
using literals::operator""_u64;
using literals::operator""_f;
using literals::operator""_d;
using literals::operator""_ld;
using literals::operator""_ull;

using operators::operator==;
using operators::operator!=;
using operators::operator>;
using operators::operator>=;
using operators::operator<;
using operators::operator<=;
using operators::operator and;
using operators::operator or;
using operators::operator not;
using operators::operator|;
using operators::operator/;
using operators::operator>>;
}  // namespace boost::inline ext::ut::inline v1_1_9
#endif
#ifndef GRAPH_PROTOTYPE_BENCHMARK_HPP
#define GRAPH_PROTOTYPE_BENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <fmt/color.h>
#include <fmt/format.h>

#if __has_include(<unistd.h>)  && __has_include(<sys/ioctl.h>)  && __has_include(<sys/syscall.h>) && __has_include(<linux/perf_event.h>)
#define HAS_LINUX_PERFORMANCE_HEADER
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace benchmark {
#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__ __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

#if defined(__GNUC__) or defined(__clang__) and not defined(_LIBCPP_VERSION)

    template<typename T>
    void
    do_not_optimize(T const &val) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : : "r,m"(val) : "memory");
    }

    template<typename T>
    void
    do_not_optimize(T &val) {
#if defined(__clang__)
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : "+r,m"(val) : : "memory");
#else
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : "+m,r"(val) : : "memory");
#endif
    }

#else
#pragma optimize("", off)

    template<class T>
    void
    do_not_optimize(T &&t) {
        reinterpret_cast<char volatile &>(t) = reinterpret_cast<char const volatile &>(t);
    }

#pragma optimize("", on)
#endif

    struct perf_sub_metric {
        uint64_t misses{0};
        uint64_t total{0};
        double ratio{0.0};
    };

    struct perf_metric {
        perf_sub_metric cache;
        perf_sub_metric branch;
        uint64_t instructions{0};
        uint64_t ctx_switches{0};
    };

#ifdef HAS_LINUX_PERFORMANCE_HEADER
    /**
     * A short and sweet performance counter (only works on Linux)
     */
    class PerformanceCounter {
        static bool                       _has_required_rights;
        int                               _fd_misses;
        int                               _fd_accesses;
        int                               _fd_branch_misses;
        int                               _fd_branch;
        int                               _fd_instructions;
        int                               _fd_ctx_switches;
        constexpr static std::string_view _sys_error_message =
                R"(You may not have permission to collect perf stats data.
Consider tweaking /proc/sys/kernel/perf_event_paranoid:
 -1 - Not paranoid at all
  0 - Disallow raw tracepoint access for unpriv
  1 - Disallow cpu events for unpriv
  2 - Disallow kernel profiling for unpriv
quick_fix: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
for details see: https://www.kernel.org/doc/Documentation/sysctl/kernel.txt)";

        static void
        print_access_right_msg(std::string_view msg) noexcept {
            fmt::print(stderr, "PerformanceCounter: {} - error {}: '{}'", msg, errno, strerror(errno));
            _has_required_rights = false;
            std::cerr << std::endl;
            fmt::print(_sys_error_message);
            std::cout << std::endl;
        }

    public:
        PerformanceCounter() {
            constexpr static int PROCESS = 0; // 0: calling process
            constexpr static int ANY_CPU = -1;
            constexpr static int FLAGS   = PERF_FLAG_FD_CLOEXEC;
            if (!_has_required_rights) {
                return;
            }

            perf_event_attr attr{};
            attr.type       = PERF_TYPE_HARDWARE;
            attr.disabled   = 1;
            attr.exclude_hv = 1;

            // cache prediction metric
            attr.config = PERF_COUNT_HW_CACHE_MISSES;
            _fd_misses  = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- misses");
                return;
            }

            attr.config  = PERF_COUNT_HW_CACHE_REFERENCES;
            _fd_accesses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_accesses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- accesses");
                return;
            }

            // branch prediction metric
            attr.config       = PERF_COUNT_HW_BRANCH_MISSES;
            _fd_branch_misses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_branch_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch misses");
                return;
            }

            attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
            _fd_branch  = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_branch == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch accesses");
                return;
            }

            // instruction count metric
            attr.config      = PERF_COUNT_HW_INSTRUCTIONS;
            _fd_instructions = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_instructions == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- instruction count");
                return;
            }

            // ctx switch count metric
            attr.type        = PERF_TYPE_SOFTWARE;
            attr.config      = PERF_COUNT_SW_CONTEXT_SWITCHES;
            _fd_ctx_switches = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_ctx_switches == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- ctx switches");
                return;
            }

            if (ioctl(_fd_misses, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_accesses, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_branch_misses, PERF_EVENT_IOC_ENABLE) == -1
                || ioctl(_fd_branch, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_instructions, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_ctx_switches, PERF_EVENT_IOC_ENABLE) == -1) {
                print_access_right_msg("could not PERF_EVENT_IOC_ENABLE");
                return;
            }
        }

        ~PerformanceCounter() {
            if (_has_required_rights
                && (ioctl(_fd_misses, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_accesses, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_branch_misses, PERF_EVENT_IOC_DISABLE) == -1
                    || ioctl(_fd_branch, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_instructions, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_ctx_switches, PERF_EVENT_IOC_DISABLE) == -1)) {
                print_access_right_msg("could not PERF_EVENT_IOC_DISABLE");
            }
            close(_fd_misses);
            close(_fd_accesses);
            close(_fd_branch_misses);
            close(_fd_branch);
            close(_fd_instructions);
            close(_fd_ctx_switches);
        }

        PerformanceCounter(const PerformanceCounter &) = delete;
        auto &
        operator=(const PerformanceCounter &)
                = delete;

        [[nodiscard]] static bool
        available() noexcept {
            return _has_required_rights;
        }

        /**
         * @return Linux HW/CPU performance counter, best consumed as:
         * @code auto [misses, accesses, branch_misses, branch_total, instructions, ctx_switches] =
         * execMetrics.results();
         */
        [[nodiscard]] auto
        results() const noexcept -> perf_metric {
            if (!_has_required_rights) {
                return {};
            }
            perf_metric           ret;
            constexpr static auto read_metric = [](int metric_fd, auto &data) noexcept -> bool { return read(metric_fd, &data, sizeof(data)) != sizeof(data); };
            if (read_metric(_fd_misses, ret.cache.misses) || read_metric(_fd_accesses, ret.cache.total) || read_metric(_fd_branch_misses, ret.branch.misses) || read_metric(_fd_branch, ret.branch.total)
                || read_metric(_fd_instructions, ret.instructions) || read_metric(_fd_ctx_switches, ret.ctx_switches)) {
                return {};
            }
            using T          = decltype(ret.cache.ratio);
            ret.cache.ratio  = static_cast<T>(ret.cache.misses) / static_cast<T>(ret.cache.total);
            ret.branch.ratio = static_cast<T>(ret.branch.misses) / static_cast<T>(ret.branch.total);
            return ret;
        }
    };

    inline bool PerformanceCounter::_has_required_rights = true;
#else

    class PerformanceCounter {
    public:
        [[nodiscard]] constexpr static bool
        available() noexcept {
            return false;
        }

        /**
         * This OS is not supported
         */
        [[nodiscard]] auto
        results() const noexcept -> perf_metric {
            return {};
        }
    };

#endif

    namespace ut = boost::ut;

/**
 * little compile-time string class (N.B. ideally std::string should become constexpr (everything!!
 * ;-)))
 */
    template<typename CharT, std::size_t SIZE>
    struct fixed_string {
        constexpr static std::size_t N = SIZE;
        CharT _data[N + 1] = {};

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

        [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }

        [[nodiscard]] explicit           operator std::string() const noexcept { return {_data, N}; }

        [[nodiscard]]                    operator const char *() const noexcept { return _data; }

        [[nodiscard]] constexpr bool
        operator==(const fixed_string &other) const noexcept {
            return std::string_view{_data, N} == std::string_view(other);
        }

        template<std::size_t N2>
        [[nodiscard]] friend constexpr bool
        operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
            return false;
        }
    };

    template<typename CharT, std::size_t N>
    fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

    template<fixed_string... key>
    constexpr bool key_not_found = false;

/**
 * constexpr const key map that allows modification of the values during run-time while the
 * compile-time look up of keys is free <-> similar to using structs and refl-cpp-based compile-time
 * reflection
 *
 * @author: Ralph J. Steinhagen
 */
    template<typename Value, fixed_string... Keys>
    class const_key_map {
        constexpr static std::size_t SIZE = sizeof...(Keys);
        constexpr static std::array<const std::string_view, SIZE> _keys = {std::string_view(Keys)...};
        std::array<Value, SIZE> _storage;

        template<fixed_string key>
        constexpr static std::size_t
        get_index_by_name() noexcept {
            if constexpr (constexpr auto itr = std::find_if(_keys.cbegin(), _keys.cend(),
                                                            [](auto const &v) { return v == std::string_view(key); });
                    itr != std::cend(_keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                static_assert(key_not_found<key>, "key not found");
            }
        }

        constexpr static std::size_t
        get_index_by_name(std::string_view key) {
            if (const auto itr = std::find_if(_keys.cbegin(), _keys.cend(),
                                              [&key](const auto &v) { return v == std::string_view(key); }); itr !=
                                                                                                             std::cend(
                                                                                                                     _keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                throw std::range_error("key not found");
            }
        }

        constexpr static std::size_t
        get_index_by_ID(const std::size_t key_ID) {
            if (key_ID < SIZE) {
                return key_ID;
            } else {
                throw std::range_error("key ID not found");
            }
        }

    public:
        [[nodiscard]] static constexpr std::size_t
        size() noexcept {
            return SIZE;
        }

        [[nodiscard]] std::string_view
        key(std::size_t key_ID) const {
            return _keys[get_index_by_ID(key_ID)];
        }

        template<fixed_string key>
        [[nodiscard]] constexpr Value const &
        at() const {
            return _storage[get_index_by_name<key>()];
        }

        template<fixed_string key>
        [[nodiscard]] constexpr Value &
        at() {
            return _storage[get_index_by_name<key>()];
        }

        [[nodiscard]] constexpr Value const &
        at(const std::string_view key) const {
            return _storage[get_index_by_name(key)];
        }

        [[nodiscard]] constexpr Value &
        at(const std::string_view key) {
            return _storage[get_index_by_name(key)];
        }

        [[nodiscard]] constexpr Value const &
        at(const std::size_t key_ID) const {
            return _storage[get_index_by_ID(key_ID)];
        }

        [[nodiscard]] constexpr Value &
        at(const std::size_t key_ID) {
            return _storage[get_index_by_ID(key_ID)];
        }

        [[nodiscard]] constexpr bool
        contains(const std::string_view key) const {
            return std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto &v) { return v == key; }) !=
                   std::cend(_keys);
        }
    };

    template<std::floating_point T>
    struct StatisticsType {
        T min{std::numeric_limits<T>::quiet_NaN()};
        T mean{std::numeric_limits<T>::quiet_NaN()};
        T stddev{std::numeric_limits<T>::quiet_NaN()};
        T median{std::numeric_limits<T>::quiet_NaN()};
        T max{std::numeric_limits<T>::quiet_NaN()};
    };

    struct StringHash {
        using is_transparent = void; // enables heterogeneous lookup

        std::size_t
        operator()(std::string_view sv) const {
            std::hash<std::string_view> hasher;
            return hasher(sv);
        }
    };

    using ResultMap = std::unordered_map<std::string, std::tuple<std::variant<std::monostate, long double, uint64_t, perf_sub_metric>, std::string, std::size_t>, StringHash, std::equal_to<>>;

    class results {
        using EntryType = std::pair<std::string, ResultMap>;
        using Data = std::vector<EntryType>;
        static std::mutex _lock;
        static Data _data;

    public:
        [[nodiscard]] static ResultMap &
        add_result(std::string_view name) noexcept {
            std::lock_guard guard(_lock);
            return _data.emplace_back(std::string(name), ResultMap()).second;
        }

        [[nodiscard]] static constexpr Data const &
        data() noexcept {
            return _data;
        };
    };

    inline results::Data results::_data;
    inline std::mutex    results::_lock;

    class time_point {
        using timePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using timeDiff = std::chrono::nanoseconds;
        timePoint _time_point;

    public:
        time_point &
        now() noexcept {
            _time_point = std::chrono::high_resolution_clock::now();
            return *this;
        }

        timeDiff
        operator-(const time_point &start_marker) const noexcept {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(_time_point - start_marker._time_point);
        }
    };

    namespace utils {

        template<std::size_t N, typename T>
        requires (N > 0)
        constexpr std::vector<T>
        diff(const std::vector<time_point> &stop, time_point start) {
            std::vector<T> ret(N);
            for (auto i = 0LU; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start).count());
                start = stop[i];
            }
            return ret;
        }

        template<std::size_t N, typename T>
        requires (N > 0)
        constexpr std::vector<T>
        diff(const std::vector<time_point> &stop, const std::vector<time_point> &start) {
            std::vector<T> ret(N);
            for (auto i = 0LU; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start[i]).count());
            }
            return ret;
        }

        template<std::size_t n_iterations, typename MapType>
        requires (n_iterations > 0)
        auto
        convert(const std::vector<MapType> &in) {
            std::vector<std::pair<std::string, std::vector<time_point>>> ret;
            ret.resize(in[0].size());

            for (auto keyID = 0LU; keyID < in[0].size(); keyID++) {
                ret[keyID].first = std::string(in[0].key(keyID));
                ret[keyID].second.resize(n_iterations);
                for (auto i = 0LU; i < n_iterations; i++) {
                    ret[keyID].second[i] = in[i].at(keyID);
                }
            }
            return ret;
        }

        template<typename T>
        [[nodiscard]] StatisticsType<T>
        compute_statistics(const std::vector<T> &values) {
            const std::size_t N = values.size();
            if (N < 1) {
                return {};
            }
            const auto minmax = std::minmax_element(values.cbegin(), values.cend());
            const auto mean = std::accumulate(values.begin(), values.end(), T{}) / static_cast<T>(N);

            T stddev{};
            std::for_each(values.cbegin(), values.cend(), [&](const auto x) { stddev += (x - mean) * (x - mean); });
            stddev /= static_cast<T>(N);
            stddev = std::sqrt(stddev);

            // Compute the median value
            std::vector<T> sorted_values(values);
            std::sort(sorted_values.begin(), sorted_values.end());
            const auto median = sorted_values[N / 2];
            return {*minmax.first, mean, stddev, median, *minmax.second};
        }

        template<typename T>
        concept Numeric = std::integral<T> || std::floating_point<T>;

        template<Numeric T>
        std::string
        to_si_prefix(T value_base, std::string_view unit = "s", std::size_t significant_digits = 0) {
            static constexpr std::array si_prefixes{'q', 'r', 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M',
                                                    'G', 'T', 'P', 'E', 'Z', 'Y', 'R', 'Q'};
            static constexpr double base = 1000.0;
            long double value = value_base;

            std::size_t exponent = 10;
            if (value == 0) {
                return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ",
                                   si_prefixes[exponent], unit);
            }
            while (value >= base && exponent < si_prefixes.size()) {
                value /= base;
                ++exponent;
            }
            while (value < 1.0 && exponent > 0) {
                value *= base;
                --exponent;
            }

            return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ",
                               si_prefixes[exponent], unit);
        }

    } // namespace utils

    namespace detail {

        template<typename T>
        struct class_with_call_operator;

// For const operator()
        template<typename ClassType, typename ResultType, typename... Args>
        struct class_with_call_operator<ResultType (ClassType::*)(Args...) const> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

// For non-const operator()
        template<typename ClassType, typename ResultType, typename... Args>
        struct class_with_call_operator<ResultType (ClassType::*)(Args...)> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename T>
        struct fn_traits;

        template<typename ClassType, typename ResultType, typename... Args>
        struct fn_traits<ResultType (ClassType::*)(Args...) const> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename ResultType, typename... Args>
        struct fn_traits<ResultType(Args...)> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename T>
        struct fn_traits : detail::class_with_call_operator<decltype(&T::operator())> {
        };

        template<typename T>
        using return_type_of_t = typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::result_type;

        template<typename T>
        using first_arg_of_t = std::tuple_element_t<0, typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::args_tuple>;
    } // namespace detail

    template<typename TestFunction>
    constexpr std::size_t
    argument_size() noexcept {
        if constexpr (std::invocable<TestFunction>) {
            return 0;
        } else {
            using arg_t = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
            if constexpr (requires { arg_t::size(); }) {
                return arg_t::size();
            } else {
                return 0;
            }
        }
    }

    template<fixed_string... meas_marker_names>
    struct MarkerMap : const_key_map<time_point, meas_marker_names...> {
    };

    template<typename TestFunction, std::size_t N>
    [[nodiscard]] constexpr auto
    get_marker_array() {
        if constexpr (std::invocable<TestFunction>) {
            return std::vector<bool>(N);
        } else {
            using MarkerMapType = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
            return std::vector<MarkerMapType>(N);
        }
    }

    template<std::size_t N_ITERATIONS = 1LU, typename ResultType = results, fixed_string... meas_marker_names>
    class benchmark : public ut::detail::test {
        std::size_t _n_scale_results;
    public:
        benchmark() = delete;

        explicit benchmark(std::string_view _name, std::size_t n_scale_results = 1LU) : ut::detail::test{"benchmark",
                                                                                                         _name},
                                                                                        _n_scale_results(
                                                                                                n_scale_results) {}

        template<class TestFunction, std::size_t MARKER_SIZE = argument_size<TestFunction>(), bool has_arguments =
        MARKER_SIZE != 0>
        // template<fixed_string ...meas_marker, Callback<meas_marker...> Test>
        constexpr benchmark &
        operator=(TestFunction &&_test) {
            static_cast<ut::detail::test &>(*this) = [&_test, this] {
                auto &result_map = ResultType::add_result(name);
                if constexpr (N_ITERATIONS != 1) {
                    result_map.try_emplace("#N", N_ITERATIONS, "", 0);
                } else {
                    result_map.try_emplace("#N", std::monostate{}, "", 0);
                }

                std::vector<time_point> stop_iter(N_ITERATIONS);
                auto marker_iter = get_marker_array<TestFunction, N_ITERATIONS>();

                PerformanceCounter execMetrics;
                const auto start = time_point().now();

                if constexpr (N_ITERATIONS == 1) {
                    if constexpr (std::invocable<TestFunction>) {
                        _test();
                    } else {
                        _test(marker_iter[0]);
                    }
                    stop_iter[0].now();
                } else if constexpr (N_ITERATIONS >= 1) {
                    for (auto i = 0LU; i < N_ITERATIONS; i++) {
                        if constexpr (std::invocable<TestFunction>) {
                            _test();
                        } else {
                            _test(marker_iter[i]);
                        }
                        stop_iter[i].now();
                    }
                } else {
                    throw std::invalid_argument("benchmark n_iteration := 0 parameter not (yet) implemented");
                }
                // N.B. need to retrieve CPU performance count here no to spoil the result by further
                // post-processing

                if (PerformanceCounter::available()) {
                    const perf_metric perf_data = execMetrics.results();
                    result_map.try_emplace("CPU cache misses", perf_data.cache, "", 0);
                    result_map.try_emplace("CPU branch misses", perf_data.branch, "", 0);
                    result_map.try_emplace("<CPU-I>", static_cast<double>(perf_data.instructions) /
                                                      static_cast<double>(N_ITERATIONS * _n_scale_results), "", 1);
                    result_map.try_emplace("CTX-SW", perf_data.ctx_switches, "", 0);
                }
                // not time-critical post-processing starts here
                const auto time_differences_ns = utils::diff<N_ITERATIONS, long double>(stop_iter, start);
                const auto ns = stop_iter[N_ITERATIONS - 1] - start;
                const long double duration_s = 1e-9 * static_cast<long double>(ns.count());

                const auto add_statistics = [&duration_s]<typename T>(ResultMap &map, const T &time_diff) {
                    if constexpr (N_ITERATIONS != 1) {
                        const auto [min, mean, stddev, median, max] = utils::compute_statistics(time_diff);
                        map.try_emplace("min", min, "s", 0);
                        map.try_emplace("mean", mean, "s", 0);
                        if (stddev == 0) {
                            map.try_emplace("stddev", std::monostate{}, "s", 0);
                        } else {
                            map.try_emplace("stddev", stddev, "s", 0);
                        }
                        map.try_emplace("median", median, "s", 0);
                        map.try_emplace("max", max, "s", 0);
                    } else {
                        map.try_emplace("min", std::monostate{}, "s", 0);
                        map.try_emplace("mean", duration_s / N_ITERATIONS, "s", 0);
                        map.try_emplace("stddev", std::monostate{}, "s", 0);
                        map.try_emplace("median", std::monostate{}, "s", 0);
                        map.try_emplace("max", std::monostate{}, "s", 0);
                    }
                };
                add_statistics(result_map, time_differences_ns);

                result_map.try_emplace("total time", duration_s, "s", 0);
                result_map.try_emplace("ops/s", _n_scale_results * N_ITERATIONS / duration_s, "", 1);

                if constexpr (MARKER_SIZE > 0) {
                    auto transposed_map = utils::convert<N_ITERATIONS>(marker_iter);
                    for (auto keyID = 0LU; keyID < transposed_map.size(); keyID++) {
                        if (keyID > 0) {
                            const auto meas = fmt::format("  {}─Marker{}: '{}'→'{}' ", //
                                                          keyID < transposed_map.size() - 1 ? "├" : "└", keyID,
                                                          transposed_map[0].first, transposed_map[keyID].first);

                            auto &marker_result_map = ResultType::add_result(meas);
                            add_statistics(marker_result_map,
                                           utils::diff<N_ITERATIONS, long double>(transposed_map[keyID].second,
                                                                                  transposed_map[0].second));
                        }
                    }
                }
            };
            return *this;
        }

        template<std::size_t N>
        auto
        repeat(std::size_t n_scale_results = 1LU) {
            return ::benchmark::benchmark<N, ResultType, meas_marker_names...>(this->name, n_scale_results);
        }
    };

    [[nodiscard]] auto operator ""_benchmark(const char *name, std::size_t size) {
        return ::benchmark::benchmark<1LU>{{name, size}};
    }

} // namespace benchmark

namespace cfg {
    namespace ut = boost::ut;

    template<class TPrinter = ut::printer>
    class reporter {
        struct {
            std::size_t fail{};
        } _benchmarks{};

        struct {
            std::size_t pass{};
            std::size_t fail{};
        } _asserts{};

        TPrinter _printer{};

    public:
        constexpr reporter &
        operator=(TPrinter printer) {
            _printer = std::move(printer);
            return *this;
        }

        constexpr void
        on(const ut::events::test_begin &) const noexcept { /* not needed */
        }

        void
        on(const ut::events::test_run &test_run) {
            _printer << "\n \"" << test_run.name << "\"...";
        }

        constexpr void
        on(const ut::events::test_skip &bench) const noexcept {
            std::cerr << fmt::format("SKIPPED - {}", bench.name) << std::endl;
            [[maybe_unused]] const auto &map = benchmark::results::add_result(bench.name);
        }

        void
        on(const ut::events::test_end &test_end) {
            if (_asserts.fail > 0) {
                ++_benchmarks.fail;
                _printer << _printer.colors().fail << fmt::format("... in benchmark '{}'", test_end.name)
                         << _printer.colors().none << '\n';
                _asserts.fail--;
            }
        }

        template<class TMsg>
        void
        on(ut::events::log <TMsg> l) {
            _printer << l.msg;
        }

        void
        on(ut::events::exception exception) {
            _printer << fmt::format("\033[31munexpected exception: \"{}\"\n\033[0m", exception.what());
            ++_asserts.fail;
        }

        template<class TExpr>
        void
        on(ut::events::assertion_pass <TExpr>) {
            ++_asserts.pass;
        }

        template<class TExpr>
        void
        on(ut::events::assertion_fail <TExpr> assertion) {
            constexpr auto short_name = [](std::string_view name) {
                return name.rfind('/') != std::string_view::npos ? name.substr(name.rfind('/') + 1) : name;
            };
            _printer << "\n  " << short_name(assertion.location.file_name()) << ':' << assertion.location.line() << ':'
                     << _printer.colors().fail << "FAILED" << _printer.colors().none << " ["
                     << std::boolalpha << assertion.expr << _printer.colors().none << ']';
            ++_asserts.fail;
        }

        void
        on(const ut::events::fatal_assertion &) const { /* not needed testing interface */
        }

        void
        on(const ut::events::summary &) {
            if (_benchmarks.fail || _asserts.fail) {
                std::cout << _printer.str() << std::endl;
                std::cout << fmt::format("\033[31m{} micro-benchmark(s) failed:\n\033[m", _benchmarks.fail);
            } else {
                std::cout << _printer.colors().pass << "all micro-benchmarks passed:\n" << _printer.colors().none;
            }
            print();
            std::cerr.flush();
            std::cout.flush();
        }

        template<std::size_t SIGNIFICANT_DIGITS = 3>
        static void
        print() {
            const auto &data = benchmark::results::data();
            if (data.empty()) {
                fmt::print("no benchmark tests executed\n");
            }
            std::vector<std::size_t> v(data.size());
            // N.B. using <algorithm> rather than <ranges> to be compatible with libc/Emscripten
            // for details see: https://libcxx.llvm.org/Status/Ranges.html
            // not as of clang 15: https://compiler-explorer.com/z/8arxzodh3 ('trunk' seems to be OK)
            std::transform(data.cbegin(), data.cend(), v.begin(), [](auto val) { return val.first.size(); });

            const std::string test_case_label = "benchmark:";
            const std::size_t name_max_size =
                    std::max(*std::max_element(v.cbegin(), v.cend()), test_case_label.size()) + 1LU;

            const auto format = [](auto &value, const std::string &unit, std::size_t digits) -> std::string {
                using ::benchmark::utils::to_si_prefix;
                if (std::holds_alternative<std::monostate>(value)) {
                    return "";
                } else if (std::holds_alternative<long double>(value)) {
                    return fmt::format("{}", to_si_prefix(std::get<long double>(value), unit, digits));
                } else if (std::holds_alternative<uint64_t>(value)) {
                    return fmt::format("{}", to_si_prefix(std::get<uint64_t>(value), unit, digits));
                } else if (std::holds_alternative<benchmark::perf_sub_metric>(value)) {
                    const auto stat = std::get<benchmark::perf_sub_metric>(value);
                    return fmt::format("{:>4} / {:>4} = {:4.1f}%", //
                                       to_si_prefix(stat.misses, unit, 0), to_si_prefix(stat.total, unit, 0),
                                       100.0 * stat.ratio);
                }
                throw std::invalid_argument("benchmark::results: unhandled ResultMap type");
            };

            // compute minimum colum width for each benchmark case and metric
            std::unordered_map<std::string, std::size_t> metric_keys;
            for (auto &[test_name, result_map]: data) {
                for (auto &[metric_key, value]: result_map) {
                    const auto &[value_name, value_unit, value_digits] = value;
                    if (!metric_keys.contains(metric_key)) {
                        metric_keys.try_emplace(metric_key, metric_key.size());
                    }
                    std::size_t value_size = format(value_name, value_unit, value_digits).size();
                    metric_keys.at(metric_key) = std::max(metric_keys.at(metric_key), value_size);
                }
            }
            v.clear();

            bool first_row = true;
            for (auto &[test_name, result_map]: data) {
                if (first_row) {
                    fmt::print("┌{1:─^{0}}", name_max_size + 2UL, test_case_label);
                    fmt::print("┬{1:─^{0}}", sizeof("PASS") + 1UL, "");
                    for (auto const &[metric_key, max_width]: metric_keys) {
                        fmt::print("┬{1:─^{0}}", max_width + 2UL, metric_key);
                    }
                    fmt::print("┐\n");
                    first_row = false;
                }
                fmt::print("│ {1:<{0}} ", name_max_size, test_name);
                if (result_map.size() == 0) {
                    fmt::print("│ \033[33mSKIP\033[0m ");
                } else if (result_map.size() == 1) {
                    fmt::print("│ \033[31mFAIL\033[0m ");
                } else {
                    fmt::print("│ \033[32mPASS\033[0m ");
                }

                for (auto &[metric_key, max_width]: metric_keys) {
                    if (result_map.contains(metric_key)) {
                        const auto &[value, unit, digits] = result_map.at(metric_key);
                        fmt::print("│ {1:>{0}} ", max_width, format(value, unit, digits));
                    } else {
                        fmt::print("│ {1:>{0}} ", max_width, "");
                    }
                }
                fmt::print("│\n");
            }

            fmt::print("└{1:─^{0}}", name_max_size + 2UL, "");
            fmt::print("┴{1:─^{0}}", sizeof("PASS") + 1UL, "");
            for (auto const &[metric_key, max_width]: metric_keys) {
                fmt::print("┴{1:─^{0}}", max_width + 2UL, "");
            }
            fmt::print("┘\n");
        }
    };
} // namespace cfg

template<>
inline auto boost::ut::cfg<boost::ut::override> = ut::runner < cfg::reporter<printer>>
{
};

#endif // GRAPH_PROTOTYPE_BENCHMARK_HPP
