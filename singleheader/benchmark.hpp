#ifndef GNURADIO_BENCHMARK_HPP
#define GNURADIO_BENCHMARK_HPP

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

// #include <boost/ut.hpp>
//
// Copyright (c) 2019-2021 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#if defined(BOOST_UT_CXX_MODULES)
#define BOOST_UT_EXPORT export
#else

#define BOOST_UT_EXPORT
#endif

#if !defined(BOOST_UT_CXX_MODULES)
#include <version>
#endif

#if defined(_MSC_VER)
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif
// Before libc++ 17 had experimental support for format and it required a
// special build flag. Currently libc++ has not implemented all C++20 chrono
// improvements. Therefore doesn't define __cpp_lib_format, instead query the
// library version to detect the support status.
//
// MSVC STL and libstdc++ provide __cpp_lib_format.
#if defined(__cpp_lib_format) or \
    (defined(_LIBCPP_VERSION) and _LIBCPP_VERSION >= 170000)
#define BOOST_UT_HAS_FORMAT
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
#define BOOST_UT_VERSION 2'3'1

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

#if !defined(BOOST_UT_CXX_MODULES)
#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#if __has_include(<unistd.h>) and __has_include(<sys/wait.h>)
#include <sys/wait.h>
#include <unistd.h>
#endif
#if defined(__cpp_exceptions)
#include <exception>
#endif

#if __has_include(<format>)
#include <format>
#endif
#if __has_include(<source_location>)
#include <source_location>
#endif
#endif // cxx modules

struct unique_name_for_auto_detect_prefix_and_suffix_length_0123456789_struct_ {
};

BOOST_UT_EXPORT
namespace boost::inline ext::ut::inline v2_3_1 {
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
constexpr auto regex_match(const char* str, const char* pattern) -> bool {
  if (*pattern == '\0' && *str == '\0') {
    return true;
  }
  if (*pattern == '\0' && *str != '\0') {
    return false;
  }
  if (*str == '\0' && *pattern != '\0') {
    return false;
  }
  if (*pattern == '.') {
    return regex_match(str + 1, pattern + 1);
  }
  if (*pattern == *str) {
    return regex_match(str + 1, pattern + 1);
  }
  return false;
}
}  // namespace utility

namespace reflection {
#if defined(__cpp_lib_source_location) && !defined(_LIBCPP_APPLE_CLANG_VER)
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
namespace detail {
template <typename TargetType>
[[nodiscard]] constexpr auto get_template_function_name_use_type()
    -> std::string_view {
// for over compiler need over macros
#if defined(_MSC_VER) && !defined(__clang__)
  return {&__FUNCSIG__[0], sizeof(__FUNCSIG__)};
#else
  return {&__PRETTY_FUNCTION__[0], sizeof(__PRETTY_FUNCTION__)};
#endif
}

// decay allows you to highlight a cleaner name
template <typename TargetType>
[[nodiscard]] constexpr auto get_template_function_name_use_decay_type()
    -> std::string_view {
  return get_template_function_name_use_type<std::decay_t<TargetType>>();
}

inline constexpr const std::string_view raw_type_name =
    get_template_function_name_use_decay_type<
        unique_name_for_auto_detect_prefix_and_suffix_length_0123456789_struct_>();

inline constexpr const std::size_t raw_length = raw_type_name.length();
inline constexpr const std::string_view need_name =
#if defined(_MSC_VER) and not defined(__clang__)
    "struct "
    "unique_name_for_auto_detect_prefix_and_suffix_length_0123456789_struct_";
#else
    "unique_name_for_auto_detect_prefix_and_suffix_length_0123456789_struct_";
#endif
inline constexpr const std::size_t need_length = need_name.length();
static_assert(need_length <= raw_length,
              "Auto find prefix and suffix length broken error 1");
inline constexpr const std::size_t prefix_length =
    raw_type_name.find(need_name);
static_assert(prefix_length != std::string_view::npos,
              "Auto find prefix and suffix length broken error 2");
static_assert(prefix_length <= raw_length,
              "Auto find prefix and suffix length broken error 3");
inline constexpr const std::size_t tail_length = raw_length - prefix_length;
static_assert(need_length <= tail_length,
              "Auto find prefix and suffix length broken error 4");
inline constexpr const std::size_t suffix_length = tail_length - need_length;

}  // namespace detail

template <typename TargetType>
[[nodiscard]] constexpr auto type_name() -> std::string_view {
  const std::string_view raw_type_name =
      detail::get_template_function_name_use_type<TargetType>();
  const std::size_t end = raw_type_name.length() - detail::suffix_length;
  const std::size_t len = end - detail::prefix_length;
  std::string_view result = raw_type_name.substr(detail::prefix_length, len);
  return result;
}

// decay allows you to highlight a cleaner name
template <typename TargetType>
[[nodiscard]] constexpr auto decay_type_name() -> std::string_view {
  const std::string_view raw_type_name =
      detail::get_template_function_name_use_decay_type<TargetType>();
  const std::size_t end = raw_type_name.length() - detail::suffix_length;
  const std::size_t len = end - detail::prefix_length;
  std::string_view result = raw_type_name.substr(detail::prefix_length, len);
  return result;
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

template <class T, class = void>
struct has_static_member_object_value : std::false_type {};

template <class T>
struct has_static_member_object_value<
    T, std::void_t<decltype(std::declval<T>().value)>>
    : std::bool_constant<!std::is_member_pointer_v<decltype(&T::value)> &&
                         !std::is_function_v<decltype(T::value)>> {};

template <class T>
inline constexpr bool has_static_member_object_value_v =
    has_static_member_object_value<T>::value;

template <class T, class = void>
struct has_static_member_object_epsilon : std::false_type {};

template <class T>
struct has_static_member_object_epsilon<
    T, std::void_t<decltype(std::declval<T>().epsilon)>>
    : std::bool_constant<!std::is_member_pointer_v<decltype(&T::epsilon)> &&
                         !std::is_function_v<decltype(T::epsilon)>> {};

template <class T>
inline constexpr bool has_static_member_object_epsilon_v =
    has_static_member_object_epsilon<T>::value;

}  // namespace type_traits

namespace concepts {

// std::convertible_to also requires implicit conversion to work
// See https://stackoverflow.com/a/76547623
template <class From, class To>
concept explicitly_convertible_to =
    requires { static_cast<To>(std::declval<From>()); };

template <class T>
concept ostreamable = requires(std::ostringstream& os, T t) { os << t; };

}  // namespace concepts

template <typename CharT, std::size_t SIZE>
struct fixed_string {
  constexpr static std::size_t N = SIZE;
  CharT _data[N + 1] = {};

  constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
    if constexpr (N != 0) {
      for (std::size_t i = 0; i < N; ++i) {
        _data[i] = str[i];
      }
    }
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }
  [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }
  [[nodiscard]] constexpr explicit operator std::string_view() const noexcept {
    return {_data, N};
  }
  [[nodiscard]] explicit operator std::string() const noexcept {
    return {_data, N};
  }
  [[nodiscard]] operator const char*() const noexcept { return _data; }
  [[nodiscard]] constexpr bool operator==(
      const fixed_string& other) const noexcept {
    return std::string_view{_data, N} == std::string_view(other);
  }

  template <std::size_t N2>
  [[nodiscard]] friend constexpr bool operator==(
      const fixed_string&, const fixed_string<CharT, N2>&) {
    return false;
  }
};

template <typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

struct none {};

namespace events {
struct run_begin {
  int argc{};
  const char** argv{};
};
struct test_begin {
  std::string_view type{};
  std::string_view name{};
  reflection::source_location location{};
};
struct suite_begin {
  std::string_view type{};
  std::string_view name{};
  reflection::source_location location{};
};
struct suite_end {
  std::string_view type{};
  std::string_view name{};
  reflection::source_location location{};
};
template <class Test, class TArg = none>
struct test {
  std::string_view type{};
  std::string name{};  /// might be dynamic
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
  std::string_view name{};
  constexpr auto operator()() { run(); }
  constexpr auto operator()() const { run(); }
};
template <class TSuite>
suite(TSuite) -> suite<TSuite>;
struct test_run {
  std::string_view type{};
  std::string_view name{};
};
struct test_finish {
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
struct fatal_assertion : std::exception {};
struct exception {
  const char* msg{};
  [[nodiscard]] auto what() const -> const char* { return msg; }
};
struct summary {};
}  // namespace events

namespace detail {
struct op {};

template <class>
struct fatal_;

struct fatal {
  template <class T>
  [[nodiscard]] auto operator()(const T& t) const {
    return detail::fatal_{t};
  }
};
struct cfg {
  using value_ref = std::variant<std::monostate, std::reference_wrapper<bool>,
                                 std::reference_wrapper<std::size_t>,
                                 std::reference_wrapper<std::string>>;
  using option = std::tuple<std::string, std::string, value_ref, std::string>;
  static inline reflection::source_location location{};
  static inline bool wip{};

#if defined(_MSC_VER)
  static inline int largc = __argc;
  static inline const char** largv = const_cast<const char**>(__argv);
#else
  static inline int largc = 0;
  static inline const char** largv = nullptr;
#endif

  static inline std::string executable_name = "unknown executable";
  static inline std::string query_pattern;           // <- done
  static inline bool invert_query_pattern = false;   // <- done
  static inline std::string query_regex_pattern;     // <- done
  static inline bool show_help = false;              // <- done
  static inline bool show_tests = false;             // <- done
  static inline bool list_tags = false;              // <- done
  static inline bool show_successful_tests = false;  // <- done
  static inline std::string output_filename;
  static inline std::string use_reporter = "console";  // <- done
  static inline std::string suite_name;
  static inline bool abort_early = false;  // <- done
  static inline std::size_t abort_after_n_failures =
      std::numeric_limits<std::size_t>::max();  // <- done
  static inline bool show_duration = false;     // <- done
  static inline std::size_t show_min_duration = 0;
  static inline std::string input_filename;
  static inline bool show_test_names = false;  // <- done
  static inline bool show_reporters = false;   // <- done
  static inline std::string sort_order = "decl";
  static inline std::size_t rnd_seed = 0;        // 0: use time
  static inline std::string use_colour = "yes";  // <- done
  static inline bool show_lib_identity = false;  // <- done
  static inline std::string wait_for_keypress = "never";

  static inline const std::vector<option> options = {
      // clang-format off
  // <short long option name>, <option arg>, <ref to cfg>, <description>
  {"-? -h --help", "", std::ref(show_help), "display usage information"},
  {"-l --list-tests", "", std::ref(show_tests), "list all/matching test cases"},
  {"-t, --list-tags", "", std::ref(list_tags), "list all/matching tags"},
  {"-s, --success", "", std::ref(show_successful_tests), "include successful tests in output"},
  {"-o, --out", "<filename>", std::ref(output_filename), "output filename"},
  {"-r, --reporter", "<name>", std::ref(use_reporter), "reporter to use (defaults to console)"},
  {"-n, --name", "<name>", std::ref(suite_name), "suite name"},
  {"-a, --abort", "", std::ref(abort_early), "abort at first failure"},
  {"-x, --abortx", "<no. failures>", std::ref(abort_after_n_failures), "abort after x failures"},
  {"-d, --durations", "", std::ref(show_duration), "show test durations"},
  {"-D, --min-duration", "<seconds>", std::ref(show_min_duration), "show test durations for [...]"},
  {"-f, --input-file", "<filename>", std::ref(input_filename), "load test names to run from a file"},
  {"--list-test-names-only", "", std::ref(show_test_names), "list all/matching test cases names only"},
  {"--list-reporters", "", std::ref(show_reporters), "list all reporters"},
  {"--order <decl|lex|rand>", "", std::ref(sort_order), "test case order (defaults to decl)"},
  {"--rng-seed", "<'time'|number>", std::ref(rnd_seed), "set a specific seed for random numbers"},
  {"--use-colour", "<yes|no>", std::ref(use_colour), "should output be colourised"},
  {"--libidentify", "", std::ref(show_lib_identity), "report name and version according to libidentify standard"},
  {"--wait-for-keypress", "<never|start|exit|both>", std::ref(wait_for_keypress), "waits for a keypress before exiting"}
      // clang-format on
  };

  static std::optional<cfg::option> find_arg(std::string_view arg) {
    for (const auto& option : cfg::options) {
      if (std::get<0>(option).find(arg) != std::string::npos) {
        return option;
      }
    }
    return std::nullopt;
  }

  static void print_usage() {
    std::size_t opt_width = 30;
    std::cout << cfg::executable_name
              << " [<test name|pattern|tags> ... ] options\n\nwith options:\n";
    for (const auto& [cmd, arg, val, description] : cfg::options) {
      std::string s = cmd;
      s.append(" ");
      s.append(arg);
      // pad fixed column width
      const auto pad_by = (s.size() <= opt_width) ? opt_width - s.size() : 0;
      s.insert(s.end(), pad_by, ' ');
      std::cout << "  " << s << description << std::endl;
    }
  }

  static void print_identity() {
    // according to: https://github.com/janwilmans/LibIdentify
    std::cout << "description:    A UT / μt test executable\n";
    std::cout << "category:       testframework\n";
    std::cout << "framework:      UT: C++20 μ(micro)/Unit Testing Framework\n";
    std::cout << "version:        " << BOOST_UT_VERSION << std::endl;
  }

  static void parse_arg_with_fallback(int argc, const char* argv[]) {
    //int before call main
    if (argc > 0 && argv != nullptr) {
      cfg::largc = argc;
      cfg::largv = argv;
    }
    else
    {
      cfg::largc = 0;
      cfg::largv = nullptr;
    }
    parse(cfg::largc, cfg::largv);
  }

  static void parse(int argc, const char* argv[]) {
    const std::size_t n_args = argc > 0 ? static_cast<std::size_t>(argc) : 0U;
    if (n_args > 0 && argv != nullptr) {
      executable_name = argv[0];
    }
    query_pattern = "";
    bool found_first_option = false;
    for (auto i = 1U; i < n_args && argv != nullptr; i++) {
      std::string cmd(argv[i]);
      auto cmd_option = find_arg(cmd);
      if (!cmd_option.has_value()) {
        if (found_first_option) {
          std::cerr << "unknown option: '" << cmd << "' run:" << std::endl;
          std::cerr << "'" << executable_name << " --help'" << std::endl;
          std::cerr << "for additional help" << std::endl;
          std::exit(-1);
        } else {
          if (i > 1U) {
            query_pattern.append(" ");
          }
          query_pattern.append(cmd);
        }
        continue;
      }
      found_first_option = true;
      auto var = std::get<value_ref>(*cmd_option);
      const bool has_option_arg = !std::get<1>(*cmd_option).empty();
      if (!has_option_arg &&
          std::holds_alternative<std::reference_wrapper<bool>>(var)) {
        std::get<std::reference_wrapper<bool>>(var).get() = true;
        continue;
      }
      if ((i + 1) >= n_args) {
        std::cerr << "missing argument for option " << argv[i] << std::endl;
        std::exit(-1);
      }
      i += 1;  // skip to next argv for parsing
      if (std::holds_alternative<std::reference_wrapper<std::size_t>>(var)) {
        // parse size argument
        std::size_t last;
        std::string argument(argv[i]);
        auto val = static_cast<std::size_t>(std::stoull(argument, &last));
        if (last != argument.length()) {
          std::cerr << "cannot parse option of " << argv[i - 1] << " "
                    << argv[i] << std::endl;
          std::exit(-1);
        }
        std::get<std::reference_wrapper<std::size_t>>(var).get() = val;
      }
      if (std::holds_alternative<std::reference_wrapper<std::string>>(var)) {
        // parse string argument
        std::get<std::reference_wrapper<std::string>>(var).get() = argv[i];
        continue;
      }
    }

    if (show_help) {
      print_usage();
      std::exit(0);
    }

    if (show_lib_identity) {
      print_identity();
      std::exit(0);
    }

    if (!query_pattern.empty()) {  // simple glob-like search
      query_regex_pattern = "";
      for (const char c : query_pattern) {
        if (c == '!') {
          invert_query_pattern = true;
        } else if (c == '*') {
          query_regex_pattern += ".*";
        } else if (c == '?') {
          query_regex_pattern += '.';
        } else if (c == '.') {
          query_regex_pattern += "\\.";
        } else if (c == '\\') {
          query_regex_pattern += "\\\\";
        } else {
          query_regex_pattern += c;
        }
      }
    }
  }
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
  // NOLINTNEXTLINE(readability-const-return-type)
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

template <std::floating_point T>
struct value<T> : op {
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
            return lhs.value == rhs.value;
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
            return lhs.value != rhs.value;
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
            return lhs.value > rhs.value;
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
            return lhs.value >= rhs.value;
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
#if defined(_MSC_VER) && !defined(__clang__)
            // for some reason, accessing the static member via :: does not compile on MSVC,
            // and the next line does not compile on clang, so we have to use the ifdef here
            return lhs.value < rhs.value;
#else
            return TLhs::value < TRhs::value;
#endif
          } else {
            return get(lhs_) < get(rhs_);
          }
        }()} {
  }

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
            return lhs.value <= rhs.value;
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
            return false;
          } catch (const TException&) {
            return true;
          } catch (...) {
            return false;
          }
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
            return false;
          } catch (...) {
            return true;
          }
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
            return true;
          } catch (...) {
            return false;
          }
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
            std::exit(0);
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
concept is_op = std::derived_from<T, detail::op>;

template <typename T, typename = void>
struct is_stream_insertable : std::false_type {};

template <typename T>
struct is_stream_insertable<
    T, std::void_t<decltype(std::declval<std::ostream&>()
                            << detail::get(std::declval<T>()))>>
    : std::true_type {};
template <typename T>
inline constexpr bool is_stream_insertable_v = is_stream_insertable<T>::value;
}  // namespace type_traits

struct colors {
  std::string_view none = "\033[0m";
  std::string_view pass = "\033[32m";
  std::string_view fail = "\033[31m";
  std::string_view skip = "\033[33m";
};

class printer {
  [[nodiscard]] auto color(const bool cond) {
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

  template <class T>
    requires std::ranges::range<T> && (!concepts::ostreamable<T>)
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

  auto on(events::run_begin) -> void {}

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

  auto on(const events::fatal_assertion&) -> void {}

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

template <class TPrinter = printer>
class reporter_junit {
  template <typename Key, typename T>
  using map = std::unordered_map<Key, T>;
  using clock_ref = std::chrono::high_resolution_clock;
  using timePoint = std::chrono::time_point<clock_ref>;
  using timeDiff = std::chrono::milliseconds;
  enum class ReportType : std::uint8_t { CONSOLE, JUNIT } report_type_;
  static constexpr ReportType CONSOLE = ReportType::CONSOLE;
  static constexpr ReportType JUNIT = ReportType::JUNIT;

  struct test_result {
    test_result* parent = nullptr;
    std::string class_name;
    std::string suite_name;
    std::string test_name;
    std::string status = "STARTED";
    timePoint run_start = clock_ref::now();
    timePoint run_stop = clock_ref::now();
    std::size_t n_tests = 0LU;
    std::size_t assertions = 0LU;
    std::size_t passed = 0LU;
    std::size_t skipped = 0LU;
    std::size_t fails = 0LU;
    std::string report_string{};
    std::unique_ptr<map<std::string, test_result>> nested_tests =
        std::make_unique<map<std::string, test_result>>();
  };
  colors color_{};
  map<std::string, test_result> results_;
  std::string active_suite_{"global"};
  test_result* active_scope_ = &results_[active_suite_];
  std::stack<std::string> active_test_{};

  std::streambuf* cout_save = std::cout.rdbuf();
  std::ostream lcout_;
  TPrinter printer_;
  std::stringstream ss_out_{};

  void reset_printer() {
    ss_out_.str("");
    ss_out_.clear();
  }

  void check_for_scope(std::string_view test_name) {
    const std::string str_name(test_name);
    active_test_.push(str_name);
    const auto [iter, inserted] = active_scope_->nested_tests->try_emplace(
        str_name, test_result{active_scope_, detail::cfg::executable_name,
                              active_suite_, str_name});
    active_scope_ = &active_scope_->nested_tests->at(str_name);
    if (active_test_.size() == 1) {
      reset_printer();
    }
    active_scope_->run_start = clock_ref::now();
    if (!inserted) {
      std::cout << "WARNING test '" << str_name << "' for test suite '"
                << active_suite_ << "' already present\n";
    }
  }

  void pop_scope(std::string_view test_name_sv) {
    const std::string test_name(test_name_sv);
    active_scope_->run_stop = clock_ref::now();
    if (active_scope_->skipped) {
      active_scope_->status = "SKIPPED";
    } else {
      active_scope_->status = active_scope_->fails > 0 ? "FAILED" : "PASSED";
    }
    active_scope_->assertions =
        active_scope_->assertions + active_scope_->fails;

    if (active_test_.top() == test_name) {
      active_test_.pop();
      auto old_scope = active_scope_;
      if (active_scope_->parent != nullptr) {
        active_scope_ = active_scope_->parent;
      } else {
        active_scope_ = &results_[std::string{"global"}];
      }
      active_scope_->n_tests += old_scope->n_tests + 1LU;
      active_scope_->assertions += old_scope->assertions;
      active_scope_->passed += old_scope->passed;
      active_scope_->skipped += old_scope->skipped;
      active_scope_->fails += old_scope->fails;
      return;
    }
    std::stringstream ss("runner returned from test w/o signaling: ");
    ss << "not popping because '" << active_test_.top() << "' differs from '"
       << test_name << "'" << std::endl;
#if defined(__cpp_exceptions)
    throw std::logic_error(ss.str());
#else
    std::abort();
#endif
  }

 public:
  constexpr auto operator=(TPrinter printer) {
    printer_ = static_cast<TPrinter&&>(printer);
  }
  reporter_junit() : lcout_(std::cout.rdbuf()) {}
  ~reporter_junit() { std::cout.rdbuf(cout_save); }

  auto on(events::run_begin run) {
    ::boost::ut::detail::cfg::parse_arg_with_fallback(run.argc, run.argv);

    if (detail::cfg::show_reporters) {
      std::cout << "available reporter:\n";
      std::cout << "  console (default)\n";
      std::cout << "  junit" << std::endl;
      std::exit(0);
    }
    if (detail::cfg::use_reporter.starts_with("junit")) {
      report_type_ = JUNIT;
    } else {
      report_type_ = CONSOLE;
    }
    if (!detail::cfg::use_colour.starts_with("yes")) {
      color_ = {"", "", "", ""};
    }
    if (!detail::cfg::show_tests && !detail::cfg::show_test_names) {
      std::cout.rdbuf(ss_out_.rdbuf());
    }
  }

  auto on(events::suite_begin suite) -> void {
    while (active_test_.size() > 0) {
      pop_scope(active_test_.top());
    }
    active_suite_ = suite.name;
    active_scope_ = &results_[active_suite_];
  }

  auto on(events::suite_end) -> void {
    while (active_test_.size() > 0) {
      pop_scope(active_test_.top());
    }
    active_suite_ = "global";
    active_scope_ = &results_[active_suite_];
  }

  auto on(events::test_begin test_event) -> void {  // starts outermost test
    check_for_scope(test_event.name);

    if (report_type_ == CONSOLE) {
      ss_out_ << "\n";
      ss_out_ << std::string((2 * active_test_.size()) - 2, ' ');
      ss_out_ << "Running " << test_event.type << " \"" << test_event.name
              << "\"... ";
    }
  }

  auto on(events::test_end test_event) -> void {
    if (active_scope_->fails > 0) {
      reset_printer();
    } else {
      active_scope_->report_string = ss_out_.str();
      active_scope_->passed += 1LU;
      if (report_type_ == CONSOLE) {
        if (detail::cfg::show_successful_tests) {
          if (!active_scope_->nested_tests->empty()) {
            ss_out_ << "\n";
            ss_out_ << std::string((2 * active_test_.size()) - 2, ' ');
            ss_out_ << "Running test \"" << test_event.name << "\" - ";
          }
          ss_out_ << color_.pass << "PASSED" << color_.none;
          print_duration(ss_out_);
          lcout_ << ss_out_.str();
          reset_printer();
        }
      }
    }

    pop_scope(test_event.name);
  }

  auto on(events::test_run test_event) -> void {  // starts nested test
    on(events::test_begin{.type = test_event.type, .name = test_event.name});
  }

  auto on(events::test_finish test_event) -> void {  // finishes nested test
    on(events::test_end{.type = test_event.type, .name = test_event.name});
  }

  auto on(events::test_skip test_event) -> void {
    ss_out_.clear();
    if (!active_scope_->nested_tests->contains(std::string(test_event.name))) {
      check_for_scope(test_event.name);
      active_scope_->status = "SKIPPED";
      active_scope_->skipped += 1;
      if (report_type_ == CONSOLE) {
        lcout_ << '\n' << std::string((2 * active_test_.size()) - 2, ' ');
        lcout_ << "Running \"" << test_event.name << "\"... ";
        lcout_ << color_.skip << "SKIPPED" << color_.none << '\n';
      }
      reset_printer();
      pop_scope(test_event.name);
    }
  }

  template <class TMsg>
  auto on(events::log<TMsg> log) -> void {
    ss_out_ << log.msg;
    if (report_type_ == CONSOLE) {
      lcout_ << log.msg;
    }
  }

  auto on(events::exception exception) -> void {
    active_scope_->fails++;
    if (!active_test_.empty()) {
      active_scope_->report_string += color_.fail;
      active_scope_->report_string += "Unexpected exception with message:\n";
      active_scope_->report_string += exception.what();
      active_scope_->report_string += color_.none;
    }
    if (report_type_ == CONSOLE) {
      lcout_ << std::string((2 * active_test_.size()) - 2, ' ');
      lcout_ << "Running test \"" << active_test_.top() << "\"... ";
      lcout_ << color_.fail << "FAILED" << color_.none;
      print_duration(lcout_);
      lcout_ << '\n';
      lcout_ << active_scope_->report_string << '\n';
    }
    if (detail::cfg::abort_early ||
        active_scope_->fails >= detail::cfg::abort_after_n_failures) {
      std::cerr << "early abort for test : " << active_test_.top() << "after ";
      std::cerr << active_scope_->fails << " failures total." << std::endl;
      std::exit(-1);
    }
  }

  template <class TExpr>
  auto on(events::assertion_pass<TExpr>) -> void {
    active_scope_->assertions++;
  }

  template <class TExpr>
  auto on(events::assertion_fail<TExpr> assertion) -> void {
    TPrinter ss{};
    ss << ss_out_.str();
    if (report_type_ == CONSOLE) {
      ss << color_.fail << "FAILED\n" << color_.none;
      print_duration(ss);
    }
    ss << "in: " << assertion.location.file_name() << ':'
       << assertion.location.line();
    ss << color_.fail << " - test condition: ";
    ss << " [" << std::boolalpha << assertion.expr;
    ss << color_.fail << ']' << color_.none;
    active_scope_->report_string += ss.str();
    active_scope_->fails++;
    reset_printer();
    if (report_type_ == CONSOLE) {
      lcout_ << active_scope_->report_string << "\n\n";
    }
    if (detail::cfg::abort_early ||
        active_scope_->fails >= detail::cfg::abort_after_n_failures) {
      std::cerr << "early abort for test : " << active_test_.top() << "after ";
      std::cerr << active_scope_->fails << " failures total." << std::endl;
      std::exit(-1);
    }
  }

  auto on(const events::fatal_assertion&) -> void { active_scope_->fails++; }

  auto on(events::summary) -> void {
    std::cout.flush();
    std::cout.rdbuf(cout_save);
    std::ofstream maybe_of;
    if (detail::cfg::output_filename != "") {
      maybe_of = std::ofstream(detail::cfg::output_filename);
    }

    if (report_type_ == JUNIT) {
      print_junit_summary(detail::cfg::output_filename != "" ? maybe_of
                                                             : std::cout);
      return;
    }
    print_console_summary(
        detail::cfg::output_filename != "" ? maybe_of : std::cout,
        detail::cfg::output_filename != "" ? maybe_of : std::cerr);
  }

 protected:
  void print_duration(auto& printer) const noexcept {
    if (detail::cfg::show_duration) {
      std::int64_t time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              active_scope_->run_stop - active_scope_->run_start)
              .count();
      // rounded to nearest ms
      double time_s = static_cast<double>(time_ms) / 1000.0;
      printer << " after " << time_s << " seconds";
    }
  }

  void print_console_summary(std::ostream& out_stream,
                             std::ostream& err_stream) {
    for (const auto& [suite_name, suite_result] : results_) {
      if (suite_result.fails) {
        err_stream
            << "\n========================================================"
               "=======================\n"
            << "Suite " << suite_name << '\n'  //
            << "tests:   " << (suite_result.n_tests) << " | " << color_.fail
            << suite_result.fails << " failed" << color_.none << '\n'
            << "asserts: " << (suite_result.assertions) << " | "
            << suite_result.passed << " passed"
            << " | " << color_.fail << suite_result.fails << " failed"
            << color_.none << '\n';
        std::cerr << std::endl;
      } else {
        out_stream << color_.pass << "Suite '" << suite_name
                   << "': all tests passed" << color_.none << " ("
                   << suite_result.assertions << " asserts in "
                   << suite_result.n_tests << " tests)\n";

        if (suite_result.skipped) {
          std::cout << suite_result.skipped << " tests skipped\n";
        }

        std::cout.flush();
      }
    }
  }

  void print_junit_summary(std::ostream& stream) {
    // aggregate results
    size_t n_tests = 0;
    size_t n_fails = 0;
    double total_time = 0.0;
    auto suite_time = [](auto const& suite_result) {
      std::int64_t time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              suite_result.run_stop - suite_result.run_start)
              .count();
      return static_cast<double>(time_ms) / 1000.0;
    };
    for (const auto& [suite_name, suite_result] : results_) {
      n_tests += suite_result.assertions;
      n_fails += suite_result.fails;
      total_time += suite_time(suite_result);
    }

    // mock junit output:
    stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    stream << "<testsuites";
    stream << " name=\"all\"";
    stream << " tests=\"" << n_tests << '\"';
    stream << " failures=\"" << n_fails << '\"';
    stream << " time=\"" << total_time << '\"';
    stream << ">\n";

    for (const auto& [suite_name, suite_result] : results_) {
      stream << "<testsuite";
      stream << " classname=\"" << detail::cfg::executable_name << '\"';
      stream << " name=\"" << suite_name << '\"';
      stream << " tests=\"" << suite_result.assertions << '\"';
      stream << " errors=\"" << suite_result.fails << '\"';
      stream << " failures=\"" << suite_result.fails << '\"';
      stream << " skipped=\"" << suite_result.skipped << '\"';
      stream << " time=\"" << suite_time(suite_result) << '\"';
      stream << " version=\"" << BOOST_UT_VERSION << "\">\n";
      print_result(stream, suite_name, " ", suite_result);
      stream << "</testsuite>\n";
      stream.flush();
    }
    stream << "</testsuites>";
  }
  void print_result(std::ostream& stream, const std::string& suite_name,
                    const std::string& indent, const test_result& parent) {
    for (const auto& [name, result] : *parent.nested_tests) {
      stream << indent;
      stream << "<testcase classname=\"" << result.suite_name << '\"';
      stream << " name=\"" << name << '\"';
      stream << " tests=\"" << result.assertions << '\"';
      stream << " errors=\"" << result.fails << '\"';
      stream << " failures=\"" << result.fails << '\"';
      stream << " skipped=\"" << result.skipped << '\"';
      std::int64_t time_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              result.run_stop - result.run_start)
              .count();
      stream << " time=\"" << (static_cast<double>(time_ms) / 1000.0) << "\"";
      stream << " status=\"" << result.status << '\"';
      if (result.report_string.empty() && result.nested_tests->empty()) {
        stream << " />\n";
      } else if (!result.nested_tests->empty()) {
        stream << " />\n";
        print_result(stream, suite_name, indent + "  ", result);
        stream << indent << "</testcase>\n";
      } else if (!result.report_string.empty()) {
        stream << ">\n";
        stream << indent << indent << "<system-out>\n";
        stream << result.report_string << "\n";
        stream << indent << indent << "</system-out>\n";
        stream << indent << "</testcase>\n";
      }
    }
  }
};

struct options {
  std::string_view filter{};
  std::vector<std::string_view> tag{};
  ut::colors colors{};
  bool dry_run{};
};

struct run_cfg {
  bool report_errors{false};
  int argc{0};
  const char** argv{nullptr};
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
    suites_.emplace_back(suite.run, suite.name);
  }

  template <class... Ts>
  auto on(events::test<Ts...> test) {
    path_[level_] = test.name;

    if (detail::cfg::list_tags) {
      std::for_each(test.tag.cbegin(), test.tag.cend(), [](const auto& tag) {
        std::cout << "tag: " << tag << std::endl;
      });
      return;
    }

    auto execute = std::empty(test.tag);
    for (const auto& tag_element : test.tag) {
      if (utility::is_match(tag_element, "skip") && !detail::cfg::show_tests &&
          !detail::cfg::show_test_names) {
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

    if (!detail::cfg::query_pattern.empty()) {
      const static auto regex = detail::cfg::query_regex_pattern;
      bool matches = utility::regex_match(test.name.data(), regex.c_str());
      for (const auto& tag2 : test.tag) {
        matches |= utility::regex_match(tag2.data(), regex.c_str());
      }
      if (matches) {
        execute = !detail::cfg::invert_query_pattern;
      } else {
        execute = detail::cfg::invert_query_pattern;
      }
    }

    if (detail::cfg::show_tests || detail::cfg::show_test_names) {
      if (!detail::cfg::show_test_names) {
        std::cout << "matching test: ";
      }
      std::cout << test.name << std::endl;
      return;
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

      if (not--level_) {
        reporter_.on(events::test_end{.type = test.type, .name = test.name});
      } else {  // N.B. prev. only root-level tests were signalled on finish
        if constexpr (requires {
                        reporter_.on(events::test_finish{.type = test.type,
                                                         .name = test.name});
                      }) {
          reporter_.on(
              events::test_finish{.type = test.type, .name = test.name});
        }
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
    reporter_.on(events::run_begin{.argc = rc.argc, .argv = rc.argv});
    for (const auto& [suite, suite_name] : suites_) {
      // add reporter in/out
      if constexpr (requires { reporter_.on(events::suite_begin{}); }) {
        reporter_.on(events::suite_begin{.type = "suite", .name = suite_name});
      }
      suite();
      if constexpr (requires { reporter_.on(events::suite_end{}); }) {
        reporter_.on(events::suite_end{.type = "suite", .name = suite_name});
      }
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
  std::vector<std::pair<void (*)(), std::string_view>> suites_{};
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
//[[maybe_unused]] inline auto cfg = runner<reporter<printer>>{};// alt reporter
[[maybe_unused]] inline auto cfg = runner<reporter_junit<printer>>{};

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
                                       .name = std::string{name},
                                       .tag = tag,
                                       .location = _test.location,
                                       .arg = none{},
                                       .run = _test.test});
    return _test.test;
  }

  template <class Test>
    requires std::invocable<Test> && (!std::convertible_to<Test, void (*)()>)
  constexpr auto operator=(Test _test) {
    on<Test>(events::test<Test>{.type = type,
                                .name = std::string{name},
                                .tag = tag,
                                .location = {},
                                .arg = none{},
                                .run = static_cast<Test&&>(_test)});
    return _test;
  }

  constexpr void operator=(void (*_test)(std::string_view,
                                         std::string_view)) const {
    _test(type, name);
  }

  template <class Test>
    requires std::invocable<Test, std::string_view, std::string_view> &&
             (!std::convertible_to<Test, void (*)(std::string_view,
                                                  std::string_view)>)
  constexpr auto operator=(Test _test) {
    return _test(type, name);
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

#if defined(BOOST_UT_HAS_FORMAT)
#if __cpp_lib_format >= 202207L
  template <class... Args>
  void operator()(std::format_string<Args...> fmt, Args&&... args) {
    on<std::string>(
        events::log{std::vformat(fmt.get(), std::make_format_args(args...))});
  }
#else
  template <class... Args>
  void operator()(std::string_view fmt, Args&&... args) {
    on<std::string>(
        events::log{std::vformat(fmt, std::make_format_args(args...))});
  }
#endif
#endif
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
      if constexpr (requires {
                      requires std::invocable<TMsg> and
                                   not std::is_void_v<
                                       std::invoke_result_t<TMsg>>;
                    }) {
        on<T>(events::log{std::invoke(msg)});
      } else {
        on<T>(events::log{msg});
      }
    }
    return *this;
  }

  auto& operator<<(detail::fatal) {
    if (not value_) {
      on<T>(events::fatal_assertion{});
    }
    return *this;
  }

  [[nodiscard]] constexpr operator bool() const { return value_; }

  bool value_{};
};
}  // namespace detail

namespace literals {
[[nodiscard]] inline auto operator""_test(const char* name, std::size_t size) {
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

[[nodiscard]] constexpr auto get_ordinal_suffix(int number) {
  // See https://stackoverflow.com/a/13627586
  const auto last_digit = number % 10;
  const auto last_two_digits = number % 100;
  if (last_digit == 1 && last_two_digits != 11) {
    return "st";
  }
  if (last_digit == 2 && last_two_digits != 12) {
    return "nd";
  }
  if (last_digit == 3 && last_two_digits != 13) {
    return "rd";
  }
  return "th";
};

template <class TArg>
inline std::string format_test_parameter([[maybe_unused]] const TArg& arg,
                                         const int counter) {
  return std::to_string(counter) + get_ordinal_suffix(counter) + " parameter";
}

template <class F>
  requires(std::integral<F> || std::floating_point<F>) &&
          (!std::same_as<F, bool>)
inline std::string
    format_test_parameter(const F& arg, [[maybe_unused]] const int counter) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

inline std::string format_test_parameter(const bool& arg,
                                         [[maybe_unused]] const int counter) {
  return arg ? "true" : "false";
}

namespace operators {
[[nodiscard]] constexpr auto operator==(std::string_view lhs,
                                        std::string_view rhs) {
  return detail::eq_{lhs, rhs};
}

[[nodiscard]] constexpr auto operator!=(std::string_view lhs,
                                        std::string_view rhs) {
  return detail::neq_{lhs, rhs};
}

template <std::ranges::range T>
[[nodiscard]] constexpr auto operator==(T&& lhs, T&& rhs) {
  return detail::eq_{static_cast<T&&>(lhs), static_cast<T&&>(rhs)};
}

template <std::ranges::range T>
[[nodiscard]] constexpr auto operator!=(T&& lhs, T&& rhs) {
  return detail::neq_{static_cast<T&&>(lhs), static_cast<T&&>(rhs)};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator==(const TLhs& lhs, const TRhs& rhs) {
  return detail::eq_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator!=(const TLhs& lhs, const TRhs& rhs) {
  return detail::neq_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator>(const TLhs& lhs, const TRhs& rhs) {
  return detail::gt_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator>=(const TLhs& lhs, const TRhs& rhs) {
  return detail::ge_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator<(const TLhs& lhs, const TRhs& rhs) {
  return detail::lt_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator<=(const TLhs& lhs, const TRhs& rhs) {
  return detail::le_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator and(const TLhs& lhs, const TRhs& rhs) {
  return detail::and_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
[[nodiscard]] constexpr auto operator or(const TLhs& lhs, const TRhs& rhs) {
  return detail::or_{lhs, rhs};
}

template <class T>
  requires type_traits::is_op<T>
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
  std::vector<std::string_view> tag;
  tag.reserve(lhs.name.size() + rhs.name.size());
  for (const auto& name : lhs.name) {
    tag.push_back(name);
  }
  for (const auto& name : rhs.name) {
    tag.push_back(name);
  }
  return detail::tag{tag};
}

template <class F, class T>
  requires std::ranges::range<T>
[[nodiscard]] constexpr auto operator|(const F& f, const T& t) {
  return [f, t](std::string_view type, std::string_view name) {
    for (int counter = 1; const auto& arg : t) {
      detail::on<F>(events::test<F, decltype(arg)>{
          .type = type,
          .name = std::string{name} + " (" +
                  format_test_parameter(arg, counter) + ")",
          .tag = {},
          .location = {},
          .arg = arg,
          .run = f});
      ++counter;
    }
  };
}

template <class F, template <class...> class T, class... Ts>
  requires(!std::ranges::range<T<Ts...>>)
[[nodiscard]] constexpr auto operator|(const F& f, const T<Ts...>& t) {
  constexpr auto unique_name = []<class TArg>(std::string_view name,
                                              const TArg& arg, int& counter) {
    auto ret = std::string{name} + " (";
    if (std::invocable<F, TArg>) {
      ret += format_test_parameter(arg, counter) + ", ";
    }
    ret += std::string(reflection::type_name<TArg>()) + ")";
    ++counter;
    return ret;
  };

  return [f, t, unique_name](std::string_view type, std::string_view name) {
    int counter = 1;
    apply(
        [=, &counter](const auto&... args) {
          (detail::on<F>(events::test<F, Ts>{
               .type = type,
               .name = unique_name.template operator()<Ts>(name, args, counter),
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

[[maybe_unused]] constexpr struct placeholder_gcc_t {
} _t;

template <class T>
constexpr auto operator%(const T& t, const placeholder_gcc_t&) {
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class T>
  requires type_traits::is_op<T>
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

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
constexpr auto operator and(const TLhs& lhs, const TRhs& rhs) {
  using and_t = detail::and_<typename TLhs::type, typename TRhs::type>;
  struct and_ : and_t, detail::log {
    using type = and_t;
    using and_t::and_t;
    const detail::terse_<type> _{*this};
  };
  return and_{lhs, rhs};
}

template <class TLhs, class TRhs>
  requires type_traits::is_op<TLhs> || type_traits::is_op<TRhs>
constexpr auto operator or(const TLhs& lhs, const TRhs& rhs) {
  using or_t = detail::or_<typename TLhs::type, typename TRhs::type>;
  struct or_ : or_t, detail::log {
    using type = or_t;
    using or_t::or_t;
    const detail::terse_<type> _{*this};
  };
  return or_{lhs, rhs};
}

template <class T>
  requires type_traits::is_op<T>
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

template <class TExpr>
  requires type_traits::is_op<TExpr> ||
           concepts::explicitly_convertible_to<TExpr, bool>
constexpr auto expect(const TExpr& expr,
                      const reflection::source_location& sl =
                          reflection::source_location::current()) {
  return detail::expect_<TExpr>{detail::on<TExpr>(
      events::assertion<TExpr>{.expr = expr, .location = sl})};
}

[[maybe_unused]] inline constexpr auto fatal = detail::fatal{};

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

template <fixed_string suite_name = "unnamed suite">
struct suite {
  reflection::source_location location{};
  std::string_view name = std::string_view(suite_name);
  template <class TSuite>
  constexpr /*explicit(false)*/ suite(TSuite _suite) {
    static_assert(1 == sizeof(_suite));
    detail::on<decltype(+_suite)>(
        events::suite<decltype(+_suite)>{.run = +_suite, .name = name});
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
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto eq(const TLhs& lhs, const TRhs& rhs) {
  return detail::eq_{lhs, rhs};
}
template <class TLhs, class TRhs, class TEpsilon>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto approx(const TLhs& lhs, const TRhs& rhs,
                                    const TEpsilon& epsilon) {
  return detail::approx_{lhs, rhs, epsilon};
}
template <class TLhs, class TRhs>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto neq(const TLhs& lhs, const TRhs& rhs) {
  return detail::neq_{lhs, rhs};
}
template <class TLhs, class TRhs>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto gt(const TLhs& lhs, const TRhs& rhs) {
  return detail::gt_{lhs, rhs};
}
template <class TLhs, class TRhs>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto ge(const TLhs& lhs, const TRhs& rhs) {
  return detail::ge_{lhs, rhs};
}
template <class TLhs, class TRhs>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
[[nodiscard]] constexpr auto lt(const TLhs& lhs, const TRhs& rhs) {
  return detail::lt_{lhs, rhs};
}
template <class TLhs, class TRhs>
  requires type_traits::is_stream_insertable_v<TLhs> &&
           type_traits::is_stream_insertable_v<TRhs>
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
}  // namespace boost::inline ext::ut::inline v2_3_1

#if (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)) && \
    !defined(__EMSCRIPTEN__)
__attribute__((constructor(101))) inline void cmd_line_args(
    int argc, const char* argv[]) {
  ::boost::ut::detail::cfg::largc = argc;
  ::boost::ut::detail::cfg::largv = argv;
}
#else
// For MSVC, largc/largv are initialized with __argc/__argv
#endif

#if defined(_MSC_VER)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif

#endif


#if __has_include(<unistd.h>)  && __has_include(<sys/ioctl.h>)  && __has_include(<sys/syscall.h>) && __has_include(<linux/perf_event.h>) && !defined(BENCHMARK_NO_PERF_COUNTER)
#define HAS_LINUX_PERFORMANCE_HEADER
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace benchmark {
#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__                __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

#if defined(__x86_64__) or defined(__i686__)
#define SIMD_REG "x,"
#else
#define SIMD_REG
#endif

/**
 * Tell the compiler that all arguments to this function are read and modified in the most
 * efficient way possible. This may force a value to memory, but generally tries to avoid doing so.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void fake_modify(T& x, Ts&... more) {
#ifdef __GNUC__
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        asm volatile("" : "+" SIMD_REG "g,m"(x));
    } else {
        asm volatile("" : "+g,m"(x));
    }
#else
    const volatile T y = x;
    x                  = y;
#endif
    (fake_modify(more), ...);
}

/**
 * Tell the compiler that all arguments to this function are read in the most efficient way
 * possible. This may force a value to memory, but generally tries to avoid doing so.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void fake_read(const T& x, const Ts&... more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" ::SIMD_REG "g,m"(x));
    } else {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" ::"g,m"(x));
    }
#else
    const volatile T y = x;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        fake_read(more...);
    }
}

/**
 * Tell the compiler that all arguments to this function must be stored to memory (stack) and
 * reloaded before reading.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void force_to_memory(T& x, Ts&... more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    // NOLINTNEXTLINE(hicpp-no-assembler)
    asm volatile("" : "+m"(x));
#else
    const volatile T y = x;
    x                  = y;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        force_to_memory(more...);
    }
}

/**
 * Tell the compiler that all arguments to this function must be stored to memory (stack).
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void force_store(const T& x, const Ts&... more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    // NOLINTNEXTLINE(hicpp-no-assembler)
    asm volatile("" ::"m"(x));
#else
    const volatile T y = x;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        force_store(more...);
    }
}

#undef SIMD_REG

struct perf_sub_metric {
    uint64_t misses{0};
    uint64_t total{0};
    double   ratio{0.0};
};

struct perf_metric {
    perf_sub_metric cache;
    perf_sub_metric branch;
    uint64_t        instructions{0};
    uint64_t        ctx_switches{0};
};

#ifdef HAS_LINUX_PERFORMANCE_HEADER
template<__u32 TypeId, __u64 ConfigId>
class PerfEventHandler {
    int _fd = -1;

    constexpr static std::string_view _sysErrorMessage = R"(You may not have permission to collect perf stats data.
Consider tweaking /proc/sys/kernel/perf_event_paranoid:
 -1 - Not paranoid at all
  0 - Disallow raw tracepoint access for unpriv
  1 - Disallow cpu events for unpriv
  2 - Disallow kernel profiling for unpriv
quick_fix: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
for details see: https://www.kernel.org/doc/Documentation/sysctl/kernel.txt)";

    constexpr static std::string_view eventIdToStringView() noexcept {
        if constexpr (TypeId == PERF_TYPE_HARDWARE) {
            switch (ConfigId) {
            case PERF_COUNT_HW_CPU_CYCLES: return "CPU cycles";
            case PERF_COUNT_HW_INSTRUCTIONS: return "Instructions";
            case PERF_COUNT_HW_CACHE_REFERENCES: return "Cache references";
            case PERF_COUNT_HW_CACHE_MISSES: return "Cache misses";
            case PERF_COUNT_HW_BRANCH_INSTRUCTIONS: return "Branch instructions";
            case PERF_COUNT_HW_BRANCH_MISSES: return "Branch misses";
            case PERF_COUNT_HW_BUS_CYCLES: return "Bus cycles";
            case PERF_COUNT_HW_STALLED_CYCLES_FRONTEND: return "Stalled cycles frontend";
            case PERF_COUNT_HW_STALLED_CYCLES_BACKEND: return "Stalled cycles backend";
            case PERF_COUNT_HW_REF_CPU_CYCLES: return "Ref CPU cycles";
            default: return "Unknown hardware config";
            }
        } else if constexpr (TypeId == PERF_TYPE_SOFTWARE) {
            switch (ConfigId) {
            case PERF_COUNT_SW_CONTEXT_SWITCHES: return "Context switches";
            case PERF_COUNT_SW_CPU_MIGRATIONS: return "CPU migrations";
            case PERF_COUNT_SW_PAGE_FAULTS: return "Page faults";
            case PERF_COUNT_SW_TASK_CLOCK: return "Task clock";
            default: return "Unknown software config";
            }
        } else {
            return "Unknown type";
        }
    }

    static void printAccessRightMsg(std::string_view method = "") noexcept {
        try {
            if (errno != 0) {
                std::println(stderr, "PerformanceCounter: could not perform {} in method '{}' - error {}: '{}'", eventIdToStringView(), method, errno, strerror(errno));
            } else {
                std::println(stderr, "PerformanceCounter: could not perform {} in method '{}'", eventIdToStringView(), method);
            }
            std::println(stderr, _sysErrorMessage);
        } catch (const std::exception& e) {
            std::cerr << "Error during logging: " << e.what() << '\n';
        } catch (...) {
            std::cerr << "Unknown error during logging\n";
        }

        std::cerr.flush();
        std::cout.flush();
    }

public:
    PerfEventHandler() = default;

    PerfEventHandler(perf_event_attr& attr, int pid, int cpu, unsigned long flags, int group_fd = -1) {
        attr.type   = TypeId;
        attr.config = ConfigId;
        _fd         = static_cast<int>(syscall(SYS_perf_event_open, &attr, pid, cpu, group_fd, flags));
        if (_fd < -1) {
            printAccessRightMsg();
        }
    }

    ~PerfEventHandler() {
        disable();
        if (_fd != -1) {
            close(_fd);
            _fd = -1;
        }
    }

    PerfEventHandler(const PerfEventHandler&)            = delete;
    PerfEventHandler& operator=(const PerfEventHandler&) = delete;
    PerfEventHandler(PerfEventHandler&& other) noexcept : _fd(other._fd) { other._fd = -1; }
    PerfEventHandler& operator=(PerfEventHandler&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        _fd       = other._fd;
        other._fd = -1;
        return *this;
    }

    int getFd() const noexcept { return _fd; }

    [[nodiscard]] bool isValid() const noexcept { return _fd != -1; }

    void enable() {
        if (_fd != -1 && ioctl(_fd, PERF_EVENT_IOC_ENABLE) == -1) {
            _fd = -1;
            printAccessRightMsg("PERF_EVENT_IOC_ENABLE");
        }
    }

    void disable() {
        if (_fd != -1 && ioctl(_fd, PERF_EVENT_IOC_DISABLE) == -1) {
            _fd = -1;
            printAccessRightMsg("PERF_EVENT_IOC_DISABLE");
        }
    }
};

/**
 * A short and sweet performance counter (only works on Linux)
 */
class PerformanceCounter {
    static bool _has_required_rights;

    PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES>        _fd_misses;
    PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES>    _fd_accesses;
    PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES>       _fd_branch_misses;
    PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS> _fd_branch;
    PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS>        _fd_instructions;
    PerfEventHandler<PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES>    _fd_ctx_switches;

public:
    PerformanceCounter() {
        constexpr static int PROCESS = 0; // 0: calling process
        constexpr static int ANY_CPU = -1;
        constexpr static int FLAGS   = PERF_FLAG_FD_CLOEXEC;
        if (!_has_required_rights) {
            return;
        }

        perf_event_attr attr{};
        attr.disabled   = 1;
        attr.exclude_hv = 1;

        // cache prediction metric
        _fd_misses = PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES>(attr, PROCESS, ANY_CPU, FLAGS, -1);
        if (_fd_misses.isValid()) {
            _fd_accesses = PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES>(attr, PROCESS, ANY_CPU, FLAGS, _fd_misses.getFd());
        } else {
            _has_required_rights = false;
            return;
        }

        // branch prediction metric
        _fd_branch_misses = PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES>(attr, PROCESS, ANY_CPU, FLAGS);
        if (_fd_branch_misses.isValid()) {
            _fd_branch = PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS>(attr, PROCESS, ANY_CPU, FLAGS, _fd_branch_misses.getFd());
        } else {
            _has_required_rights = false;
            return;
        }

        // instruction count metric
        _fd_instructions = PerfEventHandler<PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS>(attr, PROCESS, ANY_CPU, FLAGS, _fd_misses.getFd());
        if (!_fd_instructions.isValid()) {
            _has_required_rights = false;
            return;
        }

        // ctx switch count metric
        _fd_ctx_switches = PerfEventHandler<PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES>(attr, PROCESS, ANY_CPU, FLAGS, _fd_misses.getFd());
        if (!_fd_instructions.isValid()) {
            _has_required_rights = false;
            return;
        }

        // Enable all valid file descriptors
        _fd_misses.enable();
        _fd_accesses.enable();
        _fd_branch_misses.enable();
        _fd_branch.enable();
        _fd_instructions.enable();
        _fd_ctx_switches.enable();
    }

    PerformanceCounter(const PerformanceCounter&)            = delete;
    PerformanceCounter& operator=(const PerformanceCounter&) = delete;

    [[nodiscard]] static bool available() noexcept { return _has_required_rights; }

    [[nodiscard]] auto results() const noexcept -> perf_metric {
        if (!_has_required_rights) {
            return {};
        }
        perf_metric           ret;
        constexpr static auto readMetric = [](const auto& metric_fd, auto& data) noexcept -> bool { return metric_fd.isValid() && read(metric_fd.getFd(), &data, sizeof(data)) == sizeof(data); };
        if (!readMetric(_fd_misses, ret.cache.misses) || !readMetric(_fd_accesses, ret.cache.total) || !readMetric(_fd_branch_misses, ret.branch.misses) || !readMetric(_fd_branch, ret.branch.total) || !readMetric(_fd_instructions, ret.instructions) || !readMetric(_fd_ctx_switches, ret.ctx_switches)) {
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
    [[nodiscard]] constexpr static bool available() noexcept { return false; }

    /**
     * This OS is not supported
     */
    [[nodiscard]] auto results() const noexcept -> perf_metric { return {}; }
};

#endif

namespace ut = boost::ut;

/**
 * little compile-time string class (N.B. ideally std::string should become constexpr (everything!!
 * ;-)))
 */
template<typename CharT, std::size_t SIZE>
struct fixed_string {
    constexpr static std::size_t N            = SIZE;
    CharT                        _data[N + 1] = {};

    constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
        if constexpr (N != 0) {
            for (std::size_t i = 0; i < N; ++i) {
                _data[i] = str[i];
            }
        }
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }

    [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }

    [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }

    [[nodiscard]] explicit operator std::string() const noexcept { return {_data, N}; }

    [[nodiscard]] constexpr operator const char*() const noexcept { return _data; }

    [[nodiscard]] constexpr bool operator==(const fixed_string& other) const noexcept { return std::string_view{_data, N} == std::string_view(other); }

    template<std::size_t N2>
    [[nodiscard]] friend constexpr bool operator==(const fixed_string&, const fixed_string<CharT, N2>&) {
        return false;
    }
};

template<typename CharT, std::size_t N>
fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

template<fixed_string... key>
constexpr bool key_not_found = false;

/**
 * constexpr const key map that allows mdification of the values during run-time while the
 * compile-time look up of keys is free
 *
 * @author: Ralph J. Steinhagen
 */
template<typename Value, fixed_string... Keys>
class const_key_map {
    constexpr static std::size_t                              SIZE  = sizeof...(Keys);
    constexpr static std::array<const std::string_view, SIZE> _keys = {std::string_view(Keys)...};
    std::array<Value, SIZE>                                   _storage;

    template<fixed_string key, std::size_t Index = 0>
    constexpr static std::size_t get_index_by_name() noexcept {
        if constexpr (Index == SIZE) {
            static_assert(key_not_found<key>, "key not found");
            return SIZE; // This line is never reached but added to silence compiler warnings
        } else if constexpr (_keys[Index] == std::string_view(key)) {
            return Index;
        } else {
            return get_index_by_name<key, Index + 1>();
        }
    }

    constexpr static std::size_t get_index_by_name(std::string_view key) {
        if (const auto itr = std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto& v) { return v == std::string_view(key); }); itr != std::cend(_keys)) {
            return std::distance(std::cbegin(_keys), itr);
        } else {
            throw std::range_error("key not found");
        }
    }

    constexpr static std::size_t get_index_by_ID(const std::size_t key_ID) {
        if (key_ID < SIZE) {
            return key_ID;
        } else {
            throw std::range_error("key ID not found");
        }
    }

public:
    [[nodiscard]] static constexpr std::size_t size() noexcept { return SIZE; }

    [[nodiscard]] std::string_view key(std::size_t key_ID) const { return _keys[get_index_by_ID(key_ID)]; }

    template<fixed_string key>
    [[nodiscard]] constexpr Value const& at() const {
        return _storage[get_index_by_name<key>()];
    }

    template<fixed_string key>
    [[nodiscard]] constexpr Value& at() {
        return _storage[get_index_by_name<key>()];
    }

    [[nodiscard]] constexpr Value const& at(const std::string_view key) const { return _storage[get_index_by_name(key)]; }

    [[nodiscard]] constexpr Value& at(const std::string_view key) { return _storage[get_index_by_name(key)]; }

    [[nodiscard]] constexpr Value const& at(const std::size_t key_ID) const { return _storage[get_index_by_ID(key_ID)]; }

    [[nodiscard]] constexpr Value& at(const std::size_t key_ID) { return _storage[get_index_by_ID(key_ID)]; }

    [[nodiscard]] constexpr bool contains(const std::string_view key) const {
        return std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto& v) { return v == key; }) != std::cend(_keys);
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

    std::size_t operator()(std::string_view sv) const {
        std::hash<std::string_view> hasher;
        return hasher(sv);
    }
};

using ResultMap = std::unordered_map<std::string, std::tuple<std::variant<std::monostate, long double, uint64_t, perf_sub_metric>, std::string, std::size_t>, StringHash, std::equal_to<>>;

class results {
    using EntryType = std::pair<std::string, ResultMap>;
    using Data      = std::vector<EntryType>;
    static std::mutex _lock;
    static Data       _data;

public:
    [[nodiscard]] static ResultMap& add_result(std::string_view name) noexcept {
        std::lock_guard guard(_lock);
        return _data.emplace_back(std::string(name), ResultMap()).second;
    }

    static void add_separator() noexcept {
        std::lock_guard guard(_lock);
        _data.emplace_back(std::string{}, ResultMap{});
    }

    [[nodiscard]] static constexpr Data const& data() noexcept { return _data; };
};

inline results::Data results::_data;
inline std::mutex    results::_lock;

class time_point {
    using timePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using timeDiff  = std::chrono::nanoseconds;
    timePoint _time_point;

public:
    time_point& now() noexcept {
        _time_point = std::chrono::high_resolution_clock::now();
        return *this;
    }

    timeDiff operator-(const time_point& start_marker) const noexcept { return std::chrono::duration_cast<std::chrono::nanoseconds>(_time_point - start_marker._time_point); }
};

namespace utils {

template<std::size_t N, typename T>
requires(N > 0)
constexpr std::vector<T> diff(const std::vector<time_point>& stop, time_point start) {
    std::vector<T> ret(N);
    for (auto i = 0LU; i < N; i++) {
        ret[i] = 1e-9L * static_cast<T>((stop[i] - start).count());
        start  = stop[i];
    }
    return ret;
}

template<std::size_t N, typename T>
requires(N > 0)
constexpr std::vector<T> diff(const std::vector<time_point>& stop, const std::vector<time_point>& start) {
    std::vector<T> ret(N);
    for (auto i = 0LU; i < N; i++) {
        ret[i] = 1e-9L * static_cast<T>((stop[i] - start[i]).count());
    }
    return ret;
}

template<std::size_t n_iterations, typename MapType>
requires(n_iterations > 0)
auto convert(const std::vector<MapType>& in) {
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
[[nodiscard]] StatisticsType<T> compute_statistics(const std::vector<T>& values) {
    const std::size_t N = values.size();
    if (N < 1) {
        return {};
    }
    const auto [min, max] = std::ranges::minmax_element(values);
    const auto mean       = std::accumulate(values.begin(), values.end(), T{}) / static_cast<T>(N);

    T stddev{};
    std::ranges::for_each(values, [&](const auto x) { stddev += (x - mean) * (x - mean); });
    stddev /= static_cast<T>(N);
    stddev = std::sqrt(stddev);

    // Compute the median value
    std::vector<T> sorted_values(values);
    std::ranges::sort(sorted_values);
    const auto median = sorted_values[N / 2];
    return {*min, mean, stddev, median, *max};
}

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
std::string to_si_prefix(T value_base, std::string_view unit = "s", std::size_t significant_digits = 0) {
    static constexpr std::array  si_prefixes{'q', 'r', 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y', 'R', 'Q'};
    static constexpr long double base  = 1000.0L;
    auto                         value = static_cast<long double>(value_base);

    std::size_t exponent = 10U;
    if (value == 0.0L) {
        return std::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
    }
    while (value >= base && exponent < si_prefixes.size()) {
        value /= base;
        ++exponent;
    }
    while (value < 1.0L && exponent > 0U) {
        value *= base;
        --exponent;
    }
    if (significant_digits == 0 && exponent > 10) {
        if (value < 10.0L) {
            significant_digits = 2u;
        } else if (value < 100.0L) {
            significant_digits = 1u;
        }
    } else if (significant_digits == 1 && value >= 100.0L) {
        --significant_digits;
    } else if (significant_digits >= 2U) {
        if (value >= 100.0L) {
            significant_digits -= 2U;
        } else if (value >= 10.0L) {
            --significant_digits;
        }
    }

    return std::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ", si_prefixes[exponent], unit);
}

} // namespace utils

namespace detail {

template<typename T>
struct class_with_call_operator;

// For const operator()
template<typename ClassType, typename ResultType, typename... Args>
struct class_with_call_operator<ResultType (ClassType::*)(Args...) const> {
    constexpr static std::size_t arity = sizeof...(Args);
    using result_type                  = ResultType;
    using args_tuple                   = std::tuple<Args...>;
};

// For non-const operator()
template<typename ClassType, typename ResultType, typename... Args>
struct class_with_call_operator<ResultType (ClassType::*)(Args...)> {
    constexpr static std::size_t arity = sizeof...(Args);
    using result_type                  = ResultType;
    using args_tuple                   = std::tuple<Args...>;
};

template<typename T>
struct fn_traits;

template<typename ClassType, typename ResultType, typename... Args>
struct fn_traits<ResultType (ClassType::*)(Args...) const> {
    constexpr static std::size_t arity = sizeof...(Args);
    using result_type                  = ResultType;
    using args_tuple                   = std::tuple<Args...>;
};

template<typename ResultType, typename... Args>
struct fn_traits<ResultType(Args...)> {
    constexpr static std::size_t arity = sizeof...(Args);
    using result_type                  = ResultType;
    using args_tuple                   = std::tuple<Args...>;
};

template<typename T>
struct fn_traits : detail::class_with_call_operator<decltype(&T::operator())> {};

template<typename T>
using return_type_of_t = typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::result_type;

template<typename T>
using first_arg_of_t = std::tuple_element_t<0, typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::args_tuple>;
} // namespace detail

template<typename TestFunction>
constexpr std::size_t argument_size() noexcept {
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
struct MarkerMap : const_key_map<time_point, meas_marker_names...> {};

template<typename TestFunction, std::size_t N>
[[nodiscard]] constexpr auto get_marker_array() {
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
    int         _precision = 0;

public:
    benchmark() = delete;

    explicit benchmark(std::string_view _name, std::size_t n_scale_results = 1LU) : ut::detail::test{"benchmark", _name}, _n_scale_results(n_scale_results) {
        const char* ptr = std::getenv("BM_DIGITS");
        if (ptr != nullptr) {
            const std::string_view env{ptr};
            auto [_, ec] = std::from_chars(env.data(), env.data() + env.size(), _precision);
            if (ec == std::errc()) {
                _precision = std::clamp(_precision, 1, 6) - 1;
            } else {
                std::print("Invalid value for BM_DIGITS: '{}'\n", env);
            }
        }
    }

    template<class TestFunction, std::size_t MARKER_SIZE = argument_size<TestFunction>(), bool has_arguments = MARKER_SIZE != 0>
    // template<fixed_string ...meas_marker, Callback<meas_marker...> Test>
    constexpr benchmark& operator=(TestFunction&& _test) {
        static_cast<ut::detail::test&>(*this) = [&_test, this] {
            auto& result_map = ResultType::add_result(name);
            if constexpr (N_ITERATIONS != 1) {
                result_map.try_emplace("#N", N_ITERATIONS, "", 0);
            } else {
                result_map.try_emplace("#N", std::monostate{}, "", 0);
            }

            std::vector<time_point> stop_iter(N_ITERATIONS);
            auto                    marker_iter = get_marker_array<TestFunction, N_ITERATIONS>();

            PerformanceCounter execMetrics;
            const auto         start = time_point().now();

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
                result_map.try_emplace("<CPU-I>", static_cast<double>(perf_data.instructions) / static_cast<double>(N_ITERATIONS * _n_scale_results), "", std::max(1, _precision));
                result_map.try_emplace("CTX-SW", perf_data.ctx_switches, "", 0);
            }
            // not time-critical post-processing starts here
            const auto        time_differences_ns = utils::diff<N_ITERATIONS, long double>(stop_iter, start);
            const auto        ns                  = stop_iter[N_ITERATIONS - 1] - start;
            const long double duration_s          = 1e-9L * static_cast<long double>(ns.count());

            const auto add_statistics = [&]<typename T>(ResultMap& map, const T& time_diff) {
                if constexpr (N_ITERATIONS != 1) {
                    const auto [min, mean, stddev, median, max] = utils::compute_statistics(time_diff);
                    map.try_emplace("min", min, "s", _precision);
                    map.try_emplace("mean", mean, "s", _precision);
                    if (stddev == 0) {
                        map.try_emplace("stddev", std::monostate{}, "s", _precision);
                    } else {
                        map.try_emplace("stddev", stddev, "s", _precision);
                    }
                    map.try_emplace("median", median, "s", _precision);
                    map.try_emplace("max", max, "s", _precision);
                } else {
                    map.try_emplace("min", std::monostate{}, "s", _precision);
                    map.try_emplace("mean", duration_s / N_ITERATIONS, "s", _precision);
                    map.try_emplace("stddev", std::monostate{}, "s", _precision);
                    map.try_emplace("median", std::monostate{}, "s", _precision);
                    map.try_emplace("max", std::monostate{}, "s", _precision);
                }
            };
            add_statistics(result_map, time_differences_ns);

            result_map.try_emplace("total time", duration_s, "s", _precision);
            result_map.try_emplace("ops/s", static_cast<long double>(_n_scale_results * N_ITERATIONS) / duration_s, "", std::max(1, _precision));

            if constexpr (MARKER_SIZE > 0) {
                auto transposed_map = utils::convert<N_ITERATIONS>(marker_iter);
                for (auto keyID = 0LU; keyID < transposed_map.size(); keyID++) {
                    if (keyID > 0) {
                        const auto meas = std::format("  {}─Marker{}: '{}'→'{}' ", //
                            keyID < transposed_map.size() - 1 ? "├" : "└", keyID, transposed_map[0].first, transposed_map[keyID].first);

                        auto& marker_result_map = ResultType::add_result(meas);

                        const auto marker_latencies = utils::diff<N_ITERATIONS, long double>(transposed_map[keyID].second, transposed_map[0].second);

                        if constexpr (N_ITERATIONS > 1) {
                            add_statistics(marker_result_map, marker_latencies);
                        } else {
                            // even with N_ITER=1: show "min" and "mean" as the actual latency
                            marker_result_map.try_emplace("min", marker_latencies[0], "s", _precision);
                            marker_result_map.try_emplace("mean", marker_latencies[0], "s", _precision);
                            marker_result_map.try_emplace("stddev", std::monostate{}, "s", _precision);
                            marker_result_map.try_emplace("median", std::monostate{}, "s", _precision);
                            marker_result_map.try_emplace("max", marker_latencies[0], "s", _precision);
                        }
                    }
                }
            }
        };
        return *this;
    }

    template<std::size_t N>
    auto repeat(std::size_t n_scale_results = 1LU) {
        return ::benchmark::benchmark<N, ResultType, meas_marker_names...>(this->name, n_scale_results);
    }
};

[[nodiscard]] auto operator""_benchmark(const char* name, std::size_t size) { return ::benchmark::benchmark<1LU>{{name, size}}; }

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
    constexpr reporter& operator=(TPrinter printer) {
        _printer = std::move(printer);
        return *this;
    }

    constexpr void on(const ut::events::test_begin&) const noexcept { /* not needed */ }

    void on(const ut::events::test_run& test_run) { _printer << "\n \"" << test_run.name << "\"..."; }

    constexpr void on(const ut::events::test_skip& bench) const {
        std::cerr << std::format("SKIPPED - {}", bench.name) << std::endl;
        [[maybe_unused]] const auto& map = benchmark::results::add_result(bench.name);
    }

    void on(const ut::events::test_end& test_end) {
        if (_asserts.fail > 0) {
            ++_benchmarks.fail;
            _printer << _printer.colors().fail << std::format("... in benchmark '{}'", test_end.name) << _printer.colors().none << '\n';
            _asserts.fail--;
        }
    }

    template<class TMsg>
    void on(ut::events::log<TMsg> l) {
        _printer << l.msg;
    }

    void on(ut::events::exception exception) {
        _printer << std::format("\033[31munexpected exception: \"{}\"\n\033[0m", exception.what());
        ++_asserts.fail;
    }

    auto on(ut::events::run_begin) -> void {} // no-op (required for Boost.UT >= v2.3.1)

    template<class TExpr>
    void on(ut::events::assertion_pass<TExpr>) {
        ++_asserts.pass;
    }

    template<class TExpr>
    void on(ut::events::assertion_fail<TExpr> assertion) {
        constexpr auto short_name = [](std::string_view name) { return name.rfind('/') != std::string_view::npos ? name.substr(name.rfind('/') + 1) : name; };
        _printer << "\n  " << short_name(assertion.location.file_name()) << ':' << assertion.location.line() << ':' << _printer.colors().fail << "FAILED" << _printer.colors().none << " [" << std::boolalpha << assertion.expr << _printer.colors().none << ']';
        ++_asserts.fail;
    }

    void on(const ut::events::fatal_assertion&) const { /* not needed testing interface */ }

    void on(const ut::events::summary&) {
        if (_benchmarks.fail || _asserts.fail) {
            std::cout << _printer.str() << std::endl;
            std::cout << std::format("\033[31m{} micro-benchmark(s) failed:\n\033[m", _benchmarks.fail);
        } else {
            std::cout << _printer.colors().pass << "all micro-benchmarks passed:\n" << _printer.colors().none;
        }
        print();
        std::cerr.flush();
        std::cout.flush();
    }

    template<std::size_t SIGNIFICANT_DIGITS = 3>
    static void print() {
        const auto& data = benchmark::results::data();
        if (data.empty()) {
            std::print("no benchmark tests executed\n");
        }
        std::vector<std::size_t> v(data.size());
        // N.B. using <algorithm> rather than <ranges> to be compatible with libc/Emscripten
        // for details see: https://libcxx.llvm.org/Status/Ranges.html
        // not as of clang 15: https://compiler-explorer.com/z/8arxzodh3 ('trunk' seems to be OK)
        std::ranges::transform(data, v.begin(), [](auto val) { return val.first.size(); });

        const std::string test_case_label = "benchmark:";
        const std::size_t name_max_size   = std::max(*std::ranges::max_element(v), test_case_label.size()) + 1LU;

        const auto format = [](auto& value, const std::string& unit, std::size_t digits) -> std::string {
            using ::benchmark::utils::to_si_prefix;
            if (std::holds_alternative<std::monostate>(value)) {
                return "";
            } else if (std::holds_alternative<long double>(value)) {
                return std::format("{}", to_si_prefix(std::get<long double>(value), unit, digits));
            } else if (std::holds_alternative<uint64_t>(value)) {
                return std::format("{}", to_si_prefix(std::get<uint64_t>(value), unit, digits));
            } else if (std::holds_alternative<benchmark::perf_sub_metric>(value)) {
                const auto stat = std::get<benchmark::perf_sub_metric>(value);
                return std::format("{:>4} / {:>4} = {:4.1f}%", //
                    to_si_prefix(stat.misses, unit, 0), to_si_prefix(stat.total, unit, 0), 100.0 * stat.ratio);
            }
            throw std::invalid_argument("benchmark::results: unhandled ResultMap type");
        };

        // compute minimum colum width for each benchmark case and metric
        std::unordered_map<std::string, std::size_t> metric_keys;
        for (auto& [test_name, result_map] : data) {
            for (auto& [metric_key, value] : result_map) {
                const auto& [value_name, value_unit, value_digits] = value;
                if (!metric_keys.contains(metric_key)) {
                    metric_keys.try_emplace(metric_key, metric_key.size());
                }
                std::size_t value_size     = format(value_name, value_unit, value_digits).size();
                metric_keys.at(metric_key) = std::max(metric_keys.at(metric_key), value_size);
            }
        }
        v.clear();

        bool first_row = true;
        for (auto& [test_name, result_map] : data) {
            if (first_row) {
                std::print("┌{1:─^{0}}", name_max_size + 2UL, test_case_label);
                std::print("┬{1:─^{0}}", sizeof("PASS") + 1UL, "");
                for (auto const& [metric_key, max_width] : metric_keys) {
                    std::print("┬{1:─^{0}}", max_width + 2UL, metric_key);
                }
                std::print("┐\n");
                first_row = false;
            } else if (test_name.empty() and result_map.empty()) {
                std::print("├{1:─^{0}}", name_max_size + 2UL, "");
                std::print("┼{1:─^{0}}", sizeof("PASS") + 1UL, "");
                for (auto const& [metric_key, max_width] : metric_keys) {
                    std::print("┼{1:─^{0}}", max_width + 2UL, "");
                }
                std::print("┤\n");
                continue;
            }
            std::print("│ {1:<{0}} ", name_max_size, test_name);
            if (result_map.empty()) {
                std::print("│ \033[33mSKIP\033[0m ");
            } else if (result_map.size() == 1) {
                std::print("│ \033[31mFAIL\033[0m ");
            } else {
                std::print("│ \033[32mPASS\033[0m ");
            }

            for (auto& [metric_key, max_width] : metric_keys) {
                if (result_map.contains(metric_key)) {
                    const auto& [value, unit, digits] = result_map.at(metric_key);
                    std::print("│ {1:>{0}} ", max_width, format(value, unit, digits));
                } else {
                    std::print("│ {1:>{0}} ", max_width, "");
                }
            }
            std::print("│\n");
        }

        std::print("└{1:─^{0}}", name_max_size + 2UL, "");
        std::print("┴{1:─^{0}}", sizeof("PASS") + 1UL, "");
        for (auto const& [metric_key, max_width] : metric_keys) {
            std::print("┴{1:─^{0}}", max_width + 2UL, "");
        }
        std::print("┘\n");
    }
};
} // namespace cfg

template<>
inline auto boost::ut::cfg<boost::ut::override> = ut::runner<cfg::reporter<printer>>{};

#endif // GNURADIO_BENCHMARK_HPP
