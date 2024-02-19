#ifndef GNURADIO_BUFFER2_H
#define GNURADIO_BUFFER2_H

#include <bit>
#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>

namespace gr {
namespace util {
template<typename T, typename...>
struct first_template_arg_helper;

template<template<typename...> class TemplateType, typename ValueType, typename... OtherTypes>
struct first_template_arg_helper<TemplateType<ValueType, OtherTypes...>> {
    using value_type = ValueType;
};

template<typename T>
constexpr auto *
value_type_helper() {
    if constexpr (requires { typename T::value_type; }) {
        return static_cast<typename T::value_type *>(nullptr);
    } else {
        return static_cast<typename first_template_arg_helper<T>::value_type *>(nullptr);
    }
}

template<typename T>
using value_type_t = std::remove_pointer_t<decltype(value_type_helper<T>())>;

template<typename... A>
struct test_fallback {};

template<typename, typename ValueType>
struct test_value_type {
    using value_type = ValueType;
};

static_assert(std::is_same_v<int, value_type_t<test_fallback<int, float, double>>>);
static_assert(std::is_same_v<float, value_type_t<test_value_type<int, float>>>);
static_assert(std::is_same_v<int, value_type_t<std::span<int>>>);
static_assert(std::is_same_v<double, value_type_t<std::array<double, 42>>>);

} // namespace util

// TODO: find a better place for these 3 concepts
template<typename From, typename To>
concept convertible_from_lvalue_to = std::convertible_to<const std::remove_cvref_t<From> &, To>;

template<typename From, typename To>
concept convertible_from_rvalue_to = std::convertible_to<std::remove_cvref_t<From> &&, To>;

template<typename From, typename To>
concept convertible_from_lvalue_only = convertible_from_lvalue_to<From, To> && !convertible_from_rvalue_to<From, To>;

template<typename T>
concept SpanLike = std::convertible_to<T, std::span<std::remove_cvref_t<typename T::value_type>>>;

template<typename T>
concept ConstSpanLike = std::convertible_to<T, std::span<const std::remove_cvref_t<typename T::value_type>>>;

template<typename T>
concept ConstSpanLvalueLike = convertible_from_lvalue_only<T, std::span<const std::remove_cvref_t<typename T::value_type>>>;

template<typename T>
concept ConsumableSpan = std::ranges::contiguous_range<T> && ConstSpanLvalueLike<T> && requires(T &s) { s.consume(0); };

template<typename T>
concept PublishableSpan = std::ranges::contiguous_range<T> && std::ranges::output_range<T, std::remove_cvref_t<typename T::value_type>> && SpanLike<T> && requires(T &s) { s.publish(0UZ); };

// clang-format off
// disable formatting until clang-format (v16) supporting concepts
template<class T>
concept BufferReader = requires(T /*const*/ t, const std::size_t n_items) {
    { t.get(n_items) } ; // TODO: get() returns CircularBuffer::buffer_reader::ConsumableInputRange
    { t.position() }       -> std::same_as<std::make_signed_t<std::size_t>>;
    { t.available() }      -> std::same_as<std::size_t>;
    { t.buffer() };
};

template<class Fn, typename T, typename ...Args>
concept WriterCallback = std::is_invocable_v<Fn, std::span<T>&, std::make_signed_t<std::size_t>, Args...> || std::is_invocable_v<Fn, std::span<T>&, Args...>;

template<class T, typename ...Args>
concept BufferWriter = requires(T t, const std::size_t n_items, std::pair<std::size_t, std::make_signed_t<std::size_t>> token, Args ...args) {
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                                 -> std::same_as<void>;
    { t.publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::make_signed_t<std::size_t> /* writePos */, Args ...) { /* */  }, n_items, args...) }   -> std::same_as<void>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, Args ...) { /* */ }, n_items, args...) }                             -> std::same_as<bool>;
    { t.try_publish([](std::span<util::value_type_t<T>> &/*writable_data*/, std::make_signed_t<std::size_t> /* writePos */, Args ...) { /* */  }, n_items, args...) }-> std::same_as<bool>;
    { t.reserve(n_items) };// TODO: reserve() returns CircularBuffer::buffer_writer::PublishableOutputRange
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
using WithSequenceParameter = decltype([](std::span<T>&, std::make_signed_t<std::size_t>, Args...) { /* */ });
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
