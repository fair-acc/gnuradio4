#ifndef GNURADIO_IMMUTABLE_HPP
#define GNURADIO_IMMUTABLE_HPP

#include <format>
#include <type_traits>
#include <utility>

namespace gr::meta {

// const disables moved. immutable<T> is const by all means,
// but it allows the moved-from state.
template<typename T>
class immutable {
private:
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>, "T needs to be a value type");
    static_assert(std::is_move_constructible_v<T>, "T needs to be move-constuctible");
    static_assert(std::is_default_constructible_v<T>, "T needs to be default-constuctible");

    T _value;

public:
    using value_type = T;

    template<typename... Args>
    immutable(Args&&... args) : _value(std::forward<Args>(args)...) {}

    // Only construction is allowed
    immutable(immutable<T>&& other) : _value(std::move(other._value)) { other._value = T{}; }
    immutable(const immutable<T>& other) : _value(other._value) {}

    // No assignment, these are const values
    immutable<T>& operator=(const immutable<T>& other) = delete;
    immutable<T>& operator=(immutable<T>&& other)      = delete;

    const T& value() const { return _value; }

    operator const T&() const { return _value; }

    // Explicitly enable immutable<std::string> to std::string_view conversion
    // as string and string_views' relationship is special
    operator std::string_view() const
    requires(std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view>)
    {
        return _value;
    }

    auto operator<=>(const immutable<T>& other) const { return _value <=> other._value; }

    bool operator==(const immutable<T>& other) const = default;

    template<typename U>
    auto operator<=>(const U& other) const {
        return _value <=> other;
    }

    template<typename U>
    bool operator==(const U& other) const {
        return _value == other;
    }

    template<typename U>
    decltype(auto) operator|(U&& other) const {
        return _value | std::forward<U>(other);
    }

    template<typename U>
    friend auto operator<=>(const U& other, const immutable<T>& self) {
        return other <=> self._value;
    }

    template<typename U>
    friend bool operator==(const U& other, const immutable<T>& self) {
        return other == self._value;
    }

    friend auto& operator<<(std::ostream& out, const immutable<T>& self) { return out << self._value; }
};

template<typename T>
struct is_immutable : std::false_type {};

template<typename T>
struct is_immutable<immutable<T>> : std::true_type {};

template<typename T>
concept ImmutableType = is_immutable<T>::value;

} // namespace gr::meta

template<typename T>
struct std::formatter<gr::meta::immutable<T>> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin(); // no format-spec yet
    }

    template<typename FormatContext>
    auto format(const gr::meta::immutable<T>& immutable, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "{}", immutable.value());
    }
};

#endif
