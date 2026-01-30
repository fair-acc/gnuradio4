#ifndef GNURADIO_INDIRECT_HPP
#define GNURADIO_INDIRECT_HPP

#include <format>
#include <memory>
#include <utility>

namespace gr::meta {

// A simple implementation of std::indirect-like structure until
// we switch to C++26
template<typename T>
class indirect {
private:
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>, "T needs to be a value type");
    std::unique_ptr<T> _value;

public:
    indirect() : _value(std::make_unique<T>()) {}

    template<typename Arg, typename... Args>
    indirect(Arg&& arg, Args&&... args) : _value(std::make_unique<T>(std::forward<Arg>(arg), std::forward<Args>(args)...)) {}

    indirect(const indirect<T>& other)
    requires std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>
        : _value(std::make_unique<T>(*other)) {}

    indirect(indirect<T>&& other) : _value(std::move(other._value)) {}

    indirect<T>& operator=(const indirect<T>& other)
    requires std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>
    {
        auto tmp = other;
        std::swap(_value, tmp._value);
        return *this;
    }

    indirect<T>& operator=(indirect<T>&& other) {
        _value = std::exchange(other._value, nullptr);
        return *this;
    }

    ~indirect() = default;

    T&       operator*() { return *_value; }
    const T& operator*() const { return *_value; }
    T*       operator->() { return _value.get(); }
    const T* operator->() const { return _value.get(); }

    bool valueless_after_move() const { return _value == nullptr; }
};

} // namespace gr::meta

template<typename T>
struct std::formatter<gr::meta::indirect<T>> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin(); // no format-spec yet
    }

    template<typename FormatContext>
    auto format(const gr::meta::indirect<T>& indirect, FormatContext& ctx) const {
        if (indirect.valueless_after_move()) {
            return std::format_to(ctx.out(), "{}", "moved-from-indirect-value");

        } else {
            return std::format_to(ctx.out(), "{}", indirect.value);
        }
    }
};

#endif
