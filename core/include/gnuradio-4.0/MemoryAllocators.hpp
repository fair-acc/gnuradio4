#ifndef MEMORYALLOCATORS_HPP
#define MEMORYALLOCATORS_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <print>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::allocator {

/** @brief STL allocator guaranteeing `alignment` byte alignment. */
template<typename T, std::size_t alignment = 64UZ>
requires(std::has_single_bit(alignment) && alignment >= alignof(T))
struct Aligned {
    using value_type = T;

    constexpr Aligned() noexcept = default;

    template<typename U>
    constexpr Aligned(const Aligned<U, alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);
        if (bytes == 0) {
            return nullptr; // fine per standard
        }
        if (void* p = ::operator new(bytes, std::align_val_t{alignment})) {
            return static_cast<T*>(p);
        }
        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept { ::operator delete(p, std::align_val_t{alignment}); }

    template<typename U>
    struct rebind {
        using other = Aligned<U, alignment>;
    };

    friend constexpr bool operator==(const Aligned&, const Aligned&) noexcept = default;
};

template<typename T>
using Default = Aligned<T, 64UZ>;

namespace detail {
enum class Event { allocate, deallocate, allocate_at_least };

struct DefaultLogger {
    void operator()(Event ev, std::size_t count, std::size_t bytes, std::source_location loc, std::string_view type_name) const noexcept { //
        std::println("[{:10}] type={} count={:3} bytes={:3} @ {}:{}:{}", ev, type_name, count, bytes, loc.file_name(), loc.line(), loc.column());
    }
};

template<class Ptr>
struct allocation_result {
    Ptr         ptr;
    std::size_t count;
};

template<class Alloc>
concept has_allocate_at_least = requires(Alloc a, std::size_t n) {
    { std::allocator_traits<Alloc>::allocate_at_least(a, n) };
};

} // namespace detail

template<typename T, typename Underlying = Default<T>, typename Logger = detail::DefaultLogger>
struct Logging {
    using value_type      = T;
    using underlying_type = Underlying;
    using logger_type     = Logger;

    static_assert(std::is_same_v<typename underlying_type::value_type, T>, "Underlying allocator must use same value_type");

    underlying_type      _underlying{};
    logger_type          _logger{};
    std::string          _type_name = gr::meta::type_name<T>();
    std::source_location _origin    = std::source_location::current();

    explicit Logging(underlying_type underlying = {}, logger_type logger = {}, std::source_location loc = std::source_location::current()) noexcept : _underlying(std::move(underlying)), _logger(std::move(logger)), _origin(loc) {}

    template<typename U, typename OU, typename L>
    Logging(const Logging<U, OU, L>& rhs) noexcept : _underlying(rhs.underlying()), _logger(rhs.logger()), _origin(rhs.origin()) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);
        _logger(detail::Event::allocate, n, bytes, _origin, _type_name);
        return _underlying.allocate(n);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        const std::size_t bytes = n * sizeof(T);
        _logger(detail::Event::deallocate, n, bytes, _origin, _type_name);
        _underlying.deallocate(p, n);
    }

    [[nodiscard]] auto allocate_at_least(std::size_t n) {
        if constexpr (gr::allocator::detail::has_allocate_at_least<underlying_type>) {
            auto r = std::allocator_traits<underlying_type>::allocate_at_least(_underlying, n);
            _logger(detail::Event::allocate_at_least, r.count, r.count * sizeof(T), _origin, _type_name);
            return r;
        } else {
            T*   p = allocate(n);
            auto r = gr::allocator::detail::allocation_result<T*>{p, n};
            _logger(detail::Event::allocate_at_least, r.count, r.count * sizeof(T), _origin, _type_name);
            return r;
        }
    }

    [[nodiscard]] underlying_type&       underlying() noexcept { return _underlying; }
    [[nodiscard]] const underlying_type& underlying() const noexcept { return _underlying; }
    [[nodiscard]] logger_type&           logger() noexcept { return _logger; }
    [[nodiscard]] const logger_type&     logger() const noexcept { return _logger; }
    [[nodiscard]] std::source_location   origin() const noexcept { return _origin; }

    template<typename U>
    struct rebind {
        using other = Logging<U, typename std::allocator_traits<underlying_type>::template rebind_alloc<U>, logger_type>;
    };

    friend bool operator==(const Logging& a, const Logging& b) noexcept {
        if constexpr (std::allocator_traits<underlying_type>::is_always_equal::value) {
            return &a == &b || (a._origin.file_name() == b._origin.file_name() && a._origin.line() == b._origin.line());
        } else {
            return a._underlying == b._underlying && a._origin.file_name() == b._origin.file_name() && a._origin.line() == b._origin.line();
        }
    }
};

} // namespace gr::allocator

#endif // MEMORYALLOCATORS_HPP
