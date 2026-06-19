#ifndef MEMORYALLOCATORS_HPP
#define MEMORYALLOCATORS_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <print>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/meta/CacheLineSize.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::allocator {
template<std::size_t Align, typename T>
[[nodiscard]] constexpr bool isAligned(const T* p) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % Align == 0UZ;
}

template<typename T>
[[nodiscard]] constexpr bool isAligned(const T* p, std::size_t alignment) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % alignment == 0UZ;
}

/** @brief STL allocator guaranteeing `alignment` byte alignment. */
template<typename T, std::size_t alignment = gr::kCacheLine>
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
#if __cpp_exceptions
        throw std::bad_alloc();
#else
        std::abort(); // freestanding: the throwing operator new above already terminates on OOM
#endif
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept { ::operator delete(p, std::align_val_t{alignment}); }

    template<typename U>
    struct rebind {
        using other = Aligned<U, alignment>;
    };

    friend constexpr bool operator==(const Aligned&, const Aligned&) noexcept = default;
};

template<typename T>
using Default = Aligned<T, gr::kCacheLine>;

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

template<typename Alloc>
struct is_aligned_allocator : std::false_type {};

template<typename T, std::size_t Alignment>
struct is_aligned_allocator<gr::allocator::Aligned<T, Alignment>> : std::true_type {};

template<typename Alloc>
inline constexpr bool is_aligned_allocator_v = is_aligned_allocator<Alloc>::value;

template<typename Container>
struct container_allocator {
    using type = std::allocator<typename Container::value_type>;
};

template<typename T, typename Alloc>
struct container_allocator<std::vector<T, Alloc>> {
    using type = Alloc;
};

template<typename Container>
using container_allocator_t = typename container_allocator<Container>::type;

template<typename Alloc, typename NewType>
struct rebind_allocator {
    using type = typename std::allocator_traits<Alloc>::template rebind_alloc<NewType>;
};

template<typename T, std::size_t Alignment, typename NewType>
struct rebind_allocator<gr::allocator::Aligned<T, Alignment>, NewType> {
    using type = gr::allocator::Aligned<NewType, Alignment>;
};

template<typename Alloc, typename NewType>
using rebind_allocator_t = typename rebind_allocator<Alloc, NewType>::type;

template<typename InputContainer, typename OutputType>
struct deduce_output_allocator {
    using input_alloc = container_allocator_t<std::remove_cvref_t<InputContainer>>;
    using type        = rebind_allocator_t<input_alloc, OutputType>;
};

template<typename InputContainer, typename OutputType>
using deduce_output_allocator_t = typename deduce_output_allocator<InputContainer, OutputType>::type;
} // namespace detail

template<typename T, typename Underlying = Default<T>, typename Logger = detail::DefaultLogger>
struct Logging {
    using value_type      = T;
    using underlying_type = Underlying;
    using logger_type     = Logger;

    static_assert(std::is_same_v<typename underlying_type::value_type, T>, "Underlying allocator must use same value_type");

    underlying_type      _underlying{};
    logger_type          _logger{};
    std::string_view     _type_name = gr::meta::type_name<T>(); // static consteval storage
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

namespace pmr {

template<class T>
[[nodiscard]] T* migrate(std::pmr::memory_resource& target_resource, std::pmr::memory_resource& source_resource, T* source_ptr, std::size_t count) {
    if (source_ptr == nullptr || count == 0) {
        return nullptr;
    }

    if (&target_resource == &source_resource) {
        return source_ptr;
    }

    T*                           target_ptr  = static_cast<T*>(target_resource.allocate(count * sizeof(T), alignof(T)));
    [[maybe_unused]] std::size_t constructed = 0;

#if __cpp_exceptions
    try {
#endif
        // move or copy elements
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(target_ptr, source_ptr, count * sizeof(T));
            constructed = count;
        } else if constexpr (std::is_nothrow_move_constructible_v<T>) {
            std::uninitialized_move_n(source_ptr, count, target_ptr);
            constructed = count;
        } else { // copy with exception safety for throwing move constructors
            for (; constructed < count; ++constructed) {
                std::construct_at(target_ptr + constructed, source_ptr[constructed]);
            }
        }
#if __cpp_exceptions
    } catch (...) { // clean up partially constructed target
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_n(target_ptr, constructed);
        }
        target_resource.deallocate(target_ptr, count * sizeof(T), alignof(T));
        throw; // Source remains intact (strong guarantee)
    }
#endif

    // Destroy source objects
    if constexpr (!std::is_trivially_destructible_v<T>) {
        std::destroy_n(source_ptr, count);
    }

    // Deallocate source memory
    source_resource.deallocate(source_ptr, count * sizeof(T), alignof(T));

    return target_ptr;
}

/// convert std:: container to std::pmr:: equivalent (element-wise copy, explicit resource)
template<typename T>
[[nodiscard]] auto to_pmr(const std::vector<T>& src, std::pmr::memory_resource* mr) {
    return std::pmr::vector<T>(src.begin(), src.end(), mr);
}

[[nodiscard]] inline std::pmr::string to_pmr(std::string_view s, std::pmr::memory_resource* mr) { return std::pmr::string(s, mr); }

/// convert std::pmr:: container to std:: equivalent (element-wise copy)
template<typename T>
[[nodiscard]] auto to_std(const std::pmr::vector<T>& src) {
    return std::vector<T>(src.begin(), src.end());
}

[[nodiscard]] inline std::string to_std(std::string_view s) { return std::string(s); }

/// satisfied by any type with an extended move constructor accepting a pmr allocator
template<typename T>
concept PmrMigratable = std::uses_allocator_v<T, std::pmr::polymorphic_allocator<>> && std::is_constructible_v<T, T&&, std::pmr::polymorphic_allocator<>>;

/// migrate a single pmr-aware value to a new memory_resource (in-place destroy + reconstruct)
/// safe: after move, field is empty — reconstruct from empty rebound is non-throwing
template<PmrMigratable T>
void migrateField(T& field, std::pmr::memory_resource* mr) {
    std::pmr::polymorphic_allocator<> alloc{mr};
    T                                 rebound{std::move(field), alloc};
    std::destroy_at(&field);
    std::construct_at(&field, std::move(rebound));
}

struct ResourceProfile {
    std::pmr::memory_resource* data      = nullptr; /// streaming sample buffers
    std::pmr::memory_resource* tag       = nullptr; /// tag rings + per-payload property_map
    std::pmr::memory_resource* mechanics = nullptr; /// block strings, port metadata, settings, BlockWrapper / Sequence / thread_pool queues

    [[nodiscard]] std::pmr::memory_resource* dataResource() const noexcept { return data ? data : std::pmr::get_default_resource(); }
    [[nodiscard]] std::pmr::memory_resource* tagResource() const noexcept { return tag ? tag : std::pmr::get_default_resource(); }
    [[nodiscard]] std::pmr::memory_resource* mechanicsResource() const noexcept { return mechanics ? mechanics : std::pmr::get_default_resource(); }

    // TLS stash read by Block's member initialiser (emplaceBlock guards via ResourceProfileScope).
    // single-thread construction only — not seen by threads spawned mid-construction. [design-review]
    [[nodiscard]] static ResourceProfile& currentTls() noexcept {
        thread_local ResourceProfile profile{};
        return profile;
    }
};

/// RAII swap of `ResourceProfile::currentTls()`. Graph::emplaceBlock pushes its graph's
/// profile before constructing the child Block, which reads currentTls() into `_resources`.
struct ResourceProfileScope {
    ResourceProfile _previous;

    explicit ResourceProfileScope(ResourceProfile profile) : _previous(ResourceProfile::currentTls()) { ResourceProfile::currentTls() = profile; }
    ~ResourceProfileScope() { ResourceProfile::currentTls() = _previous; }

    ResourceProfileScope(const ResourceProfileScope&)            = delete;
    ResourceProfileScope& operator=(const ResourceProfileScope&) = delete;
    ResourceProfileScope(ResourceProfileScope&&)                 = delete;
    ResourceProfileScope& operator=(ResourceProfileScope&&)      = delete;
};

/// PMR resource whose `do_allocate` aborts via `gr::log::fatal`. Install as the default
/// resource in heap-discipline tests / MCU builds.
struct NoHeapResource : std::pmr::memory_resource {
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        gr::log::fatal(std::format("gr::pmr::NoHeapResource: refused {} byte / {}-aligned allocation", bytes, alignment));
        return nullptr;
    }
    void               do_deallocate(void*, std::size_t, std::size_t) override {}
    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

/// RAII swap of `std::pmr::get_default_resource()` for the lifetime of the scope.
struct ScopedDefaultResource {
    std::pmr::memory_resource* _previous;

    explicit ScopedDefaultResource(std::pmr::memory_resource* resource) : _previous(std::pmr::set_default_resource(resource)) {}
    ~ScopedDefaultResource() { std::pmr::set_default_resource(_previous); }

    ScopedDefaultResource(const ScopedDefaultResource&)            = delete;
    ScopedDefaultResource& operator=(const ScopedDefaultResource&) = delete;
    ScopedDefaultResource(ScopedDefaultResource&&)                 = delete;
    ScopedDefaultResource& operator=(ScopedDefaultResource&&)      = delete;
};

/// PMR resource that counts allocate / deallocate calls and tracks live byte usage,
/// then forwards to an upstream resource. Test-only analogue of allocator-side `Logging`.
struct CountingResource : std::pmr::memory_resource {
    std::pmr::memory_resource* upstream{std::pmr::new_delete_resource()};
    std::size_t                allocCount{0UZ};
    std::size_t                deallocCount{0UZ};
    std::size_t                liveBytes{0UZ};

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocCount;
        liveBytes += bytes;
        return upstream->allocate(bytes, alignment);
    }
    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        ++deallocCount;
        liveBytes -= bytes;
        upstream->deallocate(ptr, bytes, alignment);
    }
    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

} // namespace pmr
} // namespace gr::allocator

namespace gr::pmr {

/**
 * @brief non-owning bump-arena over externally supplied storage
 *
 * Bump-pointer `std::pmr::memory_resource` over a caller-supplied byte span. Deallocate
 * is a no-op (monotonic semantics); the only way to reclaim is `reset()`, which
 * invalidates every live pointer. On exhaustion `do_allocate` calls `gr::log::fatal`
 * (`throws gr::exception` on hosted, `std::abort` on AOT).
 *
 * Non-owning by design: real embedded use places the storage via the linker
 * (DTCM/ITCM, DMA-coherent SRAM, USM, backup-SRAM) and hands the resource a span over
 * that region. For tests / generic pools use `OwnedStaticArenaResource<N>` below.
 *
 * **Thread safety:** single-threaded by design (matches `std::pmr::monotonic_buffer_resource`).
 * `do_allocate` mutates `_used` without synchronisation; wrap with `std::pmr::synchronized_pool_resource`
 * or external locking for multi-threaded use.
 */
class StaticArenaResource : public std::pmr::memory_resource {
    std::byte*  _data;
    std::size_t _capacity;
    std::size_t _used = 0UZ;

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0UZ) { // PMR allows zero-byte requests; do not consume padding for them
            return _data + _used;
        }
        void*       aligned = _data + _used;
        std::size_t space   = _capacity - _used;
        if (std::align(alignment, bytes, aligned, space) == nullptr) { // also guards overflow: returns nullptr when the request cannot fit
            gr::log::fatal(std::format("gr::pmr::StaticArenaResource: exhausted (request {} bytes, alignment {}; {} used / {} capacity)", bytes, alignment, _used, _capacity));
        }
        const std::size_t padding = static_cast<std::size_t>(static_cast<std::byte*>(aligned) - (_data + _used));
        _used += padding + bytes;
        return aligned;
    }

    void do_deallocate(void*, std::size_t, std::size_t) noexcept override {}

    [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

public:
    explicit StaticArenaResource(std::span<std::byte> storage) noexcept : _data(storage.data()), _capacity(storage.size()) {}

    StaticArenaResource(const StaticArenaResource&)            = delete;
    StaticArenaResource& operator=(const StaticArenaResource&) = delete;

    [[nodiscard]] std::size_t used() const noexcept { return _used; }
    [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }
    [[nodiscard]] std::size_t available() const noexcept { return _capacity - _used; }

    void reset() noexcept { _used = 0UZ; } // invalidates every live allocation
};

static_assert(!std::is_copy_constructible_v<StaticArenaResource>);
static_assert(!std::is_copy_assignable_v<StaticArenaResource>);
// non-movable is load-bearing, not incidental: a move would byte-copy `_data` into the destination, leaving
// OwnedStaticArenaResource's base pointing into the moved-from object's `_bytes` array (deleted copy suppresses the implicit move).
static_assert(!std::is_move_constructible_v<StaticArenaResource>);
static_assert(!std::is_move_assignable_v<StaticArenaResource>);

/**
 * @brief owning, compile-time-sized convenience over `StaticArenaResource`
 *
 * BSS/stack-resident storage with a `StaticArenaResource` view onto it. Useful for
 * tests, generic pools, and any case that does not need linker-controlled placement.
 */
template<std::size_t N, std::size_t kAlignment = alignof(std::max_align_t)>
class OwnedStaticArenaResource : public StaticArenaResource {
    alignas(kAlignment) std::array<std::byte, N> _bytes{};

public:
    OwnedStaticArenaResource() noexcept : StaticArenaResource(std::span<std::byte>{_bytes}) {}
};

static_assert(std::is_nothrow_default_constructible_v<OwnedStaticArenaResource<1024UZ>>);

} // namespace gr::pmr

namespace gr {
using gr::allocator::pmr::migrateField;
using gr::allocator::pmr::PmrMigratable;
using gr::allocator::pmr::ResourceProfile;
using gr::allocator::pmr::ResourceProfileScope;
using gr::allocator::pmr::to_pmr;
using gr::allocator::pmr::to_std;
} // namespace gr

#endif // MEMORYALLOCATORS_HPP
