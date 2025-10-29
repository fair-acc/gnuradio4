#ifndef GNURADIO_COMPUTEDOMAIN_HPP
#define GNURADIO_COMPUTEDOMAIN_HPP

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gr {

enum class Access : std::uint8_t { HostOnly, Shared, DeviceOnly };

struct ComputeDomain {
    // All names are lower-case canonical; no case-folding is performed.
    std::string_view kind{"host"};             // "host","gpu","tpu","fpga", user-defined
    Access           access{Access::HostOnly}; // hint to provider
    std::string_view backend{"none"};          // "none","sycl","cuda","hip", user-defined
    int              deviceIndex{-1};          // -1 = provider default
    std::string_view tag{};                    // optional (“gpu0”, "gpu1", “fpgaA”, ...)
    void*            user{nullptr};            // optional opaque payload

    // sugar (string-based)
    static constexpr ComputeDomain host() noexcept { return {}; }

    static constexpr ComputeDomain gpu_shared(std::string_view be = "sycl", int idx = -1) noexcept {
        ComputeDomain d;
        d.kind        = "gpu";
        d.access      = Access::Shared;
        d.backend     = be;
        d.deviceIndex = idx;
        return d;
    }
    static constexpr ComputeDomain gpu_device(std::string_view be = "sycl", int idx = -1) noexcept {
        ComputeDomain d;
        d.kind        = "gpu";
        d.access      = Access::DeviceOnly;
        d.backend     = be;
        d.deviceIndex = idx;
        return d;
    }
};

// Provider API: given a domain + optional backend context, return a PMR.
// Returned resource must outlive all allocators bound to it (static/thread_local typically).
using ProviderFn = std::pmr::memory_resource* (*)(const ComputeDomain& dom, void* ctx);

// transparent hashing for heterogenous lookup
struct KeyHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};
struct KeyEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
};

class ComputeRegistry {
    mutable std::mutex                                          _mtx;
    std::unordered_map<std::string, ProviderFn, KeyHash, KeyEq> _providers;

public:
    static ComputeRegistry& instance() {
        static ComputeRegistry r;
        return r;
    }

    void register_provider(std::string_view backend, ProviderFn fn) {
        std::scoped_lock lk(_mtx);
        _providers[std::string(backend)] = fn; // replace-or-insert
    }

    [[nodiscard]] std::pmr::memory_resource* resolve(const ComputeDomain& dom, void* ctx) const {
        if (dom.kind == "host" || dom.backend == "none") {
            return std::pmr::new_delete_resource();
        }
        std::scoped_lock lk(_mtx);
        auto             it = _providers.find(dom.backend); // heterogenous lookup
        if (it == _providers.end()) {
            throw std::runtime_error("no provider for backend '" + std::string(dom.backend) + "'");
        }
        if (auto* mr = it->second(dom, ctx)) {
            return mr;
        }
        throw std::runtime_error("provider returned null resource");
    }
};

struct BoundDomain {
    std::pmr::memory_resource* mr{std::pmr::new_delete_resource()};
    explicit BoundDomain(std::pmr::memory_resource* p) : mr(p) {}
    template<typename T>
    [[nodiscard]] std::pmr::polymorphic_allocator<T> allocator() const noexcept {
        return std::pmr::polymorphic_allocator<T>{mr};
    }
};

[[nodiscard]] inline BoundDomain bind(const ComputeDomain& dom = {}, void* backend_ctx = nullptr) { return BoundDomain{ComputeRegistry::instance().resolve(dom, backend_ctx)}; }

} // namespace gr

#endif // GNURADIO_COMPUTEDOMAIN_HPP
