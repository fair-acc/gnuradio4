#ifndef GNURADIO_COMPUTEDOMAIN_HPP
#define GNURADIO_COMPUTEDOMAIN_HPP

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory_resource>
#include <mutex>
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

    /// parse "kind[:backend[:deviceIndex]]" into a ComputeDomain
    /// known kinds: "gpu", "fpga", "tpu" — anything else maps to host()
    /// backend strings are passed through (may be SYCL/AdaptiveCpp-reported device names);
    /// the returned string_views point into `s`, so `s` must outlive the result
    static ComputeDomain parse(std::string_view s) noexcept {
        if (s.empty() || s == "host" || s == "default_cpu" || s == "default_io") {
            return host();
        }

        auto mapKind = [](std::string_view k) -> std::string_view {
            if (k == "gpu") {
                return "gpu";
            }
            if (k == "fpga") {
                return "fpga";
            }
            if (k == "tpu") {
                return "tpu";
            }
            return "host";
        };

        const auto colon1 = s.find(':');
        const auto kindSv = mapKind(s.substr(0, colon1));
        if (kindSv == "host") {
            return host();
        }

        std::string_view backendSv = (kindSv == "gpu") ? std::string_view("sycl") : std::string_view("none");
        int              devIdx    = -1;

        if (colon1 != std::string_view::npos) {
            const auto rest   = s.substr(colon1 + 1);
            const auto colon2 = rest.find(':');
            const auto rawBe  = rest.substr(0, colon2);
            if (!rawBe.empty()) {
                backendSv = rawBe; // pass through — may be SYCL-reported device/backend name
            }
            if (colon2 != std::string_view::npos) {
                const auto idxStr = rest.substr(colon2 + 1);
                auto [ptr, ec]    = std::from_chars(idxStr.data(), idxStr.data() + idxStr.size(), devIdx);
                if (ec != std::errc{}) {
                    devIdx = -1;
                }
            }
        }

        ComputeDomain d;
        d.kind        = kindSv;
        d.access      = Access::Shared;
        d.backend     = backendSv;
        d.deviceIndex = devIdx;
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

    [[nodiscard]] std::expected<std::pmr::memory_resource*, std::string> resolve(const ComputeDomain& dom, void* ctx = nullptr) const {
        if (dom.kind == "host" || dom.backend == "none") {
            return std::pmr::new_delete_resource();
        }
        std::scoped_lock lk(_mtx);
        auto             it = _providers.find(dom.backend); // heterogenous lookup
        if (it == _providers.end()) {
            return std::unexpected("no provider for backend '" + std::string(dom.backend) + "'");
        }
        if (auto* mr = it->second(dom, ctx)) {
            return mr;
        }
        return std::unexpected("provider returned null resource for backend '" + std::string(dom.backend) + "'");
    }

    /// non-throwing resolve — returns nullptr if no provider is registered or the provider returns null
    [[nodiscard]] std::pmr::memory_resource* tryResolve(const ComputeDomain& dom, void* ctx = nullptr) const noexcept {
        if (dom.kind == "host" || dom.backend == "none") {
            return std::pmr::new_delete_resource();
        }
        std::scoped_lock lk(_mtx);
        auto             it = _providers.find(dom.backend);
        if (it == _providers.end()) {
            return nullptr;
        }
        try {
            return it->second(dom, ctx);
        } catch (...) {
            return nullptr;
        }
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

[[nodiscard]] inline BoundDomain bind(const ComputeDomain& dom = {}, void* backend_ctx = nullptr) { return BoundDomain{ComputeRegistry::instance().resolve(dom, backend_ctx).value_or(std::pmr::new_delete_resource())}; }

} // namespace gr

#endif // GNURADIO_COMPUTEDOMAIN_HPP
