#ifndef GNURADIO_COMPUTEDOMAIN_HPP
#define GNURADIO_COMPUTEDOMAIN_HPP

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/Export.hpp>

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
    bool             required{false};          // spelled with a trailing '!': the block must reach this domain or stop

    // a domain selects a device when its memory is not host-resident, or when a backend executes it
    // (a SYCL CPU device is `host:sycl`: host memory, SYCL execution)
    [[nodiscard]] constexpr bool isDevice() const noexcept { return kind != "host" || backend != "none"; }

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
    /// device-only memory: the device may dereference it, the host may not
    static constexpr ComputeDomain gpu_device(std::string_view be = "sycl", int idx = -1) noexcept {
        ComputeDomain d;
        d.kind        = "gpu";
        d.access      = Access::DeviceOnly;
        d.backend     = be;
        d.deviceIndex = idx;
        return d;
    }

    /// parse "kind[:backend[:deviceIndex]]" into a ComputeDomain
    /// known kinds: "host", "gpu", "fpga", "tpu" — anything else maps to host()
    /// a SYCL CPU device is `host:sycl`: host-resident memory, SYCL execution
    /// backend strings are passed through (may be SYCL/AdaptiveCpp-reported device names);
    /// the returned string_views point into `s`, so `s` must outlive the result
    static ComputeDomain parse(std::string_view s) noexcept {
        bool required = false;
        while (s.ends_with('!')) { // "gpu:sycl!" — reaching the domain is a requirement, not a preference
            required = true;
            s.remove_suffix(1UZ);
        }
        if (s.empty() || s == "host" || s == "default_cpu" || s == "default_io") {
            ComputeDomain plain = host();
            plain.required      = required;
            return plain;
        }

        static constexpr std::array kKinds{std::string_view{"gpu"}, std::string_view{"fpga"}, std::string_view{"tpu"}, std::string_view{"host"}};
        const auto                  mapKind = [](std::string_view k) -> std::string_view {
            const auto it = std::ranges::find(kKinds, k);
            return it != kKinds.end() ? *it : std::string_view{};
        };

        const auto       colon1   = s.find(':');
        std::string_view kindWord = s.substr(0, colon1);
        while (kindWord.ends_with('!')) { // "gpu!:sycl" — a misplaced marker still names a requirement
            required = true;
            kindWord.remove_suffix(1UZ);
        }
        const auto kindSv = mapKind(kindWord);
        if (kindSv.empty()) { // plain host, but still required if it was marked: a typo must not relax the contract
            ComputeDomain plain = host();
            plain.required      = required;
            return plain;
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
                const auto        idxStr = rest.substr(colon2 + 1);
                const char* const last   = idxStr.data() + idxStr.size();
                auto [ptr, ec]           = std::from_chars(idxStr.data(), last, devIdx);
                if (ec != std::errc{} || ptr != last || devIdx < -1) {
                    devIdx = -1;
                }
            }
        }

        ComputeDomain d;
        d.kind        = kindSv;
        d.access      = kindSv == "host" ? Access::HostOnly : Access::Shared; // host-kind memory stays host-resident
        d.backend     = backendSv;
        d.deviceIndex = devIdx;
        d.required    = required;
        return d;
    }
};

/// canonical spelling of a parsed domain: "kind[:backend[:index]]". The index survives a backend-less kind
/// as "tpu::4", which parses back to the same domain: it is the registry key, so two devices must not share one.
[[nodiscard]] inline std::string canonicalDomainName(const ComputeDomain& domain) {
    std::string name(domain.kind);
    if (domain.backend != "none") {
        name += ':';
        name += domain.backend;
    }
    if (domain.deviceIndex >= 0) {
        name += domain.backend != "none" ? ":" : "::";
        name += std::to_string(domain.deviceIndex);
    }
    return name;
}

struct DomainResolution {
    std::string declared;          // canonical spelling of what was asked for
    std::string resolved;          // the name of whatever actually serves it, after aliases
    bool        downgraded{false}; // a lower rung of the ladder answered
};

/// answers which indexed name owns a rung of the ladder, or nothing when no context serves it
template<typename T>
concept DomainOwnerLookup = requires(const T& lookup, std::string_view rung) {
    { lookup(rung) } -> std::convertible_to<std::optional<std::string>>;
};

template<DomainOwnerLookup TOwnerLookup>
[[nodiscard]] DomainResolution resolveComputeDomain(std::string_view declaredDomain, const TOwnerLookup& ownerOf) {
    const ComputeDomain parsed = ComputeDomain::parse(declaredDomain);
    DomainResolution    result{.declared = canonicalDomainName(parsed), .resolved = {}, .downgraded = false};
    if (!parsed.isDevice()) {
        result.resolved = result.declared;
        return result;
    }

    ComputeDomain withoutIndex = parsed;
    withoutIndex.deviceIndex   = -1;

    ComputeDomain hostRung = withoutIndex;
    hostRung.kind          = "host"; // the CPU rung follows the backend that was asked for: gpu:cuda falls back to
                                     // host:cuda, not to whatever backend the grammar happened to prefer
    const std::array<std::string, 3> ladder{result.declared, canonicalDomainName(withoutIndex), canonicalDomainName(hostRung)};
    for (std::size_t rung = 0UZ; rung < ladder.size(); ++rung) {
        if (rung > 0UZ && ladder[rung] == ladder[rung - 1UZ]) {
            continue; // an already-index-less request repeats its own first rung
        }
        if (std::optional<std::string> owner = ownerOf(ladder[rung]); owner.has_value()) {
            result.resolved   = std::move(*owner);
            result.downgraded = rung > 0UZ;
            return result;
        }
    }

    result.resolved   = "host";
    result.downgraded = true;
    return result;
}

using DomainResolverFn = std::string (*)(std::string_view declaredDomain);

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

// Static-init hazard for the heap-discipline / MCU target: the registry's lazy unordered_map fires
// before `main()`, so no ResourceProfile installed via Graph() can intercept it. Policy for the
// embedded target: do NOT instantiate any GR object (Graph, scheduler, ComputeRegistry-dependent
// blocks) at namespace scope. Construct them inside `main()` so the user-installed PMR default is
// already live. The map's own heap use is one-shot at first lookup.
class ComputeRegistry {
    mutable std::mutex                                          _mtx;
    std::unordered_map<std::string, ProviderFn, KeyHash, KeyEq> _providers;
    DomainResolverFn                                            _domainResolver = nullptr;

public:
    GNURADIO_EXPORT static ComputeRegistry& instance() {
        static ComputeRegistry r;
        return r;
    }

    void registerProvider(std::string_view backend, ProviderFn fn) {
        std::scoped_lock lk(_mtx);
        _providers[std::string(backend)] = fn; // replace-or-insert
    }

    void registerDomainResolver(DomainResolverFn fn) {
        std::scoped_lock lk(_mtx);
        _domainResolver = fn;
    }

    [[nodiscard]] std::string resolvedDomainName(std::string_view declaredDomain) const {
        DomainResolverFn resolver = nullptr;
        {
            std::scoped_lock lk(_mtx);
            resolver = _domainResolver; // called outside the lock: it takes the device registry's own
        }
        return resolver != nullptr ? resolver(declaredDomain) : std::string(declaredDomain);
    }

    [[nodiscard]] std::expected<std::pmr::memory_resource*, std::string> resolve(const ComputeDomain& dom, void* ctx = nullptr) const {
        if (dom.backend == "none") {
            if (dom.isDevice()) { // `fpga`/`tpu` name a device but no backend that could serve one
                return std::unexpected("compute domain '" + canonicalDomainName(dom) + "' names a device but no backend to serve it");
            }
            return std::pmr::new_delete_resource(); // `host:sycl` still needs device-accessible (USM host/shared) storage
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
        if (dom.backend == "none") {
            return dom.isDevice() ? nullptr : std::pmr::new_delete_resource(); // a device kind with no backend has nothing to serve it
        }
        std::scoped_lock lk(_mtx);
        auto             it = _providers.find(dom.backend);
        if (it == _providers.end()) {
            return nullptr;
        }
#if __cpp_exceptions
        try {
            return it->second(dom, ctx);
        } catch (...) {
            return nullptr;
        }
#else
        return it->second(dom, ctx);
#endif
    }
};

/// A `ComputeDomain` resolved to the allocator it names. The framework does not use it: a block gets its resource
/// from the graph's precedence chain (device > block > edge > graph > default), never by binding one itself. It is
/// the API for code *outside* a graph -- a test, a benchmark, a standalone algorithm -- that wants the same USM a
/// device block would get.
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
