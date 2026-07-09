#ifndef GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
#define GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/ComputeDomain.hpp>
#include <gnuradio-4.0/Export.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief Singleton registry mapping `compute_domain` strings to device contexts.
 *
 * Contexts are keyed by canonical name, so every spelling of one device names one context. A domain that no
 * rung of the ladder serves yields nullptr — the caller decides whether that is a skip or an error.
 */
class DeviceContextRegistry {
    mutable std::mutex _mtx;

    // the transparent hashing `ComputeDomain.hpp` already defines, rather than a second copy of it to keep in step
    std::unordered_map<std::string, std::unique_ptr<DeviceContext>, KeyHash, KeyEq> _contexts;

    std::unordered_map<std::string, std::string, KeyHash, KeyEq> _aliases; // canonical spelling -> the indexed name that owns the context

    /// the entry serving `canonical`, following an alias when one is registered, or `_contexts.end()` when none does
    [[nodiscard]] auto entryUnlocked(std::string_view canonical) {
        const auto alias = _aliases.find(canonical);
        return _contexts.find(alias != _aliases.end() ? std::string_view(alias->second) : canonical);
    }

    [[nodiscard]] std::optional<std::string> ownerOfUnlocked(std::string_view canonical) {
        const auto it = entryUnlocked(canonical);
        return it != _contexts.end() ? std::optional<std::string>(it->first) : std::nullopt;
    }

    [[nodiscard]] DeviceContext* servedUnlocked(std::string_view canonical) {
        const auto it = entryUnlocked(canonical);
        return it != _contexts.end() ? it->second.get() : nullptr;
    }

    [[nodiscard]] DomainResolution resolveUnlocked(std::string_view computeDomain) {
        return resolveComputeDomain(computeDomain, [this](std::string_view rung) { return ownerOfUnlocked(rung); });
    }

    /// `host:native` — the CPU backing, served in every build whatever backends were compiled in. It is spelled with
    /// a backend for symmetry with `host:sycl`/`gpu:sycl`, and it is deliberately NOT the default: plain `host` stays
    /// unserved, so `tryResolve("host")` still reports absence and a device domain that downgrades to `host` still
    /// refuses rather than silently running on a CPU context. A block reaches this only by asking for it by name.
    /// It is also off the downgrade ladder (`resolveComputeDomain` walks declared → un-indexed → `host:sycl`).
    DeviceContextRegistry() { _contexts[canonicalDomainName(ComputeDomain::parse("host:native"))] = std::make_unique<DeviceContextCpu>(); }

public:
    GNURADIO_EXPORT static DeviceContextRegistry& instance() {
        // never destroyed, as the queues it holds: a `DeviceContextSycl` owns a `sycl::queue` by value, and
        // destroying one at exit races the CUDA driver's own deinitialisation (`~cuda_queue`, CUDA error 4).
        static auto& r = *new DeviceContextRegistry();
        return r;
    }

    /// false when the name is already served: `tryResolve` hands out raw pointers that dispatch and block shadows
    /// latch, so replacing an entry would free a context still in use
    bool registerContext(std::string_view name, std::unique_ptr<DeviceContext> ctx) {
        std::scoped_lock lk(_mtx);
        return _contexts.try_emplace(canonicalDomainName(ComputeDomain::parse(name)), std::move(ctx)).second;
    }

    void registerAlias(std::string_view spelling, std::string_view owner) {
        std::scoped_lock lk(_mtx);
        _aliases[canonicalDomainName(ComputeDomain::parse(spelling))] = canonicalDomainName(ComputeDomain::parse(owner));
    }

    [[nodiscard]] DomainResolution resolve(std::string_view computeDomain) {
        std::scoped_lock lk(_mtx);
        return resolveUnlocked(computeDomain);
    }

    /// nullptr when no rung of the ladder is served — never a silent CPU fallback
    [[nodiscard]] DeviceContext* tryResolve(std::string_view computeDomain) {
        std::scoped_lock lk(_mtx);
        return servedUnlocked(resolveUnlocked(computeDomain).resolved);
    }
};

/// What a declared compute domain resolves to, and what the caller owes the user if it is not what was asked for.
struct DomainOutcome {
    DeviceContext* context = nullptr; // nullptr means run on the host, whether by choice or by refusal
    bool           refused = false;   // a domain spelled with '!' that could not be served: the caller must stop
    std::string    declared;          // canonical spelling of the request, for the message
    std::string    reason;            // empty when the request was met exactly
};

/**
 * @brief Resolve a declared compute domain, applying the '!'-required and downgrade policy.
 *
 * One decision for every driver of blocks. A scheduler and a sub-graph both place blocks, and when each resolved
 * for itself the two disagreed: a member declaring a required domain could be answered by a different device and
 * run there silently, because only one of them checked. What the caller still owns is how to say so -- a scheduler
 * raises an error message, a group puts its members into ERROR -- which is why this reports rather than logs.
 */
[[nodiscard]] inline DomainOutcome resolveDeclaredDomain(std::string_view declaredDomain) {
    const ComputeDomain parsed = ComputeDomain::parse(declaredDomain);
    if (!parsed.isDevice()) {
        return {};
    }
    const DomainResolution resolution = DeviceContextRegistry::instance().resolve(declaredDomain);
    DeviceContext* const   served     = DeviceContextRegistry::instance().tryResolve(resolution.resolved);
    if (served != nullptr && !resolution.downgraded) {
        return {.context = served, .refused = false, .declared = resolution.declared, .reason = {}};
    }
    return {.context = nullptr, .refused = parsed.required, .declared = resolution.declared, //
        .reason = resolution.downgraded ? std::format("is not available, and '{}' answered instead", resolution.resolved) : "is not available"};
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
