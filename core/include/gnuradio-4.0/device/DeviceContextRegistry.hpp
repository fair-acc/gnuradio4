#ifndef GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
#define GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/ComputeDomain.hpp>
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

    struct Hash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    };
    struct Eq {
        using is_transparent = void;
        bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
    };

    std::unordered_map<std::string, std::unique_ptr<DeviceContext>, Hash, Eq> _contexts;

    std::unordered_map<std::string, std::string, Hash, Eq> _aliases; // canonical spelling -> the indexed name that owns the context

    /// the name under which `canonical` is published, following one alias hop; nullopt when nothing serves it
    [[nodiscard]] std::optional<std::string> ownerOfUnlocked(std::string_view canonical) {
        const auto alias = _aliases.find(canonical);
        const auto owner = alias != _aliases.end() ? std::string_view(alias->second) : canonical;
        const auto it    = _contexts.find(owner);
        return it != _contexts.end() ? std::optional<std::string>(it->first) : std::nullopt;
    }

    [[nodiscard]] DeviceContext* servedUnlocked(std::string_view canonical) {
        const auto owner = ownerOfUnlocked(canonical);
        if (!owner.has_value()) {
            return nullptr;
        }
        return _contexts.find(*owner)->second.get();
    }

    [[nodiscard]] DomainResolution resolveUnlocked(std::string_view computeDomain) {
        return resolveComputeDomain(computeDomain, [this](std::string_view rung) { return ownerOfUnlocked(rung); });
    }

public:
    static DeviceContextRegistry& instance() {
        static DeviceContextRegistry r;
        return r;
    }

    void registerContext(std::string_view name, std::unique_ptr<DeviceContext> ctx) {
        std::scoped_lock lk(_mtx);
        _contexts[canonicalDomainName(ComputeDomain::parse(name))] = std::move(ctx);
    }

    /// a second spelling of a device that `registerContext` already published, so both resolve to one context
    void registerAlias(std::string_view spelling, std::string_view owner) {
        std::scoped_lock lk(_mtx);
        _aliases[canonicalDomainName(ComputeDomain::parse(spelling))] = canonicalDomainName(ComputeDomain::parse(owner));
    }

    /// how `computeDomain` resolves, including which rung of the ladder answered and whether that is a downgrade
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

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
