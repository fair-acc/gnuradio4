#ifndef GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
#define GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <gnuradio-4.0/device/DeviceContext.hpp>

namespace gr::device {

/**
 * @brief Singleton registry mapping `compute_domain` strings to device contexts.
 *
 * Resolves by exact match, then progressively shorter `:`-separated prefixes. An unregistered domain yields
 * nullptr — the caller decides whether that is a skip, a fallback or an error.
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

    // a withdrawn domain behaves as absent, so the prefix fallback below keeps trying shorter domains
    [[nodiscard]] static DeviceContext* servedOrNull(std::unordered_map<std::string, std::unique_ptr<DeviceContext>, Hash, Eq>& contexts, std::string_view key) {
        auto it = contexts.find(key);
        return (it != contexts.end() && it->second->served()) ? it->second.get() : nullptr;
    }

    [[nodiscard]] DeviceContext* longestRegisteredPrefixOf(std::string_view computeDomain) {
        if (auto* served = servedOrNull(_contexts, computeDomain)) {
            return served;
        }
        auto shorter = std::string(computeDomain);
        for (std::size_t lastSeparator = shorter.rfind(':'); lastSeparator != std::string::npos; lastSeparator = shorter.rfind(':')) {
            shorter.resize(lastSeparator);
            if (auto* served = servedOrNull(_contexts, shorter)) {
                return served;
            }
        }
        return nullptr;
    }

public:
    static DeviceContextRegistry& instance() {
        static DeviceContextRegistry r;
        return r;
    }

    void registerContext(std::string_view name, std::unique_ptr<DeviceContext> ctx) {
        std::scoped_lock lk(_mtx);
        _contexts[std::string(name)] = std::move(ctx);
    }

    /// nullptr when no context is registered for the domain — never a silent CPU fallback
    [[nodiscard]] DeviceContext* tryResolve(std::string_view computeDomain) {
        std::scoped_lock lk(_mtx);
        return longestRegisteredPrefixOf(computeDomain);
    }

    /// marks a published domain gone so a fresh tryResolve() stops finding it, while an already-cached context
    /// pointer stays dereferenceable and is caught by dispatch()'s served() check. Never a deallocation.
    /// A name that was never registered is a silent no-op.
    void withdraw(std::string_view name) {
        std::scoped_lock lk(_mtx);
        if (auto it = _contexts.find(name); it != _contexts.end()) {
            it->second->withdraw();
        }
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_CONTEXT_REGISTRY_HPP
