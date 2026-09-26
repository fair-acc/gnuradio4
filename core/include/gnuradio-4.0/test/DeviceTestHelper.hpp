#ifndef GNURADIO_TEST_DEVICE_TEST_HELPER_HPP
#define GNURADIO_TEST_DEVICE_TEST_HELPER_HPP

#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/DeviceLog.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/test/DeviceExpectation.hpp>
#include <gnuradio-4.0/test/GraphFixture.hpp>

namespace gr::testing {

inline constexpr std::string_view kHostDomain = "host";

/// deliberately not a range: `boost::ut` claims `body | range` for its own parameterised tests, so the domain
/// list must be a type its `operator|` cannot accept
struct DomainSet {
    std::span<const std::string_view> domains;

    [[nodiscard]] constexpr std::size_t size() const noexcept { return domains.size(); }
};

inline constexpr std::array<std::string_view, 3UZ> kAllDomainNames     = {"host", "host:sycl", "gpu:sycl"};
inline constexpr std::array<std::string_view, 2UZ> kOffloadDomainNames = {"host:sycl", "gpu:sycl"};

inline constexpr DomainSet kAllDomains{kAllDomainNames};
inline constexpr DomainSet kOffloadDomains{kOffloadDomainNames};

inline constexpr std::size_t kDeviceTestLogSlots = 64UZ;

/// the compute domains this run actually serves: the host always, plus each offload domain a backend registered
[[nodiscard]] inline std::vector<std::string_view> servedDomains() {
    std::vector<std::string_view> domains{kHostDomain};
    std::ignore = gr::device::registerSyclRuntime(); // false without a SYCL backend, and then nothing is added
    for (std::string_view candidate : kOffloadDomainNames) {
        if (gr::device::DeviceContextRegistry::instance().isServedExactly(candidate)) {
            domains.push_back(candidate);
        }
    }
    return domains;
}

/// domains this run was configured to reach but did not, which a caller must treat as a failure rather than a skip
[[nodiscard]] inline std::vector<std::string_view> missingRequiredDomains() {
    std::vector<std::string_view> missing;
    const auto                    served = servedDomains();
    for (std::string_view candidate : kOffloadDomainNames) {
        if (deviceDomainRequired(candidate) && std::ranges::find(served, candidate) == served.end()) {
            missing.push_back(candidate);
        }
    }
    return missing;
}

/// the first of `preference` this run serves, for a test that wants one domain rather than a sweep
[[nodiscard]] inline std::optional<std::string_view> firstServedDomain(std::initializer_list<std::string_view> preference) {
    std::ignore      = gr::device::registerSyclRuntime();
    const auto match = std::ranges::find_if(preference, [](std::string_view domain) { return gr::device::DeviceContextRegistry::instance().isServedExactly(domain); });
    return match == preference.end() ? std::nullopt : std::optional<std::string_view>(*match);
}

/// a SYCL CPU device stands in for a GPU wherever a test checks API behaviour rather than device performance,
/// so a machine without a GPU still exercises the path instead of skipping it
[[nodiscard]] inline std::optional<std::string_view> firstServedSyclDomain() { return firstServedDomain({"gpu:sycl", "host:sycl"}); }

/// runs `body(domain)` once per served domain and answers how many ran, so a test can insist it exercised one
template<typename TBody>
[[nodiscard]] std::size_t forEachDomain(TBody&& body) {
    std::size_t exercised = 0UZ;
    for (std::string_view domain : servedDomains()) {
        body(domain);
        ++exercised;
    }
    return exercised;
}

/// what a kernel body receives: somewhere to report, and whether anything has already gone wrong
struct DeviceTestHandle {
    gr::log::DeviceLogger log{};
    std::uint32_t*        failures{};

    [[nodiscard]] bool failed() const noexcept { return failures != nullptr && gr::atomic_ref(*failures).load_acquire() != 0U; }

    void countFailure() const noexcept {
        if (failures != nullptr) {
            gr::atomic_ref(*failures).fetch_add(1U);
        }
    }
};

// unqualified lookup stops at the first namespace that declares the name, so a device `expect` alone in
// `gr::testing` would hide `boost::ut::expect` from every test inside `gr::` that says
// `using namespace gr::testing;`. Re-exporting ut's overload keeps both in one set.
using boost::ut::expect;

/// `expect(handle, condition)` — reports without a call site: `__builtin_FILE()` in a defaulted argument resolves
/// where that default is written, not where `expect` is called, so a site captured here would name this header.
/// Pass a message to have the failure attributed to your own line.
inline void expect(const DeviceTestHandle& handle, bool condition) noexcept {
    if (condition) {
        return;
    }
    handle.countFailure();
    handle.log.failure(gr::log::DeviceFormatString<>{std::string_view{"expectation failed"}, std::string_view{}, 0U});
}

/// `expect(handle, condition, "len {} != {}", n, 7UZ)` — the format string carries the call site with it
template<typename... Args>
void expect(const DeviceTestHandle& handle, bool condition, gr::log::DeviceFormatString<std::type_identity_t<Args>...> fmt, Args... args) noexcept {
    if (condition) {
        return;
    }
    handle.countFailure();
    handle.log.failure(fmt, args...);
}

/**
 * one domain under test: allocate through it, launch through it, and it releases what it handed out.
 *
 * Everything here goes through `gr::device::DeviceContext`, so a test needs no backend header and no
 * `#if` — the host domain is a `DeviceContextCpu` and `parallelFor` runs the same kernel as a host loop.
 */
class DomainContext {
public:
    DomainContext(std::string_view domain, gr::device::DeviceContext& context) : _domain(domain), _context(&context) {
        const gr::device::DeviceBuffer slots = allocate(kDeviceTestLogSlots * gr::log::kDeviceLogSlotBytes, gr::pmt::kBlobAlignment, gr::device::Residency::shared);
        _slots                               = slots.devicePointer<std::byte>();
        _counters                            = alloc<std::uint64_t>(3UZ);
        _failures                            = alloc<std::uint32_t>(1UZ);
        _counters[0]                         = 0ULL;
        _counters[1]                         = 0ULL;
        _counters[2]                         = 0ULL;
        _failures[0]                         = 0U;
    }
    DomainContext(const DomainContext&)            = delete;
    DomainContext& operator=(const DomainContext&) = delete;

    ~DomainContext() {
        for (const gr::device::DeviceBuffer& buffer : _owned) {
            _context->deallocate(buffer);
        }
    }

    [[nodiscard]] std::string_view           domain() const noexcept { return _domain; }
    [[nodiscard]] gr::device::DeviceContext& context() const noexcept { return *_context; }

    /// host-writable and device-readable, released with the context
    template<typename T>
    [[nodiscard]] T* alloc(std::size_t count) {
        const gr::device::DeviceBuffer buffer = _context->template allocateShared<T>(count);
        _owned.push_back(buffer);
        return buffer.template devicePointer<T>();
    }

    /// when a test needs device-only memory or explicit transfers; an empty buffer means this domain cannot serve it
    [[nodiscard]] gr::device::DeviceBuffer allocate(std::size_t bytes, std::size_t align, gr::device::Residency residency) {
        const gr::device::DeviceBuffer buffer = _context->allocate(bytes, align, residency);
        if (buffer) {
            _owned.push_back(buffer);
        }
        return buffer;
    }

    /// device-only where the domain has such memory, shared where it does not -- the host has no device-only
    /// residency, so a test written once for every domain would otherwise be handed a null pointer
    template<typename T>
    [[nodiscard]] T* allocDeviceOrShared(std::size_t count) {
        if (const gr::device::DeviceBuffer buffer = allocate(count * sizeof(T), alignof(T), gr::device::Residency::devicePtr); buffer) {
            return buffer.template devicePointer<T>();
        }
        return alloc<T>(count);
    }

    void upload(const void* host, gr::device::DeviceBuffer destination, std::size_t bytes) { _context->upload(host, destination, bytes); }
    void download(gr::device::DeviceBuffer source, void* host, std::size_t bytes) { _context->download(source, host, bytes); }

    /// one invocation of the kernel, on the device where there is one and as a host loop where there is not
    template<typename TKernel>
    void launch(TKernel kernel) {
        launchRange(1UZ, [kernel](const DeviceTestHandle& handle, std::size_t) { kernel(handle); });
    }

    /// `count` invocations, each given its index
    template<typename TKernel>
    void launchRange(std::size_t count, TKernel kernel) {
        const DeviceTestHandle handle = deviceHandle();
        gr::device::parallelFor(*_context, count, [kernel, handle](std::size_t i) { kernel(handle, i); });
    }

    [[nodiscard]] DeviceTestHandle deviceHandle() const noexcept { return DeviceTestHandle{.log = gr::log::DeviceLogger{.slab = slab()}, .failures = _failures}; }

    [[nodiscard]] gr::log::DeviceLogSlab slab() const noexcept { return gr::log::DeviceLogSlab{.slots = _slots, .cursor = &_counters[0], .dropped = &_counters[1], .truncated = &_counters[2], .slotCount = static_cast<std::uint32_t>(kDeviceTestLogSlots), .slotBytes = gr::log::kDeviceLogSlotBytes}; }

    /// every failure the kernels reported, rendered, each prefixed with the file and line that raised it
    [[nodiscard]] std::vector<std::string> failures() const {
        std::vector<std::string> reported;
        const std::uint32_t      published = slab().published();
        for (std::uint32_t i = 0U; i < published; ++i) {
            const std::span<const std::byte> bytes = slab().slot(i);
            if (!gr::log::detail::hasBlobMagic(bytes)) {
                continue; // claimed but never formatted
            }
            const gr::pmt::ValueMap record = gr::pmt::ValueMap::makeView(bytes);
            std::string_view        file   = gr::log::detail::readString(record, gr::log::kKeyFile);
            if (const std::size_t slash = file.find_last_of('/'); slash != std::string_view::npos) {
                file = file.substr(slash + 1UZ);
            }
            const std::string message = gr::log::detail::renderMessage(record);
            reported.push_back(file.empty() ? message : std::format("{}:{}: {}", file, gr::log::detail::readScalar(record, gr::log::kKeyLine), message));
        }
        if (const std::uint64_t dropped = slab().droppedRecords(); dropped != 0ULL) {
            reported.push_back(std::format("({} further records dropped: the slab holds {})", dropped, kDeviceTestLogSlots));
        }
        return reported;
    }

private:
    std::string_view                      _domain;
    gr::device::DeviceContext*            _context{};
    std::vector<gr::device::DeviceBuffer> _owned;
    std::byte*                            _slots{};
    std::uint64_t*                        _counters{};
    std::uint32_t*                        _failures{};
};

namespace detail {

/// the host domain is a CPU context of our own; the rest come from the registry
[[nodiscard]] inline gr::device::DeviceContext* contextForDomain(std::string_view domain, gr::device::DeviceContextCpu& hostContext) {
    if (domain == kHostDomain) {
        return &hostContext;
    }
    std::ignore = gr::device::registerSyclRuntime();
    if (!gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
        return nullptr;
    }
    return gr::device::DeviceContextRegistry::instance().tryResolve(domain);
}

} // namespace detail

template<typename TBody>
struct SweptBody {
    TBody     body;
    DomainSet domains;
};

/// `|` binds tighter than `=`, so the body and the domain list pair up first and the name is applied to the pair
template<typename TBody>
[[nodiscard]] SweptBody<std::decay_t<TBody>> operator|(TBody&& body, const DomainSet& domains) {
    return SweptBody<std::decay_t<TBody>>{std::forward<TBody>(body), domains};
}

/// runs the body once per listed domain as its own boost::ut case, and stops at the first domain that fails
template<typename TBody>
void runSweep(std::string_view name, TBody& body, const DomainSet& domains) {
    gr::device::DeviceContextCpu hostContext;
    for (std::string_view domain : domains.domains) {
        gr::device::DeviceContext* context = detail::contextForDomain(domain, hostContext);
        if (context == nullptr) {
            // a lane configured for this domain must reach it rather than quietly skip
            boost::ut::expect(!deviceDomainRequired(domain)) << std::format("GR4_REQUIRE_DEVICE names '{}', so it must be served", domain);
            continue;
        }

        bool domainFailed                                                     = false;
        boost::ut::test(std::string(name) + " [" + std::string(domain) + "]") = [&] {
            DomainContext domainContext{domain, *context};
            if constexpr (std::invocable<TBody&, DomainContext&>) {
                body(domainContext); // a generic `auto&` body lands here, which is the common case
            } else {
                domainContext.launch(body); // a body naming `const DeviceTestHandle&` IS the kernel
            }
            for (const std::string& failure : domainContext.failures()) {
                domainFailed = true;
                boost::ut::expect(false) << std::format("{}: {}", domain, failure);
            }
        };
        if (domainFailed) {
            return; // the same failure repeated per domain is noise
        }
    }
}

/// a graph test parameterised by compute domain: the fixture keeps the graph and scheduler alive, and the
/// domain is what the blocks under test should declare. Bodies take `(GraphFixture&, std::string_view domain)`.
template<typename TBody>
struct SweptGraphBody {
    TBody     body;
    DomainSet domains;
};

struct GraphSweep {
    DomainSet domains;
};

[[nodiscard]] inline GraphSweep overGraphs(const DomainSet& domains) { return GraphSweep{domains}; }

template<typename TBody>
[[nodiscard]] SweptGraphBody<std::decay_t<TBody>> operator|(TBody&& body, const GraphSweep& sweep) {
    return SweptGraphBody<std::decay_t<TBody>>{std::forward<TBody>(body), sweep.domains};
}

template<typename TBody>
void runGraphSweep(std::string_view name, TBody& body, const DomainSet& domains) {
    std::ignore = gr::device::registerSyclRuntime(); // without this only the host is ever reported as served
    for (std::string_view domain : domains.domains) {
        if (domain != kHostDomain && !gr::device::DeviceContextRegistry::instance().isServedExactly(domain)) {
            boost::ut::expect(!deviceDomainRequired(domain)) << std::format("GR4_REQUIRE_DEVICE names '{}', so it must be served", domain);
            continue;
        }
        boost::ut::test(std::string(name) + " [" + std::string(domain) + "]") = [&] {
            GraphFixture fixture;
            body(fixture, domain);
        };
    }
}

struct DomainTestName {
    std::string_view name;

    template<typename TBody>
    void operator=(const SweptBody<TBody>& swept) const {
        runSweep(name, const_cast<TBody&>(swept.body), swept.domains);
    }

    template<typename TBody>
    void operator=(const SweptGraphBody<TBody>& swept) const {
        runGraphSweep(name, const_cast<TBody&>(swept.body), swept.domains);
    }
};

/**
 * `"description"_domain_test = [](auto& ctx) { ... } | gr::testing::kAllDomains;`
 *
 * A body taking only a `DeviceTestHandle` is the kernel and is launched for you; one taking a `DomainContext&`
 * drives its own setup and launches. Write these inside `main()`: AdaptiveCpp does not run namespace-scope static
 * initialisers in a translation unit holding a kernel, so a `boost::ut` suite object there never registers and
 * the test passes green having run nothing.
 */
[[nodiscard]] inline DomainTestName operator""_domain_test(const char* text, std::size_t length) { return DomainTestName{std::string_view{text, length}}; }

} // namespace gr::testing

#endif // GNURADIO_TEST_DEVICE_TEST_HELPER_HPP
