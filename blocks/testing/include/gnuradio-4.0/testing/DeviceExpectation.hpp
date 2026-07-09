#ifndef GNURADIO_TESTING_DEVICE_EXPECTATION_HPP
#define GNURADIO_TESTING_DEVICE_EXPECTATION_HPP

#include <cstdlib>
#include <string_view>

namespace gr::testing {

/**
 * @brief Whether this run is required to exercise a device domain, rather than being allowed to skip it.
 *
 * A device test skips on a machine that cannot serve the backend — otherwise nobody could run the suite on a
 * laptop — but a lane configured FOR a backend that then stopped serving it would pass green having asserted
 * nothing. `GR4_REQUIRE_DEVICE` lists the domains this run must actually reach, comma-separated
 * (`gpu:sycl,host:sycl`); the build sets it from its own configuration. Matching is on whole entries, so
 * `gpu:sycl` does not satisfy a required `gpu:sycl:0`.
 */
[[nodiscard]] inline bool deviceDomainRequired(std::string_view domain) noexcept {
    const char* const required = std::getenv("GR4_REQUIRE_DEVICE");
    if (required == nullptr) {
        return false;
    }
    std::string_view list{required};
    while (!list.empty()) {
        const std::size_t      comma = list.find(',');
        const std::string_view entry = list.substr(0UZ, comma);
        if (entry == domain) {
            return true;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        list.remove_prefix(comma + 1UZ);
    }
    return false;
}

} // namespace gr::testing

#endif // GNURADIO_TESTING_DEVICE_EXPECTATION_HPP
