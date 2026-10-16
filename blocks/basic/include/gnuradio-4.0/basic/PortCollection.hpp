#ifndef GNURADIO_BASIC_PORTCOLLECTION_HPP
#define GNURADIO_BASIC_PORTCOLLECTION_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic::detail {

/** `nPorts` 0: a port vector sized by a setting at run time; N: N ports fixed at compile time */
template<typename TPort, std::size_t nPorts>
using PortsOf = std::conditional_t<nPorts == 0UZ, std::vector<TPort>, std::array<TPort, nPorts>>;

inline constexpr std::size_t kMaxPorts     = 32UZ;
inline constexpr gr::Size_t  kMaxRunLength = 1U << 24U;

[[nodiscard]] inline property_map sampleRateTag(float rate) { return property_map{{std::string(gr::tag::SAMPLE_RATE.key()), rate}}; }

/** the data pointers of a port set, for a kernel that indexes ports by number */
template<typename TPointer>
[[nodiscard]] std::array<TPointer, kMaxPorts> dataOf(auto& spans) noexcept {
    std::array<TPointer, kMaxPorts> pointers{};
    for (std::size_t port = 0UZ; port < spans.size(); ++port) {
        pointers[port] = spans[port].data();
    }
    return pointers;
}

/** true when a connected port's buffer holds `chunk` samples; else the block reports `what` and stops */
[[nodiscard]] bool chunkFitsBuffer(auto& block, const auto& port, std::size_t chunk, std::string_view what) {
    if (port.isConnected() && chunk > port.bufferSize()) [[unlikely]] {
        block.emitErrorMessage("settingsChanged()", std::format("{} exceeds the buffer of {} samples", what, port.bufferSize()));
        block.requestStop();
        return false;
    }
    return true;
}

/** a changed `n_ports` of a block whose ports are fixed, at compile time or once one is connected, is refused: the setting keeps the port count */
template<std::size_t nPorts>
void refuseNewPortCount(auto& block, const property_map& newSettings, const auto& ports) {
    const bool connected = std::ranges::any_of(ports, [](const auto& port) { return port.isConnected(); });
    if (newSettings.contains("n_ports") && (nPorts != 0UZ || connected) && block.n_ports.value != ports.size()) {
        block.emitErrorMessage("settingsChanged()", std::format("n_ports is fixed at compile time or once connected: keeps {}", ports.size()));
        block.n_ports = static_cast<gr::Size_t>(ports.size());
    }
}

} // namespace gr::basic::detail

#endif // GNURADIO_BASIC_PORTCOLLECTION_HPP
