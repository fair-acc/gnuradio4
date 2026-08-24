#ifndef GNURADIO_DEVICE_SUBGRAPH_HPP
#define GNURADIO_DEVICE_SUBGRAPH_HPP

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/SubGraph.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>

#include <expected>
#include <format>
#include <source_location>
#include <string>

namespace gr::basic {

/**
 * @brief Wraps a graph into a SubGraph, inserting the host/device transfers its boundary needs.
 *
 * Every port no interior edge claims gets a transfer in front of it — `HostToDevice` before a boundary input,
 * `DeviceToHost` after a boundary output — and the group exports the transfers' outer ports. Membership is what
 * declares the boundary, so nothing is ever inferred.
 *
 * `T` is named rather than deduced because the graph is type-erased here, and resolving the port type through the
 * block registry would make this unavailable in the registry-free builds the device path exists to serve.
 * `deviceDomain` is not decoration: a transfer without one takes the host `processBulk` and copies on the CPU.
 */
template<typename T>
[[nodiscard]] inline std::expected<gr::SubGraphHandle, Error> makeDeviceSubGraph(gr::Graph&& members, std::string_view deviceDomain, std::source_location location = std::source_location::current()) {
    const auto boundaries = gr::boundaryPorts(members, {}, location);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }

    // a transfer only transfers when its compute_domain names a device; without one it takes the host processBulk
    // and copies on the CPU, so a host group leaves the setting alone
    const bool namesADevice = !deviceDomain.empty() && deviceDomain != "host" && deviceDomain != gr::thread_pool::kDefaultIoPoolId && deviceDomain != gr::thread_pool::kDefaultCpuPoolId;

    std::size_t nInserted        = 0UZ;
    const auto  transferSettings = [&](std::string_view rolePrefix) {
        gr::property_map init{{"name", std::format("{}_{}", rolePrefix, nInserted++)}}; // an index: a unique name carries "::" and "#"
        if (namesADevice) {
            init["compute_domain"] = std::string(deviceDomain);
        }
        return init;
    };

    const auto uploadInFrontOf = [&](const gr::BoundaryPort& boundary) -> std::expected<void, Error> {
        auto member = gr::graph::findBlock(members, std::string_view(boundary.unique), location);
        if (!member) {
            return std::unexpected(member.error());
        }
        auto& transfer = members.emplaceBlock<HostToDevice<T>>(transferSettings("h2d"));
        auto  inserted = gr::graph::findBlock(members, std::string_view(transfer.unique_name), location);
        if (!inserted) {
            return std::unexpected(inserted.error());
        }
        return members.connect(*inserted, PortDefinition("out"), *member, PortDefinition(boundary.port), {}, location);
    };

    const auto downloadBehind = [&](const gr::BoundaryPort& boundary) -> std::expected<void, Error> {
        auto member = gr::graph::findBlock(members, std::string_view(boundary.unique), location);
        if (!member) {
            return std::unexpected(member.error());
        }
        auto& transfer = members.emplaceBlock<DeviceToHost<T>>(transferSettings("d2h"));
        auto  inserted = gr::graph::findBlock(members, std::string_view(transfer.unique_name), location);
        if (!inserted) {
            return std::unexpected(inserted.error());
        }
        return members.connect(*member, PortDefinition(boundary.port), *inserted, PortDefinition("in"), {}, location);
    };

    for (const gr::BoundaryPort& boundary : boundaries->inputs) {
        if (auto inserted = uploadInFrontOf(boundary); !inserted) {
            return std::unexpected(inserted.error());
        }
    }
    for (const gr::BoundaryPort& boundary : boundaries->outputs) {
        if (auto inserted = downloadBehind(boundary); !inserted) {
            return std::unexpected(inserted.error());
        }
    }

    return gr::makeSubGraph(std::move(members), {}, location);
}

} // namespace gr::basic

#endif // GNURADIO_DEVICE_SUBGRAPH_HPP
