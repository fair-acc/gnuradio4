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

/// wraps `members` into a SubGraph, inserting a `HostToDevice`/`DeviceToHost` transfer at every boundary port
/// that no interior edge already claims, then exports the transfers' outer ports.
template<typename T>
[[nodiscard]] inline std::expected<gr::SubGraphHandle, Error> makeDeviceSubGraph(gr::Graph&& members, std::source_location location = std::source_location::current()) {
    const auto boundaries = gr::boundaryPorts(members, {}, location);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }

    // the group's domain is the graph's own: `Graph::addBlock` hands it to every member's `init()`, so a member that
    // named no domain of its own already carries it, and so will the transfers emplaced below. A member that DID
    // name one keeps it -- its init parameters are applied after `init()` assigns the graph's -- which is what makes
    // a host member inside a device group legal (`refuseTwoDeviceDomains`).
    std::size_t nInserted        = 0UZ;
    const auto  transferSettings = [&](std::string_view rolePrefix) {
        return gr::property_map{{"name", std::format("{}_{}", rolePrefix, nInserted++)}}; // an index: a unique name carries "::" and "#"
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
