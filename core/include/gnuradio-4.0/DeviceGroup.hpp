#ifndef GNURADIO_DEVICE_GROUP_HPP
#define GNURADIO_DEVICE_GROUP_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>
#include <gnuradio-4.0/meta/indirect.hpp>

#include <algorithm>
#include <expected>
#include <format>
#include <memory>
#include <ranges>
#include <set>
#include <source_location>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace gr {

[[nodiscard]] inline std::expected<void, Error> refuseTwoDeviceDomains(const gr::Graph& members, std::source_location location = std::source_location::current());

/**
 * @brief A group of blocks sharing one compute domain, run synchronously by one in-built scheduler from work().
 *
 * The caller drives it rather than it running a work loop of its own, which is what single-threaded device
 * submission needs. Members run in topological order over the inner graph's edges.
 */
struct DeviceGroup : gr::Block<DeviceGroup> {
    using Description = Doc<"a block group whose in-built scheduler runs its members synchronously from work()">;

    GR_MAKE_REFLECTABLE(DeviceGroup);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    meta::indirect<gr::Graph>                _graph{};
    std::vector<std::shared_ptr<BlockModel>> _order; // topological; empty until startDispatch()
    bool                                     _quiescent    = false;
    device::DeviceContext*                   _groupContext = nullptr; // the group's single device domain, latched at start; NOT Block::_deviceContext, which is this group block's own residency

    [[nodiscard]] const gr::Graph& graph() const noexcept { return *_graph; }
    [[nodiscard]] gr::Graph&       graph() noexcept { return *_graph; }

    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<Edge>                              edges() noexcept { return _graph->edges(); }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept { return _graph->edges(); }

    void setGraph(gr::Graph&& newGraph) { _graph = meta::indirect<gr::Graph>(std::move(newGraph)); } // gr::Graph move-assignment is deleted

    // not only in makeDeviceGroup(): setGraph() replaces the graph wholesale, so the invariant belongs to a RUNNING group
    void startDispatch() {
        _order        = topologicalOrder();
        _groupContext = nullptr; // cleared before the refusal below can return, or a restart keeps the last one
        if (auto singleDomain = gr::refuseTwoDeviceDomains(*_graph); !singleDomain) {
            failGroup(singleDomain.error().message);
            return;
        }
        // a member that is itself a group would need its own children started, its own interior edges connected and
        // its own backend propagated -- none of which this walk does. Left in, it reports OK with zero progress
        // forever and the parent spins with no error, so it is refused where it can still be diagnosed.
        for (const auto& member : _order) {
            if (member->blockCategory() != block::Category::NormalBlock) {
                failGroup(std::format("device group '{}': member '{}' is itself a block group, which a group cannot drive", this->name.value, member->name()));
                return;
            }
        }

        std::ignore = _graph->connectPendingEdges(); // interior edges only; the boundary ones belong to the parent

        // at most one device domain across the members is already an invariant, so the first one found is the
        // group's: it is the queue every member's work is enqueued on, and therefore the one barrier to take.
        // This has to settle before any member starts, because starting is when a member decides its residency --
        // a member that reaches RUNNING without the group's context would re-seat its fields onto device memory and
        // then be run on the host.
        for (const auto& member : _order) {
            const auto             setting = member->settings().get("compute_domain");
            const std::string_view domain  = setting ? setting->value_or(std::string_view{}) : std::string_view{};
            if (!ComputeDomain::parse(domain).isDevice()) {
                continue;
            }
            // the same decision a scheduler makes for a loose block, so a member cannot be answered by a device it
            // did not ask for merely because it is inside a group
            const device::DomainOutcome outcome = device::resolveDeclaredDomain(domain);
            if (outcome.refused) {
                failGroup(std::format("device group '{}': member '{}' compute_domain '{}' is required and {}", this->name.value, member->name(), outcome.declared, outcome.reason));
                return;
            }
            if (outcome.context == nullptr) {
                gr::log::warning("device group '{}': member '{}' compute_domain '{}' {} — functional fallback to 'host'; spell it '{}!' to make this a stop", this->name.value, member->name(), outcome.declared, outcome.reason, outcome.declared);
                continue; // keep scanning: a later member may name the same domain as a requirement
            }
            if (_groupContext == nullptr) {
                _groupContext = outcome.context; // at most one device domain across the members is already an invariant
            }
        }
        if (_groupContext != nullptr) {
            for (auto& member : _order) {
                member->setComputeBackend(*_groupContext);
            }
        }

        for (auto& member : _order) {
            if (auto initialised = member->changeStateTo(lifecycle::State::INITIALISED); !initialised) {
                failGroup(std::format("device group '{}': member '{}' refused to initialise: {}", this->name.value, member->name(), initialised.error().message));
                return;
            }
            if (auto running = member->changeStateTo(lifecycle::State::RUNNING); !running) {
                failGroup(std::format("device group '{}': member '{}' refused to start: {}", this->name.value, member->name(), running.error().message));
                return;
            }
        }
    }

    /// a member's refusal is the group's: it names the member, because the parent only ever sees the group
    void failGroup(std::string_view reason) {
        gr::log::error("{}", reason);
        for (auto& member : _order) {
            std::ignore = member->changeStateTo(lifecycle::State::ERROR);
        }
    }

    void stopDispatch() {
        // no group drain here: every member drains the shared queue on its own way to STOPPED, and the first one to
        // do so has already waited for every member's work -- a group-level drain would only add a barrier to count
        for (auto& member : _order) {
            std::ignore = member->changeStateTo(lifecycle::State::REQUESTED_STOP);
            std::ignore = member->changeStateTo(lifecycle::State::STOPPED);
        }
    }

    // drops everything derived from the old topology; the next startDispatch() recomputes it
    void removeMembers(std::span<const std::shared_ptr<BlockModel>> blocksToRemove) {
        _order.clear();
        for (const auto& block : blocksToRemove) {
            std::ignore = _graph->removeBlockByName(block->uniqueName());
        }
    }

    // no wait loop: work() only ever runs on the thread that calls it
    void requestWorkQuiescence() { _quiescent = true; }
    void releaseWorkQuiescence() { _quiescent = false; }

    // the group's own backend is not used to run its members: each member carries the one the group resolved for it
    // before it started, and passing anything else would contradict the residency it already committed to
    [[nodiscard]] work::Result work(std::size_t requestedWork, [[maybe_unused]] device::DeviceContext& computeBackend = device::hostBackend()) noexcept {
        if (_quiescent) {
            return {requestedWork, 0UZ, work::Status::OK};
        }
        // one cut for the chain: every member commits what is staged before any member runs, so a settings change
        // lands on the same chunk throughout instead of one chunk later at each hop
        for (auto& member : _order) {
            if (member->settings().changed()) {
                member->applyStagedSettings();
            }
        }

        std::size_t performed  = 0UZ;
        bool        unfinished = false;
        for (auto& member : _order) {
            const work::Result result = member->work(requestedWork, member->computeBackend());
            performed += result.performed_work;
            if (result.status == work::Status::ERROR) {
                return {requestedWork, performed, work::Status::ERROR};
            }
            unfinished = unfinished || result.status != work::Status::DONE;
        }
        // peek, not poll: the members' kernels were enqueued rather than awaited, and taking a barrier here would
        // add one per work() call to a chain whose interior hops already cost none. The boundary transfers drain the
        // queue; this only gives a fault a place to be attributed to the group rather than to whoever polls next.
        if (_groupContext != nullptr) {
            if (auto deviceErr = _groupContext->peekDeviceError()) {
                gr::log::error("device group '{}': device fault in a member's kernel: {}", this->name.value, *deviceErr);
                return {requestedWork, performed, work::Status::ERROR};
            }
        }
        return {requestedWork, performed, unfinished ? work::Status::OK : work::Status::DONE};
    }
    [[nodiscard]] std::vector<std::shared_ptr<BlockModel>> topologicalOrder() const {
        const auto blocks = _graph->blocks();
        const auto edges  = _graph->edges();

        std::unordered_map<const BlockModel*, std::size_t> unresolvedPredecessors;
        for (const auto& member : blocks) {
            unresolvedPredecessors[member.get()] = 0UZ;
        }
        for (const Edge& edge : edges) {
            if (edge.destinationBlock()) {
                ++unresolvedPredecessors[edge.destinationBlock().get()];
            }
        }

        const auto releaseSuccessorsOf = [&](const BlockModel* member) {
            for (const Edge& edge : edges) {
                if (edge.sourceBlock().get() == member && edge.destinationBlock()) {
                    --unresolvedPredecessors[edge.destinationBlock().get()];
                }
            }
        };

        std::vector<std::shared_ptr<BlockModel>> order;
        order.reserve(blocks.size());
        std::unordered_set<const BlockModel*> ordered;

        for (bool progressed = true; progressed;) {
            progressed = false;
            for (const auto& member : blocks) {
                if (ordered.contains(member.get()) || unresolvedPredecessors[member.get()] != 0UZ) {
                    continue;
                }
                order.push_back(member);
                ordered.insert(member.get());
                releaseSuccessorsOf(member.get());
                progressed = true;
            }
        }

        std::ranges::copy_if(blocks, std::back_inserter(order), [&](const auto& member) { return !ordered.contains(member.get()); });
        return order;
    }
};

// SchedulerWrapper without the std::thread: start() primes the members and returns
class DeviceGroupWrapper : public GraphWrapper<DeviceGroup, gr::Graph>, public SchedulerModel {
public:
    explicit DeviceGroupWrapper(gr::property_map props = {}) : GraphWrapper<DeviceGroup, gr::Graph>(std::move(props)) {}

    void            setGraph(gr::Graph&& graph) final { this->blockRef().setGraph(std::move(graph)); }
    BlockModel*     asBlockModel() final { return static_cast<BlockModel*>(this); }
    SchedulerModel* asSchedulerModel() noexcept override { return this; }

    void start() override { this->blockRef().startDispatch(); }
    void stop() override { this->blockRef().stopDispatch(); }

    // members run on whichever thread calls work(), so quiescence is a flag rather than a wait
    void requestWorkQuiescenceAll() override { this->blockRef().requestWorkQuiescence(); }
    void releaseWorkQuiescenceAll() override { this->blockRef().releaseWorkQuiescence(); }

    void blockUntilWorking() override {}

    // contractually called only within quiescence
    void removeBlocks(std::span<const std::shared_ptr<BlockModel>> blocksToRemove) final { this->blockRef().removeMembers(blocksToRemove); }
};

struct DeviceGroupHandle {
    std::shared_ptr<BlockModel> block;
    std::vector<std::string>    inputs; // exported boundary port names, in member order
    std::vector<std::string>    outputs;
};

namespace detail {

[[nodiscard]] inline std::string resolvePortName(BlockModel& block, PortDirection direction, const PortDefinition& definition) {
    if (const auto* named = std::get_if<PortDefinition::StringBased>(&definition.definition)) {
        return named->name;
    }
    auto&             ports = direction == PortDirection::INPUT ? block.dynamicInputPorts() : block.dynamicOutputPorts();
    const std::size_t index = std::get<PortDefinition::IndexBased>(definition.definition).topLevel;
    return index < ports.size() ? BlockModel::portName(ports[index]) : std::string{};
}

[[nodiscard]] inline std::string boundaryName(std::string_view blockName, std::string_view portName) { return std::format("{}:{}", blockName, portName); }

} // namespace detail

/**
 * @brief Wraps an already-built device group into a domain block, exporting every port no interior edge claims.
 *
 * The returned handle carries the block to add to the parent graph and its boundary port names. Two members that
 * would export the same name are refused with an error naming them.
 */
/// a port no interior edge claims
struct BoundaryPort {
    std::string unique; // the member's unique name, for looking the port up
    std::string name;   // the member's user-provided name, for the exported label
    std::string port;
};

struct Boundaries {
    std::vector<BoundaryPort> inputs;
    std::vector<BoundaryPort> outputs;
};

/// what makeDeviceGroup would export, for callers that must act on the boundary before the group exists
[[nodiscard]] inline std::expected<Boundaries, Error> boundaryPorts(const gr::Graph& members, const std::set<std::string>& doNotExport = {}, std::source_location location = std::source_location::current()) {
    using PortKey = std::pair<std::string, std::string>; // block unique name, port name
    auto keyOf    = [](const BoundaryPort& b) { return PortKey{b.unique, b.port}; };

    std::set<PortKey> claimedInputs;
    std::set<PortKey> claimedOutputs;
    for (const Edge& edge : members.edges()) {
        if (edge.sourceBlock()) {
            claimedOutputs.emplace(std::string(edge.sourceBlock()->uniqueName()), detail::resolvePortName(*edge.sourceBlock(), PortDirection::OUTPUT, edge.sourcePortDefinition()));
        }
        if (edge.destinationBlock()) {
            claimedInputs.emplace(std::string(edge.destinationBlock()->uniqueName()), detail::resolvePortName(*edge.destinationBlock(), PortDirection::INPUT, edge.destinationPortDefinition()));
        }
    }

    std::vector<BoundaryPort> boundaryInputs;
    std::vector<BoundaryPort> boundaryOutputs;
    for (const auto& member : members.blocks()) {
        const std::string memberName(member->uniqueName());
        const std::string memberLabel(member->name());

        const auto collect = [&](PortDirection direction, const std::set<PortKey>& claimed, std::vector<BoundaryPort>& out) -> std::expected<void, Error> {
            auto& ports = direction == PortDirection::INPUT ? member->dynamicInputPorts() : member->dynamicOutputPorts();
            for (const auto& portOrCollection : ports) {
                if (!std::holds_alternative<gr::DynamicPort>(portOrCollection)) {
                    return std::unexpected(Error(std::format("block '{}' exposes a port collection; a domain cannot export collections yet", memberName), location));
                }
                BoundaryPort candidate{memberName, memberLabel, BlockModel::portName(portOrCollection)};
                if (claimed.contains(keyOf(candidate)) || doNotExport.contains(detail::boundaryName(candidate.name, candidate.port))) {
                    continue; // claimed by an interior edge, or the caller asked for it to stay private
                }
                out.push_back(std::move(candidate));
            }
            return {};
        };

        if (auto result = collect(PortDirection::INPUT, claimedInputs, boundaryInputs); !result) {
            return std::unexpected(result.error());
        }
        if (auto result = collect(PortDirection::OUTPUT, claimedOutputs, boundaryOutputs); !result) {
            return std::unexpected(result.error());
        }
    }

    return Boundaries{.inputs = std::move(boundaryInputs), .outputs = std::move(boundaryOutputs)};
}

/// two DEVICE domains cannot share a group: only one can own its residency, the other falls back silently
[[nodiscard]] inline std::expected<void, Error> refuseTwoDeviceDomains(const gr::Graph& members, std::source_location location) {
    const auto deviceDomainOf = [](const std::shared_ptr<BlockModel>& member) -> std::string {
        const auto setting = member->settings().get("compute_domain");
        if (!setting) {
            return {};
        }
        const std::string_view domain = setting->value_or(std::string_view{});
        // one grammar, one reader: `parse` already maps "", "host" and every thread-pool spelling to the host
        return ComputeDomain::parse(domain).isDevice() ? ComputeRegistry::instance().resolvedDomainName(domain) : std::string{};
    };
    auto declared = members.blocks()                                                       // filter_view is not
                    | std::views::transform(deviceDomainOf)                                // const-iterable, hence
                    | std::views::filter([](const std::string& d) { return !d.empty(); }); // a non-const `declared`

    const std::set<std::string> deviceDomains(declared.begin(), declared.end());
    if (deviceDomains.size() > 1UZ) {
        return std::unexpected(Error(std::format("a group may hold members of at most one device compute_domain, but these members declare {}: build one group per device domain", deviceDomains.size()), location));
    }
    return {};
}

/**
 * @brief Wraps an already-built device group into a domain block, exporting every port no interior edge claims.
 *
 * The returned handle carries the block to add to the parent graph and its boundary port names. Two members that
 * would export the same name are refused with an error naming them.
 */
[[nodiscard]] inline std::expected<DeviceGroupHandle, Error> makeDeviceGroup(gr::Graph&& members, const std::set<std::string>& doNotExport = {}, std::source_location location = std::source_location::current()) {
    if (auto refusal = refuseTwoDeviceDomains(members, location); !refusal) {
        return std::unexpected(refusal.error());
    }
    auto boundaries = boundaryPorts(members, doNotExport, location);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }
    const std::vector<BoundaryPort>& boundaryInputs  = boundaries->inputs;
    const std::vector<BoundaryPort>& boundaryOutputs = boundaries->outputs;

    // the wrapper carries the group's own domain outward. Without it the group block defaults to a host thread pool
    // and an edge between two groups is a host edge, so two device groups cannot chain device-to-device however
    // their members are placed -- the exported port would be the only host hop in an otherwise device chain.
    gr::property_map wrapperSettings;
    if (const std::string groupDomain(members.compute_domain.value); ComputeDomain::parse(groupDomain).isDevice()) {
        wrapperSettings["compute_domain"] = groupDomain;
    }

    DeviceGroupHandle handle;
    handle.block  = std::static_pointer_cast<BlockModel>(std::make_shared<DeviceGroupWrapper>(std::move(wrapperSettings)));
    auto* wrapper = static_cast<DeviceGroupWrapper*>(handle.block.get());
    wrapper->setGraph(std::move(members));

    std::set<std::string> exportedNames;
    const auto            exportAll = [&](const std::vector<BoundaryPort>& toExport, PortDirection direction, std::vector<std::string>& names) -> std::expected<void, Error> {
        for (const auto& boundary : toExport) {
            std::string exported = detail::boundaryName(boundary.name, boundary.port);
            if (!exportedNames.insert(exported).second) {
                return std::unexpected(Error(std::format("two members would export the port '{}': block names must be unique within a domain. A block with no name set takes its type name, so two unnamed members of the same type always collide", exported), location));
            }
            if (auto result = wrapper->exportPort(true, boundary.unique, direction, boundary.port, exported, location); !result) {
                return std::unexpected(result.error());
            }
            names.push_back(std::move(exported));
        }
        return {};
    };

    if (auto result = exportAll(boundaryInputs, PortDirection::INPUT, handle.inputs); !result) {
        return std::unexpected(result.error());
    }
    if (auto result = exportAll(boundaryOutputs, PortDirection::OUTPUT, handle.outputs); !result) {
        return std::unexpected(result.error());
    }

    return handle;
}

} // namespace gr

#endif // GNURADIO_DEVICE_GROUP_HPP
