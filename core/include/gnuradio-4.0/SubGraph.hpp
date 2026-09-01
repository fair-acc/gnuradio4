#ifndef GNURADIO_SUBGRAPH_HPP
#define GNURADIO_SUBGRAPH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/SchedulerModel.hpp>

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
struct SubGraph : gr::Block<SubGraph> {
    using Description = Doc<"a block group whose in-built scheduler runs its members synchronously from work()">;

    GR_MAKE_REFLECTABLE(SubGraph);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    meta::indirect<gr::Graph>                _graph{};
    std::vector<std::shared_ptr<BlockModel>> _order; // topological; empty until startDispatch()
    bool                                     _quiescent = false;

    [[nodiscard]] const gr::Graph& graph() const noexcept { return *_graph; }
    [[nodiscard]] gr::Graph&       graph() noexcept { return *_graph; }

    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<Edge>                              edges() noexcept { return _graph->edges(); }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept { return _graph->edges(); }

    void setGraph(gr::Graph&& newGraph) { _graph = meta::indirect<gr::Graph>(std::move(newGraph)); } // gr::Graph move-assignment is deleted

    // not only in makeSubGraph(): setGraph() replaces the graph wholesale, so the invariant belongs to a RUNNING group
    void startDispatch() {
        if (auto singleDomain = gr::refuseTwoDeviceDomains(*_graph); !singleDomain) {
            gr::log::error("{}", singleDomain.error().message);
        }
        _order      = topologicalOrder();
        std::ignore = _graph->connectPendingEdges(); // interior edges only; the boundary ones belong to the parent
        for (auto& member : _order) {
            std::ignore = member->changeStateTo(lifecycle::State::INITIALISED);
            std::ignore = member->changeStateTo(lifecycle::State::RUNNING);
        }
    }

    void stopDispatch() {
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

    [[nodiscard]] work::Result work(std::size_t requestedWork) noexcept {
        if (_quiescent) {
            return {requestedWork, 0UZ, work::Status::OK};
        }
        std::size_t performed  = 0UZ;
        bool        unfinished = false;
        for (auto& member : _order) {
            const work::Result result = member->work(requestedWork);
            performed += result.performed_work;
            if (result.status == work::Status::ERROR) {
                return {requestedWork, performed, work::Status::ERROR};
            }
            unfinished = unfinished || result.status != work::Status::DONE;
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
class SubGraphWrapper : public GraphWrapper<SubGraph, gr::Graph>, public SchedulerModel {
public:
    explicit SubGraphWrapper(gr::property_map props = {}) : GraphWrapper<SubGraph, gr::Graph>(std::move(props)) {}

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

struct SubGraphHandle {
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
 * @brief Wraps an already-built sub-graph into a domain block, exporting every port no interior edge claims.
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

/// what makeSubGraph would export, for callers that must act on the boundary before the group exists
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
        const std::string_view domain          = setting->value_or(std::string_view{});
        const bool             namesThreadPool = domain.empty() || domain == "host" || domain == gr::thread_pool::kDefaultIoPoolId || domain == gr::thread_pool::kDefaultCpuPoolId;
        return namesThreadPool ? std::string{} : std::string(domain);
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
 * @brief Wraps an already-built sub-graph into a domain block, exporting every port no interior edge claims.
 *
 * The returned handle carries the block to add to the parent graph and its boundary port names. Two members that
 * would export the same name are refused with an error naming them.
 */
[[nodiscard]] inline std::expected<SubGraphHandle, Error> makeSubGraph(gr::Graph&& members, const std::set<std::string>& doNotExport = {}, std::source_location location = std::source_location::current()) {
    if (auto refusal = refuseTwoDeviceDomains(members, location); !refusal) {
        return std::unexpected(refusal.error());
    }
    auto boundaries = boundaryPorts(members, doNotExport, location);
    if (!boundaries) {
        return std::unexpected(boundaries.error());
    }
    const std::vector<BoundaryPort>& boundaryInputs  = boundaries->inputs;
    const std::vector<BoundaryPort>& boundaryOutputs = boundaries->outputs;

    // deliberately untouched: `settings().set()` through a BlockModel never reaches the member's field

    SubGraphHandle handle;
    handle.block  = std::static_pointer_cast<BlockModel>(std::make_shared<SubGraphWrapper>());
    auto* wrapper = static_cast<SubGraphWrapper*>(handle.block.get());
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

#endif // GNURADIO_SUBGRAPH_HPP
