#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

namespace gr {

Graph::Graph(property_map settings) : gr::Block<Graph>(std::move(settings)), _pluginLoader(std::addressof(gr::globalPluginLoader())) {
    _blocks.reserve(100); // TODO: remove

    propertyCallbacks[graph::property::kInspectBlock]           = std::mem_fn(&Graph::propertyCallbackInspectBlock);
    propertyCallbacks[graph::property::kGraphInspect]           = std::mem_fn(&Graph::propertyCallbackGraphInspect);
    propertyCallbacks[graph::property::kRegistryBlockTypes]     = std::mem_fn(&Graph::propertyCallbackRegistryBlockTypes);
    propertyCallbacks[graph::property::kRegistrySchedulerTypes] = std::mem_fn(&Graph::propertyCallbackRegistrySchedulerTypes);
}

[[maybe_unused]] std::shared_ptr<BlockModel> const& Graph::emplaceBlock(std::string_view type, property_map initialSettings) {
    if (type.starts_with("gr::Graph")) {
        auto subGraphModel = std::unique_ptr<BlockModel>(std::make_unique<GraphWrapper<Graph>>().release());
        return addBlock(std::move(subGraphModel));
    } else if (std::shared_ptr<BlockModel> block_load = _pluginLoader->instantiate(type, std::move(initialSettings)); block_load) {
        const std::shared_ptr<BlockModel>& newBlock = addBlock(block_load);
        return newBlock;
    } else if (std::shared_ptr<SchedulerModel> scheduler_load = _pluginLoader->instantiateScheduler(type, std::move(initialSettings)); scheduler_load) {
        const std::shared_ptr<BlockModel>& newBlock = addBlock(SchedulerModel::asBlockModelPtr(scheduler_load));
        return newBlock;
    }
    throw gr::exception(std::format("Cannot create block '{}'", type));
}

std::pair<std::shared_ptr<BlockModel>, std::shared_ptr<BlockModel>> Graph::replaceBlock(const std::string& uniqueName, const std::string& type, const property_map& properties) {
    auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
    if (it == _blocks.end()) {
        throw gr::exception(std::format("Block {} was not found in {}", uniqueName, this->unique_name));
    }

    auto newBlock = gr::globalPluginLoader().instantiate(type, properties);
    if (!newBlock) {
        throw gr::exception(std::format("Can not create block {}", type));
    }

    addBlock(newBlock);

    for (auto& edge : _edges) {
        if (edge._sourceBlock == *it) {
            edge._sourceBlock = newBlock;
        }

        if (edge._destinationBlock == *it) {
            edge._destinationBlock = newBlock;
        }
    }

    std::shared_ptr<BlockModel> oldBlock = std::move(*it);
    _blocks.erase(it);

    return {std::move(oldBlock), newBlock};
}

std::optional<Message> Graph::propertyCallbackRegistryBlockTypes([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kRegistryBlockTypes);
    message.data = property_map{{"types", _pluginLoader->availableBlocks()}};
    return message;
}

std::optional<Message> Graph::propertyCallbackRegistrySchedulerTypes([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kRegistryBlockTypes);
    message.data = property_map{{"types", _pluginLoader->availableSchedulers()}};
    return message;
}

std::optional<Message> Graph::propertyCallbackInspectBlock([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kInspectBlock);
    using namespace std::string_literals;
    const auto&        data       = message.data.value();
    const std::string& uniqueName = std::get<std::string>(data.at("uniqueName"s));
    using namespace std::string_literals;

    auto it = std::ranges::find_if(_blocks, [&uniqueName](const auto& block) { return block->uniqueName() == uniqueName; });
    if (it == _blocks.end()) {
        throw gr::exception(std::format("Block {} was not found in {}", uniqueName, this->unique_name));
    }

    gr::Message reply;
    reply.endpoint = graph::property::kBlockInspected;
    reply.data     = serializeBlock(*_pluginLoader, *it, BlockSerializationFlags::All);
    return {reply};
}

std::optional<Message> Graph::propertyCallbackGraphInspect([[maybe_unused]] std::string_view propertyName, Message message) {
    assert(propertyName == graph::property::kGraphInspect);
    message.data = [&] {
        property_map result;
        result[std::string(serialization_fields::BLOCK_NAME)]        = std::string(name);
        result[std::string(serialization_fields::BLOCK_UNIQUE_NAME)] = std::string(unique_name);
        result[std::string(serialization_fields::BLOCK_CATEGORY)]    = std::string(magic_enum::enum_name(blockCategory));

        property_map serializedChildren;
        for (const auto& child : blocks()) {
            serializedChildren[std::string(child->uniqueName())] = serializeBlock(*_pluginLoader, child, BlockSerializationFlags::All);
        }
        result[std::string(serialization_fields::BLOCK_CHILDREN)] = std::move(serializedChildren);

        property_map serializedEdges;
        std::size_t  index = 0UZ;
        for (const auto& edge : edges()) {
            serializedEdges[std::to_string(index)] = serializeEdge(edge);
            index++;
        }
        result[std::string(serialization_fields::BLOCK_EDGES)] = std::move(serializedEdges);
        return result;
    }();

    message.endpoint = graph::property::kGraphInspected;
    return message;
}
} // namespace gr
