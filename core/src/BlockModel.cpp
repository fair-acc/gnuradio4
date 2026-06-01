#include <string>

#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

namespace {
void serializeBlockSettings(gr::property_map& output, gr::BlockModel& block) {
    using namespace gr;
    auto writeParameters = [&](const property_map& settingsMap) {
        property_map parameters;
        for (const auto& [settingsKey, settingsValue] : settingsMap) {
            parameters.insert_or_assign(std::string_view{settingsKey}, settingsValue);
        }
        return parameters;
    };

    std::ignore        = block.settings().applyStagedParameters(); // called for its side effect; unapplied-set unused here
    const auto& stored = block.settings().getStoredAll();

    output.emplace(serialization_fields::BLOCK_PARAMETERS, writeParameters(block.settings().get()));

    Tensor<Value> ctxParamsSeq;
    ctxParamsSeq.reserve(stored.size());
    for (const auto& [ctx, ctxParameters] : stored) {
        if (ctx.empty()) { // skip default context
            continue;
        }
        for (const auto& [ctxTime, settingsMap] : ctxParameters) {
            property_map ctxParam;
            ctxParam.emplace(gr::tag::CONTEXT.shortKey(), ctxTime.context);
            ctxParam.emplace(gr::tag::CONTEXT_TIME.shortKey(), ctxTime.time);
            ctxParam.emplace(serialization_fields::BLOCK_PARAMETERS, writeParameters(settingsMap));
            ctxParamsSeq.emplace_back(std::move(ctxParam));
        }
    }
    output.emplace(serialization_fields::BLOCK_CTX_PARAMETERS, std::move(ctxParamsSeq));
}
} // namespace

namespace gr {
property_map serializeBlockImpl(gr::PluginLoader& pluginLoader, const std::shared_ptr<BlockModel>& block, int flags) {
    using namespace std::string_literals;

    property_map result;
    result.emplace(serialization_fields::BLOCK_ID, pluginLoader.registry().typeName(block));
    result.emplace(serialization_fields::BLOCK_UNIQUE_NAME, std::string(block->uniqueName()));
    result.emplace(serialization_fields::BLOCK_NAME, std::string(block->name()));
    result.emplace(serialization_fields::BLOCK_CATEGORY, std::string(gr::meta::enumName(block->blockCategory()).value_or("")));

    if (!block->metaInformation().empty()) {
        result.emplace(serialization_fields::BLOCK_META_INFORMATION, block->metaInformation());
    }

    if (flags & BlockSerializationFlags::Settings) {
        serializeBlockSettings(result, *block);
    }

    if (flags & BlockSerializationFlags::Ports) {
        auto serializePortOrCollection = [](const auto& portOrCollection) {
            // TODO: Type names can be mangled. We need proper type names...
            if (auto* port = std::get_if<gr::DynamicPort>(&portOrCollection)) {
                return property_map{
                    {"name", std::string(port->metaInfo.name)}, //
                    {"type", port->typeName()}                  //
                };
            } else {
                auto& coll = std::get<BlockModel::NamedPortCollection>(portOrCollection);
                return property_map{
                    {"name", std::string(coll.name)},                                                    //
                    {"size", static_cast<gr::Size_t>(coll.ports.size())},                                //
                    {"type", coll.ports.empty() ? std::string() : std::string(coll.ports[0].typeName())} //
                };
            }
        };

        property_map inputPorts;
        for (const auto& portOrCollection : block->dynamicInputPorts()) {
            inputPorts[convert_string_domain(BlockModel::portName(portOrCollection))] = serializePortOrCollection(portOrCollection);
        }
        result.emplace(serialization_fields::BLOCK_INPUT_PORTS, std::move(inputPorts));

        property_map outputPorts;
        for (const auto& portOrCollection : block->dynamicOutputPorts()) {
            outputPorts[convert_string_domain(BlockModel::portName(portOrCollection))] = serializePortOrCollection(portOrCollection);
        }
        result.emplace(serialization_fields::BLOCK_OUTPUT_PORTS, std::move(outputPorts));
    }

    return result;
}

property_map serializeBlock(PluginLoader& pluginLoader, const std::shared_ptr<BlockModel>& block, int flags) {
    property_map map;

    if (const gr::Graph* subgraph = block->graph()) {
        map.emplace(serialization_fields::BLOCK_ID, "SUBGRAPH");
        map.emplace(serialization_fields::BLOCK_UNIQUE_NAME, std::string(block->uniqueName()));
        map.emplace(serialization_fields::BLOCK_NAME, std::string(block->name()));
        map.emplace(serialization_fields::BLOCK_CATEGORY, std::string(gr::meta::enumName(block->blockCategory()).value_or("")));

        {
            property_map subgraphMap;

            if (flags & BlockSerializationFlags::Children) {
                subgraphMap = detail::saveGraphToMap(pluginLoader, *subgraph);
            }

            const std::size_t nExportedPorts = block->exportedInputPorts().size() + block->exportedOutputPorts().size();
            Tensor<Value>     exportedPortsData;
            exportedPortsData.reserve(nExportedPorts);
            for (const auto& [blockName, portName] : block->exportedInputPorts()) {
                exportedPortsData.push_back(Tensor<Value>(data_from, {gr::Value(blockName), gr::Value("INPUT"s), gr::Value(portName)}));
            }
            for (const auto& [blockName, portName] : block->exportedOutputPorts()) {
                exportedPortsData.push_back(Tensor<Value>(data_from, {gr::Value(blockName), gr::Value("OUTPUT"s), gr::Value(portName)}));
            }

            subgraphMap.insert_or_assign(std::string_view{"exported_ports"}, std::move(exportedPortsData));
            map.insert_or_assign(std::string_view{"graph"}, std::move(subgraphMap));
        }

        if (const auto* schedulerModel = gr::scheduler::detail::asSchedulerModel(*block); schedulerModel != nullptr) {
            property_map schedulerMap;
            schedulerMap.insert_or_assign(std::string_view{serialization_fields::BLOCK_ID}, std::string{pluginLoader.schedulerRegistry().typeName(block)});
            if (flags & BlockSerializationFlags::Settings) {
                serializeBlockSettings(schedulerMap, *block);
            }
            map.insert_or_assign(std::string_view{"scheduler"}, std::move(schedulerMap));
        }

    } else {
        map = serializeBlockImpl(pluginLoader, block, flags);
    }

    return map;
}
} // namespace gr
