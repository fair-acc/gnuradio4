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
            ctxParam.emplace(std::string_view{gr::tag::CONTEXT}, ctxTime.context);
            ctxParam.emplace(std::string_view{gr::tag::CONTEXT_TIME}, ctxTime.time);
            ctxParam.emplace(serialization_fields::BLOCK_PARAMETERS, writeParameters(settingsMap));
            ctxParamsSeq.emplace_back(std::move(ctxParam));
        }
    }
    output.emplace(serialization_fields::BLOCK_CTX_PARAMETERS, std::move(ctxParamsSeq));
}

void appendSerializedExportedPorts(gr::Tensor<gr::Value>& destination, const gr::Graph& subgraph, const gr::property_map& exportedPorts, std::string_view direction) {
    for (const auto& [blockUniqueName, portMappings_] : exportedPorts) {
        const auto portMappings = portMappings_.get_if<gr::property_map>();
        if (!portMappings) {
            continue;
        }

        auto block = gr::graph::findBlock(subgraph, std::string_view(blockUniqueName));
        if (!block) {
            continue;
        }

        for (const auto& [internalPortName, exportInfo_] : *portMappings) {
            const gr::Value exportInfoValue = exportInfo_;
            const auto      exportInfo      = exportInfoValue.get_if<gr::property_map>();
            if (!exportInfo) {
                continue;
            }

            auto exportedNameIt = exportInfo->find("exportedName");
            if (exportedNameIt == exportInfo->end()) {
                continue;
            }

            const auto exportedName = exportedNameIt->second.value_or(std::string_view{});
            if (exportedName.data() == nullptr) {
                continue;
            }

            destination.emplace_back(gr::Tensor<gr::Value>(gr::data_from, {
                                                                              gr::Value(std::string(block.value()->name())),
                                                                              gr::Value(std::string(direction)),
                                                                              gr::Value(std::string(internalPortName)),
                                                                              gr::Value(std::string(exportedName)),
                                                                          }));
        }
    }
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
        map = serializeBlockImpl(pluginLoader, block, flags);
        map.insert_or_assign(serialization_fields::BLOCK_ID, "SUBGRAPH");

        {
            property_map subgraphMap;

            if (flags & BlockSerializationFlags::Children) {
                subgraphMap = detail::saveGraphToMap(pluginLoader, *subgraph);
            }

            Tensor<Value> exportedPortsData;
            appendSerializedExportedPorts(exportedPortsData, *subgraph, block->exportedInputPorts(), "INPUT");
            appendSerializedExportedPorts(exportedPortsData, *subgraph, block->exportedOutputPorts(), "OUTPUT");

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
