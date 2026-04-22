#include <string>

#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

namespace gr {
property_map serializeBlockImpl(gr::PluginLoader& pluginLoader, const std::shared_ptr<BlockModel>& block, int flags) {
    using namespace std::string_literals;

    property_map result;
    result.emplace(serialization_fields::BLOCK_ID, pluginLoader.registry().typeName(block));
    result.emplace(serialization_fields::BLOCK_UNIQUE_NAME, std::string(block->uniqueName()));
    result.emplace(serialization_fields::BLOCK_CATEGORY, std::string(gr::meta::enumName(block->blockCategory()).value_or("")));

    if (!block->metaInformation().empty()) {
        result.emplace(serialization_fields::BLOCK_META_INFORMATION, block->metaInformation());
    }

    if (flags & BlockSerializationFlags::Settings) {
        // Helper function to write parameters
        auto writeParameters = [&](const property_map& settingsMap) {
            pmt::Value::Map parameters;
            auto            writeMap = [&](const auto& localMap) {
                for (const auto& [settingsKey, settingsValue] : localMap) {
                    parameters.insert_or_assign(std::string_view{settingsKey}, settingsValue);
                }
            };
            writeMap(settingsMap);
            return parameters;
        };

        // We don't have a use for info which parameters weren't applied here
        const auto& applyResult = block->settings().applyStagedParameters();
        const auto& stored      = block->settings().getStoredAll();

        result.emplace(serialization_fields::BLOCK_PARAMETERS, writeParameters(block->settings().get()));

        using namespace std::string_literals;
        Tensor<pmt::Value> ctxParamsSeq;
        ctxParamsSeq.reserve(stored.size());
        for (const auto& [ctx, ctxParameters] : stored) {
            if (ctx.holds<std::string>()) {
                if (auto str = ctx.value_or(std::string_view{}); str.empty()) {
                    continue;
                }
            }

            for (const auto& [ctxTime, settingsMap] : ctxParameters) {
                pmt::Value::Map ctxParam;

                // Convert ctxTime.context to a string, regardless of its actual type
                std::string contextStr;
                pmt::ValueVisitor([&contextStr]<typename T>(const T& arg) {
                    if constexpr (std::is_same_v<T, std::string>) {
                        contextStr = arg;
                    } else if constexpr (std::is_same_v<std::string_view, T> || std::is_same_v<std::pmr::string, T>) {
                        contextStr = std::string(arg);
                    } else if constexpr (std::is_arithmetic_v<T>) {
                        contextStr = std::to_string(arg);
                    } else {
                        contextStr.clear();
                    }
                }).visit(ctxTime.context);

                ctxParam.emplace(gr::tag::CONTEXT.shortKey(), contextStr);
                ctxParam.emplace(gr::tag::CONTEXT_TIME.shortKey(), ctxTime.time);
                ctxParam.emplace(serialization_fields::BLOCK_PARAMETERS, writeParameters(settingsMap));
                ctxParamsSeq.emplace_back(std::move(ctxParam));
            }
        }
        result.emplace(serialization_fields::BLOCK_CTX_PARAMETERS, std::move(ctxParamsSeq));
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
        map.emplace("id", std::string{"SUBGRAPH"});
        map.insert_or_assign(std::string_view{"unique_name"}, std::string(block->uniqueName()));
        map.insert_or_assign(std::string_view{"name"}, std::string(block->name()));

        {
            property_map subgraphMap;

            if (flags & BlockSerializationFlags::Children) {
                subgraphMap = detail::saveGraphToMap(pluginLoader, *subgraph);
            }

            const std::size_t  nExportedPorts = block->exportedInputPorts().size() + block->exportedOutputPorts().size();
            Tensor<pmt::Value> exportedPortsData;
            exportedPortsData.reserve(nExportedPorts);
            for (const auto& [blockName, portName] : block->exportedInputPorts()) {
                exportedPortsData.push_back(Tensor<pmt::Value>(data_from, {gr::pmt::Value(blockName), gr::pmt::Value("INPUT"s), gr::pmt::Value(portName)}));
            }
            for (const auto& [blockName, portName] : block->exportedOutputPorts()) {
                exportedPortsData.push_back(Tensor<pmt::Value>(data_from, {gr::pmt::Value(blockName), gr::pmt::Value("OUTPUT"s), gr::pmt::Value(portName)}));
            }

            subgraphMap.insert_or_assign(std::string_view{"exported_ports"}, std::move(exportedPortsData));
            map.insert_or_assign(std::string_view{"graph"}, std::move(subgraphMap));
        }

        if (const auto* schedulerModel = dynamic_cast<const SchedulerModel*>(block.get()); schedulerModel != nullptr) {
            property_map schedulerMap;
            schedulerMap.insert_or_assign(std::string_view{"id"}, std::string{pluginLoader.schedulerRegistry().typeName(block)});
            map.insert_or_assign(std::string_view{"scheduler"}, std::move(schedulerMap));
        }

    } else {
        map = serializeBlockImpl(pluginLoader, block, flags);
    }

    return map;
}
} // namespace gr
