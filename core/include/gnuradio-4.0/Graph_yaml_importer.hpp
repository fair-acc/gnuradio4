#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <charconv>

#include <gnuradio-4.0/YamlPmt.hpp>

#include "Graph.hpp"
#include "PluginLoader.hpp"

namespace gr {

inline gr::Graph loadGrc(PluginLoader& loader, std::string_view yamlSrc) {
    Graph testGraph;

    std::map<std::string, BlockModel*> createdBlocks;

    const auto yaml = pmtv::yaml::deserialize(yamlSrc);
    if (!yaml) {
        throw gr::exception(fmt::format("Could not parse yaml: {}:{}\n{}", yaml.error().message, yaml.error().line, yamlSrc));
    }

    auto blks = std::get<std::vector<pmtv::pmt>>(yaml.value().at("blocks"));
    for (const auto& blk : blks) {
        auto grcBlock = std::get<property_map>(blk);

        const auto name = std::get<std::string>(grcBlock["name"]);
        const auto id   = std::get<std::string>(grcBlock["id"]);

        std::string type = "double";
        /// TODO: when using saveGrc template_args is not saved, this has to be implemented
        if (auto it = grcBlock.find("template_args"); it != grcBlock.end()) {
            type = std::get<std::string>(it->second);
        }

        auto currentBlock = loader.instantiate(id, type);
        if (!currentBlock) {
            throw fmt::format("Unable to create block of type '{}'", id);
        }

        currentBlock->setName(name);

        const auto parametersPmt = grcBlock["parameters"];
        if (const auto parameters = std::get_if<property_map>(&parametersPmt)) {
            currentBlock->settings().loadParametersFromPropertyMap(*parameters);
        } else {
            currentBlock->settings().loadParametersFromPropertyMap({});
        }

        if (auto it = grcBlock.find("ctx_parameters"); it != grcBlock.end()) {
            auto parametersCtx = std::get<std::vector<pmtv::pmt>>(it->second);
            for (const auto& ctxPmt : parametersCtx) {
                auto       ctxPar        = std::get<property_map>(ctxPmt);
                const auto ctxName       = std::get<std::string>(ctxPar["context"]);
                const auto ctxTime       = std::get<std::uint64_t>(ctxPar["time"]); // in ns
                const auto ctxParameters = std::get<property_map>(ctxPar["parameters"]);

                currentBlock->settings().loadParametersFromPropertyMap(ctxParameters, SettingsCtx{ctxTime, ctxName});
            }
        }
        if (const auto failed = currentBlock->settings().activateContext(); failed == std::nullopt) {
            throw gr::exception("Settings for context could not be activated");
        }
        createdBlocks[name] = &testGraph.addBlock(std::move(currentBlock));
    } // for blocks

    auto connections = std::get<std::vector<pmtv::pmt>>(yaml.value().at("connections"));
    for (const auto& conn : connections) {
        auto connection = std::get<std::vector<pmtv::pmt>>(conn);
        if (connection.size() != 4) {
            throw fmt::format("Unable to parse connection ({} instead of 4 elements)", connection.size());
        }

        auto parseBlockPort = [&](const auto& blockField, const auto& portField) {
            const auto blockName = std::get<std::string>(blockField);
            auto       node      = createdBlocks.find(blockName);
            if (node == createdBlocks.end()) {
                throw fmt::format("Unknown node '{}'", blockName);
            }

            struct result {
                decltype(node) block_it;
                PortDefinition port_definition;
            };

            if (const auto portFields = std::get_if<std::vector<pmtv::pmt>>(&portField)) {
                if (portFields->size() != 2) {
                    throw fmt::format("Port definition has invalid length ({} instead of 2)", portFields->size());
                }
                const auto index    = std::get<std::int64_t>(portFields->at(0));
                const auto subIndex = std::get<std::int64_t>(portFields->at(1));
                return result{node, {static_cast<std::size_t>(index), static_cast<std::size_t>(subIndex)}};

            } else {
                const auto index = std::get<std::int64_t>(portField);
                return result{node, {static_cast<std::size_t>(index)}};
            }
        };

        auto src = parseBlockPort(connection[0], connection[1]);
        auto dst = parseBlockPort(connection[2], connection[3]);
        testGraph.connect(*src.block_it->second, src.port_definition, *dst.block_it->second, dst.port_definition);
    } // for connections

    return testGraph;
}

inline std::string saveGrc(const gr::Graph& testGraph) {

    pmtv::map_t yaml;

    std::vector<pmtv::pmt> blocks;
    testGraph.forEachBlock([&](const auto& node) {
        pmtv::map_t map;
        map["name"] = std::string(node.name());

        const auto& fullTypeName = node.typeName();
        std::string typeName(fullTypeName.cbegin(), std::find(fullTypeName.cbegin(), fullTypeName.cend(), '<'));
        map.emplace("id", std::move(typeName));

        // Helper function to write parameters
        auto writeParameters = [&](const property_map& settingsMap, const property_map& metaInformation = {}) {
            pmtv::map_t parameters;
            auto        writeMap = [&](const auto& localMap) {
                for (const auto& [settingsKey, settingsValue] : localMap) {
                    std::visit([&]<typename T>(const T& value) { parameters[settingsKey] = value; }, settingsValue);
                }
            };
            writeMap(settingsMap);
            if (!metaInformation.empty()) {
                writeMap(metaInformation);
            }
            return parameters;
        };

        const auto& stored = node.settings().getStoredAll();
        if (stored.contains("")) {
            const auto& ctxParameters = stored.at("");
            const auto& settingsMap   = ctxParameters.back().second; // write only the last parameters
            if (!node.metaInformation().empty() || !settingsMap.empty()) {
                map["parameters"] = writeParameters(settingsMap, node.metaInformation());
            }
        }

        std::vector<pmtv::pmt> ctxParamsSeq;
        for (const auto& [ctx, ctxParameters] : stored) {
            if (ctx == "") {
                continue;
            }

            for (const auto& [ctxTime, settingsMap] : ctxParameters) {
                pmtv::map_t ctxParam;

                // Convert ctxTime.context to a string, regardless of its actual type
                std::string contextStr = std::visit(
                    [](const auto& arg) -> std::string {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::string>) {
                            return arg;
                        } else if constexpr (std::is_arithmetic_v<T>) {
                            return std::to_string(arg);
                        }
                        return "";
                    },
                    ctxTime.context);

                ctxParam["context"]    = contextStr;
                ctxParam["time"]       = ctxTime.time;
                ctxParam["parameters"] = writeParameters(settingsMap);
                ctxParamsSeq.emplace_back(std::move(ctxParam));
            }
        }
        map["ctx_parameters"] = ctxParamsSeq;

        blocks.emplace_back(std::move(map));
    });
    yaml["blocks"] = blocks;

    std::vector<pmtv::pmt> connections;
    testGraph.forEachEdge([&](const auto& edge) {
        std::vector<pmtv::pmt> seq;

        auto writePortDefinition = [&](const auto& definition) { //
            std::visit(meta::overloaded(                         //
                           [&](const PortDefinition::IndexBased& _definition) {
                               if (_definition.subIndex != meta::invalid_index) {
                                   std::vector<pmtv::pmt> seqPort;

                                   seqPort.push_back(std::int64_t(_definition.topLevel));
                                   seqPort.push_back(std::int64_t(_definition.subIndex));
                                   seq.push_back(seqPort);
                               } else {
                                   seq.push_back(std::int64_t(_definition.topLevel));
                               }
                           }, //
                           [&](const PortDefinition::StringBased& _definition) { seq.push_back(_definition.name); }),
                definition.definition);
        };

        seq.push_back(edge.sourceBlock().name().data());
        writePortDefinition(edge.sourcePortDefinition());

        seq.push_back(edge.destinationBlock().name().data());
        writePortDefinition(edge.destinationPortDefinition());

        connections.emplace_back(seq);
    });
    yaml["connections"] = connections;

    return pmtv::yaml::serialize(yaml);
}

} // namespace gr

#endif // include guard
