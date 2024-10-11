#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <charconv>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <yaml-cpp/yaml.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "Graph.hpp"
#include "PluginLoader.hpp"
#include "YamlUtils.hpp"

namespace gr {

inline gr::Graph loadGrc(PluginLoader& loader, const std::string& yamlSrc) {
    Graph testGraph;

    std::map<std::string, BlockModel*> createdBlocks;

    YAML::Node tree   = YAML::Load(yamlSrc);
    auto       blocks = tree["blocks"];
    for (const auto& grcBlock : blocks) {
        auto name = grcBlock["name"].as<std::string>();
        auto id   = grcBlock["id"].as<std::string>();
        /// TODO: when using saveGrc template_args is not saved, this has to be implemented
        auto templateArgs = grcBlock["template_args"];

        auto currentBlock = loader.instantiate(id, templateArgs.IsDefined() ? templateArgs.as<std::string>() : "double");
        if (!currentBlock) {
            throw fmt::format("Unable to create block of type '{}'", id);
        }

        currentBlock->setName(name);
        property_map newProperties;

        auto parameters = grcBlock["parameters"];
        currentBlock->settings().loadParametersFromYAML(parameters);
        auto parametersCtx = grcBlock["ctx_parameters"];
        if (parametersCtx.IsDefined()) {
            for (const auto& ctxPar : parametersCtx) {
                auto ctxName       = ctxPar["context"].as<std::string>();
                auto ctxTime       = ctxPar["time"].as<std::uint64_t>(); // in ns
                auto ctxParameters = ctxPar["parameters"];

                currentBlock->settings().loadParametersFromYAML(ctxParameters, SettingsCtx{ctxTime, ctxName});
            }
        }
        if (const auto failed = currentBlock->settings().activateContext(); failed == std::nullopt) {
            throw gr::exception("Settings for context could not be activated");
        }
        createdBlocks[name] = &testGraph.addBlock(std::move(currentBlock));
    } // for blocks

    for (const auto& connection : tree["connections"]) {
        if (connection.size() != 4) {
            throw fmt::format("Unable to parse connection ({} instead of 4 elements)", connection.size());
        }

        auto parseBlockPort = [&](const auto& blockField, const auto& portField) {
            auto blockName = blockField.template as<std::string>();
            auto node      = createdBlocks.find(blockName);
            if (node == createdBlocks.end()) {
                throw fmt::format("Unknown node '{}'", blockName);
            }

            struct result {
                decltype(node) block_it;
                PortDefinition port_definition;
            };

            if (portField.IsSequence()) {
                if (portField.size() != 2) {
                    throw fmt::format("Port definition has invalid length ({} instead of 2)", portField.size());
                }
                const auto indexStr    = portField[0].template as<std::string>();
                const auto subIndexStr = portField[1].template as<std::string>();
                return result{node, {detail::parseIndex(indexStr), detail::parseIndex(subIndexStr)}};
            } else {
                const auto indexStr = portField.template as<std::string>();
                return result{node, {detail::parseIndex(indexStr)}};
            }
        };

        if (connection.size() == 4) {
            auto src = parseBlockPort(connection[0], connection[1]);
            auto dst = parseBlockPort(connection[2], connection[3]);
            testGraph.connect(*src.block_it->second, src.port_definition, *dst.block_it->second, dst.port_definition);
        } else {
        }
    } // for connections

    return testGraph;
}

inline std::string saveGrc(const gr::Graph& testGraph) {
    YAML::Emitter out;
    {
        detail::YamlMap root(out);

        root.writeFn("blocks", [&]() {
            detail::YamlSeq nodes(out);

            auto writeBlock = [&](const auto& node) {
                detail::YamlMap map(out);
                map.write("name", std::string(node.name()));

                const auto& fullTypeName = node.typeName();
                std::string typeName(fullTypeName.cbegin(), std::find(fullTypeName.cbegin(), fullTypeName.cend(), '<'));
                map.write("id", std::move(typeName));

                const auto& stored = node.settings().getStoredAll();
                // Helper function to write parameters
                auto writeParameters = [&](const property_map& settingsMap, const property_map& metaInformation = {}) {
                    detail::YamlMap parameters(out);
                    auto            writeMap = [&](const auto& localMap) {
                        for (const auto& [settingsKey, settingsValue] : localMap) {
                            std::visit([&]<typename T>(const T& value) { parameters.write(settingsKey, value); }, settingsValue);
                        }
                    };
                    writeMap(settingsMap);
                    if (!metaInformation.empty()) {
                        writeMap(metaInformation);
                    }
                };

                if (stored.contains("")) {
                    const auto& ctxParameters = stored.at("");
                    const auto& settingsMap   = ctxParameters.back().second; // write only the last parameters
                    if (!node.metaInformation().empty() || !settingsMap.empty()) {
                        map.writeFn("parameters", [&]() { writeParameters(settingsMap, node.metaInformation()); });
                    }
                }

                // write context parameters
                map.writeFn("ctx_parameters", [&]() {
                    detail::YamlSeq ctxParamsSeq(out);

                    for (const auto& [ctx, ctxParameters] : stored) {
                        if (ctx == "") {
                            continue;
                        }

                        for (const auto& [ctxTime, settingsMap] : ctxParameters) {
                            detail::YamlMap ctxParamMap(out);

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

                            ctxParamMap.write("context", contextStr);
                            ctxParamMap.write("time", ctxTime.time);
                            ctxParamMap.writeFn("parameters", [&]() { writeParameters(settingsMap); });
                        }
                    }
                });
            };

            testGraph.forEachBlock(writeBlock);
        });

        root.writeFn("connections", [&]() {
            detail::YamlSeq nodes(out);

            auto writePortDefinition = [&](const auto& definition) { //
                std::visit(meta::overloaded(                         //
                               [&](const PortDefinition::IndexBased& _definition) {
                                   if (_definition.subIndex != meta::invalid_index) {
                                       detail::YamlSeq seqPort(out);
                                       out << _definition.topLevel;
                                       out << _definition.subIndex;
                                   } else {
                                       out << _definition.topLevel;
                                   }
                               }, //
                               [&](const PortDefinition::StringBased& _definition) { out << _definition.name; }),
                    definition.definition);
            };

            auto writeEdge = [&](const auto& edge) {
                out << YAML::Flow;
                detail::YamlSeq seq(out);
                out << edge.sourceBlock().name().data();
                writePortDefinition(edge.sourcePortDefinition());

                out << edge.destinationBlock().name().data();
                writePortDefinition(edge.destinationPortDefinition());
            };

            testGraph.forEachEdge(writeEdge);
        });
    }

    return out.c_str();
}

} // namespace gr

#endif // include guard
