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

namespace YAML {
// YAML custom converter for complex numbers
template<typename T>
requires std::same_as<T, double> || std::same_as<T, float>
struct convert<std::complex<T>> {
    static Node encode(const std::complex<T>& rhs) {
        Node node;
        node.push_back(rhs.real());
        node.push_back(rhs.imag());
        return node;
    }

    static bool decode(const Node& node, std::complex<T>& rhs) {
        if (!node.IsSequence() || node.size() != 2) {
            return false;
        }
        rhs = std::complex<T>(node[0].as<T>(), node[1].as<T>());
        return true;
    }
};
} // namespace YAML

namespace gr {

namespace detail {

template<typename T>
inline auto toYamlString(const T& value) {
    if constexpr (std::is_same_v<std::string, std::remove_cvref_t<T>>) {
        return value;
    } else if constexpr (std::is_same_v<bool, std::remove_cvref_t<T>>) {
        return value ? "true" : "false";
    } else if constexpr (requires { std::to_string(value); }) {
        return std::to_string(value);
    } else {
        return "";
    }
}

struct YamlSeq {
    YAML::Emitter& out;

    YamlSeq(YAML::Emitter& out_) : out(out_) { out << YAML::BeginSeq; }

    ~YamlSeq() { out << YAML::EndSeq; }

    template<typename F>
    requires std::is_invocable_v<F>
    void writeFn(const char* /*key*/, F&& fun) {
        fun();
    }
};

struct YamlMap {
    YAML::Emitter& out;

    YamlMap(YAML::Emitter& out_) : out(out_) { out << YAML::BeginMap; }

    ~YamlMap() { out << YAML::EndMap; }

    template<typename T>
    void write(const std::string_view& key, const std::vector<T>& value) {
        out << YAML::Key << key.data();
        YamlSeq seq(out);
        for (const auto& elem : value) {
            if constexpr (std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>) {
                writeComplexValue(out, elem);
            } else {
                out << YAML::Value << toYamlString(elem);
            }
        }
    }

    template<typename T>
    void write(const std::string_view& key, const T& value) {
        out << YAML::Key << key.data();
        if constexpr (std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>) {
            writeComplexValue(out, value);
        } else {
            out << YAML::Value << toYamlString(value);
        }
    }

    template<typename F>
    void writeFn(const std::string_view& key, F&& fun) {
        out << YAML::Key << key.data();
        out << YAML::Value;
        fun();
    }

private:
    template<typename T>
    requires std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>
    void writeComplexValue(YAML::Emitter& outEmitter, const T& value) {
        YamlSeq seq(outEmitter);
        outEmitter << YAML::Value << toYamlString(value.real());
        outEmitter << YAML::Value << toYamlString(value.imag());
    }
};

inline std::size_t parseIndex(std::string_view str) {
    std::size_t index{};
    auto [_, src_ec] = std::from_chars(str.begin(), str.end(), index);
    if (src_ec != std::errc()) {
        throw fmt::format("Unable to parse the index");
    }
    return index;
}

} // namespace detail

inline gr::Graph loadGrc(PluginLoader& loader, const std::string& yamlSrc) {
    Graph testGraph;

    std::map<std::string, BlockModel*> createdBlocks;

    YAML::Node tree   = YAML::Load(yamlSrc);
    auto       blocks = tree["blocks"];
    for (const auto& grcBlock : blocks) {
        auto name = grcBlock["name"].as<std::string>();
        auto id   = grcBlock["id"].as<std::string>();

        // TODO: Discuss how GRC should store the node types, how we should
        // in general handle nodes that are parametrised by more than one type
        auto currentBlock = loader.instantiate(id, "double");
        if (!currentBlock) {
            throw fmt::format("Unable to create block of type '{}'", id);
        }

        currentBlock->setName(name);
        property_map newProperties;

        auto parameters = grcBlock["parameters"];
        if (parameters && parameters.IsMap()) {
            // TODO this applyStagedParameters is a workaround to make sure that currentBlock_settings is not empty
            // but contains the default values of the block (needed to covert the parameter values to the right type) should this be based on metadata/reflection?
            currentBlock->settings().updateActiveParameters();
            auto currentBlockSettings = currentBlock->settings().get();
            for (const auto& kv : parameters) {
                const auto& key = kv.first.as<std::string>();

                if (auto it = currentBlockSettings.find(key); it != currentBlockSettings.end()) {
                    using variant_type_list    = meta::to_typelist<pmtv::pmt>;
                    const YAML::Node& grcValue = kv.second;

                    // This is a known property of this node
                    auto tryType = [&]<typename T>() {
                        if (it->second.index() == variant_type_list::index_of<T>()) {
                            const auto& value  = grcValue.template as<T>();
                            newProperties[key] = value;
                            return true;
                        }

                        if (it->second.index() == variant_type_list::index_of<std::vector<T>>()) {
#if (defined __clang__) && (!defined __EMSCRIPTEN__)
                            if constexpr (std::is_same_v<T, bool>) {
                                // gcc-stdlibc++/clang-libc++ have different implementations for std::vector<bool>
                                // see https://en.cppreference.com/w/cpp/container/vector_bool for details
                                const auto&       value = grcValue.template as<std::vector<int>>(); // need intermediary vector
                                std::vector<bool> boolVector;
                                for (int intValue : value) {
                                    boolVector.push_back(intValue != 0);
                                }
                                newProperties[key] = boolVector;
                                return true;
                            }
#endif
                            const auto& value  = grcValue.template as<std::vector<T>>();
                            newProperties[key] = value;
                            return true;
                        }

                        return false;
                    };

                    // clang-format off
                    tryType.operator()<std::int8_t>() ||
                    tryType.operator()<std::int16_t>() ||
                    tryType.operator()<std::int32_t>() ||
                    tryType.operator()<std::int64_t>() ||
                    tryType.operator()<std::uint8_t>() ||
                    tryType.operator()<std::uint16_t>() ||
                    tryType.operator()<std::uint32_t>() ||
                    tryType.operator()<std::uint64_t>() ||
                    tryType.operator()<bool>() ||
                    tryType.operator()<float>() ||
                    tryType.operator()<double>() ||
                    tryType.operator()<std::string>() ||
                    tryType.operator()<std::complex<float>>() ||
                    tryType.operator()<std::complex<double>>() ||
                    [&] {
                        // Fallback to string, and non-defined property
                        const auto& value = grcValue.template as<std::string>();
                        currentBlock->metaInformation()[key] = value;
                        return true;
                    }();
                    // clang-format on

                } else {
                    const auto& value                    = kv.second.as<std::string>();
                    currentBlock->metaInformation()[key] = value;
                }
            }
        }

        std::ignore         = currentBlock->settings().set(newProperties);
        std::ignore         = currentBlock->settings().activateContext();
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

                const auto& settingsMap = node.settings().get();
                if (!node.metaInformation().empty() || !settingsMap.empty()) {
                    map.writeFn("parameters", [&]() {
                        detail::YamlMap parameters(out);
                        auto            writeMap = [&](const auto& localMap) {
                            for (const auto& [settingsKey, settingsValue] : localMap) {
                                std::visit([&]<typename T>(const T& value) { parameters.write(settingsKey, value); }, settingsValue);
                            }
                        };

                        writeMap(settingsMap);
                        writeMap(node.metaInformation());
                    });
                }
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
