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

namespace gr {

namespace detail {

template<typename T>
inline auto
toYamlString(const T &value) {
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
    YAML::Emitter &out;

    YamlSeq(YAML::Emitter &out_) : out(out_) { out << YAML::BeginSeq; }

    ~YamlSeq() { out << YAML::EndSeq; }

    template<typename F>
        requires std::is_invocable_v<F>
    void
    write_fn(const char * /*key*/, F &&fun) {
        fun();
    }
};

struct YamlMap {
    YAML::Emitter &out;

    YamlMap(YAML::Emitter &out_) : out(out_) { out << YAML::BeginMap; }

    ~YamlMap() { out << YAML::EndMap; }

    template<typename T>
    void
    write(const std::string_view &key, const std::vector<T> &value) {
        out << YAML::Key << key.data();
        YamlSeq seq(out);
        for (const auto &elem : value) out << YAML::Value << toYamlString(elem);
    }

    template<typename T>
    void
    write(const std::string_view &key, const T &value) {
        out << YAML::Key << key.data();
        out << YAML::Value << toYamlString(value);
    }

    template<typename F>
    void
    write_fn(const std::string_view &key, F &&fun) {
        out << YAML::Key << key.data();
        out << YAML::Value;
        fun();
    }
};

inline std::size_t
parseIndex(std::string_view str) {
    std::size_t index{};
    auto [_, src_ec] = std::from_chars(str.begin(), str.end(), index);
    if (src_ec != std::errc()) {
        throw fmt::format("Unable to parse the index");
    }
    return index;
}

} // namespace detail

inline gr::Graph
load_grc(PluginLoader &loader, const std::string &yaml_source) {
    Graph testGraph;

    std::map<std::string, BlockModel *> createdBlocks;

    YAML::Node tree   = YAML::Load(yaml_source);
    auto       blocks = tree["blocks"];
    for (const auto &grc_block : blocks) {
        auto name = grc_block["name"].as<std::string>();
        auto id   = grc_block["id"].as<std::string>();

        // TODO: Discuss how GRC should store the node types, how we should
        // in general handle nodes that are parametrised by more than one type
        auto currentBlock = loader.instantiate(id, "double");
        if (!currentBlock) {
            throw fmt::format("Unable to create block of type '{}'", id);
        }

        currentBlock->setName(name);
        property_map new_properties;

        auto parameters = grc_block["parameters"];
        if (parameters && parameters.IsMap()) {
            // TODO this applyStagedParameters is a workaround to make sure that currentBlock_settings is not empty
            // but contains the default values of the block (needed to covert the parameter values to the right type)
            // should this be based on metadata/reflection?
            std::ignore                = currentBlock->settings().applyStagedParameters();
            auto currentBlock_settings = currentBlock->settings().get();
            for (const auto &kv : parameters) {
                const auto &key = kv.first.as<std::string>();

                if (auto it = currentBlock_settings.find(key); it != currentBlock_settings.end()) {
                    using variant_type_list     = meta::to_typelist<pmtv::pmt>;
                    const YAML::Node &grc_value = kv.second;

                    // This is a known property of this node
                    auto try_type = [&]<typename T>() {
                        if (it->second.index() == variant_type_list::index_of<T>()) {
                            const auto &value   = grc_value.template as<T>();
                            new_properties[key] = value;
                            return true;
                        }

                        if (it->second.index() == variant_type_list::index_of<std::vector<T>>()) {
#if (defined __clang__) && (!defined __EMSCRIPTEN__)
                            if constexpr (std::is_same_v<T, bool>) {
                                // gcc-stdlibc++/clang-libc++ have different implementations for std::vector<bool>
                                // see https://en.cppreference.com/w/cpp/container/vector_bool for details
                                const auto       &value = grc_value.template as<std::vector<int>>(); // need intermediary vector
                                std::vector<bool> boolVector;
                                for (int intValue : value) {
                                    boolVector.push_back(intValue != 0);
                                }
                                new_properties[key] = boolVector;
                                return true;
                            }
#endif
                            const auto &value   = grc_value.template as<std::vector<T>>();
                            new_properties[key] = value;
                            return true;
                        }

                        return false;
                    };

                    // clang-format off
                    try_type.operator()<std::int8_t>() ||
                    try_type.operator()<std::int16_t>() ||
                    try_type.operator()<std::int32_t>() ||
                    try_type.operator()<std::int64_t>() ||
                    try_type.operator()<std::uint8_t>() ||
                    try_type.operator()<std::uint16_t>() ||
                    try_type.operator()<std::uint32_t>() ||
                    try_type.operator()<std::uint64_t>() ||
                    try_type.operator()<bool>() ||
                    try_type.operator()<float>() ||
                    try_type.operator()<double>() ||
                    try_type.operator()<std::string>() ||
                    [&] {
                        // Fallback to string, and non-defined property
                        const auto& value = grc_value.template as<std::string>();
                        currentBlock->metaInformation()[key] = value;
                        return true;
                    }();
                    // clang-format on

                } else {
                    const auto &value                    = kv.second.as<std::string>();
                    currentBlock->metaInformation()[key] = value;
                }
            }
        }

        std::ignore         = currentBlock->settings().set(new_properties);
        createdBlocks[name] = &testGraph.addBlock(std::move(currentBlock));
    }

    for (const auto &connection : tree["connections"]) {
        if (connection.size() != 4) {
            throw fmt::format("Unable to parse connection ({} instead of 4 elements)", connection.size());
        }

        auto parseBlock_port = [&](const auto &blockField, const auto &portField) {
            auto blockName = blockField.template as<std::string>();
            auto node      = createdBlocks.find(blockName);
            if (node == createdBlocks.end()) {
                throw fmt::format("Unknown node '{}'", blockName);
            }

            struct result {
                decltype(node)                   block_it;
                PortIndexDefinition<std::size_t> port_definition;
            };

            if (portField.IsSequence()) {
                if (portField.size() != 2) {
                    throw fmt::format("Port definition has invalid length ({} instead of 2)", portField.size());
                }
                const auto indexStr    = portField[0].template as<std::string>();
                const auto subIndexStr = portField[1].template as<std::string>();
                return result{ node, { detail::parseIndex(indexStr), detail::parseIndex(subIndexStr) } };
            } else {
                const auto indexStr = portField.template as<std::string>();
                return result{ node, { detail::parseIndex(indexStr) } };
            }
        };

        if (connection.size() == 4) {
            auto src = parseBlock_port(connection[0], connection[1]);
            auto dst = parseBlock_port(connection[2], connection[3]);
            testGraph.connect(*src.block_it->second, src.port_definition, *dst.block_it->second, dst.port_definition);
        } else {
        }
    }

    return testGraph;
}

inline std::string
save_grc(const gr::Graph &testGraph) {
    YAML::Emitter out;
    {
        detail::YamlMap root(out);

        root.write_fn("blocks", [&]() {
            detail::YamlSeq nodes(out);

            auto writeBlock = [&](const auto &node) {
                detail::YamlMap map(out);
                map.write("name", std::string(node.name()));

                const auto &full_type_name = node.typeName();
                std::string type_name(full_type_name.cbegin(), std::find(full_type_name.cbegin(), full_type_name.cend(), '<'));
                map.write("id", std::move(type_name));

                const auto &settings_map = node.settings().get();
                if (!node.metaInformation().empty() || !settings_map.empty()) {
                    map.write_fn("parameters", [&]() {
                        detail::YamlMap parameters(out);
                        auto            write_map = [&](const auto &local_map) {
                            for (const auto &[settingsKey, settingsValue] : local_map) {
                                std::visit([&]<typename T>(const T &value) { parameters.write(settingsKey, value); }, settingsValue);
                            }
                        };

                        write_map(settings_map);
                        write_map(node.metaInformation());
                    });
                }
            };

            testGraph.forEachBlock(writeBlock);
        });

        root.write_fn("connections", [&]() {
            detail::YamlSeq nodes(out);
            auto            write_edge = [&](const auto &edge) {
                out << YAML::Flow;
                detail::YamlSeq seq(out);
                out << edge.sourceBlock().name().data();
                const auto sourcePort = edge.sourcePortDefinition();
                if (sourcePort.subIndex == meta::invalid_index) {
                    out << sourcePort.topLevel;
                } else {
                    detail::YamlSeq seqPort(out);
                    out << std::to_string(sourcePort.topLevel);
                    out << std::to_string(sourcePort.subIndex);
                }
                out << edge.destinationBlock().name().data();
                const auto destinationPort = edge.destinationPortDefinition();
                if (destinationPort.subIndex == meta::invalid_index) {
                    out << destinationPort.topLevel;
                } else {
                    detail::YamlSeq seqPort(out);
                    out << std::to_string(destinationPort.topLevel);
                    out << std::to_string(destinationPort.subIndex);
                }
            };

            testGraph.forEachEdge(write_edge);
        });
    }

    return out.c_str();
}

} // namespace gr

#endif // include guard
