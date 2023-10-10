#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <charconv>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <yaml-cpp/yaml.h>
#pragma GCC diagnostic pop

#include "graph.hpp"
#include "plugin_loader.hpp"

namespace gr {

namespace detail {
struct YamlMap {
    YAML::Emitter &out;

    YamlMap(YAML::Emitter &out_) : out(out_) { out << YAML::BeginMap; }

    ~YamlMap() { out << YAML::EndMap; }

    template<typename T>
    void
    write(const std::string_view &key, const T &value) {
        out << YAML::Key << key.data();
        out << YAML::Value << value;
    }

    template<typename F>
    void
    write_fn(const std::string_view &key, F &&fun) {
        out << YAML::Key << key.data();
        out << YAML::Value;
        fun();
    }
};

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
} // namespace detail

inline gr::graph
load_grc(plugin_loader &loader, const std::string &yaml_source) {
    graph                               testGraph;

    std::map<std::string, BlockModel *> createdBlocks;

    YAML::Node                          tree   = YAML::Load(yaml_source);
    auto                                blocks = tree["blocks"];
    for (const auto &grc_block : blocks) {
        auto name = grc_block["name"].as<std::string>();
        auto id   = grc_block["id"].as<std::string>();

        // TODO: Discuss how GRC should store the node types, how we should
        // in general handle nodes that are parametrised by more than one type
        auto &currentBlock = loader.instantiate_in_graph(testGraph, id, "double");

        currentBlock.set_name(name);
        createdBlocks[name]                = &currentBlock;

        auto         currentBlock_settings = currentBlock.settings().get();

        property_map new_properties;

        auto         parameters = grc_block["parameters"];
        if (parameters && parameters.IsMap()) {
            for (const auto &kv : parameters) {
                const auto &key = kv.first.as<std::string>();

                if (auto it = currentBlock_settings.find(key); it != currentBlock_settings.end()) {
                    using variant_type_list = meta::to_typelist<pmtv::pmt>;
                    const auto &grc_value   = kv.second;

                    // This is a known property of this node
                    auto try_type = [&]<typename T>() {
                        if (it->second.index() != variant_type_list::index_of<T>()) {
                            return false;
                        }

                        const auto &value   = grc_value.template as<T>();
                        new_properties[key] = value;

                        return true;
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
                    try_type.operator()<std::string>() ||
                    [&] {
                        // Fallback to string, and non-defined property
                        const auto& value = grc_value.template as<std::string>();
                        currentBlock.meta_information()[key] = value;
                        return true;
                    }();
                    // clang-format on

                } else {
                    const auto &value                    = kv.second.as<std::string>();
                    currentBlock.meta_information()[key] = value;
                }
            }
        }

        std::ignore = currentBlock.settings().set(new_properties);
        // currentBlock.init(); TODO: reverse and first initialise block via property_map constructor and then add to flow-graph -> does the init implicitely then, this is a workaround for the
        // apply_staged_settigns
        std::ignore = currentBlock.settings().apply_staged_parameters();
    }

    for (const auto &connection : tree["connections"]) {
        assert(connection.size() == 4);

        auto parseBlock_port = [&](const auto &block_field, const auto &port_field) {
            auto block_name = block_field.template as<std::string>();
            auto port_str   = port_field.template as<std::string>();
            auto node       = createdBlocks.find(block_name);
            if (node == createdBlocks.end()) {
                throw fmt::format("Unknown node");
            }
            std::size_t port{};
            {
                auto [_, src_ec] = std::from_chars(port_str.data(), port_str.data() + port_str.size(), port);
                if (src_ec != std::errc()) {
                    throw fmt::format("Unable to parse the port index");
                }
            }

            struct result {
                decltype(node) block_it;
                std::size_t    port;
            };

            return result{ node, port };
        };

        auto src = parseBlock_port(connection[0], connection[1]);
        auto dst = parseBlock_port(connection[2], connection[3]);

        testGraph.dynamic_connect(*src.block_it->second, src.port, *dst.block_it->second, dst.port);
    }

    return testGraph;
}

inline std::string
save_grc(const gr::graph &testGraph) {
    YAML::Emitter out;
    {
        detail::YamlMap root(out);

        root.write_fn("blocks", [&]() {
            detail::YamlSeq nodes(out);

            auto            writeBlock = [&](const auto &node) {
                detail::YamlMap map(out);
                map.write("name", std::string(node.name()));

                const auto &full_type_name = node.type_name();
                std::string type_name(full_type_name.cbegin(), std::find(full_type_name.cbegin(), full_type_name.cend(), '<'));
                map.write("id", std::move(type_name));

                const auto &settings_map = node.settings().get();
                if (!node.meta_information().empty() || !settings_map.empty()) {
                    map.write_fn("parameters", [&]() {
                        detail::YamlMap parameters(out);
                        auto            write_map = [&](const auto &local_map) {
                            for (const auto &settings_pair : local_map) {
                                std::visit(
                                        [&]<typename T>(const T &value) {
                                            if constexpr (std::is_same_v<std::string, std::remove_cvref_t<T>>) {
                                                parameters.write(settings_pair.first, value);
                                            } else if constexpr (requires { std::to_string(value); }) {
                                                parameters.write(settings_pair.first, std::to_string(value));
                                            } else {
                                                // not supported
                                            }
                                        },
                                        settings_pair.second);
                            }
                        };

                        write_map(settings_map);
                        write_map(node.meta_information());
                    });
                }
            };

            testGraph.for_each_block(writeBlock);
        });

        root.write_fn("connections", [&]() {
            detail::YamlSeq nodes(out);
            auto            write_edge = [&](const auto &edge) {
                out << YAML::Flow;
                detail::YamlSeq seq(out);
                out << edge.src_block().name().data() << std::to_string(edge.src_port_index());
                out << edge.dst_block().name().data() << std::to_string(edge.dst_port_index());
            };

            testGraph.for_each_edge(write_edge);
        });
    }

    return out.c_str();
}

} // namespace gr

#endif // include guard
