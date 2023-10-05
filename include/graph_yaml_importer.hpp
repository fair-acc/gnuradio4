#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <charconv>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <yaml-cpp/yaml.h>
#pragma GCC diagnostic pop

#include <graph.hpp>
#include <plugin_loader.hpp>

namespace fair::graph {

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

inline fair::graph::graph
load_grc(plugin_loader &loader, const std::string &yaml_source) {
    graph                                flow_graph;

    std::map<std::string, block_model *> created_blocks;

    YAML::Node                           tree   = YAML::Load(yaml_source);
    auto                                 blocks = tree["blocks"];
    for (const auto &grc_block : blocks) {
        auto name = grc_block["name"].as<std::string>();
        auto id   = grc_block["id"].as<std::string>();

        // TODO: Discuss how GRC should store the block types, how we should
        // in general handle blocks that are parametrised by more than one type
        auto &current_block = loader.instantiate_in_graph(flow_graph, id, "double");

        current_block.set_name(name);
        created_blocks[name]                = &current_block;

        auto         current_block_settings = current_block.settings().get();

        property_map new_properties;

        auto         parameters = grc_block["parameters"];
        if (parameters && parameters.IsMap()) {
            for (const auto &kv : parameters) {
                const auto &key = kv.first.as<std::string>();

                if (auto it = current_block_settings.find(key); it != current_block_settings.end()) {
                    using variant_type_list = meta::to_typelist<pmtv::pmt>;
                    const auto &grc_value   = kv.second;

                    // This is a known property of this block
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
                        current_block.meta_information()[key] = value;
                        return true;
                    }();
                    // clang-format on

                } else {
                    const auto &value                     = kv.second.as<std::string>();
                    current_block.meta_information()[key] = value;
                }
            }
        }

        std::ignore = current_block.settings().set(new_properties);
        // current_block.init(); TODO: reverse and first initialise block via property_map constructor and then add to flow-graph -> does the init implicitely then, this is a workaround for the
        // apply_staged_settigns
        std::ignore = current_block.settings().apply_staged_parameters();
    }

    for (const auto &connection : tree["connections"]) {
        assert(connection.size() == 4);

        auto parse_block_port = [&](const auto &block_field, const auto &port_field) {
            auto block_name = block_field.template as<std::string>();
            auto port_str   = port_field.template as<std::string>();
            auto block      = created_blocks.find(block_name);
            if (block == created_blocks.end()) {
                throw fmt::format("Unknown block");
            }
            std::size_t port{};
            {
                auto [_, src_ec] = std::from_chars(port_str.data(), port_str.data() + port_str.size(), port);
                if (src_ec != std::errc()) {
                    throw fmt::format("Unable to parse the port index");
                }
            }

            struct result {
                decltype(block) block_it;
                std::size_t     port;
            };

            return result{ block, port };
        };

        auto src = parse_block_port(connection[0], connection[1]);
        auto dst = parse_block_port(connection[2], connection[3]);

        flow_graph.dynamic_connect(*src.block_it->second, src.port, *dst.block_it->second, dst.port);
    }

    return flow_graph;
}

inline std::string
save_grc(const fair::graph::graph &flow_graph) {
    YAML::Emitter out;
    {
        detail::YamlMap root(out);

        root.write_fn("blocks", [&]() {
            detail::YamlSeq blocks(out);

            auto            write_block = [&](const auto &block) {
                detail::YamlMap map(out);
                map.write("name", std::string(block.name()));

                const auto &full_type_name = block.type_name();
                std::string type_name(full_type_name.cbegin(), std::find(full_type_name.cbegin(), full_type_name.cend(), '<'));
                map.write("id", std::move(type_name));

                const auto &settings_map = block.settings().get();
                if (!block.meta_information().empty() || !settings_map.empty()) {
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
                        write_map(block.meta_information());
                    });
                }
            };

            flow_graph.for_each_block(write_block);
        });

        root.write_fn("connections", [&]() {
            detail::YamlSeq blocks(out);
            auto            write_edge = [&](const auto &edge) {
                out << YAML::Flow;
                detail::YamlSeq seq(out);
                out << edge.src_block().name().data() << std::to_string(edge.src_port_index());
                out << edge.dst_block().name().data() << std::to_string(edge.dst_port_index());
            };

            flow_graph.for_each_edge(write_edge);
        });
    }

    return out.c_str();
}

} // namespace fair::graph

#endif // include guard
