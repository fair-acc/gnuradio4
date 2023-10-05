#ifndef GRAPH_PROTOTYPE_BLOCK_REGISTRY_H
#define GRAPH_PROTOTYPE_BLOCK_REGISTRY_H

#include <memory>
#include <string>
#include <string_view>

#include <graph.hpp>
#include <utils.hpp>

namespace fair::graph {

using namespace std::string_literals;
using namespace std::string_view_literals;

class block_registry {
private:
    using block_type_handler = std::function<void(std::unique_ptr<fair::graph::block_model> &, const property_map &)>;
    std::vector<std::string>                                                             _block_types;
    std::unordered_map<std::string, std::unordered_map<std::string, block_type_handler>> _block_type_handlers;

    template<template<typename...> typename BlockTemplate, typename... BlockParameters>
    static auto
    create_handler() {
        return [](std::unique_ptr<fair::graph::block_model> &result, const property_map &params) {
            using BlockType = BlockTemplate<BlockParameters...>;

            if constexpr (std::is_constructible_v<BlockType, const property_map &>) {
                result = std::make_unique<fair::graph::block_wrapper<BlockType>>(params); // gp_plugin_block::wrap(std::make_shared<BlockType>(params));
            } else if constexpr (std::is_constructible_v<BlockType>) {
                result = std::make_unique<fair::graph::block_wrapper<BlockType>>();
            } else {
                fair::meta::print_types<fair::meta::message_type<"Can not default-construct the block instance, nor to construct it from const property_map&">, BlockType>{};
            }
            return true;
        };
    }

    auto &
    find_block_type_handlers_map(const std::string &block_type) {
        if (auto it = _block_type_handlers.find(block_type); it != _block_type_handlers.end()) {
            return it->second;
        } else {
            _block_types.emplace_back(block_type);
            return _block_type_handlers[block_type];
        }
    }

    template<typename... Types>
    static std::string
    encoded_list_of_types() {
        struct accumulator {
            std::string value;

            accumulator &
            operator%(const std::string &type) {
                if (value.empty()) value = type;
                else
                    value += ";"s + type;
                return *this;
            }
        };

        return (accumulator{} % ... % meta::type_name<Types>()).value;
    }

public:
    template<template<typename...> typename BlockTemplate, typename... BlockParameters>
    void
    add_block_type(std::string block_type) {
        auto &block_handlers                                        = find_block_type_handlers_map(block_type);
        block_handlers[encoded_list_of_types<BlockParameters...>()] = create_handler<BlockTemplate, BlockParameters...>();
        fmt::print("Registered {} {}\n", block_type, encoded_list_of_types<BlockParameters...>());
    }

    std::span<const std::string>
    provided_blocks() const {
        return _block_types;
    }

    std::unique_ptr<fair::graph::block_model>
    create_block(std::string_view name, std::string_view type, const property_map &params) {
        std::unique_ptr<fair::graph::block_model> result;
        auto                                      block_it = _block_type_handlers.find(std::string(name));
        if (block_it == _block_type_handlers.end()) return nullptr;

        auto &block_handlers = block_it->second;
        auto  handler_it     = block_handlers.find(std::string(type));
        if (handler_it == block_handlers.end()) return nullptr;

        handler_it->second(result, params);
        return result;
    }

    auto
    known_blocks() const {
        return _block_types;
    }
};

} // namespace fair::graph

#endif // include guard
