#ifndef GNURADIO_NODE_REGISTRY_H
#define GNURADIO_NODE_REGISTRY_H

#include <memory>
#include <string>
#include <string_view>

#include <gnuradio-4.0/meta/utils.hpp>

#include "Graph.hpp"

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

class BlockRegistry {
private:
    using block_type_handler = std::function<void(std::unique_ptr<gr::BlockModel> &, const property_map &)>;
    std::vector<std::string>                                                             _block_types;
    std::unordered_map<std::string, std::unordered_map<std::string, block_type_handler>> _block_type_handlers;

    template<template<typename...> typename TBlock, typename... TBlockParameters>
    static auto
    createHandler() {
        return [](std::unique_ptr<gr::BlockModel> &result, const property_map &params) {
            using BlockType = TBlock<TBlockParameters...>;

            if constexpr (std::is_constructible_v<BlockType, const property_map &>) {
                result = std::make_unique<gr::BlockWrapper<BlockType>>(params); // gp_pluginBlock::wrap(std::make_shared<BlockType>(params));
            } else if constexpr (std::is_constructible_v<BlockType>) {
                result = std::make_unique<gr::BlockWrapper<BlockType>>();
            } else {
                gr::meta::print_types<gr::meta::message_type<"Can not default-construct the node instance, nor to construct it from const property_map&">, BlockType>{};
            }
            return true;
        };
    }

    auto &
    findBlock_type_handlers_map(const std::string &block_type) {
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
    template<template<typename...> typename TBlock, typename... TBlockParameters>
    void
    addBlockType(std::string block_type) {
        auto &block_handlers                                         = findBlock_type_handlers_map(block_type);
        block_handlers[encoded_list_of_types<TBlockParameters...>()] = createHandler<TBlock, TBlockParameters...>();
        fmt::print("Registered {} {}\n", block_type, encoded_list_of_types<TBlockParameters...>());
    }

    std::span<const std::string>
    providedBlocks() const {
        return _block_types;
    }

    std::unique_ptr<gr::BlockModel>
    createBlock(std::string_view name, std::string_view type, const property_map &params) {
        std::unique_ptr<gr::BlockModel> result;
        auto                            block_it = _block_type_handlers.find(std::string(name));
        if (block_it == _block_type_handlers.end()) return nullptr;

        auto &block_handlers = block_it->second;
        auto  handler_it     = block_handlers.find(std::string(type));
        if (handler_it == block_handlers.end()) return nullptr;

        handler_it->second(result, params);
        return result;
    }

    auto
    knownBlocks() const {
        return _block_types;
    }

    friend inline BlockRegistry &
    globalBlockRegistry();
};

inline BlockRegistry &
globalBlockRegistry() {
    static BlockRegistry s_instance;
    return s_instance;
}

} // namespace gr

#endif // include guard
