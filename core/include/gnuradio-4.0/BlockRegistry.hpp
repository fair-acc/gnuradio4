#ifndef BLOCK_REGISTRY_HPP
#define BLOCK_REGISTRY_HPP

#include <memory>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

class BlockRegistry {
private:
    using block_type_handler = std::function<void(std::unique_ptr<gr::BlockModel> &, const property_map &)>;
    std::vector<std::string>                                                             _block_types;
    std::unordered_map<std::string, std::unordered_map<std::string, block_type_handler>> _block_type_handlers;

    template<typename TBlock>
    static auto
    createHandler() {
        return [](std::unique_ptr<gr::BlockModel> &result, const property_map &params) {
            if constexpr (std::is_constructible_v<TBlock, const property_map &>) {
                result = std::make_unique<gr::BlockWrapper<TBlock>>(params); // gr_pluginBlock::wrap(std::make_shared<TBlock>(params));
            } else if constexpr (std::is_constructible_v<TBlock>) {
                result = std::make_unique<gr::BlockWrapper<TBlock>>();
            } else {
                gr::meta::print_types<gr::meta::message_type<"Can not default-construct the node instance, nor to construct it from const property_map&">, TBlock>{};
            }
            return true;
        };
    }

    template<template<typename...> typename TBlock, typename... TBlockParameters>
    static auto
    createHandler() {
        return createHandler<TBlock<TBlockParameters...>>();
    }

    auto &
    findBlock_type_handlers_map(const std::string &blockType) {
        if (auto it = _block_type_handlers.find(blockType); it != _block_type_handlers.end()) {
            return it->second;
        } else {
            _block_types.emplace_back(blockType);
            return _block_type_handlers[blockType];
        }
    }

public:
    template<typename TBlock>
    void
    addBlockType(std::string blockType = {}, std::string blockParams = {}) {
        if (blockType.empty()) {
            auto fullName = meta::type_name<TBlock>();
            auto sep      = std::ranges::find(fullName, '<');
            if (sep == fullName.end()) {
                blockParams = "";
            } else {
                blockType   = std::string(fullName.begin(), sep);
                blockParams = std::string(sep + 1, fullName.end() - 1); // Trim < and >
                std::erase(blockParams, ' ');
            }
        }
        auto &block_handlers = findBlock_type_handlers_map(blockType);
        block_handlers[std::move(blockParams)] = createHandler<TBlock>();
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

#endif // BLOCK_REGISTRY_HPP
