#ifndef GNURADIO_BLOCK_REGISTRY_HPP
#define GNURADIO_BLOCK_REGISTRY_HPP

#include <memory>
#include <string>
#include <string_view>

#include <gnuradio-4.0/config.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "BlockModel.hpp"

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace detail {

template<typename TBlock>
requires std::is_constructible_v<TBlock, property_map>
std::unique_ptr<gr::BlockModel> blockFactory(property_map params) {
    return std::make_unique<gr::BlockWrapper<TBlock>>(std::move(params));
}

} // namespace detail

class BlockRegistry {
    using TBlockTypeHandler = std::function<std::unique_ptr<gr::BlockModel>(property_map)>;
    std::vector<std::string>                                                            _blockTypes;
    std::unordered_map<std::string, std::unordered_map<std::string, TBlockTypeHandler>> _blockTypeHandlers;

    auto& findBlockTypeHandlersMap(const std::string& blockType) {
        if (auto it = _blockTypeHandlers.find(blockType); it != _blockTypeHandlers.end()) {
            return it->second;
        }
        _blockTypes.emplace_back(blockType);
        return _blockTypeHandlers[blockType];
    }

public:
#ifdef ENABLE_BLOCK_REGISTRY
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    void addBlockType(std::string blockType, std::string blockParams) {
        auto& block_handlers                   = findBlockTypeHandlersMap(blockType);
        block_handlers[std::move(blockParams)] = detail::blockFactory<TBlock>;
    }
#else
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    void addBlockType(std::string, std::string) {
        // disables plugin system in favour of faster compile-times and when runtime or Python wrapping APIs are not requrired
        // e.g. for compile-time only flow-graphs or for CI runners
    }
#endif

    [[nodiscard]] std::span<const std::string> providedBlocks() const { return _blockTypes; }

    [[nodiscard]] std::unique_ptr<gr::BlockModel> createBlock(std::string_view name, std::string_view type, property_map params) const {
        if (auto blockIt = _blockTypeHandlers.find(std::string(name)); blockIt != _blockTypeHandlers.end()) {
            if (auto handlerIt = blockIt->second.find(std::string(type)); handlerIt != blockIt->second.end()) {
                return handlerIt->second(std::move(params));
            }
        }
        return nullptr;
    }

    [[nodiscard]] auto knownBlocks() const { return _blockTypes; }

    bool isBlockKnown(std::string_view block) const { return _blockTypeHandlers.find(std::string(block)) != _blockTypeHandlers.end(); }

    auto knownBlockParameterizations(std::string_view block) const {
        std::vector<std::string> result;
        if (auto it = _blockTypeHandlers.find(std::string(block)); it != _blockTypeHandlers.end()) {
            const auto& map = it->second;
            result.reserve(map.size());
            for (const auto& [key, _] : map) {
                result.push_back(key);
            }
        }

        return result;
    }

    friend inline BlockRegistry& globalBlockRegistry();
};

inline BlockRegistry& globalBlockRegistry() {
    static BlockRegistry s_instance;
    return s_instance;
}

} // namespace gr

#endif // GNURADIO_BLOCK_REGISTRY_HPP
