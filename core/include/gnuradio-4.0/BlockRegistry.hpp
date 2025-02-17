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
    struct TBlockTypeHandler {
        std::string                                                  alias;
        std::function<std::unique_ptr<gr::BlockModel>(property_map)> createFunction;
    };

    std::vector<std::string>                              _blockTypes;
    std::map<std::string, TBlockTypeHandler, std::less<>> _blockTypeHandlers;

public:
#ifdef ENABLE_BLOCK_REGISTRY
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    void addBlockType(std::string_view alias = "", std::string_view aliasParameters = "") {
        const std::string name      = gr::meta::type_name<TBlock>();
        const std::string fullAlias = [&] {
            if (alias.empty()) {
                return std::string{};
            }
            if (aliasParameters.empty()) {
                return meta::detail::makePortableTypeName(alias);
            }
            return meta::detail::makePortableTypeName(std::string{alias} + "<" + std::string{aliasParameters} + ">");
        }();
        _blockTypes.push_back(name);
        auto handler             = TBlockTypeHandler{.alias = fullAlias, .createFunction = detail::blockFactory<TBlock>};
        _blockTypeHandlers[name] = handler;
        if (!fullAlias.empty()) {
            _blockTypes.push_back(fullAlias);
            handler.alias.clear();
            _blockTypeHandlers[fullAlias] = handler;
        }
    }
#else
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    void addBlockType(std::string_view alias = "", std::string_view aliasParameters = "") {
        std::ignore = alias;
        std::ignore = aliasParameters;
        // disables plugin system in favour of faster compile-times and when runtime or Python wrapping APIs are not requrired
        // e.g. for compile-time only flow-graphs or for CI runners
    }
#endif

    [[nodiscard]] std::span<const std::string> providedBlocks() const { return _blockTypes; }

    [[nodiscard]] std::unique_ptr<gr::BlockModel> createBlock(std::string_view name, property_map params) const {
        if (auto blockIt = _blockTypeHandlers.find(name); blockIt != _blockTypeHandlers.end()) {
            return blockIt->second.createFunction(std::move(params));
        }
        return nullptr;
    }

    [[nodiscard]] auto knownBlocks() const { return _blockTypes; }

    bool isBlockKnown(std::string_view block) const { return _blockTypeHandlers.find(block) != _blockTypeHandlers.end(); }

    template<typename TBlock>
    std::string blockTypeName(const TBlock& block) {
        auto name = block.typeName();
        auto it   = _blockTypeHandlers.find(name);
        if (it != _blockTypeHandlers.end() && !it->second.alias.empty()) {
            return it->second.alias;
        }
        return std::string(name);
    }

    friend inline BlockRegistry& globalBlockRegistry();
};

inline BlockRegistry& globalBlockRegistry() {
    static BlockRegistry s_instance;
    return s_instance;
}

} // namespace gr

#endif // GNURADIO_BLOCK_REGISTRY_HPP
