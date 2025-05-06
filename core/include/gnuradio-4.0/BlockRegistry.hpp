#ifndef GNURADIO_BLOCK_REGISTRY_HPP
#define GNURADIO_BLOCK_REGISTRY_HPP

#include <memory>
#include <string>
#include <string_view>

#include <gnuradio-4.0/config.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "BlockModel.hpp"

#include <gnuradio-4.0/Export.hpp>

/**
 *  namespace gr {
 *  template<typename T> struct AlgoImpl1 {};
 *  template<typename T> struct AlgoImpl2 {};
 *
 *  // register block with arbitrary NTTPs (here: 3UZ) and expand T in [float,double], U in [short, int, long, long long]
 *  GR_REGISTER_BLOCK(gr::basic::BlockN, ([T], [U], 3UZ), [ float, double ], [ short, int, long, long long ])
 *  // register block with arbitrary NTTPs (here: 4UZ) and expand T for [short], U for [short] only
 *  GR_REGISTER_BLOCK("CustomBlockNameN", gr::basic::BlockN, ([T], [U], 4UZ, gr::basic::AlgoImpl2<[T]>), [ short ], [ short ])
 *
 *  template<typename T, typename U, std::size_t N, typename Alog = AlgoImpl1<T>>
 *  struct BlockN : public gr::IBlock { ... };
 *
 *  } // namespace gr::basic
 *
 * other macro variants options:
 * GR_REGISTER_BLOCK("MyBlockName", gr::basic::Block1, ([T], [U]), [ float, double ], [int])
 * GR_REGISTER_BLOCK(gr::basic::Block0)
 * GR_REGISTER_BLOCK("blockN.hpp", gr::basic::BlockN, ([T],[U],3UZ,SomeAlgo<[T]>), [ short, int], [double])
 */
#define GR_REGISTER_BLOCK(...) /* Marker macro for parse_registrations */

namespace gr {

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace detail {

template<typename TBlock>
requires std::is_constructible_v<TBlock, property_map>
std::unique_ptr<gr::BlockModel> blockFactory(property_map params) {
    return std::make_unique<gr::BlockWrapper<TBlock>>(std::move(params));
}

std::unique_ptr<gr::BlockModel> blockFactoryProto(property_map params);

} // namespace detail

class BlockRegistry {
    struct TBlockTypeHandler {
        std::string                          alias;
        decltype(detail::blockFactoryProto)* createFunction = nullptr;
    };

    std::map<std::string, TBlockTypeHandler, std::less<>> _blockTypeHandlers;

public:
    BlockRegistry()                                      = default;
    BlockRegistry(const BlockRegistry& other)            = delete;
    BlockRegistry& operator=(const BlockRegistry& other) = delete;

    BlockRegistry(BlockRegistry&& other) noexcept : _blockTypeHandlers(std::exchange(other._blockTypeHandlers, {})) {}
    BlockRegistry& operator=(BlockRegistry&& other) noexcept {
        auto tmp = std::move(other);
        std::swap(_blockTypeHandlers, tmp._blockTypeHandlers);
        return *this;
    }

#ifdef GR_ENABLE_BLOCK_REGISTRY
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    bool insert(std::string_view alias = "", std::string_view aliasParameters = "") {
        const std::string name      = gr::meta::type_name<TBlock>();
        const std::string fullAlias = [&] {
            if (alias.empty()) {
                return std::string{};
            }
            if (alias[0] == '=') {
                return std::string(alias.substr(1));
            }
            if (aliasParameters.empty()) {
                return meta::detail::makePortableTypeName(alias);
            }
            return meta::detail::makePortableTypeName(std::string{alias} + "<" + std::string{aliasParameters} + ">");
        }();

        auto handler = TBlockTypeHandler{.alias = fullAlias, .createFunction = detail::blockFactory<TBlock>};

        auto resName = _blockTypeHandlers.insert_or_assign(name, handler);

        bool aliasInserted = false;
        if (!fullAlias.empty()) {
            handler.alias.clear();
            _blockTypeHandlers[fullAlias] = handler;
            auto resAlias                 = _blockTypeHandlers.insert_or_assign(fullAlias, handler);
            aliasInserted                 = resAlias.second;
        }

        return resName.second || aliasInserted;
    }
#else
    template<BlockLike TBlock>
    requires std::is_constructible_v<TBlock, property_map>
    bool insert([[maybe_unused]] std::string_view alias = "", [[maybe_unused]] std::string_view aliasParameters = "") {
        return false;
        // disables plugin system in favour of faster compile-times and when runtime or Python wrapping APIs are not requrired
        // e.g. for compile-time only flow-graphs or for CI runners
    }
#endif

    [[nodiscard]] std::unique_ptr<gr::BlockModel> create(std::string_view blockName, property_map blockParams) const {
        if (auto blockIt = _blockTypeHandlers.find(blockName); blockIt != _blockTypeHandlers.end()) {
            return blockIt->second.createFunction(std::move(blockParams));
        }
        return nullptr;
    }

    [[nodiscard]] std::vector<std::string> keys() const {
        auto view = _blockTypeHandlers | std::views::keys;
        return {view.begin(), view.end()};
    }

    [[nodiscard]] bool contains(std::string_view blockName) const { return _blockTypeHandlers.contains(blockName); }

    template<typename TBlock>
    std::string typeName(const TBlock& block) {
        auto name = block.typeName();
        auto it   = _blockTypeHandlers.find(name);
        if (it != _blockTypeHandlers.end() && !it->second.alias.empty()) {
            return it->second.alias;
        }
        return std::string(name);
    }

    void merge(gr::BlockRegistry& anotherRegistry) {
        if (this == std::addressof(anotherRegistry)) {
            return;
        }

        _blockTypeHandlers.insert(anotherRegistry._blockTypeHandlers.cbegin(), anotherRegistry._blockTypeHandlers.cend());
    }

    friend BlockRegistry& globalBlockRegistry(std::source_location location);
};

GNURADIO_EXPORT
BlockRegistry& globalBlockRegistry(std::source_location location = std::source_location::current());

} // namespace gr

extern "C" {
GNURADIO_EXPORT
gr::BlockRegistry* grGlobalBlockRegistry(std::source_location location = std::source_location::current());
}

#endif // GNURADIO_BLOCK_REGISTRY_HPP
