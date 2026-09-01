#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

namespace gr {

namespace detail {
inline std::expected<std::string, ParseError> readUriToString(std::string_view uri) {
    gr::algorithm::fileio::ReaderConfig config;
    auto                                readerExp = gr::algorithm::fileio::readAsync(uri, config);
    if (!readerExp) {
        return std::unexpected(ParseError{.message = "Failed to read URI"});
    }
    auto bytesExp = readerExp->get();
    if (!bytesExp) {
        return std::unexpected(ParseError{.message = "Failed to read URI"});
    }
    return std::string(bytesExp->begin(), bytesExp->end());
}
} // namespace detail
PluginLoader& globalPluginLoader() {
    auto pluginPaths = [] {
        std::vector<std::string> result;

        auto* envpath = ::getenv("GNURADIO4_PLUGIN_DIRECTORIES");
        if (envpath == nullptr) {
            // TODO choose proper paths when we get the system GR installation done
            result.emplace_back("core/test/plugins");

        } else {
            std::string_view paths(envpath);

            auto i = paths.cbegin();

            // TODO If we want to support Windows, this should be ; there
            auto isSeparator = [](char c) { return c == ':'; };

            while (i != paths.cend()) {
                i      = std::find_if_not(i, paths.cend(), isSeparator);
                auto j = std::find_if(i, paths.cend(), isSeparator);

                if (i != paths.cend()) {
                    result.emplace_back(std::string_view(i, j));
                }
                i = j;
            }
        }

        return result;
    };

    static PluginLoader instance(gr::globalBlockRegistry(), gr::globalSchedulerRegistry(), {pluginPaths()});
    return instance;
}

namespace detail {

std::expected<void, gr::Error> checkEmbeddedVersionConsistency(const std::unordered_map<std::string, YamlDefinitionsLoader::Definition>& knownDefs, const YamlDefinitionsLoader::Definition& def) {
    const auto blocksIt = def.definition.find("blocks");
    if (blocksIt == def.definition.cend()) {
        return {};
    }
    const gr::pmt::Value blocksEntry = (*blocksIt).second;
    const auto           outerBlocks = blocksEntry.get_if<gr::TensorView<gr::pmt::Value>>();
    if (!outerBlocks) {
        return {};
    }

    for (const gr::pmt::Value& outerBlockVal : *outerBlocks) {
        const auto graphMapOpt = outerBlockVal.get_if<gr::property_map>();
        if (!graphMapOpt) {
            continue;
        }
        const auto graphIt = graphMapOpt->find("graph");
        if (graphIt == graphMapOpt->cend()) {
            continue;
        }
        const gr::pmt::Value graphEntry = (*graphIt).second;
        const auto           graphPM    = graphEntry.get_if<gr::property_map>();
        if (!graphPM) {
            continue;
        }
        const auto innerBlocksIt = graphPM->find("blocks");
        if (innerBlocksIt == graphPM->cend()) {
            continue;
        }
        const gr::pmt::Value innerBlocksEntry = (*innerBlocksIt).second;
        const auto           innerBlocks      = innerBlocksEntry.get_if<gr::TensorView<gr::pmt::Value>>();
        if (!innerBlocks) {
            continue;
        }

        for (const gr::pmt::Value& innerBlockVal : *innerBlocks) {
            const auto blockType = getProperty<std::string>(innerBlockVal, "id");
            if (!blockType) {
                continue;
            }

            const auto embeddedVersion = getProperty<std::string>(innerBlockVal, "yaml_definition_information", "PLUGIN_VERSION");
            if (!embeddedVersion) {
                continue;
            }

            if (auto knownIt = knownDefs.find(*blockType); knownIt != knownDefs.end()) {
                const auto& currentVersion = knownIt->second.metadata.plugin_version;
                if (!currentVersion.empty() && *embeddedVersion != currentVersion) {
                    return std::unexpected(gr::Error(std::format("warning: inner block '{}' in definition '{}' was authored against version '{}' but current version is '{}'", *blockType, def.metadata.block_type, *embeddedVersion, currentVersion)));
                }
            }
        }
    }

    return {};
}

} // namespace detail
} // namespace gr
