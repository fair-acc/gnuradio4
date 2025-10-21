#include <gnuradio-4.0/PluginLoader.hpp>

namespace gr {
PluginLoader& globalPluginLoader() {
    auto pluginPaths = [] {
        std::vector<std::filesystem::path> result;

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
} // namespace gr
