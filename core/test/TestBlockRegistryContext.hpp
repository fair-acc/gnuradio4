#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

#include <boost/ut.hpp>

using paths = std::vector<std::filesystem::path>;

struct TestContext {
    gr::BlockRegistry registry;
    gr::PluginLoader  loader;

    template<typename... Args>
    gr::BlockRegistry initRegistry(Args*... args) {
        gr::BlockRegistry _registry;
        ((args(_registry)), ...);
        return _registry;
    }

    template<typename... Args>
    TestContext(std::vector<std::filesystem::path> pluginPaths, Args*... args) : registry(initRegistry(args...)), loader(registry, std::move(pluginPaths)) {}
};
