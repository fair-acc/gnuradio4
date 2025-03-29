#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

#include <boost/ut.hpp>

extern "C" {
bool gr_blocklib_init_module_GrBasicBlocks(gr::BlockRegistry&);
bool gr_blocklib_init_module_GrTestingBlocks(gr::BlockRegistry&);
bool gr_blocklib_init_module_qa_grc(gr::BlockRegistry&);
}

using paths = std::vector<std::filesystem::path>;

struct TestContext {
    gr::BlockRegistry registry;
    gr::PluginLoader  loader;

    template<typename... Args>
    gr::BlockRegistry initRegistry(Args*... args) {
        auto _registry = gr::globalBlockRegistry();
        ((args(_registry)), ...);
        return _registry;
    }

    template<typename... Args>
    TestContext(std::vector<std::filesystem::path> pluginPaths, Args*... args) : registry(initRegistry(args...)), loader(registry, std::move(pluginPaths)) {}
};

TestContext* context = nullptr;

class RunnerContext {
public:
    template<typename... Args>
    RunnerContext(Args&&... args) {
        context = std::make_unique<TestContext>(std::forward<Args>(args)...).release();
    }

    ~RunnerContext() { delete context; }

    RunnerContext(const RunnerContext&)            = delete;
    RunnerContext& operator=(const RunnerContext&) = delete;

    template<class... Ts>
    auto on(boost::ext::ut::events::test<Ts...> test) {
        test();
    }
    auto on(boost::ext::ut::v2_0_1::events::suite<void (*)()> suite) { suite(); }
    template<class... Ts>
    auto on(boost::ext::ut::events::skip<Ts...>) {}
    template<class TExpr>
    auto on(boost::ext::ut::events::assertion<TExpr>) -> bool {
        return true;
    }
    auto on(boost::ext::ut::events::fatal_assertion) {}
    template<class TMsg>
    auto on(boost::ext::ut::events::log<TMsg>) {}
};
