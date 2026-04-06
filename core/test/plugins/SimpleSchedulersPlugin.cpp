#include <gnuradio-4.0/Plugin.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

GR_PLUGIN("Simple Schedulers Plugin", "Unknown", "MIT", "v1")

static const bool registerSimpleSchedulers = [] {
    auto& registry = static_cast<gr::SchedulerRegistry&>(grPluginInstance());

    registry.template insert<gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::singleThreaded>>("=gr::scheduler::SimpleSingle");
    registry.template insert<gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>>("=gr::scheduler::SimpleMulti");

    return true;
}();
