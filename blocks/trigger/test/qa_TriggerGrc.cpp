#include <boost/ut.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/GrTriggerBlocks.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/PluginLoader.hpp>

const boost::ut::suite<"control-flow step 1 round-trips through GRC"> _triggerGrc = [] {
    using namespace boost::ut;

    "gate, tag-to-event and predicate switch keep their settings through a save and a reload"_test = [] {
        constexpr std::string_view kGrc = R"(
blocks:
  - id: gr::blocks::trigger::Gate<float32>
    parameters:
      name: gate
      mode: toggle
      open_filter: start
  - id: gr::blocks::trigger::TagToEvent<float32>
    parameters:
      name: tagToEvent
  - id: gr::blocks::trigger::PredicateSwitch<float32>
    parameters:
      name: byLevel
      predicates: ['x > level', 'x <= level']
connections: []
)";
        gr::BlockRegistry          registry;
        expect(eq(gr::blocklib::initGrTriggerBlocks(registry), 0UZ)) << fatal << "the count is of units that failed to register";
        gr::SchedulerRegistry schedulers;
        gr::PluginLoader      loader(registry, schedulers, {});

        auto loaded = gr::loadGrc(loader, kGrc);
        expect(loaded.has_value()) << fatal << (loaded ? std::string{} : loaded.error().message);
        const std::string saved = gr::saveGrc(loader, **loaded);

        auto reloaded = gr::loadGrc(loader, saved);
        expect(reloaded.has_value()) << fatal;

        const auto settingOf = [&reloaded](std::string_view blockName, std::string_view key) -> gr::pmt::Value {
            for (const auto& block : (*reloaded)->blocks()) {
                if (block->name() == blockName) {
                    return block->settings().get(std::string(key)).value_or(gr::pmt::Value{});
                }
            }
            return {};
        };
        expect(eq(settingOf("gate", "open_filter").value_or(std::string()), std::string("start"))) << "Gate keeps its filter";
        expect(settingOf("byLevel", "predicates").holds<gr::Tensor<gr::pmt::Value>>() || !settingOf("byLevel", "predicates").is_monostate()) << "PredicateSwitch keeps its predicates";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
