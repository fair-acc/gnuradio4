#include <plugin.hpp>
#include <vector>

namespace {
gp_plugin_metadata plugin_metadata{ "Bad Plugin", "Unknown", "Public Domain", "v0" };

class bad_plugin : public gp_plugin_base {
private:
    std::vector<std::string> node_types;

public:
    std::unique_ptr<fair::graph::node_model>
    create_node(std::string_view name, std::string_view type, fair::graph::node_construction_params) override {
        return {};
    }

    std::uint8_t
    abi_version() const override {
        return 0;
    }

    std::span<const std::string>
    provided_nodes() const override {
        return node_types;
    }
};

} // namespace

extern "C" {
void GRAPH_PROTOTYPE_PLUGIN_EXPORT
gp_plugin_make(gp_plugin_base **plugin) {
    *plugin = nullptr;
}

void GRAPH_PROTOTYPE_PLUGIN_EXPORT
gp_plugin_free(gp_plugin_base *plugin) {
    delete plugin;
}
}
