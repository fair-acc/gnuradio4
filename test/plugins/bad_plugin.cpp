#include <plugin.hpp>
#include <vector>

namespace {
gp_plugin_metadata plugin_metadata [[maybe_unused]]{ "Bad Plugin", "Unknown", "Public Domain", "v0" };

class bad_plugin : public gp_plugin_base {
private:
    std::vector<std::string> block_types;

public:
    std::unique_ptr<fair::graph::block_model>
    create_block(std::string_view /*name*/, std::string_view /*type*/, const fair::graph::property_map &) override {
        return {};
    }

    [[nodiscard]] std::uint8_t
    abi_version() const override {
        return 0;
    }

    [[nodiscard]] std::span<const std::string>
    provided_blocks() const override {
        return block_types;
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
