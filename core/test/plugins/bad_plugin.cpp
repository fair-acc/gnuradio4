#include <vector>

#include <gnuradio-4.0/plugin.hpp>

namespace {
gr_plugin_metadata plugin_metadata [[maybe_unused]]{ "Bad Plugin", "Unknown", "Public Domain", "v0" };

class bad_plugin : public gr_plugin_base {
private:
    std::vector<std::string_view> block_types;

public:
    std::unique_ptr<gr::BlockModel>
    createBlock(std::string_view /*name*/, std::string_view /*type*/, const gr::property_map &) override {
        return {};
    }

    [[nodiscard]] std::uint8_t
    abi_version() const override {
        return 0;
    }

    [[nodiscard]] std::span<const std::string_view> providedBlocks() const override { return block_types; }
};

} // namespace

extern "C" {
void GNURADIO_PLUGIN_EXPORT
gr_plugin_make(gr_plugin_base **plugin) {
    *plugin = nullptr;
}

void GNURADIO_PLUGIN_EXPORT
gr_plugin_free(gr_plugin_base *plugin) {
    delete plugin;
}
}
