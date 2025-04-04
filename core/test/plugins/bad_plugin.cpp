#include <vector>

#include <gnuradio-4.0/Plugin.hpp>

namespace {
gr_plugin_metadata plugin_metadata [[maybe_unused]]{"Bad Plugin", "Unknown", "Public Domain", "v0"};

class bad_plugin : public gr_plugin_base {
private:
    std::vector<std::string> block_types;

public:
    std::unique_ptr<gr::BlockModel> createBlock(std::string_view /*name*/, const gr::property_map&) override { return {}; }

    [[nodiscard]] std::uint8_t abi_version() const override { return 0; }

    [[nodiscard]] std::span<const std::string> providedBlocks() const override { return block_types; }
};

} // namespace

extern "C" {
void GNURADIO_EXPORT gr_plugin_make(gr_plugin_base** plugin) { *plugin = nullptr; }

void GNURADIO_EXPORT gr_plugin_free(gr_plugin_base* /*plugin*/) {
    // We can not really kill a plugin, once a dynamic libray is loaded
    // it is here to stay
}
}
