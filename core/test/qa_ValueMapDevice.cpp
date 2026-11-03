#include <boost/ut.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>

using namespace gr::testing;

int main() {
    constexpr std::uint32_t    kKeys    = 2U;
    constexpr std::uint32_t    kPayload = 64U;
    constexpr std::size_t      kSlot    = ((gr::pmt::blobBytesForKeys(kKeys, kPayload) + gr::pmt::kBlobAlignment - 1UZ) / gr::pmt::kBlobAlignment) * gr::pmt::kBlobAlignment;
    constexpr std::string_view kText    = "trigger";

    "a kernel reads a string a host wrote into a blob"_domain_test = [&](auto& ctx) {
        std::byte* slot = ctx.template alloc<std::byte>(kSlot);
        std::ranges::fill(std::span{slot, kSlot}, std::byte{});

        auto written = gr::pmt::ValueMapView::formatAt({slot, kSlot}, kPayload, gr::pmt::entryCapacityForKeys(kKeys));
        boost::ut::expect(written.try_emplace(std::string_view{"name"}, kText)) << "the host must be able to write the blob";

        // only scalars cross into the kernel: a string_view over a host literal is a pointer the device cannot
        // dereference, which surfaces as an illegal access rather than a compile error
        std::size_t expectedLength   = kText.size();
        int         expectedChecksum = 0;
        for (const char character : kText) {
            expectedChecksum += static_cast<int>(character);
        }

        ctx.launch([slot, slotSize = kSlot, expectedLength, expectedChecksum](const DeviceTestHandle& device) {
            const auto view = gr::pmt::ValueMap::makeView(std::span<const std::byte>{slot, slotSize});
            const auto text = view.get_if<std::string_view>(std::string_view{"name"});
            expect(device, text.has_value(), "the key 'name' must be readable in a kernel");
            if (device.failed()) {
                return;
            }
            expect(device, text->size() == expectedLength, "read back {} characters, wrote {}", text->size(), expectedLength);

            int checksum = 0;
            for (std::size_t i = 0UZ; i < text->size(); ++i) {
                checksum += static_cast<int>((*text)[i]);
            }
            expect(device, checksum == expectedChecksum, "checksum {} != {}", checksum, expectedChecksum);
        });
    } | kAllDomains;

    return 0;
}
