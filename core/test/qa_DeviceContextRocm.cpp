#include <boost/ut.hpp>

#include <format>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextRocm.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

/// The ROCm mirror of `qa_DeviceContextCuda`, holding HIP to the same `DeviceContext` contract: allocation
/// honours the requested alignment, a token never lies about who may dereference it, and a payload survives
/// the round trip. Verified on ROCm 6.4 with an RX 7700S (gfx1102).
const boost::ut::suite<"DeviceContextRocm"> _deviceContextRocm = [] {
    using namespace boost::ut;
    using namespace gr::device;

    if constexpr (!gr::device::kHasRocm) {
        skip / "a ROCm device context requires GR_ENABLE_ROCM and the HIP runtime headers"_test = [] {};
        return;
    }
#if GR_DEVICE_HAS_ROCM
    if (!DeviceContextRocm::isServed()) {
        skip / "no ROCm device is present on this machine"_test = [] {};
        return;
    }

    "the context names the backend it actually is"_test = [] {
        DeviceContextRocm ctx;
        expect(ctx.backend() == DeviceBackend::ROCm);
        expect(ctx.deviceType() == DeviceType::GPU);
        expect(!ctx.name().empty()) << "a served device must report a name";
        expect(ctx.shortName().starts_with("ROCm:")) << "the short name must say which backend answered";
        expect(ctx.hasRemoteMemory());
    };

    "every residency round-trips a payload"_test = [] {
        DeviceContextRocm ctx;
        for (const auto residency : {Residency::host, Residency::shared, Residency::devicePtr}) {
            std::vector<float> sent(1024);
            std::vector<float> received(sent.size(), 0.f);
            std::iota(sent.begin(), sent.end(), 1.f);

            DeviceBuffer buffer = ctx.allocate(sent.size() * sizeof(float), 64UZ, residency);
            expect(static_cast<bool>(buffer)) << std::format("residency {} could not be served", static_cast<int>(residency));
            if (!buffer) {
                continue;
            }
            expect(eq(buffer.token % 64UZ, 0UZ)) << "the token must honour the alignment that was asked for";
            ctx.upload(sent.data(), buffer, sent.size() * sizeof(float));
            ctx.download(buffer, received.data(), received.size() * sizeof(float));
            ctx.wait();
            expect(sent == received) << std::format("residency {} did not round-trip", static_cast<int>(residency));
            ctx.deallocate(buffer);
        }
    };

    "a token never lies about who may dereference it"_test = [] {
        DeviceContextRocm ctx;
        for (const auto residency : {Residency::host, Residency::shared, Residency::devicePtr}) {
            DeviceBuffer buffer = ctx.allocate(256UZ, 64UZ, residency);
            if (!buffer) {
                continue;
            }
            const void* pointer = reinterpret_cast<const void*>(buffer.token);
            expect(ctx.isDeviceAccessible(pointer) == (residency != Residency::host)) << "pinned host memory is not device memory";
            expect(ctx.isDeviceOnly(pointer) == (residency == Residency::devicePtr)) << "managed memory is reachable from the host, so it is not device-ONLY";
            ctx.deallocate(buffer);
        }
    };

    "ordinary host memory is not mistaken for device memory"_test = [] {
        DeviceContextRocm  ctx;
        std::vector<float> onHost(16, 1.f);
        expect(!ctx.isDeviceAccessible(onHost.data()));
        expect(!ctx.peekDeviceError().has_value()) << "probing a host pointer must not leave a sticky HIP error behind";
    };
#endif
};

int main() { /* tests run from the suite */ }
