#include <boost/ut.hpp>

#include <format>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/device/DeviceContextCuda.hpp>
#include <gnuradio-4.0/device/DeviceContextRocm.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

/// The contract these cases hold CUDA to is `DeviceContext`'s, not CUDA's: allocation honours the requested
/// alignment, a token never lies about what may dereference it, and a payload survives the round trip. They
/// prove the device abstraction carries a SECOND backend -- they do not prove any block runs on CUDA, which
/// needs a launch path this context deliberately does not have.
const boost::ut::suite<"DeviceContextCuda"> _deviceContextCuda = [] {
    using namespace boost::ut;
    using namespace gr::device;

    if constexpr (!gr::device::kHasCuda) {
        skip / "a CUDA device context requires GR_ENABLE_CUDA and the CUDA runtime headers"_test = [] {};
        return;
    }
#if GR_DEVICE_HAS_CUDA
    if (!DeviceContextCuda::isServed()) {
        skip / "no CUDA device is present on this machine"_test = [] {};
        return;
    }

    "the context names the backend it actually is"_test = [] {
        DeviceContextCuda ctx;
        expect(ctx.backend() == DeviceBackend::CUDA);
        expect(ctx.deviceType() == DeviceType::GPU);
        expect(!ctx.name().empty()) << "a served device must report a name";
        expect(ctx.shortName().starts_with("CUDA:")) << "the short name must say which backend answered";
        expect(ctx.hasRemoteMemory()) << "a discrete GPU has memory the host cannot simply dereference";
    };

    "every residency round-trips a payload"_test = [] {
        DeviceContextCuda ctx;
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
            expect(eq(buffer.bytes, sent.size() * sizeof(float)));

            ctx.upload(sent.data(), buffer, sent.size() * sizeof(float));
            ctx.download(buffer, received.data(), received.size() * sizeof(float));
            ctx.wait();
            expect(sent == received) << std::format("residency {} did not round-trip", static_cast<int>(residency));
            ctx.deallocate(buffer);
        }
    };

    "a token never lies about who may dereference it"_test = [] {
        DeviceContextCuda ctx;
        for (const auto residency : {Residency::host, Residency::shared, Residency::devicePtr}) {
            DeviceBuffer buffer = ctx.allocate(256UZ, 64UZ, residency);
            if (!buffer) {
                continue;
            }
            const void* pointer = reinterpret_cast<const void*>(buffer.token);
            expect(ctx.isDeviceAccessible(pointer) == (residency != Residency::host)) << "pinned host memory is not device memory";
            expect(ctx.isDeviceOnly(pointer) == (residency == Residency::devicePtr)) << "managed memory is reachable from the host, so it is not device-ONLY";
            expect((buffer.devicePointer<float>() != nullptr) == (residency != Residency::host)) << "devicePointer must be null exactly where a kernel could not index it";
            ctx.deallocate(buffer);
        }
    };

    "an unserveable request returns an invalid token rather than a lying one"_test = [] {
        DeviceContextCuda ctx;
        expect(!ctx.allocate(64UZ, 64UZ, Residency::invalid)) << "`invalid` is never a request to honour";
        expect(!ctx.allocate(0UZ, 64UZ, Residency::devicePtr)) << "a zero-byte request has no valid token to return";
    };

    "ordinary host memory is not mistaken for device memory"_test = [] {
        DeviceContextCuda  ctx;
        std::vector<float> onHost(16, 1.f);
        expect(!ctx.isDeviceAccessible(onHost.data()));
        expect(!ctx.isDeviceOnly(onHost.data()));
        expect(!ctx.peekDeviceError().has_value()) << "probing a host pointer must not leave a sticky CUDA error behind";
    };
#endif
};

int main() { /* tests run from the suite */ }
