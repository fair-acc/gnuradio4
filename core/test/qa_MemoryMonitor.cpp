#include <boost/ut.hpp>

#include <gnuradio-4.0/thread/MemoryMonitor.hpp>

#include <cstddef>
#include <string>
#include <vector>

const boost::ut::suite<"MemoryMonitor"> memoryMonitorTests = [] {
    using namespace boost::ut;

    "getPlatformName is non-empty on supported hosts"_test = [] {
        const std::string name = gr::memory::getPlatformName();
#if defined(_WIN32) || defined(_WIN64) || defined(__linux__) || defined(__APPLE__)
        expect(!name.empty());
#if defined(__linux__)
        expect(eq(name, std::string{"Linux"}));
#elif defined(__APPLE__)
        expect(eq(name, std::string{"macOS"}));
#elif defined(_WIN32) || defined(_WIN64)
        expect(eq(name, std::string{"Windows"}));
#endif
#else
        expect(name.empty());
#endif
    };

    "getUsage reports resident size on Linux"_test = [] {
        const auto usage = gr::memory::getUsage();
#if defined(__EMSCRIPTEN__)
        expect(eq(usage.residentSize, 0UZ));
#elif defined(__linux__)
        expect(usage.residentSize > 0UZ) << "Linux RSS should be positive for a running process";
        // Touch a large heap buffer and keep it live so RSS must grow.
        std::vector<std::byte> pad(16 << 20);
        for (std::size_t i = 0; i < pad.size(); i += 4096UZ) {
            pad[i] = std::byte{1};
        }
        const auto usageAfterAllocation = gr::memory::getUsage();
        expect(usageAfterAllocation.residentSize > usage.residentSize) << "RSS after touching 16 MiB should increase";
        expect(pad.front() == std::byte{1}); // keep pad live across getUsage
#else
        expect(nothrow([] { static_cast<void>(gr::memory::getUsage()); }));
#endif
    };
};

int main() { return 0; }
