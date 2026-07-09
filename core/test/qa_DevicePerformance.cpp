#include <boost/ut.hpp>

#include <chrono>
#include <cstdio>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

/*
 * Throughput and host-memory comparison of the device dispatch flavours, in the style of qa_PerformanceMonitor.
 * Every backend's output is asserted against the host result, so a green run also certifies the device paths compute
 * correctly.
 */
namespace gr::perf {

/// a kernel body (processOne), so it runs on CPU and SYCL from one definition
struct Gain : Block<Gain> {
    PortIn<float>  in;
    PortOut<float> out;

    Annotated<float, "gain"> gain = 3.f;
    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

} // namespace gr::perf

namespace {

struct CountingResource : std::pmr::memory_resource {
    std::pmr::memory_resource* upstream    = std::pmr::new_delete_resource();
    std::size_t                allocations = 0UZ;
    std::size_t                bytes       = 0UZ;

    void* do_allocate(std::size_t nBytes, std::size_t alignment) override {
        ++allocations;
        bytes += nBytes;
        return upstream->allocate(nBytes, alignment);
    }
    void do_deallocate(void* ptr, std::size_t nBytes, std::size_t alignment) override { upstream->deallocate(ptr, nBytes, alignment); }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

struct RunResult {
    std::vector<float> samples;
    double             megaSamplesPerSecond = 0.0;
    std::size_t        hostAllocations      = 0UZ;
    double             hostMiB              = 0.0;
    std::size_t        ringCapacity         = 0UZ; // gain.in's connected buffer size, in samples
};

// `minBufferSize` defaults to gr::undefined_size, byte-identical to the implicit default: a caller that passes it
// gets a known-small ring, forcing many cursor wraps without an enormous sample count.
RunResult runGain(std::string_view domain, gr::Size_t nSamples, std::size_t minBufferSize = gr::undefined_size) {
    using namespace gr::testing;

    CountingResource counter;
    auto*            previousDefault = std::pmr::set_default_resource(&counter);

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     gain   = flow.emplaceBlock<gr::perf::Gain>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"n_samples_expected", nSamples}, {"log_samples", true}});
    std::ignore      = flow.connect<"out", "in">(source, gain, gr::EdgeParameters{.minBufferSize = minBufferSize});
    std::ignore      = flow.connect<"out", "in">(gain, sink, gr::EdgeParameters{.minBufferSize = minBufferSize});

    gr::scheduler::Simple<> scheduler;
    std::ignore          = scheduler.exchange(std::move(flow));
    const auto start     = std::chrono::steady_clock::now();
    const auto completed = scheduler.runAndWait();
    const auto stop      = std::chrono::steady_clock::now();
    // edges resize lazily from inside runAndWait(), so capacity is only known after the run; `gain` stays valid
    // because it references into a heap-held BlockModel that moving `flow` does not relocate.
    const std::size_t ringCapacity = gain.in.bufferSize();
    std::pmr::set_default_resource(previousDefault);

    RunResult result;
    result.samples.assign(sink._samples.begin(), sink._samples.end());
    const double seconds        = std::chrono::duration<double>(stop - start).count();
    result.megaSamplesPerSecond = seconds > 0.0 ? static_cast<double>(nSamples) / seconds / 1e6 : 0.0;
    result.hostAllocations      = counter.allocations;
    result.hostMiB              = static_cast<double>(counter.bytes) / 1048576.0;
    result.ringCapacity         = ringCapacity;
    boost::ut::expect(completed.has_value()) << domain << ": the graph must complete";
    return result;
}

[[nodiscard]] bool sameSamples(const std::vector<float>& reference, const std::vector<float>& other) { return reference.size() == other.size() && std::ranges::equal(reference, other); }

} // namespace

// launches kernels, so the tests are registered and run from main() while the AdaptiveCpp runtime is alive (gotcha G10)
int main() {
    using namespace boost::ut;
    using namespace std::string_view_literals;

    const bool syclAvailable = gr::device::registerSyclRuntime();

    constexpr gr::Size_t kSamples = 1U << 20; // 1,048,576

    const RunResult host = runGain("host"sv, kSamples);
    std::printf("\ngain block, %u samples, source -> gain -> sink\n", static_cast<unsigned>(kSamples));
    std::printf("  %-12s %10.2f MS/s   %8zu host allocs   %7.2f MiB\n", "host", host.megaSamplesPerSecond, host.hostAllocations, host.hostMiB);

    struct Candidate {
        std::string_view domain;
        bool             available;
    };
    for (const Candidate candidate : {Candidate{"host:sycl"sv, syclAvailable}, Candidate{"gpu:sycl"sv, syclAvailable}}) {
        const std::string name{candidate.domain};
        if (!candidate.available || gr::device::DeviceContextRegistry::instance().tryResolve(candidate.domain) == nullptr) {
            std::printf("  %-12s (backend not present in this build)\n", name.c_str());
            continue;
        }
        const RunResult run = runGain(candidate.domain, kSamples);
        std::printf("  %-12s %10.2f MS/s   %8zu host allocs   %7.2f MiB   %5.2fx\n", name.c_str(), run.megaSamplesPerSecond, run.hostAllocations, run.hostMiB, run.megaSamplesPerSecond / host.megaSamplesPerSecond);

        test(name + ": reproduces the host result") = [&] { expect(sameSamples(host.samples, run.samples)) << candidate.domain << ": the device backend must compute what the CPU computes"; };
    }
    std::printf("\n");

    "the host baseline produced the expected samples"_test = [&] {
        expect(eq(host.samples.size(), static_cast<std::size_t>(kSamples)));
        expect(host.samples.size() < 2UZ || eq(host.samples[1], 3.f)) << "1 * gain(3)";
    };

    return 0;
}
