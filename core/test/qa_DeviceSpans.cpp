#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <print>
#include <ranges>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/device/DeviceContextRegistry.hpp>
#include <gnuradio-4.0/device/ExecutionStrategy.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include "device_test_helpers.hpp"

/*
 * The classical `processBulk(InputSpanLike, OutputSpanLike)` on a device: one work item, its own accounting, and
 * state that survives across work() calls. Not a parallelisation -- a zero-crossing detector is sequential by
 * nature. The point is that such a block can sit between two device blocks without dragging the stream back to
 * the host.
 *
 * The chain is source -> trigger -> sink:
 *   - the source emits a sine, phase-shifted by half a sample so no sample sits exactly on zero
 *   - the trigger passes samples through, counts negative->positive crossings, and tags every SECOND one, so a
 *     tag marks the start of every other sine period
 *   - the sink collects samples and tags
 *
 * Publishing a tag is the one thing a kernel cannot do (a tag payload is not device-constructible), so the
 * dispatcher detects it on a throwaway copy BEFORE launching anything and takes the host path. The block is
 * therefore written the natural way and must produce identical tags on every domain -- which is what this asserts.
 */
namespace gr::device_spans_test {

using namespace boost::ut;

constexpr gr::Size_t  kSamplesPerPeriod = 16U;
constexpr gr::Size_t  kSamples          = 128U; // 8 periods
constexpr std::size_t kTriggerEvery     = 2UZ;  // one tag per two periods

struct SineSource : Block<SineSource> {
    PortOut<float> out;

    Annotated<gr::Size_t, "samples per period"> samples_per_period = kSamplesPerPeriod;
    Annotated<gr::Size_t, "max samples">        n_samples_max      = kSamples;
    gr::Size_t                                  count              = 0U;

    GR_MAKE_REFLECTABLE(SineSource, out, samples_per_period, n_samples_max, count);

    [[nodiscard]] constexpr float processOne() noexcept {
        // +0.5 so no sample lands exactly on zero and every crossing is unambiguous
        const float phase = 2.f * std::numbers::pi_v<float> * (static_cast<float>(count) + 0.5f) / static_cast<float>(samples_per_period);
        ++count;
        if (n_samples_max > 0U && count >= n_samples_max) {
            this->requestStop();
        }
        return std::sin(phase);
    }
};

/// which compilation pass produced this code. MEASURED: every SYCL domain reports "device", the OpenMP host backend
/// included -- SSCP compiles one generic kernel whatever later executes it. So this says "ran as a kernel rather than
/// through the CPU fallback"; which physical device ran it is what the `domain` key alongside it records.
[[nodiscard]] constexpr std::string_view executionTarget() noexcept {
    std::string_view target{"host", 4UZ};
#ifdef __acpp_if_target_device
    __acpp_if_target_device(target = std::string_view{"device", 6UZ};)
#endif
        return target;
}

/// sequential by nature: the decision for sample i depends on sample i-1 and on how many crossings came before
struct ZeroCrossingTrigger : Block<ZeroCrossingTrigger> {
    PortIn<float>  in;
    PortOut<float> out;

    float      _previous  = 0.f; // last sample of the previous chunk; 0 so the very first sample is not a rise
    gr::Size_t _crossings = 0U;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(ZeroCrossingTrigger, in, out, _previous, _crossings);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        const std::size_t count = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < count; ++i) {
            const float sample = input[i];
            output[i]          = sample;
            if (_previous < 0.f && sample >= 0.f) {
                ++_crossings;
                if (_crossings % kTriggerEvery == 0U) {
                    // the same provenance keys the view form writes, so the two are directly comparable in the console
                    output.publishTag(property_map{{std::string(std::string_view(gr::tag::TRIGGER_NAME)), std::string("zero-crossing")},                                       //
                                          {std::string(std::string_view(gr::tag::TRIGGER_TIME)), static_cast<std::uint64_t>(i) * 1'000UZ},                                     //
                                          {std::string(std::string_view(gr::tag::TRIGGER_TIME_ERROR)), static_cast<std::uint64_t>(0U)},                                        //
                                          {std::string(std::string_view(gr::tag::TRIGGER_OFFSET)), 0.f},                                                                       //
                                          {std::string(std::string_view(gr::tag::TRIGGER_META_INFO)), property_map{{std::string("domain"), std::string(this->compute_domain)}, //
                                                                                                          {std::string("execution_target"), std::string(executionTarget())}}}},
                        i);
                }
            }
            _previous = sample;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

/// the canonical trigger contract: `gr:trigger_name`, `_time`, `_time_error` and `_offset` are all required;
/// `gr:trigger_meta_info` is the only optional key and is deliberately absent — it is a nested `property_map`,
/// which a kernel cannot build. The `DefaultTag`s convert to a constexpr `string_view`, so no key lowers to
/// `strlen` (an unresolved libc symbol in device code).
/// the canonical trigger contract, provenance included: `gr:trigger_meta_info` is a nested map, which a kernel can
/// now build in place, so the same source produces the same tag on the host and on a device
[[nodiscard]] constexpr bool writeTriggerContract(gr::pmt::ValueMapView& payload, std::size_t sampleIndex, std::string_view domain) noexcept {
    constexpr std::uint64_t kNanosPerSample = 1'000U; // deterministic stand-in: a kernel has no clock
    bool                    complete        = payload.try_emplace(std::string_view(gr::tag::TRIGGER_NAME), std::string_view{"zero-crossing", 13UZ});
    complete                                = payload.try_emplace(std::string_view(gr::tag::TRIGGER_TIME), static_cast<std::uint64_t>(sampleIndex) * kNanosPerSample) && complete;
    complete                                = payload.try_emplace(std::string_view(gr::tag::TRIGGER_TIME_ERROR), static_cast<std::uint64_t>(0U)) && complete;
    complete                                = payload.try_emplace(std::string_view(gr::tag::TRIGGER_OFFSET), 0.f) && complete;

    gr::pmt::ValueMapView meta = payload.try_emplace_map(std::string_view(gr::tag::TRIGGER_META_INFO), 2U, 96U);
    if (meta._header == nullptr) {
        return false;
    }
    complete = meta.try_emplace(std::string_view{"domain", 6UZ}, domain) && complete;
    complete = meta.try_emplace(std::string_view{"execution_target", 16UZ}, executionTarget()) && complete;
    return complete;
}

/// the same detector, but the payload is built IN the kernel: `formatAt` + `try_emplace` into a local buffer,
/// then published as a non-owning view. That form needs no allocation, so this block really runs as a kernel
/// instead of taking the host fallback its `property_map` sibling above triggers.
struct ZeroCrossingTriggerView : Block<ZeroCrossingTriggerView> {
    PortIn<float>  in;
    PortOut<float> out;

    float                _previous  = 0.f;
    gr::Size_t           _crossings = 0U;
    std::array<char, 16> _domain{}; // compute_domain in fixed storage: trivially copyable, so the kernel reads it from its own mirror
    std::uint8_t         _domainLength = 0U;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(ZeroCrossingTriggerView, in, out, _previous, _crossings);

    void settingsChanged(const gr::property_map&, const gr::property_map&) {
        const std::string_view domain{this->compute_domain};
        _domainLength = static_cast<std::uint8_t>(std::min(domain.size(), _domain.size()));
        std::copy_n(domain.begin(), _domainLength, _domain.begin());
    }

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        const std::size_t count = std::min(input.size(), output.size());
        for (std::size_t i = 0UZ; i < count; ++i) {
            const float sample = input[i];
            output[i]          = sample;
            if (_previous < 0.f && sample >= 0.f) {
                ++_crossings;
                if (_crossings % kTriggerEvery == 0U) {
                    gr::pmt::StackValueMap<5UZ, 384UZ> payload; // four trigger keys + a nested meta map, allocation-free
                    static_assert(decltype(payload)::kCapacity <= gr::device::kDeviceTagSlotBytes, "a blob larger than a slot is rejected on publish");
                    if (payload.isFormatted() && writeTriggerContract(payload.view(), i, std::string_view{_domain.data(), _domainLength})) {
                        output.publishTag(payload.view(), i);
                    }
                }
            }
            _previous = sample;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

/// reads a value out of every input tag it was given and folds the total into each sample, so the sink's samples
/// are proof that a kernel both received the tags and could read their payloads
struct InputTagCounter : Block<InputTagCounter> {
    PortIn<float>  in;
    PortOut<float> out;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(InputTagCounter, in, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) const {
        const std::size_t count = std::min(input.size(), output.size());

        float total = 0.f;
        for (const auto& tag : input.rawTags()) {
            if (const auto* level = tag.map.template get_if<std::int32_t>(std::string_view{"level", 5UZ})) {
                total += static_cast<float>(*level);
            }
        }
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = input[i] + total;
        }
        std::ignore = input.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

/// reads tags from BOTH input ports, weighting them differently. Every per-port tag offset in the span tier is
/// scaled by the port index, so all of them evaluate identically while only port 0 ever carries a tag.
struct TwoPortTagReader : gr::Block<TwoPortTagReader> {
    gr::PortIn<float>  in0;
    gr::PortIn<float>  in1;
    gr::PortOut<float> out;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(TwoPortTagReader, in0, in1, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& a, gr::InputSpanLike auto& b, gr::OutputSpanLike auto& output) const {
        const auto levelSum = [](const auto& span) {
            float total = 0.f;
            for (const auto& tag : span.rawTags()) {
                if (const auto* level = tag.map.template get_if<std::int32_t>(std::string_view{"level", 5UZ})) {
                    total += static_cast<float>(*level);
                }
            }
            return total;
        };
        const float bias  = levelSum(a) + 100.f * levelSum(b); // weighted, so port 1's tags cannot masquerade as port 0's
        const std::size_t count = std::min({a.size(), b.size(), output.size()});
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = bias;
        }
        std::ignore = a.consume(count);
        std::ignore = b.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

[[nodiscard]] float runTwoPortTagsOn(std::string_view domain) {
    using namespace boost::ut;
    using namespace gr::testing;

    gr::Graph flow;
    auto&     srcA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(64)}, {"mark_tag", false}});
    auto&     srcB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(64)}, {"mark_tag", false}});
    // one tag per port, both at index 0, so a single work() call sees both and the expected value is exact
    srcA._tags = {{0UZ, {{"level", 1}}}};
    srcB._tags = {{0UZ, {{"level", 5}}}}; // weighted by 100 -> 501; swapped ports would give 105
    auto& reader   = flow.emplaceBlock<TwoPortTagReader>({{"gr:compute_domain", std::string(domain)}});
    auto& sink     = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    expect(flow.connect<"out", "in0">(srcA, reader).has_value());
    expect(flow.connect<"out", "in1">(srcB, reader).has_value());
    expect(flow.connect<"out", "in">(reader, sink).has_value());

    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value()) << std::format("the two-port tag chain must run on '{}'", domain);
    return sink._samples.empty() ? -1.f : std::ranges::max(sink._samples);
}

struct TagPayload {
    std::string   name;
    std::uint64_t time      = 0U;
    std::uint64_t timeError = 0U;
    float         offset    = 0.f;
    std::size_t   entries   = 0UZ;
    std::string   domain;          // what the block reported from inside the kernel
    std::string   executionTarget; // "device" only when the device compilation pass produced this code

    [[nodiscard]] bool operator==(const TagPayload&) const = default;
    /// the trigger contract alone, so runs on different domains stay comparable despite differing provenance
    [[nodiscard]] bool sameTrigger(const TagPayload& other) const { return name == other.name && time == other.time && timeError == other.timeError && offset == other.offset; }
};

/// provenance rides in the nested `gr:trigger_meta_info`, exactly where the tag contract says it belongs
[[nodiscard]] inline std::string readMeta(const gr::pmt::ValueMap& map, std::string_view key) {
    const auto meta = map.get_if<gr::pmt::ValueMap>(std::string_view(gr::tag::TRIGGER_META_INFO));
    return meta == nullptr ? std::string("<none>") : meta->value_or<std::string>(key, std::string("<none>"));
}

/// consumes n and publishes 2n: the tier used to hand both spans one count, which bounded the output by the input
struct Upsampler : gr::Block<Upsampler, gr::Resampling<1UZ, 2UZ, true>> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(Upsampler, in, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        const std::size_t nConsumed = std::min(input.size(), output.size() / 2UZ);
        for (std::size_t i = 0UZ; i < nConsumed; ++i) {
            output[2UZ * i]       = input[i];
            output[2UZ * i + 1UZ] = -input[i]; // negated, so a body that merely duplicated its input cannot pass
        }
        std::ignore = input.consume(nConsumed);
        output.publish(2UZ * nConsumed);
        return gr::work::Status::OK;
    }
};

[[nodiscard]] std::vector<float> runUpsamplerOn(std::string_view domain, gr::Size_t nSamples) {
    using namespace boost::ut;
    using namespace gr::testing;

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     up     = flow.emplaceBlock<Upsampler>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink   = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    expect(flow.connect<"out", "in">(source, up).has_value());
    expect(flow.connect<"out", "in">(up, sink).has_value());

    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value()) << std::format("the upsampler chain must run to completion on '{}'", domain);
    return {sink._samples.begin(), sink._samples.end()};
}

/// two inputs through the span tier, each consuming its own port: every port now carries its own accounting
/// record, and `a - 2b` is asymmetric so crossed ports cannot pass
struct WeightedDifferenceSpans : gr::Block<WeightedDifferenceSpans> {
    gr::PortIn<float>  in0;
    gr::PortIn<float>  in1;
    gr::PortOut<float> out;

    using DeviceStateIsReflected = void;
    GR_MAKE_REFLECTABLE(WeightedDifferenceSpans, in0, in1, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& a, gr::InputSpanLike auto& b, gr::OutputSpanLike auto& output) {
        const std::size_t count = std::min({a.size(), b.size(), output.size()});
        for (std::size_t i = 0UZ; i < count; ++i) {
            output[i] = a[i] - 2.f * b[i];
        }
        std::ignore = a.consume(count);
        std::ignore = b.consume(count);
        output.publish(count);
        return gr::work::Status::OK;
    }
};

[[nodiscard]] std::vector<float> runTwoInputSpansOn(std::string_view domain, gr::Size_t nSamples) {
    using namespace boost::ut;
    using namespace gr::testing;

    gr::Graph flow;
    auto&     sourceA = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     sourceB = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", nSamples}, {"mark_tag", false}});
    auto&     scaleB  = flow.emplaceBlock<Upsampler>({{"gr:compute_domain", std::string(domain)}}); // 1 -> 2, so the arms differ in rate as well as value
    auto&     combine = flow.emplaceBlock<WeightedDifferenceSpans>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", true}});

    expect(flow.connect<"out", "in0">(sourceA, combine).has_value());
    expect(flow.connect<"out", "in">(sourceB, scaleB).has_value());
    expect(flow.connect<"out", "in1">(scaleB, combine).has_value());
    expect(flow.connect<"out", "in">(combine, sink).has_value());

    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value()) << std::format("the two-input span chain must run to completion on '{}'", domain);
    return {sink._samples.begin(), sink._samples.end()};
}

struct RunResult {
    std::vector<float>       samples;
    std::vector<std::size_t> tagIndices;
    std::vector<TagPayload>  payloads;
    std::size_t              cpuFallbacks = 0UZ; // >0 means the block never reached a kernel
};

template<typename TTrigger>
[[nodiscard]] RunResult runOn(std::string_view domain) {
    using namespace gr::testing;

    gr::log::HistoryLoggerBackend recorded; // the dispatcher announces every CPU fallback
    gr::log::Backend*             previousBackend = gr::log::setBackend(&recorded);

    // flush
    std::fflush(stdout);
    std::fflush(stderr);

    gr::Graph flow;
    auto&     src     = flow.emplaceBlock<SineSource>();
    auto&     trigger = flow.emplaceBlock<TTrigger>({{"gr:compute_domain", std::string(domain)}});
    auto&     sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", true}, {"log_tags", true}});

    expect(flow.connect<"out", "in">(src, trigger).has_value());
    expect(flow.connect<"out", "in">(trigger, sink).has_value());

    gr::scheduler::Simple<> sched;
    expect(sched.exchange(std::move(flow)).has_value());
    expect(sched.runAndWait().has_value()) << std::format("the chain must run to completion on '{}'", domain);

    gr::log::setBackend(previousBackend);

    RunResult result;
    result.samples.assign(sink._samples.begin(), sink._samples.end());
    for (const auto& tag : sink._tags) {
        result.tagIndices.push_back(tag.index);
        result.payloads.push_back({.name = tag.map.template value_or<std::string>(std::string_view(gr::tag::TRIGGER_NAME), std::string("<MISSING>")), .time = tag.map.template value_or<std::uint64_t>(std::string_view(gr::tag::TRIGGER_TIME), std::uint64_t{0U}), .timeError = tag.map.template value_or<std::uint64_t>(std::string_view(gr::tag::TRIGGER_TIME_ERROR), std::uint64_t{0U}), .offset = tag.map.template value_or<float>(std::string_view(gr::tag::TRIGGER_OFFSET), -1.f), .entries = tag.map.size(), .domain = readMeta(tag.map, "domain"), .executionTarget = readMeta(tag.map, "execution_target")});
    }
    std::println("  ┌─ tags that arrived back at the host from '{}' {}", domain, result.cpuFallbacks > 0UZ ? "(ran on the CPU: dispatch refused the kernel)" : "");
    std::println("  │ {:>7}  {:>14}  {:>12}  {:>10}  {:>7}  {:>10}  {:>16}", "index", "trigger_name", "trigger_time", "time_error", "offset", "domain", "execution_target");
    for (const auto& [index, payload] : std::views::zip(result.tagIndices, result.payloads)) {
        std::println("  │ {:>7}  {:>14}  {:>12}  {:>10}  {:>7.3f}  {:>10}  {:>16}", index, payload.name, payload.time, payload.timeError, payload.offset, payload.domain, payload.executionTarget);
    }
    std::println("  └─ {} tag(s), {} sample(s)", result.tagIndices.size(), result.samples.size());

    std::ignore = recorded.snapshot(
        [](const gr::log::LogRecord& record, void* user) noexcept {
            if (std::string_view(record.text, record.textLength).contains("a device kernel cannot build")) {
                ++*static_cast<std::size_t*>(user);
            }
        },
        &result.cpuFallbacks);
    return result;
}

} // namespace gr::device_spans_test

// AdaptiveCpp aborts if a kernel launches while Boost.UT runs suites from ~runner, so tests run from main (G10)
int main() {
    using namespace boost::ut;
    using namespace gr::device_spans_test;

    // so it orders deterministically against the log lines it is explaining
    std::println("note: the two 'a device kernel cannot build a tag' warnings below are EXPECTED -- the owning-payload");
    std::println("      block falls back to the CPU on each SYCL domain, and that is what the third test discriminates on.");

    const bool syclAvailable = gr::device::registerSyclRuntime();

    "a stateful zero-crossing trigger tags every second sine period"_test = [] {
        const RunResult onHost = runOn<ZeroCrossingTrigger>("host");

        expect(eq(onHost.samples.size(), static_cast<std::size_t>(kSamples))) << "every sample must reach the sink";
        // with a half-sample phase shift the rise happens at 16, 32, 48, ... and every second one is tagged
        const std::vector<std::size_t> expected{32UZ, 64UZ, 96UZ};
        expect(eq(onHost.tagIndices.size(), expected.size())) << "one tag per two periods over 8 periods";
        expect(std::ranges::equal(onHost.tagIndices, expected)) << "tags mark the start of every other sine wave";
    };

    "the same block, the same tags, on every SYCL domain"_test = [syclAvailable] {
        if (!syclAvailable) {
            return;
        }
        const RunResult onHost = runOn<ZeroCrossingTrigger>("host");
        expect(eq(onHost.cpuFallbacks, 0UZ)) << "the plain host domain never goes through device dispatch";

        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue; // not served on this machine
            }
            const RunResult onDevice = runOn<ZeroCrossingTrigger>(domain);
            expect(eq(onDevice.cpuFallbacks, 1UZ)) << std::format("an owning payload cannot be built in a kernel, so '{}' must refuse it", domain);
            expect(std::ranges::all_of(onDevice.payloads, [domain](const TagPayload& p) { return p.domain == domain && p.executionTarget == "host"; })) //
                << std::format("the tags must show the block was sent to '{}' yet built its payload on the host", domain);
            expect(std::ranges::equal(onDevice.samples, onHost.samples)) << std::format("samples must match the host on '{}'", domain);
            expect(std::ranges::equal(onDevice.tagIndices, onHost.tagIndices)) << std::format("tags must match the host on '{}'", domain);
        }
    };

    "a device body publishes more than it consumes"_test = [syclAvailable] {
        constexpr gr::Size_t kN     = 64U;
        const auto           onHost = runUpsamplerOn("host", kN);
        expect(eq(onHost.size(), std::size_t{2UZ * kN})) << "the reference: one input sample becomes two";

        if (!syclAvailable) {
            return;
        }
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue; // not served on this machine
            }
            const auto onDevice = runUpsamplerOn(domain, kN);
            expect(eq(onDevice.size(), std::size_t{2UZ * kN})) << std::format("'{}' must not bound the output by the input count", domain);
            expect(std::ranges::equal(onDevice, onHost)) << std::format("'{}' must produce the same interpolated stream as the host", domain);
        }
    };

    "a two-input span body consumes each port on its own"_test = [syclAvailable] {
        constexpr gr::Size_t kN     = 64U;
        const auto onHost = runTwoInputSpansOn("host", kN);
        expect(!onHost.empty()) << "the reference must produce something to compare against";
        // derived independently of the block: the upsampled arm repeats each sample as {v, -v}, so for even i the
        // second input is i/2 and for odd i it is -(i/2)
        const bool hostMatchesFormula = std::ranges::all_of(std::views::iota(0UZ, onHost.size()), [&onHost](std::size_t i) {
            const float a = static_cast<float>(i);
            const float b = (i % 2UZ == 0UZ) ? static_cast<float>(i / 2UZ) : -static_cast<float>(i / 2UZ);
            return onHost[i] == a - 2.f * b;
        });
        expect(hostMatchesFormula) << "the host reference must match a formula derived without the block, else the device comparison proves only agreement";

        if (!syclAvailable) {
            return;
        }
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue; // not served on this machine
            }
            std::vector<float> onDevice;
            expect(eq(gr::test::cpuFallbacksDuring([&] { onDevice = runTwoInputSpansOn(domain, kN); }), 0UZ)) //
                << std::format("'{}' must reach the kernel: a two-input body that fell back returns these very same numbers", domain);
            expect(eq(onDevice.size(), onHost.size())) << std::format("'{}' must consume both ports at the same rate the host does", domain);
            expect(std::ranges::equal(onDevice, onHost)) << std::format("'{}' must combine the two ports exactly as the host does", domain);
        }
    };

    "each input port's tags stay on that port"_test = [syclAvailable] {
        const float onHost = runTwoPortTagsOn("host");
        expect(eq(onHost, 501.f)) << "the reference: port 0 contributes 1, port 1 contributes 5 weighted by 100";

        if (!syclAvailable) {
            return;
        }
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue; // not served on this machine
            }
            float onDevice = -1.f;
            expect(eq(gr::test::cpuFallbacksDuring([&] { onDevice = runTwoPortTagsOn(domain); }), 0UZ)) //
                << std::format("'{}' must read the tags inside the kernel, not on the host", domain);
            expect(eq(onDevice, 501.f)) << std::format("'{}' must stage each port's tags into that port's own slots", domain);
        }
    };

    "input tags reach a kernel"_test = [syclAvailable] {
        using namespace gr::testing;
        // a HOST source emits the tags, so this exercises the case that matters: the ring they live in belongs to
        // the host side of the boundary
        const auto runWithTags = [](std::string_view domain) {
            gr::Graph flow;
            auto&     src = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(64)}, {"mark_tag", false}});
            src._tags     = {{0UZ, {{"level", 1}}}, {16UZ, {{"level", 2}}}, {32UZ, {{"level", 4}}}};
            auto& counter = flow.emplaceBlock<InputTagCounter>({{"gr:compute_domain", std::string(domain)}});
            auto& sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"log_samples", true}, {"log_tags", true}});
            expect(flow.connect<"out", "in">(src, counter).has_value());
            expect(flow.connect<"out", "in">(counter, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value()) << std::format("the chain must run on '{}'", domain);
            return std::vector<float>(sink._samples.begin(), sink._samples.end());
        };

        const std::vector<float> onHost = runWithTags("host");
        const float              seenOnHost = std::ranges::max(onHost) - std::ranges::max(std::views::iota(0UZ, onHost.size()) | std::views::transform([](std::size_t i) { return static_cast<float>(i); }));
        // without this the comparison below could hold with every read returning nothing on both sides
        const bool anyTagValueApplied = std::ranges::any_of(std::views::iota(0UZ, onHost.size()), [&onHost](std::size_t i) { return onHost[i] != static_cast<float>(i); });
        expect(anyTagValueApplied) << "the host run must actually read a tag value, else the device comparison proves nothing";
        std::println(stderr, "  tag values read on 'host': {}", seenOnHost);

        if (!syclAvailable) {
            return;
        }
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue;
            }
            const std::vector<float> onDevice = runWithTags(domain);
            std::println(stderr, "  input tags reached the kernel on '{}': {}", domain, std::ranges::equal(onDevice, onHost) ? "yes" : "NO");
            expect(std::ranges::equal(onDevice, onHost)) << std::format("a kernel on '{}' must see the same input tags the host sees", domain);
        }
    };

    "a kernel-built tag payload survives the device publish path"_test = [syclAvailable] {
        // the discriminator: this block publishes a VIEW, so no fallback warning is emitted and the tags really
        // travel kernel -> formatAt/try_emplace -> pre-reserved slot -> host replay -> the real tag buffer.
        const RunResult                onHost = runOn<ZeroCrossingTriggerView>("host");
        const std::vector<std::size_t> expected{32UZ, 64UZ, 96UZ};
        expect(std::ranges::equal(onHost.tagIndices, expected)) << "the view form must tag the same samples as the owning form";
        expect(std::ranges::all_of(onHost.payloads, [](const TagPayload& p) { return p.executionTarget == "host"; })) << "the plain host domain runs host code, not a kernel";
        expect(eq(onHost.payloads.size(), 3UZ));
        if (onHost.payloads.size() == 3UZ) {
            expect(onHost.payloads[0] == TagPayload{.name = "zero-crossing", .time = 32'000U, .timeError = 0U, .offset = 0.f, .entries = 5UZ, .domain = "host", .executionTarget = "host"}) //
                << "the four required trigger keys plus the nested meta map, with the values the block wrote";
        }

        if (!syclAvailable) {
            return;
        }
        for (std::string_view domain : {"host:sycl", "gpu:sycl"}) {
            if (gr::device::DeviceContextRegistry::instance().tryResolve(domain) == nullptr) {
                continue;
            }
            std::println("  kernel-built tags exercised on '{}'", domain);
            const RunResult onDevice = runOn<ZeroCrossingTriggerView>(domain);
            expect(eq(onDevice.cpuFallbacks, 0UZ)) << std::format("the view form must run as a kernel on '{}', not fall back", domain);
            expect(std::ranges::equal(onDevice.samples, onHost.samples)) << std::format("samples must match the host on '{}'", domain);
            expect(std::ranges::equal(onDevice.tagIndices, onHost.tagIndices)) << std::format("kernel-published tags must match the host on '{}'", domain);
            expect(std::ranges::equal(onDevice.payloads, onHost.payloads, [](const TagPayload& a, const TagPayload& b) { return a.sameTrigger(b); })) << std::format("the kernel-built trigger contract must arrive intact on '{}'", domain);
            const bool reportsItsOwnDomain = std::ranges::all_of(onDevice.payloads, [domain](const TagPayload& p) { return p.domain == domain; });
            expect(reportsItsOwnDomain) << std::format("every tag must name '{}', the domain the block was told to use", domain);
            expect(std::ranges::all_of(onDevice.payloads, [](const TagPayload& p) { return p.executionTarget == "device"; })) //
                << std::format("on '{}' the payload must report the device compilation pass, i.e. it really ran as a kernel", domain);
        }
    };

    return 0;
}
