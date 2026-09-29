#include <boost/ut.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/device/BackendDetect.hpp>
#include <gnuradio-4.0/test/DeviceTestHelper.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

namespace gr::event_transmission_test {

constexpr std::uint32_t kKeys         = 1U;
constexpr std::uint32_t kPayload      = 32U;
constexpr std::size_t   kKernelEvents = 16UZ;
constexpr std::int64_t  kKernelSum    = static_cast<std::int64_t>(kKernelEvents * (kKernelEvents - 1UZ) / 2UZ);
constexpr std::size_t   kSlotBytes    = ((gr::pmt::blobBytesForKeys(kKeys, kPayload) + gr::pmt::kBlobAlignment - 1UZ) / gr::pmt::kBlobAlignment) * gr::pmt::kBlobAlignment;

struct alignas(gr::pmt::kBlobAlignment) EventSlot {
    std::array<std::byte, kSlotBytes> bytes{};
};

static_assert(sizeof(EventSlot) == kSlotBytes);

[[nodiscard]] bool writeEvent(std::span<std::byte> slot, std::int64_t seq) noexcept { return gr::property_map_view::formatAt(slot, kPayload, gr::pmt::entryCapacityForKeys(kKeys)).try_emplace(std::string_view{"seq"}, seq); }

[[nodiscard]] std::int64_t readEvent(std::span<const std::byte> slot) noexcept {
    const std::int64_t* seq = gr::pmt::ValueMap::makeView(slot).get_if<std::int64_t>(std::string_view{"seq"});
    return seq == nullptr ? -1 : *seq;
}

using SteppedScheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::externalStep>;

template<typename TGraph>
[[nodiscard]] std::size_t runToCompletion(SteppedScheduler& scheduler, TGraph&& graph, std::size_t maxSteps = 4096UZ) {
    if (!scheduler.exchange(std::forward<TGraph>(graph)) || !scheduler.changeStateTo(gr::lifecycle::State::INITIALISED) || !scheduler.changeStateTo(gr::lifecycle::State::RUNNING)) {
        return maxSteps;
    }
    std::size_t steps = 0UZ;
    while (steps < maxSteps) {
        const gr::work::Result result = scheduler.step();
        ++steps;
        if (result.status == gr::work::Status::DONE || !gr::lifecycle::isActive(scheduler.state())) {
            return steps;
        }
        if (result.status == gr::work::Status::ERROR) {
            return maxSteps;
        }
    }
    return steps;
}

struct EventEmitter : gr::Block<EventEmitter> {
    gr::EventPortOut evtOut;

    gr::Size_t n_events = 32U;

    GR_MAKE_REFLECTABLE(EventEmitter, evtOut, n_events);

    gr::Size_t _emitted = 0U;

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        if (_emitted >= n_events || outSpan.size() == 0UZ) {
            outSpan.publish(0UZ);
            this->requestStop();
            return gr::work::Status::DONE;
        }
        gr::pmt::StackValueMap<kKeys, kPayload> event;
        if (!event.view().try_emplace(std::string_view{"seq"}, static_cast<std::int64_t>(_emitted))) {
            return gr::work::Status::ERROR;
        }
        if (!gr::emitEvent(outSpan, 0UZ, event.view())) {
            return gr::work::Status::ERROR;
        }
        outSpan.publish(1UZ);
        ++_emitted;
        return gr::work::Status::OK;
    }
};

struct EventReceiver : gr::Block<EventReceiver> {
    gr::EventPortIn evtIn;

    gr::Size_t n_events_max = 0U; // 0 -> run until the sources finish

    GR_MAKE_REFLECTABLE(EventReceiver, evtIn, n_events_max);

    std::vector<std::int64_t> _received;
    std::size_t               _skipped = 0UZ;
    std::size_t               _calls   = 0UZ;

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan) {
        ++_calls;
        for (const gr::property_map_view& event : inSpan) {
            if (event.empty()) {
                ++_skipped;
                continue;
            }
            const std::int64_t* seq = event.get_if<std::int64_t>(std::string_view{"seq"});
            _received.push_back(seq == nullptr ? -1 : *seq);
        }
        if (!inSpan.consume(inSpan.size())) {
            return gr::work::Status::ERROR;
        }
        if (n_events_max != 0U && _received.size() >= static_cast<std::size_t>(n_events_max)) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

struct SilentEmitter : gr::Block<SilentEmitter> {
    gr::EventPortOut evtOut;

    GR_MAKE_REFLECTABLE(SilentEmitter, evtOut);

    gr::Size_t               _calls = 0U;
    std::vector<std::size_t> _spanSizes;

    gr::work::Status processBulk(gr::OutputSpanLike auto& outSpan) {
        ++_calls;
        _spanSizes.push_back(outSpan.size());
        outSpan.publish(0UZ);
        if (_calls >= 3U) {
            this->requestStop();
            return gr::work::Status::DONE;
        }
        return gr::work::Status::OK;
    }
};

struct GateConsumingEvents : gr::Block<GateConsumingEvents> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    gr::EventPortIn    evtIn;

    GR_MAKE_REFLECTABLE(GateConsumingEvents, in, out, evtIn);

    gr::Size_t  _forwarded    = 0U;
    std::size_t _eventsSeen   = 0UZ;
    std::size_t _calls        = 0UZ;
    std::size_t _evtAvailable = 0UZ;

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSpan) {
        _eventsSeen += evtSpan.size();
        _calls++;
        _evtAvailable = evtIn.streamReader().available();
        if (!evtSpan.consume(evtSpan.size())) {
            return gr::work::Status::ERROR;
        }

        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        std::ranges::copy(inSpan | std::views::take(n), outSpan.begin());
        _forwarded += static_cast<gr::Size_t>(n);
        outSpan.publish(n);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }
};

struct GateEmittingEvents : gr::Block<GateEmittingEvents> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    gr::EventPortOut   evtOut;

    GR_MAKE_REFLECTABLE(GateEmittingEvents, in, out, evtOut);

    gr::Size_t _forwarded = 0U;

    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, gr::OutputSpanLike auto& outSpan, gr::OutputSpanLike auto& eventSpan) {
        const std::size_t n = std::min(inSpan.size(), outSpan.size());
        for (std::size_t i = 0UZ; i < n; ++i) {
            outSpan[i] = inSpan[i];
        }
        _forwarded += static_cast<gr::Size_t>(n);
        if (!inSpan.consume(n)) {
            return gr::work::Status::ERROR;
        }
        outSpan.publish(n);
        eventSpan.publish(0UZ);
        return gr::work::Status::OK;
    }
};

} // namespace gr::event_transmission_test

int main() {
    using namespace boost::ut;
    using namespace gr::event_transmission_test;
    using gr::testing::operator""_domain_test;

    "an emitter block transmits events to a receiver block through a scheduler"_test = [] {
        constexpr gr::Size_t kEvents = 32U;

        gr::Graph graph;
        auto&     emitter  = graph.emplaceBlock<EventEmitter>({{"n_events", kEvents}});
        auto&     receiver = graph.emplaceBlock<EventReceiver>();
        expect(graph.connect<"evtOut", "evtIn">(emitter, receiver, {.minBufferSize = 16UZ}).has_value());

        SteppedScheduler  scheduler;
        const std::size_t steps = runToCompletion(scheduler, std::move(graph));

        expect(lt(steps, 4096UZ)) << "the graph must end on its own once its source signals end-of-stream";
        expect(eq(receiver._received.size(), std::size_t{kEvents})) << std::format("receiver saw {} of {} events", receiver._received.size(), kEvents);
        expect(std::ranges::is_sorted(receiver._received)) << "one producer must arrive in publication order";
        expect(lt(receiver._skipped, std::size_t{kEvents})) << std::format("claiming one slot per publish must not waste the ring, skipped {}", receiver._skipped);
    };

    "a fan-in receiver ends only once every source has finished"_test = [] {
        constexpr gr::Size_t kPerSource = 8U;

        gr::Graph graph;
        auto&     fast     = graph.emplaceBlock<EventEmitter>({{"n_events", kPerSource}});
        auto&     slow     = graph.emplaceBlock<EventEmitter>({{"n_events", gr::Size_t{4U * kPerSource}}});
        auto&     receiver = graph.emplaceBlock<EventReceiver>();
        expect(graph.connect<"evtOut", "evtIn">(fast, receiver, {.minBufferSize = 16UZ}).has_value());
        expect(graph.connect<"evtOut", "evtIn">(slow, receiver, {.minBufferSize = 16UZ}).has_value());

        SteppedScheduler  scheduler;
        const std::size_t steps = runToCompletion(scheduler, std::move(graph));

        expect(lt(steps, 4096UZ)) << "both sources signalled end-of-stream, so the graph must end";
        expect(eq(receiver._received.size(), std::size_t{5U * kPerSource})) << "the early finisher must not cut the other one short";
        expect(eq(receiver.evtIn.nWriters(), 0UZ)) << "a source that has stopped must let go of the shared ring";
    };

    "a receiver that finishes first shuts its sources down"_test = [] {
        gr::Graph graph;
        auto&     endless  = graph.emplaceBlock<EventEmitter>({{"n_events", std::numeric_limits<gr::Size_t>::max()}});
        auto&     receiver = graph.emplaceBlock<EventReceiver>({{"n_events_max", gr::Size_t{8U}}});
        expect(graph.connect<"evtOut", "evtIn">(endless, receiver, {.minBufferSize = 16UZ}).has_value());

        SteppedScheduler  scheduler;
        const std::size_t steps = runToCompletion(scheduler, std::move(graph));

        expect(lt(steps, 4096UZ)) << "a source with no consumer left must stop itself";
        expect(!endless.evtOut.isConnected()) << "the finished receiver must have dropped the edge";
    };

    "two sources finishing together still end the graph"_test = [] {
        constexpr gr::Size_t kPerSource = 8U;
        gr::Graph            graph;
        auto&                left     = graph.emplaceBlock<EventEmitter>({{"n_events", kPerSource}});
        auto&                right    = graph.emplaceBlock<EventEmitter>({{"n_events", kPerSource}});
        auto&                receiver = graph.emplaceBlock<EventReceiver>();
        expect(graph.connect<"evtOut", "evtIn">(left, receiver, {.minBufferSize = 16UZ}).has_value());
        expect(graph.connect<"evtOut", "evtIn">(right, receiver, {.minBufferSize = 16UZ}).has_value());

        SteppedScheduler  scheduler;
        const std::size_t steps = runToCompletion(scheduler, std::move(graph));
        expect(lt(steps, 4096UZ)) << "two sources finishing together must still end the graph";
        expect(eq(receiver._received.size(), std::size_t{2U * kPerSource}));
    };

    "an event port does not gate the stream ports beside it"_test = [] {
        constexpr gr::Size_t kSamples = 1024U;

        gr::Graph graph;
        auto&     source = graph.emplaceBlock<gr::testing::ConstantSource<float>>({{"n_samples_max", kSamples}});
        auto&     gate   = graph.emplaceBlock<GateEmittingEvents>();
        auto&     sink   = graph.emplaceBlock<gr::testing::CountingSink<float>>();
        auto&     events = graph.emplaceBlock<EventReceiver>();
        expect(graph.connect<"out", "in">(source, gate).has_value());
        expect(graph.connect<"out", "in">(gate, sink).has_value());
        expect(graph.connect<"evtOut", "evtIn">(gate, events).has_value());

        SteppedScheduler  scheduler;
        const std::size_t steps = runToCompletion(scheduler, std::move(graph));
        expect(lt(steps, 4096UZ)) << "a stream graph with a silent event port must still end";

        const std::size_t tagsOnEventEdge = events.evtIn.tagReader().get().size();
        expect(eq(tagsOnEventEdge, 0UZ)) << "a stream tag must not be forwarded onto the event edge beside it";
        expect(eq(gate._forwarded, kSamples)) << "a connected but silent event port must not stall the stream";
        expect(eq(static_cast<gr::Size_t>(sink.count), kSamples));
        expect(lt(events._calls, std::size_t{16})) << std::format("an idle event input must not be polled per iteration, saw {} calls for {} samples", events._calls, kSamples);
    };

    "an edge sizes the event ring"_test = [] {
        gr::Graph graph;
        auto&     emitter  = graph.emplaceBlock<EventEmitter>();
        auto&     receiver = graph.emplaceBlock<EventReceiver>();
        expect(graph.connect<"evtOut", "evtIn">(emitter, receiver, {.minBufferSize = 8192UZ}).has_value());
        expect(graph.connectPendingEdges());

        expect(ge(emitter.evtOut.bufferSize(), 8192UZ)) << "a size the default would not have given must reach the ring the edge uses";
        expect(eq(emitter.evtOut.bufferIdentity(), receiver.evtIn.bufferIdentity())) << "both ports must share one ring";
    };

    "an event wider than a chunk is stored whole rather than refused"_test = [] {
        gr::EventPortOut out({.streamBufferSize = 16UZ});
        gr::EventPortIn  in;
        expect(out.connect(in).has_value());

        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, kSlotBytes * 4UZ> oversized{};
        auto                                                                     wide = gr::property_map_view::formatAt(oversized, kPayload * 4UZ, gr::pmt::entryCapacityForKeys(kKeys * 4UZ));
        expect(wide.try_emplace(std::string_view{"seq"}, std::int64_t{1}));

        {
            auto span = out.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
            expect(gr::emitEvent(span, 0UZ, wide).has_value()) << "a blob too wide for a pooled chunk gets one of its own";
            span.publish(1UZ);
        }

        auto received = in.streamReader().get();
        expect(eq(received.size(), 1UZ));
        if (!received.empty()) {
            const std::int64_t* seq = received[0].get_if<std::int64_t>(std::string_view{"seq"});
            expect(seq != nullptr && *seq == 1) << "and it survives the trip intact";
        }
    };

    "a tag offered to an event edge is placed at its claim"_test = [] {
        gr::EventPortOut out;
        gr::EventPortIn  in;
        expect(out.connect(in).has_value());

        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, kSlotBytes> scratch{};
        {
            auto span = out.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
            expect(writeEvent(scratch, 7));
            expect(gr::emitEvent(span, 0UZ, gr::pmt::ValueMap::makeView(std::span<const std::byte>{scratch})).has_value());
            gr::property_map ignored;
            ignored["marker"] = true;
            span.publishTag(ignored, 0UZ);
            span.publish(1UZ);
        }

        expect(eq(in.tagReader().get().size(), 1UZ)) << "a span knows the claim its tag belongs to, so the position is well defined";
        expect(eq(in.streamReader().available(), 1UZ)) << "and the event itself is unaffected";
    };

    "an input that still holds events has not reached its end"_test = [] {
        gr::EventPortIn                                                    in;
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, kSlotBytes> scratch{};
        {
            gr::EventPortOut out;
            expect(out.connect(in).has_value());
            auto span = out.reserve<gr::SpanReleasePolicy::ProcessNone>(1UZ);
            expect(writeEvent(scratch, 5));
            expect(gr::emitEvent(span, 0UZ, gr::pmt::ValueMap::makeView(std::span<const std::byte>{scratch})).has_value());
            span.publish(1UZ);
        }

        expect(eq(in.streamReader().available(), 1UZ)) << "the event outlives the source that wrote it";
        expect(!in.pollEndOfStream()) << "an end declared here would discard an event nobody has read";

        {
            auto received = in.streamReader().get<gr::SpanReleasePolicy::ProcessAll>(1UZ);
            expect(eq(received.size(), 1UZ));
            std::ignore = received.consume(1UZ);
        }
        expect(in.pollEndOfStream()) << "once drained, the edge really is finished";
    };

    "a drained event input finishes the block even with stream data buffered"_test = [] {
        GateConsumingEvents gate;
        gr::PortOut<float>  source;
        gr::PortIn<float>   sink;
        gr::EventPortOut    events;
        expect(source.connect(gate.in).has_value()) << fatal;
        expect(gate.out.connect(sink).has_value()) << fatal;
        expect(events.connect(gate.evtIn).has_value()) << fatal;
        expect(gate.changeStateTo(gr::lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(gate.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;
        {
            auto samples = source.streamWriter().reserve<gr::SpanReleasePolicy::ProcessAll>(8UZ);
            std::ranges::fill(samples, 1.f);
            samples.publish(8UZ);
        }
        expect(events.disconnect().has_value()) << fatal;
        expect(eq(gate.in.streamReader().available(), 8UZ));
        expect(gate.evtIn.pollEndOfStream());

        const gr::work::Result result = gate.work(8UZ);

        expect(result.status == gr::work::Status::DONE);
        expect(gate.state() == gr::lifecycle::State::STOPPED);
        expect(eq(gate._forwarded, 0U)) << "event completion intentionally ends a mixed-input block";
    };

    "a finished event source closes its ring in the same work call"_test = [] {
        EventEmitter    source;
        gr::EventPortIn sink;
        source.n_events = 0U;
        expect(source.evtOut.connect(sink).has_value()) << fatal;
        expect(eq(sink.streamReader().buffer().n_writers(), 1UZ));
        expect(source.changeStateTo(gr::lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(source.changeStateTo(gr::lifecycle::State::RUNNING).has_value()) << fatal;

        const gr::work::Result result = source.work(1UZ);

        expect(result.status == gr::work::Status::DONE);
        expect(eq(sink.streamReader().buffer().n_writers(), 0UZ)) << "DONE must release the producer without requiring another work call";
        expect(sink.pollEndOfStream());
    };

    "fan-in shares one tag buffer"_test = [] {
        gr::EventPortOut first;
        gr::EventPortOut second;
        gr::EventPortIn  in;
        expect(first.connect(in).has_value());
        expect(second.connect(in).has_value());

        expect(eq(in.streamReader().buffer().n_writers(), 2UZ));
        expect(eq(in.tagReader().buffer().n_writers(), 2UZ)) << "both sources join the one tag ring";
    };

    expect(gr::testing::missingRequiredDomains().empty()) << "a domain this run was configured for must be served";

    "an event crosses to a kernel, back, and between two kernels"_domain_test = [](auto& ctx) {
        EventSlot*    shared = ctx.template alloc<EventSlot>(kKernelEvents);
        std::int64_t* sum    = ctx.template alloc<std::int64_t>(3UZ);
        std::ranges::fill(std::span{shared, kKernelEvents}, EventSlot{});
        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;

        for (std::size_t i = 0UZ; i < kKernelEvents; ++i) {
            boost::ut::expect(writeEvent(shared[i].bytes, static_cast<std::int64_t>(i)));
        }
        ctx.launch([shared, sum](const gr::testing::DeviceTestHandle& device) {
            std::int64_t total = 0;
            for (std::size_t i = 0UZ; i < kKernelEvents; ++i) {
                total += readEvent(shared[i].bytes);
            }
            sum[0] = total;
            gr::testing::expect(device, total == kKernelSum, "a kernel decoded {} of an expected {}", total, kKernelSum);
        });

        EventSlot* const deviceOnly = ctx.template allocDeviceOrShared<EventSlot>(kKernelEvents);

        ctx.launchRange(kKernelEvents, [deviceOnly](const gr::testing::DeviceTestHandle& device, std::size_t i) {
            const bool written = writeEvent(deviceOnly[i].bytes, static_cast<std::int64_t>(i));
            gr::testing::expect(device, written, "a kernel-side write of slot {} failed", i);
        });
        ctx.launch([deviceOnly, sum](const gr::testing::DeviceTestHandle& device) {
            std::int64_t total = 0;
            for (std::size_t i = 0UZ; i < kKernelEvents; ++i) {
                total += readEvent(deviceOnly[i].bytes);
            }
            sum[1] = total;
            gr::testing::expect(device, total == kKernelSum, "device-only slots carried {} of an expected {}", total, kKernelSum);
        });

        ctx.context().copy(shared, deviceOnly, kKernelEvents * sizeof(EventSlot));
        std::int64_t onHost = 0;
        for (std::size_t i = 0UZ; i < kKernelEvents; ++i) {
            onHost += readEvent(shared[i].bytes);
        }
        boost::ut::expect(boost::ut::eq(onHost, kKernelSum)) << "the host must decode what a kernel wrote";
    } | gr::testing::kAllDomains;

    return 0;
}
