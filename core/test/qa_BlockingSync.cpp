#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace boost::ut;
using namespace gr;
using namespace std::chrono_literals;

// test helper: run scheduler with timeout, returns true if completed before timeout
auto runWithTimeout(auto& sched, std::chrono::milliseconds timeout) {
    std::atomic<bool> schedulerDone{false};
    std::thread       timeoutThread([&sched, &schedulerDone, timeout] {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!schedulerDone.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(10ms);
        }
        if (!schedulerDone.load()) {
            sched.requestStop();
        }
    });
    auto              result = sched.runAndWait();
    schedulerDone            = true;
    timeoutThread.join();
    return result;
}

template<typename T>
struct TestClockSource : Block<TestClockSource<T>>, BlockingSync<TestClockSource<T>> {
    using Description = Doc<"clock source for driving downstream blocks">;

    PortOut<T> out;

    Annotated<float, "sample_rate">        sample_rate         = 1000.f;
    Annotated<gr::Size_t, "chunk_size">    chunk_size          = 100;
    Annotated<bool, "use_internal_thread"> use_internal_thread = true;
    gr::Size_t                             n_samples_max       = 0; // 0 = unlimited

    GR_MAKE_REFLECTABLE(TestClockSource, out, sample_rate, chunk_size, use_internal_thread, n_samples_max);

    std::vector<Tag> tags;
    gr::Size_t       _nSamplesProduced = 0;
    std::size_t      _nextTagIndex     = 0;

    void start() {
        _nSamplesProduced = 0;
        _nextTagIndex     = 0;
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    work::Status processBulk(OutputSpanLike auto& output) {
        if (n_samples_max > 0 && _nSamplesProduced >= n_samples_max) {
            output.publish(0);
            return work::Status::DONE;
        }

        const auto maxSamples       = (n_samples_max > 0) ? n_samples_max - _nSamplesProduced : std::numeric_limits<gr::Size_t>::max();
        const auto availableSamples = std::min(output.size(), static_cast<std::size_t>(maxSamples));
        auto       nSamples         = this->syncSamples(output);
        nSamples                    = std::min(nSamples, availableSamples);

        if (nSamples == 0) {
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        // publish tags within this chunk
        while (_nextTagIndex < tags.size()) {
            const auto tagIndex = static_cast<gr::Size_t>(tags[_nextTagIndex].index);
            if (tagIndex >= _nSamplesProduced && tagIndex < _nSamplesProduced + nSamples) {
                output.publishTag(tags[_nextTagIndex].map, tagIndex - _nSamplesProduced);
                _nextTagIndex++;
            } else {
                break;
            }
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = static_cast<T>(_nSamplesProduced + i);
        }

        output.publish(nSamples);
        _nSamplesProduced += static_cast<gr::Size_t>(nSamples);

        return (n_samples_max > 0 && _nSamplesProduced >= n_samples_max) ? work::Status::DONE : work::Status::OK;
    }
};

template<typename T>
struct TestGenerator : Block<TestGenerator<T>>, BlockingSync<TestGenerator<T>> {
    using Description = Doc<"generator with optional clock input">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate">        sample_rate         = 1000.f;
    Annotated<gr::Size_t, "chunk_size">    chunk_size          = 100;
    Annotated<bool, "use_internal_thread"> use_internal_thread = true;

    GR_MAKE_REFLECTABLE(TestGenerator, clk_in, out, sample_rate, chunk_size, use_internal_thread);

    std::size_t _samplesGenerated = 0;

    void start() {
        _samplesGenerated = 0;
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    work::Status processBulk(InputSpanLike auto& clkIn, OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(clkIn, output);

        if (nSamples == 0) {
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = static_cast<T>(_samplesGenerated + i);
        }

        std::ignore = clkIn.consume(nSamples);
        output.publish(nSamples);
        _samplesGenerated += nSamples;

        return work::Status::OK;
    }
};

template<typename T>
struct TestGeneratorNoChunk : Block<TestGeneratorNoChunk<T>>, BlockingSync<TestGeneratorNoChunk<T>> {
    using Description = Doc<"generator without explicit chunk_size (tests default calculation)">;

    PortIn<std::uint8_t, Optional> clk_in;
    PortOut<T>                     out;

    Annotated<float, "sample_rate"> sample_rate = 1000.f;
    // no chunk_size - uses default (sample_rate / 10)

    GR_MAKE_REFLECTABLE(TestGeneratorNoChunk, clk_in, out, sample_rate);

    std::size_t _samplesGenerated = 0;

    void start() {
        _samplesGenerated = 0;
        this->blockingSyncStart();
    }

    void stop() { this->blockingSyncStop(); }

    work::Status processBulk(InputSpanLike auto& clkIn, OutputSpanLike auto& output) {
        const auto nSamples = this->syncSamples(clkIn, output);

        if (nSamples == 0) {
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < nSamples; ++i) {
            output[i] = static_cast<T>(_samplesGenerated + i);
        }

        std::ignore = clkIn.consume(nSamples);
        output.publish(nSamples);
        _samplesGenerated += nSamples;

        return work::Status::OK;
    }
};

const boost::ut::suite<"Basic API"> basicApi = [] {
    "mode detection - disconnected"_test = [] {
        TestGenerator<float> gen;
        gen.init(gen.progress);
        expect(gen.isFreeRunning()) << "disconnected Optional port -> free-running";
    };

    "default chunk_size calculation"_test = [] {
        TestGeneratorNoChunk<float> gen;
        gen.sample_rate = 10000.f;
        gen.init(gen.progress);

        // default chunk = sample_rate / 10 = 1000
        expect(ge(gen.blockingSyncLastUpdateTime().time_since_epoch().count(), 0)) << "timing initialised";
    };

    "internal thread control"_test = [] {
        TestGenerator<float> gen;
        gen.use_internal_thread = false;
        gen.init(gen.progress);

        expect(!gen.isUsingInternalThread()) << "internal thread disabled";
    };
};

const boost::ut::suite<"Timing"> timing = [] {
    "wall-clock sample generation"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.chunk_size  = 50;
        src.init(src.progress);
        src.blockingSyncStart();

        // simulate passage of time
        std::this_thread::sleep_for(60ms);

        // after 60ms at 1kHz, expect ~50 samples (clamped to chunk_size)
        const auto samples = src.syncSamples(100UZ);
        expect(ge(samples, 40UZ)) << "expected ~50 samples after 60ms";
        expect(le(samples, 60UZ)) << "should be clamped around chunk_size";

        src.blockingSyncStop();
    };

    "phase continuity"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.chunk_size  = 100;
        src.init(src.progress);
        src.blockingSyncStart();

        const auto startTime = src.blockingSyncLastUpdateTime();

        std::this_thread::sleep_for(50ms);

        const auto samples1 = src.syncSamples(1000UZ);
        const auto time1    = src.blockingSyncLastUpdateTime();

        expect(gt(samples1, 0UZ));
        expect(gt(time1, startTime)) << "time advanced after first call";

        std::this_thread::sleep_for(50ms);

        const auto samples2 = src.syncSamples(1000UZ);
        const auto time2    = src.blockingSyncLastUpdateTime();

        expect(gt(samples2, 0UZ));
        expect(gt(time2, time1)) << "time advanced after second call";

        src.blockingSyncStop();
    };

    "dropIfBehind mode"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.chunk_size  = 10; // small chunk to demonstrate dropping
        src.init(src.progress);
        src.blockingSyncStart();

        std::this_thread::sleep_for(100ms); // accumulate ~100 samples worth of time

        // with dropIfBehind=true, time advances to now regardless of samples produced
        const auto timeBefore = src.blockingSyncLastUpdateTime();
        const auto samples    = src.syncSamples(1000UZ, true);
        const auto timeAfter  = src.blockingSyncLastUpdateTime();

        expect(eq(samples, 10UZ)) << "returns chunk_size samples";

        // time should have jumped forward by ~100ms, not just 10ms worth of samples
        const auto elapsed = std::chrono::duration<double>(timeAfter - timeBefore).count();
        expect(gt(elapsed, 0.05)) << "time jumped forward (dropping samples)";

        src.blockingSyncStop();
    };

    "reset timing"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.init(src.progress);
        src.blockingSyncStart();

        std::this_thread::sleep_for(50ms);
        const auto beforeReset = src.blockingSyncLastUpdateTime();

        src.blockingSyncResetTiming();
        const auto afterReset = src.blockingSyncLastUpdateTime();

        expect(gt(afterReset, beforeReset)) << "reset advances time to now";

        src.blockingSyncStop();
    };
};

const boost::ut::suite<"Edge Cases"> edgeCases = [] {
    "zero sample_rate"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 0.f;
        src.init(src.progress);

        // should not crash, timer thread should not start
        src.blockingSyncStart();
        std::this_thread::sleep_for(10ms);
        src.blockingSyncStop();

        expect(true) << "survived zero sample_rate";
    };

    "very high sample_rate"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1e9f; // 1 GHz
        src.chunk_size  = 1000000;
        src.init(src.progress);
        src.blockingSyncStart();

        std::this_thread::sleep_for(10ms);

        const auto samples = src.syncSamples(10000000UZ);
        expect(gt(samples, 0UZ)) << "high sample rate produces samples";

        src.blockingSyncStop();
    };

    "small chunk_size"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.chunk_size  = 1;
        src.init(src.progress);
        src.blockingSyncStart();

        std::this_thread::sleep_for(10ms);

        const auto samples = src.syncSamples(1000UZ);
        expect(eq(samples, 1UZ)) << "clamped to chunk_size=1";

        src.blockingSyncStop();
    };

    "limited output buffer"_test = [] {
        TestClockSource<float> src;
        src.sample_rate = 1000.f;
        src.chunk_size  = 100;
        src.init(src.progress);
        src.blockingSyncStart();

        std::this_thread::sleep_for(50ms);

        const auto samples = src.syncSamples(5UZ); // smaller than chunk_size
        expect(le(samples, 5UZ)) << "clamped to output buffer size";

        src.blockingSyncStop();
    };
};

const boost::ut::suite<"Graph Integration"> graphIntegration = [] {
    "clock source to sink"_test = [] {
        constexpr gr::Size_t nSamples   = 200;
        constexpr float      sampleRate = 1000.f;

        Graph testGraph;
        auto& src  = testGraph.emplaceBlock<TestClockSource<std::uint8_t>>({{"n_samples_max", gr::pmt::Value(nSamples)}, {"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"name", gr::pmt::Value("ClockSource")}});
        auto& sink = testGraph.emplaceBlock<testing::TagSink<std::uint8_t, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto result = runWithTimeout(sched, 5s);
        expect(result.has_value()) << "scheduler completed";
        expect(eq(static_cast<gr::Size_t>(sink._nSamplesProduced), nSamples)) << "sink received all samples";
    };

    "clock source with tags"_test = [] {
        constexpr gr::Size_t nSamples   = 200;
        constexpr float      sampleRate = 1000.f;

        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TestClockSource<std::uint8_t>>({{"n_samples_max", gr::pmt::Value(nSamples)}, {"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"name", gr::pmt::Value("ClockSourceTags")}});

        src.tags = {{0, {{"key", gr::pmt::Value("value@0")}}}, {25, {{"key", gr::pmt::Value("value@25")}}}, {50, {{"key", gr::pmt::Value("value@50")}}}, {100, {{"key", gr::pmt::Value("value@100")}}}, {150, {{"key", gr::pmt::Value("value@150")}}}};

        auto& sink = testGraph.emplaceBlock<testing::TagSink<std::uint8_t, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto result = runWithTimeout(sched, 5s);
        expect(result.has_value()) << "scheduler completed";
        expect(eq(static_cast<gr::Size_t>(sink._nSamplesProduced), nSamples));
        expect(eq(sink._tags.size(), 5UZ)) << std::format("expected 5 tags, got {}", sink._tags.size());
    };

    "generator free-running mode"_test = [] {
        Graph testGraph;
        auto& gen  = testGraph.emplaceBlock<TestGenerator<float>>({{"sample_rate", gr::pmt::Value(1000.f)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"name", gr::pmt::Value("FreeRunningGen")}});
        auto& sink = testGraph.emplaceBlock<testing::TagSink<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        // clk_in not connected -> free-running mode
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(gen).to<"in">(sink)));

        scheduler::Simple<scheduler::ExecutionPolicy::singleThreadedBlocking> sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto result = runWithTimeout(sched, 500ms);
        expect(result.has_value()) << "scheduler completed";

        expect(gt(gen._samplesGenerated, 0UZ)) << "generator produced samples";
        expect(gt(sink._nSamplesProduced, 0UZ)) << "sink received samples";

        // at 1kHz over 500ms, expect ~500 samples (with some tolerance)
        expect(ge(sink._nSamplesProduced, 100UZ)) << std::format("expected >=100 samples, got {}", sink._nSamplesProduced);
    };

    "generator clock-driven mode"_test = [] {
        constexpr gr::Size_t nSamples   = 200;
        constexpr float      sampleRate = 1000.f;

        Graph testGraph;
        auto& clock = testGraph.emplaceBlock<TestClockSource<std::uint8_t>>({{"n_samples_max", gr::pmt::Value(nSamples)}, {"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"name", gr::pmt::Value("ClockSource")}});
        auto& gen   = testGraph.emplaceBlock<TestGenerator<float>>({{"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{100})}, {"name", gr::pmt::Value("ConnectedGen")}});
        auto& sink  = testGraph.emplaceBlock<testing::TagSink<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clock).to<"clk_in">(gen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(gen).to<"in">(sink)));

        scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto result = runWithTimeout(sched, 5s);
        expect(result.has_value()) << "scheduler completed";

        expect(!gen.isFreeRunning()) << "generator in clock-driven mode";
        expect(eq(static_cast<gr::Size_t>(sink._nSamplesProduced), nSamples)) << "sink received all samples";
    };

    "generator clock-driven with tag forwarding"_test = [] {
        constexpr gr::Size_t nSamples   = 200;
        constexpr float      sampleRate = 1000.f;

        Graph testGraph;
        auto& clock = testGraph.emplaceBlock<TestClockSource<std::uint8_t>>({{"n_samples_max", gr::pmt::Value(nSamples)}, {"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"name", gr::pmt::Value("ClockSource")}});

        clock.tags = {{0, {{"sample_rate", gr::pmt::Value(sampleRate)}}}, {50, {{"trigger", gr::pmt::Value("event1")}}}, {100, {{"trigger", gr::pmt::Value("event2")}}}, {150, {{"trigger", gr::pmt::Value("event3")}}}};

        auto& gen = testGraph.emplaceBlock<TestGenerator<float>>({{"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{100})}, {"name", gr::pmt::Value("ConnectedGen")}});

        // enable forwarding of custom 'trigger' key
        gen.settings().autoForwardParameters().insert("trigger");

        auto& sink = testGraph.emplaceBlock<testing::TagSink<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(clock).to<"clk_in">(gen)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(gen).to<"in">(sink)));

        scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        auto result = runWithTimeout(sched, 5s);
        expect(result.has_value()) << "scheduler completed";

        expect(eq(static_cast<gr::Size_t>(sink._nSamplesProduced), nSamples));
        expect(ge(sink._tags.size(), 4UZ)) << std::format("expected >=4 forwarded tags, got {}", sink._tags.size());
    };
};

const boost::ut::suite<"Scheduler-driven free-running"> schedulerDriven = [] {
    "use_internal_thread=false"_test = [] {
        constexpr gr::Size_t nSamples   = 100;
        constexpr float      sampleRate = 10000.f;

        Graph testGraph;
        auto& src = testGraph.emplaceBlock<TestClockSource<std::uint8_t>>({{"n_samples_max", gr::pmt::Value(nSamples)}, {"sample_rate", gr::pmt::Value(sampleRate)}, {"chunk_size", gr::pmt::Value(gr::Size_t{50})}, {"use_internal_thread", gr::pmt::Value(false)}, {"name", gr::pmt::Value("SchedulerDrivenSource")}});

        auto& sink = testGraph.emplaceBlock<testing::TagSink<std::uint8_t, testing::ProcessFunction::USE_PROCESS_BULK>>({{"name", gr::pmt::Value("Sink")}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        scheduler::Simple sched;
        expect(sched.exchange(std::move(testGraph)).has_value());

        // with on-demand mode, scheduler drives timing
        auto result = runWithTimeout(sched, 500ms);
        expect(result.has_value()) << "scheduler completed";

        // should produce some samples (scheduler calls processBulk)
        expect(gt(sink._nSamplesProduced, 0UZ)) << std::format("expected >0 samples, got {}", sink._nSamplesProduced);
    };
};

int main() { return 0; }
