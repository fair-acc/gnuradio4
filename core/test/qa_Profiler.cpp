#include <boost/ut.hpp>

#include <gnuradio-4.0/Profiler.hpp>

#ifndef __EMSCRIPTEN__
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace {

std::string readFile(const std::filesystem::path& path) {
    std::ifstream     file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool isValidJson(const std::string& json) {
    if (json.empty()) {
        return false;
    }
    // minimal validation: starts with '[', ends with ']', contains event markers
    const auto trimmed_start = json.find_first_not_of(" \t\n\r");
    const auto trimmed_end   = json.find_last_not_of(" \t\n\r");
    if (trimmed_start == std::string::npos) {
        return false;
    }
    return json[trimmed_start] == '[' && json[trimmed_end] == ']';
}

bool containsEvent(const std::string& json, std::string_view event_name) { return json.find(std::format("\"name\": \"{}\"", event_name)) != std::string::npos; }

bool containsEventType(const std::string& json, char ph) { return json.find(std::format("\"ph\": \"{}\"", ph)) != std::string::npos; }

} // anonymous namespace

const boost::ut::suite<"gr::profiling basic events"> _0 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "instant event"_test = [] {
        const std::string output_file = "test_instant.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("test_instant", "category", {{"key", "value"}});
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << "test file " << json << "\n";
        expect(containsEvent(json, "test_instant")) << "test file " << json << "\n";
        expect(containsEventType(json, 'I')) << "test file " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "counter event"_test = [] {
        const std::string output_file = "test_counter.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->counterEvent("test_counter", "metrics", {{"count", 42}});
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << "test file " << json << "\n";
        expect(containsEvent(json, "test_counter")) << "test file " << json << "\n";
        expect(containsEventType(json, 'C')) << "test file " << json << "\n";
        expect(json.find("\"count\":42") != std::string::npos) << "counter value present, test file " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "complete event"_test = [] {
        const std::string output_file = "test_complete.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            {
                auto event = handler->startCompleteEvent("test_complete", "work");
                std::this_thread::sleep_for(std::chrono::milliseconds{5});
            }
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << "test file " << json << "\n";
        expect(containsEvent(json, "test_complete")) << "test file " << json << "\n";
        expect(containsEventType(json, 'X')) << "test file " << json << "\n";
        expect(json.find("\"dur\":") != std::string::npos) << "duration field present " << "test file " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "async event with steps"_test = [] {
        const std::string output_file = "test_async.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            {
                auto event = handler->startAsyncEvent("test_async", "flow", {{"phase", "init"}});
                event.step();
                event.step();
            }
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << "test file " << json << "\n";
        expect(containsEvent(json, "test_async")) << "test file " << json << "\n";
        expect(containsEventType(json, 'b')) << "async start" << " test file " << json << "\n";
        expect(containsEventType(json, 'n')) << "async step" << " test file " << json << "\n";
        expect(containsEventType(json, 'e')) << "async end" << " test file " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "multiple event types"_test = [] {
        const std::string output_file = "test_mixed.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();

            handler->instantEvent("instant1");
            handler->counterEvent("counter1", "cat", {{"val", 1}});
            { auto complete = handler->startCompleteEvent("complete1"); }
            {
                auto async = handler->startAsyncEvent("async1");
                async.step();
            }
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << " test file " << json << "\n";
        expect(containsEvent(json, "instant1")) << " test file " << json << "\n";
        expect(containsEvent(json, "counter1")) << " test file " << json << "\n";
        expect(containsEvent(json, "complete1")) << " test file " << json << "\n";
        expect(containsEvent(json, "async1")) << " test file " << json << "\n";
        std::println("test file:\n{}\n", json);
        std::filesystem::remove(output_file);
    };
};

const boost::ut::suite<"gr::profiling multi-threaded"> _1 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "events from multiple threads"_test = [] {
        const std::string     output_file      = "test_multithread.trace";
        constexpr std::size_t kNumThreads      = 4UZ;
        constexpr std::size_t kEventsPerThread = 10UZ;
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});

            std::vector<std::thread> threads;
            threads.reserve(kNumThreads);
            for (std::size_t t = 0UZ; t < kNumThreads; ++t) {
                threads.emplace_back([&prof, t] {
                    auto handler = prof.forThisThread();
                    for (std::size_t i = 0UZ; i < kEventsPerThread; ++i) {
                        handler->instantEvent(std::format("thread{}_{}", t, i));
                    }
                });
            }
            for (auto& th : threads) {
                th.join();
            }
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << " test file " << json << "\n";

        std::size_t event_count = 0UZ;
        for (std::size_t t = 0UZ; t < kNumThreads; ++t) {
            for (std::size_t i = 0UZ; i < kEventsPerThread; ++i) {
                if (containsEvent(json, std::format("thread{}_{}", t, i))) {
                    ++event_count;
                }
            }
        }
        expect(eq(event_count, kNumThreads * kEventsPerThread)) << "all events recorded" << " test file " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "concurrent complete events"_test = [] {
        const std::string     output_file = "test_concurrent_complete.trace";
        constexpr std::size_t kNumThreads = 4UZ;
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});

            std::vector<std::thread> threads;
            threads.reserve(kNumThreads);
            for (std::size_t t = 0UZ; t < kNumThreads; ++t) {
                threads.emplace_back([&prof, t] {
                    auto handler = prof.forThisThread();
                    auto event   = handler->startCompleteEvent(std::format("work_thread{}", t));
                    std::this_thread::sleep_for(std::chrono::milliseconds{5});
                });
            }
            for (auto& th : threads) {
                th.join();
            }
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json));
        for (std::size_t t = 0UZ; t < kNumThreads; ++t) {
            expect(containsEvent(json, std::format("work_thread{}", t))) << " test file " << json << "\n";
        }
        std::filesystem::remove(output_file);
    };
};

const boost::ut::suite<"gr::profiling Options"> _2 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "custom output file"_test = [] {
        const std::string custom_file = "custom_output.trace";
        {
            Profiler prof(Options{.output_file = custom_file});
            auto     handler = prof.forThisThread();
            handler->instantEvent("custom_test");
        }
        expect(std::filesystem::exists(custom_file)) << " custom_file " << custom_file << "\n";
        std::filesystem::remove(custom_file);
    };

    "auto-generated filename"_test = [] {
        std::string generated_file;
        {
            Profiler prof(Options{});
            auto     handler = prof.forThisThread();
            handler->instantEvent("auto_name_test");
        }
        // find the generated file
        for (const auto& entry : std::filesystem::directory_iterator(".")) {
            const auto filename = entry.path().filename().string();
            if (filename.starts_with("profile.") && filename.ends_with(".trace")) {
                generated_file = filename;
                break;
            }
        }
        expect(!generated_file.empty()) << "auto-generated file created";
        if (!generated_file.empty()) {
            std::filesystem::remove(generated_file);
        }
    };

    "stdout output mode"_test = [] {
        std::stringstream captured;
        auto              old_buf = std::cout.rdbuf(captured.rdbuf());
        {
            Profiler prof(Options{.output_mode = OutputMode::StdOut, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("stdout_test");
        }
        std::cout.rdbuf(old_buf);
        const auto output = captured.str();
        expect(isValidJson(output)) << " output: " << output << "\n";
        expect(containsEvent(output, "stdout_test")) << " output: " << output << "\n";
    };

    "custom buffer size"_test = [] {
        const std::string output_file = "test_buffer_size.trace";
        {
            Profiler prof(Options{.output_file = output_file, .buffer_size = 1024UZ});
            auto     handler = prof.forThisThread();
            handler->instantEvent("buffer_test");
        }
        expect(std::filesystem::exists(output_file)) << " output: " << output_file << "\n";
        std::filesystem::remove(output_file);
    };
};

const boost::ut::suite<"gr::profiling consumer timing"> _3 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "update period respected"_test = [] {
        const std::string output_file = "test_timing.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{50}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("timing_test");

            // allow time for at least one update cycle
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        expect(std::filesystem::exists(output_file)) << " output: " << output_file << "\n";
        std::filesystem::remove(output_file);
    };

    "fast update period"_test = [] {
        const std::string output_file = "test_fast_update.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{1}});
            auto     handler = prof.forThisThread();
            for (int i = 0; i < 100; ++i) {
                handler->instantEvent(std::format("fast_{}", i));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << " json: " << json << "\n";
        expect(containsEvent(json, "fast_0")) << " json: " << json << "\n";
        expect(containsEvent(json, "fast_99")) << " json: " << json << "\n";
        std::filesystem::remove(output_file);
    };
};

const boost::ut::suite<"gr::profiling null profiler"> _4 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "null profiler compiles and runs"_test = [] {
        null::Profiler prof;
        auto           handler = prof.forThisThread();

        expect(nothrow([&] { handler->instantEvent("null_instant"); }));
        expect(nothrow([&] { handler->counterEvent("null_counter", "cat"); }));
        expect(nothrow([&] {
            auto event = handler->startCompleteEvent("null_complete");
            event.finish();
        }));
        expect(nothrow([&] {
            auto event = handler->startAsyncEvent("null_async");
            event.step();
            event.finish();
        }));
    };

    "null profiler reset"_test = [] {
        null::Profiler prof;
        expect(nothrow([&] { prof.reset(); }));
    };
};

const boost::ut::suite<"gr::profiling JSON format"> _5 = [] {
    using namespace boost::ut;
    using namespace gr::profiling;

    "valid JSON structure"_test = [] {
        const std::string output_file = "test_json_format.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("json_test", "cat", {{"str_arg", std::string{"hello"}}, {"int_arg", 42}, {"dbl_arg", 3.14}});
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json)) << " json: " << json << "\n";
        expect(json.find("\"str_arg\":\"hello\"") != std::string::npos) << " json: " << json << "\n";
        expect(json.find("\"int_arg\":42") != std::string::npos) << " json: " << json << "\n";
        expect(json.find("\"dbl_arg\":3.14") != std::string::npos) << " json: " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "pid and tid present"_test = [] {
        const std::string output_file = "test_pid_tid.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("pid_tid_test");
        }
        const auto json = readFile(output_file);
        expect(json.find("\"pid\":") != std::string::npos) << " json: " << json << "\n";
        expect(json.find("\"tid\":") != std::string::npos) << " json: " << json << "\n";
        std::filesystem::remove(output_file);
    };

    "timestamp ordering"_test = [] {
        const std::string output_file = "test_ts_order.trace";
        {
            Profiler prof(Options{.output_file = output_file, .update_period = std::chrono::milliseconds{10}});
            auto     handler = prof.forThisThread();
            handler->instantEvent("first");
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
            handler->instantEvent("second");
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
            handler->instantEvent("third");
        }
        const auto json = readFile(output_file);
        expect(isValidJson(json));

        const auto pos_first  = json.find("\"name\": \"first\"");
        const auto pos_second = json.find("\"name\": \"second\"");
        const auto pos_third  = json.find("\"name\": \"third\"");
        expect(lt(pos_first, pos_second)) << "first before second";
        expect(lt(pos_second, pos_third)) << "second before third";
        std::filesystem::remove(output_file);
    };
};

const boost::ut::suite<"gr::profiling concepts"> _6 = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::profiling;

    "ProfilerLike satisfied"_test = [] {
        expect(ProfilerLike<Profiler>);
        expect(ProfilerLike<null::Profiler>);
    };

    "SimpleEvent satisfied"_test = [] {
        expect(SimpleEvent<null::SimpleEvent>);
        expect(SimpleEvent<CompleteEvent<Handler<Profiler, decltype(std::declval<gr::CircularBuffer<gr::profiling::detail::TraceEvent, std::dynamic_extent, ProducerType::Multi>>().new_writer())>>>);
    };

    "StepEvent satisfied"_test = [] {
        expect(StepEvent<null::StepEvent>);
        expect(StepEvent<AsyncEvent<Handler<Profiler, decltype(std::declval<gr::CircularBuffer<gr::profiling::detail::TraceEvent, std::dynamic_extent, ProducerType::Multi>>().new_writer())>>>);
    };
};
#endif

int main() { /* tests are statically registered as suites */ }
