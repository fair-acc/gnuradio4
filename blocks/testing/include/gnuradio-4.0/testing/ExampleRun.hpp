#ifndef GNURADIO_TESTING_EXAMPLE_RUN_HPP
#define GNURADIO_TESTING_EXAMPLE_RUN_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <csignal>
#include <cstddef>
#include <expected>
#include <format>
#include <future>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/Logger.hpp>
#include <gnuradio-4.0/algorithm/ArrayGeometry.hpp>
#include <gnuradio-4.0/device/SyclRuntime.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr::testing {

inline constexpr double      kExampleSceneSeconds = 10.0;
inline constexpr double      kExampleSampleRateHz = 48000.0;
inline constexpr double      kSpeedOfSoundInAir   = 343.0;
inline constexpr std::size_t kMaxExampleElements  = 32UZ;

inline constexpr std::size_t kMaxCorrelationFrameSamples = 8192UZ;

/** not required (no trailing '!'): a build or machine without a GPU falls back to the host with a framework warning */
inline constexpr std::string_view kExampleDeviceDomain = "gpu:sycl";

/** the compute domain of an example's array blocks, named on stdout: the GPU, or the host with --nogpu */
inline std::string_view chooseExampleDomain(bool noGpu) {
    const std::string_view domain = noGpu ? gr::thread_pool::kDefaultIoPoolId : kExampleDeviceDomain;
    if (!noGpu) {
        std::ignore = gr::device::registerSyclRuntime(); // without it no `gpu:sycl` context exists
    }
    std::println("compute domain of the array blocks: {}", domain);
    return domain;
}

struct ChannelArray {
    std::string_view layout; /**< a `gr::algorithm::geometry::layoutOf` name */
    std::size_t      elements;
    std::size_t      rows;        /**< a grid's; 1 otherwise */
    std::size_t      columns;     /**< a grid's; `elements` otherwise */
    double           spacing;     /**< m */
    std::string_view spacingRole; /**< what `spacing` measures: "apart", "edge" or "pitch" */
};

[[nodiscard]] constexpr ChannelArray channelArrayOf(std::size_t channels) noexcept {
    std::size_t side = 0UZ;
    while ((side + 1UZ) * (side + 1UZ) <= channels) {
        ++side;
    }
    if (channels == 4UZ) {
        return {.layout = "tetrahedron", .elements = 4UZ, .rows = 1UZ, .columns = 4UZ, .spacing = gr::algorithm::geometry::kTetrahedronEdge, .spacingRole = "edge"};
    }
    if (channels == 16UZ) { // the 16-channel array at hand is a UMA-16: its USB channel order is not the grid's row-major one
        return {.layout = "uma16", .elements = 16UZ, .rows = 4UZ, .columns = 4UZ, .spacing = gr::algorithm::geometry::kGridSpacing, .spacingRole = "pitch"};
    }
    if (side >= 3UZ && side * side == channels) {
        return {.layout = "grid", .elements = channels, .rows = side, .columns = side, .spacing = gr::algorithm::geometry::kGridSpacing, .spacingRole = "pitch"};
    }
    return {.layout = "line", .elements = channels, .rows = 1UZ, .columns = channels, .spacing = gr::algorithm::geometry::lineSpacingOf(channels), .spacingRole = "apart"};
}

/** the correlation frame holding lags up to maxLagSamples: the smallest power of two from 1024 whose half exceeds the lag, so every pair's lag stays inside the circular correlation */
[[nodiscard]] constexpr std::size_t correlationFrameSamples(std::size_t maxLagSamples) noexcept { return std::max(1024UZ, std::bit_ceil(2UZ * maxLagSamples + 1UZ)); }

/**
 * the command line of an example, read flag by flag: `flag`, `text` and `number` each consume the current argument when it is theirs;
 * `valid` turns false on a missing or malformed value, or when the caller refuses an argument nobody consumed
 *
 *   for (gr::testing::ExampleArguments args(argc, argv); args.next();) { if (!(args.flag("--synthetic", synthetic) || args.number("--seconds", seconds))) { args.refuse(); } }
 */
struct ExampleArguments {
    std::span<char*> arguments;
    std::size_t      index = 0UZ;
    bool             valid = true;

    ExampleArguments(int argc, char** argv) : arguments(argv, static_cast<std::size_t>(argc)) {}

    [[nodiscard]] bool next() noexcept { return ++index < arguments.size(); }

    [[nodiscard]] std::string_view current() const noexcept { return arguments[index]; }

    void refuse() noexcept { valid = false; }

    [[nodiscard]] bool flag(std::string_view name, bool& value) noexcept {
        if (current() != name) {
            return false;
        }
        value = true;
        return true;
    }

    [[nodiscard]] bool text(std::string_view name, std::string& value) {
        if (current() != name) {
            return false;
        }
        value = std::string(valueOf());
        return true;
    }

    /** as above; the optional tells a given empty value from an absent flag */
    [[nodiscard]] bool text(std::string_view name, std::optional<std::string>& value) {
        if (current() != name) {
            return false;
        }
        value = std::string(valueOf());
        return true;
    }

    template<typename T>
    [[nodiscard]] bool number(std::string_view name, T& value) noexcept {
        if (current() != name) {
            return false;
        }
        const std::string_view text     = valueOf();
        const auto             parsed   = std::from_chars(text.data(), text.data() + text.size(), value);
        const bool             complete = parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
        valid                           = valid && complete && std::isfinite(static_cast<double>(value));
        return true;
    }

    [[nodiscard]] std::string_view valueOf() noexcept {
        if (index + 1UZ >= arguments.size()) {
            valid = false;
            return {};
        }
        return arguments[++index];
    }
};

struct CorrelationFrame {
    gr::Size_t maxLag;  /**< samples */
    gr::Size_t samples; /**< the frame holding every pair's lag */
};

/** the lag window and correlation frame of an array of `aperture` m at `rate`; an error when the aperture exceeds the largest frame */
[[nodiscard]] inline std::expected<CorrelationFrame, std::string> correlationFrameFor(double aperture, double speed, double rate) {
    const double lagSamples = std::ceil(aperture / speed * rate) + 2.0;
    const bool   fits       = lagSamples >= 2.0 && lagSamples < static_cast<double>(kMaxCorrelationFrameSamples / 2UZ);
    if (!fits) {
        return std::unexpected(std::format("array aperture exceeds the largest correlation frame of {} samples; reduce --spacing or the element count", kMaxCorrelationFrameSamples));
    }
    const auto maxLag = static_cast<gr::Size_t>(lagSamples);
    return CorrelationFrame{.maxLag = maxLag, .samples = static_cast<gr::Size_t>(correlationFrameSamples(maxLag))};
}

/** the lag axis in ms of a correlation curve of `length` lags at `rate`, lag 0 in the middle */
[[nodiscard]] inline std::vector<double> lagAxisMilliseconds(std::size_t length, double rate) {
    return std::views::iota(0UZ, length) | std::views::transform([=](std::size_t l) { return 1000.0 * (static_cast<double>(l) - static_cast<double>(length / 2UZ)) / rate; }) | std::ranges::to<std::vector<double>>();
}

[[nodiscard]] inline std::string liveInputDescription(std::string_view device) { return std::format("true audio, device {}", device.empty() ? std::string_view("default") : device); }

namespace detail {
inline std::atomic_flag exampleInterrupted{};
inline void             interruptExample(int) noexcept { exampleInterrupted.test_and_set(std::memory_order_relaxed); }
} // namespace detail

/** one active call per process; seconds is finite and non-negative, zero means unlimited. SIGINT/SIGTERM request a graceful stop; previous handlers are restored before returning the scheduler result. */
template<typename TScheduler>
[[nodiscard]] std::expected<void, gr::Error> runExample(TScheduler& scheduler, double seconds) {
    detail::exampleInterrupted.clear(std::memory_order_relaxed);
    const auto previousInterrupt = std::signal(SIGINT, detail::interruptExample);
    if (previousInterrupt == SIG_ERR) {
        return std::unexpected(gr::Error("cannot install the example interrupt handler"));
    }
    const gr::on_scope_exit restoreInterrupt([previousInterrupt] { std::signal(SIGINT, previousInterrupt); });
    const auto              previousTerminate = std::signal(SIGTERM, detail::interruptExample);
    if (previousTerminate == SIG_ERR) {
        return std::unexpected(gr::Error("cannot install the example termination handler"));
    }
    const gr::on_scope_exit restoreTerminate([previousTerminate] { std::signal(SIGTERM, previousTerminate); });
    if (seconds == 0.0) {
        std::println("run: until Ctrl+C (--seconds N sets a time limit)");
    } else {
        std::println("run: {:.3g} s; Ctrl+C stops early", seconds);
    }
    constexpr auto kStopPollPeriod = std::chrono::milliseconds(10);
    const auto     started         = std::chrono::steady_clock::now();
    auto           run             = std::async(std::launch::async, [&scheduler] { return scheduler.runAndWait(); });
    while (run.wait_for(kStopPollPeriod) != std::future_status::ready) {
        const double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const bool   expired        = seconds > 0.0 && elapsedSeconds >= seconds;
        if ((expired || detail::exampleInterrupted.test(std::memory_order_relaxed)) && scheduler.state() == gr::lifecycle::State::RUNNING) {
            scheduler.requestStop();
        }
    }
    return run.get();
}

/** a device fault on `kExampleDeviceDomain` rebuilds the graph on the host: edges between device blocks are device-only and cannot fall back inside a running graph */
template<typename TBuildGraph>
requires std::invocable<TBuildGraph&&, std::string_view>
[[nodiscard]] int runExample(TBuildGraph&& buildGraph, std::string_view domain, double seconds, auto& scheduler) {
    constexpr std::array<std::string_view, 2> kDeviceFaultPrefixes{"device fault", "device context poisoned"}; // `device::ExecutionStrategy`'s wordings; `gr::Error` has no structured field
    for (std::string_view runDomain = domain;; runDomain = gr::thread_pool::kDefaultIoPoolId) {
        auto graph = buildGraph(runDomain);
        if (!graph) {
            std::println(stderr, "{}", graph.error());
            return 1;
        }
        if (const auto exchanged = scheduler.exchange(std::move(*graph)); !exchanged) {
            std::println(stderr, "graph refused: {}", exchanged.error().message);
            return 1;
        }
        const auto result = runExample(scheduler, seconds);
        if (result) {
            return 0;
        }
        if (runDomain != kExampleDeviceDomain || std::ranges::none_of(kDeviceFaultPrefixes, [&](std::string_view prefix) { return result.error().message.starts_with(prefix); })) {
            std::println(stderr, "graph failed: {}", result.error().message);
            return 1;
        }
        std::println(stderr, "GPU run failed: {}; running on the host (use --nogpu to skip the GPU)", result.error().message);
    }
}

} // namespace gr::testing

#endif // GNURADIO_TESTING_EXAMPLE_RUN_HPP
