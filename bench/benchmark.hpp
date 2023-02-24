#ifndef GRAPH_PROTOTYPE_BENCHMARK_HPP
#define GRAPH_PROTOTYPE_BENCHMARK_HPP

#include <algorithm>
#include <boost/ut.hpp>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <fmt/color.h>
#include <fmt/format.h>

#if __has_include(<unistd.h>)  && __has_include(<sys/ioctl.h>)  && __has_include(<sys/syscall.h>) && __has_include(<linux/perf_event.h>) && !defined(BENCHMARK_NO_PERF_COUNTER)
#define HAS_LINUX_PERFORMANCE_HEADER
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace benchmark {
#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__ __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

#if defined(__x86_64__) or defined(__i686__)
#define SIMD_REG "x,"
#else
#define SIMD_REG
#endif

/**
 * Tell the compiler that all arguments to this function are read and modified in the most
 * efficient way possible. This may force a value to memory, but generally tries to avoid doing so.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void
fake_modify(T &x, Ts &...more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : "+" SIMD_REG "g,m"(x));
    } else {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : "+g,m"(x));
    }
#else
    const volatile T y = x;
    x = y;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        fake_modify(more...);
    }
}

/**
 * Tell the compiler that all arguments to this function are read in the most efficient way
 * possible. This may force a value to memory, but generally tries to avoid doing so.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void
fake_read(const T &x, const Ts &...more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" ::SIMD_REG "g,m"(x));
    } else {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" ::"g,m"(x));
    }
#else
    const volatile T y = x;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        fake_read(more...);
    }
}

/**
 * Tell the compiler that all arguments to this function must be stored to memory (stack) and
 * reloaded before reading.
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void
force_to_memory(T &x, Ts &...more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    // NOLINTNEXTLINE(hicpp-no-assembler)
    asm volatile("" : "+m"(x));
#else
    const volatile T y = x;
    x = y;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        force_to_memory(more...);
    }
}

/**
 * Tell the compiler that all arguments to this function must be stored to memory (stack).
 */
template<typename T, typename... Ts>
BENCHMARK_ALWAYS_INLINE void
force_store(const T &x, const Ts &...more) {
#ifdef __GNUC__
    // GNU compatible compilers need to support this part
    // NOLINTNEXTLINE(hicpp-no-assembler)
    asm volatile("" ::"m"(x));
#else
    const volatile T y = x;
#endif
    if constexpr (sizeof...(Ts) > 0) {
        force_store(more...);
    }
}

#undef SIMD_REG

    struct perf_sub_metric {
        uint64_t misses{0};
        uint64_t total{0};
        double ratio{0.0};
    };

    struct perf_metric {
        perf_sub_metric cache;
        perf_sub_metric branch;
        uint64_t instructions{0};
        uint64_t ctx_switches{0};
    };

#ifdef HAS_LINUX_PERFORMANCE_HEADER
    /**
     * A short and sweet performance counter (only works on Linux)
     */
    class PerformanceCounter {
        static bool                       _has_required_rights;
        int                               _fd_misses;
        int                               _fd_accesses;
        int                               _fd_branch_misses;
        int                               _fd_branch;
        int                               _fd_instructions;
        int                               _fd_ctx_switches;
        constexpr static std::string_view _sys_error_message =
                R"(You may not have permission to collect perf stats data.
Consider tweaking /proc/sys/kernel/perf_event_paranoid:
 -1 - Not paranoid at all
  0 - Disallow raw tracepoint access for unpriv
  1 - Disallow cpu events for unpriv
  2 - Disallow kernel profiling for unpriv
quick_fix: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
for details see: https://www.kernel.org/doc/Documentation/sysctl/kernel.txt)";

        static void
        print_access_right_msg(std::string_view msg) noexcept {
            fmt::print(stderr, "PerformanceCounter: {} - error {}: '{}'", msg, errno, strerror(errno));
            _has_required_rights = false;
            std::cerr << std::endl;
            fmt::print(_sys_error_message);
            std::cout << std::endl;
        }

    public:
        PerformanceCounter() {
            constexpr static int PROCESS = 0; // 0: calling process
            constexpr static int ANY_CPU = -1;
            constexpr static int FLAGS   = PERF_FLAG_FD_CLOEXEC;
            if (!_has_required_rights) {
                return;
            }

            perf_event_attr attr{};
            attr.type       = PERF_TYPE_HARDWARE;
            attr.disabled   = 1;
            attr.exclude_hv = 1;

            // cache prediction metric
            attr.config = PERF_COUNT_HW_CACHE_MISSES;
            _fd_misses  = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- misses");
                return;
            }

            attr.config  = PERF_COUNT_HW_CACHE_REFERENCES;
            _fd_accesses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_accesses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- accesses");
                return;
            }

            // branch prediction metric
            attr.config       = PERF_COUNT_HW_BRANCH_MISSES;
            _fd_branch_misses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_branch_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch misses");
                return;
            }

            attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
            _fd_branch  = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_branch == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch accesses");
                return;
            }

            // instruction count metric
            attr.config      = PERF_COUNT_HW_INSTRUCTIONS;
            _fd_instructions = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_instructions == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- instruction count");
                return;
            }

            // ctx switch count metric
            attr.type        = PERF_TYPE_SOFTWARE;
            attr.config      = PERF_COUNT_SW_CONTEXT_SWITCHES;
            _fd_ctx_switches = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_ctx_switches == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- ctx switches");
                return;
            }

            if (ioctl(_fd_misses, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_accesses, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_branch_misses, PERF_EVENT_IOC_ENABLE) == -1
                || ioctl(_fd_branch, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_instructions, PERF_EVENT_IOC_ENABLE) == -1 || ioctl(_fd_ctx_switches, PERF_EVENT_IOC_ENABLE) == -1) {
                print_access_right_msg("could not PERF_EVENT_IOC_ENABLE");
                return;
            }
        }

        ~PerformanceCounter() {
            if (_has_required_rights
                && (ioctl(_fd_misses, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_accesses, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_branch_misses, PERF_EVENT_IOC_DISABLE) == -1
                    || ioctl(_fd_branch, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_instructions, PERF_EVENT_IOC_DISABLE) == -1 || ioctl(_fd_ctx_switches, PERF_EVENT_IOC_DISABLE) == -1)) {
                print_access_right_msg("could not PERF_EVENT_IOC_DISABLE");
            }
            close(_fd_misses);
            close(_fd_accesses);
            close(_fd_branch_misses);
            close(_fd_branch);
            close(_fd_instructions);
            close(_fd_ctx_switches);
        }

        PerformanceCounter(const PerformanceCounter &) = delete;
        auto &
        operator=(const PerformanceCounter &)
                = delete;

        [[nodiscard]] static bool
        available() noexcept {
            return _has_required_rights;
        }

        /**
         * @return Linux HW/CPU performance counter, best consumed as:
         * @code auto [misses, accesses, branch_misses, branch_total, instructions, ctx_switches] =
         * execMetrics.results();
         */
        [[nodiscard]] auto
        results() const noexcept -> perf_metric {
            if (!_has_required_rights) {
                return {};
            }
            perf_metric           ret;
            constexpr static auto read_metric = [](int metric_fd, auto &data) noexcept -> bool { return read(metric_fd, &data, sizeof(data)) != sizeof(data); };
            if (read_metric(_fd_misses, ret.cache.misses) || read_metric(_fd_accesses, ret.cache.total) || read_metric(_fd_branch_misses, ret.branch.misses) || read_metric(_fd_branch, ret.branch.total)
                || read_metric(_fd_instructions, ret.instructions) || read_metric(_fd_ctx_switches, ret.ctx_switches)) {
                return {};
            }
            using T          = decltype(ret.cache.ratio);
            ret.cache.ratio  = static_cast<T>(ret.cache.misses) / static_cast<T>(ret.cache.total);
            ret.branch.ratio = static_cast<T>(ret.branch.misses) / static_cast<T>(ret.branch.total);
            return ret;
        }
    };

    inline bool PerformanceCounter::_has_required_rights = true;
#else

    class PerformanceCounter {
    public:
        [[nodiscard]] constexpr static bool
        available() noexcept {
            return false;
        }

        /**
         * This OS is not supported
         */
        [[nodiscard]] auto
        results() const noexcept -> perf_metric {
            return {};
        }
    };

#endif

    namespace ut = boost::ut;

/**
 * little compile-time string class (N.B. ideally std::string should become constexpr (everything!!
 * ;-)))
 */
    template<typename CharT, std::size_t SIZE>
    struct fixed_string {
        constexpr static std::size_t N = SIZE;
        CharT _data[N + 1] = {};

        constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
            if constexpr (N != 0)
                for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
        }

        [[nodiscard]] constexpr std::size_t
        size() const noexcept {
            return N;
        }

        [[nodiscard]] constexpr bool
        empty() const noexcept {
            return N == 0;
        }

        [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }

        [[nodiscard]] explicit           operator std::string() const noexcept { return {_data, N}; }

        [[nodiscard]]                    operator const char *() const noexcept { return _data; }

        [[nodiscard]] constexpr bool
        operator==(const fixed_string &other) const noexcept {
            return std::string_view{_data, N} == std::string_view(other);
        }

        template<std::size_t N2>
        [[nodiscard]] friend constexpr bool
        operator==(const fixed_string &, const fixed_string<CharT, N2> &) {
            return false;
        }
    };

    template<typename CharT, std::size_t N>
    fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

    template<fixed_string... key>
    constexpr bool key_not_found = false;

/**
 * constexpr const key map that allows modification of the values during run-time while the
 * compile-time look up of keys is free <-> similar to using structs and refl-cpp-based compile-time
 * reflection
 *
 * @author: Ralph J. Steinhagen
 */
    template<typename Value, fixed_string... Keys>
    class const_key_map {
        constexpr static std::size_t SIZE = sizeof...(Keys);
        constexpr static std::array<const std::string_view, SIZE> _keys = {std::string_view(Keys)...};
        std::array<Value, SIZE> _storage;

        template<fixed_string key>
        constexpr static std::size_t
        get_index_by_name() noexcept {
            if constexpr (constexpr auto itr = std::find_if(_keys.cbegin(), _keys.cend(),
                                                            [](auto const &v) { return v == std::string_view(key); });
                    itr != std::cend(_keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                static_assert(key_not_found<key>, "key not found");
            }
        }

        constexpr static std::size_t
        get_index_by_name(std::string_view key) {
            if (const auto itr = std::find_if(_keys.cbegin(), _keys.cend(),
                                              [&key](const auto &v) { return v == std::string_view(key); }); itr !=
                                                                                                             std::cend(
                                                                                                                     _keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                throw std::range_error("key not found");
            }
        }

        constexpr static std::size_t
        get_index_by_ID(const std::size_t key_ID) {
            if (key_ID < SIZE) {
                return key_ID;
            } else {
                throw std::range_error("key ID not found");
            }
        }

    public:
        [[nodiscard]] static constexpr std::size_t
        size() noexcept {
            return SIZE;
        }

        [[nodiscard]] std::string_view
        key(std::size_t key_ID) const {
            return _keys[get_index_by_ID(key_ID)];
        }

        template<fixed_string key>
        [[nodiscard]] constexpr Value const &
        at() const {
            return _storage[get_index_by_name<key>()];
        }

        template<fixed_string key>
        [[nodiscard]] constexpr Value &
        at() {
            return _storage[get_index_by_name<key>()];
        }

        [[nodiscard]] constexpr Value const &
        at(const std::string_view key) const {
            return _storage[get_index_by_name(key)];
        }

        [[nodiscard]] constexpr Value &
        at(const std::string_view key) {
            return _storage[get_index_by_name(key)];
        }

        [[nodiscard]] constexpr Value const &
        at(const std::size_t key_ID) const {
            return _storage[get_index_by_ID(key_ID)];
        }

        [[nodiscard]] constexpr Value &
        at(const std::size_t key_ID) {
            return _storage[get_index_by_ID(key_ID)];
        }

        [[nodiscard]] constexpr bool
        contains(const std::string_view key) const {
            return std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto &v) { return v == key; }) !=
                   std::cend(_keys);
        }
    };

    template<std::floating_point T>
    struct StatisticsType {
        T min{std::numeric_limits<T>::quiet_NaN()};
        T mean{std::numeric_limits<T>::quiet_NaN()};
        T stddev{std::numeric_limits<T>::quiet_NaN()};
        T median{std::numeric_limits<T>::quiet_NaN()};
        T max{std::numeric_limits<T>::quiet_NaN()};
    };

    struct StringHash {
        using is_transparent = void; // enables heterogeneous lookup

        std::size_t
        operator()(std::string_view sv) const {
            std::hash<std::string_view> hasher;
            return hasher(sv);
        }
    };

    using ResultMap = std::unordered_map<std::string, std::tuple<std::variant<std::monostate, long double, uint64_t, perf_sub_metric>, std::string, std::size_t>, StringHash, std::equal_to<>>;

    class results {
        using EntryType = std::pair<std::string, ResultMap>;
        using Data = std::vector<EntryType>;
        static std::mutex _lock;
        static Data _data;

    public:
        [[nodiscard]] static ResultMap &
        add_result(std::string_view name) noexcept {
            std::lock_guard guard(_lock);
            return _data.emplace_back(std::string(name), ResultMap()).second;
        }

        [[nodiscard]] static constexpr Data const &
        data() noexcept {
            return _data;
        };
    };

    inline results::Data results::_data;
    inline std::mutex    results::_lock;

    class time_point {
        using timePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using timeDiff = std::chrono::nanoseconds;
        timePoint _time_point;

    public:
        time_point &
        now() noexcept {
            _time_point = std::chrono::high_resolution_clock::now();
            return *this;
        }

        timeDiff
        operator-(const time_point &start_marker) const noexcept {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(_time_point - start_marker._time_point);
        }
    };

    namespace utils {

        template<std::size_t N, typename T>
        requires (N > 0)
        constexpr std::vector<T>
        diff(const std::vector<time_point> &stop, time_point start) {
            std::vector<T> ret(N);
            for (auto i = 0LU; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start).count());
                start = stop[i];
            }
            return ret;
        }

        template<std::size_t N, typename T>
        requires (N > 0)
        constexpr std::vector<T>
        diff(const std::vector<time_point> &stop, const std::vector<time_point> &start) {
            std::vector<T> ret(N);
            for (auto i = 0LU; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start[i]).count());
            }
            return ret;
        }

        template<std::size_t n_iterations, typename MapType>
        requires (n_iterations > 0)
        auto
        convert(const std::vector<MapType> &in) {
            std::vector<std::pair<std::string, std::vector<time_point>>> ret;
            ret.resize(in[0].size());

            for (auto keyID = 0LU; keyID < in[0].size(); keyID++) {
                ret[keyID].first = std::string(in[0].key(keyID));
                ret[keyID].second.resize(n_iterations);
                for (auto i = 0LU; i < n_iterations; i++) {
                    ret[keyID].second[i] = in[i].at(keyID);
                }
            }
            return ret;
        }

        template<typename T>
        [[nodiscard]] StatisticsType<T>
        compute_statistics(const std::vector<T> &values) {
            const std::size_t N = values.size();
            if (N < 1) {
                return {};
            }
            const auto minmax = std::minmax_element(values.cbegin(), values.cend());
            const auto mean = std::accumulate(values.begin(), values.end(), T{}) / static_cast<T>(N);

            T stddev{};
            std::for_each(values.cbegin(), values.cend(), [&](const auto x) { stddev += (x - mean) * (x - mean); });
            stddev /= static_cast<T>(N);
            stddev = std::sqrt(stddev);

            // Compute the median value
            std::vector<T> sorted_values(values);
            std::sort(sorted_values.begin(), sorted_values.end());
            const auto median = sorted_values[N / 2];
            return {*minmax.first, mean, stddev, median, *minmax.second};
        }

        template<typename T>
        concept Numeric = std::integral<T> || std::floating_point<T>;

        template<Numeric T>
        std::string
        to_si_prefix(T value_base, std::string_view unit = "s", std::size_t significant_digits = 0) {
            static constexpr std::array si_prefixes{'q', 'r', 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M',
                                                    'G', 'T', 'P', 'E', 'Z', 'Y', 'R', 'Q'};
            static constexpr double base = 1000.0;
            long double value = value_base;

            std::size_t exponent = 10;
            if (value == 0) {
                return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ",
                                   si_prefixes[exponent], unit);
            }
            while (value >= base && exponent < si_prefixes.size()) {
                value /= base;
                ++exponent;
            }
            while (value < 1.0 && exponent > 0) {
                value *= base;
                --exponent;
            }

            return fmt::format("{:.{}f}{}{}{}", value, significant_digits, unit.empty() ? "" : " ",
                               si_prefixes[exponent], unit);
        }

    } // namespace utils

    namespace detail {

        template<typename T>
        struct class_with_call_operator;

// For const operator()
        template<typename ClassType, typename ResultType, typename... Args>
        struct class_with_call_operator<ResultType (ClassType::*)(Args...) const> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

// For non-const operator()
        template<typename ClassType, typename ResultType, typename... Args>
        struct class_with_call_operator<ResultType (ClassType::*)(Args...)> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename T>
        struct fn_traits;

        template<typename ClassType, typename ResultType, typename... Args>
        struct fn_traits<ResultType (ClassType::*)(Args...) const> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename ResultType, typename... Args>
        struct fn_traits<ResultType(Args...)> {
            constexpr static std::size_t arity = sizeof...(Args);
            using result_type = ResultType;
            using args_tuple = std::tuple<Args...>;
        };

        template<typename T>
        struct fn_traits : detail::class_with_call_operator<decltype(&T::operator())> {
        };

        template<typename T>
        using return_type_of_t = typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::result_type;

        template<typename T>
        using first_arg_of_t = std::tuple_element_t<0, typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::args_tuple>;
    } // namespace detail

    template<typename TestFunction>
    constexpr std::size_t
    argument_size() noexcept {
        if constexpr (std::invocable<TestFunction>) {
            return 0;
        } else {
            using arg_t = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
            if constexpr (requires { arg_t::size(); }) {
                return arg_t::size();
            } else {
                return 0;
            }
        }
    }

    template<fixed_string... meas_marker_names>
    struct MarkerMap : const_key_map<time_point, meas_marker_names...> {
    };

    template<typename TestFunction, std::size_t N>
    [[nodiscard]] constexpr auto
    get_marker_array() {
        if constexpr (std::invocable<TestFunction>) {
            return std::vector<bool>(N);
        } else {
            using MarkerMapType = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
            return std::vector<MarkerMapType>(N);
        }
    }

    template<std::size_t N_ITERATIONS = 1LU, typename ResultType = results, fixed_string... meas_marker_names>
    class benchmark : public ut::detail::test {
        std::size_t _n_scale_results;
    public:
        benchmark() = delete;

        explicit benchmark(std::string_view _name, std::size_t n_scale_results = 1LU) : ut::detail::test{"benchmark",
                                                                                                         _name},
                                                                                        _n_scale_results(
                                                                                                n_scale_results) {}

        template<class TestFunction, std::size_t MARKER_SIZE = argument_size<TestFunction>(), bool has_arguments =
        MARKER_SIZE != 0>
        // template<fixed_string ...meas_marker, Callback<meas_marker...> Test>
        constexpr benchmark &
        operator=(TestFunction &&_test) {
            static_cast<ut::detail::test &>(*this) = [&_test, this] {
                auto &result_map = ResultType::add_result(name);
                if constexpr (N_ITERATIONS != 1) {
                    result_map.try_emplace("#N", N_ITERATIONS, "", 0);
                } else {
                    result_map.try_emplace("#N", std::monostate{}, "", 0);
                }

                std::vector<time_point> stop_iter(N_ITERATIONS);
                auto marker_iter = get_marker_array<TestFunction, N_ITERATIONS>();

                PerformanceCounter execMetrics;
                const auto start = time_point().now();

                if constexpr (N_ITERATIONS == 1) {
                    if constexpr (std::invocable<TestFunction>) {
                        _test();
                    } else {
                        _test(marker_iter[0]);
                    }
                    stop_iter[0].now();
                } else if constexpr (N_ITERATIONS >= 1) {
                    for (auto i = 0LU; i < N_ITERATIONS; i++) {
                        if constexpr (std::invocable<TestFunction>) {
                            _test();
                        } else {
                            _test(marker_iter[i]);
                        }
                        stop_iter[i].now();
                    }
                } else {
                    throw std::invalid_argument("benchmark n_iteration := 0 parameter not (yet) implemented");
                }
                // N.B. need to retrieve CPU performance count here no to spoil the result by further
                // post-processing

                if (PerformanceCounter::available()) {
                    const perf_metric perf_data = execMetrics.results();
                    result_map.try_emplace("CPU cache misses", perf_data.cache, "", 0);
                    result_map.try_emplace("CPU branch misses", perf_data.branch, "", 0);
                    result_map.try_emplace("<CPU-I>", static_cast<double>(perf_data.instructions) /
                                                      static_cast<double>(N_ITERATIONS * _n_scale_results), "", 1);
                    result_map.try_emplace("CTX-SW", perf_data.ctx_switches, "", 0);
                }
                // not time-critical post-processing starts here
                const auto time_differences_ns = utils::diff<N_ITERATIONS, long double>(stop_iter, start);
                const auto ns = stop_iter[N_ITERATIONS - 1] - start;
                const long double duration_s = 1e-9 * static_cast<long double>(ns.count());

                const auto add_statistics = [&duration_s]<typename T>(ResultMap &map, const T &time_diff) {
                    if constexpr (N_ITERATIONS != 1) {
                        const auto [min, mean, stddev, median, max] = utils::compute_statistics(time_diff);
                        map.try_emplace("min", min, "s", 0);
                        map.try_emplace("mean", mean, "s", 0);
                        if (stddev == 0) {
                            map.try_emplace("stddev", std::monostate{}, "s", 0);
                        } else {
                            map.try_emplace("stddev", stddev, "s", 0);
                        }
                        map.try_emplace("median", median, "s", 0);
                        map.try_emplace("max", max, "s", 0);
                    } else {
                        map.try_emplace("min", std::monostate{}, "s", 0);
                        map.try_emplace("mean", duration_s / N_ITERATIONS, "s", 0);
                        map.try_emplace("stddev", std::monostate{}, "s", 0);
                        map.try_emplace("median", std::monostate{}, "s", 0);
                        map.try_emplace("max", std::monostate{}, "s", 0);
                    }
                };
                add_statistics(result_map, time_differences_ns);

                result_map.try_emplace("total time", duration_s, "s", 0);
                result_map.try_emplace("ops/s", _n_scale_results * N_ITERATIONS / duration_s, "", 1);

                if constexpr (MARKER_SIZE > 0) {
                    auto transposed_map = utils::convert<N_ITERATIONS>(marker_iter);
                    for (auto keyID = 0LU; keyID < transposed_map.size(); keyID++) {
                        if (keyID > 0) {
                            const auto meas = fmt::format("  {}─Marker{}: '{}'→'{}' ", //
                                                          keyID < transposed_map.size() - 1 ? "├" : "└", keyID,
                                                          transposed_map[0].first, transposed_map[keyID].first);

                            auto &marker_result_map = ResultType::add_result(meas);
                            add_statistics(marker_result_map,
                                           utils::diff<N_ITERATIONS, long double>(transposed_map[keyID].second,
                                                                                  transposed_map[0].second));
                        }
                    }
                }
            };
            return *this;
        }

        template<std::size_t N>
        auto
        repeat(std::size_t n_scale_results = 1LU) {
            return ::benchmark::benchmark<N, ResultType, meas_marker_names...>(this->name, n_scale_results);
        }
    };

    [[nodiscard]] auto operator ""_benchmark(const char *name, std::size_t size) {
        return ::benchmark::benchmark<1LU>{{name, size}};
    }

} // namespace benchmark

namespace cfg {
    namespace ut = boost::ut;

    template<class TPrinter = ut::printer>
    class reporter {
        struct {
            std::size_t fail{};
        } _benchmarks{};

        struct {
            std::size_t pass{};
            std::size_t fail{};
        } _asserts{};

        TPrinter _printer{};

    public:
        constexpr reporter &
        operator=(TPrinter printer) {
            _printer = std::move(printer);
            return *this;
        }

        constexpr void
        on(const ut::events::test_begin &) const noexcept { /* not needed */
        }

        void
        on(const ut::events::test_run &test_run) {
            _printer << "\n \"" << test_run.name << "\"...";
        }

        constexpr void
        on(const ut::events::test_skip &bench) const noexcept {
            std::cerr << fmt::format("SKIPPED - {}", bench.name) << std::endl;
            [[maybe_unused]] const auto &map = benchmark::results::add_result(bench.name);
        }

        void
        on(const ut::events::test_end &test_end) {
            if (_asserts.fail > 0) {
                ++_benchmarks.fail;
                _printer << _printer.colors().fail << fmt::format("... in benchmark '{}'", test_end.name)
                         << _printer.colors().none << '\n';
                _asserts.fail--;
            }
        }

        template<class TMsg>
        void
        on(ut::events::log <TMsg> l) {
            _printer << l.msg;
        }

        void
        on(ut::events::exception exception) {
            _printer << fmt::format("\033[31munexpected exception: \"{}\"\n\033[0m", exception.what());
            ++_asserts.fail;
        }

        template<class TExpr>
        void
        on(ut::events::assertion_pass <TExpr>) {
            ++_asserts.pass;
        }

        template<class TExpr>
        void
        on(ut::events::assertion_fail <TExpr> assertion) {
            constexpr auto short_name = [](std::string_view name) {
                return name.rfind('/') != std::string_view::npos ? name.substr(name.rfind('/') + 1) : name;
            };
            _printer << "\n  " << short_name(assertion.location.file_name()) << ':' << assertion.location.line() << ':'
                     << _printer.colors().fail << "FAILED" << _printer.colors().none << " ["
                     << std::boolalpha << assertion.expr << _printer.colors().none << ']';
            ++_asserts.fail;
        }

        void
        on(const ut::events::fatal_assertion &) const { /* not needed testing interface */
        }

        void
        on(const ut::events::summary &) {
            if (_benchmarks.fail || _asserts.fail) {
                std::cout << _printer.str() << std::endl;
                std::cout << fmt::format("\033[31m{} micro-benchmark(s) failed:\n\033[m", _benchmarks.fail);
            } else {
                std::cout << _printer.colors().pass << "all micro-benchmarks passed:\n" << _printer.colors().none;
            }
            print();
            std::cerr.flush();
            std::cout.flush();
        }

        template<std::size_t SIGNIFICANT_DIGITS = 3>
        static void
        print() {
            const auto &data = benchmark::results::data();
            if (data.empty()) {
                fmt::print("no benchmark tests executed\n");
            }
            std::vector<std::size_t> v(data.size());
            // N.B. using <algorithm> rather than <ranges> to be compatible with libc/Emscripten
            // for details see: https://libcxx.llvm.org/Status/Ranges.html
            // not as of clang 15: https://compiler-explorer.com/z/8arxzodh3 ('trunk' seems to be OK)
            std::transform(data.cbegin(), data.cend(), v.begin(), [](auto val) { return val.first.size(); });

            const std::string test_case_label = "benchmark:";
            const std::size_t name_max_size =
                    std::max(*std::max_element(v.cbegin(), v.cend()), test_case_label.size()) + 1LU;

            const auto format = [](auto &value, const std::string &unit, std::size_t digits) -> std::string {
                using ::benchmark::utils::to_si_prefix;
                if (std::holds_alternative<std::monostate>(value)) {
                    return "";
                } else if (std::holds_alternative<long double>(value)) {
                    return fmt::format("{}", to_si_prefix(std::get<long double>(value), unit, digits));
                } else if (std::holds_alternative<uint64_t>(value)) {
                    return fmt::format("{}", to_si_prefix(std::get<uint64_t>(value), unit, digits));
                } else if (std::holds_alternative<benchmark::perf_sub_metric>(value)) {
                    const auto stat = std::get<benchmark::perf_sub_metric>(value);
                    return fmt::format("{:>4} / {:>4} = {:4.1f}%", //
                                       to_si_prefix(stat.misses, unit, 0), to_si_prefix(stat.total, unit, 0),
                                       100.0 * stat.ratio);
                }
                throw std::invalid_argument("benchmark::results: unhandled ResultMap type");
            };

            // compute minimum colum width for each benchmark case and metric
            std::unordered_map<std::string, std::size_t> metric_keys;
            for (auto &[test_name, result_map]: data) {
                for (auto &[metric_key, value]: result_map) {
                    const auto &[value_name, value_unit, value_digits] = value;
                    if (!metric_keys.contains(metric_key)) {
                        metric_keys.try_emplace(metric_key, metric_key.size());
                    }
                    std::size_t value_size = format(value_name, value_unit, value_digits).size();
                    metric_keys.at(metric_key) = std::max(metric_keys.at(metric_key), value_size);
                }
            }
            v.clear();

            bool first_row = true;
            for (auto &[test_name, result_map]: data) {
                if (first_row) {
                    fmt::print("┌{1:─^{0}}", name_max_size + 2UL, test_case_label);
                    fmt::print("┬{1:─^{0}}", sizeof("PASS") + 1UL, "");
                    for (auto const &[metric_key, max_width]: metric_keys) {
                        fmt::print("┬{1:─^{0}}", max_width + 2UL, metric_key);
                    }
                    fmt::print("┐\n");
                    first_row = false;
                }
                fmt::print("│ {1:<{0}} ", name_max_size, test_name);
                if (result_map.size() == 0) {
                    fmt::print("│ \033[33mSKIP\033[0m ");
                } else if (result_map.size() == 1) {
                    fmt::print("│ \033[31mFAIL\033[0m ");
                } else {
                    fmt::print("│ \033[32mPASS\033[0m ");
                }

                for (auto &[metric_key, max_width]: metric_keys) {
                    if (result_map.contains(metric_key)) {
                        const auto &[value, unit, digits] = result_map.at(metric_key);
                        fmt::print("│ {1:>{0}} ", max_width, format(value, unit, digits));
                    } else {
                        fmt::print("│ {1:>{0}} ", max_width, "");
                    }
                }
                fmt::print("│\n");
            }

            fmt::print("└{1:─^{0}}", name_max_size + 2UL, "");
            fmt::print("┴{1:─^{0}}", sizeof("PASS") + 1UL, "");
            for (auto const &[metric_key, max_width]: metric_keys) {
                fmt::print("┴{1:─^{0}}", max_width + 2UL, "");
            }
            fmt::print("┘\n");
        }
    };
} // namespace cfg

template<>
inline auto boost::ut::cfg<boost::ut::override> = ut::runner < cfg::reporter<printer>>
{
};

#endif // GRAPH_PROTOTYPE_BENCHMARK_HPP
