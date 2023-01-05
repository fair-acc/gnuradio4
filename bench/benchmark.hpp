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

#if __has_include(<unistd.h>)  && __has_include(<sys/ioctl.h>)  && __has_include(<sys/syscall.h>) && __has_include(<linux/perf_event.h>)
#define HAS_LINUX_PERFORMANCE_HEADER
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#endif

namespace benchmark {
#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__ __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

#if defined(__GNUC__) or defined(__clang__) and not defined(_LIBCPP_VERSION)

    template<typename T>
    BENCHMARK_ALWAYS_INLINE void do_not_optimize(T const &val) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : : "r,m"(val) : "memory");
    }

    template<typename T>
    BENCHMARK_ALWAYS_INLINE void do_not_optimize(T &val) {
#if defined(__clang__)
        // NOLINTNEXTLINE(hicpp-no-assembler)
    asm volatile("" : "+r,m"(val) : : "memory");
#else
        // NOLINTNEXTLINE(hicpp-no-assembler)
        asm volatile("" : "+m,r"(val) : : "memory");
#endif
    }


#else
#pragma optimize("", off)
    template <class T>
    void do_not_optimize(T&& t) {
      reinterpret_cast<char volatile&>(t) =
          reinterpret_cast<char const volatile&>(t);
    }
#pragma optimize("", on)
#endif

#ifdef HAS_LINUX_PERFORMANCE_HEADER
    /**
     * A short and sweet performance counter (only works on Linux)
     */
    class PerformanceCounter {
        static bool _has_required_rights;
        int _fd_misses;
        int _fd_accesses;
        int _fd_branch_misses;
        int _fd_branch;
        int _fd_instructions;
        constexpr static std::string_view _sys_error_message =
R"(You may not have permission to collect perf stats data.
Consider tweaking /proc/sys/kernel/perf_event_paranoid:
 -1 - Not paranoid at all
  0 - Disallow raw tracepoint access for unpriv
  1 - Disallow cpu events for unpriv
  2 - Disallow kernel profiling for unpriv
quick_fix: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
for details see: https://www.kernel.org/doc/Documentation/sysctl/kernel.txt)";
        static void print_access_right_msg(std::string_view msg) noexcept {
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
            constexpr static int FLAGS = PERF_FLAG_FD_CLOEXEC;
            if (!_has_required_rights) {
                return;
            }


            perf_event_attr attr{};
            attr.type = PERF_TYPE_HARDWARE;
            attr.disabled = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;

            // cache prediction metric
            attr.config = PERF_COUNT_HW_CACHE_MISSES;
            _fd_misses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- misses");
                return;
            }

            attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
            _fd_accesses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_accesses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- accesses");
                return;
            }

            // branch prediction metric
            attr.config = PERF_COUNT_HW_BRANCH_MISSES;
            _fd_branch_misses = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, -1, FLAGS));
            if (_fd_branch_misses == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch misses");
                return;
            }

            attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
            _fd_branch = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses, FLAGS));
            if (_fd_branch == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- branch accesses");
                return;
            }

            // instruction count metric
            attr.config = PERF_COUNT_HW_INSTRUCTIONS;
            _fd_instructions = static_cast<int>(syscall(SYS_perf_event_open, &attr, PROCESS, ANY_CPU, _fd_misses,
                                                        FLAGS));
            if (_fd_instructions == -1) {
                print_access_right_msg("could not open SYS_perf_event_open -- instruction count");
                return;
            }


            if (ioctl(_fd_misses, PERF_EVENT_IOC_ENABLE) == -1 ||
                ioctl(_fd_accesses, PERF_EVENT_IOC_ENABLE) == -1 ||
                ioctl(_fd_branch_misses, PERF_EVENT_IOC_ENABLE) == -1 ||
                ioctl(_fd_branch, PERF_EVENT_IOC_ENABLE) == -1 ||
                ioctl(_fd_instructions, PERF_EVENT_IOC_ENABLE) == -1) {
                print_access_right_msg("could not PERF_EVENT_IOC_ENABLE");
                return;
            }
        }

        ~PerformanceCounter() {
            if (_has_required_rights && (ioctl(_fd_misses, PERF_EVENT_IOC_DISABLE) == -1 ||
                ioctl(_fd_accesses, PERF_EVENT_IOC_DISABLE) == -1 ||
                ioctl(_fd_branch_misses, PERF_EVENT_IOC_DISABLE) == -1 ||
                ioctl(_fd_branch, PERF_EVENT_IOC_DISABLE) == -1 ||
                ioctl(_fd_instructions, PERF_EVENT_IOC_DISABLE) == -1)) {
                print_access_right_msg("could not PERF_EVENT_IOC_DISABLE");
            }
            close(_fd_misses);
            close(_fd_accesses);
            close(_fd_branch_misses);
            close(_fd_branch);
            close(_fd_instructions);
        }

        [[nodiscard]] static bool available() noexcept {
            return _has_required_rights;
        }

        /**
         * @return Linux HW/CPU performance counter, best consumed as:
         * @code auto [misses, accesses, branch_misses, branch_total, instructions] = execMetrics.results();
         */
        [[nodiscard]] auto results() const noexcept -> std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> {
            if (!_has_required_rights) {
                return {0U, 0U, 0U, 0U, 0U};
            }
            uint64_t misses;
            uint64_t accesses;
            uint64_t branch_misses;
            uint64_t branch_accesses;
            uint64_t instructions;
            if (read(_fd_misses, &misses, sizeof(misses)) != sizeof(misses) ||
                read(_fd_accesses, &accesses, sizeof(accesses)) != sizeof(accesses) ||
                read(_fd_branch_misses, &branch_misses, sizeof(branch_misses)) != sizeof(branch_misses) ||
                read(_fd_branch, &branch_accesses, sizeof(branch_accesses)) != sizeof(branch_accesses) ||
                read(_fd_instructions, &instructions, sizeof(instructions)) != sizeof(instructions)) {
                return {0U, 0U, 0U, 0U, 0U};
            }
            return {misses, accesses, branch_misses, branch_accesses, instructions};
        }
    };
    inline bool PerformanceCounter::_has_required_rights = true;
#else

    class PerformanceCounter {
    public:
        [[nodiscard]] constexpr static bool available() noexcept {
            return false;
        }

        /**
        * This OS is not supported
        */
        [[nodiscard]] auto results() const noexcept -> std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> {
            return {0U, 0U, 0U, 0U, 0U};
        }
    };

#endif


    namespace ut = boost::ut;

    /**
     * little compile-time string class (N.B. ideally std::string should become constexpr (everything!! ;-)))
     */
    template<typename CharT, std::size_t SIZE>
    struct fixed_string {
        constexpr static std::size_t N = SIZE;
        CharT _data[N + 1] = {};

        constexpr explicit(false) fixed_string(const CharT (&str)[N + 1]) noexcept {
            if constexpr (N != 0) for (std::size_t i = 0; i < N; ++i) _data[i] = str[i];
        }

        [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }

        [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }

        [[nodiscard]] constexpr explicit operator std::string_view() const noexcept { return {_data, N}; }

        [[nodiscard]] constexpr explicit operator std::string() const noexcept { return {_data, N}; }

        [[nodiscard]] operator const char *() const noexcept { return _data; }

        [[nodiscard]] constexpr bool operator==(const fixed_string &other) const noexcept {
            return std::string_view{_data, N} == std::string_view(other);
        }

        template<std::size_t N2>
        [[nodiscard]] friend constexpr bool
        operator==(const fixed_string &, const fixed_string<CharT, N2> &) { return false; }
    };

    template<typename CharT, std::size_t N>
    fixed_string(const CharT (&str)[N]) -> fixed_string<CharT, N - 1>;

    template<fixed_string... key>
    constexpr bool key_not_found = false;

    /**
     * constexpr const key map that allows modification of the values during run-time while the
     * compile-time look up of keys is free <-> similar to using structs and refl-cpp-based compile-time reflection
     *
     * @author: Ralph J. Steinhagen
     */
    template<typename Value, fixed_string ...Keys>
    struct const_key_map {
        using KeyType = std::tuple<decltype(Keys)...>;
        constexpr static std::size_t SIZE = sizeof...(Keys);
        constexpr static std::array<const std::string_view, SIZE> _keys = {std::string_view(Keys)...};
        std::array<Value, SIZE> _storage;

        template<fixed_string key>
        constexpr static std::size_t get_index_by_name() noexcept {
            if constexpr (constexpr auto itr = std::find_if(_keys.cbegin(), _keys.cend(),
                                                            [](auto const &v) { return v == std::string_view(key); });
                    itr != std::cend(_keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                static_assert(key_not_found<key>, "key not found");
            }
        }

        constexpr static std::size_t get_index_by_name(std::string_view key) {
            if (const auto itr = std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto &v) {
                    return v == std::string_view(key);
                }); itr != std::cend(_keys)) {
                return std::distance(std::cbegin(_keys), itr);
            } else {
                throw std::range_error("key not found");
            }
        }

        constexpr static std::size_t get_index_by_ID(const std::size_t key_ID) {
            if (key_ID < SIZE) {
                return key_ID;
            } else {
                throw std::range_error("key ID not found");
            }
        }

    public:
        [[nodiscard]] constexpr std::size_t size() const { return SIZE; }

        [[nodiscard]] std::string_view key(std::size_t key_ID) const { return _keys[get_index_by_ID(key_ID)];; }

        template<fixed_string key>
        [[nodiscard]] constexpr Value const &at() const { return _storage[get_index_by_name<key>()]; }

        template<fixed_string key>
        [[nodiscard]] constexpr Value &at() { return _storage[get_index_by_name<key>()]; }

        [[nodiscard]] constexpr Value const &at(const std::string_view key) const {
            return _storage[get_index_by_name(key)];
        }

        [[nodiscard]] constexpr Value &at(const std::string_view key) { return _storage[get_index_by_name(key)]; }

        [[nodiscard]] constexpr Value const &at(const std::size_t key_ID) const {
            return _storage[get_index_by_ID(key_ID)];
        }

        [[nodiscard]] constexpr Value &at(const std::size_t key_ID) { return _storage[get_index_by_ID(key_ID)]; }

        [[nodiscard]] constexpr bool contains(const std::string_view key) const {
            return std::find_if(_keys.cbegin(), _keys.cend(), [&key](const auto &v) { return v == key; }) !=
                   std::cend(_keys);
        }
    };

    using StatisticsType = std::array<long double, 5>;
    using ResultMap = std::unordered_map<std::string, std::variant<std::monostate, long double, std::string>>;

    class results {
        using EntryType = std::pair<std::string, ResultMap>;
        using Data = std::vector<EntryType>;
        static std::mutex _lock;
        static Data data;

    public:
        static ResultMap &add_result(std::string_view name) {
            std::lock_guard guard(_lock);
            std::string str(name);
            return data.emplace_back(str, ResultMap()).second;
        }

        template<std::size_t SIGNIFICANT_DIGITS = 3>
        static void print() {
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

            // compute minimum colum width for each benchmark case and metric
            std::unordered_map<std::string, std::size_t> metric_keys;
            for (auto &[test_name, result_map]: data) {
                for (auto &[metric_key, value]: result_map) {
                    if (!metric_keys.contains(metric_key)) {
                        metric_keys.try_emplace(metric_key, metric_key.size());
                    }
                    std::size_t value_size;
                    if (std::holds_alternative<long double>(value)) {
                        value_size = fmt::format("{}", std::get<long double>(value)).size();
                    } else if (std::holds_alternative<std::string>(value)) {
                        value_size = fmt::format("{}", std::get<std::string>(value)).size();
                    } else {
                        throw std::invalid_argument("benchmark::results: unhandled ResultMap type");
                    }
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
                if (result_map.size() == 1) {
                    fmt::print("│ \033[31mFAIL\033[0m ");
                } else {
                    fmt::print("│ \033[32mPASS\033[0m ");
                }

                for (auto &[metric_key, max_width]: metric_keys) {
                    if (result_map.contains(metric_key)) {
                        auto value = result_map.at(metric_key);
                        if (std::holds_alternative<long double>(value)) {
                            fmt::print("│ {1:>{0}} ", max_width, std::get<long double>(value));
                        } else if (std::holds_alternative<std::string>(value)) {
                            fmt::print("│ {1:>{0}} ", max_width, std::get<std::string>(value));
                        } else {
                            fmt::print("│ {1:>{0}} ", max_width, "<variant error>");
                        }

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

    inline results::Data results::data;
    inline std::mutex results::_lock;

    class time_point {
        using timePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using timeDiff = std::chrono::nanoseconds;
        timePoint _time_point;
    public:
        time_point &now() noexcept {
            _time_point = std::chrono::high_resolution_clock::now();
            return *this;
        }

        timeDiff operator-(const time_point &start_marker) const noexcept {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(_time_point - start_marker._time_point);
        }
    };

    namespace utils {

        template<typename T, std::size_t N>
        constexpr std::array<T, N> diff(const std::array<time_point, N> stop, time_point start) {
            std::array<T, N> ret;
            for (int i = 0; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start).count());
                start = stop[i];
            }
            return ret;
        }

        template<typename T, std::size_t N>
        constexpr std::array<T, N> diff(const std::array<time_point, N> &stop, const std::array<time_point, N> &start) {
            std::array<T, N> ret;
            for (int i = 0; i < N; i++) {
                ret[i] = 1e-9 * static_cast<T>((stop[i] - start[i]).count());
            }
            return ret;
        }

        template<typename MapType, std::size_t n_iterations>
        requires (n_iterations > 0)
        constexpr auto convert(const std::array<MapType, n_iterations> &in) {
            std::vector<std::pair<std::string, std::array<time_point, n_iterations>>> ret;
            ret.resize(in[0].size());

            for (int keyID = 0; keyID < in[0].size(); keyID++) {
                ret[keyID].first = std::string(in[0].key(keyID));
                for (int i = 0; i < n_iterations; i++) {
                    ret[keyID].second[i] = in[i].at(keyID);
                }
            }
            return ret;
        }

        template<typename T, std::size_t N>
        [[nodiscard]] StatisticsType compute_statistics(const std::array<T, N> &values) {
            static_assert(N > 0, "array size must not be zero");
            const auto minmax = std::minmax_element(values.cbegin(), values.cend());
            const auto mean = std::accumulate(values.begin(), values.end(), T{}) / static_cast<T>(N);

            T stddev{};
            std::for_each(values.cbegin(), values.cend(), [&](const auto x) { stddev += (x - mean) * (x - mean); });
            stddev /= static_cast<T>(N);
            stddev = std::sqrt(stddev);

            // Compute the median value
            std::array<T, N> sorted_values(values);
            std::sort(sorted_values.begin(), sorted_values.end());
            const auto median = sorted_values[N / 2];
            return {*minmax.first, mean, stddev, median, *minmax.second};
        }

        template<typename T>
        concept Numeric = std::is_integral_v<T> || std::is_floating_point_v<T>;

        template<Numeric T>
        std::string to_si_prefix(T value_base, std::string_view unit = "s", std::size_t significant_digits = 0) {
            static constexpr std::array si_prefixes{'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T',
                                                    'P',
                                                    'E', 'Z', 'Y'};
            static constexpr double base = 1000.0;
            long double value = value_base;

            std::size_t exponent = 8;
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


    }

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
        using return_type_of_t = typename fn_traits<
                typename std::remove_const_t<typename std::remove_reference_t<T>>>::result_type;

        template<typename T>
        using first_arg_of_t =
                std::tuple_element_t<0, typename fn_traits<typename std::remove_const_t<typename std::remove_reference_t<T>>>::args_tuple>;
    } // namespace detail

    template<typename TestFunction>
    constexpr std::size_t argument_size() noexcept {
        if constexpr (std::is_invocable_v<TestFunction>) {
            return 0;
        } else {
            using arg_t = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
            if constexpr (requires { arg_t::SIZE; }) {
                return arg_t::SIZE;
            } else {
                return 0;
            }
        }
    }

    template<fixed_string ...meas_marker_names>
    struct MarkerMap : const_key_map<time_point, meas_marker_names...> {
    };

    template<std::size_t n_iterations = 1LU, typename ResultType = results, fixed_string ...meas_marker_names>
    class benchmark : ut::detail::test {

        template<typename TestFunction>
        constexpr auto get_marker_array() const {
            if constexpr (std::is_invocable_v<TestFunction>) {
                return std::array<bool, n_iterations>();
            } else {
                using MarkerMapType = std::remove_cvref_t<detail::first_arg_of_t<TestFunction>>;
                return std::array<MarkerMapType, n_iterations>();
            }
        }

    public:
        benchmark() = delete;

        explicit benchmark(std::string_view _name) : ut::detail::test{"benchmark", _name} {}

        template<class TestFunction,
                std::size_t MARKER_SIZE = argument_size<TestFunction>(),
                bool has_arguments = MARKER_SIZE != 0>
        //template<fixed_string ...meas_marker, Callback<meas_marker...> Test>
        constexpr benchmark &operator=(TestFunction &&_test) {
            static_cast<ut::detail::test &>(*this) = [&_test, this] {
                using ::benchmark::utils::to_si_prefix;
                auto &result_map = ResultType::add_result(name);
                if constexpr (n_iterations != 1) {
                    result_map.try_emplace("n-iter", fmt::format("{}", n_iterations));
                } else {
                    result_map.try_emplace("n-iter", "");
                }

                std::array<time_point, n_iterations> stop_iter;
                auto marker_iter = get_marker_array<TestFunction>();

                PerformanceCounter execMetrics;
                const auto start = time_point().now();

                if constexpr (n_iterations == 1) {
                    if constexpr (std::is_invocable_v<TestFunction>) {
                        _test();
                    } else {
                        _test(marker_iter[0]);
                    }
                    stop_iter[0].now();
                } else if constexpr (n_iterations >= 1) {
                    for (int i = 0; i < n_iterations; i++) {
                        if constexpr (std::is_invocable_v<TestFunction>) {
                            _test();
                        } else {
                            _test(marker_iter[i]);
                        }
                        stop_iter[i].now();
                    }
                } else {
                    throw std::invalid_argument("benchmark n_iteration := 0 parameter not (yet) implemented");
                }
                // N.B. need to retrieve CPU performance count here no to spoil the result by further post-processing
                auto [misses, accesses, branch_misses, branch_total, instructions] = execMetrics.results();

                // not time-critical post-processing starts here
                const auto time_differences_ns = utils::diff<long double>(stop_iter, start);
                const auto ns = stop_iter[n_iterations - 1] - start;
                const long double duration_s = 1e-9 * static_cast<long double>(ns.count()) / n_iterations;

                const auto add_statistics = []<typename T>(ResultMap &map, const T &time_diff) {
                    if constexpr (n_iterations != 1) {
                        const auto [min, mean, stddev, median, max] = utils::compute_statistics(time_diff);
                        map.try_emplace("min", to_si_prefix(min));
                        map.try_emplace("mean", to_si_prefix(mean));
                        if (stddev == 0) {
                            map.try_emplace("stddev", "");
                        } else {
                            map.try_emplace("stddev", to_si_prefix(stddev));
                        }
                        map.try_emplace("median", to_si_prefix(median));
                        map.try_emplace("max", to_si_prefix(max));
                    } else {
                        map.try_emplace("min", "");
                        map.try_emplace("mean", "");
                        map.try_emplace("stddev", "");
                        map.try_emplace("median", "");
                        map.try_emplace("max", "");
                    }
                };
                add_statistics(result_map, time_differences_ns);


                if (PerformanceCounter::available()) {
                    const long double cache_ratio =
                            100 * static_cast<long double>(misses) / static_cast<long double>(accesses);
                    result_map.try_emplace("CPU cache misses",
                                           fmt::format("({}, {}, {:4.1f} %)", to_si_prefix(misses, "", 0),
                                                       to_si_prefix(accesses, "", 0), cache_ratio));
                    const long double branch_ratio =
                            100 * static_cast<long double>(branch_misses) / static_cast<long double>(branch_total);
                    result_map.try_emplace("CPU branch misses",
                                           fmt::format("({}, {}, {:4.1f} %)", to_si_prefix(branch_misses, "", 0),
                                                       to_si_prefix(branch_total, "", 0), branch_ratio));
                    result_map.try_emplace("CPU-I",
                                           fmt::format("{}", to_si_prefix(instructions / n_iterations, "", 0)));
                }

                result_map.try_emplace("total time: ", to_si_prefix(duration_s));
                result_map.try_emplace("ops/s", to_si_prefix(1.0 / duration_s, "", 2));

                if constexpr (MARKER_SIZE > 0) {
                    auto transposed_map = utils::convert(marker_iter);
                    for (int keyID = 0; keyID < transposed_map.size(); keyID++) {
                        if (keyID > 0) {
                            const auto meas = fmt::format("  └─Marker{}: '{}'->'{}' ", keyID, transposed_map[0].first,
                                                          transposed_map[keyID].first);
                            auto &marker_result_map = ResultType::add_result(meas);

                            marker_result_map.try_emplace("n-iter", "");
                            add_statistics(marker_result_map, utils::diff<long double>(transposed_map[keyID].second,
                                                                                       transposed_map[0].second));
                            // add zero info to not duplicate the results from the superordinate benchmark
                            if (PerformanceCounter::available()) {
                                result_map.try_emplace("CPU cache misses", "");
                                result_map.try_emplace("CPU branch misses", "");
                                result_map.try_emplace("CPU-I", "");
                            }
                            result_map.try_emplace("total time: ", "");
                            result_map.try_emplace("ops/s", "");
                        }
                    }
                }

            };
            return *this;
        }

        template<size_t n>
        auto repeat() {
            return ::benchmark::benchmark<n, ResultType, meas_marker_names...>(this->name);
        }
    };

    [[nodiscard]] auto operator ""_benchmark(const char *name, std::size_t size) {
        return ::benchmark::benchmark<1LU>{{name, size}};
    }

}  // namespace benchmark



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
        constexpr reporter &operator=(TPrinter printer) {
            _printer = static_cast<TPrinter &&>(printer);
            return *this;
        }

        void on(const ut::events::test_begin &test_begin) {}

        void on(const ut::events::test_run &test_run) {
            _printer << "\n \"" << test_run.name << "\"...";
        }

        void on(const ut::events::test_skip &) {}

        void on(const ut::events::test_end &test_end) {
            if (_asserts.fail > 0) {
                ++_benchmarks.fail;
                _printer
                        << _printer.colors().fail << fmt::format("... in benchmark '{}'", test_end.name)
                        << _printer.colors().none
                        << '\n';
                _asserts.fail--;
            }
        }

        template<class TMsg>
        void on(ut::events::log <TMsg> l) {
            _printer << l.msg;
        }

        void on(ut::events::exception exception) {
            _printer << fmt::format("\033[31munexpected exception: \"{}\"\n\033[0m", exception.what());
            ++_asserts.fail;
        }

        template<class TExpr>
        void on(ut::events::assertion_pass <TExpr>) {
            ++_asserts.pass;
        }

        template<class TExpr>
        void on(ut::events::assertion_fail <TExpr> assertion) {
            constexpr auto short_name = [](std::string_view name) {
                return name.rfind('/') != std::string_view::npos
                       ? name.substr(name.rfind('/') + 1)
                       : name;
            };
            _printer << "\n  " << short_name(assertion.location.file_name()) << ':'
                     << assertion.location.line() << ':' << _printer.colors().fail
                     << "FAILED" << _printer.colors().none << " [" << std::boolalpha
                     << assertion.expr << _printer.colors().none << ']';
            ++_asserts.fail;
        }

        void on(const ut::events::fatal_assertion &) const { /* not needed testing interface */ }

        void on(const ut::events::summary &) {
            if (_benchmarks.fail || _asserts.fail) {
                std::cout << _printer.str() << std::endl;
                std::cout << fmt::format("\033[31m{} micro-benchmark(s) failed:\n\033[m", _benchmarks.fail);
            } else {
                std::cout << _printer.colors().pass << "all micro-benchmarks passed:\n" << _printer.colors().none;
            }
            benchmark::results::print();
            std::cerr.flush();
            std::cout.flush();
        }
    };
}  // namespace cfg

template<>
inline auto boost::ut::cfg<boost::ut::override> = ut::runner < cfg::reporter<printer>>
{
};

#endif //GRAPH_PROTOTYPE_BENCHMARK_HPP
