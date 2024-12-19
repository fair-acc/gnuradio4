#ifndef GNURADIO_TESTING_SETTINGS_CHANGE_RECORDER_H
#define GNURADIO_TESTING_SETTINGS_CHANGE_RECORDER_H

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>

namespace gr::testing {

namespace utils {

std::string format_variant(const auto& value) noexcept {
    return std::visit(
        [](auto& arg) {
            using Type = std::decay_t<decltype(arg)>;
            if constexpr (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || std::is_same_v<Type, std::complex<float>> || std::is_same_v<Type, std::complex<double>>) {
                return fmt::format("{}", arg);
            } else if constexpr (std::is_same_v<Type, std::monostate>) {
                return fmt::format("monostate");
            } else if constexpr (std::is_same_v<Type, std::vector<std::complex<float>>> || std::is_same_v<Type, std::vector<std::complex<double>>>) {
                return fmt::format("[{}]", fmt::join(arg, ", "));
            } else if constexpr (std::is_same_v<Type, std::vector<std::string>> || std::is_same_v<Type, std::vector<bool>> || std::is_same_v<Type, std::vector<unsigned char>> || std::is_same_v<Type, std::vector<unsigned short>> || std::is_same_v<Type, std::vector<unsigned int>> || std::is_same_v<Type, std::vector<unsigned long>> || std::is_same_v<Type, std::vector<signed char>> || std::is_same_v<Type, std::vector<short>> || std::is_same_v<Type, std::vector<int>> || std::is_same_v<Type, std::vector<long>> || std::is_same_v<Type, std::vector<float>> || std::is_same_v<Type, std::vector<double>>) {
                return fmt::format("[{}]", fmt::join(arg, ", "));
            } else {
                return fmt::format("not-yet-supported type {}", gr::meta::type_name<Type>());
            }
        },
        value);
}

void printChanges(const property_map& oldMap, const property_map& newMap) noexcept {
    for (const auto& [key, newValue] : newMap) {
        if (!oldMap.contains(key)) {
            fmt::print("    key added '{}` = {}\n", key, format_variant(newValue));
        } else {
            const auto& oldValue = oldMap.at(key);
            const bool  areEqual = std::visit(
                [](auto&& arg1, auto&& arg2) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(arg1)>, std::decay_t<decltype(arg2)>>) {
                        // compare values if they are of the same type
                        return arg1 == arg2;
                    } else {
                        return false; // values are of different type
                    }
                },
                oldValue, newValue);

            if (!areEqual) {
                fmt::print("    key value changed: '{}` = {} -> {}\n", key, format_variant(oldValue), format_variant(newValue));
            }
        }
    }
};
} // namespace utils

template<typename T>
// struct SettingsChangeRecorder : public Block<SettingsChangeRecorder<T>, BlockingIO<true>, SupportedTypes<float, double>> { // TODO: reenable BlockingIO
struct SettingsChangeRecorder : public Block<SettingsChangeRecorder<T>, SupportedTypes<float, double>> {
    using Description = Doc<R""(
some test doc documentation
)"">;
    PortIn<T>  in{};
    PortOut<T> out{};

    // settings
    Annotated<T, "scaling factor", Visible, Doc<"y = a * x">, Unit<"As">>                    scaling_factor = static_cast<T>(1); // N.B. unit 'As' = 'Coulomb'
    Annotated<std::string, "context information", Visible>                                   context{};
    gr::Size_t                                                                               n_samples_max = 0;
    Annotated<float, "sample rate", Limits<int64_t(0), std::numeric_limits<int64_t>::max()>> sample_rate   = 1.0f;
    std::vector<T>                                                                           vector_setting{T(3), T(2), T(1)};
    Annotated<std::vector<std::string>, "string vector">                                     string_vector_setting = {};

    GR_MAKE_REFLECTABLE(SettingsChangeRecorder, in, out, scaling_factor, context, n_samples_max, sample_rate, vector_setting, string_vector_setting);

    bool       _debug            = true;
    int        _updateCount      = 0;
    bool       _resetCalled      = false;
    gr::Size_t _nSamplesConsumed = 0;

    void settingsChanged(const property_map& oldSettings, property_map& newSettings, property_map& fwdSettings) noexcept {
        // optional function that is called whenever settings change
        _updateCount++;

        if (_debug) {
            fmt::println("block '{}' settings changed - update_count: {}", this->name, _updateCount);
            utils::printChanges(oldSettings, newSettings);
            for (const auto& [key, value] : fwdSettings) {
                fmt::println(" -- forward: '{}':{}", key, value);
            }
        }
    }

    void reset() {
        // optional reset function
        _nSamplesConsumed = 0;
        _resetCalled      = true;
    }

    [[nodiscard]] constexpr T processOne(const T& a) noexcept {
        _nSamplesConsumed++;
        return a * scaling_factor;
    }
};

static_assert(BlockLike<SettingsChangeRecorder<int>>);
static_assert(BlockLike<SettingsChangeRecorder<float>>);
static_assert(BlockLike<SettingsChangeRecorder<double>>);
} // namespace gr::testing
#endif // include guard
