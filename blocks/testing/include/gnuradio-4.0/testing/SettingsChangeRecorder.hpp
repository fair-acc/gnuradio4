#ifndef GNURADIO_TESTING_SETTINGS_CHANGE_RECORDER_H
#define GNURADIO_TESTING_SETTINGS_CHANGE_RECORDER_H

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>

namespace gr::testing {

namespace utils {

std::string format_variant(const auto& value) noexcept {
    std::string result;
    pmt::ValueVisitor([&result](auto& arg) {
        using Type = std::decay_t<decltype(arg)>;
        if constexpr (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || std::is_same_v<Type, std::complex<float>> || std::is_same_v<Type, std::complex<double>>) {
            result = std::format("{}", arg);
        } else if constexpr (std::is_same_v<Type, std::monostate>) {
            result = std::format("monostate");
        } else if constexpr (std::is_same_v<Type, std::vector<std::complex<float>>> || std::is_same_v<Type, std::vector<std::complex<double>>>) {
            result = std::format("[{}]", gr::join(arg, ", "));
        } else if constexpr (std::is_same_v<Type, std::vector<std::string>> || std::is_same_v<Type, std::vector<bool>> || std::is_same_v<Type, std::vector<unsigned char>> || std::is_same_v<Type, std::vector<unsigned short>> || std::is_same_v<Type, std::vector<unsigned int>> || std::is_same_v<Type, std::vector<unsigned long>> || std::is_same_v<Type, std::vector<signed char>> || std::is_same_v<Type, std::vector<short>> || std::is_same_v<Type, std::vector<int>> || std::is_same_v<Type, std::vector<long>> || std::is_same_v<Type, std::vector<float>> || std::is_same_v<Type, std::vector<double>>) {
            result = std::format("[{}]", gr::join(arg, ", "));
        } else {
            result = std::format("not-yet-supported type {}", gr::meta::type_name<Type>());
        }
    }).visit(value);
    return result;
}

void printChanges(const property_map& oldMap, const property_map& newMap) noexcept {
    for (const auto& [key, newValue] : newMap) {
        if (!oldMap.contains(key)) {
            std::print("    key added '{}` = {}\n", std::string_view(key), format_variant(newValue));
        } else {
            const auto& oldValue = oldMap.at(key);
            const bool  areEqual = oldValue == newValue;
            if (!areEqual) {
                std::print("    key value changed: '{}` = {} -> {}\n", std::string_view(key), format_variant(oldValue), format_variant(newValue));
            }
        }
    }
}
} // namespace utils

GR_REGISTER_BLOCK(gr::testing::SettingsChangeRecorder, [T], [ int32_t, float, double ])

enum class TestEnum { TEST_STATE1, TEST_STATE2, TEST_STATE3 };

template<typename T>
// struct SettingsChangeRecorder : public Block<SettingsChangeRecorder<T>, BlockingIO<true>, SupportedTypes<float, double>> { // TODO: reenable BlockingIO
struct SettingsChangeRecorder : Block<SettingsChangeRecorder<T>> {
    using Description = Doc<R""(some test doc documentation)"">;
    PortIn<T>  in{};
    PortOut<T> out{};

    // settings
    Annotated<T, "scaling factor", Visible, Doc<"y = a * x">, Unit<"As">>                    scaling_factor = static_cast<T>(1); // N.B. unit 'As' = 'Coulomb'
    Annotated<std::string, "context information", Visible>                                   context{};
    gr::Size_t                                                                               n_samples_max = 0;
    Annotated<float, "sample rate", Limits<int64_t(0), std::numeric_limits<int64_t>::max()>> sample_rate   = 1.0f;
    Tensor<T>                                                                                vector_setting{gr::data_from, {T(3), T(2), T(1)}};
    Annotated<Tensor<pmt::Value>, "string vector">                                           string_vector_setting       = {};
    TestEnum                                                                                 test_enum_setting           = TestEnum::TEST_STATE1;
    Annotated<TestEnum, "annotated enum">                                                    annotated_test_enum_setting = TestEnum::TEST_STATE1;

    GR_MAKE_REFLECTABLE(SettingsChangeRecorder, in, out, scaling_factor, context, n_samples_max, sample_rate, vector_setting, string_vector_setting, test_enum_setting, annotated_test_enum_setting);

    bool       _debug            = true;
    int        _updateCount      = 0;
    bool       _resetCalled      = false;
    gr::Size_t _nSamplesConsumed = 0;

    void settingsChanged(const property_map& oldSettings, property_map& newSettings, property_map& fwdSettings) noexcept {
        // optional function that is called whenever settings change
        _updateCount++;

        if (_debug) {
            std::println("block '{}' settings changed - update_count: {}", this->name, _updateCount);
            utils::printChanges(oldSettings, newSettings);
            for (const auto& [key, value] : fwdSettings) {
                std::println(" -- forward: '{}':{}", std::string_view(key), value);
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

} // namespace gr::testing
#endif // GNURADIO_TESTING_SETTINGS_CHANGE_RECORDER_H
