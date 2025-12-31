#ifndef CORE_TEST_VALUE_UTILS_HPP
#define CORE_TEST_VALUE_UTILS_HPP

#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Value.hpp>

namespace gr::testing {

using namespace boost::ut;
using namespace gr;

template<typename T>
auto get_value_or_fail(const gr::pmt::Value& value, std::source_location sourceLocation = std::source_location::current()) {
    if constexpr (std::is_same_v<T, std::string>) {
        auto str = value.value_or(std::string_view{});
        if (str.data() != nullptr) {
            return std::string(str);
        }

    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        auto tensor = value.get_if<Tensor<pmt::Value>>();
        if (tensor != nullptr) {
            return *tensor | std::views::transform([](const gr::pmt::Value& _value) { return _value.value_or(std::string()); }) | std::ranges::to<T>();
        }

    } else if constexpr (meta::is_instantiation_of<T, std::vector>) {
        using TValue = typename T::value_type;
        auto tensor  = value.get_if<Tensor<TValue>>();
        if (tensor != nullptr) {
            T result;
            if (auto conversionResult = pmt::assignTo(result, *tensor); conversionResult) {
                return result;
            }
        }

    } else {
        auto ptr = value.get_if<T>();
        if (ptr == nullptr) {
            std::println("ptr == nullptr calling get_value_or_fail from {}:{}", sourceLocation.file_name(), sourceLocation.line());
            assert(ptr != nullptr);
        }
        if (ptr != nullptr) {
            return *ptr;
        }
    }

    expect(false) << std::format("Required type not stored in value {}:{}", sourceLocation.file_name(), sourceLocation.line());

    // We cannot allow the test to continue
    throw std::format("Required type not stored in value {}:{}", sourceLocation.file_name(), sourceLocation.line());
}
} // namespace gr::testing

#endif // include guard
