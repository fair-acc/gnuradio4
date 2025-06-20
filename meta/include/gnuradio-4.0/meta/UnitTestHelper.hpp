#ifndef UNITTESTHELPER_HPP
#define UNITTESTHELPER_HPP

#include <boost/ut.hpp>
#include <concepts>
#include <cstddef>
#include <format>
#include <ranges>

#include "formatter.hpp"

#if defined(_WIN32)
#include <windows.h>
[[maybe_unused]] inline const int _set_console_output = [] {
    SetConsoleOutputCP(CP_UTF8);
    return 0;
}();
#endif

namespace gr::test {
using namespace boost::ut;

template<typename T>
concept HasSize = requires(const T c) {
    { c.size() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept Collection = std::ranges::range<T> && HasSize<T> && !std::is_convertible_v<T, std::string_view> && !std::is_same_v<std::remove_cvref_t<T>, std::string>;
} // namespace gr::test

namespace boost::ut {
template<gr::test::Collection RangeLHS, gr::test::Collection RangeRHS>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto eq(const RangeLHS& lhs, const RangeRHS& rhs);

template<gr::test::Collection RangeLHS, gr::test::Collection RangeRHS, typename T = std::ranges::range_value_t<RangeRHS>>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto eq(const RangeLHS& lhs, const RangeRHS& rhs, T tolerance);

template<typename Enum>
requires std::is_enum_v<Enum>
auto eq(Enum lhs, Enum rhs);
} // namespace boost::ut

namespace gr::test {
using namespace boost::ut;

struct eq_collection_result {
    bool                 success{};
    std::string          message{};
    std::source_location location = std::source_location::current();

    operator bool() const { return success; }
    friend std::ostream& operator<<(std::ostream& os, const eq_collection_result& r) { return os << r.message; }
};

template<Collection RangeLHS, Collection RangeRHS>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto eq_collections(const RangeLHS& LHS, const RangeRHS& RHS, std::size_t contextWindow = 3, std::source_location location = std::source_location::current()) -> eq_collection_result {
    const auto sizeLHS = LHS.size();
    const auto sizeRHS = RHS.size();
    if (sizeLHS != sizeRHS) {
        return {false, std::format("Collections size mismatch: LHS.size()={}, RHS.size()={}", sizeLHS, sizeRHS), location};
    }

    auto firstMismatch = std::ranges::mismatch(LHS, RHS);
    if (firstMismatch.in1 == LHS.end()) { // ferfect match
        return {true, std::format("Collections match ({} elements)", sizeLHS), location};
    }

    // define context window around first mismatched value
    const std::ptrdiff_t idx         = std::distance(LHS.begin(), firstMismatch.in1);
    const std::ptrdiff_t ctxStartIdx = idx < static_cast<std::ptrdiff_t>(contextWindow) ? 0 : (idx - static_cast<std::ptrdiff_t>(contextWindow));
    const std::ptrdiff_t ctxStopIdx  = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(sizeLHS), idx + static_cast<std::ptrdiff_t>(contextWindow) + 1);

    std::ostringstream ctxLHS, ctxRHS;
    for (auto i = ctxStartIdx; i < ctxStopIdx; ++i) {
        ctxLHS << *std::next(LHS.begin(), i) << ' ';
        ctxRHS << *std::next(RHS.begin(), i) << ' ';
    }

    return {false,
        std::format("Collections differ at index={0}; LHS[{0}]={1} vs RHS[{0}]={2}\n"
                    "Context window [{3}, {4}]:\n  left:  {5}\n  right: {6}",
            idx, *firstMismatch.in1, *firstMismatch.in2, ctxStartIdx, ctxStopIdx - 1, ctxLHS.str(), ctxRHS.str()),
        location};
}

template<Collection RangeLHS, Collection RangeRHS, typename T = std::ranges::range_value_t<RangeRHS>>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto approx_collections(const RangeLHS& LHS, const RangeRHS& RHS, T tolerance, std::size_t contextWindow = 3, std::source_location location = std::source_location::current()) -> eq_collection_result {
    const auto sizeLHS = LHS.size();
    const auto sizeRHS = RHS.size();
    if (sizeLHS != sizeRHS) {
        return {false, std::format("Collections size mismatch: LHS.size()={}, RHS.size()={}", sizeLHS, sizeRHS), location};
    }

    const auto pred = [tolerance](auto const& lhsValue, auto const& rhsValue) noexcept {
        auto diff = (lhsValue > rhsValue) ? lhsValue - rhsValue : rhsValue - lhsValue;
        return diff <= tolerance;
    };

    // define context window around first mismatched value with custom predicate
    auto firstMismatch = std::ranges::mismatch(LHS, RHS, pred);
    if (firstMismatch.in1 == LHS.end()) {
        return {true, std::format("Collections approx match ({} elements) within tolerance={}", sizeLHS, tolerance), location};
    }

    // define context window around first mismatched value
    const std::ptrdiff_t idx         = std::distance(LHS.begin(), firstMismatch.in1);
    const std::ptrdiff_t ctxStartIdx = idx < static_cast<std::ptrdiff_t>(contextWindow) ? 0 : (idx - static_cast<std::ptrdiff_t>(contextWindow));
    const std::ptrdiff_t ctxStopIdx  = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(sizeLHS), idx + static_cast<std::ptrdiff_t>(contextWindow) + 1);

    std::ostringstream ctxLHS, ctxRHS;
    for (auto i = ctxStartIdx; i < ctxStopIdx; ++i) {
        ctxLHS << *std::next(LHS.begin(), i) << ' ';
        ctxRHS << *std::next(RHS.begin(), i) << ' ';
    }

    return {false,
        std::format("Collections differ (approx) at index={0}; LHS[{0}]={1} vs RHS[{0}]={2} (tolerance={3})\n"
                    "Context window [{4}, {5}]:\n  left:  {6}\n  right: {7}",
            idx, *firstMismatch.in1, *firstMismatch.in2, tolerance, ctxStartIdx, ctxStopIdx - 1, ctxLHS.str(), ctxRHS.str()),
        location};
}

} // namespace gr::test

template<gr::test::Collection RangeLHS, gr::test::Collection RangeRHS>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto boost::ut::eq(const RangeLHS& lhs, const RangeRHS& rhs) {
    return gr::test::eq_collections(lhs, rhs);
}

template<gr::test::Collection RangeLHS, gr::test::Collection RangeRHS, typename T>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto boost::ut::eq(const RangeLHS& lhs, const RangeRHS& rhs, T tolerance) {
    return gr::test::approx_collections(lhs, rhs, tolerance);
}

template<typename Enum>
requires std::is_enum_v<Enum>
auto boost::ut::eq(Enum lhs, Enum rhs) {
    return lhs == rhs;
}

template<typename Enum>
requires std::is_enum_v<Enum>
std::ostream& operator<<(std::ostream& os, Enum e) {
    if constexpr (std::is_enum_v<Enum>) {
        if (auto name = magic_enum::enum_name(e); !name.empty()) {
            return os << name;
        }
    }
    return os << static_cast<std::underlying_type_t<Enum>>(e);
}

#endif // UNITTESTHELPER_HPP
