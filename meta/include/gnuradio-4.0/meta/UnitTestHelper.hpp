#ifndef UNITTESTHELPER_HPP
#define UNITTESTHELPER_HPP

#include <boost/ut.hpp>
#include <concepts>
#include <cstddef>
#include <fmt/format.h>
#include <ranges>

#include "formatter.hpp"

namespace gr::test {
using namespace boost::ut;

template<typename T>
concept HasSize = requires(const T c) {
    { c.size() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept Collection = std::ranges::range<T> && HasSize<T>;

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
        return {false, fmt::format("Collections size mismatch: LHS.size()={}, RHS.size()={}", sizeLHS, sizeRHS), location};
    }

    auto firstMismatch = std::ranges::mismatch(LHS, RHS);
    if (firstMismatch.in1 == LHS.end()) { // ferfect match
        return {true, fmt::format("Collections match ({} elements)", sizeLHS), location};
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
        fmt::format("Collections differ at index={idx}; LHS[{idx}]={lhs} vs RHS[{idx}]={rhs}\nContext window [{ctx_start}, {ctx_end}]:\n  left:  {lhs_context}\n  right: {rhs_context}", //
            fmt::arg("idx", idx), fmt::arg("lhs", *firstMismatch.in1), fmt::arg("rhs", *firstMismatch.in2),                                                                              //
            fmt::arg("ctx_start", ctxStartIdx), fmt::arg("ctx_end", ctxStopIdx - 1), fmt::arg("lhs_context", ctxLHS.str()), fmt::arg("rhs_context", ctxRHS.str())),
        location};
}

template<Collection RangeLHS, Collection RangeRHS, typename T = std::ranges::range_value_t<RangeRHS>>
requires std::is_same_v<std::ranges::range_value_t<RangeLHS>, std::ranges::range_value_t<RangeRHS>>
auto approx_collections(const RangeLHS& LHS, const RangeRHS& RHS, T tolerance, std::size_t contextWindow = 3, std::source_location location = std::source_location::current()) -> eq_collection_result {
    const auto sizeLHS = LHS.size();
    const auto sizeRHS = RHS.size();
    if (sizeLHS != sizeRHS) {
        return {false, fmt::format("Collections size mismatch: LHS.size()={}, RHS.size()={}", sizeLHS, sizeRHS), location};
    }

    const auto pred = [tolerance](auto const& lhsValue, auto const& rhsValue) noexcept {
        auto diff = (lhsValue > rhsValue) ? lhsValue - rhsValue : rhsValue - lhsValue;
        return diff <= tolerance;
    };

    // define context window around first mismatched value with custom predicate
    auto firstMismatch = std::ranges::mismatch(LHS, RHS, pred);
    if (firstMismatch.in1 == LHS.end()) {
        return {true, fmt::format("Collections approx match ({} elements) within tolerance={}", sizeLHS, tolerance), location};
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
        fmt::format("Collections differ (approx) at index={idx}; LHS[{idx}]={lhs} vs RHS[{idx}]={rhs} (tolerance={tol})\nContext window [{ctx_start}, {ctx_end}]:\n  left:  {lhs_context}\n  right: {rhs_context}", //
            fmt::arg("idx", idx), fmt::arg("lhs", *firstMismatch.in1), fmt::arg("rhs", *firstMismatch.in2), fmt::arg("tol", tolerance),                                                                             //
            fmt::arg("ctx_start", ctxStartIdx), fmt::arg("ctx_end", ctxStopIdx - 1), fmt::arg("lhs_context", ctxLHS.str()), fmt::arg("rhs_context", ctxRHS.str())),
        location};
}

} // namespace gr::test

#endif // UNITTESTHELPER_HPP
