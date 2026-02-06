# GR4 Compile-Time Bottleneck Analysis

**Date:** 2026-02-01
**Branch:** compileTimeTests
**Compiler:** Clang 20.1.8
**Build Type:** Debug
**Analysis Tool:** `-ftime-trace`

## Executive Summary

| Test | Total Time | InstantiateFunction | InstantiateClass |
|------|------------|---------------------|------------------|
| qa_Converter | 2176s | 288s (29K calls) | 64s (15K calls) |
| qa_Math | **6361s** | **1316s (75K calls)** | **607s (57K calls)** |

The qa_Math test has **2.9x more time** spent in template instantiation than qa_Converter, primarily due to heavier use of range adaptors and tuple operations.

---

## Top Bottlenecks by Category

### 1. PairDeduplicateView (RangesHelper.hpp:41-61)

**Impact:** ~164s in qa_Math (101s direct + 63s invoke checks)

**Problem:** Complex chain of range adaptors causes massive template instantiation:
```cpp
return std::forward<Range>(r)
    | std::views::chunk_by(eq1)           // adaptor 1
    | std::views::transform([...] {       // adaptor 2
        return iters
            | std::views::filter([...])   // adaptor 3
            | std::views::transform([...])// adaptor 4
    })
    | std::views::join;                   // adaptor 5
```

Each adaptor triggers `std::__invoke_result` and `std::__invokable_r` trait checks. The nested lambdas with captured references create complex closure types that Clang must fully instantiate.

**Top instantiations:**
- `gr::PairDeduplicateView<lambda, lambda>`: 101s
- `std::__invoke_result<PairDeduplicateView<...>>`: 34s
- `std::__invokable_r<void, PairDeduplicateView<...>>`: 34s

### 2. meta::tuple_for_each (utils.hpp:771-774)

**Impact:** ~73s in qa_Math

**Problem:** The implementation uses index sequences with lambdas:
```cpp
return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
    (([&function, &tuple, &tuples...](auto I) {
        function(std::get<I>(tuple), std::get<I>(tuples)...);
    }(std::integral_constant<std::size_t, Idx>{}), ...));
}(std::make_index_sequence<...>());
```

Each invocation creates a lambda with a complex signature that triggers extensive type trait instantiation for every tuple element.

### 3. for_each_reader_span (Block.hpp:2737-2752)

**Impact:** ~60s in qa_Math

**Problem:** Wraps tuple_for_each with additional lambda complexity:
```cpp
return gr::meta::tuple_for_each(
    [&function](auto&&... args) {
        (..., ([&function](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (ReaderSpanLike<typename ArgType::value_type>) {
                // ...
            }
        }(std::forward<decltype(args)>(args))));
    },
    std::forward<Tuple>(tuple), ...);
```

The nested fold expression with `if constexpr` branches generates multiple specializations.

### 4. MergeView (RangesHelper.hpp:93-189)

**Impact:** ~30s in qa_Math

**Problem:** Custom range view with complex iterator that uses `std::views::zip` internally.

### 5. refl::for_each_data_member_index

**Impact:** 8-11s (reduced from previous ~19s after Optimization B)

**Problem:** Reflection library iterates over all data members with template instantiation per member.

---

## Detailed Time-Trace Analysis

### qa_Math Top Instantiations (>20s each)

| Time | Category | Detail |
|------|----------|--------|
| 110s | OptModule | Code generation for qa_Math.cpp |
| 101s | InstantiateFunction | PairDeduplicateView lambda |
| 73s | InstantiateFunction | tuple_for_each lambda |
| 63s | InstantiateClass | __invoke_result<PairDeduplicateView> |
| 60s | InstantiateFunction | for_each_reader_span |
| 57s | InstantiateClass | __invokable_r<pipeable<bind_back<transform>>> |
| 45s | InstantiateClass | __invokable_r<bind_back<transform>> |
| 39s | InstantiateClass | __invoke_result<transform, chunk_by_view> |
| 39s | InstantiateFunction | __invoke_result for Ranges lambda |

### qa_Converter Top Instantiations

| Time | Category | Detail |
|------|----------|--------|
| 52s | OptModule | Code generation |
| 16s | InstantiateFunction | boost::ut operators |
| 10s | InstantiateFunction | std::apply (test framework) |
| 8s | InstantiateFunction | refl::for_each_data_member_index |
| 6.5s | InstantiateFunction | std::map<string, PropertyCallback> |
| 6s | InstantiateFunction | std::ranges::views::transform |

---

## Optimization Opportunities

### Opportunity C: Simplify PairDeduplicateView

**Current complexity:** 5 chained range adaptors with nested lambdas

**Potential approaches:**

1. **Replace with explicit loop** (for non-lazy evaluation contexts):
```cpp
template<typename Range, typename Eq1, typename Eq2>
auto pair_deduplicate_eager(Range&& r, Eq1 eq1, Eq2 eq2) {
    std::vector<std::ranges::range_value_t<Range>> result;
    // explicit chunk + dedupe logic without adaptor chains
    return result;
}
```

2. **Reduce adaptor nesting** - flatten the transform chains

3. **Use type-erased iterators** for runtime flexibility without compile-time cost

**Estimated savings:** 50-100s for qa_Math

### Opportunity D: Optimize tuple_for_each Hot Paths

**Problem:** Every call to `for_each_reader_span`, `for_each_writer_span`, `for_each_port` instantiates the full template machinery.

**Potential approaches:**

1. **Pre-instantiate common patterns** - explicit template instantiation for common tuple sizes

2. **Use function pointers** where the lambda body is simple:
```cpp
// Instead of lambda-based approach
using ProcessFn = void(*)(auto&);
void for_each_reader_span_impl(ProcessFn fn, ...);
```

3. **Reduce lambda capture complexity** - avoid capturing by reference when possible

**Estimated savings:** 30-50s for qa_Math

### Opportunity E: Defer Range Composition to Runtime

For `Block::updateMergedInputTagAndApplySettings()`, the tag merging logic uses:
```cpp
auto mergedPairsLazy = allPairViews | Merge{...};
auto nonDuplicatedInputTags = mergedPairsLazy | PairDeduplicateView(...);
```

**Alternative:** Type-erase the intermediate views or use eager evaluation:
```cpp
std::vector<PairRelIndexMapRef> mergedPairs;
// ... populate with explicit loop
std::vector<PairRelIndexMapRef> dedupedPairs;
// ... deduplicate with explicit loop
```

**Trade-off:** Runtime overhead vs compile-time savings

### Opportunity F: Reduce std::ranges Type Deduction Overhead

Many bottlenecks come from `std::__invoke_result` and `std::__invokable_r` checks.

**Approach:** Use `auto` return types with explicit constraints rather than deeply nested range adaptors:
```cpp
// Instead of
auto result = input | transform(f1) | filter(f2) | transform(f3);

// Consider
auto result = [](auto& input) -> std::vector<...> {
    std::vector<...> out;
    for (auto& x : input) {
        if (auto y = f1(x); f2(y)) {
            out.push_back(f3(y));
        }
    }
    return out;
}(input);
```

---

## Files to Investigate

| File | Bottleneck | Impact |
|------|------------|--------|
| `meta/include/.../RangesHelper.hpp` | PairDeduplicateView, MergeView | ~200s |
| `meta/include/.../utils.hpp` | tuple_for_each | ~73s |
| `core/include/.../Block.hpp` | for_each_reader/writer_span | ~60s |
| `core/include/.../Block.hpp` | updateMergedInputTagAndApplySettings | ~30s |
| `core/include/.../Settings.hpp` | refl::for_each usage | ~11s |

---

## Recommended Priority

1. **High Impact:** PairDeduplicateView simplification (~100-150s savings)
2. **Medium Impact:** tuple_for_each optimization (~30-50s savings)
3. **Medium Impact:** for_each_reader_span refactoring (~30s savings)
4. **Lower Impact:** Eager evaluation in tag processing (~20s savings)

**Total potential savings:** 180-250s for qa_Math (~3-4 minutes per file)

---

## Raw Data Location

- qa_Converter time-trace: `build-timetrace/blocks/basic/test/CMakeFiles/qa_Converter.dir/qa_Converter.cpp.json`
- qa_Math time-trace: `build-timetrace/blocks/math/test/CMakeFiles/qa_Math.dir/qa_Math.cpp.json`
