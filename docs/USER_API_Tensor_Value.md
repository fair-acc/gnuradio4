# `gr::Tensor<T>` & `gr::pmt::Value` User-API Reference

## `gr::Tensor<T, Extents...>` — Multi-Dimensional Array

GR4's high-performance signal processing needs dense N-dimensional storage with:

- **compile-time shape optimisation** — static tensors live on the stack with zero allocation
- **SIMD-friendly contiguous layout** — row-major storage enables vectorised operations
- **zero-copy views** — `TensorView` is trivially copyable for host↔device transfers (SYCL, CUDA, …)
- **PMR allocator support** — custom memory pools for arena-based allocation and streamlined CPU↔GPU interoperability

### TL;DR

```cpp
#include <gnuradio-4.0/Tensor.hpp>

Tensor<float>              dynamic{{3, 4}};       // 3×4, heap (PMR)
Tensor<float, 3, 4>        fixed;                 // 3×4, stack (no allocation)
Tensor<float, dyn, 4>      semi{{3}};             // 3 rows dynamic, 4 cols fixed
TensorView<float>          view{tensor};          // mutable non-owning view
TensorView<const float>    cview{tensor};         // const non-owning view (read-only)
```

### Construction

```cpp
// dynamic tensor — shape at runtime, PMR-allocated
Tensor<float> a{{3, 4}};                          // 3×4, zero-initialised
Tensor<float> b{{2, 3}, {1,2,3,4,5,6}};           // 2×3 with data

// static tensor — shape at compile-time (std::array storage, no heap)
Tensor<float, 2, 3> c;                            // stack-allocated 2×3
Tensor<float, 2, 3> d{1,2,3,4,5,6};               // with initialiser

// semi-static — mix of compile-time and runtime extents
Tensor<float, std::dynamic_extent, 4> e{{3}};    // 3 rows dynamic, 4 cols fixed

// view from raw pointer + shape
float* ptr = get_buffer();
TensorView<float> raw{ptr, {rows, cols}};
TensorView<float> row0 = tensor.slice(0);              // first row as 1D view
TensorView<float> sub  = tensor.slice({1, 3}, {0, 2}); // submatrix

// disambiguate ambiguous initializer_list cases
// N.B. Tensor<int>{{1,2,3}} is ambiguous: rank-3 shape or 1D data?
Tensor<int> f{data_from, {1,2,3}};               // → [1,2,3] (1D, 3 elements)
Tensor<int> g{rank_from, {1,2,3}};               // → shape 1×2×3 (6 elements)

// unambiguous cases (type mismatch resolves it):
Tensor<float> h{{2, 3}};                          // → shape 2×3 (ints → extents)
Tensor<float> i{{2, 3}, {1,2,3,4,5,6}};           // → shape + data

// pass tensors via concepts
void process(TensorLike auto& input);
```

### Element Access

```cpp
tensor[0, 1]           // multi-index — primary access
tensor[5]              // flat linear index (row-major order)
tensor.at(0, 1)        // bounds-checked (throws on out-of-range)

tensor.data()          // raw T* pointer (contiguous storage)
tensor.size()          // total element count
tensor.extents()       // shape as std::span<const std::size_t>
tensor.rank()          // number of dimensions
tensor.extent(0)       // size of dimension 0
tensor.strides()       // stride array (row-major: last stride = 1)
// + range adapters: begin(), end(), rbegin(), …
```

### Operations

```cpp
tensor.reshape({6, 2});                           // reinterpret shape (same total size)
tensor.fill(0.0f);                                // set all elements
tensor.resize({4, 5});                            // change size (dynamic only, reallocates)
tensor.reserve(100);                              // pre-allocate capacity (dynamic only)

// STL compatibility — Tensor is a contiguous range
for (auto& x : tensor) { x *= 2; }
std::ranges::sort(tensor);
std::copy(tensor.begin(), tensor.end(), output);

// conversion to STL containers
std::vector<float> vec(tensor.begin(), tensor.end());
```

### Tensor Math Operations

`TensorMath.hpp` provides SIMD-optimised operations via `std::simd` (vir/simd). Works with both `Tensor` and `TensorView`.

```cpp
#include <gnuradio-4.0/TensorMath.hpp>
using gr::math::TensorOps;
```

#### GEMM / GEMV (matrix operations)

Cache-blocked, vectorised matrix multiplication achieving near-BLAS performance for small-to-medium matrices.

```cpp
gr::Tensor<float, 64, 64> A, B, C;

// matrix-matrix: C = A × B
gr::math::gemm(A, B, C);

// with scaling: C = α·A·B + β·C
gr::math::gemm(A, B, C, /*alpha=*/1.0f, /*beta=*/0.5f);

// with explicit CPU execution policy
gemm<NoTrans, NoTrans>(gr::math::cpu_policy{}, C, A, B, /*alpha=*/1.0f, /*beta=*/0.0f);

// matrix-vector: y = A × x
gr::Tensor<float, 64> x, y;
gr::math::gemv(A, x, y);

// works with TensorView<T> (zero-copy from existing buffers)
gr::math::gemv(TensorView<const float>{A}, TensorView<const float>{x}, y);

// usual +, -, *, / operators (default to CPU/SIMD compute policy)
```

#### Reductions

```cpp
gr::Tensor<double> t{{2, 3}, {1, 2, 3, 4, 5, 6}};

// global reductions
auto total  = TensorOps<double>::sum(t);          // 21
auto prod   = TensorOps<double>::product(t);      // 720
auto avg    = TensorOps<double>::mean(t);         // 3.5
auto var    = TensorOps<double>::variance(t);
auto stddev = TensorOps<double>::std_dev(t);

auto lo     = TensorOps<double>::min(t);          // 1
auto hi     = TensorOps<double>::max(t);          // 6
auto imin   = TensorOps<double>::argmin(t);       // 0 (flat index)
auto imax   = TensorOps<double>::argmax(t);       // 5 (flat index)

// axis reductions (returns rank-1 tensor)
auto sum0   = TensorOps<double>::sum_axis(t, 0);  // sum along rows → [5, 7, 9]
auto sum1   = TensorOps<double>::sum_axis(t, 1);  // sum along cols → [6, 15]
auto mean0  = TensorOps<double>::mean_axis(t, 0);
```

#### Element-wise operations

```cpp
gr::Tensor<float> a{{3}, {1, 2, 3}};
gr::Tensor<float> b{{3}, {4, 5, 6}};

// in-place (modify first argument)
TensorOps<float>::add_inplace(a, b);                   // a = a + b
TensorOps<float>::subtract_inplace(a, b);              // a = a − b
TensorOps<float>::multiply_elementwise_inplace(a, b);  // a = a ⊙ b (Hadamard)

// non-modifying (returns new tensor)
auto c = TensorOps<float>::add(a, b);
auto d = TensorOps<float>::subtract(a, b);
auto e = TensorOps<float>::multiply_elementwise(a, b);

// scalar operations
TensorOps<float>::multiply_scalar_inplace(a, 2.0f);    // a *= 2
TensorOps<float>::divide_scalar_inplace(a, 2.0f);      // a /= 2
auto scaled = TensorOps<float>::multiply_scalar(a, 3.0f);
```

#### Special operations

```cpp
Tensor<double> t{{4}, {1.0, std::nan(""), 3.0, std::numeric_limits<double>::infinity()}};

bool has_nan = TensorOps<double>::contains_nan(t);  // true
bool has_inf = TensorOps<double>::contains_inf(t);  // true
```

---

## `gr::pmt::Value` — Polymorphic Type-Erased Container

Tags and block settings need a single type that holds scalars, strings, tensors, or maps. The previous `rva::variant`-based `pmtv::pmt` had issues:

- **compile-time overhead** — recursive variant instantiation inflates build times
- **binary bloat** — variant machinery increases symbol tables and binary size
- **exception dependency** — `std::bad_variant_access` incompatible with `-fno-exceptions`
- **RTTI dependency** — conflicts with `-fno-rtti` builds (common on microcontrollers)

`gr::pmt::Value` is a compact, hand-rolled type-erased container with:

- **24-byte footprint** — scalars stored inline (zero heap allocation)
- **PMR allocator support** — complex types use configurable memory pools
- **exception-free** — uses `assert` + `std::unreachable()` for type mismatches
- **monadic API** — `value_or`, `or_else`, `transform`, `and_then` for safe access

### TL;DR

```cpp
#include <gnuradio-4.0/Value.hpp>
using gr::pmt::Value;

Value v{42};                                      // int64 (inline, no heap)
Value s{"hello"};                                 // string (heap via PMR)
Value t{Tensor<float>{{3,3}}};                    // tensor (heap via PMR)

int64_t x = v.value_or<int64_t>(0);               // safe access with fallback
```

### Storage Layout

```
┌─────────┬─────────────┬─────────────────┬─────────────────┬─────────┐
│ 1 byte  │ 1 byte      │     8 bytes     │     8 bytes     │ 6 bytes │
│ValueType│ContainerType|  Inline Union   │   PMR Resource  │ Padding │
└─────────┴─────────────┴─────────────────┴─────────────────┴─────────┘
         ← 24 bytes total →
```

Scalars (bool, int8–64, uint8–64, float, double) stored inline. Complex numbers, strings, tensors, and maps allocate through PMR.

### Construction

```cpp
Value empty;                                      // monostate
Value i{std::int64_t{42}};                        // scalar (inline)
Value d{3.14};                                    // float64
Value c{std::complex<float>{1, 2}};               // complex (PMR heap)
Value s{std::string_view{"text"}};                // string (PMR heap)
Value t{Tensor<double>{{2, 3}}};                  // tensor (PMR heap)
Value m{Map{{"key", Value{1}}}};                  // map (PMR heap)
```

### Type Queries

```cpp
// category predicates
v.is_monostate();         // empty/null
v.is_arithmetic();        // any numeric (scalar or complex)
v.is_integral();          // int8–64, uint8–64
v.is_signed_integral();   // int8–64
v.is_unsigned_integral(); // uint8–64
v.is_floating_point();    // float, double
v.is_complex();           // complex<float/double>
v.is_string();            // pmr::string
v.is_tensor();            // Tensor<T>
v.is_map();               // Map (string → Value)

v.has_value();            // !is_monostate()
if (v) { /* has value */ }

// exact type inspection
v.value_type();           // ValueType enum
v.container_type();       // ContainerType enum
v.holds<std::int64_t>();  // true if exact match or convertible
```

### Direct Access

#### `get_if<T>()` — safe pointer access

Returns pointer to stored value, or `nullptr` on type mismatch. Use for zero-copy access when you need to check before accessing.

```cpp
Value v{std::int64_t{42}};

if (auto* p = v.get_if<std::int64_t>()) {
    *p += 1;  // modify in-place
}

v.get_if<double>();  // nullptr — type mismatch
```

> **Note:** for `std::string`/`std::string_view`, use `value_or()` instead — `get_if` returns the underlying `std::pmr::string*`.

### Monadic Access — `value_or<T>(fallback)`

The template parameter encodes **ownership semantics**:

| Syntax                   | Returns     | Value's state after        |
| ------------------------ | ----------- | -------------------------- |
| `value_or<T>(fb)`        | `T` (copy)  | unchanged                  |
| `value_or<T&>(fb)`       | `T&`        | unchanged (mutable borrow) |
| `value_or<const T&>(fb)` | `const T&`  | unchanged (const borrow)   |
| `value_or<T&&>(fb)`      | `T` (moved) | reset to monostate         |

```cpp
Value v{std::int64_t{42}};
std::int64_t fallback = -1;

// copy out (Value unchanged)
std::int64_t x = v.value_or<std::int64_t>(0);

// mutable borrow (modify in-place)
std::int64_t& ref = v.value_or<std::int64_t&>(fallback);
ref *= 2;  // v now holds 84

// const borrow (read without copy)
const auto& cref = v.value_or<const std::int64_t&>(fallback);

// ownership transfer (Value becomes monostate)
std::int64_t taken = v.value_or<std::int64_t&&>(0);
assert(v.is_monostate());
```

#### String conversion

String access handles `pmr::string` ↔ `std::string`/`std::string_view` automatically:

```cpp
Value v{std::string_view{"hello"}};

// auto-convert to std::string (allocates)
std::string s = v.value_or(std::string{""});

// zero-copy view (invalidated if Value modified)
std::string_view sv = v.value_or(std::string_view{""});
```

### Lazy Fallback — `or_else<T>(factory)`

Like `value_or` but the fallback is only evaluated on type mismatch. Use when computing the default is expensive.

```cpp
Value v{std::int64_t{42}};

// factory only called if type doesn't match
auto x = v.or_else<std::int64_t>([] { return expensive_computation(); });

// ownership transfer variant
auto y = v.or_else<std::int64_t&&>([] { return 0; });
```

String-specific variants:

```cpp
std::string s = v.or_else_string([] { return load_default_string(); });
std::string_view sv = v.or_else_string_view([] { return "default"sv; });
```

### Transform — `transform<T>(func)` / `transform_or<T>(func, default)`

Apply a function if the type matches; return default-constructed (or explicit fallback) otherwise.

```cpp
Value v{std::string_view{"hello"}};

// extract and convert — returns 0 on mismatch
std::size_t len = v.transform<std::pmr::string>([](const auto& s) {
    return s.size();
});

// with explicit fallback
std::size_t len2 = v.transform_or<std::pmr::string>(
    [](const auto& s) { return s.size(); },
    std::size_t{0}
);

// ownership transfer variant (resets Value to monostate)
auto moved = v.transform<std::pmr::string&&>([](auto&& s) {
    return process(std::move(s));
});
```

### Monadic Chaining — `and_then<T>(func)`

Like `transform` but `func` must return a `Value`. Enables fluent pipelines:

```cpp
Value v{std::int64_t{42}};

Value result = v.and_then<std::int64_t>([](auto& x) {
        return Value{x * 2};
    })
    .and_then<std::int64_t>([](auto& x) {
        return Value{std::to_string(x)};
    });

// result holds "84" as string
```

Returns default-constructed `Value` (monostate) on type mismatch.

---

## ValueHelper — Policy-Driven Conversion

`ValueHelper.hpp` provides batch conversion from `Value` to STL containers and Tensors with compile-time policy control.

```cpp
#include <gnuradio-4.0/ValueHelper.hpp>
using namespace gr::pmt;
```

### Three Orthogonal Policies

| Policy               | Options                                      | Purpose                       |
| -------------------- | -------------------------------------------- | ----------------------------- |
| **ConversionPolicy** | `Safe`, `Widening`, `Narrowing`, `Unchecked` | element type coercion rules   |
| **RankPolicy**       | `Strict`, `Flatten`, `Reshape`               | shape/dimensionality handling |
| **ResourcePolicy**   | `UseDefault`, `InheritFromSource`            | PMR memory propagation        |

**ConversionPolicy details:**

- `Safe` — `T→T` only (default)
- `Widening` — Safe + `int→larger_int`, `float→double`
- `Narrowing` — Widening + `double→float`, `int64→int32`
- `Unchecked` — any `static_cast`-able conversion

**RankPolicy details:**

- `Strict` — rank and extents must match exactly
- `Flatten` — any rank → rank-1 (linearise)
- `Reshape` — size-preserving reshape allowed

### `convertTo<T>()` — returns `std::expected`

```cpp
Value tensorVal = Tensor<float>{{2, 3}, {1, 2, 3, 4, 5, 6}};

// basic conversion (Safe policy, Strict rank)
auto vec = convertTo<std::vector<float>>(tensorVal);
if (vec) {
    // use *vec
} else {
    // vec.error().kind, vec.error().message
}

// with policies
auto wide = convertTo<std::vector<double>, ConversionPolicy::Widening>(tensorVal);
auto flat = convertTo<std::vector<float>, ConversionPolicy::Safe, RankPolicy::Flatten>(tensorVal);

// to fixed-size array
auto arr = convertTo<std::array<float, 6>>(tensorVal);

// to another Tensor
auto t2 = convertTo<Tensor<double>, ConversionPolicy::Widening>(tensorVal);
```

### `convertTo_or<T>()` — fallback variants

Never fail — return fallback on error:

```cpp
// with fallback value
std::vector<float> v = convertTo_or<std::vector<float>>(val, std::vector<float>{});

// with lazy factory
std::vector<float> v2 = convertTo_or<std::vector<float>>(val, [] {
    return std::vector<float>{1.0f, 2.0f};
});
```

### `assignTo()` — in-place assignment

Reuses existing allocation when possible:

```cpp
std::vector<float> dst;
dst.reserve(1000);  // pre-allocate

// assigns into dst, preserving capacity if sufficient
auto result = assignTo(dst, tensorVal);
if (!result) { /* handle result.error() */ }

// works with Tensors too
Tensor<float> dstTensor{{10}};
assignTo(dstTensor, sourceVal);

// and maps
std::unordered_map<std::string, Value> dstMap;
assignTo(dstMap, mapVal);
```

### Conversion Error Handling

`ConversionError` provides structured diagnostics:

```cpp
enum class Kind : std::uint8_t {
    None, TypeMismatch, ElementTypeMismatch,
    RankMismatch, ExtentsMismatch, SizeMismatch,
    NotATensor, NotAMap, EmptySourceNotAllowed,
    NarrowingNotAllowed, WideningNotAllowed
};

auto result = convertTo<std::vector<double>>(val);
if (!result) {
    switch (result.error().kind) {
    case ConversionError::Kind::NarrowingNotAllowed:
        // upgrade to Narrowing policy or handle
        break;
    case ConversionError::Kind::RankMismatch:
        // use RankPolicy::Flatten
        break;
    // …
    }
}
```

### Supported Conversions

| Source          | Target                          | Notes                             |
| --------------- | ------------------------------- | --------------------------------- |
| `Tensor<T>`     | `std::vector<U>`                | flattens if `RankPolicy::Flatten` |
| `Tensor<T>`     | `std::array<U, N>`              | size must match                   |
| `Tensor<T>`     | `Tensor<U>`                     | shape preserved                   |
| `Tensor<Value>` | typed containers                | element-wise conversion           |
| `Map`           | `std::unordered_map<string, T>` | Value extraction                  |
| `Map`           | `std::map<string, T>`           | Value extraction                  |

### Utility: Memory Usage

```cpp
std::size_t bytes = memory_usage(value);  // total allocation including PMR heap
```
