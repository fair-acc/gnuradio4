# CLAUDE.md — GNU Radio 4.0 Code & Documentation Style Guide (V1)

This is the authoritative style guide for all AI-assisted (Claude Code) and human contributions
to GNU Radio 4.0 and related/downstream projects.
It is **self-contained** — no other guideline files need to be read.

See `.claude/commands/` for review persona commands.

Think of these commands as a design and code review assistant — the same way a spell-checker catches typos
you'd otherwise miss on the fifth read-through, the `const` linter flags, or duplicate/dead/overly-complex code
you forgot at 2 o'clock. They don't replace your judgement; they free you up to focus on the actual problem domain
instead of mentally tracking whether every field follows `snake_case` or checking if you accidentally left Doxygen
boilerplate in a new block. To note: any AI-agent-driven design/implementation requires strong constraints,
domain-knowledge, a good pre-design phase, strong unit-tests, and -- this is my recommendation -- a WIP/ToDo list
so that the AI doesn't go haywire or start hallucinating.

The review commands give you a second pair of eyes before you ping a colleague. The fix commands handle the mechanical
cleanup so you can spend your time on architecture decisions, not reformatting.

**All findings still must go through human review** — these tools just raise the baseline.

---

## 0 · Core Philosophy

**Nomen est omen** — the name _is_ the documentation.
Every identifier (type, field, method, parameter, variable) must be self-explanatory.
If you need a comment to explain _what_ something does, rename it first.

**Simplicity is a feature.** Prefer the simplest correct solution.
Do not add abstraction, indirection, or generality that is not required _today_.
Code that is easy to delete is better than code that is easy to extend.

**`struct` over `class`.** Default to `struct` with public members.
Use `class` only when a genuine invariant must be enforced (RAII resource ownership,
thread-safety contract, non-trivial construction/destruction coupling).

**Terse, production-ready code.** Method names and parameters should be self-documenting.
Lean and clean — complexity must justify itself.

---

## 1 · Naming Conventions

The keywords **MUST**, **SHOULD**, **MAY** follow RFC 2119.

### 1.1 Types

| Kind                        | Convention                                    | Examples                         |
| --------------------------- | --------------------------------------------- | -------------------------------- |
| `struct` / `class` / `enum` | UpperCamelCase (proper noun)                  | `Block`, `Graph`, `Selector`     |
| `enum` values               | UpperCase if proper noun, lowerCase if common | `Planet::Earth`, `Color::red`    |
| C++ concepts                | UpperCamelCase                                | `PortLike`, `HasProcessOne`      |
| Type aliases (`using`)      | UpperCamelCase                                | `ValueType`, `InputRange`        |
| Namespaces                  | all lowercase                                 | `gr`, `gr::basic`, `gr::testing` |

### 1.2 Functions & Methods

| Kind                               | Convention            | Examples                                        |
| ---------------------------------- | --------------------- | ----------------------------------------------- |
| Methods / free functions / lambdas | lowerCamelCase (verb) | `start()`, `processOne()`, `computeMagnitude()` |

### 1.3 Fields & Variables

| Kind                                              | Convention                      | Examples                     |
| ------------------------------------------------- | ------------------------------- | ---------------------------- |
| Public reflected settings (`GR_MAKE_REFLECTABLE`) | `snake_case` MUST               | `sample_rate`, `is_valid`    |
| Public non-reflected fields                       | lowerCamelCase                  | `inputBuffer`, `fftSize`     |
| Private / non-public fields                       | `_lowerCamelCase` (leading `_`) | `_initialised`, `_cachedFft` |
| Function-local variables & parameters             | lowerCamelCase                  | `nSamples`, `inputSpan`      |
| Compile-time constants                            | `kUpperCamelCase`               | `kMaxBufferSize`             |
| Preprocessor defines / macros                     | `UPPER_SNAKE_CASE`              | `GR_ENABLE_LOGGING`          |

### 1.4 Template Parameters

| Kind                | Convention                   | Examples                    |
| ------------------- | ---------------------------- | --------------------------- |
| Type parameters     | `T` or `TSpecificName`       | `T`, `TBlock`, `TAllocator` |
| Non-type parameters | lowerCamelCase or UPPER_CASE | `nPorts`, `kSize`           |

### 1.5 Files & Directories

File names MUST reflect the primary type they define: `Block.hpp`, `Selector.hpp`.
Test files: `qa_Block.cpp`, `qa_Selector.cpp`.

### 1.6 `auto` Usage

- If the type is precisely known, **prefer naming it explicitly**.
- `auto` MAY be used when the type is generic/templated, deduced from a complex expression,
  or excessively long and obvious from context.
- At API boundaries (function return types, public fields), **always name the type**.

---

## 2 · Struct & Class Layout

Always use `struct` unless a class invariant genuinely demands `class`.
Members MUST appear in the following canonical order (blank lines separate groups):

1. type aliases & nested types
2. ports
3. settings & public fields
4. `GR_MAKE_REFLECTABLE`
5. private state (prefixed with `_`)
6. constructor
7. lifecycle methods (`start`, `stop`, `reset`)
8. processing (`processOne` / `processBulk`)
9. settings change handler
10. helper methods (public then private)

```cpp
struct MyBlock : gr::Block<MyBlock> {
    using Description = Doc<"...">;

    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    Annotated<float, "gain"> gain = 1.0f;
    float sample_rate = 1.0f;

    GR_MAKE_REFLECTABLE(MyBlock, in, out, gain, sample_rate);

    float _cachedValue = 0.0f;

    explicit MyBlock(gr::property_map init = {}) : gr::Block<MyBlock>(std::move(init)) {}

    void start() { ... }
    void stop()  { ... }
    void reset() { ... }

    [[nodiscard]] constexpr auto processOne(float x) const noexcept { ... }

    void settingsChanged(const gr::property_map&, const gr::property_map&) { ... }
};
```

**Rules:**

- **All fields at the top** — they _are_ the API. This includes private state fields (prefixed `_`).
- `processOne` and `processBulk` are mutually exclusive — implement exactly one.
- Prefer `processOne` for simple 1:1 sample transforms.
  Use `processBulk` for resampling, variable-rate, or when you need span access.
- Mark processing methods `[[nodiscard]]`, `constexpr`, and `noexcept` where possible.

---

## 3 · Documentation Policy

### What to write

- **`using Description = Doc<"...">`** — one brief sentence per type explaining its purpose and
  key usage. This is the _only_ required documentation.
- **Class/struct-level block comments** — for public core infrastructure types (`/** ... */`),
  a detailed description explaining purpose, supported operations, and usage context is encouraged.
  `@brief` may be used at this level. Example:
  ```cpp
  /**
   * @brief Global signal legend displaying all registered sinks.
   *
   * This block implements Drawable<Toolbar> and renders a horizontal legend
   * showing all sinks from SinkRegistry. It supports:
   * - left-click: toggle sink draw enabled
   * - right-click: callback for settings panel
   * - drag: start drag operation for sink transfer
   */
  struct GlobalSignalLegend : gr::Block<...> { ... };
  ```
- **Short end-of-line comments** — allowed when a field name alone cannot convey units, valid
  ranges, or non-obvious intent (e.g. `float threshold = 0.5f; // linear, not dB`).
- **"Why" comments** — for non-obvious algorithmic choices, regulatory constraints, or
  workarounds. Keep to one or two lines.

### What NOT to write

- **Method-level `@brief` / `@param` / `@return` / Doxygen boilerplate** — banned.
  If the name and type signature do not explain it, rename them.
- **Restating the code in English** — `// increment counter` above `++counter` is noise.
- **Change-log comments** — use `git log`.
- **Commented-out code** — delete it; it lives in version control.
- **ASCII art, decorative banners, or separator comments** — including `// ---`, `// ===`,
  `// --- section name ---`, etc. The code structure itself should be self-evident.

### What NOT to comment on in reviews

- Formatting — handled by `.clang-format`.
- Linting — handled by `.clang-tidy`.
- Standard C++23 usage patterns that are project-standard.
- Requesting more documentation for self-explanatory code.

### Language style

This project follows the English conventions of the EU Interinstitutional Style
Guide — the shared standard usage of Ireland and the United Kingdom. This variety
is well-codified, accessible to native and non-native speakers alike, and widely
adopted across European and international organisations to minimise ambiguity.

Adopting a single documented standard avoids unproductive debates over spelling
and formatting. When in doubt, follow these conventions rather than personal or
regional habit.

- Use sentence case for headings (capitalise only the first word and proper nouns).
- Do not capitalise list items or comments that are not complete sentences.
- Introduce abbreviations on first use: "the Fast Fourier Transform (FFT)".
- Prefer active voice and short sentences.

**Example — avoid unnecessary capitalisation in comments:**

```cpp
// ✗ avoid
enum class AxisScale {
    Linear = 0,    // Standard linear scale [min, max]
    Log10,         // Logarithmic base 10
};

// ✓ prefer
enum class AxisScale {
    Linear = 0,    // standard linear scale [min, max]
    Log10,         // logarithmic base 10
};
```

---

## 4 · Complexity Reduction

Prefer the simplest correct solution. Specifically:

- **Prefer `std::ranges` / `std::algorithms` over raw loops.**
  A named algorithm communicates intent; a `for` loop does not.
- **Prefer named lambdas over complex inline expressions.**
  If an expression needs a comment, give it a name instead.
- **Prefer composition over inheritance.**
  Use template parameters and concepts to compose behaviour.
- **Prefer flat control flow.**
  Early returns, guard clauses, and `std::expected` / `std::optional` over deeply nested
  `if`/`else` trees.
- **Prefer value semantics.**
  Pass by value or `std::span`; avoid raw pointers and manual memory management.
- **Extract, don't comment.**
  If a block of code is complex enough to need a section comment, it is complex enough to be
  a named helper function or lambda.

---

## 5 · Modern C++ Expectations

**Target:** C++23.
**Compilers:** GCC 15+ (libstdc++), Clang 20+ (libc++), Emscripten, later AdaptiveCpp (SYCL).
Only use language and library features that are **available in both libstdc++ and libc++**.

**Build system:** CMake exclusively.

**Warnings:** Compile with `-Werror`. Already enforced for GNU Radio 4.0; downstream projects
should strive for the same.

### Prefer

- **Concepts** for constraining templates — over SFINAE or `static_assert`.
- **`constexpr` / `consteval`** — as broadly as possible; runtime only when compile-time is impossible.
- **`std::expected<T, E>`** — for recoverable errors. Our code is **exception-free**
  (user-defined code outside our repositories may throw).
- **`std::optional<T>`** — for "may or may not have a value" semantics.
- **`assert` / `std::unreachable()`** — for programmer errors and broken invariants.
- **`std::span<T>`** — for non-owning views into contiguous data.
- **`std::ranges` and views** — for composable data pipelines.
- **Structured bindings** — `auto [key, value] = ...`.
- **`std::format` / `fmt::format`** — over `std::stringstream` or manual concatenation.
- **`std::variant` / `std::visit`** — for closed type sets; prefer over runtime polymorphism.
- **`if constexpr`** — for compile-time branching.
- **Designated initialisers** — `Type{.field = value}`.
- **`[[nodiscard]]`** — on functions whose return value must not be silently discarded.
- **`[[likely]]` / `[[unlikely]]`** — where semantically meaningful in branch-heavy code.
- **SIMD:** prefer `vir::simd` / `std::simd` over compiler intrinsics.
- **PMR allocators** for hot paths; stack allocation for small fixed-size buffers.

### Avoid

- **`new` / `delete`** — use RAII wrappers (`std::unique_ptr`, `std::vector`, PMR containers).
- **C-style casts** — use `static_cast`, `std::bit_cast`.
- **Macros** — except `GR_MAKE_REFLECTABLE` and other framework-required macros.
- **`std::bind`** — use lambdas.
- **Throwing exceptions in library/framework code** — use `std::expected` or `std::optional`.
- **`std::endl`** — use `'\n'`.
- **`std::mdspan`** — not operationally available yet across both standard libraries.
  For multi-dimensional data, use `Tensor[View]<T>`.
  For 1D/vector-only types, prefer `std::vector`, `std::array`, `std::span` (KISS).

---

## 6 · GR4-Specific Conventions

### Blocks

- Inherit via CRTP: `struct Foo : gr::Block<Foo> { ... }`.
- Settings are `Annotated<T, "name", ...>` fields.
  Reflected fields MUST use `snake_case` for SigMF compatibility.
- `GR_MAKE_REFLECTABLE(...)` lists the type, then all reflected members (ports + settings).
- Tags: in `processOne`, use `this->mergedInputTag()` / `this->publishTag(...)`;
  in `processBulk`, use `inSpan.rawTags` / `outSpan.publishTag(...)`.

### Ports

- `gr::PortIn<T>`, `gr::PortOut<T>` for static ports.
- `std::vector<gr::PortIn<T>>` / `std::vector<gr::PortOut<T>>` for dynamic ports —
  resize in `settingsChanged`.
- Port names must be short, descriptive nouns: `in`, `out`, `reference`, `error_signal`.

### Graphs & Scheduling

- Build graphs with `gr::Graph`; connect with `graph.connect<"out">(src).to<"in">(sink)`.
- Use `gr::scheduler::Simple` unless a custom scheduler is justified.

### Type-Erased Values

- Use `gr::pmt::Value` for type-erased values — prefer over `std::variant` for
  wire-format compatibility.

### Error Handling

- `processOne`: return the output value; signal errors via tags or `requestStop()`.
- `processBulk`: return `gr::work::Status::OK`, `ERROR`, or `INSUFFICIENT_INPUT_ITEMS` /
  `INSUFFICIENT_OUTPUT_ITEMS`.
- Lifecycle methods (`start`, `stop`, `reset`): use `std::expected` for recoverable failures.
  Do not throw.

---

## 7 · Testing Conventions

- **File naming**: `qa_<TypeUnderTest>.cpp` (e.g. `qa_Selector.cpp`).
- **Framework**: [Boost.UT](https://github.com/boost-ext/ut) (`boost::ut`).
- **Structure**:

  ```cpp
  #include <boost/ut.hpp>
  using namespace boost::ut;

  const boost::ut::suite<"BlockName"> tests = [] {
      "descriptive scenario name"_test = [] {
          // arrange
          // act
          // assert with expect(...)
      };
  };
  ```

- **Coverage requirements**:
  - Every public type / block MUST have a `qa_` file.
  - Every `processOne` / `processBulk` path MUST be tested.
  - Edge cases: empty input, single sample, maximum buffer, type boundaries.
  - Tag propagation MUST be tested when the block reads or writes tags.
  - Settings changes MUST be tested via `settingsChanged` if the block implements it.
- **Test names** are sentences describing the scenario, not function names.
- **No `sleep` / timing-based tests** — use deterministic scheduling or event signalling.

---

## 8 · AI-Generated Code: Anti-Patterns to Avoid

These are the most common mistakes made by AI code assistants in this codebase.
Treat violations as review blockers.

### 8.1 Over-Engineering

- **Do not** introduce class hierarchies, abstract base classes, or factory patterns
  unless the existing codebase already uses that pattern for the specific concern.
- **Do not** create wrapper types around standard library types without clear justification.
- **Do not** add template parameters "for future flexibility" — only template what varies _now_.

### 8.2 Premature Generalisation

- **Do not** write a generic `Processor<T, Policy, Allocator, ...>` when a concrete
  `struct FirFilter` suffices.
- **Do not** extract a "common base" from two structs that happen to share two fields.

### 8.3 Unnecessary Abstractions

- **Do not** wrap `std::vector` in a custom `Container` class.
- **Do not** create `enum class ErrorCode` when `std::expected` with a descriptive
  error type (or `gr::work::Status`) already exists.
- **Do not** introduce an `Interface` / `Impl` split for types that have exactly one
  implementation and no testing seam requirement.

### 8.4 Verbose Comments & Documentation

- **Do not** generate method-level Doxygen `@param` / `@return` documentation.
  Class-level `@brief` for public infrastructure types is acceptable (see §3).
- **Do not** add comments that restate the code.
- **Do not** add `// constructor`, `// destructor`, `// getters`, `// setters` section markers.
- **Do not** generate README or markdown for internal helper files.

### 8.5 Hallucinated APIs

- **Verify every GR4 API call** against the actual headers in `core/include/gnuradio-4.0/`.
  Do not assume APIs exist — check.
- Common traps:
  - `Block::output()` does not exist — use the port member directly.
  - `notify_settings()` does not exist — settings propagation is automatic.
  - `this->log(...)` — use `fmt::print` or the project's logging macro if it exists.

### 8.6 Style Drift

- **Do not** switch to `snake_case` for method names (they are `lowerCamelCase`).
- **Do not** use `class` when the type has no invariant.
- **Do not** reorder struct members away from the canonical order (§2).
- **Do not** add `#pragma once` style changes to files that use `#ifndef` guards (or vice versa)
  — follow what the file already uses.

### 8.7 Wrong Abstractions

- **Do not** use `std::mdspan` — it is not available across our target compilers.
- **Do not** use raw SIMD intrinsics — use `vir::simd` / `std::simd`.
- **Do not** use `std::variant` for wire-format values — use `gr::pmt::Value`.
- **Do not** throw exceptions in library code — use `std::expected` / `std::optional`.

---

## 9 · Mechanical Formatting

Handled entirely by **clang-format** (see `.clang-format` in the repository root).
Do not manually adjust whitespace, brace placement, or indentation.
Run `clang-format` before committing.

This guide does **not** prescribe formatting rules — only semantic and structural ones.

---

## 10 · Quick Reference Checklist

Before submitting any code change, verify:

- [ ] Types use `struct` unless an invariant demands `class`.
- [ ] Names are self-explanatory — no comment needed to understand _what_.
- [ ] Struct members follow canonical order (§2).
- [ ] Public reflected fields are `snake_case`; other fields are `lowerCamelCase`.
- [ ] Methods are `lowerCamelCase`.
- [ ] No method-level Doxygen boilerplate (`@param`, `@return`); class-level `@brief` is allowed.
- [ ] No commented-out code.
- [ ] No unnecessary abstraction layers.
- [ ] `processOne` xor `processBulk` — not both.
- [ ] `GR_MAKE_REFLECTABLE` lists all reflected members.
- [ ] No exceptions thrown in library/framework code.
- [ ] SIMD uses `vir::simd` / `std::simd`, not raw intrinsics.
- [ ] A `qa_` test file exists with meaningful scenario coverage.
- [ ] Compiles cleanly with `-Werror` on GCC and Clang.
- [ ] `clang-format` has been run.
