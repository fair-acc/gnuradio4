# CLAUDE.md — GR4 code, style and working agreement

> For LLM coding agents. Authoritative for GR4 conventions; machine-local files add environment
> specifics and never override this file. Rationale and prose live in `CORE_DEVELOPMENT_GUIDELINE.md`
> and `CORE_NAMING_GUIDELINE.md`, which use the same RFC 2119 keywords.

## 1 · Working agreement

**Be concise.** No preamble, no restating the request, no narrating intent. Report outcomes, numbers
and diffs. Non-functional chatter costs tokens and hides the result.

**Ask on forks.** Any decision that changes the deliverable stops and asks. Prefer multiple choice:
2-4 options, one line of context each, with a recommendation. Never proceed on an assumption and
disclose it afterwards. Use the advisor tier below to frame options when the trade-off is not obvious.

**Approvals expire.** Approval MUST be re-sought each time; it is scoped to the action named and
the turn it was given in. `commit`,
`amend`, `push`, force-push and any outward-facing action are re-asked every time. A prior yes is
never standing.

**Agent tiers** — use the cheapest that can do the job:

- _orchestrator_ (this session): all decisions, all non-mechanical edits.
- _helpers_: max 4 concurrent, weaker model, mechanical work only (rename, move, enumerate, format).
  Give each a disjoint file scope — helpers cannot see each other, overlapping scopes corrupt work.
  Helpers never commit; results return to the orchestrator.
- _advisor_: strongest model, to cross-check before committing to an approach and before declaring
  done. Not for routine steps.

**Measure, do not infer.** State a resource or timing number only from an isolated measurement, and
name the method. Timing a build-system invocation also times its dependency chain.

**Report what was not done.** Skipped scope, unverified claims and known-failing checks are stated
plainly.

**Absolute time.** Date notes absolutely (`2026-08-31`), never "yesterday". Before acting on a
recorded fact, check its age and verify the files, flags and symbols it names still exist.

**Sign-off.** Every commit MUST be signed off with the repository's configured git credentials
(`git commit -s`) and carry no other trailer — no LLM/bot co-author, no assistant attribution, no
synthesised identity. The sign-off is an assertion by the human author, who holds the copyright and
remains answerable for the quality of the code; that an assistant drafted it changes neither.

**Git.** Derive an earlier commit's file state with `git show <sha>:<path>` plus a minimal patch,
never from the final version. During a _conflicted_ rebase step use `git add` + `git rebase
--continue`; `git commit --amend` there silently squashes the commit being applied.

## 2 · Philosophy

**Nomen est omen.** Code is the single source of truth. Carry meaning in names — types, methods,
lambdas, parameters, variables — not in comments. A comment that restates code is deleted; a name
that needs a comment is renamed. Exception: a public core-API type whose intent is not evident from
its name, or which is large, may carry a short functional description with a user-level snippet.

**Simplicity is a feature.** Simplest correct solution. No abstraction, indirection or generality
that is not required today. Code that is easy to delete beats code that is easy to extend.

No wrapper types around std or gr containers, no Interface/Impl split for a single implementation,
no factories or class hierarchies the codebase does not already use for that concern.

**`struct` over `class`.** Default to `struct` with public members. Use `class` only for a genuine
invariant: RAII ownership, thread-safety contract, coupled construction/destruction.

## 3 · Naming

Reflected settings MUST be `snake_case` (SigMF wire compatibility); other public fields are
lowerCamelCase. `CORE_NAMING_GUIDELINE.md` gives the rationale.

| kind                                       | convention                               | example                              |
| ------------------------------------------ | ---------------------------------------- | ------------------------------------ |
| struct / class / enum / concept / alias    | UpperCamelCase                           | `Block`, `PortLike`, `ValueType`     |
| enum values                                | UpperCase if proper noun, else lowerCase | `Planet::Earth`, `Color::red`        |
| namespaces                                 | lowercase                                | `gr`, `gr::basic`                    |
| methods, free functions, lambdas           | lowerCamelCase verb                      | `processOne()`, `computeMagnitude()` |
| reflected settings (`GR_MAKE_REFLECTABLE`) | `snake_case`                             | `sample_rate`                        |
| public non-reflected fields                | lowerCamelCase                           | `inputBuffer`                        |
| private fields                             | `_lowerCamelCase`                        | `_cachedFft`                         |
| locals and parameters                      | lowerCamelCase                           | `nSamples`                           |
| compile-time constants                     | `kUpperCamelCase`                        | `kMaxBufferSize`                     |
| macros                                     | `UPPER_SNAKE_CASE`                       | `GR_ENABLE_LOGGING`                  |
| type template parameters                   | `T` / `TSpecificName`                    | `TBlock`                             |
| non-type template parameters               | lowerCamelCase or UPPER_CASE             | `nPorts`                             |

Files are named for the primary type: `Block.hpp`, `qa_Block.cpp`.

`auto` only when the type is generic, deduced from a complex expression, or long and obvious from
context. Name the type explicitly at API boundaries.

## 4 · Struct layout

Members in this order, blank lines between groups: type aliases · ports · settings and public
fields · `GR_MAKE_REFLECTABLE` · private state · constructor · lifecycle (`start`/`stop`/`reset`) ·
`processOne` xor `processBulk` · `settingsChanged` · helpers.

All fields at the top — they are the API. `processOne` for 1:1 sample transforms, `processBulk` for
resampling, variable rate or span access. Mark processing `[[nodiscard]] constexpr noexcept` where
possible.

## 5 · Documentation

Write: `using Description = Doc<"...">` (one sentence, the only required documentation); a
class-level block comment for public core infrastructure types; end-of-line comments where a name
cannot convey units or ranges (`float threshold = 0.5f; // linear, not dB`); one- or two-line _why_
comments for non-obvious algorithmic or regulatory choices.

Never write: method-level `@brief`/`@param`/`@return`; restatements of the code; change logs;
commented-out code; ASCII art, banners or separator comments; README or markdown for internal
helper files unless asked.

Do not review: formatting (`.clang-format`), linting (`.clang-tidy`), standard C++23 idioms, or
missing documentation on self-explanatory code.

Language: EU Interinstitutional Style Guide (Irish/UK usage). Sentence case headings, lowercase list
items and comment fragments, abbreviations introduced on first use, active voice.

## 6 · C++23

GCC 15+ (libstdc++), Clang 20+ (libc++), Emscripten, later AdaptiveCpp. Only features available in
both standard libraries. CMake only. `-Werror`.

Prefer: concepts over SFINAE · `constexpr`/`consteval` · `std::expected` for recoverable errors ·
`std::optional` · `assert`/`std::unreachable()` for programmer errors · `std::span` · ranges and
views · `std::variant`/`std::visit` for closed type sets ·
`[[nodiscard]]` · `vir::simd`/`std::simd` · PMR in hot
paths, stack for small fixed buffers.

Avoid: `new`/`delete` · C-style casts · macros other than framework ones · `std::bind` · exceptions
in library code (MUST NOT throw) · `std::endl` · `std::mdspan` (use `Tensor[View]<T>`, or `std::vector`/`array`/`span`
for 1-D).

Prefer `std::ranges` and named algorithms over raw loops; named lambdas over complex inline
expressions; composition over inheritance; early returns over nested branches; value semantics over
raw pointers. If a block needs a section comment, extract it as a named helper.

## 7 · GR4 conventions

Blocks are CRTP: `struct Foo : gr::Block<Foo>`. Settings are `Annotated<T, "name">` fields.
`GR_MAKE_REFLECTABLE` lists the type then every reflected member.

Ports: `gr::PortIn<T>`/`gr::PortOut<T>`; `std::vector<gr::PortIn<T>>` for dynamic ports, resized in
`settingsChanged`. Names are short nouns: `in`, `out`, `reference`.

Tags: `this->mergedInputTag()` / `this->publishTag(...)` in `processOne`; `inSpan.rawTags` /
`outSpan.publishTag(...)` in `processBulk`.

Graphs: `gr::Graph`, connect with `graph.connect<"out", "in">(src, sink)`. Use `gr::scheduler::Simple`
unless a custom scheduler is justified.

Type-erased values: `gr::pmt::Value`, not `std::variant`, for wire-format compatibility.

Errors: `processOne` MUST return the value and signals via tags or `requestStop()`; `processBulk` returns
`gr::work::Status`; lifecycle methods return `std::expected`. Never throw.

Logging: `gr::log::*` from `Logger.hpp`; see `docs/USER_API_Logging.md`.

Every API MUST be verified against the headers in `core/include/gnuradio-4.0/` before use.

## 8 · Testing

`qa_<TypeUnderTest>.cpp`, Boost.UT:

```cpp
const boost::ut::suite<"BlockName"> tests = [] {
    "descriptive scenario name"_test = [] { /* arrange, act, expect(...) */ };
};
```

Every public type needs a `qa_` file; every `processOne`/`processBulk` path, tag propagation and
`settingsChanged` behaviour must be covered, plus edge cases (empty, single sample, maximum buffer,
type boundaries). Test names are scenario sentences. No `sleep` or timing-dependent tests — use
deterministic scheduling or event signalling.

## 9 · Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++-15
cmake --build build --parallel <N>
ctest --test-dir build --output-on-failure
```

The binding constraint is **memory, not cores**: template-heavy translation units have exceeded 7 GB
each. Set `-DGR_BUILD_PARALLEL_LEVEL=<N>` at configure time — it drives the Ninja job pools and is
the only knob to touch; `--parallel` merely bounds it. Default `<N>=4`, raise only after measuring
peak RSS. Compiler order when testing across toolchains: GCC 15, Clang 20,
Emscripten, GCC 14.

**CPU budget.** A session gets `N` cores and a disjoint core range, so two Claude sessions on one
machine never contend and neither starves the desktop. `N` is what the user asked for, else
`nproc - 2`. The range starts at `i * N`, where `i` is this session's instance number on the machine
(first 0, second 1, …): session 0 gets `0..N-1`, session 1 gets `N..2N-1`.

Run **every** compile, test, agent and helper through it — `taskset -c <first>-<last> nice -n 5 …` —
and pass `-j <N>` as well. `-j` alone is not enough: `lld` and `mold` default to one thread per
hardware core, so a single link ignores it. Pinning bounds the whole process tree; `-j` only bounds
ninja's job count.

Never `pkill` by program name (`cc1plus`, `ninja`): it kills the other session's compiles too. Match
on the build directory instead.

Follow each file's existing include-guard style — never convert between `#pragma once` and
`#ifndef`. Formatters (`clang-format`, `cmake-format`, `black`, `prettier`, `shellcheck`) MUST be run
before committing; never adjust whitespace by hand.

## 10 · Before submitting

`struct` unless an invariant demands `class` · names self-explanatory · canonical member order ·
reflected fields `snake_case`, methods lowerCamelCase · no method-level Doxygen · no commented-out
code · no unnecessary abstraction · `processOne` xor `processBulk` · `GR_MAKE_REFLECTABLE` complete ·
no exceptions · `vir::simd` not intrinsics · a meaningful `qa_` file exists · builds `-Werror` on GCC
and Clang · formatters run.
