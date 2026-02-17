# Correctness Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

## Focus

- Off-by-one errors in loops and bounds checks
- Iterator/pointer invalidation after container mutations
- Integer overflow in size/index calculations
- Undefined behaviour: signed overflow, null deref, uninitialized reads, aliasing violations
- Floating-point edge cases: NaN propagation, Inf, denormals, precision loss
- Race conditions in lock-free code (acquire/release ordering)
- `std::span` / `std::string_view` constructed from non-contiguous or temporary storage
- Implicit narrowing conversions that lose precision or sign
- Incorrect `processOne` / `processBulk` return semantics
- Tag propagation logic errors

Do NOT flag theoretical UB that is prevented by documented preconditions or `assert`.
Do NOT flag signed/unsigned comparison warnings already handled by the build system.
Do not comment on formatting — it is handled by clang-format.

## Output Format

For each finding:

```
**[CRITICAL|WARNING|NOTE]** `file:line` — description
> quoted code
Failure scenario: ...
```

Be specific. Only report genuine issues, not style preferences.

Review: $ARGUMENTS
