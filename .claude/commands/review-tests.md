# Test Coverage Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

## Focus

Evaluate existing tests for:

- Edge cases: empty input, single element, maximum buffer size, type limits
- Numerical stability: NaN, Inf, denormals, precision loss near boundaries
- All `processOne` / `processBulk` code paths exercised
- Tag propagation tested (if block reads/writes tags)
- `settingsChanged` tested (if block implements it)
- State transitions: start → stop → start, reset cycles
- Aliasing scenarios: input buffer == output buffer (where applicable)
- Template instantiation coverage: all supported value types tested

Identify missing `qa_` files for public types.

Tests must use `boost::ut`. Test names must be descriptive sentences.
No `sleep` or timing-based assertions.

Do NOT suggest tests for trivial getters, type aliases, or code fully constrained by the type system.
Do NOT suggest tests for formatting or serialisation unless the code under review implements those.
Do not comment on formatting — it is handled by clang-format.

## Output Format

For each gap:

```
**[CRITICAL|WARNING|NOTE]** `file` or `type` — what is missing
Why it matters: what could break undetected
Test sketch: 2-4 lines of boost::ut pseudocode
```

Review: $ARGUMENTS
