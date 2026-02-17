# Performance Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

## Focus

- Unnecessary heap allocations in hot paths (`processOne`, `processBulk`, inner loops)
- Missed `vir::simd` / `std::simd` vectorisation opportunities
- Cache-unfriendly access patterns (strided access, pointer chasing, AoS vs SoA)
- Redundant computations that could be hoisted or cached
- Copy where move suffices (`std::vector`, `std::string`, large structs)
- PMR allocator opportunities for arena allocation in hot paths
- Stack allocation viable for small fixed-size buffers
- Branch-heavy tight loops (consider `[[likely]]` / `[[unlikely]]` or branchless alternatives)
- Memory bandwidth bottlenecks

Do NOT flag code that is clearly not on the hot path.
Do NOT suggest micro-optimisations that harm readability unless benchmark/profiler-justified.
Do not comment on formatting — it is handled by clang-format.

## Output Format

For each finding:

```
**[CRITICAL|WARNING|NOTE]** `file:line` — description
> quoted code
Impact: (allocations per call / cache misses / missed vectorisation / etc.)
Fix: concrete change
```

Review: $ARGUMENTS
