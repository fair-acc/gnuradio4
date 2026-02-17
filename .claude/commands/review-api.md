# API Design Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

## Focus

- CRTP usage: `Block<Derived>` applied correctly, no slicing
- Concept constraints: template parameters properly constrained with `requires`
- Type safety: could stronger types prevent misuse? (e.g. `Annotated<>` with `Limits<>`)
- Const-correctness: const methods truly non-mutating
- `std::expected` vs exceptions: appropriate for this code path?
- API surface: can callers misuse this? (wrong order, invalid states, silent defaults)
- Consistency with existing GR4 APIs (port naming, settings patterns, lifecycle methods)
- `GR_MAKE_REFLECTABLE` completeness: all reflected members listed
- `snake_case` for reflected fields, `lowerCamelCase` for methods
- Struct vs class: is `class` justified by an actual invariant?
- Rule of zero: unnecessary constructors/destructors?

Do NOT flag naming or API patterns in code outside the diff.
Do NOT request documentation beyond `using Description = Doc<"...">` for self-explanatory code (§3).
Do not comment on formatting — it is handled by clang-format.

## Output Format

For each finding:

```
**[CRITICAL|WARNING|NOTE]** `file:line` — description
> quoted code
Concern: what can go wrong or what is inconsistent
```

Review: $ARGUMENTS
