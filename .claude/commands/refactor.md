# Refactor

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

Apply structural improvements to changed files. Edit files directly, then print a summary.

This command includes all cosmetic fixes from `/fix-style` (naming §1, documentation §3,
wrong abstractions §8.7) in addition to the structural fixes below.

These changes may alter semantics — review the diff and run relevant `qa_` tests after applying.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

Determine affected files:

```bash
# default (no argument)
git diff --name-only HEAD -- '*.hpp' '*.cpp'
# with range argument
git diff --name-only <range> -- '*.hpp' '*.cpp'
```

Only modify structs/classes/functions that are part of the diff or directly affected by it.
Do not rewrite untouched code.

## Structural Fixes

**Member Ordering (§2)**
Reorder struct/class members to match canonical order:

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

Preserve all designated initialiser compatibility — if reordering would break
aggregate initialisation at a call site within the diff, flag it instead of fixing.

**Simplification (§4)**

- Replace raw loops with `std::ranges` / `std::algorithms` where intent is clearer
- Collapse nested `if`/`else` with early returns or guard clauses
- Extract complex inline expressions into named lambdas
- Replace `std::function` with templates where type erasure is unnecessary
- Apply `constexpr if` to eliminate dead branches
- Replace recursive templates with fold expressions
- Use structured bindings where applicable

Do NOT simplify if:

- The change obscures algorithmic intent
- The replacement is longer or harder to debug
- The original is already clear and idiomatic

## Output

After editing, print:

```
## refactor summary
- **Files modified**: list
- **Naming fixes**: count and brief description
- **Documentation stripped**: count
- **Member reordering**: which structs/classes were reordered
- **Simplifications**: count and brief description per file
- **Skipped**: items flagged but not auto-fixed (with reason)
```

Run `clang-format` on all modified files after editing.

Refactor: $ARGUMENTS
