# Fix Style

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

Apply safe cosmetic fixes to changed files. Edit files directly, then print a summary.

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

Only modify lines/regions that are part of the diff or directly adjacent.
Do not rewrite untouched code.

## Fixes to Apply

**Naming (§1)**

- Reflected fields in `GR_MAKE_REFLECTABLE`: ensure `snake_case`
- Public non-reflected fields: ensure `lowerCamelCase`
- Private fields: ensure `_lowerCamelCase`
- Methods: ensure `lowerCamelCase`
- Types/structs/classes: ensure `UpperCamelCase`

**Documentation (§3)**

- Strip method-level `@brief`, `@param`, `@return` Doxygen boilerplate
- Preserve class/struct-level `/** ... */` block comments for public infrastructure types
- Remove comments that restate the code
- Remove commented-out code
- Remove decorative banners, ASCII art, and section separators (`// ---`, `// ===`, `// ── name ──`)

**Wrong Abstractions (§8.7)**

- Replace `std::variant` with `gr::pmt::Value` for wire-format values
- Flag (but do not auto-fix) raw SIMD intrinsics — suggest `vir::simd`
- Flag `throw` in library/framework code — suggest `std::expected`

## Output

After editing, print:

```
## fix-style summary
- **Files modified**: list
- **Naming fixes**: count and brief description
- **Documentation stripped**: count
- **Abstraction flags**: items that need manual attention
```

Run `clang-format` on all modified files after editing.

Verify that renamed identifiers compile: if a rename could break callers within the diff,
check before committing.

Fix: $ARGUMENTS
