# Block Structure Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

Targeted at block authors: checks whether a new or modified block follows GR4 structural
conventions, the nomen-est-omen principle, and the simplicity-first philosophy.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

Determine affected files, then review only block structs (types inheriting `gr::Block<>`)
within those files:

```bash
# default (no argument)
git diff --name-only HEAD -- '*.hpp' '*.cpp'
# with range argument
git diff --name-only <range> -- '*.hpp' '*.cpp'
```

## Focus

**Structure (§2)**

- Correct CRTP: `struct Foo : gr::Block<Foo>` (not `class`, unless a genuine invariant demands it)
- `using Description = Doc<"...">` present with a brief, meaningful sentence
- Canonical member order:
  1. type aliases & nested types
  2. ports
  3. settings & public fields
  4. `GR_MAKE_REFLECTABLE`
  5. private state (`_lowerCamelCase`)
  6. constructor
  7. lifecycle (`start`, `stop`, `reset`)
  8. processing (`processOne` / `processBulk`)
  9. `settingsChanged`
  10. helper methods
- `processOne` xor `processBulk` — exactly one, not both, not neither

**Naming & nomen est omen (§0, §1)**

- Reflected fields in `GR_MAKE_REFLECTABLE`: `snake_case`
- Private fields: `_lowerCamelCase`
- Methods: `lowerCamelCase`
- Every identifier must be self-explanatory without a comment. Flag:
  - single-letter names outside tight numeric loops
  - abbreviations without established context (e.g. `proc` instead of `process`)
  - misleading names (a bool named `count`, a method named `data` that mutates state)
  - names that require a comment to understand — the fix is to rename, not to add a comment

**Simplicity (§0, §4)**

- No unnecessary abstraction: wrappers, factories, or hierarchies with a single implementation
- No premature generalisation: template parameters that could be concrete types
- Flat control flow: early returns and guard clauses over nested `if`/`else`
- Value semantics: `std::span` over raw pointers, move over copy for large types
- Terse: if a helper function is called once and its body is shorter than its call site, inline it
- DO NOT implement custom public constructors or destructors as these interfer with the ones in Block<T>,
  rely on RAII instead

**Reflectable completeness**

- `GR_MAKE_REFLECTABLE` lists the type, then all ports and settings — nothing missing, nothing extra
- All settings that appear in `settingsChanged` are reflected
- `GR_REGISTER_BLOCK` (if present) lists all intended type instantiations

**Error handling (§5, §6)**

- No `throw` in any method
- `processOne`: returns output value; errors via tags or `this->requestStop()`
- `processBulk`: returns `gr::work::Status`
- Lifecycle methods: `std::expected` for recoverable failures

**Tag handling**

- If the block reads tags: uses `this->inputTagsPresent()` guard before `this->mergedInputTag()`
- If the block writes tags: uses `this->publishTag()` (processOne) or `outSpan.publishTag()` (processBulk)

**Settings**

- If the block has mutable settings: `settingsChanged()` handles all of them
- Settings revert logic: unchanged settings are restored when a trigger does not match (if applicable)

Do NOT comment on formatting — it is handled by clang-format.
Do NOT flag naming in code outside the diff.
Do NOT request documentation beyond `using Description = Doc<"...">` for self-explanatory code.

## Output Format

For each block struct in the diff:

```
## Block: `TypeName`

**Structure:** ✓ correct | list deviations
**Naming:** ✓ correct | list violations
**Simplicity:** ✓ correct | list concerns
**Reflectable:** ✓ complete | list missing/extra members
**Processing:** ✓ correct | note issues
**Tags:** ✓ correct | N/A | note issues
**Settings:** ✓ correct | N/A | note issues

### Findings (if any)
**[CRITICAL|WARNING|NOTE]** `file:line` — description
> quoted code
Fix: concrete suggestion
```

Review: $ARGUMENTS
