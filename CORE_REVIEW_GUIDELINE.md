# Core Review Guideline

For reviewers and reviewing advisors, not for feature or refactoring iterations.
Same RFC 2119 keywords as `CLAUDE.md`; `CLAUDE.md` names when this file applies.

## Dependencies

Every new external dependency needs the author's approval, is pinned, and is declared once (`gr4_declare_dependency()` in `CMakeLists.txt`; `DEPENDENCIES.md` follows it, checked by `cmake/lint_dependencies.cmake`).

- **Core** (`core/`, `meta/`): the standard library and the dependencies already vendored only.
  An addition MUST be justified by its licence, maintenance record, transitive dependencies, added size and include reach (`devtools/code-metrics.sh`), security history,
  and why writing it here is worse.
- **Leaves** (`blocks/`, `algorithm/`, `devtools/`): a one-line reason.
- A dependency moving from a leaf into the core is reviewed as a core addition.

## Security and safety

Performance first: trust inside, guard at the project boundary.

- Interior code relies on its invariants: `assert` and `std::unreachable()` for programmer errors, no defensive checks.
- Every project boundary MUST bound its input sizes, reject path escapes, never throw, report through `std::expected`, tags or messages and `gr::log`,
  and turn raw input into checked types once ("parse, don't validate").
- A change that adds or widens a boundary names it in the commit message;
  the reviewer checks the five properties above for it.
