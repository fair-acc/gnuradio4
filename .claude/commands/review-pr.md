# Pull Request Review

Baseline conventions are in CLAUDE.md — follow them, do not restate them.

Use `/review-pr` for a quick pre-merge scan across all concerns.
Use individual review commands (`/review-correctness`, `/review-perf`, etc.) for deep-dive
analysis of a specific concern.

Perform a concise review covering correctness, performance, API design, safety,
simplification opportunities, and test coverage.

## Scope

Default: uncommitted changes (staged + unstaged vs HEAD).
If an argument is provided, treat it as a git range (e.g. `HEAD~3`, `main..feature`).

## Severity Key

- **[CRITICAL]** — must address before merge: correctness bug, safety issue, data-loss risk
- **[WARNING]** — should address: design concern, performance regression, missing coverage
- **[NOTE]** — optional improvement: clarity, idiom, simplification

## Output Format

```
## Summary
One paragraph: what this PR does and its overall state.

## Critical Issues
[CRITICAL] findings that must be addressed. Omit section if none.

## Warnings
[WARNING] findings worth addressing. Omit section if none.

## Notes
[NOTE] minor improvements. Omit section if none.

## Test Gaps
Missing coverage or test quality concerns. Omit section if none.
```

Quote exact lines (`file:line` + code) for every finding.
Be concise; omit sections with no findings. Elaborate only for critical issues.
Do not comment on formatting — it is handled by clang-format.
Do not produce a merge verdict — the human decides.

Review: $ARGUMENTS
