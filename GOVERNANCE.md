# Governance

This repository is maintained primarily by contributors at GSI/FAIR and the wider community.
Decisions are made with a focus on correctness, maintainability, performance, and long-term sustainability.

## Roles

### Maintainers

Maintainers set technical direction for the core, review and merge contributions, and triage issues/discussions.

### Contributors

Contributors may submit issues, discussions, documentation, tests, and code.

## Decision-making

Most changes are decided via pull request review.
For changes affecting architecture, APIs/ABI, scheduler/runtime semantics, performance-critical paths,
or platform support: start a discussion first, gather feedback, then proceed with a PR once there is alignment.

Maintainers have final say on merges to keep the project coherent and maintainable.

## Development model

GNU Radio 4 follows a **core + out-of-tree (OOT)** development model.
The core provides the framework; application-specific functionality lives in OOT repos.

Core changes are accepted when they:

- fix bugs, improve performance, or improve maintainability,
- enable or unblock OOT development,
- are backed by a demonstrated need (working prototype, OOT module, or concrete use case).

A concrete use case helps reviewers evaluate scope and priority — without one,
proposals are harder to assess and tend to take longer.

## Prioritisation

Maintainer time is limited. We prioritise:

- correctness and stability of core behaviour,
- maintainability and test coverage,
- performance-critical work,
- security fixes,
- changes that unblock OOT development,
- documentation that reduces support load.

Feature requests are not promises.
Small contributions are appreciated, but they do not create an obligation for maintainers
to deliver additional work.

## Current work

The team's current focus and planned work is visible on the
[project board](https://github.com/orgs/fair-acc/projects/5/views/1).

## Conduct

All participants are expected to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
