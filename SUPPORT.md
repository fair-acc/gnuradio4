# Support

This project is community-driven and maintained on a **best-effort** basis.
We aim to be helpful and responsive, but we do not provide SLAs or guaranteed timelines.

## Development model: core + out-of-tree

GNU Radio 4 is designed as a **core framework** that you extend through **out-of-tree (OOT) modules** —
not by adding everything into the core.

The core accepts changes that:

- fix bugs (with tests),
- improve performance (with benchmarks),
- enable or unblock OOT development,
- improve documentation based on real usage.

Application-specific functionality belongs in OOT repos.
If you're building something on GR4, start with an [OOT module](https://github.com/gnuradio/gr4-incubator)
and come to us when you hit a wall in the core.
Show us what you're building and what's blocking you — that's the most effective path
to getting core changes reviewed and merged.

We are specifically open to core changes that unblock OOT work.
The best way to make the case for a core change is to demonstrate the need with a working OOT prototype.

## Where to ask what

### GitHub Discussions (preferred for questions)

Use Discussions for:

- "How do I…?" usage questions,
- build/setup questions,
- design/architecture questions and early proposals,
- "Is this supported?" questions.

### GitHub Issues (bugs and actionable work items)

Use Issues for:

- reproducible bugs with a minimal test case,
- concrete, scoped work items,
- documentation problems with a clear fix.

Support requests disguised as bugs will be redirected to Discussions or closed.

## Getting help effectively

We're a small team and we prioritise helping people who are helping themselves and others.

**What gets a good response:**

- you've tried to solve the problem and can show what you tried,
- you provide a minimal reproducer and your environment details,
- you're building something on GR4 and can describe what and why,
- you've read the docs and can point to where they fell short,
- you share workarounds and solutions when you find them.

**What helps everyone:**

- answering questions from other users in Discussions,
- improving documentation based on your own learning experience,
- sharing working examples and OOT modules,
- reporting bugs with full reproduction steps.

People who contribute to the community — by answering questions, sharing knowledge, and helping others —
are the backbone of this project.
We notice and appreciate it, and they tend to get our attention first when they need help themselves.

**What's harder to help with:**

- requests without context ("make X work"),
- feature requests with no described use case — without context it's hard to evaluate scope or priority,
- issues without reproduction steps.

This isn't gatekeeping — it's how a small team stays effective.
The more context you give us, the better we can help.
The more you help others, the stronger the community becomes.

## What to include for bugs

A bug report must include:

- exact version / commit SHA,
- OS, compiler, build type, relevant flags,
- minimal reproducer (code, flowgraph, or commands),
- expected vs actual behaviour,
- logs / backtrace (if applicable).

Reports without a reproducer may be closed after a period of inactivity.

## Feature requests and larger changes

We welcome ideas, but large changes require:

- a clear problem statement and motivation,
- an outline of the proposed approach (ideally with an OOT prototype),
- compatibility/performance considerations,
- **an owner** willing to drive implementation and follow-up maintenance.

If you want a feature faster, the most effective path is to contribute (code, tests, docs, CI coverage).

## Issue hygiene

Issues without activity for 30 days may be closed.
This isn't a judgement on the issue's importance — it's housekeeping.
Closed issues can be reopened with new information.
Issues that have been assigned to the [project board](https://github.com/orgs/fair-acc/projects/5/views/1)
are tracked by the maintainer team and won't be auto-closed.

## Following development

You can follow the team's current focus and planned work on the
[project board](https://github.com/orgs/fair-acc/projects/5/views/1).

## Conduct

We collaborate in good faith.
Maintainers may close or lock discussions and disengage when interactions stop being constructive.
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
