# Contributing to GNU Radio

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing. These are mostly guidelines, not rules. Use your best judgement, and feel free to propose changes to this document in a pull request.

## Open an issue

For bugs, issues, or other discussion, please log a new issue in the GitHub repository.

GitHub supports [markdown](https://help.github.com/categories/writing-on-github/), so when filing bugs make sure you check the formatting before clicking submit.

## Other discussions

For general "how-to" and guidance questions about using GNU Radio to build and run applications, please have a look at the various
examples or if you cannot find anything that fits your use-case use GitHub's discussions forum.

## Scope & Priorities

GNU Radio 4 is maintained with limited resources.
We focus on: core correctness and stability (runtime, scheduler, graph model, buffer system, type system),
performance (SIMD, GPU, memory management, low-latency paths),
maintainability (tests, CI, docs), and security.

We welcome contributions in all areas, but prioritise review of changes aligned with these goals.

### The core + OOT model

GR4 is designed as a core framework that you extend through out-of-tree (OOT) modules.
Most things can and should be developed, prototyped, and tested in OOT repos first.

We are specifically open to core changes that **enable and unblock OOT development**.
The most effective case for a core change is a working OOT prototype that demonstrates the need —
it lets reviewers see what you're building and where the core is in the way.

Without a concrete use case it's hard for reviewers to judge a request,
so a small example (an OOT repo, a snippet, a flowgraph) tends to move things along faster than an abstract proposal.

### What needs discussion first

Major changes affecting architecture, API/ABI, scheduler semantics, performance-critical paths,
or platform support should start as a [GitHub Discussion](https://github.com/gnuradio/gnuradio4/discussions)
before a pull request. This avoids wasted effort on both sides.

### Community-maintained areas

The following are welcome as community-driven efforts but are not owned or guaranteed by the core maintainers:

- general-purpose GUI / design tools,
- digital telecommunications blocks,
- Python bindings,
- platform-specific packaging beyond what CI covers.

We will support these with architecture guidance and review, but cannot commit to maintaining the results.

## Contributing code and content

We welcome all forms of contributions from the community. Please read the following guidelines to maximise the chances of your PR being merged.

### Communication

- Before starting work on a feature, check if there isn't already an example in the block library.
  If not, then please open an issue on GitHub describing the proposed feature. We want to make sure any feature work goes smoothly.
  We're happy to work with you to determine if it fits the current project direction and make sure no one else is already working on it.

- For any work related to setting up build, test, and CI for GNU Radio on GitHub, or for small patches or bug fixes, please open an issue
  for tracking purposes, but we generally don't need a discussion prior to opening a PR.

### Development process

Please be sure to follow the usual process for submitting PRs:

- Fork the repository
- Make sure it compiles w/o errors against the current release 'main' branch:
- Write and add a descriptive/meaningful unit-test-case
- apply the default code formatter (to minimise future refactoring)
- Please check against common sanitizers, the CI/CD pipeline, or similar other QA code checker (N.B. other/further code improvements are welcome)
- Create a pull request
- Make sure your PR title is descriptive
- Include a link back to an open issue in the PR description

We reserve the right to close PRs that are not making progress. Closed PRs can be reopened again later and work can resume.

### PR etiquette

**One concept per PR.** Each PR should address a single, reviewable change — one bug fix, one feature,
one refactoring. Don't combine unrelated changes.
If a feature requires multiple steps, discuss the plan first and submit a series of focused PRs
rather than one large opus.

**If it's big, discuss first.** For any change touching multiple files or modules, or any new feature:
open a Discussion or talk to us on Matrix before writing code. A 10-minute conversation can save days of wasted work on both sides.

**Write a meaningful PR description.** Explain what the change does, why it's needed, and how to test it.
Link to the relevant issue or Discussion. If there's an OOT repo that demonstrates the need, link it.

**Keep PRs reviewable.** If your diff exceeds ~500 lines, consider splitting it.
Reviewers have limited time and large PRs tend to stall.

### LLM-assisted contributions

LLM-generated code is welcome — but the machine has no brain, please use your own.

If you use an LLM to assist with your contribution, you are responsible for:

- understanding every line you submit,
- being able to explain your design decisions in review,
- debugging and iterating when reviewers ask questions,
- ensuring the code meets the project's style and quality standards.

PRs that the author cannot explain, debug, or iterate on will be closed.

LLM guidelines for this project are available in the repository.
These reflect the GR4 design standards and are intended to constrain LLMs to produce better code —
they are not meant as human-readable documentation.

### Commit conventions

We don't currently enforce strict commit message formats, but we are gradually moving toward consistent conventions.
Good commit messages help reviewers, future maintainers, and `git log` readers.

A good commit message:

- has a concise, descriptive title (≤72 characters),
- explains **why** the change was made, not just what,
- references the relevant issue or Discussion if applicable,
- is signed off (`git commit -s`) per the DCO.

We may introduce conventional commit prefixes (e.g. `fix:`, `feat:`, `refactor:`, `docs:`) in the future.
Keep your code dry, clean, terse, and follow the 'nomen-est-omen' paradigm where code is largely self-documenting
(e.g. meaningful names for methods, functions, variables, and fields).
For now, just be concise, clear, and descriptive.

### Copyright

All copyrights remain with the original author of the contribution.
We do not need/want copyright notices in individual file headers. Please follow best scientific practise and attribute
noteworthy ideas, papers, and other work — regardless of the license.
The collective copyright statement is in the [README](README.md).

### DCO Signing

Code submitters must have the authority to merge that code into the public GNU Radio codebase.
In some cases, the rights to exploit the code may belong to the contributor's employer, depending on jurisdiction
and employment agreements.

For that purpose, we use the [Developer's Certificate of Origin](DCO.txt). It is the same document used by other
projects.
Signing the DCO states that there are no legal reasons to not merge your code.

To sign the DCO, suffix your git commits with a "Signed-off-by" line. When using the command line,
you can use `git commit -s` to automatically add this line. If there were multiple authors of the code, or other types
of stakeholders, make sure that all are listed, each with a separate Signed-off-by line.

#### License

GNU Radio 4 is licensed under LGPL-3.0-or-later with a [static linking exception](LINKING_EXCEPTION.md).

By submitting a contribution, you agree that it is licensed under the same terms.

This means:

- anyone may use the library freely, including in proprietary and statically linked applications,
- modifications to the library source must be shared back,
- your application code is never affected by the library's license.

## A Note on Expectations

Contributions — bug reports, documentation, tests, code — are genuinely valued and make the project better.
However, contributions do not create an obligation for maintainers to deliver additional features,
accept unrelated changes, or guarantee review timelines.
Feature work is prioritised based on project goals, available capacity, and long-term maintainability.

**What makes a PR likely to be accepted:**

- fixes a bug with a test,
- improves performance with a benchmark,
- improves documentation based on real usage,
- unblocks an OOT use case with a minimal, focused core change,
- the author can explain and iterate on their work.

**What takes longer or stalls:**

- large feature additions submitted without prior discussion,
- changes that increase maintenance burden without a corresponding benefit,
- submissions the author cannot explain or iterate on.

For interaction standards, see the [Code of Conduct](./CODE_OF_CONDUCT.md).

## Code of Conduct

To ensure an inclusive community, contributors and users in the GNU Radio community should follow
the [code of conduct](./CODE_OF_CONDUCT.md).
