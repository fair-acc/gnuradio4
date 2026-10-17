# Core Agent Workflow

For the orchestrating session: the one that decides, delegates to helpers and consults an advisor.
Same RFC 2119 keywords as `CLAUDE.md`.

## Agent tiers

Use the cheapest tier that does the job.

- **Orchestrator** (the session the author talks to): all decisions, all non-mechanical edits.
- **Helpers**: at most four concurrently, a weaker model, mechanical work only (rename, move, enumerate, format).
  Each gets a disjoint file scope; helpers cannot see each other, and overlapping scopes corrupt work.
  Helpers never commit; results return to the orchestrator.
  A mechanical rename is verified by the file's string literals being unchanged and by unchanged program output, not by a successful build:
  a word-bounded rename also rewrites words inside messages, and a scoped one misses call sites.
- **Waiting.** A background wait MUST watch a process ID or a file, never `pgrep -f "<text>"`: the waiting shell's own command line contains the text, so the loop never ends.
  Stop every wait of your own once its job is done.
  A scope boundary becomes a seam in the code (Conway's law):
  after a helper round the orchestrator MUST search across the scopes for duplicated constants, defaults and code.
- **Advisor**: the strongest model, independent and unprimed
  — given only the code, `CLAUDE.md` and the question, never the session history.
  Consulted before committing to an approach and before declaring done, not for routine steps;
  it also frames multiple-choice options when a trade-off is not obvious.
  Asked to argue the opposite: a model that agrees is not evidence, and sycophancy is the default failure mode.
  Its findings are relayed faithfully, each verified against the code.
  A second, primed pass MAY review a finished deliverable and is labelled as such.

## Large refactorings

List the proposals first and ask multiple-choice questions with a recommendation;
consult the advisor as the author decides (before the work, or at an impasse).

## Write it down

A job beyond a single edit gets a spec (`.md`: functional requirements, ground truth, recovery instructions) and an actionable plan before code, plus notes for off-tangent issues and spikes.
Long sessions drift and lose context: restart from the spec and plan rather than arguing with a confused session.
